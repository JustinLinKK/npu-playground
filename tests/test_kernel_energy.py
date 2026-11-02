import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


meter = module('research/energy_meter.py')


def counter_trace(power, duration, rng, tick=0.005, poll=0.02):
    """A Windows Energy Meter as seen by the sampler: integer pWh updated every `tick`, polled with jitter."""
    ticks = np.arange(0, duration + tick, tick)
    energy = np.concatenate([[0], np.cumsum(power(ticks[:-1]) * tick)])  # J at each tick
    counter = np.floor(energy / meter.PWH_TO_J)
    polls = np.sort(np.clip(np.arange(0, duration, poll) + rng.uniform(-0.003, 0.003, int(np.ceil(duration / poll))), 0, duration))
    latest = np.searchsorted(ticks, polls, side='right') - 1
    return polls, counter[latest] * meter.PWH_TO_J


def test_counter_power_recovers_rail_power_within_tick_rounding():
    rng = np.random.default_rng(1)
    t, joules = counter_trace(lambda x: np.where(x < 5, 30.0, 35.0), 15, rng)
    assert meter.counter_power(t, joules, 6.0, 12.0) == pytest.approx(35.0, abs=0.05)
    assert meter.counter_power(t, joules, 1.0, 4.0) == pytest.approx(30.0, abs=0.05)
    with pytest.raises(ValueError, match='outside'):
        meter.counter_power(t, joules, 14.0, 16.0)


def test_sampled_power_is_exact_for_a_ramp_with_edges_between_samples():
    t = np.array([0.0, 0.13, 0.31, 0.5, 0.77, 1.0, 1.42, 2.0])
    assert meter.sampled_power(t, 2 * t, 0.2, 1.3) == pytest.approx(0.2 + 1.3)  # mean of 2t over [a, b] = a + b
    step = np.where(t < 1.0, 10.0, 30.0)
    assert 10.0 < meter.sampled_power(t, step, 0.5, 1.5) < 30.0


def test_window_power_derives_uncore_from_the_rails():
    t = np.linspace(0, 10, 501)
    trace = {'t': t, 'pkg_j': 32 * t, 'pp0_j': 20 * t, 'pp1_j': 0.1 * t, 'gpu_w': np.full_like(t, 25.0)}
    power = meter.window_power(trace, 2, 8)
    assert power['pkg'] == pytest.approx(32) and power['uncore'] == pytest.approx(11.9) and power['gpu'] == pytest.approx(25)


def windows_with_drift(effect, noise, rng, length=6.0, idle=5.0, cycles=6):
    """Idle/run windows of an I R I R ... I session under slow drift; run power = idle + effect."""
    windows, clock = [], 0.0
    for index in range(2 * cycles + 1):
        kind = 'idle' if index % 2 == 0 else 'run'
        span = idle if kind == 'idle' else length
        middle = clock + span / 2
        base = 30 + 0.05 * middle + 0.5 * np.sin(middle / 40)
        windows.append({'kind': kind, 'mode': 'npu_sync' if kind == 'run' else 'idle', 'per_call_s': 1e-4,
                        'power': {'pkg': base + (effect if kind == 'run' else 0) + rng.normal(0, noise)}})
        clock += span + 0.5
    return windows


def test_paired_deltas_cancel_linear_drift_exactly():
    windows = [{'kind': k, 'power': {'pkg': 30 + 0.1 * i + (2 if k == 'run' else 0)}}
               for i, k in enumerate(['idle', 'run', 'idle', 'run', 'idle'])]
    pairs = meter.paired_deltas(windows)
    assert [p['delta']['pkg'] for p in pairs] == pytest.approx([2.0, 2.0])
    assert meter.paired_deltas([{'kind': 'run', 'power': {'pkg': 1}}]) == []


def test_baseline_is_interpolated_in_time_when_windows_are_unevenly_spaced():
    # After a GPU run the schedule waits for P8, and unsettled idles are skipped, so a run rarely sits midway.
    spans = [('idle', 0, 5), ('run', 5.5, 11.5), ('idle', 22, 27)]
    windows = [{'kind': k, 'start': a, 'end': b, 'power': {'pkg': 30 + 0.2 * (a + b) / 2 + (2 if k == 'run' else 0)}}
               for k, a, b in spans]
    pair, = meter.paired_deltas(windows)
    assert pair['delta']['pkg'] == pytest.approx(2.0) and pair['baseline_weight'] == pytest.approx(6 / 22)
    plain_mean = windows[1]['power']['pkg'] - (windows[0]['power']['pkg'] + windows[2]['power']['pkg']) / 2
    assert plain_mean == pytest.approx(1.0)  # the old estimator reported half the effect


def test_unsettled_idle_windows_are_not_baselines():
    t = np.linspace(0, 30, 1501)
    pstate = np.where((t > 10) & (t < 16), 1.0, 8.0)  # GPU holds P1 for 6 s after work
    trace = {'t': t, 'gpu_pstate': pstate}
    assert meter.gpu_settled(trace, 0, 9) and not meter.gpu_settled(trace, 11, 18) and meter.gpu_settled(trace, 17, 25)
    assert meter.gpu_settled({'t': t, 'gpu_pstate': None}, 11, 18)  # no GPU traced
    windows = [{'kind': 'idle', 'power': {'pkg': 30.0}}, {'kind': 'run', 'power': {'pkg': 33.0}},
               {'kind': 'idle', 'settled': False, 'power': {'pkg': 28.0}}, {'kind': 'run', 'power': {'pkg': 33.0}},
               {'kind': 'idle', 'power': {'pkg': 30.0}}]
    assert [p['delta']['pkg'] for p in meter.paired_deltas(windows)] == pytest.approx([3.0, 3.0])
    windows[3]['settled'] = False  # a run during which another program woke the GPU
    assert len(meter.paired_deltas(windows)) == 1


def test_estimate_matches_hand_computed_t_interval():
    result = meter.estimate([1, 2, 3, 4, 5, 6])
    se = np.std([1, 2, 3, 4, 5, 6], ddof=1) / np.sqrt(6)
    assert result['mean'] == 3.5 and result['se'] == pytest.approx(se)
    assert result['ci95'] == pytest.approx([3.5 - 2.571 * se, 3.5 + 2.571 * se])
    assert result['significant'] and result['mde80'] == pytest.approx((2.571 + 0.920) * se)
    assert result['upper95'] == pytest.approx(3.5 + 2.015 * se)
    assert meter.estimate([1.0])['ci95'] is None and meter.quantile(meter.T975, 22) == 2.086


def test_paired_estimator_is_unbiased_with_nominal_coverage_under_drift():
    rng = np.random.default_rng(20261031)
    covered, positives, estimates = 0, 0, []
    for _ in range(300):
        pairs = meter.paired_deltas(windows_with_drift(2.0, 1.0, rng))
        result = meter.estimate([p['delta']['pkg'] for p in pairs])
        estimates.append(result['mean'])
        covered += result['ci95'][0] <= 2.0 <= result['ci95'][1]
        null = meter.estimate([p['delta']['pkg'] for p in meter.paired_deltas(windows_with_drift(0.0, 1.0, rng))])
        positives += null['significant']
    assert np.mean(estimates) == pytest.approx(2.0, abs=0.1)
    assert 0.90 <= covered / 300 <= 0.98 and positives / 300 <= 0.08


def test_ratio_and_geometric_mean_of_efficiency():
    npu, gpu = meter.estimate([2000, 2100, 1900, 2000]), meter.estimate([1000, 1010, 990, 1000])
    found = meter.ratio(npu, gpu)
    assert found['ratio'] == pytest.approx(2.0) and found['ci95'][0] < 2.0 < found['ci95'][1]
    assert meter.ratio(npu, meter.estimate([1.0, -1.0, 0.5])) is None  # not significant
    overall = meter.geometric_mean([1.0, 4.0])
    assert overall['geomean'] == pytest.approx(2.0) and (overall['min'], overall['max']) == (1.0, 4.0)
    assert meter.geometric_mean([3.0])['ci95'] is None


def test_schedule_shuffles_each_cycle_reproducibly():
    order = meter.schedule(['a', 'b', 'c', 'd'], 5, seed=7)
    assert all(sorted(order[i:i + 4]) == ['a', 'b', 'c', 'd'] for i in range(0, 20, 4))
    assert order == meter.schedule(['a', 'b', 'c', 'd'], 5, seed=7) != meter.schedule(['a', 'b', 'c', 'd'], 5, seed=8)


class FakeClock:
    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        return self.now

    def advance(self):
        self.now += self.step


def test_fixed_window_counts_calls_until_the_last_completion():
    clock = FakeClock(0.01)
    result = meter.fixed_window(clock.advance, 1.0, clock)
    assert (result['calls'], result['half']) == (112, (64, pytest.approx(0.64)))  # clock read every 16 calls
    assert result['per_call_s'] == pytest.approx(0.01)
    assert meter.half_rates(result) == pytest.approx((100.0, 100.0))
    clock = FakeClock(0.01)
    drained = meter.fixed_window(clock.advance, 1.0, clock, finish=lambda: setattr(clock, 'now', clock.now + 0.05))
    assert drained['seconds'] == pytest.approx(1.17)  # queued work drains inside the measured time


def test_run_window_marks_only_inside_the_busy_span():
    clock, events = FakeClock(0.01), []

    def step():
        events.append(('step', clock()))
        clock.advance()

    result = meter.run_window(step, 0.5, 1.0, 0.3, clock, begin=lambda: events.append(('begin', clock())),
                              end=lambda r: events.append(('end', clock())))
    kinds = [kind for kind, _ in events]
    begin, end = kinds.index('begin'), kinds.index('end')
    assert 0 < begin < end < len(kinds) - 1 and 'step' in kinds[begin:end]
    assert events[-1][1] >= events[end][1] + 0.3 - 0.02 and result['calls'] == end - begin - 1


def test_run_window_drains_queued_warmup_work_before_the_counted_window():
    clock, events = FakeClock(0.01), []

    def step():
        events.append('step')
        clock.advance()

    meter.run_window(step, 0.05, 0.2, 0.0, clock, begin=lambda: events.append('begin'),
                     finish=lambda: events.append('finish'))
    assert events[events.index('begin') - 1] == 'finish'


def test_bursts_mark_every_period_and_idle_between_bursts():
    clock, marks, slept = FakeClock(0.001), [], []

    def sleep(seconds):
        slept.append(seconds)
        clock.now += seconds

    result = meter.bursts(clock.advance, 10, 0.5, 3, clock, sleep, begin=lambda: marks.append(clock()),
                          end=lambda: marks.append(clock()))
    assert marks == pytest.approx([0.0, 0.5, 1.0, 1.5], abs=1e-9) and len(result['burst_s']) == 3
    assert result['burst_s'] == pytest.approx([0.01] * 3) and not result['overrun'] and sum(slept) == pytest.approx(1.47)


def test_burst_energy_charges_the_power_tail_to_its_burst():
    t = np.linspace(0, 100, 50001)
    # Idle 30 W with drift; each burst adds 40 W for 0.1 s, then a 12 W tail for 6 s (a GPU's P1 hold).
    extra = np.zeros_like(t)
    starts = [20.0, 30.0, 40.0]
    for start in starts:
        extra[(t >= start) & (t < start + 0.1)] += 40.0
        extra[(t >= start + 0.1) & (t < start + 6.1)] += 12.0
    power = 30 + 0.01 * t + extra
    trace = {'t': t, 'pkg_j': np.concatenate([[0], np.cumsum(power[:-1] * np.diff(t))]), 'pp0_j': 0 * t, 'pp1_j': 0 * t}
    windows = [{'kind': 'idle', 'start': 5.0, 'end': 15.0}, {'kind': 'bursts', 'task': 'k', 'mode': 'gpu_sync',
               'calls': 1, 'period': 10.0, 'count': 3, 'marks': starts + [50.0], 'start': 20.0, 'end': 50.0,
               'burst_s': [0.1] * 3, 'overrun': False}, {'kind': 'idle', 'start': 55.0, 'end': 65.0}]
    for window in windows:
        window['power'] = meter.window_power(trace, window['start'], window['end'])
    found, = meter.burst_energies(trace, windows)
    assert found['energy_j']['pkg'] == pytest.approx(40 * 0.1 + 12 * 6, rel=0.01)  # 76 J per burst, tail included


def test_rounds_reports_per_call_statistics():
    clock = FakeClock(0.002)
    result = meter.rounds(clock.advance, 3, 10, clock)
    assert result['mean_s'] == pytest.approx(0.002) and result['round_spread'] == pytest.approx(0.0, abs=1e-9)


orchestrator = module('research/benchmark_kernel_energy.py')


def test_benchmark_app_output_is_parsed_to_time_per_call():
    text = """[ INFO ] Count:               60123 iterations
[ INFO ] Duration:            10002.15 ms
[ INFO ] Latency:
[ INFO ]    Median:           0.16 ms
[ INFO ]    Average:          0.17 ms
[ INFO ] Throughput:          6011.00 FPS"""
    values = orchestrator.parse_benchmark_app(text)
    assert values['count'] == 60123 and values['per_call_s'] == pytest.approx(10.00215 / 60123)
    assert values['median_ms'] == 0.16 and orchestrator.parse_benchmark_app('nothing') == {}


def test_nvbench_json_is_parsed():
    nvbench = module('research/nvbench_original_kernels.py')
    summary = lambda tag, value: {'tag': tag, 'data': [{'name': 'value', 'value': str(value)}]}
    report = {'benchmarks': [{'name': 'kb_019_relu', 'states': [{'summaries': [
        summary('nv/cold/time/gpu/median', 6.1e-06), summary('nv/batch/time/gpu/mean', 6.0e-06),
        summary('nv/cold/sample_size', 100), summary('nv/unrelated', 1)]}]}]}
    assert nvbench.parse(report) == {'kb_019_relu': {'cold_gpu_median_s': 6.1e-06, 'batch_gpu_mean_s': 6.0e-06,
                                                     'cold_samples': 100.0}}


def campaign(tmp_path, tamper=False):
    evaluation = tmp_path / 'evaluation'
    run = evaluation / 'cases' / 'c1' / 'cycle-1' / 'compile' / 'invocation-x'
    (run / 'target').mkdir(parents=True)
    (run / 'target_execution').mkdir()
    blob = run / 'target' / 'compiled.blob'
    blob.write_bytes(b'npu blob')
    digest = hashlib.sha256(b'npu blob').hexdigest()
    np.savez(run / 'target_execution' / 'normal-0.npz', output=np.arange(4, dtype=np.float16))
    saved = {'target_execution/normal-0.npz': hashlib.sha256((run / 'target_execution' / 'normal-0.npz').read_bytes()).hexdigest()}
    (run / 'execution-request.json').write_text(json.dumps({'artifacts': {'compiled.blob': str(blob)},
                                                            'artifact_sha256': {'compiled.blob': digest},
                                                            'cases': {'normal-0': {}}}))
    (run / 'validation.json').write_text('{}')
    stages = {'target_compile': {'artifacts': {'compiled.blob': digest}}, 'target_execution': {'artifacts': saved}}
    cycle = {'number': 1, 'status': 'passed', 'compile_result': {'artifacts': {'validation.json': str(run / 'validation.json')},
                                                                  'validation': {'stages': stages}}}
    (evaluation / 'cases' / 'c1' / 'report.json').write_text(json.dumps({'cycles': [cycle]}))
    (evaluation / 'campaign.json').write_text(json.dumps({'cases': [{'id': 'c1', 'kernel': 'kb_019_relu'}]}))
    if tamper:
        blob.write_bytes(b'changed')
    return evaluation, run


def test_accepted_blobs_are_hash_checked_and_outputs_compared_bit_for_bit(tmp_path):
    evaluation, run = campaign(tmp_path)
    blob = orchestrator.accepted_blobs(evaluation)['kb_019_relu']
    assert blob['blob'] == run / 'target' / 'compiled.blob' and blob['cases'] == ['normal-0']
    produced = tmp_path / 'produced'
    produced.mkdir()
    np.savez(produced / 'normal-0.npz', output=np.arange(4, dtype=np.float16))
    assert orchestrator.reproduced(run, blob['saved'], produced, ['normal-0']) == {'cases_checked': 1, 'bit_identical': 1}
    np.savez(produced / 'normal-0.npz', output=np.arange(4, dtype=np.float16) + 1)
    assert orchestrator.reproduced(run, blob['saved'], produced, ['normal-0'])['bit_identical'] == 0
    with pytest.raises(ValueError, match='changed since compilation'):
        orchestrator.accepted_blobs(campaign(tmp_path / 'second', tamper=True)[0])


def test_analysis_turns_windows_into_energy_per_inference():
    t = np.linspace(0, 200, 10001)
    # NPU runs add 1 W to the package (inside uncore); GPU runs add 3 W of host package and 20 W of board power.
    windows, clock = [], 0.0
    for kind, mode in [('idle', 'idle'), ('run', 'npu_sync'), ('idle', 'idle'), ('run', 'gpu_sync'), ('idle', 'idle'),
                       ('run', 'npu_sync'), ('idle', 'idle'), ('run', 'gpu_sync'), ('idle', 'idle')]:
        windows.append({'kind': kind, 'task': 'kb_019_relu', 'mode': mode, 'start': clock + 0.5, 'end': clock + 6.5,
                        'per_call_s': 2e-4 if mode == 'npu_sync' else 2e-5})
        clock += 7.0
    extra_pkg, extra_gpu = np.zeros_like(t), np.zeros_like(t)
    for window in windows:
        span = (t >= window['start'] - 0.5) & (t <= window['end'] + 0.5)
        extra_pkg[span] += {'npu_sync': 1.0, 'gpu_sync': 3.0}.get(window['mode'], 0.0)
        extra_gpu[span] += 20.0 if window['mode'] == 'gpu_sync' else 0.0
    pkg_power, pp0_power = 32 + extra_pkg, np.full_like(t, 20.0)
    trace = {'t': t, 'pkg_j': np.concatenate([[0], np.cumsum(pkg_power[:-1] * np.diff(t))]),
             'pp0_j': np.concatenate([[0], np.cumsum(pp0_power[:-1] * np.diff(t))]), 'pp1_j': 0.1 * t,
             'gpu_w': 25 + extra_gpu}
    results = orchestrator.analyse(trace, windows)['kb_019_relu']
    npu, gpu = results['npu_sync'], results['gpu_sync']
    assert all(w['settled'] for w in windows if w['kind'] == 'idle')  # no P-state traced
    assert npu['headline'] == 'soc' and npu['delta_w']['uncore']['mean'] == pytest.approx(1.0, abs=0.02)
    assert npu['energy_uj']['soc']['mean'] == pytest.approx(1.0 * 2e-4 * 1e6, rel=0.02)  # 200 uJ
    assert gpu['headline'] == 'system' and gpu['energy_uj']['system']['mean'] == pytest.approx(23 * 2e-5 * 1e6, rel=0.02)
    assert gpu['delta_w']['gpu']['mean'] == pytest.approx(20.0, rel=0.02) and npu['pairs'] == gpu['pairs'] == 2
    for window in windows:  # the GPU leaves P8 during the second NPU run, as when the desktop wakes it
        window.pop('settled', None)
    trace['gpu_pstate'] = np.where((t > windows[5]['start']) & (t < windows[5]['end']), 3.0, 8.0)
    assert orchestrator.analyse(trace, windows)['kb_019_relu']['npu_sync']['pairs'] == 1
    assert orchestrator.analyse(trace, windows, 'desktop')['kb_019_relu']['npu_sync']['pairs'] == 2
    assert windows[5]['gpu_p8_fraction'] < 0.1 and windows[4]['gpu_p8_fraction'] == 1.0
    del trace['gpu_pstate']
    comparison = orchestrator.compare({'kb_019_relu': results})
    assert comparison['sync_headline']['overall']['geomean'] == pytest.approx(200 / 460, rel=0.03)  # EER(GPU)/EER(NPU)
    assert comparison['throughput_headline']['overall'] is None  # no async or pipelined runs


def test_gpu_sync_gates_are_separate_from_the_ramp_gate():
    def window(per_call, first_half_rate=1.0):
        calls = int(3.0 / per_call)
        half = int(calls / 2 * first_half_rate)
        return {'calls': calls, 'seconds': 3.0, 'per_call_s': per_call, 'half': (half, 1.5)}

    row = {'gpu': {'gpu_sync': [window(19e-6, 0.97)], 'gpu_pipelined': [window(6e-6, 0.97)], 'gpu_e2e': [window(30e-6)],
                   'rounds': {'gpu_sync': {'mean_s': 20.5e-6}}}}
    gates = orchestrator.timing_gates(row, None)
    assert gates['gpu_sync_window_vs_rounds'] == pytest.approx(19 / 20.5 - 1)
    assert 0.05 < gates['gpu_sync_half_rate_change'] < orchestrator.GATE_LIMITS['gpu_sync_half_rate_change']
    assert gates['gpu_half_rate_change'] > orchestrator.GATE_LIMITS['gpu_half_rate_change']  # pipelined ramp fails


class FakeHelper:
    def __init__(self, states):
        self.states = iter(states)

    def call(self, **command):
        assert command == {'cmd': 'latest'}
        return {'t': 0.0, 'sample': {'gpu_pstate': next(self.states)}}


def test_meter_waits_for_the_gpu_idle_pstate(monkeypatch):
    clock = iter(np.arange(0, 100, 0.1))
    monkeypatch.setattr(orchestrator.time, 'perf_counter', lambda: next(clock))
    monkeypatch.setattr(orchestrator.time, 'sleep', lambda seconds: None)
    waiting = orchestrator.Meter.__new__(orchestrator.Meter)
    waiting.helper = FakeHelper([1, 1, 5, 8, 1] + [8] * 50)
    assert 1.0 < waiting.wait_gpu_idle(cap=20, hold=0.5) < 3.0  # the P8 interrupted by P1 does not count
    waiting.helper = FakeHelper([1] * 1000)
    assert waiting.wait_gpu_idle(cap=2, hold=0.5) >= 2
    waiting.helper = FakeHelper([None])
    assert waiting.wait_gpu_idle(cap=2) == pytest.approx(0, abs=0.5)
    slept = []
    monkeypatch.setattr(orchestrator.time, 'sleep', slept.append)
    desktop = SimpleNamespace(baseline='desktop', desktop_settle=10.0, settle=20.0)
    waiting.helper = FakeHelper([1] * 1000)  # the desktop keeps the GPU busy: no P8 wait, a fixed one
    assert orchestrator.settle_gpu(waiting, desktop) == 10.0 and slept == [10.0]


def test_no_code_reads_the_broken_nvml_energy_counter():
    for path in ('scripts/windows_power_sampler.py', 'research/benchmark_kernel_energy.py', 'research/energy_meter.py'):
        assert 'GetTotalEnergyConsumption' not in (ROOT / path).read_text()


def test_sweep_configs_have_consistent_shapes_and_references():
    sweep = module('research/sweep_kernels.py')
    assert sweep.SWEEP['relu_b1']['inputs'] == {'x': [1, 4, 65]}  # kb_019_relu's shape
    assert sweep.SWEEP['matmul_1024']['flops'] == 2 * 1024 ** 3 and sweep.SWEEP['relu_b4096']['instances'] == 4096
    assert all(base in sweep.SWEEP for base in sweep.BATCH_BASE.values())
    for name in ('add_n4096', 'relu_b64', 'softmax_b16', 'matmul_b16'):
        spec = sweep.SWEEP[name]
        feed = sweep.inputs(spec)
        want = sweep.reference(spec, feed)
        assert list(want.shape) == spec['output'] and want.dtype == np.float16
        assert sweep.check(spec, want, want)['within_tolerance'] == 1.0
    softmax = sweep.reference(sweep.SWEEP['softmax_32x128'], sweep.inputs(sweep.SWEEP['softmax_32x128']))
    assert np.allclose(softmax.astype(np.float32).sum(-1), 1, atol=0.02)


def test_progress_counts_runs_and_estimates_the_time_left(monkeypatch, capsys):
    monkeypatch.setattr(orchestrator, 'tqdm', None)  # the plain fallback; tqdm is optional
    clock = iter([0.0, 0.0, 60.0])
    monkeypatch.setattr(orchestrator.time, 'monotonic', lambda: next(clock))
    progress = orchestrator.Progress()
    progress.log('before start')
    progress.start(4, 'sweep')
    progress.step('relu_b1 npu_sync')
    out = capsys.readouterr()
    assert out.out == 'before start\n'
    assert out.err.strip() == '[1/4] 1.0 min elapsed, about 3 min left: relu_b1 npu_sync'
