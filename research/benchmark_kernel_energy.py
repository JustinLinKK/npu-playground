"""Measure time and energy of a campaign's generated NPU kernels and the dataset's original GPU kernels.

Time: per-call rounds and fixed wall-clock windows (N runs in T seconds gives T/N), synchronous and pipelined, on
the Intel NPU (Windows OpenVINO, scripts/intel_npu_bench_worker.py) and the NVIDIA GPU (launch closures from
research/time_original_gpu_kernels.py), cross-checked with OpenVINO benchmark_app and NVIDIA NVBench. Energy:
interleaved idle and run windows on one power clock (scripts/windows_power_sampler.py: Windows Energy Meter RAPL
rails and NVML board power); a run's power minus its neighbouring idles, times T/N, is the energy of one inference.
Method, formulas and caveats: docs/time-energy-measurement.md.

Run in the GPU venv (setup in research/time_original_gpu_kernels.py) with CC set for Triton:
    CC=~/.cache/npu-playground-gpu-venv/zigcc/cc ~/.cache/npu-playground-gpu-venv/bin/python \\
        research/benchmark_kernel_energy.py --campaign RUN_ROOT/evaluation --validate
    ... --campaign RUN_ROOT/evaluation --timing-only --tasks all --cross-check
    ... --campaign RUN_ROOT/evaluation --tasks subset        (energy: leave the machine idle)
    ... --reanalyse RUN_ROOT/kernel-energy-<UTC>               (recompute an energy run from its trace)
"""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'research'), str(ROOT / 'scripts')]

import energy_meter  # noqa: E402
import time_original_gpu_kernels as gpu_kernels  # noqa: E402
from wsl_npu_bridge import LOCK, windows_path  # noqa: E402

SAMPLER = ROOT / 'scripts' / 'windows_power_sampler.py'
WORKER = ROOT / 'scripts' / 'intel_npu_bench_worker.py'
NVBENCH = ROOT / 'research' / 'nvbench_original_kernels.py'
BENCHMARK_APP = ROOT / 'scripts' / 'benchmark_app_blob.py'
ENERGY_LOCK = ROOT / '.npu-agent' / 'energy.lock'
# One task per operation family; all are correct on both devices in the 2026-10-21 FP16 campaign.
SUBSET = ['kb_019_relu', 'kb_049_max_reduction', 'kb_027_selu', 'kb_036_rms_norm', 'cuda_tiled_matmul',
          'cuda_reduce_sum', 'hip_image_convolution', 'triton_fused_softmax']
NPU_MODES, GPU_MODES = ('npu_sync', 'npu_async'), ('gpu_sync', 'gpu_pipelined')
# Headline energy per device: the NPU lives in the package; a GPU run also costs host package power.
HEADLINE = {'npu': 'soc', 'gpu': 'system'}
ENERGY_RAILS = {'soc': ('pkg',), 'uncore': ('uncore',), 'host_cores': ('pp0',), 'gpu_board': ('gpu',),
                'system': ('pkg', 'gpu')}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path: Path, root: Path) -> Path:
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'missing or unsafe campaign artifact: {path}')
    return path


def accepted_blobs(evaluation: Path) -> dict[str, dict]:
    """Each task's accepted NPU blob, checked against its compile-time hash (plain JSON: no pydantic in this venv)."""
    campaign = json.loads((evaluation / 'campaign.json').read_text())
    found = {}
    for case in campaign['cases']:
        report = evaluation / 'cases' / case['id'] / 'report.json'
        cycles = json.loads(report.read_text())['cycles'] if report.exists() else []
        cycle = next((c for c in cycles if c['status'] == 'passed'), None)
        if cycle is None:
            continue
        compiled = cycle['compile_result']
        stages = compiled['validation']['stages']
        run = inside(Path(compiled['artifacts']['validation.json']), evaluation).parent
        request = inside(run / 'execution-request.json', evaluation)
        payload = json.loads(request.read_text())
        blob = inside(Path(payload['artifacts']['compiled.blob']), evaluation)
        if not sha256(blob) == payload['artifact_sha256']['compiled.blob'] == stages['target_compile']['artifacts']['compiled.blob']:
            raise ValueError(f'compiled blob changed since compilation: {case["kernel"]}')
        found[case['kernel']] = {'request': request, 'blob': blob, 'run': run, 'cases': list(payload['cases']),
                                 'saved': stages.get('target_execution', {}).get('artifacts', {})}
    return found


def reproduced(run: Path, saved: dict, produced: Path, cases: list[str]) -> dict:
    """Compare fresh NPU outputs with the hash-checked outputs saved when the campaign executed the blob."""
    checked = identical = 0
    for case in cases:
        old = run / 'target_execution' / f'{case}.npz'
        if not old.is_file() or saved.get(f'target_execution/{case}.npz') != sha256(old):
            continue
        before, after = gpu_kernels.load(old), gpu_kernels.load(produced / f'{case}.npz')
        checked += 1
        identical += set(before) == set(after) and all(np.array_equal(before[k], after[k], equal_nan=True) for k in before)
    return {'cases_checked': checked, 'bit_identical': identical}


class Helper:
    """A Windows-side helper process speaking one JSON object per line."""

    def __init__(self, command: list[str], log: Path):
        self.log = log
        self.stream = log.open('ab')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stream)

    def send(self, **command) -> None:
        self.process.stdin.write((json.dumps(command) + '\n').encode())
        self.process.stdin.flush()

    def receive(self) -> dict:
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'helper exited; see {self.log}')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(f'{self.log.stem}: {reply["error"]}')
        return reply

    def call(self, **command) -> dict:
        self.send(**command)
        return self.receive()

    def close(self, **command) -> None:
        try:
            self.call(**command)
            self.process.wait(timeout=30)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            self.process.kill()
        self.stream.close()


class Meter:
    """Client of the power sampler. mark() returns at once; collect() reads the marks' sampler times."""

    def __init__(self, args, output: Path):
        self.helper = Helper([args.windows_python, '-B', windows_path(SAMPLER), '--interval', str(args.interval)],
                             output / 'sampler.log')
        self.ready, self.interval = self.helper.receive(), args.interval
        self.trace = output / 'trace.csv'
        self.trace_windows = windows_path(output) + '\\trace.csv'

    def mark(self, *_) -> None:
        self.helper.send(cmd='mark')

    def collect(self, count: int) -> list[float]:
        return [self.helper.receive()['t'] for _ in range(count)]

    def round_trip(self, count: int = 20) -> float:
        times = []
        for _ in range(count):
            started = time.perf_counter()
            self.helper.call(cmd='ping')
            times.append(time.perf_counter() - started)
        return statistics.median(times)

    def wait_gpu_idle(self, cap: float, hold: float = 0.5) -> float:
        """Wait until the GPU has sat in its idle P-state for `hold` seconds, at most `cap` seconds; return the wait."""
        started, idle_since = time.perf_counter(), None
        while (waited := time.perf_counter() - started) < cap:
            state = self.helper.call(cmd='latest')['sample'].get('gpu_pstate')
            if state is None:  # no GPU traced
                return waited
            if state < energy_meter.GPU_IDLE_PSTATE:
                idle_since = None
            elif idle_since is None:
                idle_since = time.perf_counter()
            elif time.perf_counter() - idle_since >= hold:
                return waited
            time.sleep(0.1)
        return waited

    def flush(self) -> None:
        time.sleep(3 * self.interval)  # the newest sample can predate the last mark by one interval
        self.helper.call(cmd='flush', path=self.trace_windows)

    def close(self) -> None:
        self.flush()
        self.helper.close(cmd='stop')


def read_trace(path: Path) -> dict:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))

    def column(name, scale=1.0):
        values = np.array([float(row[name]) if row[name] not in ('', 'None') else np.nan for row in rows])
        if np.isnan(values).all():
            return None
        valid = ~np.isnan(values)  # fill gaps from neighbouring samples
        return np.interp(np.arange(len(values)), np.flatnonzero(valid), values[valid]) * scale

    trace = {'t': column('t'), 'gpu_w': column('gpu_w'), 'gpu_avg_w': column('gpu_avg_w'), 'cpu_util': column('cpu_util'),
             'gpu_pstate': column('gpu_pstate')}
    for rail in energy_meter.RAILS:
        trace[rail + '_j'] = column(rail + '_pwh', energy_meter.PWH_TO_J)
    return trace


def idle_window(meter: Meter, seconds: float, **labels) -> dict:
    meter.mark()
    time.sleep(seconds)
    meter.mark()
    start, end = meter.collect(2)
    return {'kind': 'idle', 'start': start, 'end': end, **labels}


def run_record(start: float, end: float, result: dict, **labels) -> dict:
    first, second = energy_meter.half_rates(result)
    return {'kind': 'run', 'start': start, 'end': end, 'calls': result['calls'], 'seconds': result['seconds'],
            'per_call_s': result['per_call_s'], 'half_rate_change': second / first - 1, **labels}


def npu_window(worker: Helper, meter: Meter | None, mode: str, warm: float, seconds: float, tail: float, nireq: int) -> dict:
    worker.send(cmd='window', mode=mode.removeprefix('npu_'), warm=warm, seconds=seconds, tail=tail, nireq=nireq)
    while (reply := worker.receive()).get('event') != 'done':
        if meter is not None:
            meter.mark()  # 'begin' and 'end', forwarded the moment they arrive
    return reply


def gpu_steps(task) -> dict:
    def sync():
        task.launch()
        task.synchronize()

    def end_to_end():
        task.upload(task.first)
        task.download(task.launch())

    return {'gpu_sync': (sync, None), 'gpu_pipelined': (task.launch, task.synchronize), 'gpu_e2e': (end_to_end, None)}


def warm_gpu(steps: dict, calls: int, seconds: float) -> None:
    """Run each GPU mode for at least `calls` calls and `seconds`: the first pipelined launches of a process speed
    up over about 3 s (host side; GPU clocks stay fixed), which 200 calls (about 1 ms) do not cover."""
    for step, finish in steps.values():
        deadline, done = time.perf_counter() + seconds, 0
        while done < calls or time.perf_counter() < deadline:
            for _ in range(16):
                step()
            done += 16
        if finish:
            finish()


def gpu_window(step, finish, meter: Meter | None, warm: float, seconds: float, tail: float) -> dict:
    mark = meter.mark if meter is not None else None
    return energy_meter.run_window(step, warm, seconds, tail, time.perf_counter, begin=mark, end=mark, finish=finish)


def device_events(task, calls: int = 2000) -> dict:
    samples = []
    for _ in range(calls):
        start, end = task.event(), task.event()
        start.record()
        task.launch()
        end.record()
        end.synchronize()
        samples.append(task.elapsed(start, end) / 1e3)
    return {'mean_s': float(np.mean(samples)), 'p50_s': float(np.median(samples))}


def start_worker(args, output: Path, blob: dict, kernel: str) -> tuple[Helper, dict, dict]:
    worker = Helper([args.windows_python, '-B', windows_path(WORKER)], output / 'npu-worker.log')
    info = worker.call(cmd='load', request=windows_path(blob['request']), blob=windows_path(blob['blob']))
    produced = output / 'npu-outputs' / kernel
    produced.mkdir(parents=True, exist_ok=True)
    worker.call(cmd='cases', out=windows_path(produced))
    check = reproduced(blob['run'], blob['saved'], produced, blob['cases'])
    worker.call(cmd='warmup', count=args.npu_warmup)
    return worker, info, check


def benchmark_app(args, blob: Path, mode: str, folder: Path) -> dict:
    """OpenVINO's own benchmark tool on the same blob (random inputs, filled once), through scripts/benchmark_app_blob.py."""
    folder.mkdir(parents=True, exist_ok=True)
    api = ['-api', 'sync', '-nireq', '1'] if mode == 'npu_sync' else ['-api', 'async', '-nireq', str(args.nireq)]
    command = [args.windows_python, '-B', windows_path(BENCHMARK_APP), '-m', windows_path(blob), '-d', 'NPU',
               '-t', str(args.cross_check_seconds), *api,
               '-hint', 'none', '-report_type', 'no_counters', '-json_stats', '-report_folder', windows_path(folder)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=600)
    (folder / 'stdout.txt').write_text(process.stdout + process.stderr)
    return parse_benchmark_app(process.stdout)


def parse_benchmark_app(text: str) -> dict:
    found = {key: re.search(pattern, text) for key, pattern in (
        ('count', r'Count:\s+(\d+)'), ('duration_ms', r'Duration:\s+([\d.]+) ms'),
        ('median_ms', r'Median:\s+([\d.]+) ms'), ('average_ms', r'Average:\s+([\d.]+) ms'),
        ('throughput_fps', r'Throughput:\s+([\d.]+) FPS'))}
    values = {key: float(match.group(1)) for key, match in found.items() if match}
    if 'count' in values and 'duration_ms' in values:
        values['per_call_s'] = values['duration_ms'] / 1e3 / values['count']
    return values


def nvbench(args, kernels: list[str], folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / 'nvbench.json'
    process = subprocess.run([sys.executable, str(NVBENCH), '--dataset', str(args.dataset), '--tasks', ','.join(kernels),
                              '--json', str(report)], capture_output=True, text=True, timeout=3600)
    (folder / 'nvbench.txt').write_text(process.stdout + process.stderr)
    import nvbench_original_kernels

    return nvbench_original_kernels.parse(json.loads(report.read_text())) if report.exists() else {}


def relative(a, b):
    return a / b - 1 if a is not None and b else None


def time_task(args, output: Path, kernel: str, blob: dict | None, cases: list[str]) -> dict:
    row = {'kernel': kernel}
    if blob is not None:
        worker, info, check = start_worker(args, output, blob, kernel)
        try:
            npu = {'info': info, 'reproduced': check,
                   'rounds': worker.call(cmd='rounds', count=args.rounds, calls=args.npu_round_calls)}
            for mode in NPU_MODES:
                npu[mode] = [npu_window(worker, None, mode, args.warm, args.timing_seconds, 0, args.nireq)
                             for _ in range(args.timing_windows)]
        finally:
            worker.close(cmd='quit')
        if args.cross_check:
            npu['benchmark_app'] = {mode: benchmark_app(args, blob['blob'], mode, output / 'crosscheck' / kernel / mode)
                                    for mode in NPU_MODES}
        row['npu'] = npu
    try:
        task = gpu_kernels.prepare(args.dataset / kernel, cases)
    except Exception as exc:  # an original source that does not compile is reported, never patched
        row['gpu'] = {'error': f'{type(exc).__name__}: {str(exc).strip().splitlines()[-1]}'}
        return row
    gpu = {'correct': task.passed == task.cases_run, 'cases_passed': task.passed, 'launch': task.launch_shape}
    steps = gpu_steps(task)
    warm_gpu(steps, args.gpu_warmup, args.gpu_warmup_seconds)
    gpu['device_events'] = device_events(task)
    gpu['rounds'] = {mode: energy_meter.rounds(steps[mode][0], args.rounds, args.gpu_round_calls, time.perf_counter)
                     for mode in ('gpu_sync', 'gpu_e2e')}
    for mode, (step, finish) in steps.items():
        gpu[mode] = [gpu_window(step, finish, None, args.warm, args.timing_seconds, 0) for _ in range(args.timing_windows)]
    row['gpu'] = gpu
    return row


def timing_gates(row: dict, nv: dict | None) -> dict:
    """Agreement between independent estimates of the same time; see the method document for the thresholds."""
    gates, npu, gpu = {}, row.get('npu'), row.get('gpu', {})

    def per_call(windows):
        return statistics.mean(w['per_call_s'] for w in windows)

    def halves(windows):
        return max(abs(b / a - 1) for a, b in map(energy_meter.half_rates, windows))

    if npu:
        gates['npu_sync_window_vs_rounds'] = relative(per_call(npu['npu_sync']), npu['rounds']['mean_s'])
        gates['npu_half_rate_change'] = max(halves(npu[mode]) for mode in NPU_MODES)
        for mode in NPU_MODES:
            tool = npu.get('benchmark_app', {}).get(mode, {})
            gates[f'{mode}_vs_benchmark_app'] = relative(per_call(npu[mode]), tool.get('per_call_s'))
    if 'gpu_sync' in gpu:
        gates['gpu_sync_window_vs_rounds'] = relative(per_call(gpu['gpu_sync']), gpu['rounds']['gpu_sync']['mean_s'])
        gates['gpu_sync_half_rate_change'] = halves(gpu['gpu_sync'])
        gates['gpu_half_rate_change'] = max(halves(gpu[mode]) for mode in ('gpu_pipelined', 'gpu_e2e'))
        if nv:
            gates['gpu_pipelined_vs_nvbench_batch'] = relative(per_call(gpu['gpu_pipelined']), nv.get('batch_gpu_mean_s'))
            # Not a gate: events around one launch on an idle stream also time the host's launch latency.
            if nv.get('cold_gpu_median_s'):
                gates['gpu_launch_gap_us'] = (gpu['device_events']['p50_s'] - nv['cold_gpu_median_s']) * 1e6
    return gates


# GPU sync calls on WSL2 switch between ~17.5 and ~22-27 us states lasting 0.1-1 s, so its gates allow 10%.
GATE_LIMITS = {'npu_sync_window_vs_rounds': 0.05, 'gpu_sync_window_vs_rounds': 0.10, 'npu_sync_vs_benchmark_app': 0.10,
               'npu_async_vs_benchmark_app': 0.10, 'gpu_pipelined_vs_nvbench_batch': 0.15, 'npu_half_rate_change': 0.05,
               'gpu_sync_half_rate_change': 0.10, 'gpu_half_rate_change': 0.05}


def summarize(pairs: list[dict]) -> dict:
    """Per task and mode: time per call, paired power deltas per rail, and energy per inference (uJ)."""
    rails = list(pairs[0]['delta'])
    out = {'pairs': len(pairs), 'per_call_us': statistics.mean(p['per_call_s'] for p in pairs) * 1e6,
           'idle_w': {r: statistics.mean(p['idle'][r] for p in pairs) for r in rails},
           'delta_w': {r: energy_meter.estimate([p['delta'][r] for p in pairs]) for r in rails},
           'energy_uj': {}, 'gross_uj': {}}
    for name, parts in ENERGY_RAILS.items():
        if all(r in rails for r in parts):
            out['energy_uj'][name] = energy_meter.estimate(
                [sum(p['delta'][r] for r in parts) * p['per_call_s'] * 1e6 for p in pairs])
            out['gross_uj'][name] = statistics.mean(sum(p['power'][r] for r in parts) * p['per_call_s'] * 1e6 for p in pairs)
    headline = out['energy_uj'].get(HEADLINE[pairs[0]['mode'].split('_')[0]])
    out['headline'] = HEADLINE[pairs[0]['mode'].split('_')[0]]
    out['inferences_per_joule'] = 1e6 / headline['mean'] if headline and headline['significant'] and headline['mean'] > 0 else None
    return out


def analyse(trace: dict, windows: list[dict]) -> dict:
    for window in windows:
        window['power'] = energy_meter.window_power(trace, window['start'], window['end'])
        # An idle window inside the GPU's post-work P1 hold is not a baseline. A GPU wake-up during an NPU run means
        # something else on the machine used it (display, another process), so that run is dropped.
        if window['kind'] == 'idle' or window['mode'].startswith('npu'):
            window['settled'] = energy_meter.gpu_settled(trace, window['start'], window['end'])
    results = {}
    for kernel in dict.fromkeys(w['task'] for w in windows):
        pairs = energy_meter.paired_deltas([w for w in windows if w['task'] == kernel])
        results[kernel] = {mode: summarize([p for p in pairs if p['mode'] == mode])
                           for mode in dict.fromkeys(p['mode'] for p in pairs)}
    return results


def energy_task(args, meter: Meter, output: Path, index: int, kernel: str, blob: dict, cases: list[str]) -> tuple[list, dict]:
    worker, info, check = start_worker(args, output, blob, kernel)
    notes = {'npu': info, 'reproduced': check}
    try:
        task = gpu_kernels.prepare(args.dataset / kernel, cases)
        notes['gpu_correct'] = task.passed == task.cases_run
    except Exception as exc:
        task, notes['gpu_error'] = None, f'{type(exc).__name__}: {exc}'
    modes = list(NPU_MODES) + (list(GPU_MODES) if task is not None and notes['gpu_correct'] else [])
    steps = gpu_steps(task) if task is not None else {}
    if 'gpu_sync' in modes:
        warm_gpu(steps, args.gpu_warmup, args.gpu_warmup_seconds)
    settle = meter.wait_gpu_idle(args.settle)  # the GPU checks and warm-up leave it in P1
    windows = [idle_window(meter, args.idle_seconds, task=kernel, mode='idle', cycle=-1, settle_s=settle)]
    try:
        for position, mode in enumerate(energy_meter.schedule(modes, args.cycles, args.seed + index)):
            if mode.startswith('npu'):
                result = npu_window(worker, meter, mode, args.warm, args.run_seconds, args.tail, args.nireq)
            else:
                step, finish = steps[mode]
                result = gpu_window(step, finish, meter, args.warm, args.run_seconds, args.tail)
            start, end = meter.collect(2)
            windows.append(run_record(start, end, result, task=kernel, mode=mode, cycle=position // len(modes)))
            settle = meter.wait_gpu_idle(args.settle) if mode.startswith('gpu') else 0.0
            time.sleep(args.guard)
            windows.append(idle_window(meter, args.idle_seconds, task=kernel, mode='idle', cycle=position // len(modes),
                                       settle_s=settle))
            print(f'  {kernel:24s} {mode:14s} {result["per_call_s"] * 1e6:8.1f} us/call', flush=True)
    finally:
        worker.close(cmd='quit')
        meter.flush()
    return windows, notes


def validate(args, meter: Meter, output: Path) -> dict:
    """V1 idle noise, V3 CPU load steps, V5 GPU power agreement and settling (docs/time-energy-measurement.md)."""
    report = {}
    print(f'V1: {args.validate_idle:.0f} s idle', flush=True)
    idle = idle_window(meter, args.validate_idle, task='V1', mode='idle', cycle=0)
    windows = []
    print('V3: CPU load steps', flush=True)
    for repeat in range(3):
        for workers in (1, 2, 4):
            time.sleep(args.guard)
            windows.append(idle_window(meter, args.idle_seconds, task='V3', mode='idle', cycle=repeat))
            deadline = time.time() + args.warm + args.run_seconds + args.tail
            busy = [multiprocessing.get_context('fork').Process(target=spin, args=(deadline,)) for _ in range(workers)]
            for process in busy:
                process.start()
            time.sleep(args.warm)
            meter.mark()
            time.sleep(args.run_seconds)
            meter.mark()
            for process in busy:
                process.join()
            start, end = meter.collect(2)
            windows.append({'kind': 'run', 'start': start, 'end': end, 'task': 'V3', 'mode': f'cpu_{workers}', 'cycle': repeat})
    windows.append(idle_window(meter, args.idle_seconds, task='V3', mode='idle', cycle=3))
    print('V5: GPU load', flush=True)
    import torch

    matrix = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    meter.wait_gpu_idle(args.settle)
    before = idle_window(meter, args.idle_seconds, task='V5', mode='idle', cycle=0)
    smi = subprocess.Popen(['nvidia-smi', '--query-gpu=power.draw.instant,power.draw.average', '--format=csv,noheader,nounits',
                            '-lms', '100'], stdout=subprocess.PIPE, text=True)
    load_seconds, started = 20.0, time.time()
    marked = []
    while (elapsed := time.time() - started) < load_seconds:
        for _ in range(10):
            matrix @ matrix
        torch.cuda.synchronize()
        if not marked and elapsed > 3:
            meter.mark()
            marked.append(True)
        if len(marked) == 1 and elapsed > load_seconds - 2:
            meter.mark()
            marked.append(True)
    smi.terminate()
    smi_samples = []
    for line in smi.communicate()[0].splitlines():
        try:
            smi_samples.append([float(value) for value in line.split(',')])
        except ValueError:  # '[N/A]' or a partial last line
            continue
    load_start, load_end = meter.collect(2)
    after = idle_window(meter, 15.0, task='V5', mode='idle', cycle=1)
    meter.flush()
    trace = read_trace(meter.trace)
    t = trace['t']
    report['V1'] = noise_table(trace, idle['start'], idle['end'], args)
    for window in windows:
        window['power'] = energy_meter.window_power(trace, window['start'], window['end'])
    pairs = energy_meter.paired_deltas(windows)
    report['V3'] = {mode: {rail: energy_meter.estimate([p['delta'][rail] for p in pairs if p['mode'] == mode])
                           for rail in ('pkg', 'pp0', 'pp1', 'uncore')} for mode in ('cpu_1', 'cpu_2', 'cpu_4')}
    middle = smi_samples[len(smi_samples) // 5: -len(smi_samples) // 10 or None]
    idle_gpu = energy_meter.sampled_power(t, trace['gpu_w'], before['start'], before['end'])
    rolling = [(x, energy_meter.sampled_power(t, trace['gpu_w'], x, x + 1.0))
               for x in np.arange(after['start'], min(after['end'], t[-1]) - 1.0, 0.1)]
    settled = next((x - after['start'] for x, power in rolling
                    if power <= idle_gpu + 2.0 and energy_meter.gpu_settled(trace, x, x + 1.0)), None)
    report['V5'] = {'sampler_instant_w': energy_meter.sampled_power(t, trace['gpu_w'], load_start, load_end),
                    'sampler_average_w': energy_meter.sampled_power(t, trace['gpu_avg_w'], load_start, load_end),
                    'nvidia_smi_instant_w': statistics.mean(s[0] for s in middle) if middle else None,
                    'nvidia_smi_average_w': statistics.mean(s[1] for s in middle) if middle else None,
                    'idle_p8_w': idle_gpu, 'seconds_to_p8_after_load': settled}
    return report


def spin(deadline: float) -> None:
    while time.time() < deadline:
        pass


def noise_table(trace: dict, start: float, end: float, args) -> dict:
    """Spread of window-mean power during idle, by window length, and the effect detectable at the chosen K."""
    table = {}
    for length in (1.0, 2.0, 5.0, 10.0):
        edges = np.arange(start, end - length + 1e-9, length)
        if len(edges) < 3:
            continue
        powers = [energy_meter.window_power(trace, a, a + length) for a in edges]
        table[f'{length:g}s'] = {rail: float(np.std([p[rail] for p in powers], ddof=1)) for rail in powers[0]}

    def nearest(seconds):
        return table[min(table, key=lambda key: abs(float(key[:-1]) - seconds))]

    run, rest = nearest(args.run_seconds), nearest(args.idle_seconds)
    df = args.cycles - 1
    factor = (energy_meter.quantile(energy_meter.T975, df) + energy_meter.quantile(energy_meter.T80, df)) / args.cycles ** 0.5
    mde = {rail: factor * (run[rail] ** 2 + rest[rail] ** 2 / 2) ** 0.5 for rail in run}
    return {'sd_w_by_window': table, 'mde80_w_at_defaults': mde, 'cycles': args.cycles,
            'run_seconds': args.run_seconds, 'idle_seconds': args.idle_seconds}


def number(value, digits=1, unit=''):
    return 'n/a' if value is None else f'{value:.{digits}f}{unit}'


def interval(estimate: dict, digits=2) -> str:
    if estimate is None:
        return 'n/a'
    if estimate.get('ci95') is None:  # a single value has no interval
        return f'{estimate["mean"]:.{digits}f} (n = {estimate["n"]})'
    low, high = estimate['ci95']
    return f'{estimate["mean"]:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]'


def energy_cell(estimate: dict) -> str:
    if estimate is None or estimate.get('ci95') is None:
        return 'n/a'
    return interval(estimate) if estimate['significant'] else f'< {estimate["upper95"]:.2f} (n.s.)'


def write_timing(path: Path, rows: list[dict], environment: dict) -> None:
    lines = ['# Kernel timing', '', 'Per-call time in microseconds. *Rounds*: mean of individually timed calls. '
             '*Window*: fixed wall-clock window, T/N. Sync = one call at a time; async/pipelined = queued calls. '
             'Method: docs/time-energy-measurement.md.', '', f'Environment: `{json.dumps(environment)}`', '',
             '| Task | NPU sync rounds | NPU sync window | NPU async window | benchmark_app sync / async | GPU events / NVBench cold (device) | NVBench batch | GPU sync rounds | GPU sync window | GPU pipelined window | GPU e2e window | Gates |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']

    def window(entry, mode):
        return number(statistics.mean(w['per_call_s'] for w in entry[mode]) * 1e6) if entry and mode in entry else 'n/a'

    for row in rows:
        npu, gpu, nv, gates = row.get('npu'), row.get('gpu', {}), row.get('nvbench', {}), row.get('gates', {})
        tool = (npu or {}).get('benchmark_app', {})
        tool_cell = ' / '.join(number(tool.get(m, {}).get('per_call_s', 0) * 1e6 if tool.get(m, {}).get('per_call_s') else None)
                               for m in NPU_MODES)
        failed = [name for name, value in gates.items()
                  if name in GATE_LIMITS and value is not None and abs(value) > GATE_LIMITS[name]]
        device = (f'{number(gpu["device_events"]["p50_s"] * 1e6, 2)} / {number(nv["cold_gpu_median_s"] * 1e6, 2) if nv.get("cold_gpu_median_s") else "n/a"}'
                  if 'device_events' in gpu else gpu.get('error', 'n/a'))
        lines.append(f'| {row["kernel"]} | {number(npu["rounds"]["mean_s"] * 1e6) if npu else "n/a"} | {window(npu, "npu_sync")} | '
                     f'{window(npu, "npu_async")} | {tool_cell} | {device} | '
                     f'{number(nv["batch_gpu_mean_s"] * 1e6, 2) if nv.get("batch_gpu_mean_s") else "n/a"} | '
                     f'{number(gpu["rounds"]["gpu_sync"]["mean_s"] * 1e6) if "rounds" in gpu else "n/a"} | {window(gpu, "gpu_sync")} | '
                     f'{window(gpu, "gpu_pipelined")} | {window(gpu, "gpu_e2e")} | {"ok" if not failed else "FAIL: " + ", ".join(failed)} |')
    lines += ['', 'Gate limits (|relative difference|): ' + ', '.join(f'{k} {v:.0%}' for k, v in GATE_LIMITS.items()), '']
    path.write_text('\n'.join(lines))


def write_energy(path: Path, results: dict, notes: dict, environment: dict, args, windows: list[dict]) -> dict:
    by_mode = {}
    for kernel, modes in results.items():
        for mode, summary in modes.items():
            headline = summary['energy_uj'].get(summary['headline'])
            if headline:
                by_mode.setdefault(mode, []).append(headline['mean'])
    overall = {mode: energy_meter.estimate(values) for mode, values in by_mode.items()}
    idles = [w for w in windows if w['kind'] == 'idle']
    npu_runs = [w for w in windows if w['kind'] == 'run' and w['mode'].startswith('npu')]
    lines = ['# Kernel energy', '',
             f'{args.cycles} cycles per task of {args.run_seconds:g} s run windows between {args.idle_seconds:g} s idle '
             'windows, run order shuffled per cycle. Energy per inference = (run power − mean of the neighbouring idle '
             'powers) × time per call. NPU headline: package (SoC) energy; GPU headline: package + GPU board ("system"). '
             'Brackets are 95% confidence intervals; "n.s." means the interval includes zero and the value shown is a '
             'one-sided 95% upper bound; n is the number of paired runs. Method and caveats: '
             'docs/time-energy-measurement.md.', '',
             f'Idle windows used as baselines: {sum(w.get("settled", True) for w in idles)} of {len(idles)}; the others '
             f'were skipped because the GPU was not in its idle P-state (P{energy_meter.GPU_IDLE_PSTATE}) for '
             f'{energy_meter.SETTLED_FRACTION:.0%} of the window. NPU runs dropped because the GPU woke up during '
             f'them (another program used it): {sum(not w.get("settled", True) for w in npu_runs)} of {len(npu_runs)}.', '',
             f'Environment: `{json.dumps(environment)}`', '',
             '## Mean over tasks (headline energy per inference, µJ)', '', '| Mode | Tasks | Mean [95% CI] |', '| --- | ---: | ---: |']
    lines += [f'| {mode} | {estimate["n"]} | {interval(estimate)} |' for mode, estimate in overall.items()]
    lines += ['', '## Per task', '',
              '| Task | Mode | n | µs/call | ΔPKG W | ΔUncore W | ΔCores W | ΔGPU W | Energy µJ/inference | Inferences/J | Gross µJ |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for kernel, modes in results.items():
        for mode, s in modes.items():
            delta = s['delta_w']
            lines.append(f'| {kernel} | {mode} | {s["pairs"]} | {number(s["per_call_us"])} | {interval(delta.get("pkg"))} | '
                         f'{interval(delta.get("uncore"))} | {interval(delta.get("pp0"))} | {interval(delta.get("gpu"))} | '
                         f'{energy_cell(s["energy_uj"].get(s["headline"]))} | {number(s["inferences_per_joule"], 0)} | '
                         f'{number(s["gross_uj"].get(s["headline"]))} |')
    flagged = {k: v['reproduced'] for k, v in notes.items() if v['reproduced']['bit_identical'] != v['reproduced']['cases_checked']}
    lines += ['', f'NPU outputs not bit-identical to the campaign\'s saved outputs: {json.dumps(flagged) if flagged else "none"}.', '']
    path.write_text('\n'.join(lines))
    return overall


def write_validation(path: Path, report: dict) -> None:
    v1, v3, v5 = report['V1'], report['V3'], report['V5']
    lines = ['# Instrument validation', '', '## V1: idle noise', '',
             'Standard deviation of window-mean power during idle (W), by window length:', '',
             '| Window | ' + ' | '.join(next(iter(v1['sd_w_by_window'].values()))) + ' |',
             '| --- |' + ' ---: |' * len(next(iter(v1['sd_w_by_window'].values())))]
    lines += [f'| {length} | ' + ' | '.join(f'{value:.2f}' for value in row.values()) + ' |'
              for length, row in v1['sd_w_by_window'].items()]
    lines += ['', f'Smallest effect detectable with 80% power at {v1["cycles"]} cycles of {v1["run_seconds"]:g} s runs and '
              f'{v1["idle_seconds"]:g} s idles (W): ' + ', '.join(f'{k} {v:.2f}' for k, v in v1['mde80_w_at_defaults'].items()),
              '', '## V3: CPU load steps (ΔW, 95% CI)', '', '| Load | ΔPKG | ΔCores | ΔGraphics | ΔUncore |',
              '| --- | ---: | ---: | ---: | ---: |']
    lines += [f'| {mode} | {interval(r["pkg"])} | {interval(r["pp0"])} | {interval(r["pp1"])} | {interval(r["uncore"])} |'
              for mode, r in v3.items()]
    lines += ['', '## V5: GPU power under a 20 s matmul load', '', '| Source | W |', '| --- | ---: |']
    lines += [f'| {key} | {number(value, 2)} |' for key, value in v5.items()]
    lines.append('')
    path.write_text('\n'.join(lines))


def environment_info(args, meter: Meter | None) -> dict:
    info = {'git': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            'gpu': gpu_kernels.environment()}
    try:
        info['power_scheme'] = subprocess.run(['/mnt/c/Windows/System32/powercfg.exe', '/getactivescheme'],
                                              capture_output=True, text=True, timeout=30).stdout.strip()
        info['nvidia_driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                               capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    if meter is not None:
        info['sampler'] = meter.ready
        info['marker_round_trip_s'] = meter.round_trip()
    return info


def select(args, dataset: dict, blobs: dict) -> list[str]:
    if args.tasks == 'subset':
        return SUBSET
    if args.tasks == 'all':
        return [task['kernel'] for task in dataset['tasks']]
    return args.tasks.split(',')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', type=Path, help='RUN_ROOT/evaluation of an Intel NPU campaign')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/intel-fp16-baseline/intel-fp16-baseline-20261020')
    parser.add_argument('--output', type=Path, help='new directory (default: beside the campaign)')
    parser.add_argument('--tasks', default='subset', help="'subset', 'all' or comma-separated task names")
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument('--timing-only', action='store_true', help='time every selected task; no energy windows')
    kind.add_argument('--validate', action='store_true', help='instrument checks only (V1, V3, V5)')
    kind.add_argument('--reanalyse', type=Path, metavar='ENERGY_RUN',
                      help='recompute an energy run from its trace.csv and windows.json into ENERGY-reanalysed.md')
    parser.add_argument('--cross-check', action='store_true', help='with --timing-only: also run benchmark_app and NVBench')
    parser.add_argument('--cycles', type=int, default=6)
    parser.add_argument('--run-seconds', type=float, default=6.0)
    parser.add_argument('--idle-seconds', type=float, default=5.0)
    parser.add_argument('--warm', type=float, default=1.0, help='busy seconds before each counted window')
    parser.add_argument('--tail', type=float, default=0.5, help='busy seconds after each counted window')
    parser.add_argument('--settle', type=float, default=20.0,
                        help='longest wait (s) after GPU work for the GPU to reach its idle P-state before an idle window')
    parser.add_argument('--guard', type=float, default=0.5, help='seconds after an NPU run before the next idle window')
    parser.add_argument('--nireq', type=int, default=4, help='NPU requests in flight in async mode')
    parser.add_argument('--npu-warmup', type=int, default=1000)
    parser.add_argument('--gpu-warmup', type=int, default=200, help='minimum GPU warm-up calls per mode')
    parser.add_argument('--gpu-warmup-seconds', type=float, default=3.0, help='minimum GPU warm-up time per mode')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--npu-round-calls', type=int, default=2000)
    parser.add_argument('--gpu-round-calls', type=int, default=20000)
    parser.add_argument('--timing-windows', type=int, default=3)
    parser.add_argument('--timing-seconds', type=float, default=3.0)
    parser.add_argument('--cross-check-seconds', type=int, default=10, help='benchmark_app -t (whole seconds)')
    parser.add_argument('--validate-idle', type=float, default=180.0)
    parser.add_argument('--interval', type=float, default=0.02, help='sampler period (s)')
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--windows-python', default='/mnt/c/npu-runner/venv/Scripts/python.exe')
    args = parser.parse_args()
    if args.reanalyse:
        reanalyse(args.reanalyse.resolve())
        return
    if args.campaign is None:
        parser.error('--campaign is required')
    evaluation, args.dataset = args.campaign.resolve(), args.dataset.resolve()
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    kind_name = 'validate' if args.validate else 'timing' if args.timing_only else 'energy'
    output = (args.output or evaluation.parent / f'kernel-{kind_name}-{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'arguments.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    locks = []
    for path in (LOCK, ENERGY_LOCK):
        path.parent.mkdir(parents=True, exist_ok=True)
        locks.append(path.open('a'))
        try:
            fcntl.flock(locks[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'{path} is held by another NPU or energy job; wait for it to finish')
    dataset = json.loads((args.dataset / 'dataset.json').read_text())
    cases = {task['kernel']: task['cases'] for task in dataset['tasks']}
    blobs = accepted_blobs(evaluation)
    kernels = select(args, dataset, blobs)
    print(f'Output: {output}', flush=True)
    if args.timing_only:
        rows = []
        for kernel in kernels:
            print(f'timing {kernel}', flush=True)
            rows.append(time_task(args, output, kernel, blobs.get(kernel), cases[kernel]))
            (output / 'timing.json').write_text(json.dumps(rows, indent=2))
        nv = nvbench(args, [r['kernel'] for r in rows if r['gpu'].get('correct')], output / 'crosscheck') if args.cross_check else {}
        for row in rows:
            row['nvbench'] = nv.get(row['kernel'], {})
            row['gates'] = timing_gates(row, row['nvbench'])
        environment = environment_info(args, None)
        (output / 'timing.json').write_text(json.dumps({'environment': environment, 'tasks': rows}, indent=2))
        write_timing(output / 'TIMING.md', rows, environment)
        print(f'Report: {output / "TIMING.md"}')
        return
    meter = Meter(args, output)
    try:
        environment = environment_info(args, meter)
        if args.validate:
            report = validate(args, meter, output)
            (output / 'validation.json').write_text(json.dumps({'environment': environment, **report}, indent=2))
            write_validation(output / 'VALIDATION.md', report)
            print(f'Report: {output / "VALIDATION.md"}')
            return
        windows, notes = [], {}
        for index, kernel in enumerate(kernels):
            print(f'energy {kernel} ({index + 1}/{len(kernels)})', flush=True)
            task_windows, notes[kernel] = energy_task(args, meter, output, index, kernel, blobs[kernel], cases[kernel])
            windows += task_windows
            (output / 'windows.json').write_text(json.dumps({'notes': notes, 'windows': windows}, indent=2))
    finally:
        meter.close()
    results = analyse(read_trace(meter.trace), windows)
    (output / 'windows.json').write_text(json.dumps({'notes': notes, 'windows': windows}, indent=2))
    overall = write_energy(output / 'ENERGY.md', results, notes, environment, args, windows)
    (output / 'summary.json').write_text(json.dumps({'environment': environment, 'notes': notes, 'overall': overall,
                                                     'tasks': results}, indent=2))
    print(f'Report: {output / "ENERGY.md"}')


def reanalyse(folder: Path) -> None:
    """Apply the current estimator to a finished energy run; its own ENERGY.md and summary.json are kept."""
    saved = json.loads((folder / 'windows.json').read_text())
    args = argparse.Namespace(**json.loads((folder / 'arguments.json').read_text()))
    environment = json.loads((folder / 'summary.json').read_text())['environment']
    results = analyse(read_trace(folder / 'trace.csv'), saved['windows'])
    overall = write_energy(folder / 'ENERGY-reanalysed.md', results, saved['notes'], environment, args, saved['windows'])
    (folder / 'summary-reanalysed.json').write_text(json.dumps(
        {'environment': environment, 'notes': saved['notes'], 'overall': overall, 'tasks': results}, indent=2))
    print(f'Report: {folder / "ENERGY-reanalysed.md"}')


if __name__ == '__main__':
    main()
