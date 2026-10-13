import json
import threading
import time
from collections import Counter

import pytest

from npu_agent.config import AMD_XDNA2_NPU2
from npu_agent.validation import write_json

from bucketed_length import run_all
from bucketed_length.render import collect, render
from bucketed_length.runner import ARMS, RunConfig, run
from bucketed_length.tests.test_part1 import Compiler, Provider


def prepared(tmp_path, *extra):
    args = run_all.parser().parse_args(['--output', str(tmp_path / 'suite'), *extra])
    return run_all.prepare(args)


def record_case(output, suite, index=0, compiler=None):
    case = suite['cases'][index]
    config = RunConfig(direction=case['direction'], arm=case['arm'], model=suite['config']['model'])
    state = run(output / 'inputs/kernel.cu', output / 'inputs/manifest.json', AMD_XDNA2_NPU2,
                Provider(), compiler or Compiler(), config, output / 'cases' / case['id'])
    case.update(process_status='finished', returncode=0 if state['status'] == 'completed' else 1, elapsed_seconds=2.5)
    write_json(output / 'suite.json', suite)
    return state


def test_plan_contains_all_arms_and_independent_outputs_libraries(tmp_path):
    output, suite = prepared(tmp_path, '--repetitions', '2')
    assert len(suite['cases']) == 32
    assert Counter(c['direction'] for c in suite['cases']) == {d: 2 * len(arms) for d, arms in ARMS.items()}
    assert len({c['id'] for c in suite['cases']}) == 32
    libraries = []
    for case in suite['cases']:
        command = case['command']
        assert command[:3] == [run_all.sys.executable, '-m', 'bucketed_length']
        assert command[command.index('--output') + 1] == str(output / 'cases' / case['id'])
        if '--library' in command:
            library = command[command.index('--library') + 1]
            libraries.append(library)
    assert len(libraries) == len(set(libraries))
    first = output / 'libraries' / suite['cases'][0]['id']
    (first / 'new-skill.json').write_text('{}')
    assert sum((p / 'new-skill.json').exists() for p in (output / 'libraries').iterdir()) == 1
    assert suite['case_token_cap_sum'] == 3200000


def test_study_report_retains_all_planned_cases_and_checks_frozen_inputs(tmp_path):
    from bucketed_length.report_study import report
    from npu_agent.validation import sha256

    for op in ('add', 'reduce', 'gemv'):
        args = run_all.parser().parse_args(['--output', str(tmp_path / ('suite-' + op)), '--repetitions', '2'])
        output, suite = run_all.prepare(args)
        record_case(output, suite)
    protocol = {key: suite['config'][key] for key in ('model', 'repetitions', 'max_repairs', 'population', 'selection_seed')}
    protocol.update(seed_shapes={'add': [64], 'reduce': [64], 'gemv': [8, 16]},
                    evaluation_shapes={'add': [128], 'reduce': [128], 'gemv': [16, 32]},
                    planned_evaluation_cases=96, token_budget_per_case=100000)
    write_json(tmp_path / 'protocol.json', protocol)
    write_json(tmp_path / 'frozen-inputs.sha256.json', {'protocol.json': sha256(tmp_path / 'protocol.json')})
    summary = report(tmp_path)
    assert summary['statuses'] == {'completed': 3, 'not_run': 93}
    assert len(summary['cases']) == 96
    assert 'SYNTHETIC TEST DATA' in (tmp_path / 'results/REPORT.md').read_text()
    (tmp_path / 'protocol.json').write_text(json.dumps({**protocol, 'model': 'changed'}))
    with pytest.raises(ValueError, match='frozen study inputs changed'):
        report(tmp_path)


def test_direction_concurrency_is_bounded_and_arms_are_sequential(tmp_path, monkeypatch):
    output, suite = prepared(tmp_path, '--workers', '3')
    lock = threading.Lock()
    barrier = threading.Barrier(3)
    active = set()
    calls = []
    peak = 0

    def execute(case, root):
        nonlocal peak
        with lock:
            assert case['direction'] not in active
            active.add(case['direction'])
            peak = max(peak, len(active))
            first_batch = len(calls) < 3
            calls.append(case['id'])
        if first_batch:
            barrier.wait(timeout=5)
        time.sleep(.002)
        with lock:
            active.remove(case['direction'])
        return {'process_status': 'finished', 'returncode': 3, 'elapsed_seconds': .01}

    monkeypatch.setattr(run_all, 'execute_case', execute)
    run_all.launch(output, suite)
    assert peak == 3 and len(calls) == 16 and len(set(calls)) == 16
    assert suite['status'] == 'finished' and suite['wall_seconds'] > 0
    assert all(c['process_status'] == 'finished' for c in suite['cases'])


def test_requested_stop_preserves_unstarted_cases(tmp_path, monkeypatch):
    output, suite = prepared(tmp_path)
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(run_all, 'execute_case', lambda *a: pytest.fail('must not launch after stop'))
    run_all.launch(output, suite, stop)
    assert suite['status'] == 'interrupted'
    assert all(c['process_status'] == 'not_run' for c in suite['cases'])


def test_child_command_uses_process_isolation_and_logs(tmp_path, monkeypatch):
    output, suite = prepared(tmp_path)
    calls = []

    def child(command, **kwargs):
        calls.append((command, kwargs))
        kwargs['stdout'].write('synthetic child failure\n')
        return type('Result', (), {'returncode': 3})()

    monkeypatch.setattr(run_all.subprocess, 'run', child)
    result = run_all.execute_case(suite['cases'][0], output)
    assert result['returncode'] == 3
    assert calls[0][1]['start_new_session'] is True
    assert calls[0][1]['env']['OMP_NUM_THREADS']
    assert 'synthetic child failure' in next((output / 'logs').glob('*.log')).read_text()


def test_worker_exception_is_recorded_and_stops_new_cases(tmp_path, monkeypatch):
    output, suite = prepared(tmp_path, '--workers', '1')

    def broken(*args):
        raise RuntimeError('test worker exception')

    monkeypatch.setattr(run_all, 'execute_case', broken)
    with pytest.raises(RuntimeError, match='worker exception'):
        run_all.launch(output, suite)
    assert suite['status'] == 'error'
    assert any(c['process_status'] == 'not_run' for c in suite['cases'])


def test_render_keeps_missing_failed_and_successful_cases_and_verifies_evidence(tmp_path):
    output, suite = prepared(tmp_path)
    record_case(output, suite)
    record_case(output, suite, 1, Compiler(failures=99))
    suite['cases'][2].update(process_status='finished', returncode=1, elapsed_seconds=.1)
    write_json(output / 'suite.json', suite)
    summary = render(output)
    assert len(summary['cases']) == 16
    assert [r['status'] for r in summary['cases'][:4]] == ['completed', 'failed', 'process_error', 'not_run']
    assert sum(g['planned'] for g in summary['groups']) == 16
    assert sum(g['active_seconds'] for g in summary['groups']) == 5.1
    assert summary['wall_seconds'] == 0  # Independent suite clock, not summed case clocks.
    for name in ('REPORT.md', 'summary.json', 'cases.csv', 'outcomes.png', 'outcomes.svg', 'resources.png', 'resources.svg'):
        assert (output / 'results' / name).stat().st_size > 100
    report = (output / 'results/REPORT.md').read_text()
    assert 'Synthetic test evidence' in report and 'seed library was empty' in report
    path = output / 'cases' / suite['cases'][0]['id'] / 'report.json'
    path.write_text('{}')
    _, rows = collect(output)
    assert rows[0]['status'] == 'invalid_evidence'
    assert rows[0]['winner'] is None


def test_render_rejects_wrong_input_binding_even_when_case_hash_is_resealed(tmp_path):
    output, suite = prepared(tmp_path)
    record_case(output, suite)
    suite['source_sha256'] = 'different-source'
    write_json(output / 'suite.json', suite)
    _, rows = collect(output)
    assert rows[0]['status'] == 'invalid_evidence'
    suite['cases'].pop()
    write_json(output / 'suite.json', suite)
    with pytest.raises(ValueError, match='every planned arm'):
        collect(output)


def test_child_exit_mismatch_cannot_be_reported_as_success(tmp_path):
    output, suite = prepared(tmp_path)
    record_case(output, suite)
    suite['cases'][0]['returncode'] = 1
    write_json(output / 'suite.json', suite)
    _, rows = collect(output)
    assert rows[0]['status'] == 'process_error'
    assert rows[0]['winner'] is None and rows[0]['first_attempt_pass'] is None


def test_saved_running_state_without_a_report_does_not_prove_interruption(tmp_path):
    output, suite = prepared(tmp_path)
    suite['cases'][0]['process_status'] = 'running'
    write_json(output / 'suite.json', suite)
    _, rows = collect(output)
    assert rows[0]['status'] == 'incomplete'
    assert 'live state unverified' in rows[0]['reason']


def test_cli_dry_run_and_rerender_never_launch_a_child_or_overwrite_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, 'execute_case', lambda *a: pytest.fail('dry/render commands must not launch children'))
    output = tmp_path / 'preview'
    assert run_all.main(['--output', str(output), '--dry-run']) == 0
    original = (output / 'suite.json').read_bytes()
    assert run_all.main(['--output', str(output), '--render-only']) == 0
    assert (output / 'suite.json').read_bytes() == original
    with pytest.raises(FileExistsError):
        run_all.main(['--output', str(output), '--dry-run'])
    assert all(c['status'] == 'not_run' for c in json.loads((output / 'results/summary.json').read_text())['cases'])


@pytest.mark.parametrize('flags', [('--workers', '0'), ('--workers', '8'), ('--repetitions', '0'),
                                 ('--max-repairs', '6'), ('--population', '9')])
def test_invalid_limits_fail_before_creating_a_suite(tmp_path, flags):
    with pytest.raises(ValueError):
        prepared(tmp_path, *flags)
    assert not (tmp_path / 'suite').exists()
