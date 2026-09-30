import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from npu_agent.validation import sha256, write_json


spec = importlib.util.spec_from_file_location('transport_case_study', Path(__file__).resolve().parents[1] / 'research/transport_case_study/cli.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def unavailable(command, path, timeout=15):
    record = {'status': 'UNAVAILABLE', 'command': command, 'returncode': 1,
              'stdout': '', 'stderr': 'synthetic unavailable-tool fixture', 'elapsed_seconds': 0}
    write_json(path, record)
    return record


def run_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'probe', unavailable)
    return study.preflight(tmp_path / 'run', build=True)


def reseal(run, name):
    hashes = json.loads((run / 'artifact_manifest.json').read_text())
    hashes[name] = {'sha256': sha256(run / name), 'bytes': (run / name).stat().st_size}
    write_json(run / 'artifact_manifest.json', hashes)


def test_unavailable_tool_stops_without_constructing_compiler(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('compiler must not be constructed when Docker is unavailable')
    monkeypatch.setattr(study, 'DockerCompiler', forbidden)
    run = run_fixture(tmp_path, monkeypatch)
    environment = json.loads((run / 'environment.json').read_text())
    assert environment['gate_a']['verdict'] == 'STOP'
    assert environment['build']['status'] == 'NOT_RUN'
    assert environment['image_identity'] is None
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 24 * 7
    assert all(r['status'] == 'NOT_RUN' and r['launch_count'] is None and not r['hardware_executed'] for r in rows)
    assert Counter(r['mutation_class'] for r in rows)['M6'] == 2 * 7
    assert study.report(run) == {'clean_not_run': 8, 'mutant_not_run': 16,
                                'baseline_rows_not_run': 168, 'detections': 0, 'false_rejections': 0}


def test_reports_are_reproducible_and_raw_records_unchanged(tmp_path, monkeypatch):
    run = run_fixture(tmp_path, monkeypatch)
    before = study.inventory(run)
    monkeypatch.chdir(tmp_path)
    study.report(Path('run'))
    assert study.inventory(run) == before
    with pytest.raises(FileExistsError):
        study.preflight(run)
    assert study.inventory(run) == before


def test_tampered_evidence_is_rejected_before_report_write(tmp_path, monkeypatch):
    run = run_fixture(tmp_path, monkeypatch)
    report_before = (run / 'REPORT.md').read_bytes()
    (run / 'results.jsonl').write_text('')
    with pytest.raises(ValueError, match='evidence hash changed'):
        study.report(run)
    assert (run / 'REPORT.md').read_bytes() == report_before


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'promoted'])
def test_denominator_and_no_false_pass_even_with_updated_hash(tmp_path, monkeypatch, change):
    run = run_fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows[-1] = rows[0]
    else:
        rows[0]['status'] = 'PASS_BOUNDED'
    (run / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    reseal(run, 'results.jsonl')
    with pytest.raises(ValueError, match='reconcile|cannot score'):
        study.report(run)


def test_probe_distinguishes_missing_tool_timeout_and_nonzero_exit(tmp_path):
    assert study.probe([str(tmp_path / 'absent')], tmp_path / 'absent.json')['status'] == 'UNAVAILABLE'
    assert study.probe([sys.executable, '-c', 'import time; time.sleep(2)'], tmp_path / 'timeout.json', timeout=.01)['status'] == 'TIMEOUT'
    result = study.probe([sys.executable, '-c', 'raise SystemExit(3)'], tmp_path / 'exit.json')
    assert result['status'] == 'UNAVAILABLE' and result['returncode'] == 3


def test_timeout_never_becomes_program_failure(tmp_path, monkeypatch):
    def timeout(command, path, **kwargs):
        result = unavailable(command, path)
        result['status'] = 'TIMEOUT'
        write_json(path, result)
        return result
    monkeypatch.setattr(study, 'probe', timeout)
    run = study.preflight(tmp_path / 'timeout', build=True)
    environment = json.loads((run / 'environment.json').read_text())
    assert environment['build']['status'] == 'INCONCLUSIVE'
    assert study.report(run)['detections'] == 0


@pytest.mark.parametrize('mismatch', [False, True])
def test_build_checks_versions_and_pins_image_without_opening_gate(tmp_path, monkeypatch, mismatch):
    calls = []
    lock = json.loads((study.ROOT / 'playground/toolchains.lock.json').read_text())
    packages = dict(p.split('==') for p in lock['amd']['python_packages'] if p.startswith(('mlir-aie==', 'llvm-aie==')))
    if mismatch:
        packages['mlir-aie'] = 'unapproved'
    identity = {'image_id': 'sha256:' + '1' * 64, 'installed_tools': {'packages': packages,
                'tools': {'aiecc': {'version': lock['amd']['mlir_aie_commit'][:12]}}}}

    class Compiler:
        def __init__(self, settings):
            assert settings.compiler_timeout_seconds == 180
            assert settings.max_compiler_jobs == 1

        def identity(self, target):
            return identity

        def compile(self, candidate, output, manifest, target):
            calls.append(target)
            assert target.compiler_image == identity['image_id']
            assert target.hardware_runner is None
            return SimpleNamespace(validation=SimpleNamespace(stages={'target_compile': SimpleNamespace(status='passed')}),
                                   model_dump=lambda **kwargs: {'synthetic_test_only': True})

    def available(command, path, **kwargs):
        result = unavailable(command, path)
        result.update(status='AVAILABLE', returncode=0, stdout='{}', stderr='')
        write_json(path, result)
        return result

    monkeypatch.setattr(study, 'DockerCompiler', Compiler)
    monkeypatch.setattr(study, 'probe', available)
    run = study.preflight(tmp_path / 'build', build=True)
    environment = json.loads((run / 'environment.json').read_text())
    assert environment['build']['status'] == ('NOT_RUN' if mismatch else 'passed')
    assert len(calls) == (0 if mismatch else 1)
    assert environment['gate_a']['verdict'] == 'STOP'
    assert study.report(run)['clean_not_run'] == 8


def historical_fixture(tmp_path):
    # Deliberately synthetic container layout; it cannot establish the real-artifact gate.
    root = tmp_path / 'historical'
    for name in ('toolchain.json', 'validation.json', 'emitted/design.mlir', 'target/insts.bin',
                 'target/final.xclbin', 'target/commands.json', 'target/core.elf',
                 'target/intermediates/input_with_addresses.mlir'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    (root / 'source').mkdir()
    for name in ('design.py', 'kernel.cc'):
        shutil.copyfile(study.ROOT / study.FIXTURE / name, root / 'source' / name)
    return root


def test_historical_snapshot_does_not_promote_gate_or_change_contract(tmp_path, monkeypatch):
    historical = historical_fixture(tmp_path)
    original = study.inventory(historical)
    monkeypatch.setattr(study, 'probe', unavailable)
    run = study.preflight(tmp_path / 'run', historical=historical)
    environment = json.loads((run / 'environment.json').read_text())
    assert environment['historical']['evidence_kind'] == 'historical_only'
    assert not environment['historical']['lock_hash_matches']
    assert environment['gate_a']['verdict'] == 'STOP'
    assert study.inventory(historical) == original == study.inventory(run / 'historical')
    assert json.loads((run / 'manifest.frozen.json').read_text()) == json.loads((study.STUDY / 'manifest.json').read_text())


def test_reject_unsafe_or_incomplete_historical_evidence(tmp_path):
    root = historical_fixture(tmp_path)
    (root / 'source/design.py').write_text('different source')
    with pytest.raises(ValueError, match='source differs'):
        study.archive_invocation(root, tmp_path / 'copy')
    (root / 'target/core.elf').unlink()
    with pytest.raises(ValueError, match='lacks core ELF'):
        study.archive_invocation(root, tmp_path / 'copy')
    (root / 'target/core.elf').symlink_to(root / 'target/insts.bin')
    with pytest.raises(ValueError, match='symlink'):
        study.archive_invocation(root, tmp_path / 'copy')
