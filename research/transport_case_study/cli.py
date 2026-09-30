"""Offline Gate A evidence collection; no artifact checker is claimed here."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from npu_agent.compilers import DockerCompiler
from npu_agent.config import AMD_XDNA2_NPU2, Settings
from npu_agent.models import KernelManifest
from npu_agent.validation import sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BASELINE = 'c1d246f4c09606d9b0ad5388ddb98c3d2a60bee7'
FIXTURE = Path('tests/fixtures/backends/amd_xdna2_npu2/add')


def probe(command: list[str], path: Path, timeout: int = 15) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        record = {'status': 'AVAILABLE' if result.returncode == 0 else 'UNAVAILABLE',
                  'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired as exc:
        record = {'status': 'TIMEOUT', 'returncode': None, 'stdout': '', 'stderr': str(exc)}
    except OSError as exc:
        record = {'status': 'UNAVAILABLE', 'returncode': None, 'stdout': '', 'stderr': str(exc)}
    record.update(command=command, elapsed_seconds=time.perf_counter() - started)
    write_json(path, record)
    return record


def inventory(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'symlink in evidence: {path}')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    return files


def archive_invocation(source: Path, output: Path) -> dict:
    if not source.is_dir() or source.is_symlink():
        raise ValueError('historical invocation must be an existing directory, not a symlink')
    before = inventory(source)
    required = {'toolchain.json', 'validation.json', 'source/design.py', 'source/kernel.cc',
                'emitted/design.mlir', 'target/insts.bin', 'target/final.xclbin',
                'target/commands.json', 'target/intermediates/input_with_addresses.mlir'}
    if not required <= before.keys() or any(before[name]['bytes'] == 0 for name in required):
        raise ValueError('historical invocation lacks required provenance or artifacts')
    if not any(name.endswith('.elf') for name in before):
        raise ValueError('historical invocation lacks core ELF')
    for name in ('design.py', 'kernel.cc'):
        if before[f'source/{name}']['sha256'] != sha256(ROOT / FIXTURE / name):
            raise ValueError('historical source differs from the unmodified feasibility fixture')
    # Copy this explicit invocation only; never search campaigns or read provider/config files.
    shutil.copytree(source, output)
    if inventory(output) != before or inventory(source) != before:
        raise ValueError('historical invocation changed during snapshot')
    identity = json.loads((output / 'toolchain.json').read_text())
    lock_hash = sha256(ROOT / 'playground/toolchains.lock.json')
    return {'origin': str(source.resolve()), 'evidence_kind': 'historical_only',
            'fixture_source_matches': True,
            'lock_hash_matches': identity.get('evaluator_hashes', {}).get('playground/toolchains.lock.json') == lock_hash,
            'identity': identity, 'files': before}


def planned_results(manifest: dict, environment: dict) -> list[dict]:
    parents = {case['case_id']: case for case in manifest['clean_cases']}
    rows = []
    for case in manifest['clean_cases'] + manifest['mutant_cases']:
        parent = parents[case.get('clean_parent_id', case['case_id'])]
        for baseline in manifest['baselines']:
            rows.append({'case_id': case['case_id'], 'clean_parent_id': case.get('clean_parent_id'),
                'topology': parent['topology'], 'shape': parent['shape'],
                'mutation_class': case.get('mutation_class'), 'injection_layer': case.get('injection_layer'),
                'repo_sha': environment['repo_sha'], 'toolchain_identity': None, 'artifact_hashes': {},
                'decoder_version': None, 'launch_protocol': None, 'launch_count': None,
                'input_seed': None, 'schedule_seed': None, 'baseline_or_checker': baseline,
                'supported_slice': 'none', 'assumptions': ['Device launch/reset contract is unestablished.'],
                'status': 'NOT_RUN', 'reason': 'GATE_A_NOT_SATISFIED',
                'violated_obligation': None, 'witness_path': None, 'elapsed_seconds': 0,
                'explored_states': 0, 'exploration_complete_within_bound': False,
                'hardware_executed': False, 'raw_log_path': 'environment.json'})
    return rows


def preflight(output: Path, docker: str = 'docker', historical: Path | None = None,
              build: bool = False) -> Path:
    # Capture Git state before creating the run, preserving even untracked inputs.
    repo_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'], cwd=ROOT, text=True)
    diff = subprocess.check_output(['git', 'diff', '--stat', BASELINE], cwd=ROOT, text=True)
    output = output.resolve()
    if historical and (output == historical.resolve() or historical.resolve() in output.parents):
        raise ValueError('output must be outside the historical invocation')
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path('playground/toolchains.lock.json'), Path('playground/tools/amd_compile.py'),
               Path('playground/tools/amd_host_runner.py'), Path('src/npu_agent/compilers.py'),
               Path('src/npu_agent/executors.py'), Path('src/npu_agent/validation.py'),
               Path('src/npu_agent/sim/amd_mlir_frontend.py'), Path('src/npu_agent/sim/amd_dataflow.py'),
               Path('src/npu_agent/sim/amd_host.py'), Path('tests/test_amd_simulation.py'),
               Path('tests/test_transport_case_study.py'),
               Path('tests/fixtures/backends/amd_xdna2_npu2/pipeline/design.py'),
               Path('docs/NPU_TRANSPORT_EARLY_CASE_STUDY.md')]
    sources += [p.relative_to(ROOT) for p in sorted((ROOT / FIXTURE).iterdir()) if p.is_file()]
    sources += [p.relative_to(ROOT) for p in sorted(STUDY.iterdir()) if p.is_file()]
    for path in sources:
        target = output / 'sources' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    manifest = json.loads((STUDY / 'manifest.json').read_text())
    write_json(output / 'manifest.frozen.json', manifest)
    for name in ('launch_contract.md', 'related_work.md'):
        shutil.copyfile(STUDY / name, output / name)
    environment = {'schema_version': 'transport-feasibility-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'repo_sha': repo_sha, 'baseline_sha': BASELINE, 'diff_from_baseline': diff,
        'dirty_files': dirty.splitlines(), 'python': sys.version, 'platform': platform.platform(),
        'toolchain_lock_sha256': sha256(ROOT / 'playground/toolchains.lock.json'),
        'limits': manifest['limits'], 'paid_calls': 0, 'hardware_executed': False,
        'image_identity': None, 'historical': None,
        'artifact_boundary': {'status': 'UNSUPPORTED', 'slice': 'none', 'decoder_version': None,
                              'reason': 'No version-matched binary decoder has been validated.'},
        'launch_contract': {'status': 'UNSUPPORTED', 'reason': 'Device launch/reset lifecycle is unknown.'}}
    environment['docker'] = probe([docker, 'version', '--format', '{{json .Server}}'], output / 'logs/docker.json')
    environment['native_tools'] = {name: shutil.which(name) for name in ('aiecc', 'aie-opt', 'xclbinutil', 'verif-opt')}
    environment['build'] = {'status': 'NOT_RUN', 'reason': 'Use the build command to attempt the clean fixture.'}
    if build:
        if environment['docker']['status'] != 'AVAILABLE':
            environment['build'] = {'status': 'INCONCLUSIVE' if environment['docker']['status'] == 'TIMEOUT' else 'NOT_RUN',
                                    'reason': 'Pinned container toolchain unavailable; see logs/docker.json.'}
        else:
            limits = manifest['limits']
            compiler = DockerCompiler(Settings(repository_path=ROOT, runs_path=output,
                compiler_timeout_seconds=limits['compile_seconds_per_stage'], docker_executable=docker,
                compiler_cpus=limits['compiler_cpus'], compiler_memory=limits['compiler_memory'],
                max_compiler_jobs=limits['max_compiler_jobs']))
            # Resolve and inspect the existing local image only; never build or pull it.
            try:
                identity = compiler.identity(AMD_XDNA2_NPU2)
                environment['image_identity'] = identity
                packages = identity['installed_tools']['packages']
                lock = json.loads((ROOT / 'playground/toolchains.lock.json').read_text())
                expected = [p for p in lock['amd']['python_packages'] if p.startswith(('mlir-aie==', 'llvm-aie=='))]
                if any(packages.get(p.split('==')[0]) != p.split('==')[1] for p in expected):
                    raise ValueError('installed compiler packages differ from frozen toolchain lock')
                version = identity['installed_tools']['tools']['aiecc']['version']
                if lock['amd']['mlir_aie_commit'][:12] not in version:
                    raise ValueError('aiecc commit differs from frozen toolchain lock')
                fixture_manifest = KernelManifest.model_validate_json((ROOT / FIXTURE / 'manifest.json').read_text())
                target = AMD_XDNA2_NPU2.model_copy(update={'compiler_image': identity['image_id']})
                result = compiler.compile(ROOT / FIXTURE, output / 'compile', fixture_manifest, target)
                write_json(output / 'compile_result.json', result.model_dump(mode='json'))
                environment['build'] = {'status': result.validation.stages['target_compile'].status,
                                        'raw_log_path': 'compile_result.json', 'scored_case': False}
            except subprocess.TimeoutExpired as exc:
                environment['build'] = {'status': 'INCONCLUSIVE', 'reason': str(exc)}
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                environment['build'] = {'status': 'NOT_RUN', 'reason': str(exc)}
    if historical:
        environment['historical'] = archive_invocation(historical, output / 'historical')
    # Compilation and archived bytes cannot substitute for extraction or lifecycle evidence.
    environment['gate_a'] = {'status': 'NOT_SATISFIED', 'verdict': 'STOP',
                             'reason': 'Fresh readable artifact and documented launch protocol not established.'}
    write_json(output / 'environment.json', environment)
    rows = planned_results(manifest, environment)
    with (output / 'results.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    write_json(output / 'artifact_manifest.json', inventory(output))
    report(output)
    return output


def report(run: Path) -> dict:
    run = run.resolve()
    hashes = json.loads((run / 'artifact_manifest.json').read_text())
    for name, expected in hashes.items():
        path = run / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(run.resolve()) or not path.is_file()):
            raise ValueError(f'invalid evidence path: {name}')
        if sha256(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError(f'evidence hash changed: {name}')
    for name in ('environment.json', 'manifest.frozen.json', 'results.jsonl', 'launch_contract.md', 'related_work.md'):
        if name not in hashes:
            raise ValueError(f'missing evidence hash: {name}')
    environment = json.loads((run / 'environment.json').read_text())
    manifest = json.loads((run / 'manifest.frozen.json').read_text())
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
    # This reporter is deliberately restricted to unscored Gate A records.
    expected = planned_results(manifest, environment)
    keys = lambda records: Counter((r['case_id'], r['baseline_or_checker']) for r in records)
    if keys(rows) != keys(expected) or any(count != 1 for count in keys(rows).values()):
        raise ValueError('result totals do not reconcile with frozen manifest')
    if rows != expected or environment['gate_a']['verdict'] != 'STOP':
        raise ValueError('Gate A reporter cannot score checker results or promote the gate')
    counts = {'clean_not_run': len(manifest['clean_cases']), 'mutant_not_run': len(manifest['mutant_cases']),
              'baseline_rows_not_run': len(rows), 'detections': 0, 'false_rejections': 0}
    with (run / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kind', 'class', 'planned_cases', 'not_run', 'detected', 'missed'])
        writer.writerow(['clean', '', counts['clean_not_run'], counts['clean_not_run'], 0, 0])
        for mutation, count in sorted(Counter(c['mutation_class'] for c in manifest['mutant_cases']).items()):
            writer.writerow(['mutant', mutation, count, count, 0, 0])
    with (run / 'support_matrix.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['case_id', 'baseline_or_checker', 'status', 'reason', 'supported_slice'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    historical = environment['historical']
    history = ('No historical invocation was supplied.' if historical is None else
               f"Historical image: `{historical['identity'].get('image_id')}`. "
               f"Fixture source matches: {historical['fixture_source_matches']}; "
               f"toolchain lock hash matches: {historical['lock_hash_matches']}. "
               'Archived validation is not a fresh build or a scored clean case.')
    reproduction = shlex.join(['.venv/bin/python', '-m', 'research.transport_case_study.cli', 'report', '--run', str(run)])
    (run / 'REPORT.md').write_text(f'''# Transport case study: Gate A

Verdict: **STOP at the feasibility gate**. This is an infrastructure/coverage
decision, not a detected program defect or a refutation of the research hypothesis.

Implemented: offline provenance capture, optional clean-fixture build, historical
artifact inventory, launch-contract audit, planned denominator and verified report
regeneration. No artifact decoder, transport model, mutations or checker was built.

Exact artifact and lifecycle coverage: **none verified**. Binary syntax,
endianness, address patches and descriptor units remain unvalidated; core
synchronization and physical state persistence are unknown.

Clean cases: 0 passed / 0 failed / 0 inconclusive / 0 unsupported;
**{counts['clean_not_run']} NOT_RUN**.
Mutant cases: 0 detected / 0 missed / 0 inconclusive / 0 not applicable;
**{counts['mutant_not_run']} NOT_RUN**. Applicability has not been established.
All {counts['baseline_rows_not_run']} case/control rows are NOT_RUN, covering B0–B3 and P1–P3.
Seeds, launch lengths and held-out cases were not executed or counted as discoveries.
The planned matrix is snapshotted but not frozen for scoring; legality is untested.

Real-program findings: none. Incremental findings beyond existing controls: none
established. Hardware actually executed: **none**. Paid calls: **0**.
Most important counterexample: none; infrastructure unavailability is not a witness.

Docker probe: **{environment['docker']['status']}**; clean build:
**{environment['build']['status']}**. See [environment](environment.json),
[Docker log](logs/docker.json), and [artifact hashes](artifact_manifest.json).
Current image identity: {'recorded in environment.json' if environment['image_identity'] else 'unavailable'}.
Repository: `{environment['repo_sha']}`; lock SHA-256:
`{environment['toolchain_lock_sha256']}`. Dirty paths and baseline diff are recorded.
{history}

Closest prior-art overlap: [source audit](related_work.md) includes the locked
MLIR-AIE rearm check, current upstream validation tooling, PEQC-MLIR and an
AccelSync lead. No competing tool was executed or shown inferior.
Main reason this may not deserve a paper: no verified extraction or incremental
failure mechanism exists yet, and substantial synchronization checks already exist.

One next experiment: restore access to the existing pinned container, compile the
unchanged feasibility fixture, and validate its instruction decoding and actual
launch/reset protocol before creating T1/T3 or scoring mutations. Do not upgrade
the lock or infer reusable-session legality from fresh simulator invocations.

See [launch contract](launch_contract.md), [planned manifest](manifest.frozen.json),
[raw rows](results.jsonl), [support matrix](support_matrix.csv) and [metrics](metrics.csv).
Checking cost: no checker ran (0 states); tool probe elapsed time is in the raw log.
Compilation and model-testing costs are distinct from device performance.

Reproduction command (verifies immutable inputs and regenerates derived tables):

```bash
{reproduction}
```
''')
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'build'):
        command = commands.add_parser(name)
        command.add_argument('--output', type=Path, required=True, help='new evidence directory; never overwritten')
        command.add_argument('--docker', default='docker')
        command.add_argument('--historical-invocation', type=Path)
    command = commands.add_parser('report')
    command.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'report':
            print(json.dumps(report(args.run), sort_keys=True))
            return 0
        output = preflight(args.output, args.docker, args.historical_invocation, args.command == 'build')
        print(f'STOP: Gate A not satisfied. Evidence: {output / "REPORT.md"}')
        return 2
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'evidence error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
