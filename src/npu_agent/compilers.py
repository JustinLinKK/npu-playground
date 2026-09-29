from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings
from .executors import encode_tensor, execute_target, unavailable_executor
from .models import Backend, CompileResult, ExecutionRequest, KernelManifest, StageResult, TargetProfile, ValidationResult
from .oracles import execute_oracle, validate_goldens
from .testcases import generate_cases, serialize_cases
from .validation import UnsupportedContract, compare_outputs, finalize_validation, json_digest, sha256, write_json


class Compiler(Protocol):
    def compile(self, candidate_dir: Path, output_dir: Path, manifest: KernelManifest,
                target: TargetProfile) -> CompileResult: ...


def _bounded(value: str | bytes, limit: int = 200_000) -> str:
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return value if len(value) <= limit else value[-limit:]


class DockerCompiler:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._slots = threading.BoundedSemaphore(settings.max_compiler_jobs)
        self._identities: dict[str, dict] = {}
        self._metadata: dict[str, dict] = {}
        if settings.compiler_cpus <= 0 or settings.max_compiler_jobs <= 0:
            raise ValueError('positive compiler CPUs and simultaneous job limit are required')

    def identity(self, target: TargetProfile) -> dict:
        # Resolve the mutable tag each time; the invocation uses the immutable ID.
        result = subprocess.run([self.settings.docker_executable, 'image', 'inspect', target.compiler_image],
                                capture_output=True, text=True, timeout=30, check=True,
                                start_new_session=self.settings.experiment_id is not None)
        image = json.loads(result.stdout)[0]
        if image['Id'] not in self._metadata:
            metadata = subprocess.run([self.settings.docker_executable, 'run', '--rm', '--network', 'none',
                '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--user', f'{os.getuid()}:{os.getgid()}', '--entrypoint', 'cat', image['Id'],
                '/opt/toolchain-metadata.json'], capture_output=True, text=True, timeout=30, check=True,
                start_new_session=self.settings.experiment_id is not None)
            self._metadata[image['Id']] = json.loads(metadata.stdout)
        tools_root = self.settings.repository_path / 'playground' / 'tools'
        paths = sorted(tools_root.rglob('*.py')) + sorted(tools_root.rglob('*.h')) + sorted(tools_root.rglob('*.hpp'))
        paths += sorted((self.settings.repository_path / 'src' / 'npu_agent').rglob('*.py'))
        lock = self.settings.repository_path / 'playground' / 'toolchains.lock.json'
        if lock.exists():
            paths.append(lock)
        identity = {'image_id': image['Id'], 'repo_digests': image.get('RepoDigests', []),
                    'profile': target.model_dump(mode='json', exclude={'hardware_runner'}),
                    'evaluator_hashes': {str(p.relative_to(self.settings.repository_path)): sha256(p) for p in paths},
                    'compile_flags': target.compiler_properties, 'installed_tools': self._metadata[image['Id']]}
        identity['fingerprint'] = json_digest(identity)
        self._identities[target.id] = identity
        return identity

    def fingerprint(self, target: TargetProfile) -> str:
        try:
            return self.identity(target)['fingerprint']
        except (OSError, ValueError, subprocess.SubprocessError):
            # Compilation records the structured infrastructure failure; retrieval
            # must not reuse lessons from an unresolved toolchain or abort the run.
            return 'unresolved'

    def _docker_prefix(self, candidate_dir: Path, output_dir: Path, target: TargetProfile,
                       mounts: dict[str, Path] | None = None, name: str | None = None) -> list[str]:
        image = self._identities.get(target.id, {}).get('image_id', target.compiler_image)
        command = [self.settings.docker_executable, 'run', '--rm', '--name', name or f'npu-agent-{uuid.uuid4().hex}',
                   '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                   '--cpus', str(self.settings.compiler_cpus), '--memory', self.settings.compiler_memory,
                   '--pids-limit', '256', '--user', f'{os.getuid()}:{os.getgid()}',
                   '--env', 'HOME=/tmp', '--env', 'NPU_CACHE_HOME=/work/cache',
                   '--env', 'PYTHONDONTWRITEBYTECODE=1',
                   '--env', 'PYTHONPATH=/framework:/opt/ironenv/lib/python3.12/site-packages/mlir_aie/python:/opt/IRON',
                   '--tmpfs', '/tmp:rw,size=2g,exec', '--tmpfs', '/work:rw,size=2g,exec,mode=1777', '--workdir', '/work',
                   '-v', f'{candidate_dir.resolve()}:/candidate:ro', '-v', f'{output_dir.resolve()}:/output:rw',
                   '-v', f'{(self.settings.repository_path / "playground/tools").resolve()}:/tools:ro',
                   '-v', f'{(self.settings.repository_path / "src").resolve()}:/framework:ro']
        for mount, path in (mounts or {}).items():
            command += ['-v', f'{path.resolve()}:{mount}:ro']
        return command + [image]

    def _run(self, candidate: Path, output: Path, target: TargetProfile, tool: str,
             mounts: dict[str, Path], args: list[str] | None = None) -> StageResult:
        output.mkdir(parents=True, exist_ok=False)
        name = f'npu-agent-{uuid.uuid4().hex}'
        command = self._docker_prefix(candidate, output, target, mounts, name) + ['python', f'/tools/{tool}', *(args or [])]
        started = time.perf_counter()
        category = None
        stdout_path = output.parent / f'{output.name}.stdout.log'
        stderr_path = output.parent / f'{output.name}.stderr.log'
        try:
            with self._slots, stdout_path.open('wb') as stdout_file, stderr_path.open('wb') as stderr_file:
                completed = subprocess.run(command, stdout=stdout_file, stderr=stderr_file,
                                           timeout=self.settings.compiler_timeout_seconds, check=False,
                                           start_new_session=self.settings.experiment_id is not None)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code, category = 124, 'timeout'
            try:
                subprocess.run([self.settings.docker_executable, 'rm', '-f', name], capture_output=True, timeout=30, check=False)
            except (OSError, subprocess.SubprocessError):
                category = 'timeout_cleanup_failed'
        except OSError as exc:
            code, category = 127, 'build_dependency'
            stderr_path.write_text(str(exc))
        duration = time.perf_counter() - started
        def excerpt(path):
            if not path.exists():
                return ''
            with path.open('rb') as handle:
                handle.seek(max(0, path.stat().st_size - 4000))
                return _bounded(handle.read(), 4000)
        stdout, stderr = excerpt(stdout_path), excerpt(stderr_path)
        if code and category is None:
            category = 'source_compile' if tool in ('amd_emit.py', 'intel_build_graph.py') else 'target_compile' if tool.endswith('compile.py') or tool == 'intel_compile_graph.py' else 'source_execution'
            for marker in ('simulation_deadlock', 'target_mismatch', 'host kernel timeout', 'immutable input', 'DMA bounds'):
                if marker in stderr:
                    category = {'host kernel timeout': 'timeout', 'immutable input': 'memory_safety', 'DMA bounds': 'memory_safety'}.get(marker, marker)
        status = 'passed' if code == 0 else 'blocked' if code in (125, 126, 127) else 'unsupported' if code == 3 else 'failed'
        write_json(output.parent / f'{output.name}.command.json', command)
        return StageResult(status=status, duration_seconds=duration,
                           error_category=category, message=_bounded(stderr, 4000),
                           details={'exit_code': code, 'command': command, 'stdout': _bounded(stdout, 4000)})

    @staticmethod
    def _files(root: Path, names: list[str], allow_empty: set[str] | None = None) -> dict[str, str]:
        result = {}
        for name in names:
            path = root / name
            if path.is_symlink() or not path.is_file() or (not path.stat().st_size and name not in (allow_empty or set())):
                raise ValueError(f'missing, empty, or unsafe artifact: {name}')
            result[name] = str(path)
        return result

    def compile(self, candidate_dir: Path, output_dir: Path, manifest: KernelManifest, target: TargetProfile) -> CompileResult:
        started = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='invocation-', dir=output_dir))
        source_hashes = {p.name: sha256(p) for p in sorted(candidate_dir.iterdir()) if p.is_file()}
        validation = ValidationResult(candidate_id=candidate_dir.name, target_id=target.id,
                                      source_sha256=json_digest(source_hashes), manifest_sha256=json_digest(manifest.model_dump(mode="json")), reference_origin='independent_builtin',
                                      stages={'target_execution': unavailable_executor(),
                                              'performance': StageResult(status='not_requested')})
        stages = validation.stages
        # Snapshot only the backend contract; fixture manifests and arbitrary extra files
        # are not candidate code and are never visible to trusted execution processes.
        from .artifacts import ALLOWED_FILES
        snapshot = run / "source"
        snapshot.mkdir()
        try:
            for name in ALLOWED_FILES[target.backend]:
                path = candidate_dir / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"missing or unsafe candidate source: {name}")
                shutil.copyfile(path, snapshot / name)
        except (OSError, ValueError) as exc:
            stages['target_compile'] = StageResult(status='failed', error_category='source_compile', message=str(exc))
            finalize_validation(validation, self.settings.validation_policy)
            write_json(run / 'validation.json', validation.model_dump(mode='json'))
            return CompileResult(success=False, exit_code=1, stderr=str(exc), compiler_fingerprint='unresolved',
                validation=validation, artifacts={'validation.json': str(run / 'validation.json')},
                duration_seconds=time.perf_counter() - started)
        candidate_dir = snapshot
        validation.source_sha256 = json_digest({p.name: sha256(p) for p in sorted(snapshot.iterdir())})
        cases = []
        oracle_started = time.perf_counter()
        try:
            validate_goldens(manifest.oracle.operation)
            cases = generate_cases(manifest)
            for case in cases:
                execute_oracle(manifest, case.inputs)
            stages['oracle_validation'] = StageResult(status='passed', correct=True, engine='independent-builtin-v1',
                cases_run=len(cases), duration_seconds=time.perf_counter() - oracle_started,
                details={'reference_origin': 'independent_builtin', 'source_equivalence': 'not_established',
                         'prose_constraints': 'descriptive_only'})
        except (ValueError, TypeError, KeyError) as exc:
            stages['oracle_validation'] = StageResult(status='unsupported' if isinstance(exc, UnsupportedContract) else 'failed',
                message=str(exc), error_category='unsupported_contract' if isinstance(exc, UnsupportedContract) else 'oracle',
                duration_seconds=time.perf_counter() - oracle_started)
        artifacts = {}
        try:
            identity = self.identity(target)
            validation.toolchain_fingerprint = identity['fingerprint']
            write_json(run / 'toolchain.json', identity)
            contract = run / 'contract'
            write_json(contract / 'manifest.json', manifest.model_dump(mode='json'))
            write_json(contract / 'target.json', target.model_dump(mode='json'))
            inputs = run / 'inputs'
            validation.input_sha256 = serialize_cases(cases, inputs)
            mounts = {'/contract': contract}
            if target.backend == Backend.INTEL_OPENVINO:
                built = self._run(candidate_dir, run / 'graph', target, 'intel_build_graph.py', mounts)
                built.representation, built.engine = 'openvino_graph', 'openvino-construction'
                stages['graph_construction'] = built
                if built.status == 'passed':
                    artifacts.update(self._files(run / 'graph', ['model.xml', 'model.bin'], {'model.bin'}))
                    mounts['/graph'] = run / 'graph'
                    if cases:
                        host = self._run(candidate_dir, run / 'host', target, 'intel_execute_graph.py',
                                         {**mounts, '/inputs': inputs})
                        stages['host_execution'] = self._compare(host, run / 'host', manifest, cases, 'openvino-cpu', 'openvino_graph')
                        write_json(run / 'validation.json', validation.model_dump(mode='json'))
                    compiled = self._run(candidate_dir, run / 'target', target, 'intel_compile_graph.py', mounts)
                    compiled.engine, compiled.representation = 'openvino-compiler-in-plugin', 'npu_blob'
                    stages['target_compile'] = compiled
                    if compiled.status == 'passed':
                        artifacts.update(self._files(run / 'target', ['compiled.blob']))
                        compiled.artifacts = {'compiled.blob': sha256(Path(artifacts['compiled.blob']))}
                else:
                    stages['target_compile'] = StageResult(status='failed', error_category='source_compile', message=built.message)
            else:
                emitted = self._run(candidate_dir, run / 'emitted', target, 'amd_emit.py', mounts)
                stages['design_emission'] = emitted
                if emitted.status == 'passed':
                    artifacts.update(self._files(run / 'emitted', ['design.mlir']))
                    mounts['/design'] = run / 'emitted'
                    compiled = self._run(candidate_dir, run / 'target', target, 'amd_compile.py', mounts)
                    compiled.engine, compiled.representation = 'mlir-aie-peano', 'aie2p_artifacts'
                    stages['target_compile'] = compiled
                    if compiled.status == 'passed':
                        artifacts.update(self._files(run / 'target', ['final.xclbin', 'insts.bin', 'abi.json', 'inspection.txt']))
                        compiled.artifacts = {k: sha256(Path(v)) for k, v in artifacts.items()}
                    # Parsing is independent of compilation; unsupported simulation must not erase compilation.
                    parsed = self._run(candidate_dir, run / 'parsed', target, 'amd_parse.py', mounts)
                    stages['dataflow_frontend'] = parsed
                    if parsed.status == 'passed' and cases:
                        mounts['/program'] = run / 'parsed'
                        for stage, mode in [('host_execution', 'host'), ('dataflow_simulation', 'dataflow')]:
                            result = self._run(candidate_dir, run / mode, target, 'amd_host_runner.py',
                                               {**mounts, '/inputs': inputs}, ['--mode', mode])
                            stages[stage] = self._compare(result, run / mode, manifest, cases,
                                f'amd-{mode}-subset-v1', 'kernel_source' if mode == 'host' else 'design_mlir')
                    elif parsed.status != 'passed':
                        for stage in ('host_execution', 'dataflow_simulation'):
                            stages[stage] = StageResult(status='unsupported' if parsed.details.get('exit_code') == 3 else 'failed',
                                error_category='unsupported_mlir' if parsed.details.get('exit_code') == 3 else 'source_compile',
                                message=parsed.message)
                else:
                    stages['target_compile'] = StageResult(status='failed', error_category='source_compile', message=emitted.message)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            stages['target_compile'] = StageResult(status='blocked' if validation.toolchain_fingerprint is None else 'failed',
                error_category='build_dependency' if validation.toolchain_fingerprint is None else 'malformed_report', message=str(exc))
        for stage in ('host_execution', 'dataflow_simulation'):
            stages.setdefault(stage, StageResult(status='not_requested' if stage == 'dataflow_simulation' and target.vendor == 'intel'
                                                else 'blocked', reason_code='PREREQUISITE_UNAVAILABLE'))
        if target.hardware_runner and stages['target_compile'].status == 'passed' and cases:
            binary_names = ['compiled.blob'] if target.vendor == 'intel' else ['final.xclbin', 'insts.bin']
            binaries = {name: artifacts[name] for name in binary_names}
            request = ExecutionRequest(run_id=run.name, target=target, artifacts=binaries,
                artifact_sha256={k: sha256(Path(v)) for k, v in binaries.items()}, manifest=manifest,
                manifest_sha256=json_digest(manifest.model_dump(mode='json')),
                abi_sha256=sha256(Path(artifacts['abi.json'])) if 'abi.json' in artifacts else json_digest([t.model_dump(mode='json') for t in manifest.tensors]),
                cases={c.id: {k: encode_tensor(v) for k, v in c.inputs.items() if isinstance(v, np.ndarray)} for c in cases},
                compiler_fingerprint=validation.toolchain_fingerprint, timeout_seconds=self.settings.compiler_timeout_seconds,
                warmup_count=target.executor_warmup_count, measurement_count=target.executor_measurement_count)
            write_json(run / 'execution-request.json', request.model_dump(mode='json'))
            stages['target_execution'] = execute_target(request, target.hardware_runner)
        for directory in ("source", "contract", "inputs", "target", "parsed", "host", "dataflow"):
            for path in sorted((run / directory).rglob("*")):
                if path.is_file() and not path.is_symlink():
                    artifacts[str(path.relative_to(run))] = str(path)
        execution = stages['target_execution']
        samples = execution.details.get('timing_samples_ms', []) if execution.engine == 'physical_npu' and execution.correct is True else []
        if samples:
            validation.hardware_latency_p50_ms = float(np.percentile(samples, 50))
            stages['performance'] = StageResult(status='passed', engine='physical_npu', representation='hardware_timing',
                details={'samples_ms': samples, 'scope': execution.details['timing_scope'],
                         'provenance': execution.details['timing_provenance']})
        finalize_validation(validation, self.settings.validation_policy)
        write_json(run / 'validation.json', validation.model_dump(mode='json'))
        artifacts['validation.json'] = str(run / 'validation.json')
        success = stages['target_compile'].status == 'passed'
        return CompileResult(success=success, exit_code=0 if success else 3 if stages['target_compile'].status == 'blocked' else 1,
            stderr='\n'.join(f'{name}: {s.message}' for name, s in stages.items() if s.status in ('failed', 'unsupported', 'blocked')),
            artifacts=artifacts, compiler_fingerprint=validation.toolchain_fingerprint or 'unresolved',
            duration_seconds=time.perf_counter() - started,
            evaluation_duration_seconds=sum(s.duration_seconds for n, s in stages.items() if n != 'target_compile'),
            host_correct=stages['host_execution'].correct, hardware_correct=stages['target_execution'].correct,
            latency_p50_ms=validation.hardware_latency_p50_ms,
            latency_p95_ms=float(np.percentile(samples, 95)) if samples else None,
            hardware_warmup_count=target.executor_warmup_count if samples else None,
            hardware_iteration_count=len(samples) if samples else None,
            validation=validation, static_metrics={'source_bytes': sum(p.stat().st_size for p in candidate_dir.iterdir() if p.is_file()),
            'artifact_bytes': sum(Path(p).stat().st_size for p in artifacts.values())})

    def _compare(self, stage: StageResult, output: Path, manifest: KernelManifest, cases: list,
                 engine: str, representation: str) -> StageResult:
        stage.engine, stage.representation = engine, representation
        if stage.status != 'passed':
            if stage.details.get('exit_code') == 3:
                stage.status, stage.error_category = 'unsupported', 'unsupported_source'
            stage.error_category = stage.error_category or 'source_execution'
            return stage
        started = time.perf_counter()
        try:
            comparisons = []
            for case in cases:
                self._files(output, [f'{case.id}.npz'])
                schedules = ['', '-schedule-7', '-schedule-31'] if engine == 'amd-dataflow-subset-v1' else ['']
                for suffix in schedules:
                    self._files(output, [f'{case.id}{suffix}.npz'])
                    with np.load(output / f'{case.id}{suffix}.npz', allow_pickle=False) as archive:
                        actual = {key: archive[key] for key in archive.files}
                    comparisons.append(compare_outputs(actual, execute_oracle(manifest, case.inputs), manifest.tolerance, case.id + suffix))
            stage.correct = all(c['correct'] for c in comparisons)
            stage.status = 'passed' if stage.correct else 'failed'
            stage.error_category = None if stage.correct else 'numerical_mismatch'
            stage.cases_run = len(cases)
            stage.details['comparisons'] = comparisons
            stage.artifacts = {p.name: sha256(p) for p in output.glob('*.npz') if not p.is_symlink()}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            stage.status, stage.error_category, stage.message = 'failed', 'malformed_report', str(exc)
        stage.duration_seconds += time.perf_counter() - started
        return stage


def build_images(settings: Settings, target_ids: list[str]) -> None:
    from .config import TARGETS
    dockerfiles = {Backend.AMD_XDNA2: 'Dockerfile.xdna2-offline', Backend.INTEL_OPENVINO: 'Dockerfile.intel-npu-offline'}
    for target_id in target_ids:
        target = TARGETS[target_id]
        subprocess.run([settings.docker_executable, 'build', '-t', target.compiler_image, '-f',
                        f'playground/dockerfiles/{dockerfiles[target.backend]}', '.'], cwd=settings.repository_path, check=True)


def smoke_images(settings: Settings, target_ids: list[str]) -> None:
    from .config import TARGETS
    compiler = DockerCompiler(settings)
    for target_id in target_ids:
        fixture = settings.repository_path / 'tests/fixtures/backends' / target_id / 'add'
        manifest = KernelManifest.model_validate_json((fixture / 'manifest.json').read_text())
        result = compiler.compile(fixture, settings.runs_path / 'smoke' / target_id, manifest, TARGETS[target_id])
        if not result.validation.offline_contract_met:
            raise RuntimeError(result.validation.model_dump_json(indent=2))


def capabilities(settings: Settings, target_ids: list[str]) -> dict:
    from .config import TARGETS
    compiler = DockerCompiler(settings)
    result = {}
    for target_id in target_ids:
        target = TARGETS[target_id]
        try:
            identity = compiler.identity(target)
            with tempfile.TemporaryDirectory(prefix='npu-capabilities-') as temporary:
                root = Path(temporary)
                source = root / 'source'
                source.mkdir()
                contract = root / 'contract'
                write_json(contract / 'target.json', target.model_dump(mode='json'))
                probe = compiler._run(source, root / 'output', target, 'capability_probe.py', {'/contract': contract})
                if probe.status != 'passed':
                    raise ValueError(probe.message)
                identity['capability_probe'] = json.loads((root / 'output/capabilities.json').read_text())
            compile_status = 'available'
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            identity = {'error': str(exc)}
            compile_status = 'blocked'
        result[target_id] = {'toolchain': identity, 'target_compile': compile_status,
            'host_execution': 'amd-host-cpp-subset-v1' if target.vendor == 'amd' else 'openvino-cpu',
            'dataflow_simulation': 'amd-dataflow-subset-v1' if target.vendor == 'amd' else 'not_requested',
            'target_execution': unavailable_executor().model_dump(mode='json') if not target.hardware_runner else
                                {'status': 'unprobed', 'command': target.hardware_runner}}
    return result


artifact_sha256 = sha256
