import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from npu_agent.config import INTEL_NPU_3720, _runner
from npu_agent.executors import encode_tensor, validate_response
from npu_agent.models import ExecutionRequest, KernelManifest
from npu_agent.testcases import generate_cases
from npu_agent.validation import json_digest, sha256


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request(tmp_path, **updates):
    manifest = KernelManifest.model_validate_json((ROOT / 'tests/fixtures/backends/intel_npu_3720/add/manifest.json').read_text())
    cases = generate_cases(manifest)[:2]
    blob = tmp_path / 'compiled.blob'
    blob.write_bytes(b'npu blob')
    values = dict(run_id='fixture', target=INTEL_NPU_3720, artifacts={'compiled.blob': str(blob)},
        artifact_sha256={'compiled.blob': sha256(blob)}, manifest=manifest,
        manifest_sha256=json_digest(manifest.model_dump(mode='json')), abi_sha256='a' * 64,
        cases={case.id: {name: encode_tensor(value) for name, value in case.inputs.items() if isinstance(value, np.ndarray)}
               for case in cases}, compiler_fingerprint='fixture', timeout_seconds=10)
    return ExecutionRequest(**(values | updates))


class Port:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_names(self):
        return {self.tensor.name}

    def get_shape(self):
        return list(self.tensor.shape)

    def get_element_type(self):
        return SimpleNamespace(to_dtype=lambda: np.dtype(self.tensor.dtype))


class Compiled:
    def __init__(self, manifest, devices):
        self.inputs = [Port(t) for t in manifest.tensors if t.direction == 'input']
        self.outputs = [Port(t) for t in manifest.tensors if t.direction == 'output']
        self.devices = devices

    def get_property(self, name):
        assert name == 'EXECUTION_DEVICES'
        return self.devices

    def create_infer_request(self):
        state = {}

        def infer(feed):
            state['output'] = (feed['x'] + feed['y']).astype(np.float32)

        return SimpleNamespace(infer=infer, get_tensor=lambda name: SimpleNamespace(data=state[name]))


def fake_openvino(monkeypatch, task, devices=('CPU', 'NPU'), architecture='3720', execution='NPU'):
    class Core:
        available_devices = list(devices)

        def get_property(self, device, name):
            assert device == 'NPU'
            return {'DEVICE_ARCHITECTURE': architecture, 'FULL_DEVICE_NAME': 'Fake NPU',
                    'NPU_DRIVER_VERSION': '1004404'}[name]

        def import_model(self, stream, device):
            assert device == 'NPU' and stream.read() == b'npu blob'
            return Compiled(task.manifest, execution)

    module = ModuleType('openvino')
    module.Core = Core
    module.get_version = lambda: 'fake-2026.3.1'
    monkeypatch.setitem(sys.modules, 'openvino', module)
    monkeypatch.delitem(sys.modules, 'intel_execute_graph', raising=False)


def test_executor_requires_openvino_and_returns_checked_npu_outputs(tmp_path, monkeypatch):
    runner = script('intel_npu_executor')
    task = request(tmp_path)
    monkeypatch.setitem(sys.modules, 'openvino', None)
    with pytest.raises(ImportError):
        runner.run(task)
    fake_openvino(monkeypatch, task)
    result = runner.run(task)
    assert result.executor_kind == 'physical_npu' and result.hardware == '3720'
    assert result.runtime_identity == 'openvino/fake-2026.3.1;npu-arch/3720;npu-driver/1004404'
    stage = validate_response(task, result)
    assert stage.status == 'passed' and stage.correct is True and stage.engine == 'physical_npu'
    assert stage.cases_run == 2 and not result.timing_samples_ms


def test_executor_records_timing_with_provenance(tmp_path, monkeypatch):
    runner = script('intel_npu_executor')
    task = request(tmp_path, warmup_count=2, measurement_count=3)
    fake_openvino(monkeypatch, task)
    result = runner.run(task)
    assert len(result.timing_samples_ms) == 3 and 'Fake NPU' in result.timing_provenance
    assert validate_response(task, result).correct is True


@pytest.mark.parametrize('options,error,match', [
    ({'architecture': '4000'}, ValueError, 'architecture 4000 does not match target platform 3720'),
    ({'devices': ('CPU', 'GPU.0')}, RuntimeError, 'no NPU device'),
    ({'execution': ['CPU']}, RuntimeError, 'did not load on the NPU alone'),
])
def test_executor_rejects_wrong_device(tmp_path, monkeypatch, options, error, match):
    runner = script('intel_npu_executor')
    task = request(tmp_path)
    fake_openvino(monkeypatch, task, **options)
    with pytest.raises(error, match=match):
        runner.run(task)


def test_executor_rejects_wrong_artifacts_and_targets(tmp_path, monkeypatch):
    runner = script('intel_npu_executor')
    task = request(tmp_path)
    fake_openvino(monkeypatch, task)
    with pytest.raises(ValueError, match='blob required'):
        runner.run(task.model_copy(update={'artifacts': {'final.xclbin': task.artifacts['compiled.blob']}}))
    mismatched = INTEL_NPU_3720.model_copy(update={'compiler_properties': {'NPU_COMPILER_TYPE': 'PLUGIN', 'NPU_PLATFORM': '4000'}})
    with pytest.raises(ValueError, match='its own platform'):
        runner.run(task.model_copy(update={'target': mismatched}))
    Path(task.artifacts['compiled.blob']).write_bytes(b'changed blob')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        runner.run(task)


def test_bridge_translates_paths_and_relays_executor_json(tmp_path, monkeypatch, capsysbinary):
    bridge = script('wsl_npu_bridge')
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == 'wslpath':
            return subprocess.CompletedProcess(command, 0, stdout='\\\\wsl.localhost\\Ubuntu' + command[2].replace('/', '\\') + '\n')
        return subprocess.CompletedProcess(command, 7, stdout=b'{"result": 1}', stderr=b'executor log')

    monkeypatch.setattr(bridge.subprocess, 'run', run)
    monkeypatch.setattr(bridge, 'LOCK', tmp_path / 'npu.lock')
    payload = {'run_id': 'x', 'artifacts': {'compiled.blob': '/home/user/run/compiled.blob'}}
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode())))
    assert bridge.main(['--windows-python', '/mnt/c/runner/python.exe']) == 7
    command, kwargs = calls[-1]
    assert command[:2] == ['/mnt/c/runner/python.exe', '-B'] and command[2].endswith('\\scripts\\intel_npu_executor.py')
    assert json.loads(kwargs['input'])['artifacts'] == {'compiled.blob': '\\\\wsl.localhost\\Ubuntu\\home\\user\\run\\compiled.blob'}
    output = capsysbinary.readouterr()
    assert output.out == b'{"result": 1}' and output.err == b'executor log'


def test_runner_environment_is_opt_in_and_outside_the_fingerprint(monkeypatch):
    monkeypatch.delenv('NPU_AGENT_INTEL_NPU_3720_RUNNER', raising=False)
    assert _runner('intel_npu_3720') == {}
    monkeypatch.setenv('NPU_AGENT_INTEL_NPU_3720_RUNNER', "python 'scripts/wsl npu_bridge.py' --windows-python py.exe")
    settings = _runner('intel_npu_3720')
    assert settings['hardware_runner'] == ['python', 'scripts/wsl npu_bridge.py', '--windows-python', 'py.exe']
    configured = INTEL_NPU_3720.model_copy(update=settings)
    # DockerCompiler.identity excludes hardware_runner, so attaching an untimed runner keeps the toolchain fingerprint.
    assert configured.model_dump(exclude={'hardware_runner'}) == INTEL_NPU_3720.model_dump(exclude={'hardware_runner'})
    monkeypatch.setenv('NPU_AGENT_INTEL_NPU_3720_TIMING', '5,50')
    assert (_runner('intel_npu_3720')['executor_warmup_count'], _runner('intel_npu_3720')['executor_measurement_count']) == (5, 50)


def test_replay_reexecutes_compiled_attempts_without_touching_the_campaign(tmp_path, monkeypatch):
    from npu_agent.models import StageResult

    replay = script('replay_intel_local_npu')
    evaluation = tmp_path / 'evaluation'
    run = evaluation / 'cases' / 'c1' / 'cycle-2' / 'compile' / 'invocation-x'
    run.mkdir(parents=True)
    task = request(run)
    (run / 'execution-request.json').write_text(task.model_dump_json())
    (run / 'validation.json').write_text('{}')
    compiled = {'artifacts': {'validation.json': str(run / 'validation.json')}, 'validation': {'stages': {
        'target_compile': {'status': 'passed', 'artifacts': {'compiled.blob': task.artifact_sha256['compiled.blob']}},
        'host_execution': {'status': 'passed', 'correct': True},
        'target_execution': {'details': {'comparisons': [{'correct': False}, {'correct': True}]}}}}}
    cycles = [{'number': 1, 'status': 'failed', 'compile_result': {'validation': {'stages': {'target_compile': {'status': 'failed'}}}}},
              {'number': 2, 'status': 'passed', 'compile_result': compiled}]
    (evaluation / 'cases' / 'c1' / 'report.json').write_text(json.dumps({'cycles': cycles}))
    campaign = {'config': {'protocol': 'intel-npu-baseline-v1', 'targets': ['intel_npu_3720']}, 'cases': [{'id': 'c1', 'kernel': 'add'}]}
    (evaluation / 'campaign.json').write_text(json.dumps(campaign))
    before = {path: path.read_bytes() for path in evaluation.rglob('*') if path.is_file()}
    calls = []

    def execute(request_, runner, output_dir):
        calls.append((request_.run_id, runner, output_dir))
        return StageResult(status='failed', correct=False, engine='physical_npu', cases_run=2, details={
            'cases_passed': 1, 'fp16_cases_passed': 2, 'fp16_correct': True,
            'fp16_comparisons': [{'outputs': {'output': {'max_error_half_ulps': 1.25}}}]})

    monkeypatch.setattr(replay, 'execute_target', execute)
    summary = replay.replay(evaluation.resolve(), (tmp_path / 'replay').resolve(), ['runner'])
    assert calls == [('fixture', ['runner'], (tmp_path / 'replay' / 'c1' / 'cycle-2').resolve())]
    assert summary['attempts_replayed'] == 1 and summary['reproduces_recorded_manifest_counts'] == 1
    assert (summary['accepted_tasks_manifest_correct'], summary['accepted_tasks_fp16_correct']) == (0, 1)
    assert (summary['accepted_case_pass_rate_manifest'], summary['accepted_case_pass_rate_fp16']) == (0.5, 1.0)
    assert summary['tasks'][0]['fp16_max_half_ulps'] == 1.25
    assert before == {path: path.read_bytes() for path in evaluation.rglob('*') if path.is_file()}
    Path(task.artifacts['compiled.blob']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed since compilation'):
        replay.replay(evaluation.resolve(), (tmp_path / 'replay-2').resolve(), ['runner'])
