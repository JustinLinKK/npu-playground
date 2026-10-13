import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from npu_agent.config import AMD_XDNA2_NPU2
from npu_agent.executors import encode_tensor, validate_response
from npu_agent.models import ExecutionRequest, KernelManifest, StageResult
from npu_agent.testcases import generate_cases, serialize_cases
from npu_agent.validation import json_digest, sha256


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request(tmp_path):
    manifest = KernelManifest.model_validate_json((ROOT / 'tests/fixtures/backends/amd_xdna2_npu2/add/manifest.json').read_text())
    case = generate_cases(manifest)[0]
    artifacts = {}
    for name in ('final.xclbin', 'insts.bin'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        artifacts[name] = str(path)
    return ExecutionRequest(run_id='fixture', target=AMD_XDNA2_NPU2, artifacts=artifacts,
        artifact_sha256={name: sha256(Path(path)) for name, path in artifacts.items()}, manifest=manifest,
        manifest_sha256=json_digest(manifest.model_dump(mode='json')), abi_sha256='a' * 64,
        cases={case.id: {name: encode_tensor(value) for name, value in case.inputs.items() if isinstance(value, np.ndarray)}},
        compiler_fingerprint='fixture', timeout_seconds=10)


def test_local_executor_requires_xrt_and_checks_physical_outputs(tmp_path, monkeypatch):
    runner = script('xdna2_local_npu_executor')
    task = request(tmp_path)
    monkeypatch.setitem(sys.modules, 'pyxrt', None)
    with pytest.raises(ModuleNotFoundError):
        runner.run(task)

    class XRTTensor:
        device = 'npu'

        def __init__(self, value, dtype=None, device=None):
            self.value = np.array(value, dtype=dtype)

        def numpy(self):
            return self.value

    class NPUKernel:
        def __init__(self, xclbin, insts):
            assert (xclbin, insts) == (task.artifacts['final.xclbin'], task.artifacts['insts.bin'])

        def __call__(self, x, y, output):
            output.value[:] = x.value + y.value

    aie = ModuleType('aie')
    aie.__path__ = []
    iron = ModuleType('aie.iron')
    iron.tensor = XRTTensor
    utils = ModuleType('aie.utils')
    utils.__path__ = []
    utils.NPUKernel = NPUKernel
    runtime = ModuleType('aie.utils.xrtruntime')
    runtime.__path__ = []
    tensor = ModuleType('aie.utils.xrtruntime.tensor')
    tensor.XRTTensor = XRTTensor
    aie.iron, aie.utils = iron, utils
    utils.xrtruntime = runtime
    runtime.tensor = tensor
    for name, module in [('pyxrt', ModuleType('pyxrt')), ('aie', aie), ('aie.iron', iron),
                         ('aie.utils', utils), ('aie.utils.xrtruntime', runtime),
                         ('aie.utils.xrtruntime.tensor', tensor)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda name: '1.4.2')
    result = runner.run(task)
    assert result.executor_kind == 'physical_npu'
    assert validate_response(task, result).correct is True


def test_replay_uses_saved_inputs_and_keeps_offline_result_separate(tmp_path, monkeypatch):
    replay = script('replay_xdna2_local_npu')
    evaluation = tmp_path / 'evaluation'
    case_id = 'fixture'
    artifact_dir = evaluation / 'cases' / case_id / 'cycle-1' / 'compile'
    artifact_dir.mkdir(parents=True)
    task = request(artifact_dir)
    manifest_path = artifact_dir / 'manifest.json'
    manifest_path.write_text(task.manifest.model_dump_json())
    abi_path = artifact_dir / 'abi.json'
    abi_path.write_text('{"device": "npu2_1col"}')
    inputs = artifact_dir / 'inputs'
    saved = serialize_cases(generate_cases(task.manifest), inputs)
    artifacts = {**task.artifacts, 'abi.json': str(abi_path), 'contract/manifest.json': str(manifest_path),
                 **{f'inputs/{name}.npz': str(inputs / f'{name}.npz') for name in saved}}
    compiled_hashes = {name: sha256(Path(artifacts[name])) for name in ('final.xclbin', 'insts.bin', 'abi.json')}
    validation = {'offline_contract_met': True, 'input_sha256': saved,
                  'manifest_sha256': task.manifest_sha256,
                  'stages': {'target_compile': {'artifacts': compiled_hashes}}}
    report = {'cycles': [{'status': 'passed', 'compile_result': {
        'artifacts': artifacts, 'compiler_fingerprint': 'fixture', 'validation': validation}}]}
    (evaluation / 'cases' / case_id / 'report.json').write_text(json.dumps(report))
    (evaluation / 'campaign.json').write_text(json.dumps({'config': {'protocol': 'xdna2-baseline-v1'},
        'cases': [{'id': case_id, 'kernel': task.manifest.name}]}))
    seen = []
    def execute(request, command):
        seen.append(request)
        assert command[-1].endswith('xdna2_local_npu_executor.py')
        return StageResult(status='passed', correct=True, engine='physical_npu')
    monkeypatch.setattr(replay, 'execute_target', execute)
    summary = replay.replay(evaluation, tmp_path / 'local-npu')
    assert summary['physical_passes'] == summary['offline_accepted'] == 1
    assert len(seen[0].cases) == 12
    assert (evaluation / 'cases' / case_id / 'report.json').read_text() == json.dumps(report)
