from pathlib import Path

import numpy as np
import pytest

from npu_agent.config import INTEL_NPU_4000
from npu_agent.executors import decode_tensor, encode_tensor, execute_target, validate_response
from npu_agent.models import ExecutionRequest, ExecutionResult, KernelManifest
from npu_agent.oracles import execute_oracle
from npu_agent.testcases import generate_cases
from npu_agent.validation import json_digest


def request():
    m = KernelManifest.model_validate_json(Path('examples/classic/01_cuda_vector_add/manifest.json').read_text())
    case = generate_cases(m)[0]
    return ExecutionRequest(run_id='test', target=INTEL_NPU_4000, artifacts={'compiled.blob': '/missing'},
        artifact_sha256={'compiled.blob': 'a' * 64}, manifest=m, manifest_sha256=json_digest(m.model_dump(mode='json')),
        abi_sha256='b' * 64, cases={case.id: {k: encode_tensor(v) for k, v in case.inputs.items() if isinstance(v, np.ndarray)}},
        compiler_fingerprint='toolchain', timeout_seconds=1)


def response(r):
    return ExecutionResult(run_id=r.run_id, target_id=r.target.id, hardware=r.target.hardware,
        executor_kind='physical_npu', executed_artifact_sha256=r.artifact_sha256,
        outputs={k: {n: encode_tensor(v) for n, v in execute_oracle(r.manifest, {n: decode_tensor(t) for n, t in case.items()}).items()}
                 for k, case in r.cases.items()}, logs='test fixture', runtime_identity='test-fixture', compatibility_checked=True)


def test_no_executor_is_blocked_and_inputs_do_not_include_expected():
    r = request()
    stage = execute_target(r, None)
    assert stage.status == 'blocked' and stage.correct is None
    assert 'expected' not in r.model_dump_json()


@pytest.mark.parametrize('field,value', [('hardware', '3720'), ('target_id', 'amd_xdna2_npu2'),
    ('executed_artifact_sha256', {'compiled.blob': 'c' * 64}), ('outputs', {})])
def test_executor_provenance_rejected(field, value):
    r = request()
    result = response(r)
    setattr(result, field, value)
    with pytest.raises(ValueError):
        validate_response(r, result)


def test_supervisor_compares_actual_tensors():
    r = request()
    result = response(r)
    assert validate_response(r, result).correct
    name = next(iter(result.outputs))
    result.outputs[name]['output'] = encode_tensor(np.zeros(4096, dtype=np.float32))
    assert validate_response(r, result).correct is False


@pytest.mark.parametrize('samples,count', [([float('nan')], 1), ([-1.], 1), ([1.], 2)])
def test_malformed_timing_is_rejected(samples, count):
    payload = response(request()).model_dump()
    payload.update(timing_samples_ms=samples, measurement_count=count, timing_provenance='test')
    with pytest.raises(ValueError):
        ExecutionResult.model_validate(payload)


def test_fp16_verdict_separates_device_precision_from_wrong_results():
    r = request()
    result = response(r)
    case_id, case = next(iter(r.cases.items()))
    rounded = {n: decode_tensor(t).astype(np.float16).astype(np.float32) for n, t in case.items()}
    # An exact FP16 device: FP16 inputs, true sum, FP16 output.
    device = {n: v.astype(np.float16).astype(np.float32) for n, v in execute_oracle(r.manifest, rounded).items()}
    result.outputs[case_id] = {n: encode_tensor(v) for n, v in device.items()}
    stage = validate_response(r, result)
    assert stage.correct is False and stage.details['cases_passed'] == 0
    assert stage.details['fp16_correct'] is True and stage.details['fp16_cases_passed'] == 1
    assert stage.details['fp16_comparisons'][0]['outputs']['output']['max_error_half_ulps'] == 0
    result.outputs[case_id] = {n: encode_tensor(np.zeros_like(v)) for n, v in device.items()}
    assert validate_response(r, result).details['fp16_correct'] is False


def test_executor_outputs_are_saved_with_hashes(tmp_path):
    import sys

    from npu_agent.validation import sha256

    blob = tmp_path / 'compiled.blob'
    blob.write_bytes(b'blob')
    r = request().model_copy(update={'artifacts': {'compiled.blob': str(blob)}, 'timeout_seconds': 60,
                                     'artifact_sha256': {'compiled.blob': sha256(blob)}})
    payload = tmp_path / 'response.json'
    payload.write_text(response(r).model_dump_json())
    script = tmp_path / 'executor.py'
    script.write_text(f'import sys\nsys.stdin.read()\nsys.stdout.write(open({str(payload)!r}).read())\n')
    stage = execute_target(r, [sys.executable, str(script)], tmp_path / 'target_execution')
    assert stage.status == 'passed' and stage.details['fp16_correct'] is True
    (name, digest), = stage.artifacts.items()
    assert name.startswith('target_execution/') and sha256(tmp_path / name) == digest
    case_id, case = next(iter(r.cases.items()))
    expected = execute_oracle(r.manifest, {n: decode_tensor(t) for n, t in case.items()})
    with np.load(tmp_path / name) as saved:
        assert np.array_equal(saved['output'], expected['output'])
