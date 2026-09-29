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
