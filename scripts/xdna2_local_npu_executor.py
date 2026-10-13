"""Execute an XDNA2 ExecutionRequest on the local XRT NPU via MLIR-AIE 1.4.2."""
from __future__ import annotations

import importlib.metadata
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from npu_agent.executors import decode_tensor, encode_tensor
from npu_agent.models import ExecutionRequest, ExecutionResult
from npu_agent.validation import json_digest, sha256


def run(request: ExecutionRequest) -> ExecutionResult:
    if request.target.id != 'amd_xdna2_npu2' or request.target.hardware != 'npu2':
        raise ValueError('local executor requires the XDNA2 NPU2 target')
    if request.manifest_sha256 != json_digest(request.manifest.model_dump(mode='json')):
        raise ValueError('manifest hash mismatch')
    if set(request.artifacts) != {'final.xclbin', 'insts.bin'}:
        raise ValueError('XDNA2 artifact pair required')
    if request.warmup_count or request.measurement_count:
        raise ValueError('local replay supports correctness only')
    for name, path in request.artifacts.items():
        if sha256(Path(path)) != request.artifact_sha256[name]:
            raise ValueError(f'artifact hash mismatch: {name}')

    import pyxrt  # Fail rather than allow IRON's CPU fallback.
    import aie.iron as iron
    from aie.utils import NPUKernel
    from aie.utils.xrtruntime.tensor import XRTTensor

    kernel = NPUKernel(request.artifacts['final.xclbin'], request.artifacts['insts.bin'])
    outputs = {}
    for case_id, encoded in request.cases.items():
        buffers = []
        output_buffers = {}
        for tensor in request.manifest.tensors:
            if tensor.direction == 'input':
                value = decode_tensor(encoded[tensor.name])
                if tuple(value.shape) != tensor.shape or str(value.dtype) != tensor.dtype:
                    raise ValueError(f'input interface mismatch: {tensor.name}')
            else:
                value = np.full(tensor.shape, np.nan if tensor.dtype.startswith('float') else 0, dtype=tensor.dtype)
            buffer = iron.tensor(value, dtype=value.dtype, device='npu')
            if not isinstance(buffer, XRTTensor) or buffer.device != 'npu':
                raise RuntimeError('XRT NPU tensor unavailable')
            buffers.append(buffer)
            if tensor.direction == 'output':
                output_buffers[tensor.name] = buffer
        kernel(*buffers)
        outputs[case_id] = {name: encode_tensor(np.asarray(buffer.numpy())) for name, buffer in output_buffers.items()}
    return ExecutionResult(run_id=request.run_id, target_id=request.target.id, hardware=request.target.hardware,
        executor_kind='physical_npu', executed_artifact_sha256=request.artifact_sha256, outputs=outputs,
        logs='XRT NPU execution completed', runtime_identity='mlir-aie/' + importlib.metadata.version('mlir-aie') +
        ';pyxrt/' + str(getattr(pyxrt, '__version__', 'installed')), compatibility_checked=True,
        warmup_count=request.warmup_count, measurement_count=request.measurement_count,
        timing_scope=request.timing_scope)


if __name__ == '__main__':
    request = ExecutionRequest.model_validate_json(sys.stdin.read())
    with redirect_stdout(sys.stderr):
        result = run(request)
    print(result.model_dump_json())
