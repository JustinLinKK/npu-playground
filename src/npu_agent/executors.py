from __future__ import annotations

import base64
import hashlib
import os
import signal
from pathlib import Path
import subprocess
import time

import numpy as np

from .models import ExecutionRequest, ExecutionResult, StageResult, TensorPayload
from .oracles import execute_oracle
from .validation import compare_outputs, sha256


def encode_tensor(value: np.ndarray) -> TensorPayload:
    value = np.ascontiguousarray(value)
    raw = value.tobytes()
    return TensorPayload(dtype=str(value.dtype), shape=list(value.shape),
                         data_base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())


def decode_tensor(payload: TensorPayload) -> np.ndarray:
    if payload.dtype not in ('float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64'):
        raise ValueError('unsupported tensor dtype')
    if not payload.shape or any(d <= 0 for d in payload.shape):
        raise ValueError('invalid tensor shape')
    raw = base64.b64decode(payload.data_base64, validate=True)
    if hashlib.sha256(raw).hexdigest() != payload.sha256:
        raise ValueError('tensor hash mismatch')
    size = int(np.prod(payload.shape, dtype=object)) * np.dtype(payload.dtype).itemsize
    if len(raw) != size:
        raise ValueError('tensor byte count mismatch')
    return np.frombuffer(raw, dtype=payload.dtype).reshape(payload.shape)


def unavailable_executor() -> StageResult:
    return StageResult(status='blocked', engine='none', reason_code='NO_VALIDATED_TARGET_EXECUTOR',
                       message='No validated executor for the emitted target artifact is configured.')


def validate_response(request: ExecutionRequest, response: ExecutionResult) -> StageResult:
    if (response.run_id != request.run_id or response.target_id != request.target.id
            or response.hardware != request.target.hardware):
        raise ValueError('target or run identity mismatch')
    if response.executed_artifact_sha256 != request.artifact_sha256:
        raise ValueError('executed artifact hash mismatch')
    if set(response.outputs) != set(request.cases):
        raise ValueError('execution case count mismatch')
    if (response.measurement_count != request.measurement_count or response.warmup_count != request.warmup_count
            or response.timing_scope != request.timing_scope):
        raise ValueError('execution timing contract mismatch')
    comparisons = []
    for case_id, encoded in request.cases.items():
        inputs = {name: decode_tensor(value) for name, value in encoded.items()}
        inputs.update({s.name: s.value for s in request.manifest.scalars})
        expected = execute_oracle(request.manifest, inputs)
        actual = {name: decode_tensor(value) for name, value in response.outputs[case_id].items()}
        comparisons.append(compare_outputs(actual, expected, request.manifest.tolerance, case_id))
    correct = all(item['correct'] for item in comparisons)
    details = response.model_dump(mode='json', exclude={'outputs'})
    details['comparisons'] = comparisons
    return StageResult(status='passed' if correct else 'failed', correct=correct,
                       engine=response.executor_kind, representation='target_binary', cases_run=len(comparisons),
                       error_category=None if correct else 'numerical_mismatch', details=details)


def execute_target(request: ExecutionRequest, command: list[str] | None) -> StageResult:
    if not command:
        return unavailable_executor()
    started = time.perf_counter()
    try:
        for name, path in request.artifacts.items():
            if sha256(Path(path)) != request.artifact_sha256[name]:
                raise ValueError('artifact changed before execution')
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(request.model_dump_json(), timeout=request.timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        if process.returncode:
            return StageResult(status='failed', error_category='execution_process', message=stderr[-4000:],
                               duration_seconds=time.perf_counter() - started)
        response = ExecutionResult.model_validate_json(stdout)
        result = validate_response(request, response)
        for name, path in request.artifacts.items():
            if sha256(Path(path)) != request.artifact_sha256[name]:
                raise ValueError('artifact changed during execution')
        result.duration_seconds = time.perf_counter() - started
        return result
    except subprocess.TimeoutExpired:
        return StageResult(status='failed', error_category='timeout', message='target executor timed out',
                           duration_seconds=time.perf_counter() - started)
    except (ValueError, KeyError, OSError) as exc:
        return StageResult(status='failed', error_category='malformed_report', message=str(exc),
                           duration_seconds=time.perf_counter() - started)
