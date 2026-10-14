from __future__ import annotations

import base64
import hashlib
import os
import signal
from pathlib import Path
import subprocess
import time

import numpy as np

from .models import ExecutionRequest, ExecutionResult, NumericTolerance, StageResult, TensorPayload
from .oracles import execute_oracle
from .validation import compare_outputs, sha256

FP16_UNIT_ROUNDOFF = 2.0 ** -11
# Allowance for rounding inside an FP16 device (intermediates, accumulation, approximate transcendentals),
# in half-ulps of each output tensor's peak magnitude. Output and input rounding are modeled exactly instead.
FP16_TOLERANCE_HALF_ULPS = 8
FP16_ABSOLUTE_FLOOR = 2.0 ** -24  # smallest FP16 subnormal; covers all-zero references
FP16_VERDICT = {'reference': 'independent oracle on FP16-rounded float inputs and scalars, result rounded to FP16',
                'tolerance_half_ulps': FP16_TOLERANCE_HALF_ULPS, 'scale': 'peak magnitude of each output tensor',
                'absolute_floor': FP16_ABSOLUTE_FLOOR}


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


def _to_fp16(value):
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float16).astype(value.dtype)
    if isinstance(value, float):
        return float(np.float16(value))
    return value


def fp16_reference(manifest, inputs: dict) -> dict[str, np.ndarray]:
    """What an exact FP16 device returns: FP16 inputs in, the true result of those inputs rounded to FP16 out."""
    return {name: _to_fp16(value) for name, value in
            execute_oracle(manifest, {name: _to_fp16(value) for name, value in inputs.items()}).items()}


def compare_fp16(actual: dict[str, np.ndarray], reference: dict[str, np.ndarray], tolerance: NumericTolerance,
                 case_id: str) -> dict:
    diagnostics = {'test_id': case_id, 'correct': True, 'outputs': {}}
    if set(actual) != set(reference):
        return {'test_id': case_id, 'correct': False, 'error': 'output_names'}
    for name, expected in reference.items():
        value = np.asarray(actual[name])
        if value.shape != expected.shape:
            diagnostics['outputs'][name] = {'correct': False, 'error': 'shape'}
            diagnostics['correct'] = False
            continue
        a, e = value.astype(np.float64), expected.astype(np.float64)
        finite = np.isfinite(e)
        peak = float(np.max(np.abs(e[finite]), initial=0.0))
        with np.errstate(invalid='ignore'):
            error = np.where(finite & np.isfinite(a), np.abs(a - e), 0.0)
        bound = (FP16_TOLERANCE_HALF_ULPS * FP16_UNIT_ROUNDOFF * peak + FP16_ABSOLUTE_FLOOR
                 if np.issubdtype(expected.dtype, np.floating) else 0.0)
        matches = (finite & np.isfinite(a) & (error <= bound)) | (np.isinf(e) & (a == e))
        if tolerance.equal_nan:
            matches |= np.isnan(a) & np.isnan(e)
        largest = float(np.max(error, initial=0.0))
        count = int(np.count_nonzero(~matches))
        diagnostics['outputs'][name] = {'correct': count == 0, 'mismatch_count': count, 'max_absolute_error': largest,
            'max_error_half_ulps': largest / (FP16_UNIT_ROUNDOFF * peak) if peak else (0.0 if largest == 0 else None),
            'first_failing_indices': np.argwhere(~matches)[:8].tolist()}
        diagnostics['correct'] &= count == 0
    return diagnostics


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
    comparisons, fp16 = [], []
    for case_id, encoded in request.cases.items():
        inputs = {name: decode_tensor(value) for name, value in encoded.items()}
        inputs.update({s.name: s.value for s in request.manifest.scalars})
        expected = execute_oracle(request.manifest, inputs)
        actual = {name: decode_tensor(value) for name, value in response.outputs[case_id].items()}
        comparisons.append(compare_outputs(actual, expected, request.manifest.tolerance, case_id))
        try:
            fp16.append(compare_fp16(actual, fp16_reference(request.manifest, inputs), request.manifest.tolerance, case_id))
        except (ValueError, ArithmeticError) as exc:
            fp16.append({'test_id': case_id, 'correct': None, 'error': f'FP16 reference unavailable: {exc}'})
    correct = all(item['correct'] for item in comparisons)
    details = response.model_dump(mode='json', exclude={'outputs'})
    # The manifest tolerance alone decides this stage; the FP16 verdict is a separate device-precision metric.
    details.update(comparisons=comparisons, cases_passed=sum(item['correct'] for item in comparisons),
                   fp16_verdict=FP16_VERDICT, fp16_comparisons=fp16,
                   fp16_cases_passed=sum(item['correct'] is True for item in fp16),
                   fp16_correct=None if any(item['correct'] is None for item in fp16) else all(item['correct'] for item in fp16))
    return StageResult(status='passed' if correct else 'failed', correct=correct,
                       engine=response.executor_kind, representation='target_binary', cases_run=len(comparisons),
                       error_category=None if correct else 'numerical_mismatch', details=details)


def execute_target(request: ExecutionRequest, command: list[str] | None, output_dir: Path | None = None) -> StageResult:
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
        if output_dir is not None:
            # Keep the device's actual tensors so later verdicts never require re-execution.
            output_dir.mkdir(parents=True, exist_ok=True)
            for case_id, encoded in response.outputs.items():
                path = output_dir / f'{case_id}.npz'
                np.savez(path, **{name: decode_tensor(value) for name, value in encoded.items()})
                result.artifacts[f'{output_dir.name}/{path.name}'] = sha256(path)
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
