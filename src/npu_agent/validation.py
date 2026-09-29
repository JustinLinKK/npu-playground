from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import KernelManifest, NumericTolerance, StageResult, ValidationPolicy, ValidationResult


class UnsupportedContract(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def check_contract(manifest: KernelManifest) -> None:
    for tensor in manifest.tensors:
        if tensor.direction == 'inout':
            raise UnsupportedContract(f'tensors.{tensor.name}.direction=inout')
        if tensor.alias_of is not None:
            raise UnsupportedContract(f'tensors.{tensor.name}.alias_of')
        if tensor.layout != 'row_major':
            raise UnsupportedContract(f'tensors.{tensor.name}.layout={tensor.layout}')
        if tensor.strides is not None:
            strides = tuple(int(np.prod(tensor.shape[i + 1:])) for i in range(len(tensor.shape)))
            if tensor.strides != strides:
                raise UnsupportedContract(f'tensors.{tensor.name}.strides={tensor.strides}')
    allowed = {
        'vector_add': set(), 'scalar_mul': {'scale'},
        'reduce_sum': {'axis', 'keepdims', 'accumulation_dtype'},
        'matmul': {'output_dtype'}, 'transpose': {'axes'}, 'softmax': {'axis'},
        'layer_norm': {'axis', 'epsilon'}, 'attention': {'scale'},
        'moving_average': {'window'}, 'conv2d': {'padding'}, 'gamma_correction': {'gamma'},
        'sigmoid': set(), 'silu': set(),
    }
    unknown = set(manifest.oracle.parameters) - allowed[manifest.oracle.operation]
    if unknown:
        raise UnsupportedContract(f'oracle.parameters.{sorted(unknown)[0]}')
    for scalar in manifest.scalars:
        if scalar.name not in {'n', 'window', 'gamma', 'scale'}:
            raise UnsupportedContract(f'scalars.{scalar.name}: no builtin scalar binding')
        if scalar.name != 'n' and scalar.name not in manifest.oracle.parameters:
            raise UnsupportedContract(f'scalars.{scalar.name}: missing oracle parameter binding')
        if scalar.name in manifest.oracle.parameters and scalar.value != manifest.oracle.parameters[scalar.name]:
            raise ValueError(f'scalar {scalar.name} disagrees with oracle parameter')
        if scalar.name == 'n':
            first = next(t for t in manifest.tensors if t.direction == 'input')
            if scalar.value != int(np.prod(first.shape)):
                raise ValueError('scalar n disagrees with static input size')
    params = manifest.oracle.parameters
    if params.get('window', 1) <= 0 or params.get('padding', 0) < 0 or params.get('epsilon', 1) <= 0:
        raise ValueError('invalid window, padding, or epsilon')
    if manifest.oracle.operation in ('sigmoid', 'silu') and manifest.numerical_domain != 'finite':
        raise UnsupportedContract('numerical_domain: composed operators require finite inputs')


def compare_outputs(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], tolerance: NumericTolerance,
                    case_id: str) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {'test_id': case_id, 'correct': True, 'outputs': {}}
    if set(actual) != set(expected):
        return {'test_id': case_id, 'correct': False, 'error': 'output_names',
                'expected': sorted(expected), 'actual': sorted(actual)}
    for name, reference in expected.items():
        value = np.asarray(actual[name])
        if value.shape != reference.shape or value.dtype != reference.dtype:
            diagnostics['outputs'][name] = {'correct': False, 'error': 'shape_or_dtype',
                'expected_shape': list(reference.shape), 'actual_shape': list(value.shape),
                'expected_dtype': str(reference.dtype), 'actual_dtype': str(value.dtype)}
            diagnostics['correct'] = False
            continue
        if np.issubdtype(reference.dtype, np.integer):
            matches = value == reference
            absolute = np.abs(value.astype(np.longdouble) - reference.astype(np.longdouble))
            scaled = absolute
        else:
            a, e = value.astype(np.float64), reference.astype(np.float64)
            finite = np.isfinite(a) & np.isfinite(e)
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                absolute = np.where(finite, np.abs(a - e), 0)
                bound = tolerance.atol + tolerance.rtol * np.abs(e)
                scaled = np.divide(absolute, bound, out=np.zeros_like(absolute), where=bound > 0)
                scaled = np.where((bound == 0) & (absolute > 0), np.inf, scaled)
                matches = finite & (absolute <= bound)
            matches |= np.isinf(e) & (a == e)
            if tolerance.equal_nan:
                matches |= np.isnan(a) & np.isnan(e)
        count = int(np.count_nonzero(~matches))
        max_scaled = float(np.max(scaled, initial=0))
        diagnostics['outputs'][name] = {
            'correct': count == 0, 'mismatch_count': count,
            'max_absolute_error': float(np.max(absolute, initial=0)),
            'max_scaled_error': max_scaled if np.isfinite(max_scaled) else None,
            'first_failing_indices': np.argwhere(~matches)[:8].tolist(),
        }
        diagnostics['correct'] &= count == 0
    return diagnostics


def meets_policy(result: ValidationResult, policy: ValidationPolicy | str) -> bool:
    policy = ValidationPolicy(policy)
    stages = result.stages
    if stages.get('target_compile', StageResult(status='not_requested')).status != 'passed':
        return False
    if policy == ValidationPolicy.COMPILE_ONLY:
        return True
    if result.legacy:
        return False
    semantic = stages.get("semantic_validation")
    if semantic and semantic.status != "passed":
        return False
    required = ['oracle_validation']
    if policy == ValidationPolicy.OFFLINE_VALIDATED:
        required += ['host_execution']
        if result.target_id.startswith('amd_'):
            required += ['dataflow_simulation']
    else:
        required += ['target_execution']
    return all(name in stages and stages[name].status == 'passed' and stages[name].correct is True
               for name in required)


def finalize_validation(result: ValidationResult, policy: ValidationPolicy | str) -> ValidationResult:
    result.policy = ValidationPolicy(policy)
    result.offline_contract_met = meets_policy(result, ValidationPolicy.OFFLINE_VALIDATED)
    result.all_three_requirements_met = result.offline_contract_met and meets_policy(result, ValidationPolicy.TARGET_EXECUTED)
    result.policy_met = meets_policy(result, policy)
    return result


def policy_exit_code(result: ValidationResult) -> int:
    if result.policy_met:
        return 0
    required = ['target_compile']
    if result.policy != ValidationPolicy.COMPILE_ONLY:
        required += ['oracle_validation']
        if 'semantic_validation' in result.stages:
            required += ['semantic_validation']
        required += ['target_execution'] if result.policy == ValidationPolicy.TARGET_EXECUTED else ['host_execution']
        if result.policy == ValidationPolicy.OFFLINE_VALIDATED and result.target_id.startswith('amd_'):
            required += ['dataflow_simulation']
    statuses = [result.stages[name].status for name in required if name in result.stages]
    return 1 if 'failed' in statuses else 3


def legacy_validation(payload: dict[str, Any], candidate_id: str, target_id: str) -> ValidationResult:
    stages = {'target_compile': StageResult(status='passed' if payload.get('success', payload.get('compile_success')) else 'failed',
                                            engine='legacy', representation='target_artifact'),
              'target_execution': StageResult(status='blocked', reason_code='LEGACY_EXECUTION_UNVERIFIED')}
    host = payload.get('host_correct', payload.get('host_correctness'))
    if host is not None:
        stages['host_execution'] = StageResult(status='passed' if host else 'failed', correct=host,
                                              engine='legacy', representation='host_graph')
    return finalize_validation(ValidationResult(candidate_id=candidate_id, target_id=target_id,
                                                legacy=True, stages=stages), ValidationPolicy.COMPILE_ONLY)
