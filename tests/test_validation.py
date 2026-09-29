import copy
from pathlib import Path

import numpy as np
import pytest

from npu_agent import ir, oracles
from npu_agent.models import KernelManifest, NumericTolerance, StageResult, ValidationResult
from npu_agent.testcases import generate_cases, tail_manifests
from npu_agent.validation import UnsupportedContract, compare_outputs, finalize_validation, policy_exit_code
from test_ir_and_models import ir_for_manifest


def manifest():
    return KernelManifest.model_validate_json(Path('examples/classic/01_cuda_vector_add/manifest.json').read_text())


@pytest.mark.parametrize('operation', [g[0] for g in oracles.GOLDENS])
def test_independent_golden(operation):
    oracles.validate_goldens(operation)


def test_ir_and_oracle_mutations_are_detected(monkeypatch):
    m = manifest()
    monkeypatch.setattr(ir, 'evaluate_operation', lambda op, inputs, attrs: inputs[0] - inputs[1])
    assert not ir.validate_ir(ir_for_manifest(m), m).passed
    monkeypatch.setattr(oracles, 'reference', lambda op, args, p: np.array([999.]))
    with pytest.raises(ValueError, match='golden failed'):
        oracles.validate_goldens('vector_add')


def test_comparison_rejects_dtype_shape_nan_and_infinity():
    tol = NumericTolerance()
    expected = {'out': np.array([0., 1., np.inf, np.nan], dtype=np.float32)}
    assert not compare_outputs(expected, expected, tol, 'nan')['correct']
    assert compare_outputs(expected, expected, NumericTolerance(equal_nan=True), 'nan')['correct']
    for values in [np.array([0., 1., -np.inf, np.nan], dtype=np.float32), np.zeros(4, np.float64), np.zeros((2, 2), np.float32)]:
        assert not compare_outputs({'out': values}, expected, tol, 'invalid')['correct']
    actual = {'out': np.array([0., 2.], dtype=np.float32)}
    report = compare_outputs(actual, {'out': np.array([0., 1.], dtype=np.float32)}, tol, 'mismatch')
    assert report['outputs']['out']['mismatch_count'] == 1
    assert report['outputs']['out']['first_failing_indices'] == [[1]]


@pytest.mark.parametrize('field,value', [('alias_of', 'y'), ('strides', [2]), ('layout', 'column_major')])
def test_unrealized_memory_contract_is_unsupported(field, value):
    m = manifest()
    setattr(m.tensors[0], field, value)
    with pytest.raises(UnsupportedContract, match=field):
        generate_cases(m)


def test_case_bytes_are_repeatable_and_static_tail_manifests_are_distinct():
    first, second = generate_cases(manifest()), generate_cases(manifest())
    assert len(first) == 12
    assert [c.id for c in first] == [c.id for c in second]
    assert all(a.inputs['x'].tobytes() == b.inputs['x'].tobytes() for a, b in zip(first, second))
    tails = tail_manifests(manifest(), 16)
    assert [m.tensors[0].shape for m in tails] == [(15,), (16,), (17,), (35,)]
    assert len({m.name for m in tails}) == 4


def test_policy_requires_independent_stages_and_exact_execution():
    result = ValidationResult(candidate_id='a', target_id='amd_xdna2_npu2', stages={
        'target_compile': StageResult(status='passed')})
    assert finalize_validation(result, 'compile-only').policy_met
    assert not finalize_validation(result, 'offline-validated').policy_met
    for name in ('oracle_validation', 'host_execution', 'dataflow_simulation'):
        result.stages[name] = StageResult(status='passed', correct=True)
    assert finalize_validation(result, 'offline-validated').policy_met
    assert not finalize_validation(result, 'target-executed').policy_met
    result.stages['target_execution'] = StageResult(status='blocked', reason_code='NO_VALIDATED_TARGET_EXECUTOR')
    assert not result.all_three_requirements_met
    result.stages['semantic_validation'] = StageResult(status='failed', correct=False)
    assert policy_exit_code(finalize_validation(result, 'offline-validated')) == 1


def test_unexecuted_stage_cannot_claim_correctness():
    with pytest.raises(ValueError, match='unknown correctness'):
        StageResult(status='blocked', correct=True)
