"""Best-of-N with private selection inputs and explicit metamorphic relations."""
from unittest.mock import patch

import numpy as np

from npu_agent.study_v4 import StudyCompiler
from npu_agent.ir import execute_ir
from npu_agent.oracles import execute_oracle
from npu_agent.scenarios import ExecutableKernelIR
from npu_agent.testcases import TestCase, generate_cases
from npu_agent.validation import compare_outputs

from ..feedback_loop import best_profile


class SelectionCompiler(StudyCompiler):
    def compile_cases(self, candidate, output, manifest, target, cases):
        # Same isolated-input mechanism as research/hidden_evaluation.py; serial only.
        with patch('npu_agent.compilers.generate_cases', return_value=cases):
            return self.compile(candidate, output, manifest, target)


def selection_cases(manifest, seed):
    if set(range(seed, seed + 5)) & set(range(manifest.seed, manifest.seed + 5)):
        raise ValueError('selection random seeds overlap the feedback suite')
    cases = [TestCase(f'selection-{i}', case.inputs) for i, case in enumerate(
        generate_cases(manifest.model_copy(update={'seed': seed}))[:5])]
    inputs = {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in cases[0].inputs.items()}
    names = [t.name for t in manifest.tensors if t.direction == 'input']
    relation = None
    if manifest.oracle.operation == 'vector_add':
        inputs[names[0]], inputs[names[1]] = inputs[names[1]], inputs[names[0]]
        relation = {'kind': 'commutativity', 'left': cases[0].id, 'right': 'metamorphic', 'scale': 1}
    elif manifest.oracle.operation in ('reduce_sum', 'matmul', 'transpose', 'scalar_mul'):
        inputs[names[0]] = inputs[names[0]] * np.asarray(2, dtype=inputs[names[0]].dtype)
        relation = {'kind': 'linearity', 'left': cases[0].id, 'right': 'metamorphic', 'scale': 2}
    if relation:
        cases.append(TestCase('metamorphic', inputs))
    return cases, relation


def check_relation(result, relation, manifest):
    if relation is None:
        return {'status': 'unsupported', 'reason': 'No declared metamorphic relation for this operation.'}
    arrays = []
    for case_id in (relation['left'], relation['right']):
        path = result.artifacts.get(f'host/{case_id}.npz')
        if path is None:
            return {'status': 'blocked', 'reason': 'Evaluator did not preserve selection outputs.'}
        with np.load(path, allow_pickle=False) as archive:
            arrays.append({key: archive[key].copy() for key in archive.files})
    expected = {name: value * relation['scale'] for name, value in arrays[0].items()}
    comparison = compare_outputs(arrays[1], expected, manifest.tolerance, 'metamorphic')
    return {'status': 'passed' if comparison['correct'] else 'failed', 'relation': relation, 'comparison': comparison}


def check_ir(result, intermediate, cases, manifest):
    ir = ExecutableKernelIR.model_validate(intermediate)
    comparisons = []
    for case in cases:
        expected = execute_ir(ir, case.inputs)
        comparisons.append(compare_outputs(expected, execute_oracle(manifest, case.inputs), manifest.tolerance, case.id + '-ir'))
        path = result.artifacts.get(f'host/{case.id}.npz')
        if path is None:
            return {'status': 'blocked', 'reason': 'Evaluator did not preserve selection outputs.'}
        with np.load(path, allow_pickle=False) as archive:
            actual = {key: archive[key] for key in archive.files}
            comparisons.append(compare_outputs(actual, expected, manifest.tolerance, case.id + '-backend'))
    return {'status': 'passed' if all(c['correct'] for c in comparisons) else 'failed', 'comparisons': comparisons}


def select(attempts):
    passed = [item for item in attempts if item['status'] == 'passed'
              and item.get('selection', {}).get('status') == 'passed']
    return best_profile(passed)
