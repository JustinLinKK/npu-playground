import json
import shutil

import numpy as np
import pytest

from npu_agent.artifacts import materialize_bundle
from npu_agent.config import AMD_XDNA2_NPU2, INTEL_NPU_4000
from npu_agent.models import CodeBundle, CompileResult, GeneratedFile, IROperation, KernelManifest, StageResult, ValidationResult
from npu_agent.oracles import execute_oracle
from npu_agent.providers import ProviderError, ProviderResponse
from npu_agent.scenarios import ExecutableKernelIR
from npu_agent.sim.amd_host import HostKernel, build_host
from npu_agent.testcases import generate_cases
from npu_agent.validation import json_digest, sha256, write_json

from bucketed_length import api_constraints, candidate_search, executable_ir, feature_retrieval, feedback_loop, template_generation, verified_skills
from bucketed_length.__main__ import main
from bucketed_length.runner import ARMS, ROOT, RunConfig, metrics, run


MANIFEST = ROOT / 'tests/fixtures/backends/amd_xdna2_npu2/add/manifest.json'
SOURCE = ROOT / 'examples/classic/01_cuda_vector_add/kernel.cu'


def manifest():
    return KernelManifest.model_validate_json(MANIFEST.read_text())


def ir(operation='add'):
    contract = manifest()
    return ExecutableKernelIR(name=contract.name, inputs=contract.tensors[:2], outputs=contract.tensors[2:],
        scalars=contract.scalars, iteration_domains=[], numeric_behavior=[], operations=[
            IROperation(id='sum', op=operation, inputs=['x', 'y'], output='output')])


def bundle(target=AMD_XDNA2_NPU2, bad=False):
    if target.vendor == 'intel':
        return CodeBundle(backend=target.backend, files=[GeneratedFile(relative_path='model.py', content='import openvino as ov\n')])
    return CodeBundle(backend=target.backend, files=[GeneratedFile(relative_path='design.py', content='print("module {}")\n'),
        GeneratedFile(relative_path='kernel.cc', content='#include <adf.h>\n' if bad else 'void compute() {}\n')])


class Provider:
    name = 'fake'

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.prompts = []
        self.schemas = []

    def generate(self, prompt, schema, model):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if self.responses:
            value = self.responses.pop(0)
        elif schema is ExecutableKernelIR:
            value = ir()
        elif schema is executable_ir.JsonSummary:
            value = schema(operation='add', indexing='each element', numeric_behavior='float32')
        elif schema is template_generation.MicroKernel:
            value = schema(body='for (int i=0; i<64; ++i) out[i] = x0[i] + x1[i];')
        else:
            value = bundle()
        if isinstance(value, Exception):
            raise value
        return ProviderResponse(value, {'usage': {'input_tokens': 10, 'output_tokens': 5}}, value.model_dump(mode='json'))


class Compiler:
    """Explicitly synthetic stage evidence; never run generated code in a unit test."""
    fingerprint = 'fixture-fingerprint'

    def __init__(self, failures=0, selection_failures=0):
        self.calls = 0
        self.failures = failures
        self.selection_failures = selection_failures
        self.selection_calls = 0

    def identity(self, target):
        return {'fingerprint': self.fingerprint, 'image_id': 'synthetic-fixture'}

    def compile(self, candidate, output, contract, target):
        self.calls += 1
        passed = self.calls > self.failures
        hashes = {p.name: sha256(p) for p in candidate.iterdir()}
        result = CompileResult(success=passed, exit_code=0 if passed else 1,
            stderr='' if passed else 'unknown symbol: missing_vector_add', compiler_fingerprint=self.fingerprint,
            validation=ValidationResult(candidate_id='fixture', target_id=target.id,
                source_sha256=json_digest(hashes), manifest_sha256=json_digest(contract.model_dump(mode='json')),
                toolchain_fingerprint=self.fingerprint, reference_origin='independent_builtin', stages={
                    'target_compile': StageResult(status='passed' if passed else 'failed'),
                    'oracle_validation': StageResult(status='passed', correct=True, cases_run=12),
                    'host_execution': StageResult(status='passed', correct=True, cases_run=12),
                    'dataflow_simulation': StageResult(status='passed', correct=True, cases_run=12),
                    'target_execution': StageResult(status='blocked'),
                }))
        return result

    def compile_cases(self, candidate, output, contract, target, cases):
        self.selection_calls += 1
        result = self.compile(candidate, output, contract, target)
        for case in cases:
            path = output / 'host' / f'{case.id}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **execute_oracle(contract, case.inputs))
            result.artifacts[f'host/{case.id}.npz'] = str(path)
        if self.selection_calls <= self.selection_failures:
            result.validation.stages['host_execution'] = StageResult(status='failed', correct=False)
        return result


def execute(tmp_path, direction='feedback_loop', arm='repair', provider=None, compiler=None, **kwargs):
    return run(SOURCE, MANIFEST, AMD_XDNA2_NPU2, provider or Provider(), compiler or Compiler(),
               RunConfig(direction=direction, arm=arm, model='test', **kwargs), tmp_path / 'run')


@pytest.mark.parametrize(('direction', 'arm'), [(d, a) for d, arms in ARMS.items() for a in arms])
def test_every_direction_and_arm_runs_without_external_calls(tmp_path, direction, arm):
    state = execute(tmp_path, direction, arm)
    assert state['status'] == 'completed'
    assert state['synthetic'] is True
    assert state['winner'] is not None
    assert (tmp_path / 'run/evidence.sha256.json').is_file()
    assert state['attempts'][0]['profile']['vectorization_percent'] is None


def test_ir_failure_prevents_compile_and_validated_ir_is_reused(tmp_path):
    provider = Provider([ir('subtract'), ir(), bundle(), bundle()])
    compiler = Compiler(failures=1)
    state = execute(tmp_path, 'executable_ir', 'executable_ir', provider, compiler)
    assert state['status'] == 'completed'
    assert compiler.calls == 2
    assert provider.schemas == [ExecutableKernelIR, ExecutableKernelIR, CodeBundle, CodeBundle]
    assert state['attempts'][0]['semantic_validation']['status'] == 'failed'
    assert 'missing_vector_add' in provider.prompts[-1]
    assert all('"oracle"' not in p for p in provider.prompts)


def test_repair_limit_means_initial_plus_five_and_stops_on_pass(tmp_path):
    compiler = Compiler(failures=99)
    state = execute(tmp_path, compiler=compiler)
    assert state['status'] == 'failed' and compiler.calls == 6
    second = execute(tmp_path / 'one', arm='one_shot', compiler=Compiler(failures=1))
    assert len(second['attempts']) == 1
    with pytest.raises(ValueError):
        RunConfig(direction='feedback_loop', arm='repair', model='x', max_repairs=6)


def test_guided_prompts_and_history_prevent_api_error_amnesia(tmp_path):
    provider = Provider()
    execute(tmp_path, provider=provider, compiler=Compiler(failures=2))
    assert 'Runtime(seq_fn, fn_args)' in provider.prompts[0]
    assert 'aie.iron.placers are not available' in provider.prompts[0]
    history = [{'number': 1, 'error': 'old API error', 'selection': {'error': 'private secret'}},
               {'number': 2, 'error': 'new API error'}]
    prompt = feedback_loop.feedback(history[-1], history)
    assert 'old API error' in prompt and 'new API error' in prompt
    assert 'private secret' not in prompt


def test_search_generates_all_initial_candidates_before_repairs(tmp_path):
    state = execute(tmp_path, 'candidate_search', 'best_of_n', compiler=Compiler(failures=99),
                    population=4, max_repairs=2, token_budget=150)
    assert [(a['member'], a['repair']) for a in state['attempts'][:4]] == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert metrics(state)['candidates_started'] == 4
    assert metrics(state)['candidates_planned'] == 4
    assert [c['schema'] for c in state['calls']].count('ExecutableKernelIR') == 4


def test_skeleton_prompt_warns_about_observed_vector_portability_failures(tmp_path):
    provider = Provider()
    state = execute(tmp_path, 'template_generation', 'skeleton', provider=provider)
    assert state['status'] == 'completed'
    assert 'scalar pointer loops for multiply/dot products' in provider.prompts[0]
    assert 'aie::mul and aie::mac are unsupported by the host evaluator' in provider.prompts[0]
    assert 'not native vector arithmetic or dynamic vector lane indexing' in provider.prompts[0]


def test_budget_includes_rejected_calls_and_never_evaluates_over_budget_response(tmp_path):
    error = ProviderError('invalid schema', {'exit_code': 0, 'schema_valid': False,
        'usage': {'input_tokens': 10, 'output_tokens': 5}})
    compiler = Compiler()
    state = execute(tmp_path, provider=Provider([error, bundle()]), compiler=compiler, token_budget=20)
    assert state['status'] == 'budget_exhausted' and state['tokens_used'] == 30
    assert compiler.calls == 0 and state['winner'] is None


def test_budget_exhaustion_retains_an_earlier_qualified_search_candidate(tmp_path):
    state = execute(tmp_path, 'candidate_search', 'best_of_n', token_budget=45)
    assert state['status'] == 'completed' and state['winner'] == 1
    assert state['terminal_reason'] == 'budget_exhausted_after_success'
    assert state['tokens_used'] == 45
    assert state['attempts'][1]['status'] == 'budget_exhausted'


def test_private_evaluation_errors_never_trigger_repairs(tmp_path):
    class BrokenSelection(Compiler):
        def compile_cases(self, *args):
            raise ValueError('private-selection-only-diagnostic')

    provider = Provider()
    state = execute(tmp_path, 'candidate_search', 'best_of_n', provider=provider, compiler=BrokenSelection(), population=2)
    assert state['status'] == 'blocked' and state['winner'] is None
    assert len(provider.prompts) == 4
    assert all('private-selection-only-diagnostic' not in p for p in provider.prompts)


def test_interrupted_call_marks_usage_unknown_and_does_not_resume(tmp_path):
    class Interrupted(Provider):
        def generate(self, *args):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, provider=Interrupted())
    state = json.loads((tmp_path / 'run/report.json').read_text())
    assert state['status'] == 'interrupted' and not state['usage_complete']
    assert state['calls'][0]['status'] == 'in_flight'


def test_unknown_usage_or_unavailable_compiler_blocks_without_retry(tmp_path):
    error = ProviderError('HTTP 429', {'usage_incomplete': True})
    provider = Provider([error])
    state = execute(tmp_path, provider=provider)
    assert state['status'] == 'blocked' and not state['usage_complete']
    assert len(provider.prompts) == 1

    class Unavailable(Compiler):
        def identity(self, target):
            raise OSError('no Docker server')

    provider = Provider()
    state = execute(tmp_path / 'unavailable', provider=provider, compiler=Unavailable())
    assert state['status'] == 'blocked' and provider.prompts == []


@pytest.mark.parametrize('corruption', ['fingerprint', 'source', 'manifest', 'oracle', 'host'])
def test_evidence_binding_and_independent_correctness_are_required(tmp_path, corruption):
    class Corrupt(Compiler):
        def compile(self, *args):
            result = super().compile(*args)
            if corruption == 'fingerprint':
                result.compiler_fingerprint = 'changed'
            elif corruption == 'source':
                result.validation.source_sha256 = 'other'
            elif corruption == 'manifest':
                result.validation.manifest_sha256 = 'other'
            else:
                result.validation.stages['oracle_validation' if corruption == 'oracle' else 'host_execution'] = StageResult(status='failed', correct=False)
            return result

    state = execute(tmp_path, compiler=Corrupt(), max_repairs=0)
    assert state['winner'] is None and state['status'] in ('failed', 'blocked')


def test_api_constraint_rejection_feeds_repair_without_compiling_bad_bundle(tmp_path):
    provider = Provider([bundle(bad=True), bundle()])
    compiler = Compiler()
    state = execute(tmp_path, 'api_constraints', 'checked', provider, compiler)
    assert state['status'] == 'completed' and compiler.calls == 1
    assert 'adf.h' in provider.prompts[-1]
    for code in ('from openvino.opset13 import hallucination', 'import openvino as ov\nx=ov.hallucination()',
                 'from openvino import opset13 as ops\nx=ops.hallucination()', 'x=getattr(ov, "anything")'):
        value = CodeBundle(backend=INTEL_NPU_4000.backend, files=[GeneratedFile(relative_path='model.py', content=code)])
        assert api_constraints.check(value, 'f', 'f')
    with pytest.raises(ValueError):
        api_constraints.check(bundle(), 'old', 'new')


def test_feature_retrieval_hard_filters_toolchain_and_prioritizes_observed_fix():
    key = feature_retrieval.features(manifest(), AMD_XDNA2_NPU2, 'f')
    def entry(name, **changes):
        return {'id': name, 'features': {**key, **changes}, 'bundle': bundle().model_dump(mode='json'), 'fixes': {}}
    exact, fixed = entry('exact'), entry('fixed', shapes=[[128], [128], [128]])
    from npu_agent.database import normalize_error

    fixed['fixes'][normalize_error('unknown vector')] = {'description': 'repaired'}
    records = [entry('wrong', fingerprint='Chess'), exact, fixed, entry('op', operation='softmax')]
    result = feature_retrieval.retrieve(records, key, diagnostic='unknown vector')
    assert [r['id'] for r in result] == ['fixed', 'exact']
    assert feature_retrieval.retrieve(records, key, mode='off') == []
    assert 'wrong' not in [r['id'] for r in feature_retrieval.retrieve(records, key, mode='text', limit=99)]


def test_selection_rejects_feedback_only_pass_and_never_leaks_inputs(tmp_path):
    provider = Provider()
    compiler = Compiler(selection_failures=1)
    state = execute(tmp_path, 'candidate_search', 'best_of_n', provider, compiler, population=4)
    assert state['status'] == 'completed' and state['winner'] == 2
    assert compiler.selection_calls == 4
    assert len(provider.prompts) == 8
    assert all('selection-0' not in p and '70123' not in p for p in provider.prompts)
    assert state['attempts'][1]['selection']['metamorphic']['status'] == 'passed'
    assert metrics(state)['first_attempt_pass'] is False
    assert metrics(state)['selection_passes'] == 3
    with pytest.raises(ValueError, match='overlap'):
        candidate_search.selection_cases(manifest(), 0)


def test_profile_does_not_promote_static_or_host_metrics():
    result = CompileResult(success=True, exit_code=0, compiler_fingerprint='f', cpu_latency_p50_ms=1,
                           static_metrics={'vectorization_percent': 100})
    assert feedback_loop.profile(result)['status'] == 'unavailable'
    result.validation = ValidationResult(candidate_id='c', target_id='amd_xdna2_npu2', hardware_latency_p50_ms=2,
        stages={'target_execution': StageResult(status='passed', correct=True, engine='physical_npu'),
                'performance': StageResult(status='passed', engine='physical_npu', details={
                    'provenance': 'fixture-only profiler', 'scope': 'execution', 'vectorization_percent': 37,
                    'dma_stall_cycles': 5, 'objectfifo_stall_cycles': -1})})
    profile = feedback_loop.profile(result)
    assert profile['latency_ms'] == 2 and profile['vectorization_percent'] == 37
    assert profile['dma_stall_cycles'] == 5 and profile['objectfifo_stall_cycles'] is None


def test_profile_optimization_retains_the_best_correct_candidate(tmp_path):
    class Profiled(Compiler):
        def compile(self, *args):
            result = super().compile(*args)
            result.validation.hardware_latency_p50_ms = [3, 1, 2][self.calls - 1]
            result.validation.stages['target_execution'] = StageResult(status='passed', correct=True, engine='physical_npu')
            result.validation.stages['performance'] = StageResult(status='passed', engine='physical_npu',
                details={'provenance': 'test-fixture-only', 'scope': 'execution'})
            return result

    provider = Provider()
    state = execute(tmp_path, provider=provider, compiler=Profiled(), optimize_profile=True, max_repairs=2)
    assert state['winner'] == 2 and len(state['attempts']) == 3
    assert '"latency_ms": 1' in provider.prompts[-1]
    attempts = [{'number': i, 'profile': {'status': 'measured', 'latency_ms': None,
                 'scope': 'execution', 'provenance': 'fixture', 'vectorization_percent': score}}
                for i, score in enumerate((10, 40, 20))]
    assert feedback_loop.best_profile(attempts)['number'] == 1


@pytest.mark.parametrize('missing', [False, True])
def test_private_outputs_must_match_the_executable_ir(tmp_path, missing):
    class BadOutputs(Compiler):
        def compile_cases(self, *args):
            result = super().compile_cases(*args)
            if missing:
                result.artifacts.clear()
            else:
                for path in result.artifacts.values():
                    np.savez(path, output=np.zeros(64, np.float32))
            return result

    state = execute(tmp_path, 'candidate_search', 'single', compiler=BadOutputs())
    assert state['winner'] is None
    assert state['status'] == ('blocked' if missing else 'failed')


def test_preserves_existing_output_and_synthetic_runs_cannot_be_skills(tmp_path):
    execute(tmp_path)
    path = tmp_path / 'run/report.json'
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        execute(tmp_path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='nonsynthetic'):
        verified_skills.admit(tmp_path / 'run', tmp_path / 'library')


def test_skill_admission_integrity_and_feature_reuse(tmp_path):
    # A trusted evaluator fixture exercises admission; it is not research evidence.
    provider = Provider()
    provider.name = 'admission-unit-test'
    execute(tmp_path, provider=provider, compiler=Compiler(failures=1))
    path = verified_skills.admit(tmp_path / 'run', tmp_path / 'library')
    entries = verified_skills.load(tmp_path / 'library')
    assert len(entries) == 1 and entries[0]['fixes']
    assert verified_skills.admit(tmp_path / 'run', tmp_path / 'library') == path
    reuse_provider = Provider()
    state = run(SOURCE, MANIFEST, AMD_XDNA2_NPU2, reuse_provider, Compiler(),
        RunConfig(direction='feature_retrieval', arm='features', model='test'), tmp_path / 'reuse', library=tmp_path / 'library')
    assert state['attempts'][0]['retrieved'] == [entries[0]['id']]
    assert 'Verified examples' in reuse_provider.prompts[0]
    content = json.loads(path.read_text())
    content['bundle']['files'][0]['content'] = 'changed'
    write_json(path, content)
    with pytest.raises(ValueError, match='hash'):
        verified_skills.load(tmp_path / 'library')
    (tmp_path / 'run/attempt-2/candidate/kernel.cc').write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        verified_skills.admit(tmp_path / 'run', tmp_path / 'other')


@pytest.mark.parametrize('kind', ['elementwise_add', 'reduction_sum', 'gemv'])
def test_native_microkernels_match_independent_oracle_on_adversarial_cases(tmp_path, kind):
    if not shutil.which('g++'):
        pytest.skip('g++ unavailable')
    data = manifest().model_dump(mode='json')
    data['scalars'] = []
    if kind == 'elementwise_add':
        for tensor in data['tensors']:
            tensor['shape'] = [17]
        body = ('aie::store_v(out, aie::add(aie::load_v<16>(x0), aie::load_v<16>(x1)));\n'
                'out[16] = x0[16] + x1[16];')
    elif kind == 'reduction_sum':
        data['tensors'] = [data['tensors'][0], data['tensors'][2]]
        data['tensors'][0]['shape'] = [16]
        data['tensors'][1]['shape'] = [1]
        data['oracle'] = {'kind': 'builtin', 'operation': 'reduce_sum', 'parameters': {'axis': 0, 'keepdims': True}}
        body = 'double sum=0; for(int i=0;i<16;++i) sum += x0[i]; out[0] = sum;'
    else:
        for tensor, shape in zip(data['tensors'], ([4, 8], [8, 1], [4, 1])):
            tensor['shape'] = shape
        data['oracle'] = {'kind': 'builtin', 'operation': 'matmul', 'parameters': {'output_dtype': 'float32'}}
        body = 'for(int i=0;i<4;++i) { float sum=0; for(int j=0;j<8;++j) sum += x0[i*8+j]*x1[j]; out[i]=sum; }'
    contract = KernelManifest.model_validate(data)
    value, plan = template_generation.instantiate(contract, AMD_XDNA2_NPU2, template_generation.MicroKernel(body=body))
    assert plan['template'] == kind
    materialize_bundle(value, tmp_path)
    abi = {'compute': [{'kind': 'buffer', 'dtype': t.dtype, 'shape': list(t.shape)} for t in contract.tensors]}
    binary = build_host(tmp_path / 'kernel.cc', abi, tmp_path, ROOT / 'playground/tools/host_compat')
    kernel = HostKernel(binary, abi, tmp_path / 'sanitizer.log')
    try:
        for case in generate_cases(contract):
            actual = np.full(contract.tensors[-1].shape, np.nan, np.float32)
            args = [case.inputs[t.name].copy() for t in contract.tensors[:-1]] + [actual]
            kernel('compute', args, [True] * (len(args) - 1) + [False])
            np.testing.assert_allclose(actual, execute_oracle(contract, case.inputs)['output'],
                                       rtol=contract.tolerance.rtol, atol=contract.tolerance.atol)
    finally:
        kernel.close()


def test_template_guards_reject_illegal_contracts_and_escaping_bodies():
    for body in ('} void wrong() {', '#include <adf.h>\n', '{'):
        with pytest.raises(ValueError):
            template_generation.instantiate(manifest(), AMD_XDNA2_NPU2, template_generation.MicroKernel(body=body))
    with pytest.raises(ValueError, match='AMD'):
        template_generation.schedule(manifest(), INTEL_NPU_4000)


def test_cli_list_and_saved_comparison_report(tmp_path, capsys):
    assert main(['list']) == 0
    assert set(json.loads(capsys.readouterr().out)) == set(ARMS)
    execute(tmp_path)
    output = tmp_path / 'comparison.json'
    assert main(['report', str(tmp_path / 'run'), '--output', str(output)]) == 0
    assert json.loads(output.read_text())['rows'][0]['synthetic'] is True
