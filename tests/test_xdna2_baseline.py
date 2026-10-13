import io
import json
from pathlib import Path
from threading import Event

import pytest

from npu_agent.config import TARGETS
from npu_agent.experiment import discover_kernels
from npu_agent.providers import ProviderError
from npu_agent.study_v4 import StudyConfig, run_case, stage_prompt
from npu_agent.study_v4_provider import ProviderConfig, StudyProvider
from npu_agent.xdna2_baseline import classify_attempt
from test_study_v4 import setup_case
from test_study_v4_provider import envelope


def baseline_config(**updates):
    values = dict(protocol='xdna2-baseline-v1', arms=['direct'], targets=['amd_xdna2_npu2'], repetitions=1)
    return StudyConfig(**(values | updates))


def test_baseline_protocol_is_separate_from_historical_design():
    assert baseline_config().max_cycles == 10
    with pytest.raises(ValueError, match='primary arm'):
        StudyConfig(arms=['direct'])
    for updates in ({'arms': ['direct', 'hinted_ir']}, {'targets': ['intel_npu_4000']}, {'repetitions': 2}):
        with pytest.raises(ValueError, match='XDNA2 baseline'):
            baseline_config(**updates)


def test_baseline_prompt_has_only_container_contract_and_no_ir_guidance(monkeypatch):
    source, _, manifest = discover_kernels(Path('examples/classic'))['cuda_vector_add']
    args = ('code', 'direct', source.read_text(), manifest, TARGETS['amd_xdna2_npu2'], None, '')
    historical = stage_prompt(*args)
    assert 'Container invocation and submission contract' not in historical

    def no_backend_guidance(*args, **kwargs):
        pytest.fail('baseline must not include the shared backend implementation guidance')

    monkeypatch.setattr('npu_agent.study_v4._backend_contract', no_backend_guidance)
    for source, _, manifest in discover_kernels(Path('examples')).values():
        prompt = stage_prompt('code', 'direct', source.read_text(), manifest, TARGETS['amd_xdna2_npu2'],
                              None, '', protocol='xdna2-baseline-v1')
        assert 'python design.py --dev npu2 --emit-mlir' in prompt
        assert 'must NOT be required' in prompt and 'revision 2, unguided' in prompt
        assert source.read_text() in prompt
        for forbidden in ('IRON', 'MLIR', 'KernelIR', 'IR_CONTRACT', 'intermediate representation',
                          'aie.', 'aiex.', 'scf.for', 'arith.constant', 'memref<', 'link_with',
                          'ObjectFifo', 'Runtime(', 'Worker(', 'DMA', 'FIFO', 'npu::', 'npu_numeric.h',
                          '"oracle"', '"provenance"'):
            assert forbidden not in prompt


def test_qwen_config_key_pinned_route_and_usage(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    (tmp_path / 'config.yaml').write_text('openrouter:\n  api_key: config-only-test-key\n')
    config = ProviderConfig(model='qwen/qwen3.8-27b', endpoint='deepinfra/bf16', upstream_provider='DeepInfra', max_tokens=32768)

    def post(request, timeout):
        assert request.get_header('Authorization') == 'Bearer config-only-test-key'
        body = json.loads(request.data)
        assert body['provider'] == {'only': ['deepinfra/bf16'], 'allow_fallbacks': False, 'require_parameters': True}
        assert body['reasoning']['enabled'] and body['max_tokens'] == 32768
        return io.BytesIO(json.dumps(envelope(model=config.model, provider='DeepInfra')).encode())

    monkeypatch.setattr('urllib.request.urlopen', post)
    from npu_agent.study_v4 import Hint

    result = StudyProvider(config, tmp_path).generate('transport check', Hint, config.model)
    assert 'config-only-test-key' not in json.dumps(result.metadata)


@pytest.mark.parametrize('socket_timeout', [False, True])
def test_qwen_heartbeats_cannot_extend_deadline_and_header_id_survives(tmp_path, monkeypatch, socket_timeout):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only')
    config = ProviderConfig(model='qwen/qwen3.8-27b', endpoint='deepinfra/bf16', upstream_provider='DeepInfra', timeout_seconds=1)
    clock = [0.]

    class Heartbeats(io.BytesIO):
        headers = {'X-Generation-Id': 'generation-stalled'}

        def read1(self, size):
            clock[0] += .6
            if socket_timeout and clock[0] >= config.timeout_seconds:
                raise TimeoutError('sensitive transport detail')
            return b' '

    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: Heartbeats())
    monkeypatch.setattr('npu_agent.study_v4_provider.time.perf_counter', lambda: clock[0])
    from npu_agent.study_v4 import Hint

    with pytest.raises(ProviderError, match='transport failed') as exc:
        StudyProvider(config, tmp_path).generate('probe', Hint, config.model)
    assert exc.value.metadata['generation_id'] == 'generation-stalled'
    assert exc.value.metadata['usage_incomplete']
    assert exc.value.metadata['timeout_kind'] == 'total_response_deadline'
    assert 'sensitive' not in str(exc.value)
    assert clock[0] == 1.2


def test_baseline_response_can_finish_after_old_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-only')
    config = ProviderConfig.model_validate(json.loads(Path('configs/xdna2-baseline-v1.json').read_text())['provider'])
    clock = [0.]

    class SlowResponse(io.BytesIO):
        def read1(self, size):
            clock[0] = 1000.
            return super().read1(size)

    def post(request, timeout):
        assert timeout == config.timeout_seconds == 1800
        return SlowResponse(json.dumps(envelope(model=config.model, provider=config.upstream_provider)).encode())

    monkeypatch.setattr('urllib.request.urlopen', post)
    monkeypatch.setattr('npu_agent.study_v4_provider.time.perf_counter', lambda: clock[0])
    from npu_agent.study_v4 import Hint

    response = StudyProvider(config, tmp_path).generate('probe', Hint, config.model)
    assert response.metadata['schema_valid'] and not response.metadata['usage_incomplete']
    assert response.metadata['duration_seconds'] == 1000.


def test_ten_rejected_attempts_are_charged_and_not_replayed(tmp_path):
    request, settings, _, case = setup_case(tmp_path)
    request = request.model_copy(update={'targets': [TARGETS['amd_xdna2_npu2']]})
    case['target'] = 'amd_xdna2_npu2'

    class Rejected:
        calls = 0

        def generate(self, *args):
            self.calls += 1
            raise ProviderError('invalid JSON', {'failure_origin': 'model_output',
                                'usage': {'input_tokens': 10, 'output_tokens': 20}}, {})

    provider = Rejected()
    config = baseline_config()
    state = run_case(request, settings, config, case, provider, None, Event())
    assert state['status'] == 'failed' and len(state['cycles']) == provider.calls == 10
    assert all(c['calls']['code']['metadata']['usage']['output_tokens'] == 20 for c in state['cycles'])
    assert run_case(request, settings, config, case, provider, None, Event()) == state
    assert provider.calls == 10
    with pytest.raises(ValueError, match='inputs changed'):
        run_case(request, settings, baseline_config(max_cycles=9), case, provider, None, Event())


@pytest.mark.parametrize('stages,category,kernel', [
    ({'design_emission': {'status': 'failed', 'message': 'error: required --xclbin-path'}}, 'emitter_api_contract', False),
    ({'target_compile': {'status': 'failed', 'message': 'PLEASE submit a bug report: unable to legalize'}}, 'toolchain_crash', False),
    ({'target_compile': {'status': 'failed', 'message': 'clang++: error: clang frontend command failed with exit code 134 (use -v to see invocation)'}}, 'toolchain_crash', False),
    ({'target_compile': {'status': 'failed', 'message': 'clang++: error: clang frontend command failed with exit code 70 (use -v to see invocation)'}}, 'toolchain_crash', False),
    ({'target_compile': {'status': 'passed'}, 'host_execution': {'status': 'unsupported', 'message': 'unsupported intrinsic mul'}}, 'simulator_coverage', False),
    ({'target_compile': {'status': 'failed', 'message': 'invalid scf.for bounds'}, 'host_execution': {'status': 'unsupported'}}, 'target_source_or_verifier', True),
    ({'host_execution': {'status': 'failed', 'error_category': 'numerical_mismatch'}}, 'numerical_mismatch', True),
    ({'host_execution': {'status': 'failed', 'message': 'mysterious failure'}}, 'unclassified', False),
    ({'host_execution': {'status': 'failed', 'message': 'AddressSanitizer: segmentation fault'}}, 'memory_abi_or_dataflow', True),
    ({'host_execution': {'status': 'failed', 'message': 'SimulationError: DMA token count or element size mismatch'}}, 'memory_abi_or_dataflow', True),
])
def test_noise_filter_preserves_real_failures(stages, category, kernel):
    cycle = {'status': 'failed', 'compile_result': {'validation': {'stages': stages}}}
    row = classify_attempt(cycle)
    assert row['category'] == category and row['kernel_failure'] == kernel


def test_generated_but_unevaluated_bundle_is_not_a_model_failure():
    cycle = {'status': 'running', 'bundle': {'files': []}, 'calls': {'code': {'status': 'completed'}}}
    row = classify_attempt(cycle)
    assert row == {'category': 'not_evaluated', 'kernel_failure': False, 'noise': True, 'stage': 'validation'}
    cycle['calls']['code']['status'] = 'blocked'
    assert classify_attempt(cycle)['category'] == 'provider_or_interruption'
    cycle.pop('bundle')
    cycle['calls']['code']['status'] = 'rejected'
    cycle['status'] = 'failed'
    assert classify_attempt(cycle)['category'] == 'model_output_or_unclassified'


def test_all_thirty_manifests_have_finite_independent_references():
    import numpy as np

    from npu_agent.oracles import execute_oracle, validate_goldens
    from npu_agent.testcases import generate_cases

    corpus = discover_kernels(Path('examples/kernelbench'))
    assert len(corpus) == 30 and len(discover_kernels(Path('examples'))) == 40
    for _, _, manifest in corpus.values():
        validate_goldens(manifest.oracle.operation)
        out = next(t for t in manifest.tensors if t.direction == 'output')
        for case in generate_cases(manifest):
            value = execute_oracle(manifest, case.inputs)[out.name]
            assert value.shape == out.shape and value.dtype == np.float32 and np.isfinite(value).all()


@pytest.mark.parametrize('stage,status,message,case_status,smoke_passed,expected', [
    ('design_emission', 'failed', 'TypeError: missing required argument', 'failed', True, 'passed'),
    ('target_compile', 'failed', 'invalid scf.for bounds', 'failed', True, 'passed'),
    ('host_execution', 'unsupported', 'unsupported intrinsic', 'failed', True, 'passed'),
    ('target_compile', 'blocked', 'container unavailable', 'failed', True, 'failed'),
    ('oracle_validation', 'failed', 'invalid reference', 'failed', True, 'failed'),
    ('design_emission', 'failed', 'TypeError: missing required argument', 'blocked', True, 'failed'),
    ('design_emission', 'failed', 'TypeError: missing required argument', 'failed', False, 'failed'),
])
def test_unguided_calibration_gates_evaluator_not_model_ability(tmp_path, stage, status, message,
                                                            case_status, smoke_passed, expected):
    from research.run_xdna2_baseline import calibration_gate

    cases = [{'id': str(i), 'kernel': str(i)} for i in range(4)]
    (tmp_path / 'campaign.json').write_text(json.dumps({'cases': cases, 'identity': {'source_hashes': {}}}))
    (tmp_path / 'preflight.json').write_text(json.dumps({'status': 'completed', 'compiler_smokes': {
        'amd_xdna2_npu2': {'validation': {'offline_contract_met': smoke_passed}}}}))
    for case in cases:
        folder = tmp_path / 'cases' / case['id']
        folder.mkdir(parents=True)
        (folder / 'report.json').write_text(json.dumps({'status': case_status, 'cycles': [{
            'status': 'failed', 'compile_result': {'validation': {'stages': {
                stage: {'status': status, 'message': message}}}}}]}))
    gate = calibration_gate(tmp_path)
    assert gate['status'] == expected
    assert gate['evaluator_ready'] == smoke_passed and gate['preflight_sha256']
    (tmp_path / 'preflight.json').unlink()
    assert calibration_gate(tmp_path)['status'] == 'failed'


def test_calibration_reference_shapes(tmp_path):
    from research.run_xdna2_baseline import calibration_inputs
    from npu_agent.oracles import execute_oracle
    from npu_agent.testcases import generate_cases

    corpus = discover_kernels(calibration_inputs(tmp_path))
    assert len(corpus) == 4
    for _, _, manifest in corpus.values():
        for case in generate_cases(manifest):
            execute_oracle(manifest, case.inputs)


def test_blocked_campaign_keeps_all_forty_unrun_tasks(tmp_path):
    from npu_agent.study_v4 import case_schedule
    from npu_agent.xdna2_baseline import build_report

    names = list(discover_kernels(Path('examples')))
    config = baseline_config()
    campaign = {'config': config.model_dump(), 'status': 'blocked', 'blocker': 'Provider calibration unavailable',
                'cases': case_schedule(config, names, {name: name for name in names})}
    (tmp_path / 'campaign.json').write_text(json.dumps(campaign))
    summary = build_report(tmp_path)
    assert summary['planned'] == 40 and summary['attempted_tasks'] == 0
    assert summary['statuses'] == {'not_run': 40} and summary['kernel_failure_attempts'] == {}
    assert 'NOT RUN' in (tmp_path / 'results/REPORT.md').read_text()


@pytest.mark.parametrize('external_budget', [False, True])
def test_baseline_launcher_selects_budget_mode_and_config_key(tmp_path, monkeypatch, external_budget):
    from types import SimpleNamespace

    from research import run_xdna2_baseline as launcher

    commands = []
    monkeypatch.setenv('OPENROUTER_API_KEY', 'unused-environment-test-key')

    def run(command, **kwargs):
        assert 'OPENROUTER_API_KEY' not in kwargs['env']
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, 'run', run)
    monkeypatch.setattr(launcher, 'calibration_gate', lambda *a: {'status': 'passed'})
    monkeypatch.setattr(launcher.sys, 'argv', ['baseline', 'calibrate', '--root', str(tmp_path)] +
                        (['--budget-on-openrouter'] if external_budget else []))
    assert launcher.main() == 0
    assert len(commands) == (1 if external_budget else 2)
    assert ('--budget-on-openrouter' in commands[-1]) == external_budget
    assert ('--ledger' in commands[-1]) != external_budget
    assert commands[-1][-4:] == ['--workers', '4', '--compiler-jobs', '2']
    if not external_budget:
        assert '--limit-usd' in commands[0] and '50' in commands[0]
    config = json.loads((tmp_path / 'calibration-v1.config.json').read_text())
    assert config['max_cycles'] == 2 and config['token_budget'] == 200000


def test_openai_baseline_launcher_keeps_key_in_yaml_and_uses_direct_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from research import run_xdna2_baseline as launcher

    commands = []
    monkeypatch.setenv('OPENAI_API_KEY', 'unused-environment-test-key')

    def run(command, **kwargs):
        assert 'OPENAI_API_KEY' not in kwargs['env'] and 'OPENROUTER_API_KEY' not in kwargs['env']
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, 'run', run)
    monkeypatch.setattr(launcher, 'calibration_gate', lambda *a: {'status': 'passed'})
    monkeypatch.setattr(launcher.sys, 'argv', ['baseline', 'calibrate', '--root', str(tmp_path),
                        '--config', 'configs/xdna2-baseline-openai.json', '--budget-on-openai'])
    assert launcher.main() == 0 and len(commands) == 1
    assert '--budget-on-openai' in commands[0] and '--ledger' not in commands[0]
    config = json.loads((tmp_path / 'calibration-v1.config.json').read_text())
    assert config['provider']['model'] == 'gpt-6-astra' and config['provider']['upstream_provider'] == 'OpenAI'
    assert config['max_cycles'] == 2 and config['token_budget'] == 200000


def test_baseline_evaluation_cannot_switch_provider_after_calibration(tmp_path, monkeypatch):
    from research import run_xdna2_baseline as launcher

    calibration = tmp_path / 'calibration-v1'
    calibration.mkdir()
    (calibration / 'campaign.json').write_text(json.dumps({'config': baseline_config().model_dump()}))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **kw: pytest.fail('must not send'))
    monkeypatch.setattr(launcher.sys, 'argv', ['baseline', 'evaluate', '--root', str(tmp_path),
                        '--config', 'configs/xdna2-baseline-openai.json', '--budget-on-openai', '--source-report', 'unused.json'])
    with pytest.raises(ValueError, match='provider configuration changed after calibration'):
        launcher.main()


@pytest.mark.container
@pytest.mark.parametrize('number', [3, 24, 36, 41, 48, 91])
def test_kernelbench_scalar_container_probe(tmp_path, number):
    import os
    import numpy as np

    from npu_agent.config import Settings
    from npu_agent.study_v4 import StudyCompiler

    if os.environ.get('NPU_AGENT_CONTAINER_TESTS') != '1':
        pytest.skip('set NPU_AGENT_CONTAINER_TESTS=1 for real XDNA2 probes')
    source, manifest_path, manifest = next(v for k, v in discover_kernels(Path('examples/kernelbench')).items()
                                          if k.startswith(f'kb_{number:03d}_'))
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    text = source.read_text().replace('extern "C" __global__ void kernel', 'extern "C" void compute')
    count = int(np.prod(next(t for t in manifest.tensors if t.direction == 'output').shape))
    text = text.replace('int i=blockIdx.x*blockDim.x+threadIdx.x;', f'for(int i=0;i<{count};++i)')
    for cuda, portable in [('fmaxf', 'maximum'), ('fminf', 'minimum'), ('expf', 'exp'), ('sqrtf', 'sqrt'), ('logf', 'log')]:
        text = text.replace(cuda + '(', 'npu::' + portable + '(')
    (candidate / 'kernel.cc').write_text('#include <npu_numeric.h>\n' + text)
    (candidate / 'design.py').write_text(Path('tests/fixtures/backends/amd_single_worker.py.template').read_text())
    compiler = StudyCompiler(Settings(repository_path=Path.cwd(), runs_path=tmp_path, validation_policy='offline-validated',
                                      compiler_cpus=2, compiler_memory='4g', compiler_timeout_seconds=600))
    result = compiler.compile(candidate, tmp_path / 'validation', manifest, TARGETS['amd_xdna2_npu2'])
    assert result.validation.offline_contract_met, result.stderr
    assert result.validation.stages['host_execution'].cases_run == 12
