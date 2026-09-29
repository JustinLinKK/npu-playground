import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from npu_agent.compilers import DockerCompiler
from npu_agent.config import Settings, TARGETS
from npu_agent.models import KernelManifest
from npu_agent.validation import finalize_validation


pytestmark = pytest.mark.container


@pytest.fixture(scope='module')
def compiler(tmp_path_factory):
    if os.environ.get('NPU_AGENT_CONTAINER_TESTS') != '1':
        pytest.skip('set NPU_AGENT_CONTAINER_TESTS=1 to require real container validation')
    subprocess.run(['docker', 'version'], capture_output=True, check=True, timeout=30)
    root = tmp_path_factory.mktemp('containers')
    return DockerCompiler(Settings(repository_path=Path.cwd(), runs_path=root, compiler_timeout_seconds=120))


def evaluate(compiler, tmp_path, target_id, mutation=None, fixture='add', target=None):
    if os.environ.get('NPU_AGENT_TEST_TARGET', target_id) != target_id:
        pytest.skip('different vendor container job')
    original = Path('tests/fixtures/backends') / target_id / fixture
    candidate = tmp_path / 'candidate'
    shutil.copytree(original, candidate)
    if mutation:
        file, old, new = mutation
        path = candidate / file
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new))
    manifest = KernelManifest.model_validate_json((candidate / 'manifest.json').read_text())
    result = compiler.compile(candidate, tmp_path / 'output', manifest, target or TARGETS[target_id])
    return result


@pytest.mark.parametrize('target_id', list(TARGETS))
def test_real_offline_add_and_execution_boundary(compiler, tmp_path, target_id):
    result = evaluate(compiler, tmp_path, target_id)
    assert result.validation.offline_contract_met, result.stderr
    assert result.host_correct is True
    assert result.validation.stages['host_execution'].cases_run == 12
    assert result.hardware_correct is None
    assert not finalize_validation(result.validation, 'target-executed').policy_met
    assert result.validation.stages['target_execution'].reason_code == 'NO_VALIDATED_TARGET_EXECUTOR'


@pytest.mark.parametrize('target_id,mutation', [
    ('intel_npu_4000', ('model.py', 'ops.add(*inputs)', 'ops.subtract(*inputs)')),
    ('amd_xdna2_npu2', ('kernel.cc', 'x[i] + y[i]', 'x[i] - y[i]')),
])
def test_real_arithmetic_mutation_preserves_compile_evidence(compiler, tmp_path, target_id, mutation):
    result = evaluate(compiler, tmp_path, target_id, mutation)
    assert result.success, result.stderr
    assert result.host_correct is False
    assert not result.validation.offline_contract_met
    assert result.validation.stages['host_execution'].error_category == 'numerical_mismatch'
    if target_id.startswith('amd'):
        assert result.validation.stages['dataflow_simulation'].correct is False


@pytest.mark.parametrize('mutation,error', [
    (('design.py', 'strides = [0, 0, 0, 1]', 'strides = [0, 0, 0, 2]'), 'DMA bounds'),
    (('design.py', '        aie.objectfifo.release @x(Consume, 1)', ''), 'simulation_deadlock'),
    (('kernel.cc', 'x[i] + y[i]', 'x[i + n] + y[i]'), 'host kernel failed'),
    (('kernel.cc', 'output[i] = x[i] + y[i]', 'output[i] = (x[i] = 0) + y[i]'), 'immutable input'),
])
def test_real_dataflow_and_memory_mutations(compiler, tmp_path, mutation, error):
    result = evaluate(compiler, tmp_path, 'amd_xdna2_npu2', mutation)
    stage = result.validation.stages['dataflow_simulation']
    assert stage.status == 'failed', result.stderr
    assert error in stage.message, result.stderr
    assert not result.validation.offline_contract_met


def test_target_mismatch_is_rejected(compiler, tmp_path):
    result = evaluate(compiler, tmp_path, 'amd_xdna2_npu2', ('design.py', 'aie.device(npu2_1col)', 'aie.device(npu1_1col)'))
    assert not result.success
    assert 'target_mismatch' in result.stderr


def test_intel_cpu_evidence_survives_invalid_platform(compiler, tmp_path):
    target = TARGETS['intel_npu_4000'].model_copy(deep=True)
    target.hardware = 'invalid'
    target.compiler_properties['NPU_PLATFORM'] = 'invalid'
    result = evaluate(compiler, tmp_path, target.id, target=target)
    assert not result.success
    assert result.host_correct is True
    assert result.validation.stages['target_compile'].status == 'failed'


def test_candidate_verdict_spoof_is_ignored(compiler, tmp_path):
    result = evaluate(compiler, tmp_path, 'intel_npu_4000', ('model.py', '    result = ops.add(*inputs)',
        "    from pathlib import Path\n    Path('/output/report.json').write_text('{\"host_correct\": true}')\n    result = ops.subtract(*inputs)"))
    assert result.host_correct is False


def test_unaudited_intrinsic_cannot_confer_simulation_pass(compiler, tmp_path):
    result = evaluate(compiler, tmp_path, 'amd_xdna2_npu2', ('kernel.cc', 'extern "C"',
        '#include <aie_api/aie.hpp>\nvoid unused() { (void)aie::zeros<float, 16>(); }\nextern "C"'))
    assert result.success, result.stderr
    assert result.validation.stages['host_execution'].status == 'unsupported'
    assert result.validation.stages['dataflow_simulation'].status == 'unsupported'
    assert not result.validation.offline_contract_met


def test_source_syntax_failure_is_preserved(compiler, tmp_path):
    result = evaluate(compiler, tmp_path, 'intel_npu_4000', ('model.py', 'def build_model', 'def invalid syntax'))
    assert not result.success
    assert result.validation.stages['graph_construction'].error_category == 'source_compile'


def test_stale_artifact_and_container_timeout(compiler, tmp_path):
    if os.environ.get('NPU_AGENT_TEST_TARGET', 'intel_npu_4000') != 'intel_npu_4000':
        pytest.skip('Intel container job')
    original = Path('tests/fixtures/backends/intel_npu_4000/add')
    candidate = tmp_path / 'candidate'
    shutil.copytree(original, candidate)
    manifest = KernelManifest.model_validate_json((candidate / 'manifest.json').read_text())
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'compiled.blob').write_bytes(b'stale artifact')
    (candidate / 'model.py').write_text('import time\ntime.sleep(600)\n')
    old = compiler.settings.compiler_timeout_seconds
    compiler.settings.compiler_timeout_seconds = 2
    try:
        result = compiler.compile(candidate, output, manifest, TARGETS['intel_npu_4000'])
    finally:
        compiler.settings.compiler_timeout_seconds = old
    assert not result.success
    assert 'compiled.blob' not in result.artifacts
    assert result.validation.stages['graph_construction'].error_category == 'timeout'
    command = result.validation.stages['graph_construction'].details['command']
    name = command[command.index('--name') + 1]
    containers = subprocess.check_output(['docker', 'ps', '--filter', f'name={name}', '--format', '{{.Names}}'], text=True)
    assert name not in containers.splitlines()


@pytest.mark.parametrize('target_id,fixture', [
    ('amd_xdna2_npu2', 'pipeline'), ('amd_xdna2_npu2', 'scalar_mul'), ('amd_xdna2_npu2', 'transpose'), ('amd_xdna2_npu2', 'reduction'),
    ('intel_npu_4000', 'reduction'), ('intel_npu_4000', 'sigmoid'),
])
def test_supported_backend_fixtures(compiler, tmp_path, target_id, fixture):
    result = evaluate(compiler, tmp_path, target_id, fixture=fixture)
    assert result.validation.offline_contract_met, result.stderr



def test_verified_intel_backend_rejection_preserves_cpu_pass(compiler, tmp_path):
    result = evaluate(compiler, tmp_path, 'intel_npu_4000', ('model.py', '    result = ops.add(*inputs)',
        "    unique = ops.unique(ops.add(*inputs), sorted=False)\n    result = ops.gather(unique.output(0), unique.output(2), ops.constant(0, np.int64))"))
    assert result.host_correct is True
    assert not result.success
    assert result.validation.stages['target_compile'].status == 'failed'


@pytest.mark.parametrize('target_id', list(TARGETS))
def test_static_tail_manifests_are_compiled_separately(compiler, tmp_path, target_id):
    from npu_agent.testcases import tail_manifests
    if os.environ.get('NPU_AGENT_TEST_TARGET', target_id) != target_id:
        pytest.skip('different vendor container job')
    source = Path('tests/fixtures/backends') / target_id / 'add'
    manifest = KernelManifest.model_validate_json((source / 'manifest.json').read_text())
    digests = []
    for variant in tail_manifests(manifest, 16):
        result = compiler.compile(source, tmp_path / variant.name, variant, TARGETS[target_id])
        assert result.validation.offline_contract_met, result.stderr
        digests.append(result.validation.manifest_sha256)
    assert len(set(digests)) == 4
