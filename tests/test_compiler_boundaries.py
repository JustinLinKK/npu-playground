import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from npu_agent.compilers import DockerCompiler, _bounded
from npu_agent.config import Settings, INTEL_NPU_4000
from npu_agent.models import StageResult, KernelManifest
from npu_agent.testcases import generate_cases


def test_timeout_bytes_are_decoded():
    assert _bounded(b'bad\xffoutput', 6) == 'output'


def test_malformed_actual_archive_is_a_stage_failure(tmp_path):
    manifest = KernelManifest.model_validate_json(Path('tests/fixtures/backends/intel_npu_4000/add/manifest.json').read_text())
    case = generate_cases(manifest)[0]
    (tmp_path / f'{case.id}.npz').write_bytes(b'truncated')
    compiler = DockerCompiler(Settings())
    stage = compiler._compare(StageResult(status='passed'), tmp_path, manifest, [case], 'openvino-cpu', 'openvino_graph')
    assert stage.status == 'failed' and stage.error_category == 'malformed_report'


def test_shared_container_isolation(tmp_path):
    compiler = DockerCompiler(Settings(repository_path=Path.cwd()))
    command = compiler._docker_prefix(tmp_path, tmp_path, INTEL_NPU_4000)
    assert '--privileged' not in command
    assert '--device' not in command
    assert command[command.index('--cpus') + 1] == '4'
    assert command[command.index('--network') + 1] == 'none'
    assert command[command.index('--user') + 1] != '0:0'
    assert '/work:rw,size=2g,exec,mode=1777' in command


def test_missing_candidate_source_produces_saved_failure(tmp_path):
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    manifest = KernelManifest.model_validate_json(Path('tests/fixtures/backends/intel_npu_4000/add/manifest.json').read_text())
    result = DockerCompiler(Settings()).compile(candidate, tmp_path / 'output', manifest, INTEL_NPU_4000)
    assert not result.success
    assert result.validation.stages['target_compile'].error_category == 'source_compile'
    assert json.loads(Path(result.artifacts['validation.json']).read_text())['policy_met'] is False


def test_parallel_container_limits_and_graceful_stop_boundary(tmp_path):
    import subprocess

    compiler = DockerCompiler(Settings(compiler_cpus=6, compiler_memory='8g', max_compiler_jobs=4,
                                       experiment_id='parallel'))
    with patch('npu_agent.compilers.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as invoke:
        stage = compiler._run(tmp_path, tmp_path / 'output', INTEL_NPU_4000, 'intel_compile_graph.py', {})
    assert stage.status == 'passed'
    command = invoke.call_args.args[0]
    assert command[command.index('--cpus') + 1] == '6'
    assert command[command.index('--memory') + 1] == '8g'
    assert invoke.call_args.kwargs['start_new_session'] is True
