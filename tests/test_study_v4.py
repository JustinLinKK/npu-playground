from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from threading import Event

import pytest

from npu_agent.config import Settings, TARGETS
from npu_agent.experiment import discover_kernels
from npu_agent.models import StageResult, TranslationRequest
from npu_agent.providers import ProviderError, ProviderResponse
from npu_agent.study_v4 import (Hint, StudyCompiler, StudyConfig, case_schedule, repair_feedback,
                                run_case, stage_prompt)
from npu_agent.study_v4_reporting import build_report, case_metrics, paired_comparison
from test_experiment import CorpusCompiler, CorpusProvider


class Provider(CorpusProvider):
    def __init__(self):
        super().__init__("deepseek/deepseek-v3.2")
        self.prompts = []

    @staticmethod
    def _manifest(prompt):
        value, _ = json.JSONDecoder().raw_decode(prompt.split("Interface:\n", 1)[1])
        return discover_kernels(Path("examples/classic"))[value["name"]][2]

    def generate(self, prompt, response_model, model=None):
        self.prompts.append(prompt)
        if response_model is Hint:
            self.calls.append("Hint")
            value = Hint(intermediate="Add x and y elementwise into output.")
            return ProviderResponse(value=value, raw=value.model_dump(), metadata={"usage": {"input_tokens": 10, "output_tokens": 5}})
        return super().generate(prompt, response_model, model)


class RepairCompiler(CorpusCompiler):
    def compile(self, *args):
        result = super().compile(*args)
        if self.calls == 1:
            result.validation.stages["host_execution"] = StageResult(status="failed", correct=False, message="wrong result")
        return result


def setup_case(tmp_path, arm="direct", **overrides):
    config = StudyConfig(**overrides)
    source, manifest, _ = discover_kernels(Path("examples/classic"))["cuda_vector_add"]
    request = TranslationRequest(source_path=str(source), manifest_path=str(manifest), targets=[TARGETS["intel_npu_4000"]],
                                 provider="openrouter", model=config.provider.model, validation_policy="offline-validated")
    settings = Settings(database_path=tmp_path / "state.sqlite", runs_path=tmp_path / "cases", experiment_id="test-v4",
                        validation_policy="offline-validated")
    settings.ensure_directories()
    case = {"id": arm, "kernel": "cuda_vector_add", "target": "intel_npu_4000", "arm": arm, "family": "add", "repetition": 1}
    return request, settings, config, case


@pytest.mark.parametrize("arm", ["direct", "hinted_ir", "structured_ir", "structured_ir_no_validation",
                                 "structured_ir_no_reuse", "hinted_ir_no_reuse"])
def test_repair_reuse_cost_and_idempotent_resume(tmp_path, arm):
    args = setup_case(tmp_path, arm)
    provider, compiler = Provider(), RepairCompiler()
    state = run_case(*args, provider, compiler, Event())
    assert state["status"] == "completed", state
    assert len(state["cycles"]) == 2
    expected = 2 if arm == "direct" else 4 if arm.endswith("no_reuse") else 3
    assert len(provider.calls) == expected
    assert compiler.calls == 2
    assert state["active_seconds"] > 0
    for prompt in provider.prompts:
        assert "Original GPU source" in prompt
        assert '"oracle"' not in prompt and '"provenance"' not in prompt
        assert "Shared numerical guidance" not in prompt and "valid transport syntax" not in prompt
    resumed = run_case(*args, provider, compiler, Event())
    assert resumed == state
    assert len(provider.calls) == expected and compiler.calls == 2


def test_graceful_pause_retains_response_and_resumes_before_compile(tmp_path):
    args = setup_case(tmp_path)
    stop = Event()

    class PausingProvider(Provider):
        def generate(self, *args):
            response = super().generate(*args)
            stop.set()
            return response

    provider, compiler = PausingProvider(), CorpusCompiler()
    state = run_case(*args, provider, compiler, stop)
    assert state["status"] == "paused" and state["phase"] is None
    assert compiler.calls == 0 and len(provider.calls) == 1
    stop.clear()
    resumed = run_case(*args, provider, compiler, stop)
    assert resumed["status"] == "completed"
    assert compiler.calls == 1 and len(provider.calls) == 1
    assert resumed["active_seconds"] >= state["active_seconds"]


def test_lost_response_is_blocked_and_never_replayed(tmp_path):
    args = setup_case(tmp_path)
    provider, compiler = Provider(), CorpusCompiler()
    stop = Event()
    stop.set()
    state = run_case(*args, provider, compiler, stop)
    state.update(phase="code", status="running")
    state["cycles"] = [{"number": 1, "status": "running", "calls": {"code": {"status": "in_flight"}}}]
    path = args[1].runs_path / "direct" / "report.json"
    path.write_text(json.dumps(state))
    stop.clear()
    resumed = run_case(*args, provider, compiler, stop)
    assert resumed["status"] == "blocked" and resumed["usage_incomplete"] and resumed["timing_incomplete"]
    assert not provider.calls and compiler.calls == 0


def test_rejected_raw_output_is_charged_but_never_reprompted(tmp_path):
    args = setup_case(tmp_path)

    class RejectedProvider(Provider):
        attempted = 0

        def generate(self, prompt, *rest):
            self.attempted += 1
            if self.attempted == 1:
                raise ProviderError("output token limit reached", {"failure_origin": "model_output",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1000}, "finish_reason": "length"},
                    {"content": "RAW_REPETITION" * 10000})
            assert "RAW_REPETITION" not in prompt
            return super().generate(prompt, *rest)

    provider = RejectedProvider()
    state = run_case(*args, provider, CorpusCompiler(), Event())
    metrics = case_metrics(args[3], state, args[2])
    assert metrics["known_tokens"] == 1115 and metrics["length_finishes"] == 1
    assert state["status"] == "completed" and metrics["cycles"] == 2
    assert metrics["llm_calls"] == 2 and metrics["usage_complete"]


def test_invalid_ir_consumes_cycle_and_is_regenerated(tmp_path):
    args = setup_case(tmp_path, "structured_ir")

    class InvalidFirstIR(Provider):
        def generate(self, prompt, response_model, model):
            response = super().generate(prompt, response_model, model)
            if len(self.calls) == 1:
                response.raw["operations"][0]["op"] = "subtract"
            return response

    provider, compiler = InvalidFirstIR(), CorpusCompiler()
    state = run_case(*args, provider, compiler, Event())
    row = case_metrics(args[3], state, args[2])
    assert state["status"] == "completed" and row["semantic_failures"] == 1
    assert row["cycles"] == 2 and row["llm_calls"] == 3 and compiler.calls == 1
    assert "intermediate_reused_from" not in state["cycles"][1]


def test_budget_stops_between_calls_and_failures_cannot_look_cheap(tmp_path):
    args = setup_case(tmp_path, max_cycles=1)
    state = run_case(*args, Provider(), RepairCompiler(), Event())
    assert state["status"] == "failed" and len(state["cycles"]) == 1
    row = case_metrics(args[3], state, args[2])
    rows = [row, dict(row, arm="structured_ir", known_tokens=1)]
    report = paired_comparison(rows, args[2], "intel_npu_4000", "direct")
    assert report["intervals"]["penalized_token_delta"]["estimate"] == 0
    assert report["verdict"] == "inconclusive"


def test_token_threshold_charges_overshoot_and_stops_before_validation(tmp_path):
    args = setup_case(tmp_path, token_budget=1024)

    class Expensive(Provider):
        def generate(self, *args):
            response = super().generate(*args)
            response.metadata["usage"] = {"input_tokens": 100, "output_tokens": 1000}
            return response

    compiler = CorpusCompiler()
    state = run_case(*args, Expensive(), compiler, Event())
    row = case_metrics(args[3], state, args[2])
    assert state["status"] == "failed" and compiler.calls == 0
    assert row["known_tokens"] == 1100 and row["llm_calls"] == 1 and not row["success"]


def test_success_without_known_usage_cannot_support_efficiency(tmp_path):
    args = setup_case(tmp_path)
    state = run_case(*args, Provider(), CorpusCompiler(), Event())
    state["cycles"][0]["calls"]["code"]["metadata"]["usage"] = {}
    row = case_metrics(args[3], state, args[2])
    assert row["offline_pass"] and not row["success"] and not row["usage_complete"]
    result = paired_comparison([row, dict(row, arm="structured_ir")], args[2], "intel_npu_4000", "direct")
    assert not result["evidence_complete"] and result["verdict"] == "inconclusive"


def test_schedule_is_paired_and_position_balanced():
    config = StudyConfig(repetitions=3)
    cases = case_schedule(config, ["a", "b"], {"a": "f1", "b": "f2"})
    assert cases == case_schedule(config, ["b", "a"], {"a": "f1", "b": "f2"})
    assert len({case["id"] for case in cases}) == 36
    for kernel in ("a", "b"):
        orders = [[case["arm"] for case in cases if case["kernel"] == kernel and case["target"] == "intel_npu_4000"
                   and case["repetition"] == repetition] for repetition in (1, 2, 3)]
        assert all(len({order[position] for order in orders}) == 3 for position in range(3))


def test_all_backend_prompts_have_same_source_interface_and_contract():
    source, _, manifest = discover_kernels(Path("examples/classic"))["triton_layer_norm"]
    for target in TARGETS.values():
        prompts = [stage_prompt("code", arm, source.read_text(), manifest, target, None if arm == "direct" else "test", "")
                   for arm in ("direct", "hinted_ir", "structured_ir")]
        suffixes = [prompt.split("\nTarget profile:", 1)[1] for prompt in prompts]
        assert len(set(suffixes)) == 1
        assert all('"oracle"' not in prompt and "Shared numerical guidance" not in prompt for prompt in prompts)
        assert "center(" not in prompts[0] and "center(" not in prompts[1]


def test_feedback_is_bounded():
    config = StudyConfig(feedback_chars=512, previous_code_chars=1024)
    cycle = {"feedback": {"error": "error" * 10000}, "bundle": {"files": [{"content": "code" * 10000}]}}
    assert len(repair_feedback([cycle], config)) < 1800


def test_partial_report_keeps_pending_outside_finished_denominator(tmp_path):
    config = StudyConfig(kernels=["a"], repetitions=1, targets=["intel_npu_4000"])
    cases = case_schedule(config, ["a"], {"a": "f"})
    (tmp_path / "campaign.json").write_text(json.dumps({"config": config.model_dump(), "cases": cases, "status": "paused"}))
    report = build_report(tmp_path)
    assert all(group["finished"] == 0 and group["planned"] == 1 for group in report["groups"])
    assert all(not item["evidence_complete"] and item["verdict"] == "inconclusive" for item in report["paired"])
    assert not report["joint_claim_supported"]


def test_candidate_stage_hides_oracle_and_evaluator_source(tmp_path, monkeypatch):
    from npu_agent.compilers import DockerCompiler

    _, _, manifest = discover_kernels(Path("examples/classic"))["cuda_vector_add"]
    contract = tmp_path / "contract"
    contract.mkdir()
    (contract / "manifest.json").write_text(manifest.model_dump_json())
    (contract / "target.json").write_text(TARGETS["intel_npu_4000"].model_dump_json())

    def fake_run(self, candidate, output, target, tool, mounts, args):
        public = json.loads((mounts["/contract"] / "manifest.json").read_text())
        assert "oracle" not in public and "provenance" not in public
        assert list(mounts["/framework"].iterdir()) == []
        assert [path.name for path in mounts["/tools"].iterdir()] == [tool]
        command = self._docker_prefix(candidate, output, target, mounts)
        assert sum(value.endswith(":/framework:ro") for value in command) == 1
        assert sum(value.endswith(":/tools:ro") for value in command) == 1
        return StageResult(status="passed")

    monkeypatch.setattr(DockerCompiler, "_run", fake_run)
    compiler = StudyCompiler(Settings(repository_path=Path.cwd()))
    compiler._run(tmp_path / "candidate", tmp_path / "graph", TARGETS["intel_npu_4000"], "intel_build_graph.py", {"/contract": contract})
    assert "oracle" in json.loads((contract / "manifest.json").read_text())


def test_repeated_sigint_drains_active_stage(tmp_path):
    # A real child process exercises signal delivery during a call, without network or containers.
    script = tmp_path / "signal_test.py"
    script.write_text("""import sys, time, signal, os
from pathlib import Path
from threading import Event
sys.path[:0] = [str(Path.cwd() / 'scripts'), str(Path.cwd() / 'tests')]
from case_study_v4 import install_pause_handlers
from test_study_v4 import setup_case, Provider, CorpusCompiler
from npu_agent.study_v4 import run_case
stop = Event()
install_pause_handlers(stop)
class Slow(Provider):
    def generate(self, *args):
        print('IN_CALL', flush=True)
        time.sleep(1)
        return super().generate(*args)
state = run_case(*setup_case(Path(sys.argv[1])), Slow(), CorpusCompiler(), stop)
assert state['status'] == 'paused'
assert state['cycles'][0]['calls']['code']['status'] == 'completed'
print('SAVED', flush=True)
""")
    process = subprocess.Popen([sys.executable, str(script), str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        assert process.stdout.readline().strip() == "IN_CALL"
        os.killpg(process.pid, signal.SIGINT)
        assert "Pause requested" in process.stdout.readline()
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        assert "SAVED" in stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_confirmatory_rejects_reused_corpus():
    with pytest.raises(ValueError, match="holdout"):
        StudyConfig(stage="confirmatory")


def runner_module():
    spec = importlib.util.spec_from_file_location("study_v4_runner", Path("scripts/case_study_v4.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signal_handler_cannot_reenter_event_set():
    module = runner_module()

    class ReentrantStop:
        calls = 0

        def set(self):
            self.calls += 1
            assert self.calls == 1
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

    stop = ReentrantStop()
    previous = module.install_pause_handlers(stop)
    try:
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert stop.calls == 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_preflight_uses_all_schemas_and_is_idempotent(tmp_path):
    from npu_agent.validation import finalize_validation

    module = runner_module()
    config = StudyConfig(kernels=["cuda_vector_add"], targets=["intel_npu_4000"], repetitions=1)
    selected, families, corpus = module.inputs(config)
    campaign = module.prepare(tmp_path, config, selected, families, corpus)

    class PreflightProvider(Provider):
        def endpoint_identity(self):
            return {"provider": "fake"}

        def generate(self, prompt, schema, model):
            source, _, manifest = selected["cuda_vector_add"]
            prompt = stage_prompt("code", "direct", source.read_text(), manifest, TARGETS["intel_npu_4000"], None, "")
            response = super().generate(prompt, schema, model)
            response.metadata["usage"]["reasoning_output_tokens"] = 1
            return response

    class PreflightCompiler(CorpusCompiler):
        def identity(self, target):
            return {"fingerprint": "fake"}

        def compile(self, *args):
            result = super().compile(*args)
            finalize_validation(result.validation, "offline-validated")
            return result

    provider, compiler = PreflightProvider(), PreflightCompiler()
    module.preflight(tmp_path, campaign, config, provider, compiler, selected, Event())
    assert provider.calls == ["CodeBundle", "Hint", "ExecutableKernelIR"]
    assert compiler.calls == 1
    module.preflight(tmp_path, campaign, config, provider, compiler, selected, Event())
    assert len(provider.calls) == 3 and compiler.calls == 1
    assert module.prepare(tmp_path, config, selected, families, corpus)["status"] == "ready"
    changed = config.model_copy(update={"max_cycles": 2})
    with pytest.raises(ValueError, match="frozen"):
        module.prepare(tmp_path, changed, selected, families, corpus)


def test_confirmatory_input_check_rejects_classic_even_with_review_notes(tmp_path):
    module = runner_module()
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"reviewed_by": "test", "review_notes": "test", "development_source_hashes": ["test"],
                                  "families": {"cuda_vector_add": "add"}}))
    config = StudyConfig(stage="confirmatory", repetitions=5, kernels=["cuda_vector_add"], holdout_manifest=str(holdout))
    with pytest.raises(ValueError, match="overlaps development"):
        module.inputs(config)


@pytest.mark.container
@pytest.mark.parametrize("target_id", list(TARGETS))
def test_v4_real_offline_add_with_isolated_candidate(tmp_path, target_id):
    if os.environ.get("NPU_AGENT_CONTAINER_TESTS") != "1":
        pytest.skip("set NPU_AGENT_CONTAINER_TESTS=1 for actual container validation")
    from npu_agent.models import KernelManifest

    fixture = Path("tests/fixtures/backends") / target_id / "add"
    manifest = KernelManifest.model_validate_json((fixture / "manifest.json").read_text())
    compiler = StudyCompiler(Settings(repository_path=Path.cwd(), runs_path=tmp_path, experiment_id="v4-test",
                                      compiler_timeout_seconds=120, validation_policy="offline-validated"))
    result = compiler.compile(fixture, tmp_path / "compile", manifest, TARGETS[target_id])
    assert result.validation.offline_contract_met, result.stderr
