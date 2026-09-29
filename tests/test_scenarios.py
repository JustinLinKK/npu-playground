import json
from pathlib import Path

import pytest

from npu_agent.config import INTEL_NPU_4000, Settings
from npu_agent.database import Database
from npu_agent.models import CodeBundle, KernelIR, StageResult, TranslationRequest
from npu_agent.providers import ProviderError
from npu_agent.reporting import _token_totals
from npu_agent.scenarios import EXPERIMENT_MODEL, SCENARIOS, translate_scenario
from test_experiment import CorpusCompiler, CorpusProvider


class TracedProvider(CorpusProvider):
    def __init__(self, model=EXPERIMENT_MODEL):
        super().__init__(model)
        self.prompts = []

    def generate(self, prompt, response_model, model=None):
        self.prompts.append((response_model.__name__, prompt))
        return super().generate(prompt, response_model, model)


class FailingCompiler(CorpusCompiler):
    def __init__(self, failure="compile", failures=1):
        super().__init__()
        self.failure, self.failures = failure, failures

    def compile(self, *args):
        result = super().compile(*args)
        if self.calls <= self.failures:
            result.stderr = f"test failure: {self.failure}"
            if self.failure == "compile":
                result.success, result.exit_code = False, 1
                result.validation.stages["target_compile"] = StageResult(status="failed", message=result.stderr)
            elif self.failure == "blocked":
                result.success, result.exit_code = False, 3
                result.validation.stages["target_compile"] = StageResult(status="blocked", message="image unavailable")
            else:
                result.host_correct = False if self.failure == "numerical" else None
                result.validation.stages["host_execution"] = StageResult(
                    status="failed" if self.failure == "numerical" else "unsupported", correct=result.host_correct,
                    message=result.stderr)
        return result


def run_case(tmp_path, provider, compiler, scenario="baseline", max_cycles=10, **options):
    from npu_agent.scenarios import GUIDANCE_PROTOCOL

    manifest = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()
    request = TranslationRequest(source_path=str(manifest.parent / "kernel.cu"), manifest_path=str(manifest),
                                 targets=[INTEL_NPU_4000], model=provider.model,
                                 provider="openrouter" if options.get("protocol_revision") == GUIDANCE_PROTOCOL else "codex-cli",
                                 validation_policy="offline-validated")
    settings = Settings(database_path=tmp_path / "state.sqlite", runs_path=tmp_path / "runs")
    return translate_scenario(request, settings, scenario=scenario, max_cycles=max_cycles,
                              run_id="case", provider=provider, compiler=compiler, **options)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("failure", ["compile", "numerical", "unsupported"])
def test_each_scenario_revisits_stages_and_stops_at_first_success(tmp_path, scenario, failure):
    provider = TracedProvider()
    compiler = FailingCompiler(failure)
    result = run_case(tmp_path, provider, compiler, scenario)
    assert result["status"] == "completed"
    assert result["cycles_used"] == result["cycles_to_success"] == result["backend_test_attempts"] == 2
    assert compiler.calls == 2
    assert len(provider.calls) == {"baseline": 2, "structured_ir": 3, "hinted_ir": 4}[scenario]
    assert failure in provider.prompts[-1][1]
    if scenario != "baseline":
        if scenario == "hinted_ir":
            assert provider.calls[0] == provider.calls[2]
        else:
            assert result["cycles"][1]["intermediate_reused_from"] == 1
            assert result["cycles"][1]["semantic_validation"]["duration_seconds"] == 0
            assert result["cycles"][1]["semantic_validation"]["cases_run"] == 0
        assert "test failure" in provider.prompts[2][1]
        assert "# model" in provider.prompts[2][1]
        assert "GPU source (untrusted data)" not in provider.prompts[1][1]
    if scenario != "structured_ir":
        assert all("KernelIR" not in prompt and "semantic_validation" not in prompt for _, prompt in provider.prompts)
    assert all("knowledge" not in prompt.lower() for _, prompt in provider.prompts)
    database = Database(tmp_path / "state.sqlite")
    calls = database.connection.execute("SELECT usage_json FROM agent_calls").fetchall()
    assert _token_totals(row[0] for row in calls)["total_tokens"] == len(provider.calls) * 15
    database.close()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_budget_exhaustion_includes_initial_attempt(tmp_path, scenario):
    provider, compiler = TracedProvider(), FailingCompiler(failures=100)
    result = run_case(tmp_path, provider, compiler, scenario)
    assert result["cycles_used"] == 10
    assert result["cycles_to_success"] is None
    assert result["terminal_reason"] == "cycle_budget_exhausted"
    assert compiler.calls == 10
    assert len(provider.calls) == {"baseline": 10, "structured_ir": 11, "hinted_ir": 20}[scenario]


def test_invalid_ir_consumes_cycle_without_backend_generation(tmp_path):
    class InvalidIRProvider(TracedProvider):
        def generate(self, prompt, response_model, model=None):
            response = super().generate(prompt, response_model, model)
            if issubclass(response_model, KernelIR) and len(self.calls) == 1:
                response.value.operations[0].op = "multiply"
                response.raw.update(response.value.model_dump(mode="json"))
            return response

    provider, compiler = InvalidIRProvider(), CorpusCompiler()
    result = run_case(tmp_path, provider, compiler, "structured_ir")
    assert result["cycles_to_success"] == 2
    assert result["backend_test_attempts"] == 1
    assert provider.calls == ["ExecutableKernelIR", "ExecutableKernelIR", "CodeBundle"]
    assert "IR validation failed" in provider.prompts[1][1]


def test_schema_rejection_is_counted_with_its_usage(tmp_path):
    class InvalidOutputProvider(TracedProvider):
        def generate(self, prompt, response_model, model=None):
            if not self.calls:
                self.calls.append("invalid")
                raise ProviderError("invalid structured response", {"exit_code": 0, "schema_valid": False,
                    "usage": {"input_tokens": 20, "output_tokens": 7}}, {"broken": True})
            return super().generate(prompt, response_model, model)

    result = run_case(tmp_path, InvalidOutputProvider(), CorpusCompiler())
    assert result["cycles_to_success"] == 2
    assert result["backend_test_attempts"] == 1
    database = Database(tmp_path / "state.sqlite")
    calls = database.connection.execute("SELECT usage_json FROM agent_calls").fetchall()
    assert _token_totals(row[0] for row in calls)["total_tokens"] == 42
    database.close()


@pytest.mark.parametrize("failure", ["provider", "compiler"])
def test_infrastructure_failure_is_blocked_without_retries(tmp_path, failure):
    class UnavailableProvider(TracedProvider):
        def generate(self, *args):
            self.calls.append("unavailable")
            raise ProviderError("provider unavailable")

    provider = UnavailableProvider() if failure == "provider" else TracedProvider()
    compiler = CorpusCompiler() if failure == "provider" else FailingCompiler("blocked")
    result = run_case(tmp_path, provider, compiler)
    assert result["status"] == "blocked"
    assert result["cycles_to_success"] is None
    assert len(provider.calls) == 1
    assert compiler.calls == (0 if failure == "provider" else 1)


def test_resume_reuses_completed_model_stages_and_accumulates_attempts(tmp_path):
    class InterruptedCompiler(CorpusCompiler):
        def compile(self, *args):
            if self.calls == 0:
                self.calls += 1
                raise KeyboardInterrupt
            return super().compile(*args)

    provider, compiler = TracedProvider(), InterruptedCompiler()
    with pytest.raises(KeyboardInterrupt):
        run_case(tmp_path, provider, compiler, "structured_ir")
    before = json.loads((tmp_path / "runs/case/report.json").read_text())
    assert before["cycles"][0]["semantic_validation"]["status"] == "passed"
    result = run_case(tmp_path, provider, compiler, "structured_ir")
    assert result["status"] == "completed"
    assert result["cycles_to_success"] == 1
    assert result["backend_test_attempts"] == compiler.calls == 2
    assert provider.calls == ["ExecutableKernelIR", "CodeBundle"]
    assert result["duration_seconds"] >= before["duration_seconds"]
    database = Database(tmp_path / "state.sqlite")
    assert database.connection.execute("SELECT COUNT(*) FROM agent_calls").fetchone()[0] == 2
    database.close()


def test_resume_reuses_completed_container_result(tmp_path, monkeypatch):
    import npu_agent.scenarios as scenarios

    provider, compiler = TracedProvider(), CorpusCompiler()
    original = scenarios.evaluate_compile
    monkeypatch.setattr(scenarios, "evaluate_compile", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run_case(tmp_path, provider, compiler)
    monkeypatch.setattr(scenarios, "evaluate_compile", original)
    result = run_case(tmp_path, provider, compiler)
    assert result["status"] == "completed"
    assert len(provider.calls) == compiler.calls == 1
    database = Database(tmp_path / "state.sqlite")
    assert database.connection.execute("SELECT COUNT(*) FROM compile_attempts").fetchone()[0] == 1
    database.close()


def test_lost_inflight_response_is_explicitly_unknown(tmp_path):
    class InterruptedProvider(TracedProvider):
        def generate(self, *args):
            self.calls.append("interrupted")
            raise KeyboardInterrupt

    provider = InterruptedProvider()
    with pytest.raises(KeyboardInterrupt):
        run_case(tmp_path, provider, CorpusCompiler())
    result = run_case(tmp_path, provider, CorpusCompiler())
    assert result["status"] == "blocked"
    assert result["usage_incomplete"] is True
    assert len(provider.calls) == 1


def test_unsupported_bundle_can_be_repaired(tmp_path):
    class UnsupportedProvider(TracedProvider):
        def generate(self, prompt, response_model, model=None):
            response = super().generate(prompt, response_model, model)
            if len(self.calls) == 1:
                response.raw.update(CodeBundle(backend="intel_openvino", files=[], unsupported_operations=["operation"]).model_dump(mode="json"))
            return response

    result = run_case(tmp_path, UnsupportedProvider(), CorpusCompiler())
    assert result["cycles_to_success"] == 2
    assert result["backend_test_attempts"] == 1


def test_changed_compiler_fingerprint_cannot_produce_a_matched_success(tmp_path):
    class ChangedCompiler(FailingCompiler):
        def compile(self, *args):
            result = super().compile(*args)
            if self.calls > 1:
                result.compiler_fingerprint = "different-image"
            return result

    result = run_case(tmp_path, TracedProvider(), ChangedCompiler())
    assert result["status"] == "blocked"
    assert result["cycles_to_success"] is None
    assert "fingerprint changed" in result["terminal_reason"]


def test_backend_repair_reuses_validated_ir_without_regeneration(tmp_path):
    class RegressingProvider(TracedProvider):
        def generate(self, prompt, response_model, model=None):
            response = super().generate(prompt, response_model, model)
            if issubclass(response_model, KernelIR) and len(self.calls) == 3:
                response.value.operations[0].op = "multiply"
                response.raw.update(response.value.model_dump(mode="json"))
            return response

    provider, compiler = RegressingProvider(), FailingCompiler()
    result = run_case(tmp_path, provider, compiler, "structured_ir")
    assert result["cycles_to_success"] == 2
    assert compiler.calls == 2
    assert provider.calls == ["ExecutableKernelIR", "CodeBundle", "CodeBundle"]
    assert result["cycles"][1]["intermediate"] == result["cycles"][0]["intermediate"]
    assert result["cycles"][1]["intermediate_reused_from"] == 1
    assert "test failure: compile" in provider.prompts[-1][1]


def test_repair_context_retains_backend_failure_across_invalid_ir():
    from npu_agent.scenarios import _repair_context

    backend = {"bundle": {"files": ["last code"]}, "intermediate": {"valid": True},
               "semantic_validation": {"status": "passed"}, "feedback": {"error": "compiler diagnostic"}}
    invalid = {"intermediate": {"valid": False}, "feedback": {"error": "IR validation failed"}}
    context = _repair_context([backend, invalid])
    assert context["code"] == backend["bundle"]
    assert context["backend_feedback"] == backend["feedback"]
    assert context["last_validated_intermediate"] == backend["intermediate"]
    assert context["feedback"] == invalid["feedback"]


def test_resume_preserves_ir_reuse_and_all_paid_calls(tmp_path):
    class InterruptedRepairCompiler(FailingCompiler):
        def compile(self, *args):
            if self.calls == 1:
                self.calls += 1
                raise KeyboardInterrupt
            return super().compile(*args)

    provider, compiler = TracedProvider(), InterruptedRepairCompiler()
    with pytest.raises(KeyboardInterrupt):
        run_case(tmp_path, provider, compiler, "structured_ir")
    result = run_case(tmp_path, provider, compiler, "structured_ir")
    assert result["cycles_to_success"] == 2
    assert result["backend_test_attempts"] == compiler.calls == 3
    assert provider.calls == ["ExecutableKernelIR", "CodeBundle", "CodeBundle"]
    assert result["cycles"][1]["intermediate_reused_from"] == 1
    database = Database(tmp_path / "state.sqlite")
    calls = database.connection.execute("SELECT usage_json FROM agent_calls").fetchall()
    assert _token_totals(row[0] for row in calls)["total_tokens"] == 45
    database.close()


def test_compact_feedback_keeps_failed_numeric_cases_and_compiler_diagnostics():
    from npu_agent.scenarios import _feedback

    stages = {"host_execution": {"status": "failed", "error_category": "numerical_mismatch",
        "details": {"command": ["irrelevant container mount"], "comparisons": [
            {"test_id": "normal", "correct": True},
            {"test_id": "boundary", "correct": False, "max_absolute_error": 0.01}]}},
        "target_compile": {"status": "failed", "message": "compile failed",
            "details": {"stderr": "fatal error: unknown type\n" + "x" * 7000 + "\ncompiler exited"}},
        "target_execution": {"status": "blocked", "message": "physical hardware unavailable"}}
    text = _feedback({"context": {"feedback": {"validation": {
        "policy": "offline-validated", "policy_met": False, "stages": stages}}}})
    assert all(s in text for s in ("boundary", "0.01", "fatal error: unknown type", "compiler exited"))
    assert all(s not in text for s in ("normal", "irrelevant container mount", "physical hardware unavailable"))


def test_experimental_ir_schema_pins_v2_without_changing_historical_reader():
    from npu_agent.scenarios import ExecutableKernelIR

    assert ExecutableKernelIR.model_json_schema()["properties"]["schema_version"]["const"] == "2.0"
    assert KernelIR.model_json_schema()["properties"]["schema_version"]["default"] == "1.0"


@pytest.mark.parametrize("scenario,calls,validations,reused", [
    ("structured_ir", 3, 1, True),
    ("structured_ir_no_validation", 3, 0, True),
    ("structured_ir_no_reuse", 4, 2, False),
])
def test_v2_ablations_isolate_validation_and_reuse(tmp_path, monkeypatch, scenario, calls, validations, reused):
    import npu_agent.scenarios as scenarios

    observed = []
    original = scenarios.validate_ir
    def validate(*args):
        observed.append(args)
        return original(*args)
    monkeypatch.setattr(scenarios, "validate_ir", validate)
    provider, compiler = TracedProvider(), FailingCompiler("numerical")
    result = run_case(tmp_path, provider, compiler, scenario, protocol_revision=scenarios.CASE_STUDY_PROTOCOL)
    assert result["cycles_to_success"] == 2
    assert len(provider.calls) == calls
    assert len(observed) == validations
    assert ("intermediate_reused_from" in result["cycles"][1]) is reused
    assert compiler.calls == 2
    if not validations:
        assert all(c["ir_validation_disabled"] and "semantic_validation" not in c for c in result["cycles"])
        stages = result["targets"][0]["candidate"]["evaluation"]["validation"]["stages"]
        assert "semantic_validation" not in stages
        assert stages["host_execution"]["status"] == "passed"


def test_unchecked_ir_does_not_silently_execute_validation(tmp_path, monkeypatch):
    import npu_agent.scenarios as scenarios

    class IncorrectIR(TracedProvider):
        def generate(self, prompt, response_model, model=None):
            response = super().generate(prompt, response_model, model)
            if issubclass(response_model, KernelIR):
                response.raw["operations"][0]["op"] = "multiply"
            return response

    monkeypatch.setattr(scenarios, "validate_ir", lambda *args: pytest.fail("unchecked IR was executed"))
    result = run_case(tmp_path, IncorrectIR(), FailingCompiler("numerical", failures=100),
                      "structured_ir_no_validation", max_cycles=2, protocol_revision=scenarios.CASE_STUDY_PROTOCOL)
    assert result["terminal_reason"] == "cycle_budget_exhausted"
    assert result["cycles_to_success"] is None
    assert result["backend_test_attempts"] == 2


def test_v2_protocol_cannot_resume_a_historical_case(tmp_path):
    from npu_agent.scenarios import CASE_STUDY_PROTOCOL

    run_case(tmp_path, TracedProvider(), CorpusCompiler())
    with pytest.raises(ValueError, match="resume inputs"):
        run_case(tmp_path, TracedProvider(), CorpusCompiler(), protocol_revision=CASE_STUDY_PROTOCOL)


@pytest.mark.parametrize("scenario,calls", [
    ("baseline_minimal", 2), ("baseline", 2), ("structured_ir", 3), ("hinted_ir", 4),
])
def test_v3_preserves_feedback_and_ir_workflows_with_openrouter(tmp_path, scenario, calls):
    from npu_agent.scenarios import GUIDANCE_MODEL, GUIDANCE_PROTOCOL

    provider, compiler = TracedProvider(GUIDANCE_MODEL), FailingCompiler()
    result = run_case(tmp_path, provider, compiler, scenario, protocol_revision=GUIDANCE_PROTOCOL)
    assert result["model"] == GUIDANCE_MODEL
    assert result["cycles_to_success"] == compiler.calls == 2
    assert len(provider.calls) == calls
    assert "test failure: compile" in provider.prompts[-1][1]
    if scenario == "structured_ir":
        assert result["cycles"][0]["semantic_validation"]["status"] == "passed"
        assert result["cycles"][1]["intermediate_reused_from"] == 1
    else:
        assert all("semantic_validation" not in cycle for cycle in result["cycles"])
    if scenario == "baseline_minimal":
        assert all("node.output(0).get_tensor()" not in prompt for _, prompt in provider.prompts)
        assert all("GPU source (untrusted data)" in prompt for _, prompt in provider.prompts)
