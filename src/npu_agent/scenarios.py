from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from threading import BoundedSemaphore, Event
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import materialize_bundle, store_artifact
from .baseline import _load_inputs, _validate_bundle
from .compilers import Compiler
from .config import Settings
from .database import Database
from .ir import validate_ir
from .mcts import evaluate_compile
from .models import Candidate, CodeBundle, CompileResult, KernelIR, StageResult, TranslationRequest
from .providers import ExperimentCodexProvider, ExperimentOpenRouterProvider, ProviderError, StructuredProvider
from .validation import json_digest, write_json
from .workflow import _analysis_prompt, _backend_contract


SCENARIOS = ("baseline", "structured_ir", "hinted_ir")
ABLATIONS = ("structured_ir_no_validation", "structured_ir_no_reuse")
EXPERIMENT_MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "xhigh"
PROTOCOL_REVISION = "validated-ir-v2"
CASE_STUDY_PROTOCOL = "case-study-v2"
GUIDANCE_PROTOCOL = "case-study-v3"
GUIDANCE_MODEL = "qwen/qwen3-coder-30b-a3b-instruct"
GUIDANCE_SCENARIOS = ("baseline_minimal", "baseline", "structured_ir", "hinted_ir")


class ExecutableKernelIR(KernelIR):
    # Historical KernelIR documents retain their original version and semantics.
    schema_version: Literal["2.0"] = "2.0"


class IntermediateRepresentation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intermediate: str = Field(min_length=1)


class _Blocked(RuntimeError):
    pass


def _generate(
    provider: StructuredProvider, database: Database, state: dict[str, Any], cycle: dict[str, Any],
    stage: str, prompt: str, response_model: type[BaseModel], report_path: Path,
    stop_event: Event | None = None,
) -> BaseModel:
    if stop_event is not None and stop_event.is_set():
        raise KeyboardInterrupt("experiment stopped between stages")
    role = f"cycle:{cycle['number']}:{stage}"
    call = cycle["calls"].get(stage)
    if call is None:
        call = {"status": "in_flight", "prompt": prompt, "schema": response_model.model_json_schema()}
        cycle["calls"][stage] = call
        write_json(report_path, state)
        try:
            response = provider.generate(prompt, response_model, state["model"])
            call.update(status="completed", metadata=response.metadata, response=response.raw)
        except ProviderError as exc:
            metadata = exc.metadata or {}
            invalid_output = metadata.get("exit_code") == 0 and metadata.get("schema_valid") is False
            call.update(status="rejected" if invalid_output else "blocked", metadata=metadata,
                        response=exc.raw, error=str(exc))
        except Exception as exc:
            call.update(status="blocked", metadata={}, response={}, error=str(exc))
        write_json(report_path, state)
    elif call["status"] == "in_flight":
        # A response lost before checkpointing cannot safely be replayed as a free call.
        call.update(status="blocked", metadata={"usage_incomplete": True}, response={},
                    error="interrupted provider call: response and exact usage unavailable")
        write_json(report_path, state)
    metadata = call["metadata"]
    metadata.setdefault("provider", provider.name)
    metadata.setdefault("model", state["model"])
    metadata.setdefault("reasoning_effort", state["reasoning_effort"])
    metadata.setdefault("prompt_sha256", hashlib.sha256(prompt.encode()).hexdigest())
    metadata.setdefault("schema_sha256", json_digest(call["schema"]))
    metadata.setdefault("schema_valid", call["status"] == "completed")
    metadata.setdefault("exit_code", 0 if call["status"] in {"completed", "rejected"} else 1)
    if metadata.get("usage_incomplete"):
        state["usage_incomplete"] = True
    if not database.connection.execute("SELECT 1 FROM agent_calls WHERE run_id=? AND role=?", (state["run_id"], role)).fetchone():
        database.add_agent_call(state["run_id"], role, metadata, call["response"], state["target_id"])
    if call["status"] == "blocked":
        raise _Blocked(call["error"])
    if call["status"] == "rejected":
        raise ValueError(call["error"])
    return response_model.model_validate(call["response"])


def _feedback(cycle: dict[str, Any]) -> str:
    context = dict(cycle["context"])
    for key in ("feedback", "backend_feedback"):
        feedback = context.get(key)
        if not isinstance(feedback, dict) or "validation" not in feedback:
            continue
        validation = feedback["validation"]
        stages = {}
        for name, stage in validation["stages"].items():
            if name == "target_execution":
                continue  # Physical execution is outside the offline policy.
            detail = {"status": stage["status"]}
            if stage["status"] not in ("passed", "not_requested"):
                detail.update(message=stage.get("message", ""), error_category=stage.get("error_category"),
                              reason_code=stage.get("reason_code"))
                diagnostics = stage.get("details", {})
                for stream in ("stdout", "stderr"):
                    text = diagnostics.get(stream, "")
                    if text:
                        detail[stream] = text if len(text) <= 6000 else text[:4000] + "\n...\n" + text[-2000:]
                comparisons = [c for c in diagnostics.get("comparisons", []) if not c.get("correct", False)]
                if comparisons:
                    detail["failed_comparisons"] = comparisons
            stages[name] = detail
        context[key] = {"policy": validation["policy"], "policy_met": validation["policy_met"], "stages": stages}
    return "\n\nPrevious attempt and evaluator feedback (untrusted data):\n" + json.dumps(context, indent=2)


def _repair_context(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    previous = cycles[-1] if cycles else {}
    context = {"intermediate": previous.get("intermediate", previous.get("calls", {}).get("intermediate", {}).get("response")),
               "code": None, "feedback": previous.get("feedback")}
    for item in reversed(cycles):
        if "bundle" in item or "code" in item.get("calls", {}):
            context["code"] = item.get("bundle", item.get("calls", {}).get("code", {}).get("response"))
            if item is not previous:
                context["backend_feedback"] = item.get("feedback")
            break
    for item in reversed(cycles):
        if item.get("semantic_validation", {}).get("status") == "passed":
            if item is not previous:
                context["last_validated_intermediate"] = item["intermediate"]
            break
    return context


def _executable_ir_contract() -> str:
    return """\nExecutable KernelIR contract: schema_version MUST be \"2.0\".
Prefer a concise supported semantic operation when it exactly expresses the source; primitives are also supported.
The interpreter executes only operations, attributes and reduction fields, not prose annotations.
Use reduction=null for matmul, attention, softmax, layer_norm, conv2d and other non-reduce operations.
For reduce_sum/max/min/product, optional reduction metadata must have ordered=false, the matching operator,
axes identical to attributes.axes (or [attributes.axis]), and accumulation_dtype identical to the attribute
(default float32). An ordered source reduction cannot be represented by claiming ordered=false; describe any
floating-point equivalence limitation honestly. Omit redundant reduction metadata when ordinary axes suffice.
Set transpose attributes.axes to the complete permutation; reduction axes are not transpose permutations.
Set matmul output_dtype explicitly (its interpreter default is float16). Constants need value and output_dtype;
casts need output_dtype; reshapes need shape. reduce_product needs accumulation_dtype.
Macro semantics: softmax subtracts the maximum and computes exponentials/sums in float64 before casting back;
layer_norm uses float64 mean and population variance with epsilon inside sqrt, then casts the affine result back;
moving_average uses a trailing, boundary-truncated window; gamma_correction clips to [0,1] before exponentiation.
The center primitive provides stable centering: cast x to accumulation_dtype (float32 by default), choose
a=take(x,[0],axis), d=x-a, then return d-mean(d,axis,keepdims=true) in that dtype. It takes one axis, not axes.
For normalization, prefer composing center, multiply, reduce_sum (keepdims=true), divide by the axis length,
add epsilon, sqrt, divide, multiply weight and add bias. This avoids subtracting a rounded mean near a large
common offset. Use this executable decomposition rather than merely describing stability in numeric_behavior.
Preserve a previously validated IR when feedback concerns backend APIs or compilation only. If the latest IR
regressed, start from last_validated_intermediate and address only an evidenced semantic error.
"""


def _test_candidate(
    compiler: Compiler, database: Database, settings: Settings, request: TranslationRequest,
    state: dict[str, Any], cycle: dict[str, Any], bundle: CodeBundle, manifest: Any, report_path: Path,
    stop_event: Event | None = None, compiler_slots: BoundedSemaphore | None = None,
) -> Candidate:
    target = request.targets[0]
    candidate = Candidate(id=f"{state['run_id']}-{cycle['number']}", run_id=state["run_id"], target_id=target.id,
                          label=f"cycle-{cycle['number']}", rationale=state["scenario"], bundle=bundle,
                          debug_attempt=cycle["number"] - 1)
    database.add_candidate(candidate)
    directory = report_path.parent / f"cycle-{cycle['number']}"
    if "compile_result" not in cycle:
        with compiler_slots if compiler_slots is not None else nullcontext():
            if stop_event is not None and stop_event.is_set():
                raise KeyboardInterrupt("experiment stopped before container validation")
            materialize_bundle(bundle, directory / "candidate", settings.max_generated_bytes)
            cycle["backend_test_attempts"] = cycle.get("backend_test_attempts", 0) + 1
            write_json(report_path, state)
            try:
                result = compiler.compile(directory / "candidate", directory / "compile", manifest, target)
            except Exception as exc:
                raise _Blocked(f"compiler infrastructure failed: {exc}") from exc
            cycle["compile_result"] = result.model_dump(mode="json")
            write_json(report_path, state)
    result = CompileResult.model_validate(cycle["compile_result"])
    if result.validation and "semantic_validation" in cycle:
        result.validation.stages["semantic_validation"] = StageResult.model_validate(cycle["semantic_validation"])
    if not database.connection.execute("SELECT 1 FROM compile_attempts WHERE candidate_id=?", (candidate.id,)).fetchone():
        database.add_compile_attempt(candidate.id, cycle["number"] - 1, result)
    candidate.evaluation = evaluate_compile(candidate.id, result, request.validation_policy, target.id)
    database.add_evaluation(candidate.evaluation)
    cycle["candidate"] = candidate.model_dump(mode="json")
    cycle["feedback"] = {
        "validation": candidate.evaluation.validation.model_dump(mode="json"),
        "exit_code": result.exit_code, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:],
    }
    scope = ("r.experiment_id", settings.experiment_id) if settings.experiment_id else ("r.id", state["run_id"])
    previous_fingerprint = database.connection.execute(
        f"""SELECT ca.compiler_fingerprint FROM compile_attempts ca JOIN candidates c ON c.id=ca.candidate_id
            JOIN app_runs r ON r.id=c.run_id WHERE {scope[0]}=? AND c.target_id=? AND c.id!=?
            AND ca.compiler_fingerprint NOT IN ('unresolved', '') ORDER BY ca.id LIMIT 1""",
        (scope[1], target.id, candidate.id),
    ).fetchone()
    if previous_fingerprint and result.compiler_fingerprint != "unresolved" and previous_fingerprint[0] != result.compiler_fingerprint:
        raise _Blocked("compiler/evaluator fingerprint changed during the experiment")
    if candidate.evaluation.validation.policy_met:
        for kind, raw_path in result.artifacts.items():
            path = Path(raw_path)
            stored, digest = store_artifact(path, settings.runs_path.parent / "artifacts")
            if not database.connection.execute("SELECT 1 FROM artifacts WHERE candidate_id=? AND kind=?", (candidate.id, kind)).fetchone():
                database.add_artifact(candidate.id, kind, str(stored), digest, {"bytes": path.stat().st_size})
    else:
        stages = candidate.evaluation.validation.stages
        for name, stage in stages.items():
            if name != "target_execution" and stage.status == "blocked" and stage.reason_code != "PREREQUISITE_UNAVAILABLE":
                raise _Blocked(stage.message or f"{name} infrastructure unavailable")
        compile_stage = stages.get("target_compile")
        oracle_stage = stages.get("oracle_validation")
        if compile_stage and compile_stage.status == "blocked":
            raise _Blocked(compile_stage.message or "target compiler unavailable")
        if oracle_stage and oracle_stage.status != "passed":
            raise _Blocked(oracle_stage.message or "independent reference unavailable")
        if candidate.evaluation.validation.legacy:
            raise _Blocked("container did not return offline validation evidence")
    return candidate


def translate_scenario(
    request: TranslationRequest, settings: Settings, *, scenario: str, max_cycles: int,
    run_id: str, provider: StructuredProvider, compiler: Compiler,
    stop_event: Event | None = None, compiler_slots: BoundedSemaphore | None = None,
    protocol_revision: str = PROTOCOL_REVISION,
) -> dict[str, Any]:
    allowed = SCENARIOS + ABLATIONS if protocol_revision == CASE_STUDY_PROTOCOL else SCENARIOS
    if protocol_revision == GUIDANCE_PROTOCOL:
        allowed = GUIDANCE_SCENARIOS
    if protocol_revision not in (PROTOCOL_REVISION, CASE_STUDY_PROTOCOL, GUIDANCE_PROTOCOL) or scenario not in allowed or max_cycles < 1:
        raise ValueError("invalid scenario or cycle budget")
    structured = scenario == "structured_ir" or scenario in ABLATIONS
    validate_intermediate = scenario != "structured_ir_no_validation"
    reuse_intermediate = scenario != "structured_ir_no_reuse"
    expected_provider = "openrouter" if protocol_revision == GUIDANCE_PROTOCOL else "codex-cli"
    provider_type = ExperimentOpenRouterProvider if protocol_revision == GUIDANCE_PROTOCOL else ExperimentCodexProvider
    reasoning_effort = "none" if protocol_revision == GUIDANCE_PROTOCOL else REASONING_EFFORT
    if provider.name != "fake" and not isinstance(provider, provider_type):
        raise ValueError("live scenarios require the protocol-pinned provider; only fake providers may be substituted")
    expected_model = GUIDANCE_MODEL if protocol_revision == GUIDANCE_PROTOCOL else EXPERIMENT_MODEL
    if request.provider != expected_provider or request.model != expected_model or request.validation_policy != "offline-validated":
        raise ValueError(f"scenarios require {expected_provider}, {expected_model}, and offline-validated")
    if len(request.targets) != 1 or request.targets[0].hardware_runner:
        raise ValueError("each scenario case requires one offline target without a hardware runner")
    if settings.role_models or settings.role_providers:
        raise ValueError("role-specific model/provider overrides are not permitted in experiments")
    manifest, source = _load_inputs(request)
    target = request.targets[0]
    report_path = settings.runs_path / run_id / "report.json"
    identity = {"source": hashlib.sha256(source.encode()).hexdigest(), "manifest": json_digest(manifest.model_dump(mode="json")),
                "target": json_digest(target.model_dump(mode="json")), "max_cycles": max_cycles, "scenario": scenario,
                "protocol_revision": protocol_revision}
    if report_path.exists():
        state = json.loads(report_path.read_text(encoding="utf-8"))
        if state["identity"] != identity or state["model"] != request.model or state["reasoning_effort"] != reasoning_effort:
            raise ValueError("resume inputs or scenario settings differ from the recorded case")
    else:
        state = {"schema_version": "2.0", "run_id": run_id, "kernel": manifest.name, "target_id": target.id,
                 "scenario": scenario, "model": request.model, "reasoning_effort": reasoning_effort,
                 "identity": identity, "status": "running", "cycles": [], "cycles_used": 0,
                 "cycles_to_success": None, "backend_test_attempts": 0, "terminal_reason": None,
                 "duration_seconds": 0.0, "targets": [], "report_path": str(report_path)}
    database = Database(settings.database_path)
    if database.get_run(run_id) is None:
        database.start_run(run_id, manifest, request, scenario, settings.experiment_id)
    started = time.perf_counter()
    try:
        if state["status"] != "running":
            latest = next((item["candidate"] for item in reversed(state["cycles"]) if "candidate" in item), None)
            state["targets"] = [{"target": target.model_dump(mode="json"), "candidate": latest, "status": state["status"]}]
            return state
        while len(state["cycles"]) < max_cycles or state["cycles"][-1]["status"] == "running":
            if stop_event is not None and stop_event.is_set():
                raise KeyboardInterrupt("experiment stopped between cycles")
            if not state["cycles"] or state["cycles"][-1]["status"] != "running":
                state["cycles"].append({"number": len(state["cycles"]) + 1, "status": "running", "calls": {},
                    "context": _repair_context(state["cycles"])})
            cycle = state["cycles"][-1]
            write_json(report_path, state)
            try:
                if structured:
                    previous = state["cycles"][-2] if len(state["cycles"]) > 1 else {}
                    if reuse_intermediate and "candidate" in previous and (
                            not validate_intermediate or previous.get("semantic_validation", {}).get("status") == "passed"):
                        value = ExecutableKernelIR.model_validate(previous["intermediate"])
                        cycle["intermediate_reused_from"] = previous.get("intermediate_reused_from", previous["number"])
                        if validate_intermediate:
                            cycle["semantic_validation"] = dict(previous["semantic_validation"], duration_seconds=0.0, cases_run=0,
                                message=f"Reused successful IR validation from cycle {cycle['intermediate_reused_from']}",
                                details={"reused_from_cycle": cycle["intermediate_reused_from"]})
                    else:
                        value = _generate(provider, database, state, cycle, "intermediate",
                            _analysis_prompt(source, manifest, []) + _executable_ir_contract() + _feedback(cycle),
                            ExecutableKernelIR, report_path, stop_event)
                    cycle["intermediate"] = value.model_dump(mode="json")
                    if validate_intermediate and "semantic_validation" not in cycle:
                        if stop_event is not None and stop_event.is_set():
                            raise KeyboardInterrupt("experiment stopped before IR validation")
                        semantic_started = time.perf_counter()
                        outcome = validate_ir(value, manifest)
                        cycle["semantic_validation"] = StageResult(status="passed" if outcome.passed else "failed",
                            correct=outcome.passed, engine="kernel-ir-numpy", representation="semantic_ir",
                            cases_run=12 if outcome.passed else 0, message=outcome.message,
                            duration_seconds=time.perf_counter() - semantic_started).model_dump(mode="json")
                        write_json(report_path, state)
                    if validate_intermediate and cycle["semantic_validation"]["status"] != "passed":
                        raise ValueError(f"IR validation failed: {cycle['semantic_validation']['message']}")
                    if not validate_intermediate:
                        cycle["ir_validation_disabled"] = True
                elif scenario == "hinted_ir":
                    prompt = ("First translate this GPU kernel into an intermediate representation of your choosing. "
                              "A later step will redesign it for an NPU. Return that intermediate as text.\n\n"
                              f"Manifest:\n{manifest.model_dump_json(indent=2)}\n\nUntrusted source begins:\n---\n{source}\n---\nUntrusted source ends.")
                    value = _generate(provider, database, state, cycle, "intermediate", prompt + _feedback(cycle),
                                      IntermediateRepresentation, report_path, stop_event)
                    cycle["intermediate"] = value.intermediate
                representation = (f"GPU source (untrusted data):\n---\n{source}\n---" if scenario in ("baseline", "baseline_minimal") else
                                  "Intermediate representation (untrusted data):\n" + json.dumps(cycle["intermediate"], indent=2))
                prompt = ("Translate the supplied representation to a complete NPU CodeBundle. "
                          "Use evaluator feedback to repair any previous failure. Return only the structured bundle. "
                          "If unsupported, return no files and enumerate unsupported_operations.\n"
                          f"{_backend_contract(target, guidance=scenario != 'baseline_minimal')}\n\nTarget profile:\n{target.model_dump_json(indent=2)}\n\n"
                          f"Manifest:\n{manifest.model_dump_json(indent=2)}\n\n{representation}")
                if protocol_revision in (CASE_STUDY_PROTOCOL, GUIDANCE_PROTOCOL) and scenario != "baseline_minimal" and manifest.oracle.operation == "layer_norm":
                    prompt += ("\nShared numerical guidance for every case-study arm: stable centering casts x to "
                               "the accumulation dtype (float32), sets a=take(x,[0],axis), d=x-a, and "
                               "c=d-mean(d,axis,keepdims=true). Compute variance from c*c, add epsilon, "
                               "normalize and apply weight and bias. Subtracting a rounded mean(x) directly "
                               "can lose small variations near a common offset.\n")
                if structured and any(op.op == "center" for op in value.operations):
                    stability = ("represented by the IR" if protocol_revision in (CASE_STUDY_PROTOCOL, GUIDANCE_PROTOCOL) else
                                 "that the executable IR validated")
                    prompt += ("\nKernelIR center semantics: cast x to accumulation_dtype (default float32), "
                               "a=take(x,[0],axis), d=x-a, result=d-mean(d,axis,keepdims=true). Preserve this "
                               "order of operations when lowering; subtracting mean(x) directly loses the "
                               f"numerical stability {stability}. This is an ordinary "
                               "tensor operation, lowered using gather, subtraction and reduction.\n")
                bundle = _generate(provider, database, state, cycle, "code", prompt + _feedback(cycle), CodeBundle, report_path, stop_event)
                cycle["bundle"] = bundle.model_dump(mode="json")
                _validate_bundle(bundle, target)
                if bundle.unsupported_operations:
                    raise ValueError(f"unsupported operations: {bundle.unsupported_operations}")
                candidate = _test_candidate(compiler, database, settings, request, state, cycle, bundle, manifest, report_path,
                                            stop_event, compiler_slots)
                cycle["status"] = "passed" if candidate.evaluation.validation.policy_met else "failed"
            except _Blocked as exc:
                cycle.update(status="blocked", error=str(exc))
                state.update(status="blocked", terminal_reason=str(exc))
            except ValueError as exc:
                cycle.update(status="failed", feedback={"error": str(exc), "stage": "code" if "code" in cycle["calls"] else "intermediate"})
            state["cycles_used"] = len(state["cycles"])
            state["backend_test_attempts"] = sum(item.get("backend_test_attempts", 0) for item in state["cycles"])
            if cycle["status"] == "passed":
                state.update(status="completed", cycles_to_success=cycle["number"], terminal_reason="solved")
            write_json(report_path, state)
            if state["status"] != "running":
                break
        if state["status"] == "running":
            state.update(status="failed", terminal_reason="cycle_budget_exhausted")
        latest = next((item["candidate"] for item in reversed(state["cycles"]) if "candidate" in item), None)
        state["targets"] = [{"target": target.model_dump(mode="json"), "candidate": latest, "status": state["status"]}]
        return state
    finally:
        state["cycles_used"] = len(state["cycles"])
        state["backend_test_attempts"] = sum(item.get("backend_test_attempts", 0) for item in state["cycles"])
        state["duration_seconds"] += time.perf_counter() - started
        write_json(report_path, state)
        database.finish_run(run_id, state["status"], state, duration_seconds=state["duration_seconds"])
        database.close()
