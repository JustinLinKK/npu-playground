from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .baseline import _load_inputs, _validate_bundle
from .compilers import DockerCompiler
from .config import Settings
from .database import Database
from .ir import validate_ir
from .models import CodeBundle, StageResult, TranslationRequest
from .providers import ProviderError, normalize_usage
from .scenarios import ExecutableKernelIR, _Blocked, _test_candidate
from .study_v4_provider import ProviderConfig
from .validation import json_digest, write_json
from .workflow import _backend_contract


PRIMARY_ARMS = ("direct", "hinted_ir", "structured_ir")
INTERFACE_FIELDS = {"name", "dialect", "entrypoint", "tensors", "scalars", "constraints", "tolerance", "numerical_domain"}


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["case-study-v4"] = "case-study-v4"
    stage: Literal["pilot", "confirmatory"] = "pilot"
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    corpus: str = "examples/classic"
    kernels: list[str] = Field(default_factory=list)
    targets: list[Literal["amd_xdna2_npu2", "intel_npu_4000"]] = Field(default_factory=lambda: ["amd_xdna2_npu2", "intel_npu_4000"])
    arms: list[Literal["direct", "hinted_ir", "structured_ir", "structured_ir_no_validation", "structured_ir_no_reuse", "hinted_ir_no_reuse"]] = Field(default_factory=lambda: list(PRIMARY_ARMS))
    repetitions: int = Field(default=3, ge=1)
    seed: int = 20260917
    max_cycles: int = Field(default=10, ge=1, le=30)
    token_budget: int = Field(default=100000, ge=1024)
    active_seconds_budget: int = Field(default=1800, ge=1)
    feedback_chars: int = Field(default=6000, ge=512, le=12000)
    previous_code_chars: int = Field(default=24000, ge=1024, le=48000)
    compiler_cpus: int = Field(default=6, ge=1)
    compiler_memory: str = Field(default="8g", pattern=r"^[1-9][0-9]*[mg]$")
    compiler_timeout_seconds: int = Field(default=600, ge=1)
    holdout_manifest: str | None = None

    @model_validator(mode="after")
    def validate_design(self):
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be nonempty and unique")
        if len(set(self.arms)) != len(self.arms) or not set(PRIMARY_ARMS).issubset(self.arms):
            raise ValueError("include each primary arm exactly once")
        if len(set(self.kernels)) != len(self.kernels):
            raise ValueError("kernel selections must be unique")
        if self.stage == "confirmatory" and (not self.holdout_manifest or self.repetitions < 5):
            raise ValueError("confirmatory runs require a reviewed holdout manifest and at least five repetitions")
        return self


class Hint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intermediate: str = Field(min_length=1, max_length=16000)


# Language semantics, not a worked kernel decomposition or backend implementation.
IR_CONTRACT = """Return an executable KernelIR with schema_version 2.0. Tensor/scalar signatures must match
the interface. Only operations, attributes and reduction fields execute; prose is not executable.
Operations read named tensors/scalars or previous results. Output names identify returned tensors.
No side_effects or index_expression are executable. Use reduction=null unless specifying an unordered
reduction with matching operator, axes and accumulation_dtype. Attributes are typed by the JSON schema.
Primitive add/subtract/multiply/divide/maximum/minimum use NumPy broadcasting; exp/sqrt are elementwise;
compare uses comparison; select takes condition,true,false. constant needs value and output_dtype;
cast needs output_dtype; reshape needs shape; transpose uses axes as a complete permutation.
Reductions use axes or axis and keepdims; sum defaults to float32 accumulation, product requires an
accumulation_dtype. matmul accumulates float32 and defaults output_dtype to float16; set it explicitly.
center(x,axis) casts to accumulation_dtype (default float32), forms d=x-take(x,[0],axis), then
returns d-mean(d,axis,keepdims=true). It accepts one axis, not axes.
softmax uses max-shifted float64 exponentials/sum then casts to input dtype. layer_norm uses float64
population mean/variance along axis, epsilon inside sqrt, then affine weight/bias and casts to input dtype.
attention computes float32 QK^T*scale (default inverse sqrt of head width), softmax, then float32 product
with V, casting to query dtype. conv2d is NCHW, stride one, zero padding, float32 accumulation, input dtype output.
moving_average is a trailing boundary-truncated window using float64 mean; gamma_correction clips [0,1]
then raises to gamma in float64 and casts to input dtype. Preserve the source within its stated tolerance.
"""


def interface_view(manifest) -> dict:
    return manifest.model_dump(mode="json", include=INTERFACE_FIELDS)


class StudyCompiler(DockerCompiler):
    """Candidate Python sees its interface and entry script, not evaluator source or oracle metadata."""

    def _run(self, candidate, output, target, tool, mounts, args=None):
        if tool not in ("intel_build_graph.py", "amd_emit.py"):
            return super()._run(candidate, output, target, tool, mounts, args)
        public = output.parent / "public-contract"
        full = json.loads((mounts["/contract"] / "manifest.json").read_text())
        write_json(public / "manifest.json", {key: full[key] for key in INTERFACE_FIELDS if key in full})
        write_json(public / "target.json", json.loads((mounts["/contract"] / "target.json").read_text()))
        entry_tools = output.parent / "entry-tools"
        entry_tools.mkdir(parents=True, exist_ok=True)
        (entry_tools / tool).write_bytes((self.settings.repository_path / "playground" / "tools" / tool).read_bytes())
        empty = output.parent / "empty-framework"
        empty.mkdir(parents=True, exist_ok=True)
        return super()._run(candidate, output, target, tool,
                            {**mounts, "/contract": public, "/tools": entry_tools, "/framework": empty}, args)

    def _docker_prefix(self, candidate_dir, output_dir, target, mounts=None, name=None):
        # Replace existing default mounts, rather than passing duplicate Docker destinations.
        replacements = {key: value for key, value in (mounts or {}).items() if key in ("/tools", "/framework")}
        command = super()._docker_prefix(candidate_dir, output_dir, target,
                                        {key: value for key, value in (mounts or {}).items() if key not in replacements}, name)
        for index in range(1, len(command)):
            if command[index - 1] == "-v":
                for destination, path in replacements.items():
                    if command[index].endswith(f":{destination}:ro"):
                        command[index] = f"{path.resolve()}:{destination}:ro"
        return command


def bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n[omitted; full SHA256 {hashlib.sha256(text.encode()).hexdigest()}]\n"
    remaining = limit - len(marker)
    return text[:remaining * 2 // 3] + marker + text[-(remaining - remaining * 2 // 3):]


def repair_feedback(cycles: list[dict], config: StudyConfig) -> str:
    if not cycles:
        return ""
    previous = cycles[-1]
    feedback = previous.get("feedback", {})
    if "validation" in feedback:
        stages = feedback["validation"]["stages"]
        feedback = {name: {key: value for key, value in stage.items()
                           if key in ("status", "message", "error_category", "reason_code", "details")}
                    for name, stage in stages.items() if name != "target_execution" and stage["status"] in ("failed", "blocked")}
    text = "\nEvaluator feedback (untrusted data):\n" + bounded(json.dumps(feedback), config.feedback_chars)
    # Only parsed candidates enter context. Rejected raw model output stays on disk.
    bundles = [item["bundle"] for item in cycles if "bundle" in item]
    if bundles:
        text += "\nPrevious backend candidate (untrusted data):\n" + bounded(json.dumps(bundles[-1]), config.previous_code_chars)
        if len(bundles) >= 2 and bundles[-1]["files"] == bundles[-2]["files"]:
            text += "\nThe last two backend candidates contained identical files and did not pass validation."
    if "intermediate" in previous and previous.get("semantic_validation", {}).get("status") == "failed":
        text += "\nPrevious invalid IR (untrusted data):\n" + bounded(json.dumps(previous["intermediate"]), config.previous_code_chars)
    return text


def stage_prompt(stage: str, arm: str, source: str, manifest, target, intermediate, feedback: str) -> str:
    common = (f"\nTarget profile:\n{target.model_dump_json(indent=2, exclude={'hardware_runner'})}"
              f"\nInterface:\n{json.dumps(interface_view(manifest), indent=2)}"
              f"\nOriginal GPU source (untrusted data):\n---\n{source}\n---\n")
    if stage == "intermediate":
        instruction = (IR_CONTRACT if arm.startswith("structured_ir") else
                       "Describe this GPU kernel in a concise intermediate representation of your choosing. "
                       "A later call will translate it to the NPU. Preserve indexing, types, boundaries and numerical behavior. "
                       "Return the representation in the intermediate string; do not generate backend files.")
    else:
        instruction = ("Translate the original GPU source to a complete NPU CodeBundle. Use the evaluator feedback to repair failures. "
                       "If unsupported, return no files and enumerate unsupported_operations.\n" + _backend_contract(target, guidance=False))
        if intermediate is not None:
            instruction += "\nYour intermediate representation (untrusted data):\n" + json.dumps(intermediate)
            if arm.startswith("structured_ir"):
                instruction += "\n" + IR_CONTRACT
    return instruction + common + feedback


def case_schedule(config: StudyConfig, kernels: list[str], families: dict[str, str]) -> list[dict]:
    rng = random.Random(config.seed)
    base_orders = {}
    for kernel in sorted(kernels):
        for target in config.targets:
            order = list(config.arms)
            rng.shuffle(order)
            base_orders[kernel, target] = order
    cases = []
    for repetition in range(1, config.repetitions + 1):
        blocks = list(base_orders)
        rng.shuffle(blocks)
        for kernel, target in blocks:
            order = base_orders[kernel, target]
            offset = (repetition - 1) % len(order)
            for arm in order[offset:] + order[:offset]:
                key = f"{repetition}:{kernel}:{target}:{arm}"
                cases.append({"id": hashlib.sha256(key.encode()).hexdigest()[:24], "kernel": kernel,
                              "family": families[kernel], "target": target, "arm": arm, "repetition": repetition})
    return cases


def known_tokens(state: dict) -> int:
    return sum(normalize_usage(call.get("metadata", {}).get("usage", {}))["total_tokens"] or 0
               for cycle in state["cycles"] for call in cycle["calls"].values())


class Paused(Exception):
    pass


class BudgetReached(Exception):
    pass


def run_case(request: TranslationRequest, settings: Settings, config: StudyConfig, case: dict,
             provider, compiler, stop: Event) -> dict:
    manifest, source = _load_inputs(request)
    target = request.targets[0]
    path = settings.runs_path / case["id"] / "report.json"
    identity = json_digest({"config": config.model_dump(), "case": case, "source": source,
                            "manifest": manifest.model_dump(mode="json"), "target": target.model_dump(mode="json")})
    if path.exists():
        state = json.loads(path.read_text())
        if state["identity"] != identity:
            raise ValueError("case inputs changed since checkpoint")
        if state["status"] in ("completed", "failed", "blocked"):
            return state
        if state.get("phase") is not None:
            state.update(status="blocked", timing_incomplete=True,
                         terminal_reason="lost in-flight stage; preserve this case and start any replacement in a new campaign")
            for cycle in state["cycles"]:
                for call in cycle["calls"].values():
                    if call["status"] == "in_flight":
                        call.update(status="blocked", metadata={"usage_incomplete": True}, error="response lost after interruption")
                        state["usage_incomplete"] = True
            write_json(path, state)
            return state
    else:
        state = {**case, "run_id": case["id"], "scenario": case["arm"], "target_id": target.id,
                 "identity": identity, "status": "running", "cycles": [], "active_seconds": 0.0,
                 "phase": None, "usage_incomplete": False, "timing_incomplete": False,
                 "model": config.provider.model, "terminal_reason": None}
    database = Database(settings.database_path)
    if database.get_run(case["id"]) is None:
        database.start_run(case["id"], manifest, request, case["arm"], settings.experiment_id)
    last_tick = time.perf_counter()

    def checkpoint():
        nonlocal last_tick
        now = time.perf_counter()
        state["active_seconds"] += now - last_tick
        last_tick = now
        write_json(path, state)

    def boundary():
        checkpoint()
        if stop.is_set():
            raise Paused()
        if known_tokens(state) >= config.token_budget or state["active_seconds"] >= config.active_seconds_budget:
            raise BudgetReached()

    def generate(cycle, stage, prompt, schema):
        if stage not in cycle["calls"]:
            boundary()
            call = {"status": "in_flight", "started_at": datetime.now(UTC).isoformat(),
                    "prompt": prompt, "schema": schema.model_json_schema()}
            cycle["calls"][stage] = call
            state["phase"] = stage
            checkpoint()
            try:
                response = provider.generate(prompt, schema, config.provider.model)
                call.update(status="completed", response=response.raw, metadata=response.metadata)
            except ProviderError as exc:
                metadata = exc.metadata or {}
                call.update(status="rejected" if metadata.get("failure_origin") == "model_output" else "blocked",
                            response=exc.raw, metadata=metadata, error=str(exc))
            except Exception as exc:
                call.update(status="blocked", response={}, metadata={"usage_incomplete": True}, error=type(exc).__name__)
            state["phase"] = None
            metadata = call["metadata"]
            counts = normalize_usage(metadata.get("usage", {}))
            if metadata.get("usage_incomplete") or counts["input_tokens"] is None or counts["output_tokens"] is None:
                state["usage_incomplete"] = True
                call.update(status="blocked", error="complete token accounting unavailable")
            checkpoint()
        call = cycle["calls"][stage]
        if call["status"] == "blocked":
            raise _Blocked(call["error"])
        if call["status"] == "rejected":
            raise ValueError(call["error"])
        return schema.model_validate(call["response"])

    state["status"] = "running"
    try:
        while True:
            if not state["cycles"] or state["cycles"][-1]["status"] != "running":
                boundary()
                if len(state["cycles"]) >= config.max_cycles:
                    raise BudgetReached()
                state["cycles"].append({"number": len(state["cycles"]) + 1, "status": "running", "calls": {}})
            cycle = state["cycles"][-1]
            feedback = repair_feedback(state["cycles"][:-1], config)
            arm = case["arm"]
            structured = arm.startswith("structured_ir")
            try:
                if arm != "direct":
                    if "intermediate" not in cycle:
                        reusable = next((item for item in reversed(state["cycles"][:-1]) if "intermediate" in item and
                            (not structured or arm == "structured_ir_no_validation" or
                             item.get("semantic_validation", {}).get("status") == "passed")), None)
                        if reusable and not arm.endswith("no_reuse"):
                            cycle["intermediate"] = reusable["intermediate"]
                            cycle["intermediate_reused_from"] = reusable.get("intermediate_reused_from", reusable["number"])
                            if "semantic_validation" in reusable:
                                cycle["semantic_validation"] = dict(reusable["semantic_validation"], duration_seconds=0.0,
                                    cases_run=0, details={"reused_from_cycle": cycle["intermediate_reused_from"]})
                        else:
                            value = generate(cycle, "intermediate", stage_prompt("intermediate", arm, source, manifest,
                                             target, None, feedback), ExecutableKernelIR if structured else Hint)
                            cycle["intermediate"] = value.model_dump(mode="json") if structured else value.intermediate
                        checkpoint()
                    if structured and arm != "structured_ir_no_validation":
                        if "semantic_validation" not in cycle:
                            boundary()
                            state["phase"] = "semantic_validation"
                            checkpoint()
                            started = time.perf_counter()
                            outcome = validate_ir(ExecutableKernelIR.model_validate(cycle["intermediate"]), manifest)
                            cycle["semantic_validation"] = StageResult(status="passed" if outcome.passed else "failed",
                                correct=outcome.passed, engine="kernel-ir-numpy", representation="semantic_ir",
                                message=outcome.message, duration_seconds=time.perf_counter() - started).model_dump(mode="json")
                            state["phase"] = None
                            checkpoint()
                        if cycle["semantic_validation"]["status"] != "passed":
                            raise ValueError("IR validation failed: " + cycle["semantic_validation"]["message"])
                if "bundle" not in cycle:
                    bundle = generate(cycle, "code", stage_prompt("code", arm, source, manifest, target,
                                      cycle.get("intermediate"), feedback), CodeBundle)
                    cycle["bundle"] = bundle.model_dump(mode="json")
                    checkpoint()
                bundle = CodeBundle.model_validate(cycle["bundle"])
                _validate_bundle(bundle, target)
                if bundle.unsupported_operations:
                    raise ValueError("unsupported operations: " + str(bundle.unsupported_operations))
                if "candidate" not in cycle:
                    boundary()
                    state["phase"] = "backend_validation"
                    checkpoint()
                    candidate = _test_candidate(compiler, database, settings, request, state, cycle, bundle, manifest, path)
                    state["phase"] = None
                    cycle["status"] = "passed" if candidate.evaluation.validation.policy_met else "failed"
                    checkpoint()
                else:
                    cycle["status"] = "passed" if cycle["candidate"]["evaluation"]["validation"]["policy_met"] else "failed"
            except ValueError as exc:
                state["phase"] = None
                cycle.update(status="failed", feedback={"error": bounded(str(exc), config.feedback_chars)})
            checkpoint()
            if cycle["status"] == "passed":
                state.update(status="completed", terminal_reason="solved")
                break
    except Paused:
        state.update(status="paused", terminal_reason="graceful_pause")
    except BudgetReached:
        state.update(status="failed", terminal_reason="budget_exhausted")
    except Exception as exc:
        state.update(status="blocked", terminal_reason=bounded(str(exc), config.feedback_chars))
        if state["phase"] is not None:
            state["phase"] = None
    finally:
        checkpoint()
        database.finish_run(case["id"], state["status"], state, duration_seconds=state["active_seconds"])
        database.close()
    return state
