from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import operator
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .artifacts import bundle_hash, materialize_bundle, store_artifact
from .compilers import Compiler, DockerCompiler
from .config import Settings
from .database import Database, normalize_error
from .ir import validate_ir
from .validation import policy_exit_code
from .mcts import evaluate_compile, select_winner
from .models import (
    Backend,
    Candidate,
    CodeBundle,
    CompileResult,
    DebugResponse,
    Evaluation,
    KernelIR,
    KernelManifest,
    ProposalSet,
    TargetProfile,
    TargetResult,
    TranslationRequest,
    TranslationResult,
)
from .providers import ProviderError, ProviderResponse, StructuredProvider, create_provider


class WorkflowState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    manifest: dict[str, Any]
    source: str
    kernel_ir: dict[str, Any]
    ir_validated: bool
    target: dict[str, Any]
    target_results: Annotated[list[dict[str, Any]], operator.add]
    result: dict[str, Any]
    started_at_ns: int


class TargetSearchState(TypedDict, total=False):
    run_id: str
    request: TranslationRequest
    manifest: KernelManifest
    source: str
    kernel_ir: KernelIR
    target: TargetProfile
    memory: dict[str, list[dict[str, Any]]]
    tree: dict[str, Any]
    selected: str
    pending: list[Candidate]
    evaluated: list[Candidate]
    all_candidates: list[Candidate]
    round_index: int
    target_result: TargetResult


@dataclass(slots=True)
class Services:
    settings: Settings
    database: Database
    provider: StructuredProvider
    compiler: Compiler
    provider_cache: dict[str, StructuredProvider] | None = None

    def provider_for(self, role: str) -> StructuredProvider:
        provider_name = self.settings.role_providers.get(role)
        if not provider_name or provider_name == self.provider.name:
            return self.provider
        if self.provider_cache is None:
            self.provider_cache = {}
        if provider_name not in self.provider_cache:
            self.provider_cache[provider_name] = create_provider(
                provider_name,
                self.settings.repository_path,
                self.settings.role_models.get(role) or self.settings.model,
                self.settings.provider_timeout_seconds,
            )
        return self.provider_cache[provider_name]


def _json(value: Any) -> str:
    if isinstance(value, BaseException):
        return str(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, indent=2)


def _candidate_id(
    run_id: str,
    target_id: str,
    parent_id: str | None,
    label: str,
    bundle: CodeBundle,
    debug_attempt: int = 0,
) -> str:
    payload = f"{run_id}:{target_id}:{parent_id}:{label}:{debug_attempt}:{bundle_hash(bundle)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _new_search_tree(candidate: Candidate, branching_factor: int) -> dict[str, Any]:
    reward = candidate.evaluation.reward if candidate.evaluation else 0.0
    return {
        "root_id": candidate.id,
        "branching_factor": branching_factor,
        "nodes": {
            candidate.id: {
                "candidate": candidate.model_dump(mode="json"),
                "parent_id": None,
                "children": [],
                "visits": 1,
                "value_sum": reward,
            }
        },
    }


def _select_tree_node(tree: dict[str, Any]) -> str:
    current_id = tree["root_id"]
    while len(tree["nodes"][current_id]["children"]) >= tree["branching_factor"]:
        parent = tree["nodes"][current_id]

        def ucb1(child_id: str) -> tuple[float, str]:
            child = tree["nodes"][child_id]
            if child["visits"] == 0:
                return (float("inf"), child_id)
            score = child["value_sum"] / child["visits"]
            score += (2.0 * max(0.0, math.log(parent["visits"] + 1)) / child["visits"]) ** 0.5
            return (score, child_id)

        current_id = max(parent["children"], key=ucb1)
    return current_id


def _add_tree_children(tree: dict[str, Any], parent_id: str, candidates: list[Candidate]) -> dict[str, Any]:
    updated = copy.deepcopy(tree)
    for candidate in candidates:
        reward = candidate.evaluation.reward if candidate.evaluation else 0.0
        updated["nodes"][candidate.id] = {
            "candidate": candidate.model_dump(mode="json"),
            "parent_id": parent_id,
            "children": [],
            "visits": 1,
            "value_sum": reward,
        }
        updated["nodes"][parent_id]["children"].append(candidate.id)
        current_id: str | None = parent_id
        while current_id is not None:
            node = updated["nodes"][current_id]
            node["visits"] += 1
            node["value_sum"] += reward
            current_id = node["parent_id"]
    return updated


def _recorded_generate(
    services: Services,
    run_id: str,
    role: str,
    prompt: str,
    response_model: type[Any],
    model: str | None,
    validate: Callable[[Any], None] | None = None,
    correction_attempts: int = 2,
    target_id: str | None = None,
) -> Any:
    current_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(correction_attempts + 1):
        try:
            provider = services.provider_for(role)
            selected_model = services.settings.role_models.get(role) or model
            response: ProviderResponse = provider.generate(current_prompt, response_model, selected_model)
            services.database.add_agent_call(run_id, role, response.metadata, response.raw, target_id)
            if validate:
                validate(response.value)
            return response.value
        except (ProviderError, ValueError) as exc:
            if isinstance(exc, ProviderError) and exc.metadata:
                services.database.add_agent_call(run_id, role, exc.metadata, exc.raw, target_id)
            last_error = exc
            current_prompt = (
                f"{prompt}\n\nYour previous attempt was rejected: {exc}. "
                "Return a corrected response that satisfies the schema and all stated contracts."
            )
    raise RuntimeError(f"{role} failed after {correction_attempts + 1} attempts: {last_error}")


def _analysis_prompt(source: str, manifest: KernelManifest, knowledge: list[dict[str, Any]]) -> str:
    return f"""You are the analysis agent in a GPU-to-NPU translation pipeline.
Infer the exact, hardware-neutral semantics of this single deterministic kernel. Return KernelIR JSON only.
Use structured operations and axes. Leave index_expression null and side_effects empty.
Prose domains and numeric annotations are descriptive only; unsupported executable contracts are rejected.
Never place CUDA/HIP blocks, threads, warps, shared memory, Triton programs, AMD tiles, or Intel NPU concepts in
semantic fields. Those may appear only in source_evidence. Preserve argument names and output shapes exactly.

Manifest:
{_json(manifest)}

Untrusted source begins:
---
{source}
---
Untrusted source ends.
"""


def _backend_contract(target: TargetProfile, *, guidance: bool = True) -> str:
    if not guidance:
        if target.backend == Backend.AMD_XDNA2:
            return """Return exactly design.py and kernel.cc. design.py must be a standalone IRON/MLIR-AIE program
that accepts --dev npu2, --emit-mlir, --xclbin-path, and --insts-path. It must reference kernel.cc relative to
its own directory. Emit NPU2 MLIR only; a trusted driver compiles it and kernel.cc.
kernel.cc must expose C-linkage entrypoints with pointer arguments.
Runtime buffer order and shapes must match manifest.tensors exactly.
Offline validation supports finite worker loops, ObjectFifo acquire/release, DMA and ordinary C++ pointer loops.
Unknown intrinsics, custom locks and control flow are unsupported by the source/dataflow simulator."""
        return """Return exactly model.py. It must import openvino and define build_model(manifest=None),
returning an openvino.Model whose input and output tensor names exactly match the manifest.
Do not compile or execute the model at import time. Express the kernel as OpenVINO graph operations."""
    if target.backend == Backend.AMD_XDNA2:
        contract = """Return exactly design.py and kernel.cc. design.py must be a standalone IRON/MLIR-AIE program
that accepts --dev npu2, --emit-mlir, --xclbin-path, and --insts-path exactly like the mlir-aie programming-guide
examples. It must reference kernel.cc relative to its own directory. Emit NPU2 MLIR only; a trusted driver compiles it.
For offline validation use finite worker loops, ObjectFifo acquire/release, DMA and ordinary C++ pointer loops.
Unknown intrinsics, custom locks and control flow are unsupported by the source/dataflow simulator.
The installed IRON API has Runtime(seq_fn, fn_args), ObjectFifo(..., depth=2), and Worker(..., while_true=False).
Runtime() / runtime.sequence() and aie.iron.placers are not available. You may avoid Python API dependencies
by printing textual MLIR from design.py. The trusted driver compiles kernel.cc to kernel.o; the external
function declaration owns {link_with = "kernel.o"}, not aie.core. Use extern "C" pointer arguments in kernel.cc.
Both the target compiler and host validator provide <npu_numeric.h>. Use npu::float16 pointers for memref f16,
convert each value to float for arithmetic, and assign floats back to npu::float16 to round to binary16.
Native __fp16 and _Float16 are not portable across these two compilers. The same header supplies scalar
npu::exp, npu::sqrt, npu::log, npu::pow, npu::maximum and npu::minimum for ordinary finite floating-point inputs.
Here is valid transport syntax for one 16-element input/output (replace shapes, symbols, topology and the
external compute signature for this manifest; this example supplies no arithmetic implementation):
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @input(%shim, {%core}, 1 : i32) : !aie.objectfifo<memref<16xf32>>
    aie.objectfifo @output(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<16xf32>>
    func.func private @compute(memref<16xf32>, memref<16xf32>) attributes {link_with = "kernel.o"}
    %worker = aie.core(%core) {
      %i = aie.objectfifo.acquire @input(Consume, 1) : !aie.objectfifosubview<memref<16xf32>>
      %iv = aie.objectfifo.subview.access %i[0] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
      %o = aie.objectfifo.acquire @output(Produce, 1) : !aie.objectfifosubview<memref<16xf32>>
      %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
      func.call @compute(%iv, %ov) : (memref<16xf32>, memref<16xf32>) -> ()
      aie.objectfifo.release @input(Consume, 1)
      aie.objectfifo.release @output(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%x: memref<16xf32>, %y: memref<16xf32>) {
      %tx = aiex.dma_configure_task_for @input {
        aie.dma_bd(%x : memref<16xf32> offset = 0 len = 16 sizes = [1, 1, 1, 16] strides = [0, 0, 0, 1])
        aie.end
      }
      %ty = aiex.dma_configure_task_for @output {
        aie.dma_bd(%y : memref<16xf32> offset = 0 len = 16 sizes = [1, 1, 1, 16] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tx)
      aiex.dma_start_task(%ty)
      aiex.dma_await_task(%ty)
      aiex.dma_free_task(%tx)
      aiex.dma_free_task(%ty)
    }
  }
}
Runtime buffer order and shapes must match manifest.tensors exactly. Use finite scf.for loops for tiles,
matching acquire/release counts and DMA lengths. Size FIFO objects to fit tile memory; distribute DMA channels
across shim tiles or explicitly link/join FIFOs when needed. Implement the source arithmetic yourself."""
    else:
        contract = """Return exactly model.py. It must import openvino and define build_model(manifest=None),
returning an openvino.Model whose input and output tensor names exactly match the manifest.
For a Node, set names with node.output(0).get_tensor().set_names({name}); Node has no get_tensor method.
An Output already supports output.get_tensor().set_names({name}); friendly names alone are insufficient. Do not
compile or execute the model at import time. Express the kernel as OpenVINO graph operations."""
    return contract


def _coding_prompt(
    kernel_ir: KernelIR,
    manifest: KernelManifest,
    target: TargetProfile,
    knowledge: list[dict[str, Any]],
) -> str:
    contract = _backend_contract(target)
    return f"""You are the NPU coding agent. Generate a complete backend artifact bundle from validated semantic IR.
If the semantics cannot be represented honestly, return no files and enumerate unsupported_operations.
{contract}

Target profile:
{_json(target)}

Manifest:
{_json(manifest)}

Validated IR:
{_json(kernel_ir)}

"""


def _optimization_prompt(
    candidate: Candidate,
    kernel_ir: KernelIR,
    target: TargetProfile,
    knowledge: list[dict[str, Any]],
    lessons: list[dict[str, Any]],
    branching_factor: int,
) -> str:
    return f"""You are the hardware optimization agent. Produce exactly {branching_factor} distinct complete
replacement bundles for the current candidate. Preserve semantic behavior and the backend file contract. Each
proposal needs a concrete rationale, expected benefit, assumptions, and risks. Favor target-specific tiling,
vectorization, data movement, fusion, and legal precision choices. Do not merely rename variables.

Target: {_json(target)}
Validated IR: {_json(kernel_ir)}
Current candidate: {_json(candidate.bundle)}
"""


def _debug_prompt(
    candidate: Candidate,
    result: CompileResult,
    kernel_ir: KernelIR,
    target: TargetProfile,
    knowledge: list[dict[str, Any]],
    lessons: list[dict[str, Any]],
) -> str:
    return f"""You are the backend debug agent. Diagnose the recorded compiler, host, or dataflow failure and return a complete
corrected bundle. Preserve the validated semantics and backend file contract. Fix only causes supported by the
stage evidence; do not hide errors or replace the operation with a constant.

Target: {_json(target)}
Validated IR: {_json(kernel_ir)}
Candidate: {_json(candidate.bundle)}
Compiler exit: {result.exit_code}
Compiler stdout: {result.stdout[-12000:]}
Compiler stderr: {result.stderr[-12000:]}
"""


def _check_bundle(bundle: CodeBundle, target: TargetProfile) -> None:
    if bundle.backend != target.backend:
        raise ValueError(f"bundle backend {bundle.backend.value} does not match {target.backend.value}")
    if bundle.unsupported_operations and bundle.files:
        raise ValueError("unsupported bundle must not also claim an implementation")


def _persist_artifacts(services: Services, candidate: Candidate, result: CompileResult) -> None:
    for kind, raw_path in result.artifacts.items():
        path = Path(raw_path)
        store_path, digest = store_artifact(path, services.settings.runs_path.parent / "artifacts")
        services.database.add_artifact(
            candidate.id,
            kind,
            str(store_path),
            digest,
            {"bytes": path.stat().st_size},
        )


def _compile_once(
    services: Services,
    candidate: Candidate,
    manifest: KernelManifest,
    target: TargetProfile,
    attempt: int,
) -> CompileResult:
    candidate_dir = services.settings.runs_path / candidate.run_id / target.id / "candidates" / candidate.id
    output_dir = services.settings.runs_path / candidate.run_id / target.id / "compiles" / candidate.id / str(attempt)
    try:
        materialize_bundle(candidate.bundle, candidate_dir, services.settings.max_generated_bytes)
        if candidate.bundle.unsupported_operations:
            result = CompileResult(
                success=False,
                exit_code=2,
                stderr=f"unsupported operations: {candidate.bundle.unsupported_operations}",
                compiler_fingerprint=f"{target.id}:{target.compiler_version}",
            )
        else:
            result = services.compiler.compile(candidate_dir, output_dir, manifest, target)
    except Exception as exc:
        result = CompileResult(
            success=False,
            exit_code=1,
            stderr=f"artifact or compiler boundary rejected candidate: {exc}",
            compiler_fingerprint=f"{target.id}:{target.compiler_version}",
        )
    services.database.add_compile_attempt(candidate.id, attempt, result)
    return result


def _evaluate_with_debug(
    services: Services,
    candidate: Candidate,
    manifest: KernelManifest,
    kernel_ir: KernelIR,
    target: TargetProfile,
    request: TranslationRequest,
    memory: dict[str, list[dict[str, Any]]],
) -> tuple[Candidate, list[Candidate]]:
    all_candidates: list[Candidate] = []
    current = candidate
    first_failure: CompileResult | None = None
    semantic_failure = False
    last_diagnosis = ""
    last_fix_summary = ""
    for attempt in range(request.debug_retries + 1):
        services.database.add_candidate(current)
        all_candidates.append(current)
        result = _compile_once(services, current, manifest, target, attempt)
        if result.validation:
            from .models import StageResult
            semantic_started = time.perf_counter()
            semantic = validate_ir(kernel_ir, manifest)
            result.validation.stages["semantic_validation"] = StageResult(
                status="passed" if semantic.passed else "failed", correct=semantic.passed,
                engine="kernel-ir-numpy", representation="semantic_ir", cases_run=12 if semantic.passed else 0,
                duration_seconds=time.perf_counter() - semantic_started, message=semantic.message,
                details={"prose_metadata": "descriptive_only"})
        evaluation = evaluate_compile(current.id, result, request.validation_policy, target.id)
        current.evaluation = evaluation
        services.database.add_evaluation(evaluation)
        if select_winner([current], request.validation_policy) is not None:
            _persist_artifacts(services, current, result)
            if first_failure is not None and not semantic_failure:
                services.database.add_lesson(
                    target,
                    result.compiler_fingerprint,
                    [operation.op for operation in kernel_ir.operations],
                    normalize_error(first_failure.stderr),
                    last_diagnosis or f"Compiler failure repaired after {attempt} debug attempt(s)",
                    last_fix_summary or current.rationale,
                    True,
                )
            return current, all_candidates
        if evaluation.validation and policy_exit_code(evaluation.validation) == 3:
            return current, all_candidates
        if first_failure is None:
            first_failure = result
            semantic_failure = result.success or evaluation.correctness is False or "does not match oracle" in result.stderr.lower()
        if attempt >= request.debug_retries:
            if first_failure is not None and not semantic_failure:
                services.database.add_lesson(
                    target,
                    result.compiler_fingerprint,
                    [operation.op for operation in kernel_ir.operations],
                    normalize_error(first_failure.stderr),
                    "Compilation remained unresolved after the configured debug budget",
                    "No verified fix",
                    False,
                )
            return current, all_candidates
        response: DebugResponse = _recorded_generate(
            services,
            candidate.run_id,
            "debug",
            _debug_prompt(
                current,
                result,
                kernel_ir,
                target,
                services.database.retrieve_knowledge(
                    "debug",
                    " ".join(
                        [*(operation.op for operation in kernel_ir.operations), result.stderr[-4000:]]
                    ),
                    manifest.dialect.value,
                    target.backend.value,
                    target.hardware,
                    target.id,
                    target.vendor,
                    result.compiler_fingerprint,
                ),
                services.database.retrieve_lessons(
                    target,
                    " ".join(
                        [*(operation.op for operation in kernel_ir.operations), result.stderr[-4000:]]
                    ),
                    compiler_fingerprint=result.compiler_fingerprint,
                ),
            ),
            DebugResponse,
            request.model,
            lambda value: _check_bundle(value.bundle, target),
            target_id=target.id,
        )
        last_diagnosis = response.diagnosis
        last_fix_summary = response.fix_summary
        repaired = Candidate(
            id=_candidate_id(
                candidate.run_id,
                target.id,
                current.id,
                f"debug-{attempt + 1}",
                response.bundle,
                attempt + 1,
            ),
            run_id=candidate.run_id,
            target_id=target.id,
            parent_id=current.id,
            depth=current.depth + 1,
            label=f"{candidate.label}-debug-{attempt + 1}",
            rationale=f"{response.diagnosis}\n{response.fix_summary}",
            bundle=response.bundle,
            debug_attempt=attempt + 1,
        )
        current = repaired
    raise AssertionError("unreachable debug loop")


def _build_target_graph(services: Services):
    builder = StateGraph(TargetSearchState)

    def retrieve_memory_node(state: TargetSearchState) -> dict[str, Any]:
        manifest = state["manifest"]
        target = state["target"]
        query = " ".join([manifest.name, manifest.oracle.operation, *manifest.constraints])
        return {
            "memory": {
                "coding": services.database.retrieve_knowledge(
                    "coding", query, manifest.dialect.value, target.backend.value, target.hardware,
                    target.id, target.vendor, f"{target.id}:{target.compiler_version}"
                ),
                "optimize": services.database.retrieve_knowledge(
                    "optimize", query, manifest.dialect.value, target.backend.value, target.hardware,
                    target.id, target.vendor, f"{target.id}:{target.compiler_version}"
                ),
                "debug": services.database.retrieve_knowledge(
                    "debug", query, manifest.dialect.value, target.backend.value, target.hardware,
                    target.id, target.vendor, f"{target.id}:{target.compiler_version}"
                ),
                "lessons": services.database.retrieve_lessons(target, query, compiler_fingerprint=services.compiler.fingerprint(target)
                    if hasattr(services.compiler, "fingerprint") else None),
            }
        }

    def draft_node(state: TargetSearchState) -> dict[str, Any]:
        target = state["target"]
        request = state["request"]
        bundle: CodeBundle = _recorded_generate(
            services,
            state["run_id"],
            "coding",
            _coding_prompt(
                state["kernel_ir"], state["manifest"], target, state["memory"].get("coding", [])
            ),
            CodeBundle,
            request.model,
            lambda value: _check_bundle(value, target),
            target_id=target.id,
        )
        candidate = Candidate(
            id=_candidate_id(state["run_id"], target.id, None, "draft", bundle),
            run_id=state["run_id"],
            target_id=target.id,
            label="draft",
            rationale="Initial backend lowering",
            bundle=bundle,
        )
        return {"pending": [candidate], "all_candidates": [], "round_index": 0}

    def compile_draft_node(state: TargetSearchState) -> dict[str, Any]:
        final, seen = _evaluate_with_debug(
            services,
            state["pending"][0],
            state["manifest"],
            state["kernel_ir"],
            state["target"],
            state["request"],
            state["memory"],
        )
        return {"tree": _new_search_tree(final, state["request"].branching_factor), "all_candidates": seen}

    def optimize_node(state: TargetSearchState) -> dict[str, Any]:
        tree = state["tree"]
        selected_id = _select_tree_node(tree)
        selected = Candidate.model_validate(tree["nodes"][selected_id]["candidate"])
        request = state["request"]
        query = " ".join(operation.op for operation in state["kernel_ir"].operations)
        current_lessons = services.database.retrieve_lessons(state["target"], query,
            compiler_fingerprint=services.compiler.fingerprint(state["target"]) if hasattr(services.compiler, "fingerprint") else None)

        def validate_proposals(value: ProposalSet) -> None:
            if len(value.proposals) != request.branching_factor:
                raise ValueError(f"expected exactly {request.branching_factor} proposals")
            labels = {proposal.label for proposal in value.proposals}
            if len(labels) != request.branching_factor:
                raise ValueError("proposal labels must be distinct")
            for proposal in value.proposals:
                _check_bundle(proposal.bundle, state["target"])

        proposal_set: ProposalSet = _recorded_generate(
            services,
            state["run_id"],
            "optimize",
            _optimization_prompt(
                selected,
                state["kernel_ir"],
                state["target"],
                state["memory"].get("optimize", []),
                current_lessons,
                request.branching_factor,
            ),
            ProposalSet,
            request.model,
            validate_proposals,
            target_id=state["target"].id,
        )
        pending = [
            Candidate(
                id=_candidate_id(
                    state["run_id"], state["target"].id, selected.id, proposal.label, proposal.bundle
                ),
                run_id=state["run_id"],
                target_id=state["target"].id,
                parent_id=selected.id,
                depth=selected.depth + 1,
                label=proposal.label,
                rationale=proposal.rationale,
                bundle=proposal.bundle,
            )
            for proposal in proposal_set.proposals
        ]
        return {"selected": selected_id, "pending": pending}

    def compile_proposals_node(state: TargetSearchState) -> dict[str, Any]:
        evaluated: list[Candidate] = []
        seen: list[Candidate] = []
        for candidate in state["pending"]:
            final, branch_seen = _evaluate_with_debug(
                services,
                candidate,
                state["manifest"],
                state["kernel_ir"],
                state["target"],
                state["request"],
                state["memory"],
            )
            evaluated.append(final)
            seen.extend(branch_seen)
        return {"evaluated": evaluated, "all_candidates": state["all_candidates"] + seen}

    def backprop_node(state: TargetSearchState) -> dict[str, Any]:
        tree = _add_tree_children(state["tree"], state["selected"], state["evaluated"])
        return {"tree": tree, "round_index": state["round_index"] + 1}

    def route_search(state: TargetSearchState) -> str:
        for candidate in state["all_candidates"]:
            validation = candidate.evaluation.validation if candidate.evaluation else None
            if validation and not validation.policy_met and any(
                stage.status == "blocked" and name in ("oracle_validation", "target_compile", "host_execution", "dataflow_simulation")
                or stage.status == "blocked" and name == "target_execution" and state["request"].validation_policy.value == "target-executed"
                for name, stage in validation.stages.items()
            ):
                return "select_winner"
        return "optimize" if state["round_index"] < state["request"].search_rounds else "select_winner"

    def select_winner_node(state: TargetSearchState) -> dict[str, Any]:
        winner = select_winner(state["all_candidates"], state["request"].validation_policy)
        blocked = any(c.evaluation and c.evaluation.validation and policy_exit_code(c.evaluation.validation) == 3
                      for c in state["all_candidates"])
        result = TargetResult(
            target=state["target"],
            winner=winner,
            candidates_evaluated=len(state["all_candidates"]),
            status="completed" if winner else "blocked" if blocked else "failed",
            error=None if winner else "no candidate met the requested validation policy",
        )
        return {"target_result": result}

    builder.add_node("retrieve_memory", retrieve_memory_node)
    builder.add_node("draft", draft_node)
    builder.add_node("compile_draft", compile_draft_node)
    builder.add_node("optimize", optimize_node)
    builder.add_node("compile_and_debug_proposals", compile_proposals_node)
    builder.add_node("backpropagate", backprop_node)
    builder.add_node("select_winner", select_winner_node)
    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "draft")
    builder.add_edge("draft", "compile_draft")
    builder.add_conditional_edges("compile_draft", route_search)
    builder.add_edge("optimize", "compile_and_debug_proposals")
    builder.add_edge("compile_and_debug_proposals", "backpropagate")
    builder.add_conditional_edges("backpropagate", route_search)
    builder.add_edge("select_winner", END)
    return builder.compile()


def _build_graph(services: Services):
    target_graph = _build_target_graph(services)
    builder = StateGraph(WorkflowState)

    def analyze_node(state: WorkflowState) -> dict[str, Any]:
        manifest = KernelManifest.model_validate(state["manifest"])
        request = TranslationRequest.model_validate(state["request"])
        if services.database.get_run(state["run_id"]) is None:
            services.database.start_run(
                state["run_id"], manifest, request, "agentic", services.settings.experiment_id
            )
        query = " ".join([manifest.name, manifest.oracle.operation, *manifest.constraints])
        knowledge = services.database.retrieve_knowledge("analysis", query, manifest.dialect.value)

        def validate_analysis(value: KernelIR) -> None:
            outcome = validate_ir(value, manifest)
            if not outcome.passed:
                raise ValueError(outcome.message)

        kernel_ir: KernelIR = _recorded_generate(
            services,
            state["run_id"],
            "analysis",
            _analysis_prompt(state["source"], manifest, knowledge),
            KernelIR,
            request.model,
            validate_analysis,
        )
        return {
            "kernel_ir": kernel_ir.model_dump(mode="json"),
            "started_at_ns": state.get("started_at_ns", time.time_ns()),
        }

    def verify_node(state: WorkflowState) -> dict[str, Any]:
        outcome = validate_ir(
            KernelIR.model_validate(state["kernel_ir"]), KernelManifest.model_validate(state["manifest"])
        )
        if not outcome.passed:
            raise RuntimeError(outcome.message)
        return {"ir_validated": True, "target_results": []}

    def route_targets(state: WorkflowState) -> list[Send]:
        request = TranslationRequest.model_validate(state["request"])
        return [Send("target_search", {**state, "target": target.model_dump(mode="json")}) for target in request.targets]

    def target_search_node(state: WorkflowState) -> dict[str, Any]:
        request = TranslationRequest.model_validate(state["request"])
        target_state: TargetSearchState = {
            "run_id": state["run_id"],
            "request": request,
            "manifest": KernelManifest.model_validate(state["manifest"]),
            "source": state["source"],
            "kernel_ir": KernelIR.model_validate(state["kernel_ir"]),
            "target": TargetProfile.model_validate(state["target"]),
        }
        completed = target_graph.invoke(target_state)
        return {"target_results": [completed["target_result"].model_dump(mode="json")]}

    def finalize_node(state: WorkflowState) -> dict[str, Any]:
        manifest = KernelManifest.model_validate(state["manifest"])
        target_results = [TargetResult.model_validate(item) for item in state["target_results"]]
        completed = sum(result.status == "completed" for result in target_results)
        status = "completed" if completed == len(target_results) else "partial" if completed else "blocked" if all(item.status == "blocked" for item in target_results) else "failed"
        report_path = services.settings.runs_path / state["run_id"] / "report.json"
        duration_seconds = max(0.0, (time.time_ns() - state["started_at_ns"]) / 1_000_000_000)
        result = TranslationResult(
            run_id=state["run_id"],
            kernel=manifest.name,
            ir=KernelIR.model_validate(state["kernel_ir"]),
            ir_validated=state["ir_validated"],
            targets=target_results,
            status=status,
            report_path=str(report_path),
            duration_seconds=duration_seconds,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        services.database.finish_run(
            state["run_id"], status, result.model_dump(mode="json"), duration_seconds=duration_seconds
        )
        return {"result": result.model_dump(mode="json")}

    builder.add_node("analysis_agent", analyze_node)
    builder.add_node("verify_ir", verify_node)
    builder.add_node("target_search", target_search_node)
    builder.add_node("finalize_report", finalize_node)
    builder.add_edge(START, "analysis_agent")
    builder.add_edge("analysis_agent", "verify_ir")
    builder.add_conditional_edges("verify_ir", route_targets, ["target_search"])
    builder.add_edge("target_search", "finalize_report")
    builder.add_edge("finalize_report", END)
    checkpoint_connection = sqlite3.connect(services.settings.database_path, check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_connection)
    graph = builder.compile(checkpointer=checkpointer)
    graph._npu_services = services  # type: ignore[attr-defined]
    graph._npu_checkpoint_connection = checkpoint_connection  # type: ignore[attr-defined]
    return graph


def build_workflow(
    settings: Settings | None = None,
    *,
    provider: StructuredProvider | None = None,
    compiler: Compiler | None = None,
    database: Database | None = None,
):
    selected = settings or Settings()
    selected.ensure_directories()
    db = database or Database(selected.database_path)
    model_provider = provider or create_provider(
        selected.provider, selected.repository_path, selected.model, selected.provider_timeout_seconds
    )
    compile_backend = compiler or DockerCompiler(selected)
    return _build_graph(Services(selected, db, model_provider, compile_backend))


async def translate(
    request: TranslationRequest,
    settings: Settings | None = None,
    *,
    provider: StructuredProvider | None = None,
    compiler: Compiler | None = None,
) -> TranslationResult:
    selected = settings or Settings(provider=request.provider, model=request.model)
    selected.validation_policy = request.validation_policy
    selected.provider = request.provider
    selected.model = request.model
    graph = build_workflow(selected, provider=provider, compiler=compiler)
    services: Services = graph._npu_services  # type: ignore[attr-defined]
    run_id = uuid.uuid4().hex
    source_path = Path(request.source_path)
    manifest_path = Path(request.manifest_path)
    if not source_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("source_path and manifest_path must reference regular files")
    manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8")
    expected_suffixes = {"cuda": {".cu"}, "triton": {".py"}, "hip": {".hip", ".cpp", ".cu"}}
    if source_path.suffix.lower() not in expected_suffixes[manifest.dialect.value]:
        raise ValueError(f"source suffix {source_path.suffix} does not match dialect {manifest.dialect.value}")
    if len(source.encode("utf-8")) > 1_000_000:
        raise ValueError("kernel source exceeds the 1 MB input limit")
    if manifest.entrypoint not in source:
        raise ValueError(f"entrypoint {manifest.entrypoint!r} was not found in source")
    initial: WorkflowState = {
        "run_id": run_id,
        "request": request.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json"),
        "source": source,
        "target_results": [],
        "started_at_ns": time.time_ns(),
    }
    try:
        state = await asyncio.to_thread(
            graph.invoke, initial, {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
        )
        return TranslationResult.model_validate(state["result"])
    except Exception as exc:
        if services.database.get_run(run_id):
            started_at_ns = initial["started_at_ns"]
            services.database.finish_run(
                run_id,
                "failed",
                error=str(exc),
                duration_seconds=max(0.0, (time.time_ns() - started_at_ns) / 1_000_000_000),
            )
        raise


def resume_run(graph: Any, run_id: str) -> TranslationResult:
    state = graph.invoke(None, {"configurable": {"thread_id": run_id}, "recursion_limit": 100})
    return TranslationResult.model_validate(state["result"])
