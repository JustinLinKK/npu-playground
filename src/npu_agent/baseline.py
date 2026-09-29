from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from pathlib import Path

from .artifacts import bundle_hash, materialize_bundle, store_artifact
from .compilers import Compiler, DockerCompiler
from .config import Settings
from .database import Database
from .mcts import evaluate_compile, select_winner
from .validation import policy_exit_code
from .models import (
    Backend,
    BaselineResult,
    BaselineTargetResult,
    Candidate,
    CodeBundle,
    CompileResult,
    KernelManifest,
    TargetProfile,
    TranslationRequest,
)
from .providers import ProviderError, StructuredProvider, create_provider


def _prompt(source: str, manifest: KernelManifest, target: TargetProfile) -> str:
    if target.backend == Backend.AMD_XDNA2:
        contract = (
            "Return exactly design.py and kernel.cc. design.py must accept --dev npu2, --emit-mlir, "
            "--xclbin-path, and --insts-path, target AIE2P/npu2, and use kernel.cc."
        )
    else:
        contract = (
            "Return exactly model.py. It must define build_model(manifest=None) and return an ov.Model "
            "whose input and output tensor names match the manifest arguments; explicitly set get_tensor().set_names."
        )
    return f"""Translate this GPU kernel directly to the requested NPU backend in one attempt.
Return only the requested structured CodeBundle. Do not ask questions. If the semantics cannot be represented,
return no files and list the unsupported operations. {contract}

Target profile:
{target.model_dump_json(indent=2)}

Manifest:
{manifest.model_dump_json(indent=2)}

Untrusted source begins:
---
{source}
---
Untrusted source ends.
"""


def _validate_bundle(bundle: CodeBundle, target: TargetProfile) -> None:
    if bundle.backend != target.backend:
        raise ValueError(f"bundle backend {bundle.backend.value} does not match {target.backend.value}")
    if bundle.unsupported_operations and bundle.files:
        raise ValueError("unsupported bundle must not also claim an implementation")


def _candidate_id(run_id: str, target: TargetProfile, bundle: CodeBundle) -> str:
    value = f"{run_id}:{target.id}:baseline:{bundle_hash(bundle)}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _compile_once(
    settings: Settings,
    database: Database,
    compiler: Compiler,
    candidate: Candidate,
    manifest: KernelManifest,
    target: TargetProfile,
) -> CompileResult:
    candidate_dir = settings.runs_path / candidate.run_id / target.id / "candidate"
    output_dir = settings.runs_path / candidate.run_id / target.id / "compile"
    try:
        materialize_bundle(candidate.bundle, candidate_dir, settings.max_generated_bytes)
        if candidate.bundle.unsupported_operations:
            result = CompileResult(
                success=False,
                exit_code=2,
                stderr=f"unsupported operations: {candidate.bundle.unsupported_operations}",
                compiler_fingerprint=f"{target.id}:{target.compiler_version}",
            )
        else:
            result = compiler.compile(candidate_dir, output_dir, manifest, target)
    except Exception as exc:
        result = CompileResult(
            success=False,
            exit_code=1,
            stderr=f"artifact or compiler boundary rejected candidate: {exc}",
            compiler_fingerprint=f"{target.id}:{target.compiler_version}",
        )
    database.add_compile_attempt(candidate.id, 0, result)
    for kind, raw_path in result.artifacts.items():
        path = Path(raw_path)
        stored, digest = store_artifact(path, settings.runs_path.parent / "artifacts")
        database.add_artifact(candidate.id, kind, str(stored), digest, {"bytes": path.stat().st_size})
    return result


def _load_inputs(request: TranslationRequest) -> tuple[KernelManifest, str]:
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
    return manifest, source


def _translate_one_shot_sync(
    request: TranslationRequest,
    settings: Settings,
    provider: StructuredProvider,
    compiler: Compiler,
    database: Database,
) -> BaselineResult:
    manifest, source = _load_inputs(request)
    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    database.start_run(run_id, manifest, request, "one_shot", settings.experiment_id)
    target_results: list[BaselineTargetResult] = []
    for target in request.targets:
        target_started = time.perf_counter()
        candidate: Candidate | None = None
        try:
            response = provider.generate(_prompt(source, manifest, target), CodeBundle, request.model)
            database.add_agent_call(run_id, "baseline", response.metadata, response.raw, target.id)
            bundle = CodeBundle.model_validate(response.value)
            _validate_bundle(bundle, target)
            candidate = Candidate(
                id=_candidate_id(run_id, target, bundle),
                run_id=run_id,
                target_id=target.id,
                label="one-shot",
                rationale="Direct one-shot source-to-target baseline",
                bundle=bundle,
            )
            database.add_candidate(candidate)
            compile_result = _compile_once(settings, database, compiler, candidate, manifest, target)
            candidate.evaluation = evaluate_compile(candidate.id, compile_result, request.validation_policy, target.id)
            database.add_evaluation(candidate.evaluation)
            accepted = select_winner([candidate], request.validation_policy) is not None
            status = "completed" if accepted else "blocked" if policy_exit_code(candidate.evaluation.validation) == 3 else "failed"
            error = None if accepted else compile_result.stderr[-4000:] or "correctness validation failed"
        except Exception as exc:
            if isinstance(exc, ProviderError) and exc.metadata:
                database.add_agent_call(run_id, "baseline", exc.metadata, exc.raw, target.id)
            empty = CodeBundle(backend=target.backend, files=[], unsupported_operations=["provider_output_invalid"])
            candidate = Candidate(
                id=_candidate_id(run_id, target, empty),
                run_id=run_id,
                target_id=target.id,
                label="one-shot-invalid",
                rationale="The only provider response was unusable",
                bundle=empty,
            )
            database.add_candidate(candidate)
            compile_result = CompileResult(
                success=False,
                exit_code=2,
                stderr=f"provider or schema boundary rejected baseline output: {exc}",
                compiler_fingerprint=f"{target.id}:{target.compiler_version}",
            )
            database.add_compile_attempt(candidate.id, 0, compile_result)
            candidate.evaluation = evaluate_compile(candidate.id, compile_result, request.validation_policy, target.id)
            database.add_evaluation(candidate.evaluation)
            status = "failed"
            error = str(exc)
        target_results.append(
            BaselineTargetResult(
                target=target,
                candidate=candidate,
                status=status,
                error=error,
                duration_seconds=time.perf_counter() - target_started,
            )
        )
    completed = sum(item.status == "completed" for item in target_results)
    status = "completed" if completed == len(target_results) else "partial" if completed else "blocked" if all(item.status == "blocked" for item in target_results) else "failed"
    report_path = settings.runs_path / run_id / "report.json"
    result = BaselineResult(
        run_id=run_id,
        kernel=manifest.name,
        targets=target_results,
        status=status,
        report_path=str(report_path),
        duration_seconds=time.perf_counter() - started,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    database.finish_run(run_id, status, result.model_dump(mode="json"), duration_seconds=result.duration_seconds)
    return result


async def translate_one_shot(
    request: TranslationRequest,
    settings: Settings | None = None,
    *,
    provider: StructuredProvider | None = None,
    compiler: Compiler | None = None,
) -> BaselineResult:
    selected = settings or Settings(provider=request.provider, model=request.model)
    selected.validation_policy = request.validation_policy
    selected.provider = request.provider
    selected.model = request.model
    selected.ensure_directories()
    database = Database(selected.database_path)
    model_provider = provider or create_provider(
        selected.provider, selected.repository_path, selected.model, selected.provider_timeout_seconds
    )
    compile_backend = compiler or DockerCompiler(selected)
    try:
        return await asyncio.to_thread(
            _translate_one_shot_sync, request, selected, model_provider, compile_backend, database
        )
    finally:
        database.close()
