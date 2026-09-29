from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from threading import BoundedSemaphore, Event, local
from typing import Any

from .compilers import Compiler, DockerCompiler
from .config import TARGETS, Settings
from .database import Database
from .models import KernelManifest, TranslationRequest
from .providers import ExperimentCodexProvider, ExperimentOpenRouterProvider, ProviderError, StructuredProvider
from .reporting import build_report
from .scenarios import ABLATIONS, CASE_STUDY_PROTOCOL, EXPERIMENT_MODEL, GUIDANCE_MODEL, GUIDANCE_PROTOCOL, GUIDANCE_SCENARIOS, PROTOCOL_REVISION, REASONING_EFFORT, SCENARIOS, translate_scenario
from .validation import sha256


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_revision(repository: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or None


def discover_kernels(corpus: Path) -> dict[str, tuple[Path, Path, KernelManifest]]:
    kernels: dict[str, tuple[Path, Path, KernelManifest]] = {}
    for manifest_path in sorted(corpus.rglob("manifest.json")):
        manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        kernels[manifest.name] = (manifest_path.parent / manifest.source_file, manifest_path, manifest)
    if not kernels:
        raise FileNotFoundError(f"no manifest.json files found under {corpus}")
    return kernels


def _new_experiment(
    experiment_id: str,
    repository: Path,
    selected_kernels: list[str],
    target_ids: list[str],
    provider_name: str,
    model: str | None,
    corpus: Path,
    max_cycles: int,
    scenarios: list[str],
) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    for method in scenarios:
        for kernel in selected_kernels:
            for target_id in target_ids:
                key = f"{method}:{kernel}:{target_id}"
                target = TARGETS[target_id]
                cases[key] = {
                    "method": method,
                    "kernel": kernel,
                    "target_id": target_id,
                    "backend": target.backend.value,
                    "status": "pending",
                    "result_status": None,
                    "run_id": None,
                    "result_path": None,
                    "duration_seconds": None,
                    "error": None,
                }
    return {
        "schema_version": "2.0",
        "protocol_revision": PROTOCOL_REVISION,
        "experiment_id": experiment_id,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "repository": str(repository),
        "corpus": str(corpus),
        "provider": provider_name,
        "model": model,
        "reasoning_effort": "none" if provider_name == "openrouter" else REASONING_EFFORT,
        "knowledge_enabled": False,
        "validation_policy": "offline-validated",
        "kernels": selected_kernels,
        "targets": target_ids,
        "max_cycles": max_cycles,
        "scenarios": scenarios,
        "reproducibility": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_revision": _git_revision(repository),
            "target_profiles": {key: TARGETS[key].model_dump(mode="json") for key in target_ids},
        },
        "cases": cases,
    }


def _resolve_resume(runs_dir: Path, resume: str) -> Path:
    direct = Path(resume)
    root = direct if direct.is_dir() else runs_dir / resume
    if not (root / "experiment.json").is_file():
        raise FileNotFoundError(f"experiment not found: {root}")
    return root.resolve()


def _provider_metadata(provider: StructuredProvider) -> dict[str, Any]:
    metadata = {
        "name": provider.name,
        "executable": getattr(provider, "executable", None),
        "version": getattr(provider, "version", None),
    }
    if isinstance(provider, ExperimentOpenRouterProvider):
        metadata["settings"] = provider.public_config
    return metadata


def run_experiment(
    *,
    repository: Path,
    corpus: Path,
    runs_dir: Path,
    kernel_names: list[str] | None = None,
    target_ids: list[str] | None = None,
    provider_name: str | None = None,
    model: str | None = None,
    resume: str | None = None,
    provider: StructuredProvider | None = None,
    compiler: Compiler | None = None,
    validation_policy: str = "offline-validated",
    max_cycles: int | None = None,
    case_workers: int | None = None,
    max_compiler_jobs: int | None = None,
    compiler_cpus: int | None = None,
    compiler_memory: str | None = None,
    scenarios: list[str] | None = None,
    protocol_revision: str | None = None,
    order_seed: int | None = None,
) -> Path:
    if provider_name not in (None, "codex-cli", "openrouter") or model not in (None, EXPERIMENT_MODEL, GUIDANCE_MODEL) or validation_policy != "offline-validated":
        raise ValueError("experiments require a protocol-pinned provider/model and offline-validated")
    if provider is not None and provider.name != "fake" and not isinstance(provider, (ExperimentCodexProvider, ExperimentOpenRouterProvider)):
        raise ValueError("live experiments require a pinned experiment provider")
    if max_cycles is not None and (isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles < 1):
        raise ValueError("max_cycles must be a positive integer")
    if protocol_revision not in (None, PROTOCOL_REVISION, CASE_STUDY_PROTOCOL, GUIDANCE_PROTOCOL):
        raise ValueError("unknown protocol revision")
    if order_seed is not None and (isinstance(order_seed, bool) or not isinstance(order_seed, int)):
        raise ValueError("order_seed must be an integer")
    if scenarios is not None and (not scenarios or len(set(scenarios)) != len(scenarios) or set(scenarios) - set(SCENARIOS + ABLATIONS + GUIDANCE_SCENARIOS)):
        raise ValueError("scenarios must be a nonempty unique subset of the supported scenarios")
    execution = {"case_workers": case_workers, "max_compiler_jobs": max_compiler_jobs,
                 "compiler_cpus": compiler_cpus, "compiler_memory": compiler_memory}
    defaults = {"case_workers": 1, "max_compiler_jobs": 2, "compiler_cpus": 4, "compiler_memory": "8g"}
    for name in ("case_workers", "max_compiler_jobs", "compiler_cpus"):
        value = execution[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if compiler_memory is not None and (not isinstance(compiler_memory, str) or not compiler_memory.strip()):
        raise ValueError("compiler_memory must be a nonempty Docker memory limit")
    repository = repository.resolve()
    corpus = corpus.resolve()
    runs_dir = runs_dir.resolve()
    if resume:
        root = _resolve_resume(runs_dir, resume)
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        if experiment.get("schema_version") != "2.0":
            raise ValueError("historical experiments remain readable; start a new version-2 experiment to run this protocol")
        recorded_protocol = experiment.get("protocol_revision")
        if recorded_protocol not in (PROTOCOL_REVISION, CASE_STUDY_PROTOCOL, GUIDANCE_PROTOCOL) or (
                protocol_revision is not None and protocol_revision != recorded_protocol):
            raise ValueError("protocol revision changed; historical reports remain readable, but start a new experiment")
        protocol_revision = recorded_protocol
        expected_model = GUIDANCE_MODEL if protocol_revision == GUIDANCE_PROTOCOL else EXPERIMENT_MODEL
        expected_provider = "openrouter" if protocol_revision == GUIDANCE_PROTOCOL else "codex-cli"
        reasoning_effort = "none" if protocol_revision == GUIDANCE_PROTOCOL else REASONING_EFFORT
        if provider_name is not None and provider_name != expected_provider:
            raise ValueError("provider cannot change on resume")
        provider_name = expected_provider
        if model is not None and model != expected_model:
            raise ValueError("model cannot change on resume")
        model = expected_model
        if order_seed is not None and order_seed != experiment.get("order_seed"):
            raise ValueError("order_seed cannot change on resume")
        order_seed = experiment.get("order_seed")
        if scenarios is not None and scenarios != experiment["scenarios"]:
            raise ValueError("scenarios cannot change on resume")
        if (experiment["provider"] != provider_name or experiment["model"] != model
                or experiment.get("reasoning_effort") != reasoning_effort
                or experiment.get("validation_policy") != validation_policy or experiment.get("knowledge_enabled") is not False):
            raise ValueError("recorded experiment settings do not match the pinned protocol")
        if max_cycles is not None and max_cycles != experiment["max_cycles"]:
            raise ValueError("max_cycles cannot change on resume")
        max_cycles = experiment["max_cycles"]
        repository = Path(experiment["repository"])
        corpus = Path(experiment["corpus"])
        provider_name = experiment["provider"]
        model = experiment["model"]
        selected_kernels = experiment["kernels"]
        target_ids = experiment["targets"]
        defaults.update(experiment.get("execution", {}))
    else:
        protocol_revision = protocol_revision or PROTOCOL_REVISION
        expected_model = GUIDANCE_MODEL if protocol_revision == GUIDANCE_PROTOCOL else EXPERIMENT_MODEL
        expected_provider = "openrouter" if protocol_revision == GUIDANCE_PROTOCOL else "codex-cli"
        if provider_name is not None and provider_name != expected_provider:
            raise ValueError(f"{protocol_revision} requires {expected_provider}")
        provider_name = expected_provider
        if model is not None and model != expected_model:
            raise ValueError(f"{protocol_revision} requires {expected_model}")
        model = expected_model
        if scenarios and set(scenarios) & set(ABLATIONS) and protocol_revision != CASE_STUDY_PROTOCOL:
            raise ValueError("ablation scenarios require the case-study-v2 protocol")
        if scenarios and "baseline_minimal" in scenarios and protocol_revision != GUIDANCE_PROTOCOL:
            raise ValueError("baseline_minimal requires the case-study-v3 protocol")
        max_cycles = 10 if max_cycles is None else max_cycles
        kernels = discover_kernels(corpus)
        selected_kernels = kernel_names or sorted(kernels)
        unknown_kernels = set(selected_kernels) - set(kernels)
        if unknown_kernels:
            raise ValueError(f"unknown kernels: {sorted(unknown_kernels)}")
        target_ids = target_ids or list(TARGETS)
        unknown_targets = set(target_ids) - set(TARGETS)
        if unknown_targets:
            raise ValueError(f"unknown targets: {sorted(unknown_targets)}")
        experiment_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = runs_dir / experiment_id
        suffix = 1
        while root.exists():
            root = runs_dir / f"{experiment_id}-{suffix}"
            suffix += 1
        root.mkdir(parents=True)
        experiment = _new_experiment(
            root.name,
            repository,
            selected_kernels,
            target_ids,
            provider_name,
            model,
            corpus,
            max_cycles,
            list(GUIDANCE_SCENARIOS if protocol_revision == GUIDANCE_PROTOCOL else SCENARIOS) if scenarios is None else scenarios,
        )
        experiment["protocol_revision"] = protocol_revision
        if order_seed is not None:
            experiment["order_seed"] = order_seed
        _atomic_json(root / "experiment.json", experiment)
    if provider is not None and provider.name != "fake" and provider.name != provider_name:
        raise ValueError("live provider does not match the pinned protocol")
    execution = {name: defaults[name] if value is None else value for name, value in execution.items()}
    experiment["execution"] = execution
    kernels = discover_kernels(corpus)
    corpus_hashes = {name: {"source": sha256(kernels[name][0]), "manifest": sha256(kernels[name][1])} for name in selected_kernels}
    if resume and experiment["reproducibility"].get("corpus_sha256") != corpus_hashes:
        raise ValueError("corpus contents changed since this experiment began")
    profiles = {key: TARGETS[key].model_dump(mode="json") for key in target_ids}
    if profiles != experiment["reproducibility"]["target_profiles"]:
        raise ValueError("target profiles changed since this experiment began")
    experiment["reproducibility"]["corpus_sha256"] = corpus_hashes
    database_path = root / "state.sqlite"
    database = Database(database_path)
    database.close()
    pending = [case for case in experiment["cases"].values() if case["status"] != "completed"]
    selected_provider = provider
    provider_error = None
    if pending and selected_provider is None:
        try:
            provider_type = ExperimentOpenRouterProvider if protocol_revision == GUIDANCE_PROTOCOL else ExperimentCodexProvider
            selected_provider = provider_type(repository, 900)
        except ProviderError as exc:
            provider_error = str(exc)
    compiler_options = {name: execution[name] for name in ("max_compiler_jobs", "compiler_cpus", "compiler_memory")}
    worker_state = local()
    compiler_slots = BoundedSemaphore(execution["max_compiler_jobs"])
    stop_event = Event()
    experiment["validation_policy"] = validation_policy
    if selected_provider:
        recorded_provider = experiment["reproducibility"].get("provider")
        if resume and protocol_revision == GUIDANCE_PROTOCOL and recorded_provider != _provider_metadata(selected_provider):
            raise ValueError("OpenRouter provider settings changed on resume")
        experiment["reproducibility"]["provider"] = _provider_metadata(selected_provider)
    _atomic_json(root / "experiment.json", experiment)
    print(f"Experiment: {root}\n{execution['case_workers']} case workers; "
          f"{execution['max_compiler_jobs']} validation jobs; "
          f"{execution['compiler_cpus']} CPUs / {execution['compiler_memory']} per container", flush=True)

    def execute_case(case: dict[str, Any]) -> dict[str, Any]:
        source_path, manifest_path, _ = kernels[case["kernel"]]
        started = time.perf_counter()
        try:
            if stop_event.is_set():
                raise KeyboardInterrupt("experiment stopped before case")
            if provider_error:
                raise ProviderError(provider_error)
            request = TranslationRequest(
                source_path=str(source_path),
                manifest_path=str(manifest_path),
                targets=[TARGETS[case["target_id"]]],
                provider=provider_name,
                model=model,
                search_rounds=0,
                debug_retries=0,
                validation_policy=validation_policy,
            )
            settings = Settings(
                database_path=database_path,
                runs_path=root / case["method"],
                repository_path=repository,
                provider=provider_name,
                model=model,
                experiment_id=experiment["experiment_id"],
                validation_policy=validation_policy,
                **compiler_options,
            )
            if compiler is None and not hasattr(worker_state, "compiler"):
                worker_state.compiler = DockerCompiler(settings)
            selected_compiler = compiler if compiler is not None else worker_state.compiler
            result = translate_scenario(request, settings, scenario=case["method"], max_cycles=max_cycles,
                                        run_id=case["run_id"], provider=selected_provider, compiler=selected_compiler,
                                        stop_event=stop_event, compiler_slots=compiler_slots,
                                        protocol_revision=protocol_revision)
            relative_result = Path(case["method"]) / case["run_id"] / "report.json"
            return {"result_path": str(relative_result), "result_status": result["status"],
                    "duration_seconds": result["duration_seconds"], "status": "completed"}
        except Exception as exc:
            return {"result_status": "blocked" if isinstance(exc, ProviderError) else "failed", "error": str(exc),
                    "duration_seconds": float(case.get("duration_seconds") or 0) + time.perf_counter() - started,
                    "status": "completed"}

    def save_progress() -> None:
        experiment["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(root / "experiment.json", experiment)

    # Interleave scenarios so a parallel batch includes all three workflows.
    methods = ("baseline_minimal",) + SCENARIOS + ABLATIONS
    pending.sort(key=lambda case: (case["kernel"], case["target_id"], methods.index(case["method"]))
                 if execution["case_workers"] > 1 else (methods.index(case["method"]), case["kernel"], case["target_id"]))
    if order_seed is not None:
        order = sorted(experiment["cases"])
        random.Random(order_seed).shuffle(order)
        positions = {key: index for index, key in enumerate(order)}
        pending.sort(key=lambda case: positions[f"{case['method']}:{case['kernel']}:{case['target_id']}"])
    remaining = iter(pending)
    active = {}
    executor = ThreadPoolExecutor(max_workers=execution["case_workers"], thread_name_prefix="experiment")
    experiment["status"] = "running"
    try:
        while True:
            while len(active) < execution["case_workers"]:
                case = next(remaining, None)
                if case is None:
                    break
                case.update(status="running", error=None, run_id=case["run_id"] or uuid.uuid4().hex)
                save_progress()
                active[executor.submit(execute_case, dict(case))] = case
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active[future].update(future.result())
                del active[future]
                save_progress()
    except BaseException:
        stop_event.set()
        print("Stopping: saving in-flight model responses and validation results before exit.", flush=True)
        executor.shutdown(wait=True, cancel_futures=True)
        for future, case in active.items():
            if future.cancelled():
                case["status"] = "pending"
                continue
            try:
                case.update(future.result())
            except (KeyboardInterrupt, SystemExit):
                pass
        experiment["status"] = "interrupted"
        save_progress()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    experiment["status"] = "completed"
    completed_at = datetime.now(UTC)
    experiment["completed_at"] = completed_at.isoformat()
    experiment["updated_at"] = experiment["completed_at"]
    experiment["active_duration_seconds"] = sum(
        float(case.get("duration_seconds") or 0) for case in experiment["cases"].values()
    )
    experiment["wall_duration_seconds"] = (
        completed_at - datetime.fromisoformat(experiment["created_at"])
    ).total_seconds()
    _atomic_json(root / "experiment.json", experiment)
    build_report(root)
    return root
