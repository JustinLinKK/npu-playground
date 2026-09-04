from __future__ import annotations

import asyncio
import json
import os
import platform
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .baseline import translate_one_shot
from .compilers import Compiler, DockerCompiler
from .config import TARGETS, Settings
from .database import Database
from .models import KernelManifest, TranslationRequest
from .providers import StructuredProvider, create_provider
from .reporting import build_report
from .workflow import translate


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
) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    for method in ("baseline", "agentic"):
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
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "repository": str(repository),
        "corpus": str(corpus),
        "provider": provider_name,
        "model": model or "CLI default",
        "kernels": selected_kernels,
        "targets": target_ids,
        "search": {"rounds": 3, "branching_factor": 3, "debug_retries": 2},
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
    return {
        "name": provider.name,
        "executable": getattr(provider, "executable", None),
        "version": getattr(provider, "version", None),
    }


def _latest_run(database_path: Path, experiment_id: str, method: str, kernel: str) -> str | None:
    connection = sqlite3.connect(database_path)
    try:
        row = connection.execute(
            """SELECT id FROM app_runs WHERE experiment_id=? AND method=? AND kernel_name=?
               ORDER BY created_at DESC, rowid DESC LIMIT 1""",
            (experiment_id, "one_shot" if method == "baseline" else method, kernel),
        ).fetchone()
    finally:
        connection.close()
    return row[0] if row else None


def run_experiment(
    *,
    repository: Path,
    corpus: Path,
    runs_dir: Path,
    kernel_names: list[str] | None = None,
    target_ids: list[str] | None = None,
    provider_name: str = "codex-cli",
    model: str | None = None,
    resume: str | None = None,
    provider: StructuredProvider | None = None,
    compiler: Compiler | None = None,
) -> Path:
    repository = repository.resolve()
    corpus = corpus.resolve()
    runs_dir = runs_dir.resolve()
    if resume:
        root = _resolve_resume(runs_dir, resume)
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        corpus = Path(experiment["corpus"])
        provider_name = experiment["provider"]
        model = None if experiment["model"] == "CLI default" else experiment["model"]
        selected_kernels = experiment["kernels"]
        target_ids = experiment["targets"]
    else:
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
        )
        _atomic_json(root / "experiment.json", experiment)
    kernels = discover_kernels(corpus)
    database_path = root / "state.sqlite"
    database = Database(database_path)
    seed_result = database.seed_builtin_knowledge()
    catalog_rows = database.connection.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0]
    database.close()
    experiment["knowledge_seed"] = {"catalog_rows": catalog_rows, "idempotent_reseed": seed_result}
    pending = [case for case in experiment["cases"].values() if case["status"] != "completed"]
    selected_provider = provider
    selected_compiler = compiler
    if pending and selected_provider is None:
        selected_provider = create_provider(provider_name, repository, model, 900)
    if pending and selected_compiler is None:
        compiler_settings = Settings(
            database_path=database_path,
            runs_path=root / "agentic",
            repository_path=repository,
            provider=provider_name,
            model=model,
            experiment_id=experiment["experiment_id"],
        )
        selected_compiler = DockerCompiler(compiler_settings)
    if selected_provider:
        experiment["reproducibility"]["provider"] = _provider_metadata(selected_provider)
    _atomic_json(root / "experiment.json", experiment)
    for _, case in sorted(
        experiment["cases"].items(), key=lambda item: ((0 if item[1]["method"] == "baseline" else 1), item[0])
    ):
        if case["status"] == "completed":
            continue
        source_path, manifest_path, _ = kernels[case["kernel"]]
        case["status"] = "running"
        case["error"] = None
        experiment["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(root / "experiment.json", experiment)
        started = time.perf_counter()
        try:
            request = TranslationRequest(
                source_path=str(source_path),
                manifest_path=str(manifest_path),
                targets=[TARGETS[case["target_id"]]],
                provider=provider_name,
                model=model,
                search_rounds=3,
                branching_factor=3,
                debug_retries=2,
            )
            settings = Settings(
                database_path=database_path,
                runs_path=root / case["method"],
                repository_path=repository,
                provider=provider_name,
                model=model,
                experiment_id=experiment["experiment_id"],
            )
            if case["method"] == "baseline":
                result = asyncio.run(
                    translate_one_shot(request, settings, provider=selected_provider, compiler=selected_compiler)
                )
            else:
                result = asyncio.run(translate(request, settings, provider=selected_provider, compiler=selected_compiler))
            relative_result = Path(case["method"]) / "cases" / f"{case['kernel']}__{case['target_id']}.json"
            _atomic_json(root / relative_result, result.model_dump(mode="json"))
            case["run_id"] = result.run_id
            case["result_path"] = str(relative_result)
            case["result_status"] = result.status
        except Exception as exc:
            case["result_status"] = "failed"
            case["error"] = str(exc)
            case["run_id"] = _latest_run(
                database_path, experiment["experiment_id"], case["method"], case["kernel"]
            )
        case["duration_seconds"] = time.perf_counter() - started
        case["status"] = "completed"
        experiment["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(root / "experiment.json", experiment)
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
