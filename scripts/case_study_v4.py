#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import platform
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npu_agent.compilers import DockerCompiler
from npu_agent.config import TARGETS, Settings
from npu_agent.experiment import discover_kernels
from npu_agent.models import CodeBundle, KernelManifest, TranslationRequest
from npu_agent.oracles import execute_oracle, validate_goldens
from npu_agent.providers import ProviderError, normalize_usage
from npu_agent.scenarios import ExecutableKernelIR
from npu_agent.study_v4 import Hint, StudyCompiler, StudyConfig, case_schedule, run_case, stage_prompt
from npu_agent.study_v4_provider import StudyProvider
from npu_agent.study_v4_reporting import build_report
from npu_agent.testcases import generate_cases
from npu_agent.validation import json_digest, sha256, write_json


def inputs(config: StudyConfig) -> tuple[dict, dict, dict]:
    corpus = (ROOT / config.corpus).resolve()
    discovered = discover_kernels(corpus)
    # discover_kernels is shared with historical runners; v4 additionally rejects duplicate names.
    if len(list(corpus.rglob("manifest.json"))) != len(discovered):
        raise ValueError("duplicate kernel names in corpus")
    names = sorted(config.kernels or discovered)
    if set(names) - set(discovered):
        raise ValueError("unknown selected kernels")
    selected = {name: discovered[name] for name in names}
    families = {name: name for name in names}
    holdout = None
    if config.holdout_manifest:
        holdout = json.loads((ROOT / config.holdout_manifest).read_text())
        if not holdout.get("reviewed_by") or not holdout.get("review_notes") or not holdout.get("development_source_hashes"):
            raise ValueError("holdout manifest needs reviewed_by, review_notes, development_source_hashes and families")
        families = {name: holdout["families"][name] for name in names}
        if any(not isinstance(family, str) or not family.strip() for family in families.values()):
            raise ValueError("each holdout kernel needs a nonempty family")
    if config.stage == "confirmatory":
        development = discover_kernels(ROOT / "examples" / "classic")
        forbidden_hashes = set(holdout["development_source_hashes"]) | {sha256(item[0]) for item in development.values()}
        if set(names) & set(development) or any(sha256(item[0]) in forbidden_hashes for item in selected.values()):
            raise ValueError("confirmatory corpus overlaps development kernels; renaming identical source is insufficient")
        if len(set(families.values())) < 10:
            raise ValueError("confirmatory corpus needs at least ten reviewed kernel families")
        evidence = {}
        for name, (source, manifest_path, _) in selected.items():
            reference = holdout.get("source_validation", {}).get(name, {})
            if reference.get("manifest_sha256") != sha256(manifest_path) or not reference.get("report"):
                raise ValueError("confirmatory kernels need source-runtime validation bound to the exact manifest")
            report_path = (ROOT / reference["report"]).resolve()
            report = json.loads(report_path.read_text())
            if (report.get("status") != "passed" or report.get("representation") != "original_gpu_source" or
                    report.get("details", {}).get("source_sha256") != sha256(source)):
                raise ValueError("source-runtime validation is missing, unsuccessful, or belongs to another source")
            evidence[name] = {"report": report, "report_sha256": sha256(report_path)}
        holdout = {**holdout, "verified_source_validation": evidence}
    hashes = {str(path.relative_to(corpus)): sha256(path) for path in sorted(corpus.rglob("*"))
              if path.is_file() and "__pycache__" not in path.parts}
    return selected, families, {"corpus": str(corpus), "hashes": hashes, "holdout": holdout}


def source_files() -> list[Path]:
    paths = []
    for directory in ("src", "scripts", "playground/tools", "playground/dockerfiles", "tests/fixtures/backends"):
        paths += [path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    paths += [ROOT / relative for relative in ("pyproject.toml", "uv.lock", "playground/toolchains.lock.json", "docs/case-study-v4.md")
              if (ROOT / relative).is_file()]
    return sorted(paths)


def freeze(config: StudyConfig, corpus: dict) -> dict:
    return {"config": config.model_dump(), "corpus": corpus,
            "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_files()},
            "python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pydantic", "langgraph")},
            "targets": {name: TARGETS[name].model_dump(mode="json") for name in config.targets}}


def prepare(root: Path, config: StudyConfig, selected: dict, families: dict, corpus: dict) -> dict:
    identity = freeze(config, corpus)
    path = root / "campaign.json"
    if path.exists():
        campaign = json.loads(path.read_text())
        if campaign["identity"] != identity:
            raise ValueError("frozen configuration, source, environment or corpus changed; use a new output directory")
        return campaign
    cases = case_schedule(config, list(selected), families)
    campaign = {"config": config.model_dump(), "identity": identity, "cases": cases, "status": "prepared",
                "created_at": datetime.now(UTC).isoformat(), "events": []}
    for path in source_files():
        destination = root / "frozen" / "repository" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    corpus_root = Path(corpus["corpus"])
    for relative in corpus["hashes"]:
        destination = root / "frozen" / "corpus" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((corpus_root / relative).read_bytes())
    write_json(root / "campaign.json", campaign)
    return campaign


def preflight(root: Path, campaign: dict, config: StudyConfig, provider: StudyProvider, compiler: DockerCompiler,
              selected: dict, stop: Event) -> None:
    path = root / "preflight.json"
    identity = {"provider": provider.endpoint_identity(), "compilers": {name: compiler.identity(TARGETS[name]) for name in config.targets}}
    if path.exists():
        preflight_state = json.loads(path.read_text())
        if preflight_state["identity"] != identity:
            raise ValueError("provider endpoint or compiler identity changed since preflight")
    else:
        preflight_state = {"identity": identity, "status": "running", "calls": {}}
        write_json(path, preflight_state)
    # Establish a working evaluator before spending model tokens. Fixtures are never model context.
    for _, _, manifest in selected.values():
        validate_goldens(manifest.oracle.operation)
        for case in generate_cases(manifest):
            execute_oracle(manifest, case.inputs)
    for target in config.targets:
        if stop.is_set():
            return
        if target not in preflight_state.get("compiler_smokes", {}):
            fixture = ROOT / "tests" / "fixtures" / "backends" / target / "add"
            manifest = KernelManifest.model_validate_json((fixture / "manifest.json").read_text())
            result = compiler.compile(fixture, root / "preflight-compiler" / target, manifest, TARGETS[target])
            preflight_state.setdefault("compiler_smokes", {})[target] = result.model_dump(mode="json")
            write_json(path, preflight_state)
        if not preflight_state["compiler_smokes"][target]["validation"]["offline_contract_met"]:
            raise ValueError("compiler/evaluator preflight failed; no model calls should be launched")
    # Validate actual response schemas, including the large IR schema, on an unrelated toy task.
    for schema in (CodeBundle, Hint, ExecutableKernelIR):
        if stop.is_set():
            return
        name = schema.__name__
        if name in preflight_state["calls"]:
            if preflight_state["calls"][name]["status"] != "completed":
                raise ValueError("preflight has an unsuccessful/lost paid call; preserve it and use a new campaign")
            continue
        prompt = ("Transport/schema preflight only. Respond using the provided JSON schema. Describe an identity function on "
                  "one float32 vector of length 4 called x, returning y. If an intermediate is requested, use concise text. "
                  "For CodeBundle use intel_openvino and no files, with unsupported_operations=['preflight only']. "
                  "For KernelIR use schema_version 2.0, a reshape from x to y with shape [4], no scalar arguments, "
                  "no side effects. This is a schema check, not a benchmark translation.")
        call = {"status": "in_flight", "prompt": prompt, "schema": schema.model_json_schema(),
                "started_at": datetime.now(UTC).isoformat()}
        preflight_state["calls"][name] = call
        write_json(path, preflight_state)
        try:
            response = provider.generate(prompt, schema, config.provider.model)
            call.update(status="completed", response=response.raw, metadata=response.metadata)
        except ProviderError as exc:
            call.update(status="blocked", response=exc.raw, metadata=exc.metadata, error=str(exc))
            preflight_state["status"] = "blocked"
            write_json(path, preflight_state)
            raise
        write_json(path, preflight_state)
    if stop.is_set():
        return
    reasoning = [normalize_usage(call["metadata"]["usage"])["reasoning_output_tokens"] for call in preflight_state["calls"].values()]
    if config.provider.reasoning and not any((value or 0) > 0 for value in reasoning):
        raise ValueError("reasoning-enabled preflight returned no observed reasoning tokens")
    # Archive representative prompts for review before the first benchmark call.
    first = next(iter(selected.values()))
    for arm in config.arms:
        for target in config.targets:
            prompt = stage_prompt("code" if arm == "direct" else "intermediate", arm, first[0].read_text(), first[2],
                                  TARGETS[target], None, "")
            destination = root / "prompt-preview" / f"{arm}-{target}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(prompt)
    preflight_state["status"] = "completed"
    write_json(path, preflight_state)
    campaign["status"] = "ready"
    write_json(root / "campaign.json", campaign)


def install_pause_handlers(stop: Event) -> dict:
    requested = False

    def pause(signum, frame):
        nonlocal requested
        del signum, frame
        if requested:
            return
        # Signal handlers can re-enter Event.set(), whose lock is not reentrant.
        requested = True
        stop.set()
        print("Pause requested; finishing and saving the current stage. Further Ctrl+C signals will not discard it.", flush=True)

    previous = {sig: signal.signal(sig, pause) for sig in (signal.SIGINT, signal.SIGTERM)}
    return previous


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen, serial, OpenRouter-only IR comparison. plan/report make no model calls.")
    parser.add_argument("command", choices=("plan", "preflight", "run", "report"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "case-study-v4.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "case-study-v4-pilot")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "report":
        print(json.dumps(build_report(root), indent=2))
        return 0
    config = StudyConfig.model_validate_json(args.config.read_text())
    selected, families, corpus = inputs(config)
    cases = case_schedule(config, list(selected), families)
    if args.command == "plan":
        print(json.dumps({"config": config.model_dump(), "cases": len(cases),
                          "max_model_calls": sum(1 if case["arm"] == "direct" else 2 for case in cases) * config.max_cycles + 3,
                          "schedule_sha256": json_digest(cases), "families": families,
                          "execution": "one case and one validation job at a time",
                          "budget_note": "token/time stopping thresholds checked between stages; a stage may overshoot. No hard spending cap.",
                          "preflight": "three paid schema probes before benchmark; plan makes no calls"}, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = prepare(root, config, selected, families, corpus)
        settings = Settings(database_path=root / "state.sqlite", runs_path=root / "cases", repository_path=ROOT,
                            provider="openrouter", model=config.provider.model, validation_policy="offline-validated",
                            experiment_id=root.name, compiler_cpus=config.compiler_cpus, compiler_memory=config.compiler_memory,
                            max_compiler_jobs=1, compiler_timeout_seconds=config.compiler_timeout_seconds)
        settings.ensure_directories()
        provider = StudyProvider(config.provider, ROOT)
        compiler = StudyCompiler(settings)
        stop = Event()
        previous = install_pause_handlers(stop)
        try:
            preflight(root, campaign, config, provider, compiler, selected, stop)
            if args.command == "run" and not stop.is_set():
                campaign["status"] = "running"
                campaign["events"].append({"event": "run_or_resume", "at": datetime.now(UTC).isoformat()})
                write_json(root / "campaign.json", campaign)
                for index, case in enumerate(campaign["cases"]):
                    if stop.is_set():
                        break
                    source, manifest, _ = selected[case["kernel"]]
                    request = TranslationRequest(source_path=str(source), manifest_path=str(manifest),
                        targets=[TARGETS[case["target"]]], provider="openrouter", model=config.provider.model,
                        validation_policy="offline-validated")
                    state = run_case(request, settings, config, case, provider, compiler, stop)
                    print(f"{index + 1}/{len(cases)} {case['kernel']} {case['target']} {case['arm']}: {state['status']}", flush=True)
                    if state["status"] == "blocked":
                        campaign["status"] = "blocked"
                        break
                else:
                    campaign["status"] = "completed"
            if stop.is_set():
                campaign["status"] = "paused"
            campaign["events"].append({"event": campaign["status"], "at": datetime.now(UTC).isoformat()})
            write_json(root / "campaign.json", campaign)
            build_report(root)
            return 2 if campaign["status"] == "blocked" else 0
        except Exception:
            campaign["status"] = "blocked"
            write_json(root / "campaign.json", campaign)
            build_report(root)
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ProviderError, OSError) as exc:
        print(f"v4 stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
