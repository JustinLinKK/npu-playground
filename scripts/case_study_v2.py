#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
import shutil
import signal
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

from npu_agent.compilers import DockerCompiler
from npu_agent.config import TARGETS, Settings
from npu_agent.experiment import _atomic_json, discover_kernels, run_experiment
from npu_agent.providers import ExperimentCodexProvider, ExperimentOpenRouterProvider, ProviderError, StructuredProvider
from npu_agent.reporting import collect_metrics
from npu_agent.scenarios import ABLATIONS, CASE_STUDY_PROTOCOL, EXPERIMENT_MODEL, GUIDANCE_MODEL, GUIDANCE_PROTOCOL, GUIDANCE_SCENARIOS, IntermediateRepresentation, SCENARIOS
from npu_agent.validation import sha256


REPOSITORY = Path(__file__).resolve().parents[1]
TOKEN_BUDGETS = (10000, 25000, 50000, 100000, 200000, 400000)


def inputs(corpus: Path, names: list[str]) -> dict:
    paths = [REPOSITORY / "pyproject.toml", REPOSITORY / "uv.lock",
             REPOSITORY / "playground/toolchains.lock.json",
             REPOSITORY / "docs/case-study-v2.md", REPOSITORY / "docs/case-study-v3.md"]
    for directory in ("src", "playground/tools", "playground/dockerfiles", "scripts"):
        paths.extend(p for p in (REPOSITORY / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    kernels = discover_kernels(corpus)
    return {"code": {str(p.relative_to(REPOSITORY)): sha256(p) for p in sorted(set(paths))},
            "corpus": {name: {"source": sha256(kernels[name][0]), "manifest": sha256(kernels[name][1])}
                       for name in names}}


def check_heldout(config: dict, identity: dict) -> None:
    if config["stage"] != "heldout":
        return
    development = Path(config["development_campaign"])
    previous = json.loads((development / "campaign.json").read_text())
    if previous["config"]["stage"] != "controlled" or previous.get("status") != "completed":
        raise ValueError("held-out evaluation requires a completed controlled campaign")
    if identity["code"] != previous["inputs"]["code"]:
        raise ValueError("held-out evaluation must use the frozen controlled-campaign code")
    for name in ("protocol_revision", "provider", "model", "reasoning_effort", "max_cycles", "knowledge_enabled", "validation_policy"):
        if config.get(name) != previous["config"].get(name):
            raise ValueError(f"held-out evaluation must preserve {name}")
    old = previous["inputs"]["corpus"]
    if set(old) & set(identity["corpus"]):
        raise ValueError("held-out kernel names overlap the development corpus")
    old_sources = {item["source"] for item in old.values()}
    if any(item["source"] in old_sources for item in identity["corpus"].values()):
        raise ValueError("held-out source duplicates development source; renaming is not a holdout")


def environment(targets: list[str], protocol_revision: str = CASE_STUDY_PROTOCOL) -> tuple[dict, StructuredProvider]:
    provider_type = ExperimentOpenRouterProvider if protocol_revision == GUIDANCE_PROTOCOL else ExperimentCodexProvider
    provider = provider_type(REPOSITORY, 900)
    compiler = DockerCompiler(Settings(repository_path=REPOSITORY))
    toolchains = {name: compiler.identity(TARGETS[name]) for name in targets}
    identity = {"python": sys.version, "provider_version": provider.version,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pydantic", "langgraph", "matplotlib")},
            "toolchains": toolchains}
    if isinstance(provider, ExperimentOpenRouterProvider):
        identity["provider_settings"] = provider.public_config
        identity["packages"]["pyyaml"] = importlib.metadata.version("pyyaml")
    return identity, provider


def summarize(output: Path, campaign: dict) -> dict:
    config = campaign["config"]
    records, clocks = [], []
    for repetition in range(1, config["repetitions"] + 1):
        roots = sorted((output / f"repeat-{repetition:03d}").glob("*/experiment.json"))
        if len(roots) > 1:
            raise ValueError("multiple experiments in one repetition directory")
        if not roots:
            continue
        root = roots[0].parent
        metadata = json.loads(roots[0].read_text())
        if not (root / "state.sqlite").exists():
            continue
        records.extend(dict(row, repetition=repetition) for row in collect_metrics(root))
        clocks.append({"repetition": repetition, "status": metadata["status"],
                       "elapsed_including_pauses_seconds": metadata.get("wall_duration_seconds"),
                       "summed_active_case_seconds": metadata.get("active_duration_seconds")})
    groups = []
    for target in config["targets"]:
        for method in config["scenarios"]:
            rows = [r for r in records if r["target_id"] == target and r["method"] == method]
            expected = len(config["kernels"]) * config["repetitions"]
            complete = len(rows) == expected and all(r["usage_complete"] for r in rows)
            groups.append({"target": target, "method": method, "requested": expected, "recorded": len(rows),
                "solved": sum(r["success"] for r in rows), "success_rate": sum(r["success"] for r in rows) / expected,
                "first_cycle_solved": sum(r["cycles_to_success"] == 1 for r in rows),
                "blocked": sum(r["outcome_status"] == "blocked" for r in rows),
                "cycles_spent": sum(r["cycles_used"] or 0 for r in rows),
                "llm_calls": sum(r["llm_calls"] for r in rows),
                "total_tokens": sum(r["total_tokens"] for r in rows) if complete else None,
                "known_tokens": sum(r["known_total_tokens"] for r in rows),
                "usage_complete_cases": sum(r["usage_complete"] for r in rows),
                "active_seconds": sum(r["end_to_end_seconds"] or 0 for r in rows),
                "successes_by_repetition": [sum(r["success"] for r in rows if r["repetition"] == n)
                                            for n in range(1, config["repetitions"] + 1)],
                "failure_categories": dict(Counter(r["failure_category"] or "unknown" for r in rows if not r["success"])),
                "success_within_token_budget": {str(b): sum(r["success"] and r["usage_complete"] and
                    r["total_tokens"] <= b for r in rows) / expected for b in TOKEN_BUDGETS}})
    lookup = {(r["repetition"], r["kernel"], r["target_id"], r["method"]): r for r in records}
    paired = []
    comparisons = [("structured_ir", other) for other in config["scenarios"] if other != "structured_ir"]
    if "baseline_minimal" in config["scenarios"]:
        comparisons.append(("baseline", "baseline_minimal"))
        comparisons.append(("hinted_ir", "baseline"))
    for target in config["targets"]:
        for method, other in comparisons:
            pairs, differences = [], []
            for kernel in config["kernels"]:
                kernel_pairs = [(lookup.get((n, kernel, target, method)), lookup.get((n, kernel, target, other)))
                                for n in range(1, config["repetitions"] + 1)]
                if not all(a and b and a["terminal_reason"] in ("solved", "cycle_budget_exhausted") and
                           b["terminal_reason"] in ("solved", "cycle_budget_exhausted") for a, b in kernel_pairs):
                    continue
                pairs.extend(kernel_pairs)
                differences.append(sum(int(a["success"]) - int(b["success"]) for a, b in kernel_pairs) / len(kernel_pairs))
            solved = [(a, b) for a, b in pairs if a["success"] and b["success"]]
            token_pairs = [(a, b) for a, b in solved if a["usage_complete"] and b["usage_complete"]]
            interval = None
            if len(differences) >= 2:
                samples = np.random.default_rng(20260916).choice(differences, size=(2000, len(differences)))
                interval = np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist()
            paired.append({"target": target, "reference": other, "method": method,
                "complete_kernel_clusters": len(differences), "requested_kernel_clusters": len(config["kernels"]),
                "success_rate_difference": float(np.mean(differences)) if differences else None,
                "kernel_bootstrap_95_interval": interval, "jointly_solved": len(solved),
                "token_complete_pairs": len(token_pairs),
                "paired_tokens": {method: sum(a["total_tokens"] for a, _ in token_pairs),
                                  "reference": sum(b["total_tokens"] for _, b in token_pairs)},
                "paired_active_seconds": {method: sum(a["end_to_end_seconds"] for a, _ in solved),
                                          "reference": sum(b["end_to_end_seconds"] for _, b in solved)}})
    summary = {"status": campaign["status"], "config": config, "groups": groups, "paired": paired,
               "experiment_clocks": clocks, "token_budgets": list(TOKEN_BUDGETS)}
    _atomic_json(output / "summary.json", summary)
    if records:
        with (output / "cases.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    title = "Case study v3: prompt guidance and IR" if config.get("protocol_revision") == GUIDANCE_PROTOCOL else "Case study v2"
    lines = [f"# {title}", "", f"Stage: {config['stage']}. Status: {campaign['status']}.", "",
             "Offline validation only; no physical NPU correctness or performance claim.", "",
             "| Target | Arm | Solved / requested | Blocked | Cycles | Tokens | Active minutes |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    if campaign.get("blocked_reason"):
        lines[4:4] = [campaign["blocked_reason"], ""]
    if config.get("protocol_revision") == GUIDANCE_PROTOCOL:
        lines[2:2] = [f"Model: {config['model']}. Provider: {config['provider']}.", ""]
    for row in groups:
        tokens = f"{row['total_tokens']:,}" if row["total_tokens"] is not None else f"unknown (known: {row['known_tokens']:,})"
        lines.append(f"| {row['target']} | {row['method']} | {row['solved']}/{row['requested']} | "
                     f"{row['blocked']} | {row['cycles_spent']} | {tokens} | {row['active_seconds'] / 60:.1f} |")
    lines.extend(["", "Active time sums concurrent case clocks; campaign elapsed clocks include any resume pauses.",
                  "Token-budget curves in summary.json replay first-success costs within the ten-cycle cap; they are not hard spending caps.",
                  "Unresolved cases remain in requested denominators; paired intervals use only complete kernel clusters, with coverage reported.",
                  "Bootstrap intervals are exploratory, resampling kernels while keeping their repetitions together.",
                  "Per-repetition report.html files contain all-arm PNG/SVG charts and detailed evidence."])
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return summary


def run_campaign(output: Path, config: dict) -> None:
    corpus = Path(config["corpus"])
    identity = inputs(corpus, config["kernels"])
    check_heldout(config, identity)
    path = output / "campaign.json"
    if path.exists():
        campaign = json.loads(path.read_text())
        if campaign["config"] != config or campaign["inputs"] != identity:
            raise ValueError("campaign configuration or source/corpus changed; restore it or use a new output directory")
        if campaign["status"] == "completed":
            summarize(output, campaign)
            print(f"Already completed; no model calls. Results: {output / 'results.md'}")
            return
    else:
        if list(output.glob("repeat-*")):
            raise ValueError("untracked repetition directories already exist")
        campaign = {"schema_version": 1, "config": config, "inputs": identity,
                    "created_at": datetime.now(UTC).isoformat(), "status": "running"}
        for relative in identity["code"]:
            destination = output / "snapshot/code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / relative, destination)
        for source, manifest, _ in discover_kernels(corpus).values():
            for item in (source, manifest):
                destination = output / "snapshot/corpus" / item.relative_to(corpus)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)
        _atomic_json(path, campaign)
    try:
        for repetition in range(1, config["repetitions"] + 1):
            if inputs(corpus, config["kernels"]) != identity:
                raise ValueError("source or corpus changed during the campaign")
            current, provider = environment(config["targets"], config.get("protocol_revision", CASE_STUDY_PROTOCOL))
            if config["stage"] == "heldout":
                development = json.loads((Path(config["development_campaign"]) / "campaign.json").read_text())
                if current != development["environment"]:
                    raise ValueError("held-out evaluation requires the controlled-campaign environment and targets")
            if "environment" in campaign and current != campaign["environment"]:
                raise ValueError("provider, Python packages, or compiler/evaluator identity changed")
            campaign["environment"] = current
            campaign["status"] = "running"
            _atomic_json(path, campaign)
            if config.get("protocol_revision") == GUIDANCE_PROTOCOL and not campaign.get("model_preflight_passed"):
                try:
                    response = provider.generate(
                        "Return the structured response with intermediate exactly equal to ready. Do not use tools.",
                        IntermediateRepresentation, config["model"])
                except ProviderError as exc:
                    _atomic_json(output / "model-preflight.json", {"status": "blocked", "error": str(exc),
                        "metadata": exc.metadata, "response": exc.raw})
                    campaign["blocked_reason"] = "Model access preflight failed; no benchmark cases were started. See model-preflight.json."
                    raise ValueError(f"model access preflight failed; inspect {output / 'model-preflight.json'}") from exc
                _atomic_json(output / "model-preflight.json", {"status": "passed", "metadata": response.metadata,
                    "response": response.raw})
                campaign["model_preflight_passed"] = True
                campaign.pop("blocked_reason", None)
                _atomic_json(path, campaign)
            trial = output / f"repeat-{repetition:03d}"
            roots = sorted(trial.glob("*/experiment.json"))
            if len(roots) > 1:
                raise ValueError("multiple experiments in one repetition directory")
            if roots and json.loads(roots[0].read_text())["status"] == "completed":
                root = roots[0].parent
            else:
                root = run_experiment(repository=REPOSITORY, corpus=corpus, runs_dir=trial,
                    kernel_names=config["kernels"], target_ids=config["targets"], scenarios=config["scenarios"],
                    provider_name=config.get("provider", "codex-cli"),
                    model=config.get("model", EXPERIMENT_MODEL), max_cycles=10, provider=provider,
                    protocol_revision=config.get("protocol_revision", CASE_STUDY_PROTOCOL), order_seed=config["order_seed"] + repetition - 1,
                    resume=str(roots[0].parent) if roots else None, **config["execution"])
            if inputs(corpus, config["kernels"]) != identity:
                raise ValueError("source or corpus changed during the repetition")
            with sqlite3.connect(root / "state.sqlite") as connection:
                fingerprints = connection.execute("SELECT DISTINCT c.target_id, ca.compiler_fingerprint "
                    "FROM compile_attempts ca JOIN candidates c ON c.id=ca.candidate_id").fetchall()
            if any(fingerprint != current["toolchains"][target]["fingerprint"] for target, fingerprint in fingerprints):
                raise ValueError("a backend attempt used a different compiler/evaluator identity")
            metadata = json.loads((root / "experiment.json").read_text())
            records = collect_metrics(root)
            problems = [r for r in records if r["terminal_reason"] not in ("solved", "cycle_budget_exhausted")
                        or not r["usage_complete"]]
            if metadata["status"] != "completed" or problems:
                raise ValueError(f"repetition {repetition} has blocked, unexpected, or incomplete-usage cases; "
                                 "inspect its report before starting another campaign")
            summarize(output, campaign)
        campaign["status"] = "completed"
        campaign["completed_at"] = datetime.now(UTC).isoformat()
    except BaseException:
        campaign["status"] = "interrupted"
        raise
    finally:
        _atomic_json(path, campaign)
        summarize(output, campaign)
    print(f"Results: {output / 'results.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled, resumable case study (no model calls for plan/report)")
    parser.add_argument("action", choices=["plan", "run", "report"], nargs="?", default="plan")
    parser.add_argument("--output", type=Path, default=Path("runs/case-study-v2"))
    parser.add_argument("--stage", choices=["controlled", "heldout"], default="controlled")
    parser.add_argument("--protocol-revision", choices=[CASE_STUDY_PROTOCOL, GUIDANCE_PROTOCOL], default=CASE_STUDY_PROTOCOL)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--development-campaign", type=Path)
    parser.add_argument("--kernels", help="comma-separated names; default: every kernel in the corpus")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=20260916)
    parser.add_argument("--case-workers", type=int, default=20)
    parser.add_argument("--max-compiler-jobs", type=int, default=4)
    parser.add_argument("--compiler-cpus", type=int, default=6)
    parser.add_argument("--compiler-memory", default="8g")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "report":
        summarize(output, json.loads((output / "campaign.json").read_text()))
        print(output / "results.md")
        return
    if args.stage == "heldout" and (args.corpus is None or args.development_campaign is None):
        parser.error("heldout requires --corpus and --development-campaign")
    corpus = (args.corpus or REPOSITORY / "examples/classic").resolve()
    kernels = discover_kernels(corpus)
    names = sorted(args.kernels.split(",")) if args.kernels else sorted(kernels)
    targets = args.targets.split(",")
    if not names or set(names) - set(kernels) or len(names) != len(set(names)):
        parser.error("unknown or duplicate kernels")
    if not targets or set(targets) - set(TARGETS) or len(targets) != len(set(targets)):
        parser.error("unknown or duplicate targets")
    if min(args.repetitions, args.case_workers, args.max_compiler_jobs, args.compiler_cpus) < 1:
        parser.error("repetitions and execution limits must be positive")
    if not args.compiler_memory.strip():
        parser.error("compiler-memory cannot be empty")
    config = {"stage": args.stage, "corpus": str(corpus), "kernels": names, "targets": targets,
              "development_campaign": str(args.development_campaign.resolve()) if args.development_campaign else None,
              "protocol_revision": args.protocol_revision,
              "model": GUIDANCE_MODEL if args.protocol_revision == GUIDANCE_PROTOCOL else EXPERIMENT_MODEL,
              "reasoning_effort": "none" if args.protocol_revision == GUIDANCE_PROTOCOL else "xhigh",
              "knowledge_enabled": False, "validation_policy": "offline-validated", "max_cycles": 10,
              "repetitions": args.repetitions, "order_seed": args.order_seed,
              "scenarios": list(GUIDANCE_SCENARIOS if args.protocol_revision == GUIDANCE_PROTOCOL else
                                SCENARIOS + ABLATIONS if args.stage == "controlled" else SCENARIOS),
              "execution": {name: getattr(args, name) for name in
                            ("case_workers", "max_compiler_jobs", "compiler_cpus", "compiler_memory")}}
    if args.protocol_revision == GUIDANCE_PROTOCOL:
        config["provider"] = "openrouter"
    check_heldout(config, inputs(corpus, names))
    cases = len(names) * len(targets) * len(config["scenarios"]) * args.repetitions
    print(json.dumps(config, indent=2))
    print(f"{cases} cases; at most {cases * 10} cycles. Live token cost is not hard-capped.")
    if args.action == "plan":
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another process is using this campaign directory")
        run_campaign(output, config)


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    except (ValueError, OSError, ProviderError) as exc:
        sys.exit(str(exc))
