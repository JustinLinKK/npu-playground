from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from npu_agent.providers import normalize_usage
from npu_agent.study_v4 import StudyConfig
from npu_agent.study_v4_reporting import case_metrics
from npu_agent.validation import write_json


def failure_category(stage: str, result: dict) -> str | None:
    if stage == "target_execution" or result.get("status") not in ("failed", "blocked", "unsupported", "rejected"):
        return None
    if result.get("reason_code") == "PREREQUISITE_UNAVAILABLE":
        return None
    if result.get("status") == "blocked":
        return "infrastructure"
    if stage.startswith("model_"):
        return "model_output"
    if stage == "semantic_validation":
        message = result.get("message", "").lower()
        return "unsupported_ir" if "unsupported" in message or "not executable" in message else "semantic"
    if stage in ("host_execution", "dataflow_simulation") and result.get("correct") is None and result.get("duration_seconds") == 0:
        return None
    if stage == "dataflow_simulation":
        return "buffer_dataflow"
    if stage == "host_execution" and result.get("correct") is False:
        return "numerical"
    message = result.get("message", "")
    if "AttributeError" in message or "ImportError" in message or "ModuleNotFoundError" in message:
        return "backend_api"
    return "needs_review"


def build_audit(root: Path) -> dict:
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    campaign = read(root / "campaign.json")
    config = StudyConfig.model_validate(campaign["config"])
    preflight = read(root / "preflight.json") if (root / "preflight.json").exists() else {}
    stages, cases = [], []

    def add(scope, case, cycle, stage, result, call=False):
        metadata = result.get("metadata", {}) if call else {}
        counts = normalize_usage(metadata.get("usage", {})) if call else {}
        stages.append({"scope": scope, "case_id": case.get("id"), "kernel": case.get("kernel"),
                       "target": case.get("target"), "arm": case.get("arm"), "cycle": cycle,
                       "stage": stage, "status": result.get("status"),
                       "failure_category": failure_category(stage, result),
                       "message": result.get("error", result.get("message", "")),
                       "known_tokens": counts.get("total_tokens") or 0,
                       "usage_complete": not call or (counts.get("total_tokens") is not None and
                           not metadata.get("usage_incomplete", False) and result.get("status") != "in_flight"),
                       "seconds": metadata.get("duration_seconds") if call else result.get("duration_seconds"),
                       "generation_id": metadata.get("generation_id"),
                       "billed_cost": metadata.get("usage", {}).get("cost") if call else None})

    for name, call in preflight.get("calls", {}).items():
        add("preflight", {}, 0, "model_" + name, call, True)
    for target, result in preflight.get("compiler_smokes", {}).items():
        for name, stage in result.get("validation", {}).get("stages", {}).items():
            add("preflight", {"target": target}, 0, name, stage)
    for case in campaign["cases"]:
        path = root / "cases" / case["id"] / "report.json"
        state = read(path) if path.exists() else None
        cases.append(case_metrics(case, state, config))
        if state is None:
            continue
        for cycle in state["cycles"]:
            for name, call in cycle["calls"].items():
                add("translation", case, cycle["number"], "model_" + name, call, True)
            if "semantic_validation" in cycle:
                add("translation", case, cycle["number"], "semantic_validation", cycle["semantic_validation"])
            # Candidate evaluation copies semantic validation and compiler stages. Read each once.
            for name, stage in cycle.get("compile_result", {}).get("validation", {}).get("stages", {}).items():
                if name != "semantic_validation":
                    add("translation", case, cycle["number"], name, stage)
            if cycle.get("status") == "failed" and cycle.get("feedback", {}).get("error"):
                if not any(item["case_id"] == case["id"] and item["cycle"] == cycle["number"] and
                           item["failure_category"] for item in stages):
                    add("translation", case, cycle["number"], "candidate_contract",
                        {"status": "failed", "message": cycle["feedback"]["error"]})
        if state["status"] == "blocked" and not any(item["case_id"] == case["id"] and
                item["failure_category"] == "infrastructure" for item in stages):
            add("translation", case, len(state["cycles"]), "case_interruption",
                {"status": "blocked", "message": state.get("terminal_reason", "")})
    calls = [item for item in stages if item["stage"].startswith("model_")]
    ids = [item["generation_id"] for item in calls if item["generation_id"]]
    distributions = []
    for target in config.targets:
        for arm in config.arms:
            selected = [case for case in cases if case["target"] == target and case["arm"] == arm]
            started = [case for case in selected if case["status"] != "pending"]
            group = {"target": target, "arm": arm, "planned": len(selected), "started": len(started),
                     "finished": sum(case["status"] in ("completed", "failed") for case in selected)}
            for metric in ("known_tokens", "active_seconds", "backend_attempts", "identical_backend_pairs"):
                values = [case[metric] for case in started]
                group[metric] = {"sum": sum(values), "quantiles": dict(zip(("min", "p25", "median", "p75", "max"),
                    np.quantile(values, (0, .25, .5, .75, 1)).tolist())) if values else None}
            distributions.append(group)
    known_tokens = sum(item["known_tokens"] for item in calls)
    preflight_tokens = sum(item["known_tokens"] for item in calls if item["scope"] == "preflight")
    return {"protocol": "research-audit-v1", "campaign_status": campaign["status"], "input_sha256": hashes,
            "cases": cases, "stages": stages, "distributions": distributions,
            "parallel_events": [event for event in campaign.get("events", []) if event.get("event") == "parallel_resume"],
            "failure_counts": dict(Counter(item["failure_category"] for item in stages if item["failure_category"])),
            "known_tokens": known_tokens, "preflight_known_tokens": preflight_tokens,
            "translation_known_tokens": known_tokens - preflight_tokens,
            "unknown_usage_calls": sum(not item["usage_complete"] for item in calls),
            "in_flight_calls": sum(item["status"] == "in_flight" for item in calls),
            "unknown_call_timing": sum(item["seconds"] is None for item in calls),
            "duplicate_generation_ids": sorted(key for key, count in Counter(ids).items() if count > 1),
            "missing_generation_ids": len(calls) - len(ids),
            "preflight_status": preflight.get("status", "not_started"),
            "accounting_complete": campaign["status"] == "completed" and preflight.get("status") == "completed" and
                set(preflight.get("calls", {})) == {"CodeBundle", "Hint", "ExecutableKernelIR"} and
                all(case["usage_complete"] and case["timing_complete"] and case["status"] in ("completed", "failed")
                    for case in cases) and all(item["usage_complete"] and item["seconds"] is not None for item in calls) and
                len(ids) == len(set(ids)) == len(calls)}


def apply_reviews(audit: dict, root: Path, reviews: Path) -> None:
    audit["automatic_failure_counts"] = dict(audit["failure_counts"])
    audit["manual_reviews"] = {}
    reviewed_cases = set()
    categories = {"semantic", "unsupported_ir", "backend_api", "buffer_dataflow", "numerical", "model_output", "infrastructure"}
    for path in sorted(reviews.glob("*.json")):
        raw = path.read_bytes()
        review = json.loads(raw)
        if review.get("status") != "terminal_review":
            continue
        source = (Path(__file__).resolve().parents[1] / review["source_report"]).resolve()
        if not source.is_relative_to(root.resolve() / "cases") or source.name != "report.json":
            raise ValueError("manual review refers to another campaign")
        relative = str(source.relative_to(root.resolve()))
        if audit["input_sha256"].get(relative) != review.get("source_report_sha256") or not review.get("reviewer"):
            raise ValueError("manual review is stale or lacks reviewer attribution")
        case_id = source.parent.name
        if case_id in reviewed_cases:
            raise ValueError("duplicate terminal reviews for a case")
        if not any(case["id"] == case_id and case["status"] in ("completed", "failed", "blocked") for case in audit["cases"]):
            raise ValueError("manual review must refer to a terminal case")
        reviewed_cases.add(case_id)
        for cycle in review["cycles"]:
            category = cycle.get("failure_category")
            if category is None:
                continue
            if category not in categories:
                raise ValueError("unknown manual failure category")
            for stage in audit["stages"]:
                if stage["case_id"] == case_id and stage["cycle"] == cycle["cycle"] and stage["failure_category"] == "needs_review":
                    stage["failure_category"] = category
        audit["manual_reviews"][str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    audit["failure_counts"] = dict(Counter(item["failure_category"] for item in audit["stages"] if item["failure_category"]))


def write_audit(root: Path, output: Path, reviews: Path | None = None) -> dict:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("audit output must be outside the frozen campaign")
    audit = build_audit(root)
    if reviews is not None:
        apply_reviews(audit, root, reviews)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", audit)
    for name in ("cases", "stages"):
        if audit[name]:
            with (output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audit[name][0]))
                writer.writeheader()
                writer.writerows(audit[name])
    lines = ["# Research pilot audit v1", "", f"Campaign status: {audit['campaign_status']}.",
             f"Preflight status: {audit['preflight_status']}. Accounting complete: {audit['accounting_complete']}.", "",
             f"Known tokens: {audit['known_tokens']:,} ({audit['preflight_known_tokens']:,} preflight; "
             f"{audit['translation_known_tokens']:,} translation). Unknown-usage calls: {audit['unknown_usage_calls']} "
             f"({audit['in_flight_calls']} currently in flight).",
             "", f"Bound terminal manual reviews applied: {len(audit.get('manual_reviews', {}))}. "
             "Unreviewed stage failure labels are triage suggestions, not causal diagnoses. "
             "Counts are stage events, not unique failed cases; downstream prerequisites are omitted.",
             "Distributions include all started cases, including failures and interruptions; pending cases are not zero-cost trials.",
             "Known stage durations omit unrecorded orchestration and interrupted work; do not equate their sum with case active time.",
             "Repeated IR evidence is charged zero validation time when the runner records reuse. "
             "Reasoning tokens are already included in output tokens.", "",
             "| Category | Stage events |", "| --- | ---: |"]
    if audit["parallel_events"]:
        lines[2:2] = ["Execution changed from serial to parallel during this campaign. Case active time includes compiler queueing; summed active time is not campaign wall time. Timing comparisons are confounded by execution mode. See parallel_events in audit.json for affected cases and worker limits.", ""]
    lines += [f"| {category} | {count} |" for category, count in sorted(audit["failure_counts"].items())]
    lines += ["", "Full evidence, input hashes, cost quantiles, case status and per-stage traces: `audit.json`, `cases.csv`, `stages.csv`.",
              "This is offline validation evidence, not physical NPU correctness or kernel runtime performance.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    if any(case["status"] != "pending" for case in audit["cases"]):
        plot_audit(audit, output)
    return audit


def plot_audit(audit: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = audit["distributions"]
    labels = [f"{group['target']} / {group['arm']}\nstarted {group['started']}/{group['planned']}, finished {group['finished']}"
              for group in groups]
    figure, axes = plt.subplots(1, 3, figsize=(18, max(5, len(groups) * .65)), sharey=True)
    positions = np.arange(len(groups))
    left = np.zeros(len(groups))
    for stage, label in (("model_intermediate", "Intermediate"), ("model_code", "Backend code")):
        values = [sum(row["known_tokens"] for row in audit["stages"] if row["scope"] == "translation" and
                      row["target"] == group["target"] and row["arm"] == group["arm"] and row["stage"] == stage)
                  for group in groups]
        axes[0].barh(positions, values, left=left, label=label)
        left += values
    axes[0].set(xlabel="Known translation tokens (all started cases)", yticks=positions, yticklabels=labels)
    axes[0].legend(loc="best", fontsize=8)
    for axis, metric, label in ((axes[1], "known_tokens", "Known tokens per started case"),
                                (axes[2], "active_seconds", "Recorded active seconds per started case")):
        for position, group in enumerate(groups):
            rows = [row for row in audit["cases"] if row["target"] == group["target"] and
                    row["arm"] == group["arm"] and row["status"] != "pending"]
            for status, marker, color in (("completed", "o", "tab:green"), ("failed", "x", "tab:red"),
                                           ("blocked", "s", "tab:orange"), ("running", ">", "tab:blue"),
                                           ("paused", "D", "tab:purple")):
                values = [row[metric] for row in rows if row["status"] == status]
                axis.scatter(values, [position] * len(values), marker=marker, color=color, label=status if position == 0 else None)
        axis.set_xlabel(label)
        axis.set_xlim(left=0)
        axis.grid(axis="x", alpha=.2)
    axes[2].legend(loc="best", fontsize=8)
    axes[0].invert_yaxis()
    figure.suptitle(f"Pilot cost audit — {audit['campaign_status']}\nAll started cases; missing usage is not zero. "
                   f"Preflight: {audit['preflight_known_tokens']:,} known tokens, excluded from translation panels.")
    if audit.get("parallel_events"):
        figure.text(.5, .01, "Mixed serial/parallel execution: timing comparisons are confounded; summed active time is not campaign wall time.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .90))
    for extension in ("png", "svg"):
        figure.savefig(output / f"cost_breakdown.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read-only research audit of a frozen v4 campaign.")
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    result = write_audit(args.campaign, args.output, args.reviews)
    print(json.dumps({key: result[key] for key in ("campaign_status", "known_tokens", "unknown_usage_calls", "accounting_complete")}, indent=2))
