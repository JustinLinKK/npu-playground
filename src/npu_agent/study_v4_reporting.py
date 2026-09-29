from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .providers import normalize_usage
from .study_v4 import StudyConfig
from .validation import write_json


def case_metrics(case: dict, state: dict | None, config: StudyConfig) -> dict:
    row = {**case, "status": "pending", "offline_pass": False, "success": False, "known_tokens": 0,
           "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_input_tokens": 0,
           "usage_complete": False, "timing_complete": False, "active_seconds": 0.0, "llm_calls": 0,
           "cycles": 0, "backend_attempts": 0, "semantic_failures": 0, "identical_backend_pairs": 0,
           "length_finishes": 0, "model_seconds": 0.0}
    if state is None:
        return row
    row.update(status=state["status"], offline_pass=state["status"] == "completed",
               usage_complete=not state.get("usage_incomplete", False),
               timing_complete=not state.get("timing_incomplete", False) and state.get("phase") is None,
               active_seconds=state["active_seconds"], cycles=len(state["cycles"]))
    bundles = []
    for cycle in state["cycles"]:
        row["backend_attempts"] += cycle.get("backend_test_attempts", 0)
        row["semantic_failures"] += cycle.get("semantic_validation", {}).get("status") == "failed"
        if "bundle" in cycle:
            bundles.append(cycle["bundle"]["files"])
        for call in cycle["calls"].values():
            row["llm_calls"] += 1
            metadata = call.get("metadata", {})
            counts = normalize_usage(metadata.get("usage", {}))
            row["known_tokens"] += counts["total_tokens"] or 0
            for dest, src in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                              ("reasoning_tokens", "reasoning_output_tokens"), ("cached_input_tokens", "cached_input_tokens")):
                if row[dest] is not None:
                    row[dest] = row[dest] + counts[src] if counts[src] is not None else None
            row["usage_complete"] &= (counts["input_tokens"] is not None and counts["output_tokens"] is not None and
                                      not metadata.get("usage_incomplete", False) and call["status"] != "in_flight")
            row["length_finishes"] += metadata.get("finish_reason") == "length"
            row["model_seconds"] += metadata.get("duration_seconds", 0.0)
    row["identical_backend_pairs"] = sum(left == right for left, right in zip(bundles, bundles[1:]))
    row["success"] = bool(row["offline_pass"] and row["usage_complete"] and row["timing_complete"] and
                          row["known_tokens"] <= config.token_budget and row["active_seconds"] <= config.active_seconds_budget)
    return row


def paired_comparison(rows: list[dict], config: StudyConfig, target: str, comparator: str) -> dict:
    selected = [row for row in rows if row["target"] == target and row["arm"] in ("structured_ir", comparator)]
    complete = all(row["status"] in ("completed", "failed") and row["usage_complete"] and row["timing_complete"] for row in selected)
    pairs = {}
    for row in selected:
        pairs.setdefault((row["kernel"], row["repetition"]), {})[row["arm"]] = row
    complete &= bool(pairs) and all(set(pair) == {"structured_ir", comparator} for pair in pairs.values())
    result = {"target": target, "comparator": comparator, "paired_cases": len(pairs), "evidence_complete": complete,
              "verdict": "inconclusive", "intervals": {}, "reasons": []}
    if not complete:
        result["reasons"].append("pending, paused, blocked, unpaired, or incomplete-usage/timing cases")
        return result
    # Unsuccessful attempts get the full budget, never a reward for failing cheaply.
    families = {}
    for pair in pairs.values():
        left, right = pair["structured_ir"], pair[comparator]
        delta = [int(left["success"]) - int(right["success"])]
        for name, budget in (("known_tokens", config.token_budget), ("active_seconds", config.active_seconds_budget),
                             ("cycles", config.max_cycles)):
            delta.append((left[name] / budget if left["success"] else 1.0) -
                         (right[name] / budget if right["success"] else 1.0))
        families.setdefault(left["family"], []).append(delta)
    values = np.asarray([np.mean(items, axis=0) for items in families.values()])
    result["families"] = len(values)
    # Predeclared family-wise alpha .05 across targets, comparators and four metrics.
    alpha = 0.05 / (len(config.targets) * 2 * 4)
    result["interval_confidence"] = 1 - alpha
    rng = np.random.default_rng(config.seed)
    samples = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    bounds = np.quantile(samples, [alpha / 2, 1 - alpha / 2], axis=0)
    for index, name in enumerate(("success_delta", "penalized_token_delta", "penalized_time_delta", "penalized_cycle_delta")):
        result["intervals"][name] = {"estimate": float(values[:, index].mean()),
                                    "low": float(bounds[0, index]), "high": float(bounds[1, index])}
    result["observed_joint_benefit"] = bool(values[:, 0].mean() >= 0 and np.all(values[:, 1:].mean(axis=0) < 0))
    if config.stage != "confirmatory":
        result["reasons"].append("development pilot; intervals are descriptive")
    if len(values) < 10:
        result["reasons"].append("fewer than ten reviewed kernel-family clusters")
    if result["reasons"]:
        return result
    intervals = result["intervals"]
    if intervals["success_delta"]["high"] < 0:
        result["verdict"] = "structured_ir_reduces_success"
        return result
    if any(intervals[name]["low"] > 0 for name in ("penalized_token_delta", "penalized_time_delta", "penalized_cycle_delta")):
        result["verdict"] = "joint_efficiency_claim_not_supported"
        return result
    # A degenerate bootstrap from all success ties cannot establish zero loss on unseen kernels.
    if np.all(values[:, 0] == 0):
        result["reasons"].append("all success differences are zero; bootstrap cannot establish population noninferiority")
        return result
    noninferior = intervals["success_delta"]["low"] >= 0
    savings = all(intervals[name]["high"] < 0 for name in ("penalized_token_delta", "penalized_time_delta", "penalized_cycle_delta"))
    if noninferior and savings:
        result["verdict"] = "supported"
    return result


def build_report(root: Path) -> dict:
    campaign = json.loads((root / "campaign.json").read_text())
    config = StudyConfig.model_validate(campaign["config"])
    rows = []
    for case in campaign["cases"]:
        path = root / "cases" / case["id"] / "report.json"
        rows.append(case_metrics(case, json.loads(path.read_text()) if path.exists() else None, config))
    groups = []
    for target in config.targets:
        for arm in config.arms:
            selected = [row for row in rows if row["target"] == target and row["arm"] == arm]
            groups.append({"target": target, "arm": arm, "planned": len(selected),
                           "finished": sum(row["status"] in ("completed", "failed") for row in selected),
                           "blocked": sum(row["status"] == "blocked" for row in selected),
                           "offline_passes": sum(row["offline_pass"] for row in selected),
                           "budget_successes": sum(row["success"] for row in selected),
                           "known_tokens": sum(row["known_tokens"] for row in selected),
                           "active_seconds": sum(row["active_seconds"] for row in selected),
                           "llm_calls": sum(row["llm_calls"] for row in selected),
                           "cycles": sum(row["cycles"] for row in selected),
                           "backend_attempts": sum(row["backend_attempts"] for row in selected),
                           "usage_complete": all(row["usage_complete"] for row in selected),
                           "timing_complete": all(row["timing_complete"] for row in selected)})
    paired = [paired_comparison(rows, config, target, comparator) for target in config.targets
              for comparator in ("direct", "hinted_ir")]
    generation_ids = []
    for path in (root / "cases").glob("*/report.json"):
        state = json.loads(path.read_text())
        for cycle in state["cycles"]:
            for call in cycle["calls"].values():
                if call.get("metadata", {}).get("generation_id"):
                    generation_ids.append(call["metadata"]["generation_id"])
    duplicate_ids = len(generation_ids) != len(set(generation_ids))
    if duplicate_ids:
        for item in paired:
            item.update(verdict="inconclusive", evidence_complete=False)
            item["reasons"].append("duplicate provider generation IDs")
    curves = []
    for group in groups:
        selected = [row for row in rows if row["target"] == group["target"] and row["arm"] == group["arm"]]
        for metric, budget in (("known_tokens", config.token_budget), ("active_seconds", config.active_seconds_budget),
                               ("cycles", config.max_cycles)):
            thresholds = {fraction * budget for fraction in (0, 0.1, 0.25, 0.5, 0.75, 1.0)}
            thresholds.update(row[metric] for row in selected if row["success"])
            for threshold in sorted(thresholds):
                curves.append({"target": group["target"], "arm": group["arm"], "metric": metric,
                               "budget": threshold, "successes": sum(row["success"] and row[metric] <= threshold for row in selected),
                               "planned": len(selected), "finished": group["finished"]})
    preflight_path = root / "preflight.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    preflight_calls = list(preflight.get("calls", {}).values())
    preflight_tokens = sum(normalize_usage((call.get("metadata") or {}).get("usage", {}))["total_tokens"] or 0 for call in preflight_calls)
    summary = {"protocol": config.protocol, "stage": config.stage, "status": campaign["status"],
               "groups": groups, "paired": paired, "budget_curves": curves, "duplicate_generation_ids": duplicate_ids,
               "preflight_known_tokens": preflight_tokens,
               "campaign_known_tokens": preflight_tokens + sum(row["known_tokens"] for row in rows),
               "joint_claim_supported": all(item["verdict"] == "supported" for item in paired)}
    write_json(root / "summary.json", summary)
    with (root / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Case study v4", "", f"Stage: {config.stage}. Status: {campaign['status']}. Model: {config.provider.model}.",
             "", "Success means the complete offline contract passed within both declared budgets. Physical NPU execution is not tested.",
             "Incomplete campaigns show progress counts, not final accuracy estimates. Token totals below are known expenditure, including failures.",
             "Active seconds include generation and validation and exclude intentional pauses. Cases run serially; campaign calendar span includes pauses.", "",
             "| Target | Arm | Budget successes | Finished / planned | Blocked | Known tokens | Active minutes | Calls | Cycles | Backend attempts |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in groups:
        lines.append(f"| {group['target']} | {group['arm']} | {group['budget_successes']} | {group['finished']}/{group['planned']} | "
                     f"{group['blocked']} | {group['known_tokens']:,}{'' if group['usage_complete'] else ' (incomplete)'} | "
                     f"{group['active_seconds']/60:.1f}{'' if group['timing_complete'] else ' (incomplete)'} | {group['llm_calls']} | "
                     f"{group['cycles']} | {group['backend_attempts']} |")
    lines += ["", "Paired decisions use family-cluster bootstrap intervals with multiplicity adjustment; repeated trials are not independent kernels.",
              "Failures receive the full budget in penalized efficiency metrics. Savings on jointly solved cases alone cannot establish the claim.", ""]
    for item in paired:
        lines.append(f"- {item['target']}, structured IR vs {item['comparator']}: **{item['verdict']}**. " + "; ".join(item["reasons"]))
    lines += ["", "Full intervals, per-case diagnostics and success-budget curves: `summary.json` and `cases.csv`.",
              f"Known preflight tokens: {preflight_tokens:,}. Known total campaign tokens: {summary['campaign_known_tokens']:,}. "
              "Unknown usage cannot be reconstructed from a lost response. Preflight is excluded from translation comparisons.", ""]
    (root / "results.md").write_text("\n".join(lines))
    if any(group["finished"] for group in groups):
        plot_curves(root, config, curves, campaign["status"])
    return summary


def plot_curves(root: Path, config: StudyConfig, curves: list[dict], status: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, len(config.targets), figsize=(6 * len(config.targets), 10), squeeze=False)
    for column, target in enumerate(config.targets):
        for row, (metric, label) in enumerate((("known_tokens", "Total tokens per case"),
                                               ("active_seconds", "Active seconds per case"), ("cycles", "Cycles per case"))):
            axis = axes[row, column]
            for arm in config.arms:
                points = [point for point in curves if point["target"] == target and point["arm"] == arm and point["metric"] == metric]
                axis.step([0] + [point["budget"] for point in points], [0] + [point["successes"] for point in points],
                          where="post", label=arm)
            axis.set(xlabel=label, ylabel="Verified successes", title=target)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"V4 {config.stage} — {status}\nSuccesses must also meet the other resource budgets; unresolved cases are not failures.")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "svg"):
        figure.savefig(root / f"success_budgets.{extension}", dpi=180)
    plt.close(figure)
