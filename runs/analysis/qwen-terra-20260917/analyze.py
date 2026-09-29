"""Read saved experiments only; write audit artifacts alongside this script."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import statistics
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
ROOTS = {
    "Qwen": REPO / "runs/case-study-v3-openrouter-interrupted-backup-20260917T031835Z",
    "Terra": REPO / "runs/case-study-v2-recovered",
}
COMMON = ("baseline", "structured_ir", "hinted_ir")
METHODS = ("baseline_minimal",) + COMMON
TARGETS = ("amd_xdna2_npu2", "intel_npu_4000")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    cases, calls, states, source_hashes = [], [], {}, {}
    for model, root in ROOTS.items():
        for path in sorted(root.glob("repeat-*/*/experiment.json")):
            repetition = int(path.parent.parent.name.split("-")[-1])
            experiment = read(path)
            metrics_path = path.parent / "metrics.jsonl"
            metrics = {r["run_id"]: r for r in map(json.loads, metrics_path.read_text().splitlines())}
            source_hashes[str(path)] = digest(path)
            source_hashes[str(metrics_path)] = digest(metrics_path)
            for key, case in experiment["cases"].items():
                report_path = path.parent / case["result_path"]
                state = read(report_path)
                states[model, repetition, key] = state
                source_hashes[str(report_path)] = digest(report_path)
                metric = metrics[case["run_id"]]
                cycles = state["cycles"]
                bundles = [json.dumps(c["bundle"]["files"], sort_keys=True) for c in cycles if "bundle" in c]
                row = {"model": model, "repetition": repetition, "key": key,
                       "method": case["method"], "target": case["target_id"], "kernel": case["kernel"],
                       "outcome": state["status"], "cycles": len(cycles),
                       "first_cycle_success": state.get("cycles_to_success") == 1,
                       "semantic_pass": any(c.get("semantic_validation", {}).get("status") == "passed" for c in cycles),
                       "target_compile_pass": any(c.get("compile_result", {}).get("validation", {}).get("stages", {}).get("target_compile", {}).get("status") == "passed" for c in cycles),
                       "repeated_adjacent_bundles": sum(a == b for a, b in zip(bundles, bundles[1:])),
                       "bundle_transitions": max(0, len(bundles) - 1),
                       "recorded_case_seconds": metric["end_to_end_seconds"],
                       "recorded_provider_seconds": metric["provider_seconds"],
                       "known_tokens": metric["known_total_tokens"], "usage_complete": metric["usage_complete"],
                       "failure_category": metric["failure_category"], "report": str(report_path)}
                cases.append(row)
                for cycle in cycles:
                    for stage, call in cycle["calls"].items():
                        meta = call.get("metadata", {})
                        usage = meta.get("usage", {})
                        content = call.get("response", {}).get("content", "")
                        nxt = cycles[cycle["number"]] if cycle["number"] < len(cycles) else {}
                        next_call = nxt.get("calls", {}).get(stage, {})
                        next_context = nxt.get("context", {}).get(stage)
                        calls.append({"model": model, "repetition": repetition, "key": key,
                                      "method": case["method"], "target": case["target_id"],
                                      "cycle": cycle["number"], "stage": stage, "status": call["status"],
                                      "finish_reason": meta.get("finish_reason"), "seconds": meta.get("duration_seconds"),
                                      "generation_id": meta.get("generation_id"),
                                      "returned_model": meta.get("returned_model"), "upstream_provider": meta.get("upstream_provider"),
                                      "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                                      "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                                      "total_tokens": usage.get("total_tokens"),
                                      "prompt_chars": len(call["prompt"]), "response_chars": len(content),
                                      "compression_ratio": len(zlib.compress(content.encode())) / len(content.encode()) if content else None,
                                      "tail": content[-180:] if call["status"] != "completed" else "",
                                      "next_prompt_chars": len(next_call.get("prompt", "")) or None,
                                      "raw_response_reused_in_next_context": bool(content and isinstance(next_context, dict) and next_context.get("content") == content),
                                      "error": call.get("error"), "report": str(report_path)})
    return cases, calls, states, source_hashes


def summarize(cases, calls, states):
    groups = []
    for model in ROOTS:
        for target in TARGETS:
            for method in METHODS:
                rows = [r for r in cases if (r["model"], r["target"], r["method"]) == (model, target, method)]
                if not rows:
                    continue
                counts = Counter(r["outcome"] for r in rows)
                groups.append({"model": model, "target": target, "method": method,
                               "recorded": len(rows), **{k: counts[k] for k in ("completed", "failed", "blocked")},
                               "first_cycle_success": sum(r["first_cycle_success"] for r in rows),
                               "semantic_pass_cases": sum(r["semantic_pass"] for r in rows),
                               "target_compile_pass_cases": sum(r["target_compile_pass"] for r in rows),
                               "cycles": sum(r["cycles"] for r in rows),
                               "successes_by_repetition": [sum(r["outcome"] == "completed" for r in rows if r["repetition"] == rep) for rep in sorted({r["repetition"] for r in rows})],
                               "failure_categories": dict(Counter(r["failure_category"] for r in rows if r["outcome"] != "completed"))})
    matches = Counter()
    for (model, repetition, key), state in states.items():
        if model != "Terra" or key.split(":")[0] not in COMMON:
            continue
        qwen = states["Qwen", 1, key]
        stage = "code" if key.startswith("baseline:") else "intermediate"
        a, b = state["cycles"][0]["calls"][stage], qwen["cycles"][0]["calls"][stage]
        matches["pairs"] += 1
        matches["identical_prompt_text"] += a["prompt"] == b["prompt"]
        matches["identical_saved_schema"] += a["schema"] == b["schema"]
        matches["identical_provider_schema_hash"] += a["metadata"].get("schema_sha256") == b["metadata"].get("schema_sha256")
    call_summary = {}
    for model in ROOTS:
        rows = [r for r in calls if r["model"] == model]
        capped = [r for r in rows if r["finish_reason"] == "length"]
        generation_ids = [r["generation_id"] for r in rows if r["generation_id"]]
        call_summary[model] = {"calls": len(rows), "status": dict(Counter(r["status"] for r in rows)),
                              "returned_generation_ids": len(generation_ids), "unique_generation_ids": len(set(generation_ids)),
                              "length_capped_calls": len(capped),
                              "length_capped_output_tokens": sum(r["output_tokens"] or 0 for r in capped),
                              "length_capped_seconds": sum(r["seconds"] or 0 for r in capped),
                              "known_call_seconds": sum(r["seconds"] or 0 for r in rows),
                              "known_output_tokens": sum(r["output_tokens"] or 0 for r in rows),
                              "median_known_call_seconds": statistics.median(r["seconds"] for r in rows if r["seconds"] is not None),
                              "capped_compression_ratio_range": [min(r["compression_ratio"] for r in capped), max(r["compression_ratio"] for r in capped)] if capped else None}
    campaign = {model: read(root / "campaign.json") for model, root in ROOTS.items()}
    controls = {"same_corpus": campaign["Qwen"]["inputs"]["corpus"] == campaign["Terra"]["inputs"]["corpus"], "toolchains": {}}
    for target in TARGETS:
        a, b = [campaign[m]["environment"]["toolchains"][target] for m in ROOTS]
        controls["toolchains"][target] = {k: a[k] == b[k] for k in ("image_id", "installed_tools", "compile_flags", "profile")}
        controls["toolchains"][target]["changed_evaluator_hashes"] = [k for k in a["evaluator_hashes"] if a["evaluator_hashes"][k] != b["evaluator_hashes"].get(k)]
    return {"groups": groups, "initial_prompt_comparison": dict(matches), "calls": call_summary, "controls": controls,
            "shared_arm_bundle_repetition": {model: {"identical": sum(r["repeated_adjacent_bundles"] for r in cases if r["model"] == model and r["method"] in COMMON),
                                                       "comparisons": sum(r["bundle_transitions"] for r in cases if r["model"] == model and r["method"] in COMMON)} for model in ROOTS},
            "qwen_outcomes": dict(Counter(r["outcome"] for r in cases if r["model"] == "Qwen")),
            "qwen_blocked_time_rows": [r for r in cases if r["model"] == "Qwen" and r["outcome"] == "blocked"],
            "qwen_capped_calls": [r for r in calls if r["model"] == "Qwen" and r["finish_reason"] == "length"]}


def plots(cases, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    colors = {"Qwen": "#d87934", "Terra": "#346a9b"}
    for ax, target, title in zip(axes, TARGETS, ("AMD XDNA2", "Intel NPU 4000")):
        for i, method in enumerate(METHODS):
            for model, offset in (("Qwen", -.19), ("Terra", .19)):
                rows = [g for g in summary["groups"] if (g["model"], g["target"], g["method"]) == (model, target, method)]
                if not rows:
                    ax.text(i + offset, 3, "n/a", ha="center", fontsize=9, color="#777777")
                    continue
                g = rows[0]
                rate = 100 * g["completed"] / g["recorded"]
                ax.bar(i + offset, rate, width=.35, color=colors[model], label=model if i == 1 else None)
                ax.text(i + offset, rate + 2, f'{g["completed"]}/{g["recorded"]}', ha="center", fontsize=10)
                if g["blocked"]:
                    upper = 100 * g["blocked"] / g["recorded"]
                    ax.bar(i + offset, upper, bottom=rate, width=.35, facecolor="none", edgecolor=colors[model], hatch="///", linewidth=1)
        ax.set_title(title, weight="bold")
        ax.set_xticks(range(4), ("Minimal\ndirect", "Guided\ndirect", "Structured\nIR", "Hinted\nIR"))
        ax.set_ylim(0, 115)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Offline acceptance among recorded cases (%)")
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle("Qwen’s gap persists after correcting the partial-run denominator", fontsize=16, weight="bold", y=.98)
    fig.text(.5, .025, "Qwen: first repetition, 10 cases per cell. Terra: 3 repetitions, 30 cases per cell.\nHatching: unresolved interrupted cases, not additional successes. Different model/provider/reasoning settings.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, .92))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"acceptance_comparison.{ext}", dpi=180, facecolor="white")
    plt.close(fig)

    kernels = sorted({r["kernel"] for r in cases})
    labels = ["Qwen" + "\n" + m.replace("_", " ") for m in METHODS] + ["Terra" + "\n" + m.replace("_", " ") for m in COMMON]
    columns = [("Qwen", m) for m in METHODS] + [("Terra", m) for m in COMMON]
    palette = LinearSegmentedColormap.from_list("outcomes", ("#efb3a3", "#f1e7ad", "#4c9a76"))
    palette.set_bad("#d9dfe5")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), sharey=True)
    for ax, target, title in zip(axes, TARGETS, ("AMD XDNA2", "Intel NPU 4000")):
        values = np.zeros((10, 7))
        for y, kernel in enumerate(kernels):
            for x, (model, method) in enumerate(columns):
                rows = [r for r in cases if (r["model"], r["target"], r["method"], r["kernel"]) == (model, target, method, kernel)]
                solved = sum(r["outcome"] == "completed" for r in rows)
                blocked = any(r["outcome"] == "blocked" for r in rows)
                values[y, x] = -.3 if blocked else solved / len(rows)
        ax.imshow(np.ma.masked_less(values, 0), cmap=palette, vmin=0, vmax=1, aspect="auto")
        for y in range(10):
            for x, (model, _) in enumerate(columns):
                value = values[y, x]
                label = "?" if value < 0 else f"{int(round(value * (1 if model == 'Qwen' else 3)))}/{1 if model == 'Qwen' else 3}"
                ax.text(x, y, label, ha="center", va="center", fontsize=10)
        ax.axvline(3.5, color="white", linewidth=3)
        ax.set_title(title, weight="bold")
        ax.set_xticks(range(7), labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(10), [k.replace("cuda_", "").replace("hip_", "").replace("triton_", "").replace("_", " ") for k in kernels])
    fig.suptitle("Recorded case outcomes: solved / available repetitions", fontsize=16, weight="bold")
    fig.text(.5, .012, "? = interrupted response and unknown outcome. Qwen has one repetition; Terra has three. Offline validation only.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .94))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"kernel_outcomes.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cases, calls, states, hashes = collect()
    result = summarize(cases, calls, states)
    assert len([r for r in cases if r["model"] == "Qwen"]) == 80
    assert len([r for r in cases if r["model"] == "Terra"]) == 300
    assert result["qwen_outcomes"] == {"completed": 28, "failed": 46, "blocked": 6}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for name, rows in (("cases.csv", cases), ("calls.csv", calls)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plots(cases, result)
    assert all(digest(Path(p)) == h for p, h in hashes.items()), "Source evidence changed during analysis"
    print(json.dumps({k: result[k] for k in ("initial_prompt_comparison", "calls", "qwen_outcomes")}, indent=2))
