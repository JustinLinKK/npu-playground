"""Read historical CSVs without changing campaigns; retain overlapping runs separately."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
paths = [REPOSITORY / "runs/ir-evolution-20260911/comparison/cases.csv"]
paths += sorted((REPOSITORY / "runs").glob("case-study*/cases.csv"))
hashes = {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
datasets = {}
summary = {}
for path in paths:
    rows = list(csv.DictReader(path.open()))
    key = str(path.parent.relative_to(REPOSITORY))
    datasets[key] = rows
    groups = {}
    for row in rows:
        name = (row.get("label") or row["method"]) + " / " + row["target_id"]
        group = groups.setdefault(name, dict(recorded=0, solved=0, exhausted=0, unresolved=0,
                                            known_tokens=0, cycles=0, active_minutes=0.0))
        group["recorded"] += 1
        group["solved"] += row["success"].lower() == "true"
        group["exhausted"] += row["terminal_reason"] == "cycle_budget_exhausted"
        group["unresolved"] += row["terminal_reason"] not in ("solved", "cycle_budget_exhausted")
        group["known_tokens"] += int(row["known_total_tokens"] or 0)
        group["cycles"] += int(row["cycles_used"] or 0)
        group["active_minutes"] += float(row["end_to_end_seconds"] or 0) / 60
    summary[key] = groups

figure, axes = plt.subplots(2, 2, figsize=(14, 9))
panels = [
    ("Original Terra — 20 cases per arm", "runs/ir-evolution-20260911/comparison", "label",
     ["Frozen baseline", "Original structured IR", "Frozen hinted IR"], ["Direct", "Structured IR", "Hinted IR"]),
    ("Terra development revisions — 20 cases per pipeline", "runs/ir-evolution-20260911/comparison", "label",
     ["Frozen baseline", "IR revision 1", "IR revision 2", "Frozen hinted IR"], ["Frozen\ndirect", "IR revision 1", "IR revision 2", "Frozen\nhinted"]),
    ("Extended Terra — 60 cases per arm", "runs/case-study-v2-recovered", "method",
     ["baseline", "structured_ir", "hinted_ir", "structured_ir_no_validation", "structured_ir_no_reuse"],
     ["Guided\ndirect", "Structured\nIR", "Hinted\nIR", "IR, no\nsemantic gate", "IR, no\nreuse"]),
    ("Qwen frozen backup — 20 recorded cases per arm", "runs/case-study-v3-openrouter-interrupted-backup-20260917T031835Z", "method",
     ["baseline_minimal", "baseline", "structured_ir", "hinted_ir"], ["Minimal\ndirect", "Guided\ndirect", "Structured IR", "Hinted IR"]),
]
colors = ["#566574", "#167b8d", "#c17920", "#735d97", "#39805a"]
for axis, (title, key, field, selected, labels) in zip(axes.flat, panels):
    counts, unresolved, denominators = [], [], []
    for value in selected:
        rows = [row for row in datasets[key] if row[field] == value]
        counts.append(sum(row["success"].lower() == "true" for row in rows))
        unresolved.append(sum(row["terminal_reason"] not in ("solved", "cycle_budget_exhausted") for row in rows))
        denominators.append(len(rows))
    assert all(denominators)
    rates = [100 * n / d for n, d in zip(counts, denominators)]
    unknown = [100 * n / d for n, d in zip(unresolved, denominators)]
    bars = axis.bar(labels, rates, color=colors[:len(labels)])
    axis.bar(labels, unknown, bottom=rates, color="none", edgecolor="#888888", hatch="///", label="Unresolved")
    for bar, count, denominator in zip(bars, counts, denominators):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{count}/{denominator}", ha="center")
    axis.set(title=title, ylim=(0, 114), ylabel="Offline passes / recorded cases (%)")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
axes[1, 1].legend(loc="upper right", fontsize=9)
figure.suptitle("Historical outcomes: promising development gains, no established IR-only advantage", fontsize=15)
figure.text(0.5, 0.015, "IR revisions also changed backend/numerical guidance; their baselines stayed frozen. Qwen is one partial repetition.\n"
            "Recovered and backup directories overlap and must not be pooled as independent observations.", ha="center", fontsize=10)
figure.tight_layout(rect=(0, 0.06, 1, 0.95))
for extension in ("png", "svg"):
    figure.savefig(OUTPUT / f"historical_outcomes.{extension}", dpi=180)
plt.close(figure)
(OUTPUT / "historical-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUTPUT / "source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
for relative, digest in hashes.items():
    assert hashlib.sha256((REPOSITORY / relative).read_bytes()).hexdigest() == digest
print("Historical CSV hashes unchanged; wrote separate summaries and PNG/SVG.")
