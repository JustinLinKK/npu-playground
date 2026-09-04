from __future__ import annotations

import csv
import html
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRIC_COLUMNS = [
    "experiment_id",
    "method",
    "kernel",
    "target_id",
    "backend",
    "run_id",
    "outcome_status",
    "success",
    "provider_schema_valid",
    "compile_success",
    "correctness",
    "amd_compile_valid",
    "intel_cpu_equivalent",
    "target_npu_correct",
    "evidence_tier",
    "reward",
    "failure_category",
    "end_to_end_seconds",
    "provider_seconds",
    "compiler_seconds",
    "debug_seconds",
    "evaluation_seconds",
    "llm_calls",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
    "candidates",
    "compile_attempts",
    "debug_attempts",
    "artifact_bytes",
    "static_score",
    "intel_cpu_p50_ms",
    "intel_cpu_p95_ms",
    "target_npu_p50_ms",
    "target_npu_p95_ms",
    "target_npu_warmups",
    "target_npu_iterations",
    "target_npu_throughput_per_second",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _token_totals(usages: Iterable[str]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }
    for raw in usages:
        usage = json.loads(raw or "{}")
        for key in totals:
            if isinstance(usage.get(key), (int, float)):
                totals[key] += int(usage[key])
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


def _failure_category(
    case: dict[str, Any], provider_valid: bool, evaluation: dict[str, Any] | None
) -> str | None:
    if case.get("error") and not provider_valid:
        return "provider_or_schema"
    if evaluation:
        if not evaluation.get("compile_success"):
            return "compile"
        if evaluation.get("correctness") is False:
            return "correctness"
    if case.get("result_status") not in {"completed", "partial"}:
        return "translation"
    return None


def collect_metrics(experiment_root: Path) -> list[dict[str, Any]]:
    experiment = _read_json(experiment_root / "experiment.json")
    connection = sqlite3.connect(experiment_root / "state.sqlite")
    connection.row_factory = sqlite3.Row
    records: list[dict[str, Any]] = []
    try:
        for case_id, case in sorted(experiment["cases"].items()):
            del case_id
            result: dict[str, Any] = {}
            result_path = case.get("result_path")
            if result_path and (experiment_root / result_path).is_file():
                result = _read_json(experiment_root / result_path)
            run_id = case.get("run_id")
            target_result = (result.get("targets") or [{}])[0]
            candidate = target_result.get("candidate") or target_result.get("winner") or {}
            evaluation = candidate.get("evaluation")
            calls: list[sqlite3.Row] = []
            candidates: list[sqlite3.Row] = []
            compiles: list[sqlite3.Row] = []
            evaluations: list[sqlite3.Row] = []
            run: sqlite3.Row | None = None
            if run_id:
                run = connection.execute("SELECT * FROM app_runs WHERE id=?", (run_id,)).fetchone()
                calls = connection.execute("SELECT * FROM agent_calls WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
                candidates = connection.execute(
                    "SELECT * FROM candidates WHERE run_id=? ORDER BY created_at, id", (run_id,)
                ).fetchall()
                compiles = connection.execute(
                    """SELECT ca.* FROM compile_attempts ca JOIN candidates c ON c.id=ca.candidate_id
                       WHERE c.run_id=? ORDER BY ca.id""",
                    (run_id,),
                ).fetchall()
                evaluations = connection.execute(
                    """SELECT e.evaluation_json FROM evaluations e JOIN candidates c ON c.id=e.candidate_id
                       WHERE c.run_id=?""",
                    (run_id,),
                ).fetchall()
            provider_valid = bool(calls) and all(bool(row["schema_valid"]) for row in calls)
            tokens = _token_totals(row["usage_json"] for row in calls)
            compile_total = sum(float(row["duration_seconds"] or 0) for row in compiles)
            evaluation_total = sum(float(row["evaluation_duration_seconds"] or 0) for row in compiles)
            evaluation_total += sum(
                float(json.loads(row["evaluation_json"]).get("duration_seconds", 0)) for row in evaluations
            )
            latest_compile: dict[str, Any] = {}
            if candidate.get("id"):
                row = connection.execute(
                    "SELECT result_json FROM compile_attempts WHERE candidate_id=? ORDER BY id DESC LIMIT 1",
                    (candidate["id"],),
                ).fetchone()
                if row:
                    latest_compile = json.loads(row["result_json"] or "{}")
            compile_success = bool(evaluation and evaluation.get("compile_success"))
            success = target_result.get("status") == "completed" and compile_success
            record = {
                "experiment_id": experiment["experiment_id"],
                "method": case["method"],
                "kernel": case["kernel"],
                "target_id": case["target_id"],
                "backend": case["backend"],
                "run_id": run_id,
                "outcome_status": case.get("result_status", "failed"),
                "success": success,
                "provider_schema_valid": provider_valid,
                "compile_success": compile_success,
                "correctness": evaluation.get("correctness") if evaluation else None,
                "amd_compile_valid": compile_success if case["backend"] == "amd_xdna2" else None,
                "intel_cpu_equivalent": (
                    evaluation.get("host_correctness") if evaluation and case["backend"] == "intel_openvino" else None
                ),
                "target_npu_correct": evaluation.get("target_correctness") if evaluation else None,
                "evidence_tier": evaluation.get("evidence_tier") if evaluation else 0,
                "reward": evaluation.get("reward") if evaluation else 0.0,
                "failure_category": _failure_category(case, provider_valid, evaluation),
                "end_to_end_seconds": (
                    case.get("duration_seconds")
                    if case.get("duration_seconds") is not None
                    else float(run["duration_seconds"] or 0) if run else None
                ),
                "provider_seconds": sum(float(row["duration_seconds"] or 0) for row in calls),
                "compiler_seconds": max(0.0, compile_total - evaluation_total),
                "debug_seconds": sum(
                    float(row["duration_seconds"] or 0) for row in calls if row["role"] == "debug"
                ),
                "evaluation_seconds": evaluation_total,
                "llm_calls": len(calls),
                **tokens,
                "candidates": len(candidates),
                "compile_attempts": len(compiles),
                "debug_attempts": sum(int(row["debug_attempt"] > 0) for row in candidates),
                "artifact_bytes": evaluation.get("artifact_bytes", 0) if evaluation else 0,
                "static_score": evaluation.get("static_score", 0.0) if evaluation else 0.0,
                "intel_cpu_p50_ms": evaluation.get("cpu_latency_p50_ms") if evaluation else None,
                "intel_cpu_p95_ms": evaluation.get("cpu_latency_p95_ms") if evaluation else None,
                "target_npu_p50_ms": evaluation.get("latency_p50_ms") if evaluation else None,
                "target_npu_p95_ms": evaluation.get("latency_p95_ms") if evaluation else None,
                "target_npu_warmups": latest_compile.get("hardware_warmup_count"),
                "target_npu_iterations": latest_compile.get("hardware_iteration_count"),
                "target_npu_throughput_per_second": latest_compile.get("hardware_throughput_per_second"),
            }
            records.append(record)
    finally:
        connection.close()
    return records


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["method"], record["target_id"])].append(record)
    summary_groups = []
    for (method, target), values in sorted(groups.items()):
        successes = sum(bool(item["success"]) for item in values)
        durations = [item["end_to_end_seconds"] for item in values if item["end_to_end_seconds"] is not None]
        summary_groups.append(
            {
                "method": method,
                "target_id": target,
                "successes": successes,
                "total": len(values),
                "success_rate": successes / len(values) if values else 0.0,
                "mean_end_to_end_seconds": sum(durations) / len(durations) if durations else None,
                "mean_llm_calls": sum(item["llm_calls"] for item in values) / len(values) if values else 0.0,
                "mean_compile_attempts": (
                    sum(item["compile_attempts"] for item in values) / len(values) if values else 0.0
                ),
                "mean_reward": sum(float(item["reward"]) for item in values) / len(values) if values else 0.0,
            }
        )
    return {"cases": len(records), "groups": summary_groups}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_metric_files(experiment_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    jsonl = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    _atomic_text(experiment_root / "metrics.jsonl", jsonl)
    csv_path = experiment_root / "metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, csv_path)
    summary = summarize_metrics(records)
    _atomic_text(experiment_root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_audit_logs(experiment_root: Path) -> None:
    connection = sqlite3.connect(experiment_root / "state.sqlite")
    connection.row_factory = sqlite3.Row
    try:
        agent_rows = connection.execute(
            """SELECT r.experiment_id, r.method, r.kernel_name, a.*
               FROM agent_calls a JOIN app_runs r ON r.id=a.run_id
               ORDER BY a.id"""
        ).fetchall()
        compile_rows = connection.execute(
            """SELECT r.experiment_id, r.method, r.kernel_name, c.target_id, c.label,
                      c.debug_attempt, ca.*
               FROM compile_attempts ca JOIN candidates c ON c.id=ca.candidate_id
               JOIN app_runs r ON r.id=c.run_id ORDER BY ca.id"""
        ).fetchall()
    finally:
        connection.close()
    _atomic_text(
        experiment_root / "logs" / "provider_calls.jsonl",
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in agent_rows),
    )
    _atomic_text(
        experiment_root / "logs" / "compile_attempts.jsonl",
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in compile_rows),
    )


def _plot_module():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["svg.hashsalt"] = "npu-agent"
    except ImportError as exc:
        raise RuntimeError("experiment charts require the package dependencies; run: uv sync") from exc
    return plt


def _save_figure(figure: Any, charts: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(charts / f"{name}.png", dpi=200, bbox_inches="tight")
    figure.savefig(charts / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})


def _ordered(records: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    kernels = sorted({item["kernel"] for item in records})
    targets = sorted({item["target_id"] for item in records})
    methods = [method for method in ("baseline", "agentic") if any(r["method"] == method for r in records)]
    return kernels, targets, methods


def _aggregate_chart(plt: Any, records: list[dict[str, Any]], charts: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["target_id"], record["method"])].append(record)
    labels = [f"{target}\n{method}" for target, method in sorted(groups)]
    values = [sum(r["success"] for r in groups[key]) / len(groups[key]) for key in sorted(groups)]
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
    bars = axis.bar(labels, values, color=["#7389ae" if "baseline" in label else "#2a9d8f" for label in labels])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Success rate")
    axis.set_title("Compile/validation success by backend (no cross-backend ranking)")
    for bar, key in zip(bars, sorted(groups), strict=True):
        values_for_key = groups[key]
        successful = sum(r["success"] for r in values_for_key)
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{successful}/{len(values_for_key)}", ha="center")
    _save_figure(figure, charts, "aggregate_success")
    plt.close(figure)


def _heatmap(plt: Any, records: list[dict[str, Any]], charts: Path) -> None:
    kernels, targets, methods = _ordered(records)
    columns = [(target, method) for target in targets for method in methods]
    lookup = {(r["kernel"], r["target_id"], r["method"]): r for r in records}
    values = []
    annotations = []
    for kernel in kernels:
        row = []
        note_row = []
        for target, method in columns:
            record = lookup.get((kernel, target, method))
            row.append(1 if record and record["success"] else 0)
            note_row.append(f"{'OK' if record and record['success'] else 'FAIL'}\nT{record['evidence_tier'] if record else 0}")
        values.append(row)
        annotations.append(note_row)
    figure, axis = plt.subplots(figsize=(max(9, len(columns) * 2), max(5, len(kernels) * 0.65)))
    axis.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(columns)), [f"{target}\n{method}" for target, method in columns])
    axis.set_yticks(range(len(kernels)), kernels)
    for y, row in enumerate(annotations):
        for x, value in enumerate(row):
            axis.text(x, y, value, ha="center", va="center", fontsize=8)
    axis.set_title("Per-kernel success and evidence tier")
    _save_figure(figure, charts, "kernel_evidence_heatmap")
    plt.close(figure)


def _paired_times(plt: Any, records: list[dict[str, Any]], charts: Path) -> None:
    kernels, targets, methods = _ordered(records)
    lookup = {(r["kernel"], r["target_id"], r["method"]): r for r in records}
    figure, axes = plt.subplots(len(targets), 1, figsize=(max(10, len(kernels) * 1.1), 4.5 * len(targets)), squeeze=False)
    width = 0.36
    for axis, target in zip(axes[:, 0], targets, strict=True):
        positions = list(range(len(kernels)))
        for offset, method in zip((-width / 2, width / 2), methods, strict=False):
            values = [float(lookup.get((k, target, method), {}).get("end_to_end_seconds") or 0) for k in kernels]
            axis.bar([value + offset for value in positions], values, width, label=method)
        axis.set_xticks(positions, kernels, rotation=35, ha="right")
        axis.set_ylabel("Seconds")
        axis.set_title(f"Paired end-to-end time: {target}")
        axis.legend()
    _save_figure(figure, charts, "paired_end_to_end_time")
    plt.close(figure)


def _breakdown(plt: Any, records: list[dict[str, Any]], charts: Path) -> None:
    labels = [f"{r['kernel']}\n{r['target_id']}\n{r['method']}" for r in records]
    figure, axis = plt.subplots(figsize=(max(12, len(labels) * 0.7), 6))
    bottom = [0.0] * len(records)
    for field, label, color in (
        ("provider_seconds", "provider", "#457b9d"),
        ("compiler_seconds", "compiler", "#e9c46a"),
        ("debug_seconds", "debug provider", "#e76f51"),
        ("evaluation_seconds", "evaluation", "#2a9d8f"),
    ):
        values = [
            max(0.0, float(record[field] or 0) - float(record["debug_seconds"] or 0))
            if field == "provider_seconds"
            else float(record[field] or 0)
            for record in records
        ]
        axis.bar(range(len(records)), values, bottom=bottom, label=label, color=color)
        bottom = [left + value for left, value in zip(bottom, values, strict=True)]
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    axis.set_ylabel("Seconds")
    axis.set_title("Recorded provider, compile, debug, and evaluation time")
    axis.legend()
    _save_figure(figure, charts, "time_breakdown")
    plt.close(figure)


def _effort(plt: Any, records: list[dict[str, Any]], charts: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["target_id"], record["method"])].append(record)
    labels = [f"{target}\n{method}" for target, method in sorted(groups)]
    fields = [("llm_calls", "LLM calls"), ("candidates", "Candidates"), ("compile_attempts", "Compiles"), ("debug_attempts", "Repairs")]
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for axis, (field, title) in zip(axes, fields, strict=True):
        values = [sum(float(r[field]) for r in groups[key]) / len(groups[key]) for key in sorted(groups)]
        axis.bar(labels, values, color="#6d597a")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
    figure.suptitle("Mean search and repair effort per kernel")
    _save_figure(figure, charts, "search_effort")
    plt.close(figure)


def _artifact_reward(plt: Any, records: list[dict[str, Any]], charts: Path) -> None:
    labels = [f"{r['kernel']}\n{r['method']}" for r in records]
    figure, axes = plt.subplots(2, 1, figsize=(max(12, len(labels) * 0.6), 8))
    axes[0].bar(range(len(records)), [r["artifact_bytes"] for r in records], color="#f4a261")
    axes[0].set_ylabel("Bytes")
    axes[0].set_title("Winner artifact size")
    axes[1].bar(range(len(records)), [r["reward"] for r in records], color="#2a9d8f")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Evidence-aware reward")
    for axis in axes:
        axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    _save_figure(figure, charts, "artifact_size_and_reward")
    plt.close(figure)


def _latency_charts(plt: Any, records: list[dict[str, Any]], charts: Path) -> None:
    intel = [r for r in records if r["target_id"] == "intel_npu_4000" and r["intel_cpu_p50_ms"] is not None]
    if intel:
        labels = [f"{r['kernel']}\n{r['method']}" for r in intel]
        figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.8), 5))
        axis.plot(range(len(intel)), [r["intel_cpu_p50_ms"] for r in intel], "o-", label="CPU p50")
        axis.plot(range(len(intel)), [r["intel_cpu_p95_ms"] for r in intel], "o-", label="CPU p95")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_ylabel("Milliseconds")
        axis.set_title("Intel OpenVINO CPU equivalence diagnostic (not NPU performance)")
        axis.legend()
        _save_figure(figure, charts, "intel_cpu_diagnostic_latency")
        plt.close(figure)
    measured = [r for r in records if r["target_npu_p50_ms"] is not None]
    if not measured:
        return
    labels = [f"{r['kernel']}\n{r['target_id']}\n{r['method']}" for r in measured]
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.8), 5))
    axis.plot(range(len(measured)), [r["target_npu_p50_ms"] for r in measured], "o-", label="NPU p50")
    axis.plot(range(len(measured)), [r["target_npu_p95_ms"] for r in measured], "o-", label="NPU p95")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_ylabel("Milliseconds")
    axis.set_title("Measured target-NPU latency")
    axis.legend()
    _save_figure(figure, charts, "target_npu_latency")
    plt.close(figure)
    lookup = {(r["kernel"], r["target_id"], r["method"]): r for r in measured}
    pairs = []
    for kernel, target in sorted({(key[0], key[1]) for key in lookup}):
        baseline = lookup.get((kernel, target, "baseline"))
        agentic = lookup.get((kernel, target, "agentic"))
        if baseline and agentic and agentic["target_npu_p50_ms"]:
            pairs.append((kernel, target, baseline["target_npu_p50_ms"] / agentic["target_npu_p50_ms"]))
    if pairs:
        figure, axis = plt.subplots(figsize=(max(8, len(pairs) * 0.9), 4.5))
        axis.bar(range(len(pairs)), [item[2] for item in pairs], color="#2a9d8f")
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_xticks(range(len(pairs)), [f"{k}\n{t}" for k, t, _ in pairs], rotation=40, ha="right")
        axis.set_ylabel("Baseline p50 / agentic p50")
        axis.set_title("Agentic-over-baseline NPU speedup, paired within backend")
        _save_figure(figure, charts, "target_npu_speedup")
        plt.close(figure)


def _search_trees(plt: Any, experiment_root: Path, records: list[dict[str, Any]], charts: Path) -> None:
    connection = sqlite3.connect(experiment_root / "state.sqlite")
    connection.row_factory = sqlite3.Row
    try:
        for record in records:
            if record["method"] != "agentic" or not record["run_id"]:
                continue
            rows = connection.execute(
                """SELECT c.id, c.parent_id, c.depth, c.label, e.evaluation_json
                   FROM candidates c LEFT JOIN evaluations e ON e.candidate_id=c.id
                   WHERE c.run_id=? AND c.target_id=? ORDER BY c.depth, c.id""",
                (record["run_id"], record["target_id"]),
            ).fetchall()
            if not rows:
                figure, axis = plt.subplots(figsize=(8, 4))
                axis.text(0.5, 0.5, "No candidates were produced", ha="center", va="center")
                axis.set_axis_off()
                axis.set_title(f"Search tree: {record['kernel']} / {record['target_id']}")
                name = f"search_tree_{record['kernel']}_{record['target_id']}"
                _save_figure(figure, charts, name)
                plt.close(figure)
                continue
            by_depth: dict[int, list[sqlite3.Row]] = defaultdict(list)
            for row in rows:
                by_depth[int(row["depth"])].append(row)
            positions: dict[str, tuple[float, float]] = {}
            for depth, depth_rows in sorted(by_depth.items()):
                for index, row in enumerate(depth_rows):
                    positions[row["id"]] = (depth, index - (len(depth_rows) - 1) / 2)
            figure, axis = plt.subplots(figsize=(max(8, len(by_depth) * 2.2), max(5, len(rows) * 0.36)))
            for row in rows:
                x, y = positions[row["id"]]
                if row["parent_id"] in positions:
                    px, py = positions[row["parent_id"]]
                    axis.plot([px, x], [py, y], color="#888888", linewidth=0.8, zorder=1)
                evaluation = json.loads(row["evaluation_json"]) if row["evaluation_json"] else {}
                color = "#2a9d8f" if evaluation.get("compile_success") else "#e76f51"
                axis.scatter([x], [y], s=180, color=color, zorder=2)
                axis.text(x + 0.05, y, f"{row['label']}\n{evaluation.get('reward', 0):.2f}", fontsize=7, va="center")
            axis.set_title(f"Search tree: {record['kernel']} / {record['target_id']}")
            axis.set_xlabel("Candidate depth")
            axis.set_yticks([])
            name = f"search_tree_{record['kernel']}_{record['target_id']}"
            _save_figure(figure, charts, name)
            plt.close(figure)
    finally:
        connection.close()


def generate_charts(experiment_root: Path, records: list[dict[str, Any]]) -> list[str]:
    plt = _plot_module()
    charts = experiment_root / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    if records:
        _aggregate_chart(plt, records, charts)
        _heatmap(plt, records, charts)
        _paired_times(plt, records, charts)
        _breakdown(plt, records, charts)
        _effort(plt, records, charts)
        _artifact_reward(plt, records, charts)
        _latency_charts(plt, records, charts)
        _search_trees(plt, experiment_root, records, charts)
    return sorted(path.name for path in charts.glob("*.png"))


def write_html_report(
    experiment_root: Path, records: list[dict[str, Any]], summary: dict[str, Any], chart_names: list[str]
) -> Path:
    group_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(group[key]))}</td>"
            for key in ("method", "target_id", "successes", "total", "success_rate", "mean_end_to_end_seconds", "mean_reward")
        )
        + "</tr>"
        for group in summary["groups"]
    )
    images = "".join(
        f'<section><h2>{html.escape(Path(name).stem.replace("_", " ").title())}</h2>'
        f'<img src="charts/{html.escape(name)}" alt="{html.escape(name)}"></section>'
        for name in chart_names
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>NPU translation experiment</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1500px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.4rem}}img{{max-width:100%;height:auto}}.note{{background:#fff4d6;padding:1rem}}</style>
</head><body><h1>NPU translation experiment</h1>
<p class="note">AMD offline compile validity, Intel CPU graph equivalence, and physical NPU measurements are separate evidence. Missing device latency is not represented as zero. Backends are not ranked against each other.</p>
<p>Recorded cases: {len(records)}</p>
<table><thead><tr><th>Method</th><th>Target</th><th>Successes</th><th>Total</th><th>Rate</th><th>Mean end-to-end seconds</th><th>Mean reward</th></tr></thead><tbody>{group_rows}</tbody></table>
{images}</body></html>"""
    path = experiment_root / "report.html"
    _atomic_text(path, content)
    return path


def build_report(experiment_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = collect_metrics(experiment_root)
    summary = write_metric_files(experiment_root, records)
    write_audit_logs(experiment_root)
    chart_names = generate_charts(experiment_root, records)
    write_html_report(experiment_root, records, summary, chart_names)
    return records, summary
