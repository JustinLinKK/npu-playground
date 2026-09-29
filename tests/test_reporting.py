import asyncio
import json
from pathlib import Path

from npu_agent.baseline import translate_one_shot
from npu_agent.config import INTEL_NPU_4000, Settings
from npu_agent.experiment import run_experiment
from npu_agent.models import TranslationRequest
from npu_agent.reporting import _token_totals, build_report
from test_experiment import CorpusCompiler, CorpusProvider


def test_token_totals_normalize_each_call_without_double_counting():
    totals = _token_totals(json.dumps(usage) for usage in [
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cached_input_tokens": 7,
         "output_tokens_details": {"reasoning_tokens": 3}},
        {"prompt_tokens": 12, "completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": 4},
         "completion_tokens_details": {"reasoning_tokens": 6}},
        {"input_tokens": 4, "output_tokens": 3, "input_token_details": {"cache_read": 2},
         "output_token_details": {"reasoning": 1}},
    ])
    assert totals == {"input_tokens": 26, "output_tokens": 16, "total_tokens": 42,
                      "known_total_tokens": 42, "cached_input_tokens": 13,
                      "reasoning_output_tokens": 10, "usage_complete": True}


def test_missing_usage_preserves_known_subtotal_without_claiming_zero():
    totals = _token_totals([json.dumps({"input_tokens": 10, "output_tokens": 5}), "{}"])
    assert totals["total_tokens"] is None
    assert totals["input_tokens"] is None
    assert totals["known_total_tokens"] == 15
    assert totals["usage_complete"] is False


def test_later_turn_without_usage_keeps_known_tokens_and_marks_total_unknown():
    from npu_agent.providers import telemetry_usage

    usage = telemetry_usage([{"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
                             {"type": "turn.completed"}])
    totals = _token_totals([json.dumps(usage), json.dumps({"total_tokens": 7})])
    assert totals["known_total_tokens"] == 22
    assert totals["total_tokens"] is None
    assert totals["usage_complete"] is False
    assert _token_totals(["null"])["total_tokens"] is None


def test_reports_include_failed_cases_and_pair_only_solved_complete_usage(tmp_path):
    class MissingUsageProvider(CorpusProvider):
        def generate(self, *args):
            response = super().generate(*args)
            if len(self.calls) == 2:
                response.metadata.pop("usage")
            return response

    root = run_experiment(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path,
                          kernel_names=["cuda_vector_add"], target_ids=["intel_npu_4000"],
                          provider=MissingUsageProvider(), compiler=CorpusCompiler())
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    structured = next(r for r in records if r["method"] == "structured_ir")
    assert structured["total_tokens"] is None
    assert structured["known_total_tokens"] == 15
    assert not structured["usage_complete"]
    summary = json.loads((root / "summary.json").read_text())
    pair = next(p for p in summary["paired_efficiency"] if p["method"] == "structured_ir")
    assert pair["jointly_solved"] == 1
    assert pair["token_complete_pairs"] == 0
    assert pair["baseline_over_method_tokens"] is None
    assert "full pipeline" in (root / "report.html").read_text()
    # A recorded unsolved case must also be removed from cycle comparisons.
    path = next(root.glob("structured_ir/*/report.json"))
    case = json.loads(path.read_text())
    case.update(status="failed", cycles_to_success=None, terminal_reason="cycle_budget_exhausted")
    case["targets"][0]["status"] = "failed"
    path.write_text(json.dumps(case))
    experiment_path = root / "experiment.json"
    experiment = json.loads(experiment_path.read_text())
    for value in experiment["cases"].values():
        if value["method"] == "structured_ir":
            value["result_status"] = "failed"
    experiment_path.write_text(json.dumps(experiment))
    records, summary = build_report(root)
    pair = next(p for p in summary["paired_efficiency"] if p["method"] == "structured_ir")
    assert pair["jointly_solved"] == pair["token_complete_pairs"] == 0
    assert pair["baseline_over_method_cycles"] is None
    assert len(records) == 3
    assert "cycle_budget_exhausted" in (root / "report.html").read_text()


def test_historical_one_shot_report_remains_readable(tmp_path):
    manifest = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()
    request = TranslationRequest(source_path=str(manifest.parent / "kernel.cu"), manifest_path=str(manifest),
                                 targets=[INTEL_NPU_4000], provider="codex-cli", model="gpt-5.6-terra")
    settings = Settings(database_path=tmp_path / "state.sqlite", runs_path=tmp_path / "baseline", experiment_id="old")
    result = asyncio.run(translate_one_shot(request, settings, provider=CorpusProvider(), compiler=CorpusCompiler()))
    (tmp_path / "experiment.json").write_text(json.dumps({"schema_version": "1.0", "experiment_id": "old", "cases": {
        "baseline:cuda_vector_add:intel_npu_4000": {"method": "baseline", "kernel": "cuda_vector_add", "target_id": "intel_npu_4000",
            "backend": "intel_openvino", "run_id": result.run_id, "result_status": result.status,
            "result_path": str(Path(result.report_path).relative_to(tmp_path))}}}))
    records, summary = build_report(tmp_path)
    assert records[0]["success"] is True
    assert records[0]["cycles_to_success"] is None
    assert records[0]["total_tokens"] == 15
    assert summary["paired_efficiency"] == []
