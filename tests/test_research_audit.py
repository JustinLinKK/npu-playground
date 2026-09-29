import importlib.util
import json
from pathlib import Path

import pytest

from npu_agent.study_v4 import StudyConfig


spec = importlib.util.spec_from_file_location("research_audit", Path(__file__).resolve().parents[1] / "research/audit_v1.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
build_audit, failure_category, write_audit = audit_module.build_audit, audit_module.failure_category, audit_module.write_audit


def campaign(tmp_path, status="completed"):
    root = tmp_path / "campaign"
    root.mkdir()
    case = {"id": "one", "kernel": "add", "target": "intel_npu_4000", "arm": "structured_ir",
            "family": "add", "repetition": 1}
    (root / "campaign.json").write_text(json.dumps({"config": StudyConfig().model_dump(), "cases": [case], "status": status}))
    call = {"status": "completed", "metadata": {"usage": {"prompt_tokens": 10, "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 5}}, "duration_seconds": 2, "generation_id": "prep"}}
    (root / "preflight.json").write_text(json.dumps({"status": "completed", "calls": {
        name: dict(call, metadata=dict(call["metadata"], generation_id=name))
        for name in ("CodeBundle", "Hint", "ExecutableKernelIR")}}))
    state = {"status": "completed", "phase": None, "active_seconds": 5, "cycles": [
        {"number": 1, "status": "passed", "calls": {"intermediate": dict(call, metadata=dict(call["metadata"], generation_id="ir"))},
         "semantic_validation": {"status": "passed", "duration_seconds": .5},
         "compile_result": {"validation": {"stages": {"semantic_validation": {"status": "passed", "duration_seconds": .5},
             "host_execution": {"status": "passed", "duration_seconds": 1},
             "target_execution": {"status": "blocked", "reason_code": "NO_VALIDATED_TARGET_EXECUTOR"}}}}}]}
    path = root / "cases" / "one" / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state))
    return root, path, state


def test_usage_stage_deduplication_and_nonmutation(tmp_path):
    root, _, _ = campaign(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    audit = write_audit(root, tmp_path / "audit")
    assert audit["known_tokens"] == 120
    assert audit["preflight_known_tokens"] == 90 and audit["translation_known_tokens"] == 30
    assert audit["accounting_complete"]
    assert sum(s["stage"] == "semantic_validation" for s in audit["stages"]) == 1
    assert audit["failure_counts"] == {}
    assert all(p.read_bytes() == raw for p, raw in before.items())
    with pytest.raises(ValueError, match="outside"):
        write_audit(root, root / "audit")


def test_unknown_preflight_usage_blocks_accounting(tmp_path):
    root, _, _ = campaign(tmp_path)
    (root / "preflight.json").write_text(json.dumps({"status": "blocked", "calls": {
        "CodeBundle": {"status": "blocked", "metadata": {}, "error": "transport failed"}}}))
    audit = build_audit(root)
    assert not audit["accounting_complete"]
    assert audit["unknown_usage_calls"] == 1
    assert audit["known_tokens"] == 30
    assert audit["failure_counts"] == {"infrastructure": 1}


def test_live_call_is_not_failed_and_pending_case_is_not_zero_cost(tmp_path):
    root, path, state = campaign(tmp_path, "running")
    state["status"] = "running"
    state["phase"] = "code"
    state["cycles"][0]["calls"]["code"] = {"status": "in_flight"}
    path.write_text(json.dumps(state))
    audit = build_audit(root)
    assert audit["in_flight_calls"] == 1 and not audit["accounting_complete"]
    assert audit["failure_counts"] == {}
    path.unlink()
    audit = build_audit(root)
    group = next(g for g in audit["distributions"] if g["target"] == "intel_npu_4000" and g["arm"] == "structured_ir")
    assert group["planned"] == 1 and group["started"] == 0
    assert group["known_tokens"]["quantiles"] is None


def test_duplicate_preflight_and_case_generation_ids(tmp_path):
    root, path, state = campaign(tmp_path)
    state["cycles"][0]["calls"]["intermediate"]["metadata"]["generation_id"] = "CodeBundle"
    path.write_text(json.dumps(state))
    audit = build_audit(root)
    assert audit["duplicate_generation_ids"] == ["CodeBundle"]
    assert not audit["accounting_complete"]


@pytest.mark.parametrize("stage,result,category", [
    ("semantic_validation", {"status": "failed", "message": "unsupported operation"}, "unsupported_ir"),
    ("semantic_validation", {"status": "failed", "message": "wrong values"}, "semantic"),
    ("host_execution", {"status": "failed", "correct": False}, "numerical"),
    ("host_execution", {"status": "blocked"}, "infrastructure"),
    ("dataflow_simulation", {"status": "failed"}, "buffer_dataflow"),
    ("model_code", {"status": "rejected"}, "model_output"),
    ("graph_construction", {"status": "failed", "message": "AttributeError"}, "backend_api"),
    ("target_compile", {"status": "failed"}, "needs_review"),
    ("host_execution", {"status": "blocked", "reason_code": "PREREQUISITE_UNAVAILABLE"}, None),
])
def test_failure_taxonomy(stage, result, category):
    assert failure_category(stage, result) == category


def test_bound_manual_review_labels_preserve_accounting_and_reject_stale_records(tmp_path):
    import hashlib

    root, path, state = campaign(tmp_path)
    state['status'] = 'failed'
    state['cycles'][0]['compile_result']['validation']['stages']['target_compile'] = {
        'status': 'failed', 'message': 'MLIR expected attribute value', 'duration_seconds': 1}
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    reviews = tmp_path / 'reviews'
    reviews.mkdir()
    review = {'status': 'terminal_review', 'source_report': str(path),
              'source_report_sha256': hashlib.sha256(before).hexdigest(), 'reviewer': 'test reviewer',
              'cycles': [{'cycle': 1, 'failure_category': 'backend_api'}]}
    review_path = reviews / 'one.json'
    review_path.write_text(json.dumps(review))
    audit = build_audit(root)
    complete = audit['accounting_complete']
    audit_module.apply_reviews(audit, root, reviews)
    assert audit['automatic_failure_counts'] == {'needs_review': 1}
    assert audit['failure_counts'] == {'backend_api': 1}
    assert audit['accounting_complete'] == complete
    assert audit['manual_reviews'][str(review_path.resolve())] == hashlib.sha256(review_path.read_bytes()).hexdigest()
    assert path.read_bytes() == before
    state['active_seconds'] = 6
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='stale'):
        audit_module.apply_reviews(build_audit(root), root, reviews)
