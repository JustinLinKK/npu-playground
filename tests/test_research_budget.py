import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from threading import Event

import pytest


spec = importlib.util.spec_from_file_location("research_budget", Path(__file__).resolve().parents[1] / "research/run_budgeted_study.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def setup_guard(tmp_path):
    stops = []
    ledger = {"limit_usd": "30", "calls": {"previous": {"cost_usd": "29.9", "campaign": "previous"}},
              "unknown_spending": False}
    guard = budget.BudgetGuard(ledger, tmp_path / "budget.json", tmp_path / "campaign", lambda: stops.append(True))
    return guard, stops


def test_crossing_limit_preserves_response_and_stops_next_request(tmp_path):
    guard, stops = setup_guard(tmp_path)
    response = SimpleNamespace(metadata={"generation_id": "last", "usage": {"cost": .11}})
    assert guard.generate(lambda: response) is response
    assert guard.spent() == Decimal("30.01") and stops
    assert json.loads(guard.path.read_text())["known_spent_usd"] == "30.01"
    called = []
    with pytest.raises(ValueError, match="no request sent"):
        guard.generate(lambda: called.append(True))
    assert called == []


def test_missing_cost_preserves_response_but_blocks_spending(tmp_path):
    guard, stops = setup_guard(tmp_path)
    response = SimpleNamespace(metadata={"generation_id": "unknown", "usage": {}})
    assert guard.generate(lambda: response) is response
    assert guard.ledger["unknown_spending"] and stops


def test_rejected_paid_response_is_charged(tmp_path):
    guard, stops = setup_guard(tmp_path)

    class Rejected(Exception):
        metadata = {"generation_id": "rejected", "usage": {"cost": .2}}

    def generate():
        raise Rejected()

    with pytest.raises(Rejected):
        guard.generate(generate)
    assert guard.spent() == Decimal("30.1") and stops


def test_hard_interruption_leaves_persistent_pending_spending(tmp_path):
    guard, _ = setup_guard(tmp_path)

    def interrupted():
        assert json.loads(guard.path.read_text())["pending_call"]
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        guard.generate(interrupted)
    assert json.loads(guard.path.read_text())["pending_call"]
    with pytest.raises(ValueError, match="no request sent"):
        guard.generate(lambda: pytest.fail("must not spend after a lost response"))


def test_success_clears_pending_spending(tmp_path):
    guard, _ = setup_guard(tmp_path)
    guard.generate(lambda: SimpleNamespace(metadata={"generation_id": "ok", "usage": {"cost": .01}}))
    assert "pending_call" not in json.loads(guard.path.read_text())


def test_limit_pause_checkpoints_paid_response_before_backend(tmp_path):
    from npu_agent.study_v4 import run_case
    from test_study_v4 import Provider, setup_case
    from test_experiment import CorpusCompiler

    args = setup_case(tmp_path)
    stop = Event()
    guard, _ = setup_guard(tmp_path)
    guard.stop = stop.set

    class GuardedProvider(Provider):
        def generate(self, *args, **kwargs):
            def paid():
                response = super(GuardedProvider, self).generate(*args, **kwargs)
                response.metadata["usage"]["cost"] = .2
                response.metadata["generation_id"] = "final_paid_response"
                return response
            return guard.generate(paid)

    compiler = CorpusCompiler()
    state = run_case(*args, GuardedProvider(), compiler, stop)
    assert state["status"] == "paused" and state["phase"] is None
    assert state["cycles"][0]["calls"]["code"]["status"] == "completed"
    assert state["cycles"][0]["bundle"]
    assert not state["usage_incomplete"] and compiler.calls == 0
    assert guard.spent() == Decimal("30.1")


@pytest.mark.parametrize("value", [None, "nan", "inf", -1, True, "invalid"])
def test_invalid_amount_fails_closed(value):
    with pytest.raises(ValueError):
        budget.amount(value)


def test_import_is_idempotent_and_refuses_in_flight(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    path = root / "preflight.json"
    call = {"status": "completed", "metadata": {"generation_id": "preflight", "usage": {"cost": .01}}}
    path.write_text(json.dumps({"calls": {"CodeBundle": call}}))
    ledger = {"calls": {}}
    budget.import_spending(ledger, [root, root])
    assert len(ledger["calls"]) == 1
    call["status"] = "in_flight"
    path.write_text(json.dumps({"calls": {"CodeBundle": call}}))
    with pytest.raises(ValueError, match="in-flight"):
        budget.import_spending(ledger, [root])


def test_acknowledged_history_preserves_gap_and_stops_on_new_gap(tmp_path):
    guard, stops = setup_guard(tmp_path)
    guard.ledger['unknown_spending'] = True
    guard.acknowledge_unknown = True
    guard.generate(lambda: SimpleNamespace(metadata={'generation_id': 'ok', 'usage': {'cost': .01}}))
    assert guard.ledger['unknown_spending'] and not stops
    guard.generate(lambda: SimpleNamespace(metadata={}))
    assert stops and not guard.acknowledge_unknown
    with pytest.raises(ValueError, match='no request sent'):
        guard.generate(lambda: pytest.fail('new gap must stop spending'))


def test_acknowledged_import_retains_unknown_call_and_rejects_pending(tmp_path):
    root = tmp_path / 'campaign'
    root.mkdir()
    path = root / 'preflight.json'
    call = {'status': 'failed', 'metadata': {}}
    path.write_text(json.dumps({'calls': {'CodeBundle': call}}))
    ledger = {'calls': {}}
    budget.import_spending(ledger, [root, root], acknowledge_unknown=True)
    assert ledger['unknown_spending'] and ledger['calls'] == {}
    assert len(ledger['acknowledged_unknown_calls']) == 1
    assert next(iter(ledger['acknowledged_unknown_calls'].values()))['call'] == call
    call['status'] = 'in_flight'
    path.write_text(json.dumps({'calls': {'CodeBundle': call}}))
    with pytest.raises(ValueError, match='in-flight'):
        budget.import_spending(ledger, [root], acknowledge_unknown=True)


@pytest.mark.parametrize('http_status,skipped', [(429, True), (402, False), (503, False)])
def test_continue_retains_only_recorded_rate_limit_case(tmp_path, http_status, skipped):
    path = tmp_path / 'case' / 'report.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'status': 'blocked', 'cycles': [{'calls': {'code': {
        'metadata': {'http_status': http_status}}}}]}))
    original = path.read_bytes()
    calls = []
    def run_case(*args, **kwargs):
        calls.append(True)
        return {'status': 'blocked'}
    result = budget.run_remaining(run_case, None, SimpleNamespace(runs_path=tmp_path), None, {'id': 'case'})
    assert (result['status'] == 'blocked_rate_limit_retained') == skipped
    assert bool(calls) != skipped
    assert path.read_bytes() == original
