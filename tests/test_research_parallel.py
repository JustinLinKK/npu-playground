import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("parallel_study", Path(__file__).resolve().parents[1] / "research/run_parallel_study.py")
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


def test_parallel_requests_overlap_and_reconcile_exactly(tmp_path):
    stop = Event()
    ledger = {"limit_usd": "30", "calls": {}, "unknown_spending": False}
    guard = parallel.ParallelBudgetGuard(ledger, tmp_path / "ledger.json", tmp_path, stop)
    barrier = Barrier(4)

    def call(index):
        def response():
            barrier.wait(timeout=5)
            with guard.lock:
                assert len(json.loads(guard.path.read_text())["pending_calls"]) == 4
            barrier.wait(timeout=5)
            return SimpleNamespace(metadata={"generation_id": str(index), "usage": {"cost": ".1"}})
        return guard.generate(response)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len(list(pool.map(call, range(4)))) == 4
    saved = json.loads(guard.path.read_text())
    assert saved["known_spent_usd"] == "0.4" and len(saved["calls"]) == 4
    assert "pending_calls" not in saved and not stop.is_set()


def test_billing_failure_drains_inflight_and_blocks_new_requests(tmp_path):
    stop = Event()
    ledger = {"limit_usd": "30", "calls": {}, "unknown_spending": False}
    guard = parallel.ParallelBudgetGuard(ledger, tmp_path / "ledger.json", tmp_path, stop)
    started = Event()

    class BillingError(Exception):
        metadata = {"http_status": 402}

    def billed():
        started.set()
        assert stop.wait(5)
        return SimpleNamespace(metadata={"generation_id": "paid", "usage": {"cost": ".1"}})

    def fail():
        raise BillingError()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(guard.generate, billed)
        assert started.wait(5)
        with pytest.raises(BillingError):
            guard.generate(fail)
        assert future.result().metadata["generation_id"] == "paid"
    with pytest.raises(ValueError, match="no request sent"):
        guard.generate(lambda: pytest.fail("must not send"))
    assert ledger["unknown_spending"] and ledger["known_spent_usd"] == "0.1"
    assert "pending_calls" not in ledger


def test_crashed_parallel_request_blocks_serial_and_parallel_resume(tmp_path):
    stop = Event()
    ledger = {"limit_usd": "30", "calls": {}, "unknown_spending": False}
    path = tmp_path / "ledger.json"
    guard = parallel.ParallelBudgetGuard(ledger, path, tmp_path, stop)

    def crash():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        guard.generate(crash)
    saved = json.loads(path.read_text())
    assert saved["pending_calls"]
    with pytest.raises(ValueError, match="pending requests"):
        parallel.ParallelBudgetGuard(saved, path, tmp_path, Event(), True)
    serial = parallel.BudgetGuard(saved, path, tmp_path, lambda: None, True)
    with pytest.raises(ValueError, match="no request sent"):
        serial.generate(lambda: pytest.fail("must not send"))


def test_fresh_database_supports_parallel_case_checkpoints(tmp_path):
    from npu_agent.study_v4 import run_case
    from test_study_v4 import Provider, RepairCompiler, setup_case

    request, settings, config, direct = setup_case(tmp_path)
    hinted = dict(direct, id="hinted_ir", arm="hinted_ir")
    parallel.Database(settings.database_path).close()
    ready = Barrier(2)

    def execute(case):
        ready.wait(timeout=5)
        return run_case(request, settings, config, case, Provider(), RepairCompiler(), Event())

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, [direct, hinted]))
    assert [result["status"] for result in results] == ["completed", "completed"]
    database = parallel.Database(settings.database_path)
    try:
        assert database.get_run("direct") is not None
        assert database.get_run("hinted_ir") is not None
    finally:
        database.close()


@pytest.mark.parametrize("billing_error", [False, True])
@pytest.mark.parametrize("provider_name", ["openrouter", "openai"])
def test_provider_budget_runs_without_ledger_and_stops_on_provider_error(tmp_path, monkeypatch, billing_error, provider_name):
    from npu_agent.providers import ProviderError
    from test_study_v4 import Provider, RepairCompiler, runner_module
    from test_study_v4_provider import openai_config

    runner = runner_module()
    preflights = []
    monkeypatch.setattr(runner, "preflight", lambda *a: preflights.append(True))
    monkeypatch.setattr(parallel, "importlib", SimpleNamespace(util=SimpleNamespace(
        spec_from_file_location=lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)),
        module_from_spec=lambda spec: runner)))
    config = parallel.StudyConfig(protocol="xdna2-baseline-v1", arms=["direct"],
                                  targets=["amd_xdna2_npu2"], repetitions=1, kernels=["cuda_vector_add"],
                                  **({"provider": openai_config()} if provider_name == "openai" else {}))
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    ledger = tmp_path / "budget.json"
    original = '{"limit_usd":"50","known_spent_usd":"100","pending_calls":{"lost":{}},"unknown_spending":true}'
    ledger.write_text(original)

    class ExternalProvider(Provider):
        def generate(self, *args):
            if billing_error:
                self.calls.append("billing_error")
                raise ProviderError("provider HTTP error", {"http_status": 429 if provider_name == "openai" else 402, "usage_incomplete": True})
            response = super().generate(*args)
            if len(self.calls) == 1:
                response.metadata["usage"]["cost"] = 100
            else:
                response.metadata["usage"].pop("cost", None)
            return response

    provider = ExternalProvider()
    provider.model = config.provider.model
    compiler = RepairCompiler()
    monkeypatch.setattr(parallel, "StudyOpenAIProvider" if provider_name == "openai" else "StudyProvider", lambda *a: provider)
    monkeypatch.setattr(parallel, "StudyCompiler", lambda settings: compiler)
    run_case = runner.run_case

    def execute(request, settings, *args):
        assert request.provider == settings.provider == provider_name
        return run_case(request, settings, *args)

    monkeypatch.setattr(runner, "run_case", execute)
    root = tmp_path / "campaign"
    command = ["parallel", "--config", str(path), "--output", str(root),
               "--workers", "1", "--compiler-jobs", "1"]
    monkeypatch.setattr(parallel.sys, "argv", command + ["--budget-on-" + provider_name])
    assert parallel.main() == (2 if billing_error else 0)
    campaign = json.loads((root / "campaign.json").read_text())
    assert campaign["budget_enforcement"] == provider_name and preflights == [True]
    assert len(provider.calls) == (1 if billing_error else 2)
    assert ledger.read_text() == original and not ledger.with_suffix(".lock").exists()
    assert not (root / "budget.json").exists()
    report = (root / "results/REPORT.md").read_text()
    assert ("Direct OpenAI has no local dollar cap" if provider_name == "openai" else "delegated to OpenRouter") in report
    assert "$50 threshold" not in report
    if provider_name == "openai":
        assert "gpt-5.6-terra" in report and "Qwen" not in report and "billed dollars unavailable" in report
    summary = json.loads((root / "summary.json").read_text())
    assert summary["known_total_spend_usd"] is None and summary["billing_complete"] is None
    if provider_name == "openai":
        assert summary["known_evaluation_cost_usd"] is None
    if billing_error:
        with pytest.raises(ValueError, match="blocked case"):
            parallel.main()
    else:
        assert parallel.main() == 0
    assert len(provider.calls) == (1 if billing_error else 2)
    monkeypatch.setattr(parallel.sys, "argv", command + ["--ledger", str(ledger)])
    with pytest.raises(ValueError, match="requires --budget-on-openai" if provider_name == "openai" else "budget enforcement changed"):
        parallel.main()
