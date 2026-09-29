"""Explicit parallel continuation of v4; preserves per-case prompts and checkpoints."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from npu_agent.config import Settings, TARGETS
from npu_agent.database import Database
from npu_agent.models import TranslationRequest
from npu_agent.study_v4 import StudyCompiler, StudyConfig
from npu_agent.study_v4_provider import StudyProvider
from npu_agent.validation import sha256, write_json
from research.run_budgeted_study import BudgetGuard, amount, import_spending, run_remaining


class ParallelBudgetGuard(BudgetGuard):
    def __init__(self, ledger, path, campaign, stop, acknowledge_unknown=False):
        super().__init__(ledger, path, campaign, stop.set, acknowledge_unknown)
        self.event = stop
        self.lock = RLock()
        if ledger.get("pending_call") or ledger.get("pending_calls"):
            raise ValueError("unresolved pending requests; reconcile before continuing")

    def generate(self, generate, *args, **kwargs):
        with self.lock:
            if (self.event.is_set() or (self.ledger.get("unknown_spending") and not self.acknowledge_unknown)
                    or self.spent() >= amount(self.ledger["limit_usd"])):
                self.stop()
                raise ValueError("OpenRouter budget or pause prevents request; no request sent")
            pending = uuid4().hex
            self.ledger.setdefault("pending_calls", {})[pending] = {"campaign": str(self.campaign.resolve())}
            write_json(self.path, self.ledger)
        try:
            response = generate(*args, **kwargs)
        except Exception as exc:
            with self.lock:
                metadata = getattr(exc, "metadata", {}) or {}
                if metadata.get("failure_origin") != "model_output":
                    self.stop()
                self.finish(pending, metadata)
            raise
        except BaseException:
            self.stop()
            raise
        with self.lock:
            self.finish(pending, response.metadata)
        return response

    def finish(self, pending, metadata):
        del self.ledger["pending_calls"][pending]
        if not self.ledger["pending_calls"]:
            del self.ledger["pending_calls"]
        self.record(metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compiler-jobs", type=int, default=4)
    parser.add_argument("--acknowledge-unknown-spending", action="store_true")
    parser.add_argument("--continue-past-rate-limit", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.compiler_jobs < 1 or args.compiler_jobs > args.workers:
        raise ValueError("require positive workers and compiler jobs <= workers")
    root = args.output.resolve()
    if not (root / "campaign.json").exists():
        raise ValueError("parallel continuation requires an existing frozen campaign")
    spec = importlib.util.spec_from_file_location("v4_runner", ROOT / "scripts/case_study_v4.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = StudyConfig.model_validate_json(args.config.read_text())
    with args.ledger.with_suffix(".lock").open("a") as ledger_lock, (root / ".campaign.lock").open("a") as campaign_lock:
        fcntl.flock(ledger_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected, families, corpus = runner.inputs(config)
        campaign = runner.prepare(root, config, selected, families, corpus)
        ledger = json.loads(args.ledger.read_text())
        if ledger.get("pending_call") or ledger.get("pending_calls"):
            raise ValueError("unresolved pending requests; reconcile before continuing")
        roots = {Path(call["campaign"]) for call in ledger["calls"].values()}
        import_spending(ledger, sorted(roots | {root}), args.acknowledge_unknown_spending)
        stop = Event()
        guard = ParallelBudgetGuard(ledger, args.ledger, root, stop, args.acknowledge_unknown_spending)
        if (ledger.get("unknown_spending") and not args.acknowledge_unknown_spending) or guard.spent() >= amount(ledger["limit_usd"]):
            raise ValueError("spending limit or unresolved billing prevents continuation")
        write_json(args.ledger, ledger)
        settings = Settings(database_path=root / "state.sqlite", runs_path=root / "cases", repository_path=ROOT,
            provider="openrouter", model=config.provider.model, validation_policy="offline-validated",
            experiment_id=root.name, compiler_cpus=config.compiler_cpus, compiler_memory=config.compiler_memory,
            max_compiler_jobs=args.compiler_jobs, compiler_timeout_seconds=config.compiler_timeout_seconds)
        settings.ensure_directories()
        provider, compiler = StudyProvider(config.provider, ROOT), StudyCompiler(settings)
        generate = provider.generate
        provider.generate = lambda *a, **kw: guard.generate(generate, *a, **kw)
        previous = runner.install_pause_handlers(stop)
        try:
            runner.preflight(root, campaign, config, provider, compiler, selected, stop)
            if stop.is_set():
                return 2
            pending = []
            for case in campaign["cases"]:
                path = settings.runs_path / case["id"] / "report.json"
                state = json.loads(path.read_text()) if path.exists() else None
                if state and state["status"] in ("completed", "failed"):
                    continue
                if state and state["status"] == "blocked":
                    if args.continue_past_rate_limit and run_remaining(lambda *a, **kw: state, None, settings, None, case)["status"] == "blocked_rate_limit_retained":
                        continue
                    raise ValueError("unresolved blocked case prevents continuation")
                pending.append(case)
            event = {"event": "parallel_resume", "at": datetime.now(UTC).isoformat(),
                "workers": args.workers, "compiler_jobs": args.compiler_jobs, "cases": [case["id"] for case in pending],
                "implementation_sha256": sha256(Path(__file__)),
                "budget_guard_sha256": sha256(ROOT / "research/run_budgeted_study.py"),
                "timing_note": "Mixed serial/parallel campaign. Active case time includes compiler queueing; sum is not campaign wall time."}
            archive = root / ("parallel-run-" + uuid4().hex)
            archive.mkdir()
            (archive / "runner.py").write_bytes(Path(__file__).read_bytes())
            (archive / "budget_guard.py").write_bytes((ROOT / "research/run_budgeted_study.py").read_bytes())
            write_json(archive / "provenance.json", event)
            campaign["events"].append(event)
            campaign["status"] = "running"
            write_json(root / "campaign.json", campaign)

            def execute(case):
                if stop.is_set():
                    return "paused"
                source, manifest, _ = selected[case["kernel"]]
                request = TranslationRequest(source_path=str(source), manifest_path=str(manifest),
                    targets=[TARGETS[case["target"]]], provider="openrouter", model=config.provider.model,
                    validation_policy="offline-validated")
                try:
                    state = runner.run_case(request, settings, config, case, provider, compiler, stop)
                    if state["status"] == "blocked":
                        stop.set()
                    return state["status"]
                except BaseException:
                    stop.set()
                    raise

            # Initialize WAL and schema before concurrent workers open connections.
            Database(settings.database_path).close()
            statuses = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(execute, case): case for case in pending}
                for future in as_completed(futures):
                    status = future.result()
                    statuses.append(status)
                    case = futures[future]
                    print(f"{case['id']} {case['kernel']} {case['arm']}: {status}", flush=True)
            campaign["status"] = "blocked" if "blocked" in statuses else "paused" if stop.is_set() else "completed"
        except Exception:
            campaign["status"] = "blocked"
            raise
        finally:
            campaign["events"].append({"event": campaign["status"], "at": datetime.now(UTC).isoformat()})
            write_json(root / "campaign.json", campaign)
            runner.build_report(root)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return 2 if campaign["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
