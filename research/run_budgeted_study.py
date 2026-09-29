"""Spending guard for v4; keeps the frozen runner and its prompts unchanged."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npu_agent.study_v4_provider import StudyProvider
from npu_agent.validation import write_json


def amount(value) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("missing or invalid billed cost; do not launch another paid request") from None
    if not result.is_finite() or result < 0:
        raise ValueError("billed cost must be finite and nonnegative")
    return result


def import_spending(ledger: dict, roots: list[Path], acknowledge_unknown: bool = False) -> None:
    for root in roots:
        preflight = root / "preflight.json"
        calls = list(json.loads(preflight.read_text()).get("calls", {}).values()) if preflight.exists() else []
        for path in (root / "cases").glob("*/report.json"):
            for cycle in json.loads(path.read_text())["cycles"]:
                calls.extend(cycle["calls"].values())
        for call in calls:
            if call.get("status") == "in_flight":
                raise ValueError("campaign still has an in-flight call; wait for its response")
            metadata = call.get("metadata", {})
            generation = metadata.get("generation_id")
            try:
                cost = amount(metadata.get("usage", {}).get("cost"))
                if not generation:
                    raise ValueError("spending record has no provider generation ID")
            except ValueError:
                if not acknowledge_unknown:
                    raise
                identity = hashlib.sha256(json.dumps(call, sort_keys=True).encode()).hexdigest()
                ledger.setdefault("acknowledged_unknown_calls", {})[identity] = {
                    "campaign": str(root.resolve()), "call": call}
                ledger["unknown_spending"] = True
                continue
            record = {"cost_usd": str(cost), "campaign": str(root.resolve())}
            if generation in ledger["calls"] and ledger["calls"][generation] != record:
                raise ValueError("conflicting spending records for the same provider generation")
            ledger["calls"][generation] = record


class BudgetGuard:
    def __init__(self, ledger: dict, path: Path, campaign: Path, stop, acknowledge_unknown: bool = False):
        self.ledger, self.path, self.campaign, self.stop = ledger, path, campaign, stop
        self.acknowledge_unknown = acknowledge_unknown

    def spent(self) -> Decimal:
        return sum((amount(call["cost_usd"]) for call in self.ledger["calls"].values()), Decimal(0))

    def generate(self, generate, *args, **kwargs):
        if (self.ledger.get("unknown_spending") and not self.acknowledge_unknown) or (self.ledger.get("pending_call") or self.ledger.get("pending_calls")) or self.spent() >= amount(self.ledger["limit_usd"]):
            self.stop()
            raise ValueError("OpenRouter spending limit reached or spending is unknown; no request sent")
        self.ledger["pending_call"] = {"campaign": str(self.campaign.resolve())}
        write_json(self.path, self.ledger)
        try:
            response = generate(*args, **kwargs)
        except Exception as exc:
            self.record(getattr(exc, "metadata", {}) or {})
            raise
        self.record(response.metadata)
        return response

    def record(self, metadata):
        try:
            generation = metadata.get("generation_id")
            cost = amount(metadata.get("usage", {}).get("cost"))
            if not generation or generation in self.ledger["calls"]:
                raise ValueError("missing or duplicate provider generation ID")
            self.ledger["calls"][generation] = {"cost_usd": str(cost), "campaign": str(self.campaign.resolve())}
        except ValueError:
            self.ledger["unknown_spending"] = True
            self.acknowledge_unknown = False
        self.ledger.pop("pending_call", None)
        self.ledger["known_spent_usd"] = str(self.spent())
        write_json(self.path, self.ledger)
        if (self.ledger.get("unknown_spending") and not self.acknowledge_unknown) or self.spent() >= amount(self.ledger["limit_usd"]):
            # The runner's handler drains/checkpoints this response before pausing.
            self.stop()


def run_remaining(run_case, request, settings, config, case, *a, **kw):
    path = settings.runs_path / case["id"] / "report.json"
    if path.exists():
        state = json.loads(path.read_text())
        calls = list(state["cycles"][-1]["calls"].values()) if state["cycles"] else []
        if state["status"] == "blocked" and calls and calls[-1].get("metadata", {}).get("http_status") == 429:
            # Only the loop-control view changes. Preserve the actual blocked record.
            return {**state, "status": "blocked_rate_limit_retained"}
    return run_case(request, settings, config, case, *a, **kw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--limit-usd", default="30")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/case-study-v4.json")
    parser.add_argument("--command", choices=("preflight", "run"), default="run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-campaign", type=Path, action="append", default=[])
    parser.add_argument("--acknowledge-unknown-spending", action="store_true",
                        help="explicit authorization to continue with historical billing gaps; new gaps still pause")
    parser.add_argument("--continue-past-rate-limit", action="store_true",
                        help="retain previously blocked HTTP 429 cases and run remaining scheduled cases")
    args = parser.parse_args()
    limit = amount(args.limit_usd)
    if limit <= 0:
        raise ValueError("limit must be positive")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else {
            "protocol": "openrouter-budget-v1", "limit_usd": str(limit), "calls": {}, "unknown_spending": False}
        if amount(ledger["limit_usd"]) != limit:
            raise ValueError("existing budget limit differs; do not reset spending")
        roots = {Path(call["campaign"]) for call in ledger["calls"].values()}
        import_spending(ledger, sorted(roots | set(args.include_campaign) | {args.output}), args.acknowledge_unknown_spending)
        guard = BudgetGuard(ledger, args.ledger, args.output, lambda: os.kill(os.getpid(), signal.SIGTERM), args.acknowledge_unknown_spending)
        ledger["known_spent_usd"] = str(guard.spent())
        write_json(args.ledger, ledger)
        if (ledger.get("unknown_spending") and not args.acknowledge_unknown_spending) or (ledger.get("pending_call") or ledger.get("pending_calls")) or guard.spent() >= limit:
            print("Budget exhausted or spending unknown; no paid requests launched.", flush=True)
            return 2
        original = StudyProvider.generate
        StudyProvider.generate = lambda provider, *a, **kw: guard.generate(original, provider, *a, **kw)
        previous_argv = sys.argv
        try:
            spec = importlib.util.spec_from_file_location("frozen_v4_runner", ROOT / "scripts/case_study_v4.py")
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            if args.continue_past_rate_limit:
                original_run_case = runner.run_case


                runner.run_case = lambda *a, **kw: run_remaining(original_run_case, *a, **kw)
            sys.argv = [str(ROOT / "scripts/case_study_v4.py"), args.command, "--config", str(args.config), "--output", str(args.output)]
            print(f"OpenRouter budget: ${guard.spent()} spent / ${limit}; pauses at a request boundary.", flush=True)
            return runner.main()
        finally:
            StudyProvider.generate = original
            sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
