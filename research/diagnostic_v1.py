"""Audited-reference diagnostic using the unchanged v4 backend prompt and repair loop."""
from __future__ import annotations

import argparse
import fcntl
import json
import signal
import sys
import time
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from npu_agent.config import Settings, TARGETS
from npu_agent.experiment import discover_kernels
from npu_agent.ir import validate_ir
from npu_agent.models import StageResult, TranslationRequest
from npu_agent.scenarios import ExecutableKernelIR
from npu_agent.study_v4 import StudyCompiler, StudyConfig, run_case
from npu_agent.study_v4_provider import StudyProvider
from npu_agent.study_v4_reporting import case_metrics
from npu_agent.validation import json_digest, sha256, write_json
from research.audit_v1 import build_audit
from research.run_budgeted_study import BudgetGuard, amount, import_spending


def seed_reference(request, settings, config, case, reference, ir_path):
    """Seed the documented v4 checkpoint fields; no synthetic model call or free generation."""
    started = time.perf_counter()
    source = Path(request.source_path)
    manifest_path = Path(request.manifest_path)
    manifest = discover_kernels(manifest_path.parent)[case["kernel"]][2]
    if (sha256(source) != reference["source_sha256"] or sha256(manifest_path) != reference["manifest_sha256"] or
            sha256(ir_path) != reference["ir_sha256"]):
        raise ValueError("reference/source/manifest binding changed")
    ir = ExecutableKernelIR.model_validate_json(ir_path.read_text())
    identity = json_digest({"config": config.model_dump(), "case": case, "source": source.read_text(),
                            "manifest": manifest.model_dump(mode="json"), "target": request.targets[0].model_dump(mode="json")})
    path = settings.runs_path / case["id"] / "report.json"
    if path.exists():
        if json.loads(path.read_text())["identity"] != identity:
            raise ValueError("diagnostic checkpoint identity changed")
        return
    outcome = validate_ir(ir, manifest)
    duration = time.perf_counter() - started
    if not outcome.passed:
        raise ValueError("audited reference no longer passes semantic validation: " + outcome.message)
    state = {**case, "run_id": case["id"], "scenario": case["arm"], "target_id": request.targets[0].id,
             "identity": identity, "status": "running", "active_seconds": duration, "phase": None,
             "usage_incomplete": False, "timing_incomplete": False, "model": config.provider.model,
             "terminal_reason": None, "reference_preparation": reference,
             "cycles": [{"number": 1, "status": "running", "calls": {}, "intermediate": ir.model_dump(mode="json"),
                 "semantic_validation": StageResult(status="passed", correct=True, engine="kernel-ir-numpy",
                     representation="semantic_ir", message=outcome.message, duration_seconds=duration).model_dump(mode="json")} ]}
    write_json(path, state)


def run(selection_path: Path, output: Path, ledger_path: Path, acknowledge_unknown: bool = False) -> dict:
    selection = json.loads(selection_path.read_text())
    pilot = ROOT / selection["pilot"]
    if output.resolve().is_relative_to(pilot.resolve()):
        raise ValueError("diagnostic output must be outside the frozen pilot")
    audit = build_audit(pilot)
    if not audit["accounting_complete"]:
        raise ValueError("complete and reconcile the unchanged pilot before diagnostic execution")
    review_path = pilot.parent / (pilot.name + "-review.json")
    review = json.loads(review_path.read_text())
    if (review.get("decision") != "proceed_to_diagnostic" or review.get("pilot_hashes") != audit["input_sha256"] or
            not review.get("reviewer") or not review.get("review_notes")):
        raise ValueError("diagnostic needs a pilot review bound to the current evidence")
    campaign = json.loads((pilot / "campaign.json").read_text())
    config = StudyConfig.model_validate(campaign["config"])
    corpus = Path(campaign["identity"]["corpus"]["corpus"])
    selected = discover_kernels(corpus)
    references = {reference["kernel"]: reference for reference in selection["references"]}
    if selection["kernels"] != config.kernels or selection["targets"] != config.targets:
        raise ValueError("diagnostic selection must retain the preregistered pilot kernels and targets")
    provider = StudyProvider(config.provider, ROOT)
    settings = Settings(repository_path=ROOT, database_path=output / "state.sqlite", runs_path=output / "cases",
                        experiment_id=output.name, provider="openrouter", model=config.provider.model,
                        validation_policy="offline-validated", compiler_cpus=config.compiler_cpus,
                        compiler_memory=config.compiler_memory, max_compiler_jobs=1,
                        compiler_timeout_seconds=config.compiler_timeout_seconds)
    compiler = StudyCompiler(settings)
    preflight = json.loads((pilot / "preflight.json").read_text())
    live = {"provider": provider.endpoint_identity(),
            "compilers": {name: compiler.identity(TARGETS[name]) for name in config.targets}}
    if live != preflight["identity"]:
        raise ValueError("paired diagnostic provider/compiler identity differs from pilot")
    for relative, digest in campaign["identity"]["source_hashes"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError("paired diagnostic implementation differs from pilot")
    identity = {"protocol": "audited-ir-diagnostic-v1", "selection_sha256": sha256(selection_path),
                "implementation_sha256": sha256(Path(__file__)), "budget_guard_sha256": sha256(ROOT / "research/run_budgeted_study.py"),
                "pilot_identity": json_digest(campaign["identity"]), "review_sha256": sha256(review_path), "services": live}
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "diagnostic.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("diagnostic identity changed; preserve previous output")
    write_json(identity_path, identity)
    settings.ensure_directories()
    with ledger_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text())
        if amount(ledger["limit_usd"]) != amount("30"):
            raise ValueError("use the existing user-authorized $30 ledger")
        roots = {Path(call["campaign"]) for call in ledger["calls"].values()}
        import_spending(ledger, sorted(roots | {pilot, output}), acknowledge_unknown)
        if ledger.get("pending_call") or ledger.get("pending_calls") or (ledger.get("unknown_spending") and not acknowledge_unknown):
            raise ValueError("reconcile unresolved spending before a new diagnostic")
        stop = Event()
        guard = BudgetGuard(ledger, ledger_path, output, stop.set, acknowledge_unknown)
        generate = provider.generate
        provider.generate = lambda *a, **kw: guard.generate(generate, *a, **kw)
        previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        rows = []
        try:
            for kernel in selection["kernels"]:
                for target in selection["targets"]:
                    case = {"kernel": kernel, "family": kernel, "target": target, "arm": "structured_ir", "repetition": 1,
                            "treatment": "audited_ir_current_backend", "reference_sha256": references.get(kernel, {}).get("ir_sha256")}
                    case["id"] = json_digest(case)[:24]
                    if kernel not in references:
                        rows.append({**case, "status": "reference_unresolved", "known_tokens": 0})
                    elif stop.is_set() or guard.spent() >= amount("30"):
                        rows.append({**case, "status": "not_started_budget_or_pause", "known_tokens": 0})
                    else:
                        source, manifest, _ = selected[kernel]
                        request = TranslationRequest(source_path=str(source), manifest_path=str(manifest),
                            targets=[TARGETS[target]], provider="openrouter", model=config.provider.model,
                            validation_policy="offline-validated")
                        seed_reference(request, settings, config, case, references[kernel], ROOT / references[kernel]["ir_path"])
                        state = run_case(request, settings, config, case, provider, compiler, stop)
                        rows.append(case_metrics(case, state, config))
                        if state["status"] == "blocked":
                            stop.set()
                    report = {"protocol": identity["protocol"], "status": "running", "cases": rows,
                              "preparation_accounting": selection["preparation_accounting"],
                              "shared_preflight": str(pilot / "preflight.json"),
                              "interpretation": "Diagnostic control; reference preparation is not free deployable work."}
                    write_json(output / "results.json", report)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        report["status"] = "blocked" if any(row["status"] == "blocked" for row in rows) else "paused" if any(
            row["status"] in ("paused", "not_started_budget_or_pause") for row in rows) else "completed"
        report["shared_budget_spent_usd"] = str(guard.spent())
        write_json(output / "results.json", report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "configs/research-next-steps-v1/diagnostic-selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=ROOT / "runs/research-next-steps-budget.json")
    parser.add_argument("--acknowledge-unknown-spending", action="store_true")
    args = parser.parse_args()
    run(args.selection, args.output, args.ledger, args.acknowledge_unknown_spending)
