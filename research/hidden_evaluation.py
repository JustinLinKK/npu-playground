"""Separate final-input evaluation; never called by the translation/repair runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from npu_agent.artifacts import materialize_bundle
from npu_agent.config import Settings, TARGETS
from npu_agent.experiment import discover_kernels
from npu_agent.models import CodeBundle
from npu_agent.study_v4 import StudyCompiler, StudyConfig
from npu_agent.study_v4_reporting import case_metrics
from npu_agent.testcases import TestCase, generate_cases
from npu_agent.validation import json_digest, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def input_digest(inputs: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        array = np.asarray(value)
        digest.update(json.dumps([name, str(array.dtype), array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_hidden(suite_path: Path, selected: dict) -> tuple[dict, dict]:
    suite = json.loads(suite_path.read_text())
    if suite.get("protocol") != "hidden-inputs-v1" or not suite.get("reviewed_by") or not suite.get("review_notes"):
        raise ValueError("hidden inputs require a reviewed, versioned suite")
    if set(suite["kernels"]) != set(selected):
        raise ValueError("hidden suite must retain every selected kernel")
    result, hashes = {}, {}
    for name, (_, manifest_path, manifest) in selected.items():
        entry = suite["kernels"][name]
        if entry["manifest_sha256"] != sha256(manifest_path):
            raise ValueError("hidden suite manifest binding changed")
        feedback = {input_digest(case.inputs) for case in generate_cases(manifest)}
        cases, seen = [], set()
        inputs = {t.name: t for t in manifest.tensors if t.direction == "input"}
        for case in entry["cases"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) or case["id"] in seen:
                raise ValueError("hidden case IDs must be safe and unique")
            seen.add(case["id"])
            path = (suite_path.parent / case["path"]).resolve()
            if sha256(path) != case["sha256"]:
                raise ValueError("hidden input hash changed")
            hashes[f"{name}/{case['id']}"] = case["sha256"]
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(inputs):
                    raise ValueError("hidden inputs must match the input tensor signature")
                values = {key: archive[key].copy() for key in inputs}
            for key, tensor in inputs.items():
                value = values[key]
                if list(value.shape) != list(tensor.shape) or str(value.dtype) != tensor.dtype:
                    raise ValueError("hidden input shape/dtype mismatch")
                if manifest.numerical_domain == "finite" and not np.isfinite(value).all():
                    raise ValueError("hidden input violates the finite numerical domain")
            values.update({scalar.name: scalar.value for scalar in manifest.scalars})
            identity = input_digest(values)
            if identity in feedback:
                raise ValueError("hidden inputs overlap feedback inputs or another hidden case")
            feedback.add(identity)
            cases.append(TestCase(case["id"], values))
        if not cases:
            raise ValueError("each kernel requires hidden final inputs")
        result[name] = cases
    return result, hashes


def campaign_inputs(root: Path):
    campaign = json.loads((root / "campaign.json").read_text())
    config = StudyConfig.model_validate(campaign["config"])
    if config.stage != "confirmatory":
        raise ValueError("hidden final evaluation requires a separate confirmatory campaign")
    corpus = campaign["identity"]["corpus"]
    for relative, digest in corpus["hashes"].items():
        if sha256(Path(corpus["corpus"]) / relative) != digest:
            raise ValueError("frozen corpus changed")
    all_kernels = discover_kernels(Path(corpus["corpus"]))
    names = {case["kernel"] for case in campaign["cases"]}
    selected = {name: all_kernels[name] for name in names}
    return campaign, config, selected


def seal(root: Path, suite_path: Path, commitment: Path) -> dict:
    campaign, _, selected = campaign_inputs(root)
    if campaign["status"] not in ("prepared", "ready") or any((root / "cases").glob("*/report.json")):
        raise ValueError("seal hidden inputs before the first benchmark case")
    if commitment.exists():
        raise ValueError("do not overwrite an existing hidden-input commitment")
    _, hashes = load_hidden(suite_path, selected)
    record = {"protocol": "hidden-evaluation-v1", "campaign_identity": json_digest(campaign["identity"]),
              "suite_sha256": sha256(suite_path), "input_sha256": hashes, "evaluator_sha256": sha256(Path(__file__)),
              "candidate_rule": "first offline-passing bundle; otherwise the last submitted bundle; retain absent/unsupported cases"}
    write_json(commitment, record)
    return record


def evaluate(root: Path, suite_path: Path, commitment: Path, output: Path) -> dict:
    campaign, config, selected = campaign_inputs(root)
    if campaign["status"] != "completed":
        raise ValueError("final outcomes are unavailable until the campaign completes")
    if output.resolve().is_relative_to(root.resolve()) or output.exists():
        raise ValueError("use a new final-output directory outside the campaign")
    record = json.loads(commitment.read_text())
    if (record["campaign_identity"] != json_digest(campaign["identity"]) or
            record["suite_sha256"] != sha256(suite_path) or record["evaluator_sha256"] != sha256(Path(__file__))):
        raise ValueError("committed inputs or evaluator changed")
    for relative, digest in campaign["identity"]["source_hashes"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError("frozen evaluator implementation changed")
    hidden, hashes = load_hidden(suite_path, selected)
    if hashes != record["input_sha256"]:
        raise ValueError("hidden input commitment mismatch")
    states = {}
    for case in campaign["cases"]:
        state = json.loads((root / "cases" / case["id"] / "report.json").read_text())
        if state["status"] not in ("completed", "failed") or state.get("phase") is not None:
            raise ValueError("all cases must be terminal before final evaluation")
        states[case["id"]] = state
    output.mkdir(parents=True, mode=0o700)
    settings = Settings(repository_path=ROOT, runs_path=output, database_path=output / "unused.sqlite",
                        validation_policy="offline-validated", experiment_id="hidden-final-v1",
                        compiler_cpus=config.compiler_cpus, compiler_memory=config.compiler_memory,
                        max_compiler_jobs=1, compiler_timeout_seconds=config.compiler_timeout_seconds)
    compiler = StudyCompiler(settings)
    preflight = json.loads((root / "preflight.json").read_text())
    for target in config.targets:
        if compiler.identity(TARGETS[target]) != preflight["identity"]["compilers"][target]:
            raise ValueError("compiler identity changed since feedback evaluation")
    rows = []
    for case in campaign["cases"]:
        state = states[case["id"]]
        metrics = case_metrics(case, state, config)
        bundles = [cycle for cycle in state["cycles"] if "bundle" in cycle]
        cycle = next((cycle for cycle in bundles if cycle["status"] == "passed"), bundles[-1] if bundles else None)
        row = {**case, "feedback_success": metrics["success"], "hidden_pass": False, "final_success": False,
               "status": "no_candidate", "state_sha256": sha256(root / "cases" / case["id"] / "report.json")}
        if cycle:
            bundle = CodeBundle.model_validate(cycle["bundle"])
            row.update(cycle=cycle["number"], bundle_sha256=json_digest(cycle["bundle"]))
            if bundle.unsupported_operations:
                row["status"] = "unsupported"
            else:
                directory = output / case["id"]
                materialize_bundle(bundle, directory / "candidate")
                # Standalone, serial evaluation only: no provider or repair loop is invoked.
                started = time.perf_counter()
                try:
                    with patch("npu_agent.compilers.generate_cases", return_value=hidden[case["kernel"]]):
                        result = compiler.compile(directory / "candidate", directory / "validation",
                                                  selected[case["kernel"]][2], TARGETS[case["target"]])
                except Exception as exc:
                    row.update(status="blocked", error=type(exc).__name__, duration_seconds=time.perf_counter() - started)
                    write_json(output / "results.json", {"protocol": "hidden-evaluation-v1", "commitment": record,
                               "status": "blocked", "cases": [*rows, row]})
                    raise
                write_json(directory / "result.json", result.model_dump(mode="json"))
                row.update(status="evaluated", hidden_pass=result.validation.offline_contract_met,
                           final_success=metrics["success"] and result.validation.offline_contract_met,
                           duration_seconds=result.duration_seconds)
        rows.append(row)
        write_json(output / "results.json", {"protocol": "hidden-evaluation-v1", "commitment": record,
                   "status": "running", "cases": rows})
    report = {"protocol": "hidden-evaluation-v1", "commitment": record, "status": "completed", "cases": rows}
    write_json(output / "results.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "evaluate"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "seal":
        seal(args.campaign, args.suite, args.commitment)
    elif args.output is None:
        parser.error("evaluate requires --output")
    else:
        evaluate(args.campaign, args.suite, args.commitment, args.output)
