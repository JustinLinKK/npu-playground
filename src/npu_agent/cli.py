from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .baseline import translate_one_shot
from .compilers import DockerCompiler, build_images, smoke_images, capabilities
from .config import TARGETS, Settings
from .database import Database
from .models import BaselineResult, KernelManifest, TranslationRequest, TranslationResult, ValidationPolicy
from .validation import finalize_validation, policy_exit_code, write_json
from .workflow import build_workflow, resume_run, translate


def _target_ids(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(result) - set(TARGETS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown targets: {sorted(unknown)}")
    return result


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(
        database_path=Path(args.database),
        runs_path=Path(args.runs_path),
        repository_path=Path(args.repository).resolve(),
        provider=getattr(args, "provider", "openai"),
        model=getattr(args, "model", None),
        compiler_cpus=args.compiler_cpus, compiler_memory=args.compiler_memory,
        compiler_timeout_seconds=args.compiler_timeout_seconds,
        validation_policy=ValidationPolicy(getattr(args, "validation_policy", "compile-only")),
    )


def _request(args: argparse.Namespace, source: Path, manifest: Path) -> TranslationRequest:
    return TranslationRequest(
        source_path=str(source.resolve()),
        manifest_path=str(manifest.resolve()),
        targets=[TARGETS[item] for item in args.targets],
        provider=args.provider,
        model=args.model,
        search_rounds=getattr(args, "search_rounds", 3),
        branching_factor=getattr(args, "branching_factor", 3),
        debug_retries=getattr(args, "debug_retries", 2),
        validation_policy=ValidationPolicy(args.validation_policy),
    )


def _add_search_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--validation-policy", choices=[p.value for p in ValidationPolicy], default="offline-validated")
    parser.add_argument("--targets", type=_target_ids, default=list(TARGETS))
    parser.add_argument("--provider", choices=["openai", "codex-cli", "claude-cli"], default="openai")
    parser.add_argument("--model")
    parser.add_argument("--search-rounds", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--debug-retries", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="npu-agent")
    parser.add_argument("--database", default=".npu-agent/state.sqlite")
    parser.add_argument("--runs-path", default=".npu-agent/runs")
    parser.add_argument("--repository", default=".")
    parser.add_argument("--compiler-cpus", type=int, default=4)
    parser.add_argument("--compiler-memory", default="8g")
    parser.add_argument("--compiler-timeout-seconds", type=int, default=1200)
    commands = parser.add_subparsers(dest="command", required=True)

    environment = commands.add_parser("env", help="build or smoke-test compiler environments")
    environment_commands = environment.add_subparsers(dest="env_command", required=True)
    for name in ("build", "smoke", "capabilities"):
        command = environment_commands.add_parser(name)
        command.add_argument("--targets", type=_target_ids, default=list(TARGETS))
        if name == "capabilities":
            command.add_argument("--json", action="store_true")

    translate_command = commands.add_parser("translate", help="translate one kernel")
    translate_command.add_argument("source", type=Path)
    translate_command.add_argument("--manifest", type=Path, required=True)
    _add_search_options(translate_command)

    baseline = commands.add_parser("baseline", help="run a direct one-shot source-to-target baseline")
    baseline.add_argument("source", type=Path)
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--targets", type=_target_ids, default=list(TARGETS))
    baseline.add_argument("--provider", choices=["openai", "codex-cli", "claude-cli"], default="openai")
    baseline.add_argument("--model")
    baseline.add_argument("--validation-policy", choices=[p.value for p in ValidationPolicy], default="offline-validated")

    suite = commands.add_parser("suite", help="translate every manifest under a corpus")
    suite.add_argument("corpus", type=Path)
    _add_search_options(suite)

    for name in ("validate-candidate", "validate-suite"):
        command = commands.add_parser(name, help="validate deterministic backend code without a provider")
        command.add_argument("path", type=Path)
        command.add_argument("--validation-policy", choices=[p.value for p in ValidationPolicy], default="offline-validated")
        command.add_argument("--report-dir", type=Path, default=Path("runs/container-validation"))
        if name == "validate-candidate":
            command.add_argument("--manifest", required=True, type=Path)
            command.add_argument("--target", required=True, choices=list(TARGETS))
        else:
            command.add_argument("--targets", type=_target_ids, default=list(TARGETS))

    runs = commands.add_parser("runs", help="inspect or resume runs")
    runs_commands = runs.add_subparsers(dest="runs_command", required=True)
    show = runs_commands.add_parser("show")
    show.add_argument("run_id")
    resume = runs_commands.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--provider", choices=["openai", "codex-cli", "claude-cli"])
    resume.add_argument("--model")

    memory = commands.add_parser("memory", help="inspect or import agent knowledge")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    memory_list = memory_commands.add_parser("list")
    memory_list.add_argument("--hardware")
    memory_list.add_argument("--target", choices=list(TARGETS))
    memory_commands.add_parser("seed")
    memory_search = memory_commands.add_parser("search")
    memory_search.add_argument("query")
    memory_search.add_argument("--role", choices=["analysis", "coding", "optimize", "debug"], required=True)
    memory_search.add_argument("--target", choices=list(TARGETS))
    memory_search.add_argument("--dialect", choices=["cuda", "triton", "hip"])
    memory_import = memory_commands.add_parser("import")
    memory_import.add_argument("file", type=Path)
    memory_import.add_argument("--role", choices=["analysis", "coding", "optimize", "debug"], required=True)
    memory_import.add_argument("--title", required=True)
    memory_import.add_argument("--tags", default="")
    memory_import.add_argument("--dialect", choices=["cuda", "triton", "hip"])
    memory_import.add_argument("--backend", choices=["amd_xdna2", "intel_openvino"])
    memory_import.add_argument("--hardware")
    memory_import.add_argument("--target", choices=list(TARGETS))
    return parser


def _print_result(result: TranslationResult | BaselineResult) -> None:
    print(result.model_dump_json(indent=2))
    if result.status != "completed":
        raise SystemExit(3 if result.status == "blocked" else 1)


def _run_suite(args: argparse.Namespace, settings: Settings) -> int:
    manifests = sorted(args.corpus.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no manifest.json files found under {args.corpus}")
    results: list[TranslationResult] = []
    for manifest_path in manifests:
        manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        source = manifest_path.parent / manifest.source_file
        result = asyncio.run(translate(_request(args, source, manifest_path), settings))
        results.append(result)
        print(json.dumps({"kernel": result.kernel, "run_id": result.run_id, "status": result.status}))
    summary = {
        "provider": args.provider,
        "kernels": len(results),
        "completed": sum(item.status == "completed" for item in results),
        "failed": [item.kernel for item in results if item.status != "completed"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failed"] else 3 if all(item.status == "blocked" for item in results) else 1


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = _settings(args)
    settings.ensure_directories()
    try:
        if args.command in ("validate-candidate", "validate-suite"):
            compiler = DockerCompiler(settings)
            fixtures = [(args.path, args.manifest, args.target)] if args.command == "validate-candidate" else [
                (p.parent, p, target) for target in args.targets for p in sorted((args.path / target).rglob("manifest.json"))]
            if not fixtures:
                raise ValueError("no backend fixtures found")
            reports, codes = [], []
            for candidate, manifest_path, target_id in fixtures:
                manifest = KernelManifest.model_validate_json(manifest_path.read_text())
                result = compiler.compile(candidate, args.report_dir / target_id / candidate.name, manifest, TARGETS[target_id])
                report = finalize_validation(result.validation, args.validation_policy)
                reports.append(report.model_dump(mode="json"))
                codes.append(policy_exit_code(report))
                print(report.model_dump_json(indent=2))
            coverage = {}
            for target_id in sorted({r["target_id"] for r in reports}):
                coverage[target_id] = {}
                for stage in ("oracle_validation", "host_execution", "dataflow_simulation", "target_compile", "target_execution"):
                    values = [r["stages"].get(stage, {"status": "not_requested", "cases_run": 0}) for r in reports if r["target_id"] == target_id]
                    coverage[target_id][stage] = {"requested_candidates": len(values),
                        "supported_candidates": sum(s["status"] in ("passed", "failed") for s in values),
                        "executed_input_cases": sum(s.get("cases_run", 0) for s in values),
                        **{status: sum(s["status"] == status for s in values) for status in
                           ("passed", "failed", "blocked", "unsupported", "not_requested")}}
            summary = {"coverage": coverage, "total_requested": len(reports), "policy_passed": codes.count(0),
                       "failed": codes.count(1), "blocked_or_unsupported": codes.count(3), "results": reports}
            write_json(args.report_dir / "report.json", summary)
            raise SystemExit(1 if 1 in codes else 3 if 3 in codes else 0)
        if args.command == "env":
            if args.env_command == "build":
                build_images(settings, args.targets)
            elif args.env_command == "capabilities":
                print(json.dumps(capabilities(settings, args.targets), indent=2))
            else:
                smoke_images(settings, args.targets)
            return
        if args.command == "translate":
            _print_result(asyncio.run(translate(_request(args, args.source, args.manifest), settings)))
            return
        if args.command == "baseline":
            _print_result(asyncio.run(translate_one_shot(_request(args, args.source, args.manifest), settings)))
            return
        if args.command == "suite":
            raise SystemExit(_run_suite(args, settings))
        database = Database(settings.database_path)
        if args.command == "runs":
            row = database.get_run(args.run_id)
            if row is None:
                raise KeyError(f"unknown run: {args.run_id}")
            if args.runs_command == "show" or row["status"] in {"completed", "partial"}:
                for field in ("request_json", "result_json"):
                    if row.get(field):
                        row[field] = json.loads(row[field])
                print(json.dumps(row, indent=2))
                return
            request = TranslationRequest.model_validate_json(row["request_json"])
            settings.validation_policy = request.validation_policy
            settings.provider = args.provider or request.provider
            settings.model = args.model or request.model
            graph = build_workflow(settings, database=database)
            _print_result(resume_run(graph, args.run_id))
            return
        if args.memory_command == "list":
            print(json.dumps(database.list_memory(args.hardware, args.target), indent=2))
        elif args.memory_command == "seed":
            print(json.dumps(database.seed_builtin_knowledge(), indent=2))
        elif args.memory_command == "search":
            target = TARGETS.get(args.target) if args.target else None
            print(
                json.dumps(
                    database.retrieve_knowledge(
                        args.role,
                        args.query,
                        args.dialect,
                        target.backend.value if target else None,
                        target.hardware if target else None,
                        target.id if target else None,
                        target.vendor if target else None,
                        f"{target.id}:{target.compiler_version}" if target else None,
                    ),
                    indent=2,
                )
            )
        else:
            target = TARGETS.get(args.target) if args.target else None
            item_id = database.import_knowledge(
                args.role,
                args.title,
                args.file.read_text(encoding="utf-8"),
                args.tags,
                args.dialect,
                target.backend.value if target else args.backend,
                target.hardware if target else args.hardware,
                target_id=target.id if target else None,
                vendor=target.vendor if target else None,
                compiler_fingerprint=(f"{target.id}:{target.compiler_version}" if target else None),
            )
            print(json.dumps({"knowledge_id": item_id}))
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1 if isinstance(exc, RuntimeError) else 2) from exc


if __name__ == "__main__":
    main()
