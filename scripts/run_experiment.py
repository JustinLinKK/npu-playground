#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path

# Avoid a host BLAS thread pool per concurrent IR/reference validation.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from npu_agent.config import TARGETS
from npu_agent.experiment import run_experiment
from npu_agent.scenarios import ABLATIONS, CASE_STUDY_PROTOCOL, GUIDANCE_PROTOCOL, PROTOCOL_REVISION, SCENARIOS


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare controlled NPU translation loops")
    parser.add_argument("--kernels", type=_comma_list, help="comma-separated manifest kernel names")
    parser.add_argument("--scenarios", type=_comma_list, help=f"comma-separated {','.join(SCENARIOS + ABLATIONS + ('baseline_minimal',))} (default: protocol scenarios; recorded value on resume)")
    parser.add_argument("--protocol-revision", choices=[PROTOCOL_REVISION, CASE_STUDY_PROTOCOL, GUIDANCE_PROTOCOL], help="default: validated-ir-v2; recorded value on resume")
    parser.add_argument("--order-seed", type=int, help="shuffle case scheduling reproducibly; recorded value on resume")
    parser.add_argument("--targets", type=_comma_list, help=f"comma-separated: {','.join(TARGETS)}")
    parser.add_argument("--provider", choices=["openai", "codex-cli", "claude-cli", "openrouter"], help="default: protocol provider")
    parser.add_argument("--model", help="must match the protocol: qwen/qwen3-coder-30b-a3b-instruct for v3, gpt-5.6-terra otherwise")
    parser.add_argument("--max-cycles", type=int, help="initial attempt plus repairs (default: 10; recorded value on resume)")
    parser.add_argument("--case-workers", type=int, help="concurrent scenario cases (default: 1; recorded value on resume)")
    parser.add_argument("--max-compiler-jobs", type=int, help="global concurrent container validations (default: 2; recorded value on resume)")
    parser.add_argument("--compiler-cpus", type=int, help="CPU quota per container (default: 4; recorded value on resume)")
    parser.add_argument("--compiler-memory", help="memory limit per container (default: 8g; recorded value on resume)")
    parser.add_argument("--validation-policy", choices=["compile-only", "offline-validated", "target-executed"], default="offline-validated")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--resume", help="experiment ID or experiment directory")
    parser.add_argument("--corpus", type=Path, default=Path("examples/classic"))
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    output = run_experiment(
        repository=args.repository,
        corpus=args.corpus,
        runs_dir=args.runs_dir,
        kernel_names=args.kernels,
        scenarios=args.scenarios,
        target_ids=args.targets,
        provider_name=args.provider,
        model=args.model,
        resume=args.resume,
        validation_policy=args.validation_policy,
        max_cycles=args.max_cycles,
        case_workers=args.case_workers,
        max_compiler_jobs=args.max_compiler_jobs,
        compiler_cpus=args.compiler_cpus,
        compiler_memory=args.compiler_memory,
        protocol_revision=args.protocol_revision,
        order_seed=args.order_seed,
    )
    print(output)


if __name__ == "__main__":
    def stop(signum, frame):
        # Let the coordinator checkpoint in-flight stages on SIGTERM as on Ctrl+C.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    main()
