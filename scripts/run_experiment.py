#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from npu_agent.config import TARGETS
from npu_agent.experiment import run_experiment


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one-shot and agentic GPU-to-NPU translation")
    parser.add_argument("--kernels", type=_comma_list, help="comma-separated manifest kernel names")
    parser.add_argument("--targets", type=_comma_list, help=f"comma-separated: {','.join(TARGETS)}")
    parser.add_argument("--provider", choices=["openai", "codex-cli", "claude-cli"], default="codex-cli")
    parser.add_argument("--model")
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
        target_ids=args.targets,
        provider_name=args.provider,
        model=args.model,
        resume=args.resume,
    )
    print(output)


if __name__ == "__main__":
    main()
