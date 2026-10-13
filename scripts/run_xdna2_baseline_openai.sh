#!/usr/bin/env bash
set -euo pipefail
repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository"
if [[ $# -gt 2 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s [RUN_ROOT [SOURCE_REPORT]]\n' "$0"
    printf 'Runs direct OpenAI calibration, then the 40-task baseline, using openai.api_key in config.yaml.\n'
    printf 'Attempt/token/time limits apply; no local dollar cap. Check billing on OpenAI.\n'
    exit 0
fi
run_root="${1:-$repository/runs/xdna2-baseline-openai-$(date -u +%Y%m%dT%H%M%S%NZ)}"
source_report="${2:-$repository/examples/kernelbench/source-validation-report.json}"
if [[ ! -f "$source_report" ]]; then
    printf 'Source validation report not found: %s\n' "$source_report" >&2
    exit 1
fi
unset OPENAI_API_KEY
printf 'Campaign: %s\nProvider: direct OpenAI (key from config.yaml).\nNo local dollar cap; check billing on OpenAI.\n' "$run_root"
if [[ ! -f "$run_root/evaluation/campaign.json" ]]; then
    "$repository/.venv/bin/python" research/run_xdna2_baseline.py calibrate \
        --config "$repository/configs/xdna2-baseline-openai.json" --root "$run_root" --budget-on-openai
fi
"$repository/.venv/bin/python" research/run_xdna2_baseline.py evaluate \
    --config "$repository/configs/xdna2-baseline-openai.json" --root "$run_root" \
    --budget-on-openai --source-report "$source_report"
printf 'Report: %s/evaluation/results/REPORT.md\n' "$run_root"
