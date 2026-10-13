#!/usr/bin/env bash
set -euo pipefail
repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository"
if [[ $# -gt 2 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s [RUN_ROOT [SOURCE_REPORT]]\n' "$0"
    printf 'Runs calibration, then the 40-task baseline, with dollar limits enforced only by OpenRouter.\n'
    exit 0
fi
run_root="${1:-$repository/runs/xdna2-baseline-openrouter-budget-$(date -u +%Y%m%dT%H%M%S%NZ)}"
source_report="${2:-$repository/examples/kernelbench/source-validation-report.json}"
if [[ ! -f "$source_report" ]]; then
    printf 'Source validation report not found: %s\n' "$source_report" >&2
    exit 1
fi
unset OPENROUTER_API_KEY
printf 'Campaign: %s\nDollar limits: OpenRouter only (key from config.yaml).\n' "$run_root"
if [[ ! -f "$run_root/evaluation/campaign.json" ]]; then
    "$repository/.venv/bin/python" research/run_xdna2_baseline.py calibrate \
        --root "$run_root" --budget-on-openrouter
fi
"$repository/.venv/bin/python" research/run_xdna2_baseline.py evaluate \
    --root "$run_root" --budget-on-openrouter --source-report "$source_report"
printf 'Report: %s/evaluation/results/REPORT.md\n' "$run_root"
