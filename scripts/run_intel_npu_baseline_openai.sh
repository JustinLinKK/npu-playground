#!/usr/bin/env bash
set -euo pipefail
repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository"
if [[ $# -gt 2 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s [RUN_ROOT [SOURCE_REPORT]]\n' "$0"
    printf 'Runs direct OpenAI calibration, then the 40-task Intel NPU 3720 baseline, using openai.api_key in config.yaml.\n'
    printf 'Requires NPU_AGENT_INTEL_NPU_3720_RUNNER (see docs/intel-npu-local.md); NPU results are observational.\n'
    printf 'INTEL_NPU_BASELINE_CONFIG selects the campaign config (default configs/intel-npu-baseline-openai.json).\n'
    printf 'NPU_AGENT_INTEL_NPU_3720_TIMING sets NPU latency runs per compiled blob as WARMUP,MEASURE (default 1000,200; 0,0 disables).\n'
    printf 'Attempt/token/time limits apply; no local dollar cap. Check billing on OpenAI.\n'
    exit 0
fi
if [[ -z "${NPU_AGENT_INTEL_NPU_3720_RUNNER:-}" ]]; then
    printf 'NPU_AGENT_INTEL_NPU_3720_RUNNER is not set; refusing to run the Intel baseline without the physical NPU.\n' >&2
    printf 'See docs/intel-npu-local.md. Keep the same value for every resume of a campaign.\n' >&2
    exit 2
fi
# Physical-NPU latency: warmup and timed runs of every compiled blob. Frozen into the campaign identity with the runner.
# Each executor process starts the NPU in a low-power state; latency settles only after a few hundred inferences.
export NPU_AGENT_INTEL_NPU_3720_TIMING="${NPU_AGENT_INTEL_NPU_3720_TIMING:-1000,200}"
config="${INTEL_NPU_BASELINE_CONFIG:-$repository/configs/intel-npu-baseline-openai.json}"
if [[ ! -f "$config" ]]; then
    printf 'Campaign config not found: %s\n' "$config" >&2
    exit 1
fi
config="$(cd -- "$(dirname -- "$config")" && pwd)/$(basename -- "$config")"
run_root="${1:-$repository/runs/$(basename -- "$config" .json)-$(date -u +%Y%m%dT%H%M%S%NZ)}"
source_report="${2:-$repository/examples/kernelbench/source-validation-report.json}"
if [[ ! -f "$source_report" ]]; then
    printf 'Source validation report not found: %s\n' "$source_report" >&2
    exit 1
fi
unset OPENAI_API_KEY
printf 'Campaign: %s\nConfig: %s\nProvider: direct OpenAI (key from config.yaml).\nNo local dollar cap; check billing on OpenAI.\n' \
    "$run_root" "$config"
printf 'NPU runner: %s\nNPU timing (warmup,measure): %s\nPhysical NPU results are recorded beside each attempt and do not affect acceptance.\n' \
    "$NPU_AGENT_INTEL_NPU_3720_RUNNER" "$NPU_AGENT_INTEL_NPU_3720_TIMING"
if [[ ! -f "$run_root/evaluation/campaign.json" ]]; then
    "$repository/.venv/bin/python" research/run_xdna2_baseline.py calibrate \
        --config "$config" --root "$run_root" --budget-on-openai
fi
"$repository/.venv/bin/python" research/run_xdna2_baseline.py evaluate \
    --config "$config" --root "$run_root" \
    --budget-on-openai --source-report "$source_report"
printf 'Report: %s/evaluation/results/REPORT.md\n' "$run_root"
