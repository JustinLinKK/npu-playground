#!/usr/bin/env bash
# The Intel NPU 3720 baseline on the prepared intel-fp16-v1 dataset (FP16 inputs, references and tolerance).
set -euo pipefail
repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INTEL_NPU_BASELINE_CONFIG="$repository/configs/intel-npu-fp16-baseline-openai.json" \
    exec bash "$repository/scripts/run_intel_npu_baseline_openai.sh" "$@"
