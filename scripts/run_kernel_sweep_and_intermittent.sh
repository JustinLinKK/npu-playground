#!/usr/bin/env bash
# NPU vs GPU energy, steps 1 and 2: the size and batching sweep (about 90 min), then the intermittent-load run
# (about 30 min). Both use research/benchmark_kernel_energy.py; method and results: docs/time-energy-measurement.md.
#
#   bash scripts/run_kernel_sweep_and_intermittent.sh                       # both steps
#   STEPS=sweep bash scripts/run_kernel_sweep_and_intermittent.sh           # or STEPS=intermittent
#   BASELINE=p8 bash scripts/run_kernel_sweep_and_intermittent.sh           # quiet-machine baseline (waits for GPU P8)
#
# With the default desktop baseline, keep desktop use light and steady while it runs (no video, games or other GPU
# work started midway): the idle windows measure the machine as it is being used.
set -euo pipefail
repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${GPU_VENV:-$HOME/.cache/npu-playground-gpu-venv}"
python="$venv/bin/python"
windows_python="${WINDOWS_PYTHON:-/mnt/c/npu-runner/venv/Scripts/python.exe}"
baseline="${BASELINE:-desktop}"
steps=",${STEPS:-sweep,intermittent},"
export CC="$venv/zigcc/cc"  # Triton's launcher needs a C compiler; the machine has none (see research/time_original_gpu_kernels.py)
cd "$repository"

[[ -x "$python" ]] || { echo "GPU venv not found: $venv (setup: research/time_original_gpu_kernels.py)" >&2; exit 1; }
[[ -x "$windows_python" ]] || { echo "Windows NPU runner not found: $windows_python (setup: docs/intel-npu-local.md)" >&2; exit 1; }
"$python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available in WSL"'
"$python" -c 'import tqdm' 2>/dev/null || echo 'tqdm is not installed in the GPU venv: progress is shown as counter lines.'
[[ "$baseline" == p8 ]] && idle=(--start-when-idle 60) || idle=()

if [[ "$steps" == *,sweep,* ]]; then
    echo "== Step 1: size and batching sweep (baseline: $baseline, about 90 min)"
    "$python" research/benchmark_kernel_energy.py --sweep all --baseline "$baseline" --cycles 4 --run-seconds 5 \
        --idle-seconds 4 --windows-python "$windows_python" "${idle[@]}"
fi
if [[ "$steps" == *,intermittent,* ]]; then
    echo "== Step 2: intermittent load (baseline: $baseline, about 30 min)"
    "$python" research/benchmark_kernel_energy.py --intermittent relu_b1 --baseline "$baseline" \
        --windows-python "$windows_python" "${idle[@]}"
fi
echo '== Done. The reports are the "Report:" paths above (runs/kernel-sweep-*/SWEEP.md, runs/kernel-intermittent-*/INTERMITTENT.md).'
