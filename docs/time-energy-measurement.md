# Time and energy measurement for NPU and GPU kernels

This document describes how we measure the execution time and energy of the generated Intel NPU kernels and of
the original CUDA, HIP and Triton kernels they translate. It covers what each number means, how it is computed, and
which tools we use or rejected. Update it whenever the method changes (see the revision history at the end).

Code:

- `research/benchmark_kernel_energy.py`: orchestrator; runs in WSL.
- `research/energy_meter.py`: pure computation, shared with the Windows side.
- `scripts/windows_power_sampler.py`: Windows power sampler.
- `scripts/intel_npu_bench_worker.py`: Windows NPU worker.
- `research/nvbench_original_kernels.py`: NVBench timing of the GPU kernels.
- `scripts/benchmark_app_blob.py`: OpenVINO `benchmark_app` shim for compiled blobs.
- `research/sweep_kernels.py`: synthetic single-op kernels for the size and batching sweep.

Results go to `RUN_ROOT/kernel-{validate,timing,energy}-<UTC>/`. Sweep and intermittent runs need no campaign, so
they go to `runs/kernel-{sweep,intermittent}-<UTC>/`.

## What is compared

| | NPU | GPU |
| --- | --- | --- |
| Device | Intel NPU 3720 ("Intel AI Boost") in a Core Ultra 7 270K Plus | NVIDIA GeForce RTX 5070 (sm_120), also driving the display |
| Code | Each task's accepted blob from a finished campaign, checked against its compile-time hash | The dataset's original source, unmodified, at its original dtype |
| Runtime | Windows-native OpenVINO 2026.3.1 (`C:\npu-runner\venv`) | WSL2: CuPy NVRTC for CUDA and HIP; torch and Triton for Triton |
| Correctness gate | Outputs bit-identical to the campaign's saved device outputs | All 12 stored cases within rtol = atol = 2⁻¹⁰ |
| Inputs | The first stored case, bound once | The first stored case, resident on the device |

Precision differs: the NPU kernels compute in FP16. The original GPU kernels compute in float32, except
`cuda_tiled_matmul` and `triton_fused_attention`, which use FP16. Shapes are the dataset's reduced test shapes.

## Quick start

All commands run in the GPU venv. `CC` is needed for Triton's launcher because the machine has no system compiler.

```bash
PY="env CC=$HOME/.cache/npu-playground-gpu-venv/zigcc/cc $HOME/.cache/npu-playground-gpu-venv/bin/python"
CAMPAIGN=runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z/evaluation
$PY research/benchmark_kernel_energy.py --campaign $CAMPAIGN --validate                          # instruments, ~6 min
$PY research/benchmark_kernel_energy.py --campaign $CAMPAIGN --timing-only --tasks all --cross-check   # ~65 min
$PY research/benchmark_kernel_energy.py --campaign $CAMPAIGN --tasks subset                      # energy, ~60 min
$PY research/benchmark_kernel_energy.py --reanalyse RUN_ROOT/kernel-energy-<UTC>   # current estimator on an old run
bash scripts/run_kernel_sweep_and_intermittent.sh      # the two commands below, with checks; BASELINE=p8, STEPS=sweep
$PY research/benchmark_kernel_energy.py --sweep all --baseline desktop --cycles 4 --run-seconds 5 \
    --idle-seconds 4                                                                   # size and batching, ~90 min
$PY research/benchmark_kernel_energy.py --intermittent relu_b1 --baseline desktop       # bursts, ~30 min
```

- **Leave the machine idle during `--validate` and energy runs.** Background load is the main source of energy noise.
- **Locks:** the orchestrator holds `.npu-agent/npu.lock` and `.npu-agent/energy.lock`, so campaign NPU executions
  wait for it.
- **Setup:** the GPU venv also needs `uv pip install "cuda-bench[cu13]"`. See
  `research/time_original_gpu_kernels.py` for the rest.
- **Re-analysis:** `--reanalyse` recomputes an energy, sweep or intermittent run from its `trace.csv` and
  `windows.json` with the current estimator. It writes `<REPORT>-reanalysed.md` and `summary-reanalysed.json`, and
  leaves the run's own report untouched.
- **Progress:** timing, energy, sweep and intermittent runs show a progress bar on stderr. It counts measurements
  (one run or burst series plus the idle after it) and shows elapsed time and an ETA. It uses `tqdm` if installed
  (`uv pip install tqdm` into the GPU venv), otherwise prints one counter line per measurement.
- **Baseline (`--baseline`):** see "Two baselines" under Formulas. `p8` (the default) needs a quiet GPU; `desktop`
  measures against normal use.
- **Waiting for an idle machine:** with `--baseline p8`, `--start-when-idle S` holds the run until the GPU has sat at
  P8 for S seconds (at most 12 h). Desktop programs can keep the GPU out of P8 indefinitely: on 2026-11-01, with
  Chrome, WeChat, Steam and the NVIDIA overlay open, it spent 2% of 30 s at P8 and 20% at P0. Every idle window was
  then unusable as a baseline. Even after closing some of those programs it was at P8 only 55% of the time, so the
  wait never ended. This is why `desktop` exists.

## Time

### Metrics

Every metric is the time of one kernel call on the first input case. All are reported in µs.

| Metric | NPU | GPU | Includes |
| --- | --- | --- | --- |
| **Rounds** | 5 × 2000 synchronous `infer()` calls, each timed with `perf_counter` | 5 × 20000 launch+sync calls (also end to end) | Host dispatch, execution, completion wait |
| **Sync window** | Back-to-back `infer()` for T s; reports T/N | Back-to-back launch+sync for T s; reports T/N | Same as rounds, plus every outlier |
| **Pipelined window** | `AsyncInferQueue` with 4 requests in flight for T s; reports T/N | Back-to-back launches, one synchronize at the end; reports T/N | Steady-state cost per call when dispatch overlaps execution |
| **End-to-end window** | n/a | Upload inputs, launch, download output, for T s; reports T/N | Host↔device copies |
| **Device time** | No comparable timer (see below) | NVBench *cold* GPU time | Kernel execution only |
| **Batch** | n/a | NVBench *batch* GPU time | Back-to-back launches until 0.5 s of GPU time has accumulated |

The fixed windows answer "how many runs fit in T seconds". `T` is measured from the first submission to the last
completion, because the final synchronize or `wait_all` happens inside the measured time. The warm-up's queued work
is drained (synchronize or `wait_all`) just before the window starts. Otherwise up to about a thousand queued GPU
launches would execute inside the window without being counted. That is negligible for launch-bound tiny kernels,
whose queue stays nearly empty, but not for the sweep's large kernels, where the device is the bottleneck. The clock is read only
every 16 calls, so timer overhead does not inflate short calls. Rounds report the mean, p50, p95, p99 and the spread
between rounds.

**Inputs and outputs.**

- *NPU:* inputs are bound once into plugin-owned buffers by `infer(feed)`, and later calls run `infer()` without
  inputs. Sync calls return copies of the outputs, as `InferRequest.infer` does by default.
- *GPU:* inputs stay on the device except in end-to-end mode.
- The earlier replay timing (`replay_intel_local_npu.py --timing`) used `infer(feed)` on every call. It is a few
  percent slower.

**Warmup.**

- *NPU:* 1000 synchronous calls after loading, plus 1 s of the same mode immediately before every window. Each
  executor process starts the NPU in a low-power state. After a 5 s idle gap, the first 0.25 s of calls were slower
  (212 µs per call for relu), after which the per-call time oscillated between about 150 and 190 µs on a cycle of
  roughly 0.5 s.
- *GPU:* each mode first runs for at least 200 calls and 3 s (`--gpu-warmup`, `--gpu-warmup-seconds`), plus 1 s of
  the same mode before every window. The first pipelined launches of a process speed up from about 6.2 to 5.6 µs per
  call over about 3 s. GPU clocks stay at P1 and 2902 MHz throughout, so the warm-up is on the host side. With only
  200 warm-up calls (about 1 ms), the first pipelined window ran 3–8% faster in its second half than its first. After
  the change, first-window changes were +3.2, −4.0, +0.5 and +0.5% on four tasks.

### Cross-checks with standard tools

- **OpenVINO `benchmark_app`** runs on the same blob in sync mode (`-api sync -nireq 1`) and async mode
  (`-api async -nireq 4`), with `-hint none` and inputs filled once with random data. It is called through
  `scripts/benchmark_app_blob.py`. The `benchmark_app` shipped with 2026.3.1 cannot load a `.blob`: its `-ip`, `-op`
  and `-iop` options default to `None`, which its compiled-model check treats as a requested precision, and it passes
  a path to `Core.import_model`, which accepts only bytes. The shim fixes both without touching the measurement loop.
- **NVIDIA NVBench** (PyPI `cuda-bench`) runs the same launch closures on its own stream:
  - *Cold:* each launch is held behind a blocking kernel, so the time is device-only. Sampling stops once relative
    noise falls to 0.5%, with a minimum of 0.5 s and a timeout of 15 s.
  - *Batch:* back-to-back launches until 0.5 s of GPU time has accumulated.
  - The current stream is set once around `state.exec`. Entering a stream context on every launch added several µs
    of Python overhead, which inflated Triton's batch time from 6.9 to 12.5 µs.
  - NVBench discards and repeats samples when GPU clocks fall below 75% of the default. This happens often with
    these tiny kernels.

### Consistency gates

These are reported in TIMING.md. A failed gate is a reason to inspect the result, not to drop it.

| Gate | Limit | Why |
| --- | ---: | --- |
| NPU sync window vs rounds mean | 5% | Two estimates of the same quantity |
| GPU sync window vs rounds mean | 10% | Same, allowing for the WSL2 launch-latency states below |
| NPU sync and async window vs `benchmark_app` | 10% | Independent implementation |
| GPU pipelined window vs NVBench batch | 15% | Independent implementation |
| Throughput in the first vs second half of each window: NPU modes, GPU pipelined and end to end | 5% | Catches power-state changes inside a window |
| The same for GPU sync | 10% | Same, allowing for the WSL2 launch-latency states below |

**GPU sync calls have latency states.** On WSL2, a launch+sync call takes about 17.5 µs for a while, then 22–27 µs
for a while, switching every 0.1–1 s. In the 2026-11-01 timing run, the five rounds of 5000 calls (0.1 s each) had
means varying by up to 50% within a task. The 3 s windows of the same task agreed within about 2%, but their 1.5 s
halves differed by up to 6.3%, with either sign. A 5% limit therefore failed on noise rather than on a method error.
Rounds now take 20000 calls each (2 s in total per mode instead of 0.5 s), and the GPU sync gates allow 10%. On a
4-task check after the change, GPU sync windows and rounds agreed within 3.5%. A real warm-up ramp, as in the
pipelined case above, shows up as a consistently positive change, mostly in the first window.

In the first smoke run, `benchmark_app` agreed within 3% (sync) and 2% (async), and NVBench batch agreed with the
pipelined window within 8%.

**CUDA events around one launch are not device time.** On an idle stream, the start event fires immediately, and
the kernel starts only after the host has finished launching it. So events include the launch latency, which is
large for Triton's Python launcher (11.4 µs by events vs 6.0 µs NVBench cold in one run). TIMING.md shows events next
to NVBench cold, and their difference is reported as `gpu_launch_gap_us`. NVBench cold is the device time.

**NPU device time.** The NPU plugin reports per-layer profiling only for blobs compiled with profiling enabled. On
these blobs it returns "No available profiling data". The driver-level mode (`NPU_PROFILING_TYPE=INFER`) reports only
AVG/MIN/MAX aggregates and slows inference itself. So for the NPU we report host-observed time only.

## Energy

### Telemetry

| Source | Read through | Rails | Resolution | Notes |
| --- | --- | --- | --- | --- |
| Windows Energy Meter (Intel RAPL) | PDH, `\Energy Meter(RAPL_Package0_*)\Energy` and `\Time`; no admin rights | PKG (package), PP0 (CPU cores), PP1 (integrated graphics), DRAM | Updates about every 5 ms; energy in pWh, time in ms | DRAM reads 0 on this platform. There is no NPU rail. |
| NVML (`C:\Windows\System32\nvml.dll`) | `nvmlDeviceGetFieldValues(NVML_FI_DEV_POWER_INSTANT)` | GPU board | Updates about every 100 ms | `nvmlDeviceGetPowerUsage` is a 1 s average and is logged for comparison only. |

- **Uncore.** `uncore = PKG − PP0 − PP1` is the package outside the CPU cores and the integrated graphics. It contains
  the NPU, so it is our best proxy for NPU power, and it excludes the host CPU thread that drives the NPU.
- **Never use NVML's total-energy counter.** On this RTX 5070, `nvmlDeviceGetTotalEnergyConsumption` reads about 25×
  higher than the integrated power, on both WSL and native Windows. A similar report exists on NVIDIA's forum for an
  Ada card. No code reads it, and a test enforces this.
- **Sampler.** It polls every 20 ms and also logs CPU utility, GPU P-state, SM clock and utilization. The P-state
  also decides which idle windows count as baselines (next item). A `latest` command returns the newest sample.
- **The GPU holds a busy P-state after work.** After the last launch, the RTX 5070 stays in P1 for 5–7 s, at about
  23 W of board power while its clocks step down. It then passes through P3 or P5 and reaches P8, about 11 W, 7–9 s
  after the run. During the P1 hold, uncore reads about 1.8 W *lower* than with the GPU at P8 (10.7 vs 12.5 W); the
  cause is unknown. An idle window inside this hold is therefore not a baseline: it understates the GPU's board
  increment and overstates the NPU's uncore increment. The NPU shows no such tail: uncore is back at its idle value
  within 0.5 s of an NPU run.

### Session layout

1. **Idle means everything loaded, no work submitted.** The NPU worker has its blob imported and waits on stdin, and
   the orchestrator's CUDA context is alive.
2. **Per task, windows alternate `I R I R … I`.** Run types are NPU-sync, NPU-async, GPU-sync and GPU-pipelined. The
   order is shuffled in each of K cycles with a fixed seed.
3. **Defaults:** 6 s runs, 5 s idles, K = 6, 1 s warm-up before each run and 0.5 s of busy tail after it.
   - After a GPU run, the orchestrator polls the sampler until the GPU has been at P8 for 0.5 s (at most `--settle`,
     20 s). It then waits 0.5 s (`--guard`) before the next idle window, which is all it waits after an NPU run. The
     same P8 wait precedes each task's first idle window, because the GPU correctness checks leave the GPU in P1.
   - Each task takes about 7 minutes, so the 8-task subset takes about an hour.
4. **Marking windows.** All window marks are timestamps on the sampler's clock. Each run's power window lies strictly
   inside the device's busy span:
   - The NPU worker prints `begin` and `end` the moment they happen, and the orchestrator forwards them to the sampler.
   - GPU windows send marks without waiting for a reply.
   - The WSL → Windows marker round trip is about 0.1 ms.
5. **Time and power come from different clocks.** Time per call (T/N) comes from the worker's own clock and power
   from the sampler's clock, so the WSL and Windows clocks never need to be aligned.
6. **The 8-task subset:** `kb_019_relu`, `kb_049_max_reduction`, `kb_027_selu`, `kb_036_rms_norm`, `cuda_tiled_matmul`,
   `cuda_reduce_sum`, `hip_image_convolution`, `triton_fused_softmax`. It has one task per operation family, all
   correct on both devices, and includes multi-launch and Triton kernels.

### Formulas

For a window `[a, b]` on the sampler clock:

- Rail energy: `E_r = C_r(b) − C_r(a)`, with the cumulative counter `C_r` converted to joules (pWh × 3.6·10⁻⁹) and
  interpolated at the window edges.
- Rail power: `P_r = E_r / (b − a)`.
- GPU board power: the trapezoid integral of the instant-power samples over `[a, b]`, with interpolated edges, divided
  by `b − a`.
- **Paired difference** for run k: `ΔP_k = P(R_k) − [P(I_before) + w_k·(P(I_after) − P(I_before))]`, with
  `w_k = (m(R_k) − m(I_before)) / (m(I_after) − m(I_before))` and `m` a window's midpoint on the sampler clock. The idle
  power is interpolated linearly in time to the run's midpoint, which cancels linear drift exactly however the windows
  are spaced. Until 2026-11-01 the estimator used the plain mean `½·(P(I_before) + P(I_after))`, which cancels drift
  only for a run exactly midway. In practice runs sat 11–89% of the way between their baselines, because of the P8
  wait and the skipped idles below. Re-analysis with interpolation moved no headline mean by more than 0.5%.
- **Only settled idle windows are baselines.** `I_before` and `I_after` are the nearest idle windows in which the GPU
  sat at P8 for at least 90% of the samples (`energy_meter.gpu_settled`). Others are skipped, and ENERGY.md states how
  many were used. A run with no settled idle window on one side is dropped, so n can fall below K.
- **NPU runs with a GPU wake-up are dropped.** The GPU should stay at P8 throughout an NPU run. If it leaves P8,
  another program (the desktop, a browser) used it, and that activity also disturbs the package rails. The same 90%
  rule applies, and ENERGY.md reports the count: 1 of 96 NPU runs in the first energy run.
- **Two baselines (`--baseline`).** Both subtract the interpolated idle power above; they differ in which idle
  windows count and how long the schedule waits after GPU work.

  | | `p8` (default) | `desktop` |
  | --- | --- | --- |
  | Question answered | Extra energy over a quiet machine | Extra energy over the machine as it is being used |
  | Idle windows used | Only those with the GPU at P8 for ≥ 90% of samples | All of them, whatever state other programs keep the GPU in |
  | NPU runs and bursts | Dropped if the GPU leaves P8 during them | Never dropped |
  | Wait after GPU work | Until P8 is held for 0.5 s (at most `--settle`, 20 s) | A fixed `--desktop-settle`, 10 s: the GPU's own P1 hold lasts 5–7 s, and P8 followed 7–9 s after the run on a quiet desktop |
  | Needs | A quiet GPU; desktop programs can prevent P8 for good | Desktop activity that is steady over the run, so the K shuffled repeats turn it into noise, not bias |

  The `desktop` baseline is a different quantity, not a noisier estimate of the same one:
  - *GPU:* a GPU already clocked up by desktop programs costs less extra power to run a kernel than one at P8, so
    ΔGPU board is smaller than with `p8`.
  - *Variance:* desktop programs add CPU and GPU noise to both runs and idles, so CIs are wider.
  - *Reporting:* every report states its baseline, the mean and range of idle GPU board power, and the share of idle
    samples at P8, so readers can see what the desktop was doing.
- **Energy per inference** (µJ): `e_k = ΔP_k · t_k`, where `t_k = T_k / N_k` comes from the same window.
- **Gross energy** `P(R_k) · t_k` includes idle power as well. It is dominated by the ~32 W package idle, so it is
  reported but not used as the headline.

What counts for each device:

| Device | Headline (`HEADLINE` in the code) | Breakdown |
| --- | --- | --- |
| NPU | `soc` = ΔPKG · t | `uncore` (NPU and fabric), `host_cores` (PP0, the host thread driving the NPU) |
| GPU | `system` = (ΔPKG + ΔGPU board) · t | `host_cores`, `gpu_board` |

**Energy efficiency ratio (EER).** A device's EER on a task is inferences per joule, `1 / ē`, from the mean energy
per inference `ē`. It is shown only when `ē` is significant. Devices are compared task by task, then summarized:

- **Per task:** `EER(GPU) / EER(NPU) = ē_NPU / ē_GPU`; above 1 means the GPU is more efficient. Its 95% CI uses the
  delta method on the log scale, `SE(log r) = √((SE_NPU/ē_NPU)² + (SE_GPU/ē_GPU)²)`, with the t quantile at the smaller
  n − 1 (`energy_meter.ratio`).
- **Across tasks:** the geometric mean of the per-task ratios, with the range and a t-interval of the log ratios
  (`energy_meter.geometric_mean`). Ratios are never formed from the arithmetic means over tasks. Those weight the
  slowest tasks most, and until 2026-11-01 the Results section quoted them (1.4× and 3.5×; the geometric means are
  1.42× and 3.59×).
- **Pairs compared:** `sync` = NPU sync vs GPU sync; `throughput` = NPU async vs GPU pipelined. Each uses the headline
  energies (NPU SoC vs GPU system) and the device-only energies (NPU uncore vs GPU board). ENERGY.md and
  `summary.json` (`comparison`) carry all four.

### Statistics

- For the K paired values per task and mode: the mean, `SE = sd/√K`, and a 95% CI of mean ± t(0.975, K−1) · SE.
  A t-table is embedded because the GPU venv has no SciPy.
- A result is **significant** only if its CI excludes zero. Otherwise the report shows a one-sided 95% upper bound,
  marked "n.s.".
- `MDE80 = (t(0.975) + t(0.80)) · SE` is the smallest effect detectable with 80% power.
- Across tasks, the report gives the mean of the per-task headline energies with a between-task CI.
- A fixed-seed Monte Carlo test (`tests/test_kernel_energy.py`) checks that, under drift, the estimator is unbiased,
  CI coverage is 0.90–0.98, and the false-positive rate under the null is at most 8%.

**Noise floor.** Measured on 2026-10-31 over 15 one-second idle windows: PKG 32.2 ± 3.0 W, cores 20.0 ± 2.6 W,
integrated graphics 0.11 ± 0.02 W. `--validate` measures the spread for 1 to 10 s windows and prints the MDE at the
chosen K. Background bursts land mostly in PP0, so the uncore rail is much quieter, which is why it is the NPU proxy.

## Validation (`--validate`)

- **V1, idle noise:** 180 s idle. Reports the standard deviation of window-mean power per rail for 1, 2, 5 and 10 s
  windows, and the MDE at the default T and K. Use it to confirm or retune `--run-seconds`, `--idle-seconds` and
  `--cycles`.
- **V3, CPU load steps:** 1, 2 and 4 busy processes, 3 repeats each. Expect ΔPP0 to rise with load, ΔPKG ≥ ΔPP0 and
  ΔPP1 ≈ 0. This confirms the rails and the paired estimator respond correctly.
- **V5, GPU power:** a 20 s FP16 matmul load. The integrated NVML instant power, the NVML 1 s average and
  `nvidia-smi power.draw.instant` / `power.draw.average` should agree within about 5%. It also measures how long the
  GPU takes after the load to reach P8 with board power within 2 W of its P8 idle (`seconds_to_p8_after_load`). The
  idle reference is taken after waiting for P8.

Not yet implemented, optional: a null run with a sleeping worker, and a large NPU matmul as a positive control for
ΔUncore.

**Results on 2026-10-31** (`kernel-validate-20261031T231029Z`, machine left idle):

| Check | Result |
| --- | --- |
| V1 idle noise (SD of window means, W) | PKG 1.80 (1 s), 1.38 (2 s), 0.99 (5 s), 0.81 (10 s). Cores track PKG (1.72 at 1 s). Uncore is only 0.26 (1 s) and 0.17 (5 s). GPU board 2.66 (1 s). |
| Smallest detectable effect at the defaults (K = 6, 6 s / 5 s) | PKG 1.73 W, cores 1.69 W, **uncore 0.29 W**, GPU board 3.89 W |
| V3 CPU load, ΔW for 1 / 2 / 4 busy processes | PKG 11.9 / 22.8 / 40.8, all in PP0 (11.8 / 22.8 / 40.8). PP1 and uncore within ±0.15 W of 0. |
| V5 GPU, 20 s FP16 matmul | Integrated NVML instant 249.95 W, NVML 1 s average 250.00 W, `nvidia-smi` instant and average 250.00 W. The load is power-capped at 250 W, so this shows agreement only at the cap. Back to within 2 W of the reference idle (31.4 W) after 1.9 s; that reference was itself taken in the P1 hold, so this settle time was wrong (see the correction below). |

What this tells us:

- The rails respond as expected.
- Host-core load does not leak into uncore, so uncore isolates the NPU side from the CPU thread that drives it.
- Uncore noise is low enough to resolve sub-watt NPU increments.
- The defaults are kept.

**Correction (2026-11-01).** The conclusion "the 3 s settle after GPU runs is enough" was wrong. V5's idle reference
was measured right after the matmul setup, while the GPU was still in its P1 hold, so "within 2 W of idle" meant
"back to about 33 W", not back to P8. The first energy run then showed GPU board power still at 18–21 W in idle
windows after GPU runs, against about 12 W after NPU runs. The schedule now waits for P8, the estimator skips
unsettled idle windows, and V5 measures the time to P8.

## Size and batching sweep (`--sweep`)

The campaign's shapes are 1–24 KB, so both devices spend nearly all of each call on dispatch: a generated one-op
ReLU takes 165 µs on the NPU, the same as any other one-op graph. The sweep asks how time and energy per unit of work
change when each call carries more work.

- **Kernels** (`research/sweep_kernels.py`): one native op per configuration, FP16 inputs and outputs on both devices.
  - *NPU:* an OpenVINO graph with a single op (`Add`, `ReLU`, `SoftMax`, `MatMul`). This is the best graph a
    generator could emit, so the sweep measures the hardware and runtime, not the generated code.
  - *GPU:* the matching torch op (`add`, `clamp`, `softmax`, `matmul` on cuBLAS), with outputs preallocated where
    torch accepts `out=`.
  - Inputs: one standard-normal FP16 draw per configuration (seed 20261101), saved and fed to both devices.
- **Size axis:** one instance grows. Vector add with 4096 to 16.8M elements; square matmul from 64 to 2048; softmax
  from 32×128 to 1024×4096.
- **Batch axis:** B independent copies of a dataset task's shape in one call. ReLU `[B, 4, 65]` (kb_019_relu, B up to
  65536), matmul `[B, 64, 64]` (cuda_tiled_matmul) and softmax `[B, 32, 128]` (triton_fused_softmax), B up to 256.
  B = 1 is the dataset shape itself.
- **Measurement:** the energy schedule of the campaign runs (`I R I R … I`, all four modes, paired and
  time-interpolated baselines), with K = 4, 5 s runs and 4 s idles. Time per call comes from the same run windows.
- **Correctness:** both devices' outputs are compared with a float32 reference rounded to FP16, at rtol = atol = 2⁻¹⁰.
  This is reported, not gated.
- **Per unit of work:** SWEEP.md divides the throughput-mode energy by output elements and, on the batch axis, by
  instances. It also gives rates: GB/s moved (FP16 in and out) or TFLOP/s for matmul.
- **NPU compiler.** The sweep compiles with `NPU_COMPILER_TYPE=PLUGIN`, the compiler the campaign used for its blobs.
  The default on Windows is the compiler inside the NPU driver, here a 2024 release (32.0.100.3053). In a probe on
  2026-11-01 it was slower on large shapes: matmul 1024² took 3.7 vs 2.8 ms per call, and a 16.8M-element add took
  31.9 vs 13.6 ms. It also needed 736 s to compile that add (1.2 s with the plugin compiler). For the one-op ReLU at the
  dataset shape, both give 156–169 µs, the same as the campaign blob (165 µs).

## Intermittent load (`--intermittent`)

The steady-state energy above charges a device only while it runs back to back. A real workload often arrives in
bursts, and devices behave differently around them:
- after sustained GPU work, the board stays in P1 (about 12 W above P8) for 5–7 s (bursts of up to 25 ms did not
  trigger this; see Results);
- after an idle gap, the first NPU calls are slower.

The intermittent metric charges every burst for everything that follows it.

- **Load:** every period P, one burst of N synchronous calls; the device then idles until the next burst. One
  measurement covers at least 3 periods and 30 s. The NPU worker times the bursts itself, so bridge latency does not
  move them. The sampler gets a mark at every burst start and at the end of the last period.
- **Energy per burst:** for each period `[m_k, m_(k+1)]`, `E_k = (P̄_run − P_idle(t_k)) · (m_(k+1) − m_k)`, where
  `P̄_run` is the mean power over the period and `P_idle(t_k)` is the idle power interpolated to the period's midpoint
  between the nearest settled idle windows (`energy_meter.burst_energies`). A measurement's value is the mean over its
  periods. A period holds the burst and everything after it until the next burst, so a GPU power tail is charged to
  the burst that caused it, and a tail that outlasts P carries into the next period. Rails as before: NPU package,
  GPU package + board.
- **Repeats and CIs:** each (device, N, P) condition is measured R = 3 times in shuffled order. Each repeat has its own
  idle windows, with the baseline's wait after GPU bursts (P8 or 10 s). The CI is the t-interval over repeats, so
  baseline error is included.
- **Defaults:** `relu_b1` (the kb_019_relu shape), N ∈ {1, 1000}, P ∈ {1, 5, 20} s. The run also measures steady
  state for the same kernel, in the same session (3 cycles of the sync modes), so each burst result can be compared
  with N × the steady energy per inference.
- **EER per burst:** NPU energy per burst ÷ GPU energy per burst, with the delta-method CI. If the NPU value is not
  significant, its one-sided upper bound gives an upper bound on the ratio.
- **A GPU kept only for this load:** the incremental metric leaves out the GPU's own idle power (about 11 W at P8,
  more under desktop use), because the card idles whether or not it is used. INTERMITTENT.md also shows the ratio with
  the mean idle board power × P added to each GPU burst, the case where the GPU is installed only for this workload. The NPU's own idle power cannot
  be separated from the rest of the package, so it has no such term.

## Results

The timing and energy results come from the October 21 FP16 campaign
(`runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z`). The sweep and intermittent results use synthetic
kernels and are in `runs/kernel-sweep-*` and `runs/kernel-intermittent-*`. The NPU driver was 32.0.100.3053 and the
NVIDIA driver 617.14 throughout. Geometric means are taken over tasks or configurations.

**Timing, all tasks** (`kernel-timing-20261101T000341Z`, 2026-11-01).

- **Coverage:** 40 NPU tasks and 39 GPU tasks. The original `triton_fused_attention` does not compile.
- **NPU gates:** all passed. Every task's NPU outputs were bit-identical to the campaign's (12 of 12 cases).
  `benchmark_app` agreed within 3.5% (sync) and within 2% (async).
- **GPU gates:** 26 tasks failed a GPU gate. Most failures came from the latency states described under Consistency
  gates; the rest came from the pipelined warm-up ramp, since fixed. These failures move the GPU means by no more than
  a few percent.

| Per call (µs) | Geomean | Range |
| --- | ---: | ---: |
| NPU sync | 197 | 164–314 |
| NPU async (4 in flight) | 99 | 93–178 |
| GPU sync (launch + sync) | 19.8 | 18.5–24.1 |
| GPU pipelined | 6.2 | 5.6–13.0 |
| GPU end to end (with copies) | 32.3 | 29.7–85.7 |
| GPU device (NVBench cold) | 6.6 | 5.6–12.4 |

How the NPU compares (geomean ratio, with the range across tasks):

- NPU sync is 10.0× GPU sync (7.0–16.6×).
- NPU async is 16.0× GPU pipelined (7.4–30.6×).
- NPU sync is 6.1× GPU end to end (2.6–10.4×).

**Energy, 8-task subset** (`kernel-energy-20261031T231640Z`, re-analysed with settled baselines;
`ENERGY-reanalysed.md`). 101 of the 200 idle windows were settled, which left n = 5–6 paired runs per task and mode.
The table gives means over tasks: rail power increments in W and energy per inference in µJ.

| Mode | ΔPKG | ΔUncore | ΔCores | ΔGPU board | Headline µJ | Device-only µJ | Host cores µJ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NPU sync | 9.8 | 1.96 | 7.8 | +0.1 | 2014 (SoC) | 411 (uncore) | 1589 |
| NPU async | 15.4 | 2.38 | 12.9 | +0.1 | 1665 (SoC) | 264 (uncore) | 1394 |
| GPU sync | 27.4 | −0.59 | 28.0 | 41.6 | 1411 (system) | 850 (board) | 573 |
| GPU pipelined | 27.4 | −0.31 | 27.7 | 41.5 | 476 (system) | 288 (board) | 190 |

Every per-task headline value is significant, with 95% CIs within ±12%. The GPU board increment during NPU runs is
consistent with zero in every task (the GPU stays at P8), so leaving it out of the NPU headline adds no bias.

EER, GPU : NPU (geometric mean over the 8 tasks [95% CI], range):

| Comparison | Headline energy (NPU SoC vs GPU system) | Device only (NPU uncore vs GPU board) |
| --- | ---: | ---: |
| sync | 1.42 [1.20, 1.67], 1.09–2.00 | 0.47 [0.36, 0.61], 0.32–0.80 |
| throughput | 3.59 [2.67, 4.84], 1.72–5.60 | 0.93 [0.64, 1.33], 0.40–1.81 |

What the measurements show:

- **The NPU itself draws about 2–2.4 W.** Its uncore increment has CIs of about ±0.05 W. The NPU's whole-package
  energy is dominated by the host thread that drives it, 8–13 W on the CPU cores.
- **The GPU wins on total energy at these shapes.** It draws about 41 W on the board plus about 27 W of host cores,
  but it is 10–16× faster. Its EER is 1.42× the NPU's in sync mode and 3.59× in throughput mode.
- **Device-only, the NPU is as efficient or better.** Counting only the NPU's uncore increment and the GPU's board
  increment, the NPU's EER is 2.1× the GPU's in sync mode (ratio 0.47) and about equal in throughput mode (0.93, CI
  includes 1). Uncore excludes DRAM while the GPU board includes its memory, which favours the NPU once data volumes
  grow.
- **Effect of the correction.** The original analysis, with a 3 s settle and every idle window counted, overstated
  NPU ΔUncore by 0.2–0.3 W and understated GPU ΔBoard by 5–7 W. It moved the headline means by at most 7%.

**Size and batching sweep** (`runs/kernel-sweep-20261102T033004Z`, 2026-11-01, `--baseline desktop`, K = 4).

- **Correctness:** every configuration was 100% within 2⁻¹⁰ on both devices.
- **Baseline quality:** the GPU was at P8 in 99% of idle samples, and the idle uncore power had an SD of 0.12 W.
  - The first two configurations ran while the desktop was busier: 10 idle windows had GPU board power of 13–28 W,
    and one had +23 W of package power. This widened the CIs of `add_n4096` and `add_n65536`.
  - The later configurations ran with a quiet desktop: idle board power 10.1–10.6 W.

Throughput modes (NPU async with 4 in flight, GPU pipelined) and the EERs. *thr* compares the throughput modes and
*sync* the sync modes. *dev* compares device only (NPU uncore vs GPU board); otherwise NPU SoC vs GPU system.

| Config | NPU µs | GPU µs | NPU ÷ GPU time | EER thr | EER thr, dev | EER sync | EER sync, dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_n4096 | 93.0 | 5.9 | 15.8 | 4.41 | 0.80 | 1.59 | 0.39 |
| add_n65536 | 100.9 | 5.8 | 17.5 | 4.08 | 0.90 | 1.41 | 0.50 |
| add_n1048576 | 256.7 | 6.7 | 38.1 | 5.22 | 2.00 | 7.73 | 2.63 |
| add_n16777216 | 3673 | 156.1 | 23.5 | 1.35 | 0.88 | 6.81 | 1.67 |
| matmul_64 | 95.8 | 6.5 | 14.8 | 3.73 | 0.77 | 1.49 | 0.35 |
| matmul_256 | 115.1 | 7.3 | 15.7 | 2.99 | 0.80 | 1.77 | 0.51 |
| matmul_1024 | 1942 | 40.9 | 47.5 | 2.40 | 1.25 | 3.09 | 1.38 |
| matmul_2048 | 15388 | 268.8 | 57.2 | 1.49 | 1.30 | 2.36 | 1.32 |
| softmax_32x128 | 95.0 | 6.5 | 14.7 | 4.23 | 0.81 | 1.85 | 0.44 |
| softmax_256x1024 | 174.2 | 7.0 | 24.8 | 5.69 | 1.18 | 2.59 | 0.87 |
| softmax_1024x4096 | 1761 | 40.5 | 43.5 | 2.51 | 0.99 | 8.02 | 2.01 |
| relu_b1 | 92.1 | 6.1 | 15.2 | 4.13 | 0.81 | 1.71 | 0.39 |
| relu_b64 | 88.4 | 5.8 | 15.3 | 4.38 | 0.85 | 1.71 | 0.38 |
| relu_b4096 | 276.5 | 6.7 | 41.4 | 6.61 | 2.43 | 9.10 | 2.66 |
| relu_b65536 | 3614 | 93.4 | 38.7 | 1.96 | 1.35 | 10.86 | 2.59 |
| matmul_b16 | 117.9 | 7.1 | 16.7 | 3.44 | 1.05 | 1.98 | 0.60 |
| softmax_b16 | 110.4 | 6.5 | 17.1 | 4.19 | 0.95 | 2.14 | 0.58 |
| matmul_b256 | 1538 | 9.2 | 167.7 | 6.88 | 2.77 | 8.00 | 2.81 |
| softmax_b256 | 777.3 | 7.5 | 103.2 | 5.71 | 1.87 | 7.87 | 2.39 |
| **Geomean [95% CI]** | | | | **3.62** [2.89, 4.53] | **1.15** [0.95, 1.39] | **3.29** [2.29, 4.72] | **0.97** [0.66, 1.42] |

Per-configuration CIs are in SWEEP.md; all EERs above are significant. The geomean CIs reflect the spread across
configurations, which differ by design, rather than measurement precision.

What the sweep shows:

- **More work per call does not let the NPU catch up in time.**
  - Up to 65K elements, both devices are dispatch-bound: the NPU takes 88–118 µs and the GPU 6–7 µs (15–18×).
  - The GPU absorbs much more work at almost no extra time: ReLU goes from B = 1 to B = 4096 (4096× the work) in
    6.1 → 6.7 µs.
  - The NPU becomes throughput-bound near 1M elements, and the gap widens to 24–57× on the size axis and 39–168×
    on the batch axis.
  - NPU plateaus: streaming adds at 25–27 GB/s, ReLU at 15–19 GB/s, matmul at 1.1 TFLOP/s.
  - GPU at the same sizes: 645 GB/s from DRAM (930 GB/s while the data fits in L2) and 52–64 TFLOP/s.
- **Batched small matmuls suit the NPU poorly.** `matmul_b256` (256 independent 64³ matmuls) reaches 0.09 TFLOP/s,
  12× below one 1024² matmul with the same output size. Batched softmax is fine: `softmax_b256` runs within 11% of
  `softmax_256x1024` per element.
- **The GPU wins on total energy at every size.** Its throughput-mode EER over the NPU is:
  - 3.0–4.4 for small calls;
  - 5.2–6.9 near 1M elements, where the GPU is still launch-bound but the NPU is already bandwidth-bound;
  - 1.35–2.5 at the largest sizes. There the GPU draws 165–257 W (package + board) against the NPU's 7–12 W, and
    that 17–38× power gap nearly offsets its 23–57× speed lead. Both devices' rates have levelled off, so these
    values are close to the asymptotes.
- **On small calls, most of the NPU's energy is the host thread.** Up to 65K elements:
  - the CPU cores account for 79–89% of the NPU's package increment (ΔCores 15–19 W async, 8–13 W sync);
  - the NPU's own uncore increment is 1.8–3.6 W.

  At the largest sizes, calls are fewer and the host share falls (ΔCores 1.2–6.8 W, ΔUncore 3.5–5.7 W).
- **Device only, the two are close.** NPU uncore vs GPU board in throughput mode:
  - 0.77–1.05 for small calls (the NPU is slightly better or equal);
  - 1.9–2.8 near 1M elements and for B = 256 batches (the GPU is better);
  - 0.88–1.35 at the largest sizes.

  Two biases favour the NPU here (see Caveats): uncore misses DRAM, and uncore reads low under CPU load.
- **NPU sync mode is costly at large sizes.** At ≥ 1M elements, sync is 1.2–4× slower than async while the host
  thread spins for the whole call. Its sync EER reaches 6.8–10.9 for the streaming ops.

**Intermittent load** (`runs/kernel-intermittent-20261102T045906Z`, `relu_b1`, `--baseline desktop`, R = 3; the GPU
was at P8 in 100% of idle samples; `INTERMITTENT-reanalysed.md` has the corrected header text).

- **Energy per burst was not resolved, except in one condition: N = 1000, P = 1 s on the NPU.**
  - That condition gave 3.06 J [1.70, 4.43] per burst: 3.06 mJ per inference, 1.3× the steady 2.29 mJ.
  - Every other condition is n.s. The noise, which the desktop baseline contributes over a whole period, has CI
    half-widths of ±1.4–6.6 J at P = 1 s, ±5–36 J at 5 s and ±13–62 J at 20 s.
  - The expected burst energy is about 2 mJ for one call and 1.5–3 J for 1000 calls, so the noise is 10²–10⁴× the
    signal for single calls and comparable to it for 1000 calls.
- **Short GPU bursts do not trigger the GPU's P1 hold.**
  - In every GPU burst condition (1 call, or 1000 sync calls lasting about 25 ms), every P-state sample stayed at
    P8, and board power stayed at or below 12.4 W, apart from one 19.5 W sample.
  - The steady GPU runs in the same session ran at P1 at 52 W.
  - So the 5–7 s tail the metric was designed to charge did not occur at this load. A heavier burst would trigger it.
- **NPU wake-up latency.**
  - A single NPU call takes 1.6–1.7 ms after 1–5 s idle and 12.4 ms after 20 s idle, against 0.17 ms in steady
    state.
  - The first call of a measurement, after at least 5 s of idle, took about 12 ms in 6 of 9 cases (1.6–2.2 ms in
    two, 273 ms once).
  - So the NPU enters a deeper idle state after between 5 and 20 s, and leaving it costs about 10 ms.
  - 1000-call bursts take 160 ms at P = 1 s, about steady speed, but 233–256 ms at P = 5–20 s.
  - The GPU takes 0.4–0.6 ms for one call at any gap, against 0.02 ms steady, and 24–25 ms for 1000 calls.
- **With the GPU's idle board power counted** (a GPU installed only for this load), every condition favours the NPU.
  The GPU : NPU upper bounds are 0.08–0.63, so the NPU is at least 1.6× more efficient. The GPU's 10.6 W × P
  dominates.
- **What would resolve it** (exploratory analysis of this trace, not yet in the code):
  - Take each burst's baseline from the 1 s before it, and integrate over a fixed horizon. This narrows the
    P = 20 s CIs 1.1–7×: NPU N = 1000 goes from ±62 J to ±8.9 J.
  - Use uncore rather than package for the NPU. Its noise is ±0.2–1.0 J, and NPU N = 1000, P = 5 s gives
    0.27 ± 0.20 J of uncore, the same as 1000 steady calls (0.33 J). So the NPU device shows no wake-up overhead
    at that scale.
  - Run longer, with more work per burst, and add a burst heavy enough to wake the GPU.

## Tools considered

| Tool | Used? | Reason |
| --- | --- | --- |
| OpenVINO `benchmark_app` | Yes, NPU timing cross-check | Standard tool; needs the blob shim above |
| NVIDIA NVBench (`cuda-bench`) | Yes, GPU device and batch time | Standard kernel benchmark; Linux wheel, runs in WSL |
| `triton.testing.do_bench` | No | NVBench covers device time for every dialect |
| Zeus, CodeCarbon | No | Rely on the broken NVML energy counter, on Linux RAPL (absent in WSL), or on the discontinued Intel Power Gadget |
| Intel SoC Watch, PCM, VTune; Scaphandre | No | Need admin kernel drivers; no documented NPU rail |
| HWiNFO | No | GUI only; its NPU power sensor covers only Panther Lake and later |
| MLPerf Power | No | Needs an external power analyzer |

## Caveats

- **What PKG misses.** PKG excludes DRAM, the motherboard and power-supply losses for both devices, so wall-plug
  energy is higher. The RAPL DRAM rail reads 0 on this machine, so device-only comparisons leave out the NPU's
  LPDDR5x traffic but include the GPU's GDDR7 (on the board). This favours the NPU on memory-bound shapes: the
  16.8M-element add moves 100 MB per call.
- **Uncore is not a clean NPU meter.** During GPU runs, when the NPU is idle, ΔUncore reads −0.05 to −0.75 W for small
  calls and −1.1 to −1.8 W for large ones, with tight CIs. So uncore has an error of up to a watt or two that depends
  on the load elsewhere in the system. NPU runs put 1–19 W on the cores, so their ΔUncore may be understated in the
  same way, which would favour the NPU in device-only ratios. This matters most for small calls, where the NPU
  adds only 2–3 W.
- **Host side of GPU runs.** It runs in WSL2, so host core power includes virtualization overhead. The GPU also
  drives the display. Its idle board power is about 11 W at P8, about 23 W during the P1 hold that follows any GPU
  work, and up to about 37 W while the desktop is active.
- **Tiny shapes.** Both devices are dominated by fixed dispatch cost, so per-inference energy mostly reflects static
  power and the launch path, not arithmetic efficiency.
- **Driver versions matter and are recorded.** The NPU driver changed from 32.0.100.4404 (campaigns of October 13–22)
  to 32.0.100.3053 by 2026-10-31. Outputs stayed bit-identical, and per-call times matched the earlier
  measurements (relu sync about 166 µs).
- **Bias from correlated noise.** Paired idles cancel linear drift, but not background bursts that coincide with a run.
  K repetitions in shuffled order turn such bursts into variance rather than bias, as long as they are not
  synchronized with the schedule.

## Revision history

| Date | Change |
| --- | --- |
| 2026-10-31 | First version: fixed-window and per-call timing for both devices; `benchmark_app` and NVBench cross-checks; Windows Energy Meter plus NVML instant-power energy protocol with paired idles and t-intervals; validation V1, V3, V5. |
| 2026-10-31 | The sampler waits three intervals before writing its trace, so the trace always covers the last mark (the first validation run failed on this). Added the validation results; defaults confirmed. |
| 2026-11-01 | First timing (40 tasks) and energy (8 tasks) runs; added the Results section. **Energy fix:** the GPU holds P1 for 5–7 s after work, so the 3 s settle left about half the idle windows contaminated. The schedule now waits for P8 (`--settle` is the cap, 20 s), only idle windows with the GPU at P8 are baselines, and NPU runs during which the GPU woke up are dropped. V5 measures the time to P8 against a P8 idle. Added `--reanalyse`, and re-analysed the first energy run with it. A 1-task check (`kb_019_relu`, K = 2) confirmed the P8 wait: 8–12 s after GPU runs. That check also showed spontaneous GPU wake-ups, which the new filters caught; the machine was probably in use. **Timing fixes:** time-based GPU warm-up (3 s per mode), GPU rounds of 20000 calls, and 10% limits for the GPU sync gates. |
| 2026-11-01 | **EER fix:** device ratios are per-task `ē_NPU / ē_GPU` with delta-method CIs, summarized by geometric means; the Results section had divided means over tasks (1.4×, 3.5× → 1.42×, 3.59×). **Baseline fix:** idle power is interpolated in time to each run's midpoint instead of averaging the two neighbours, which cancels drift only for a run midway; runs sat 11–89% of the way. Re-analysed the first energy run (headline means moved < 0.5%). **Counted windows** drain the warm-up's queued work before they start. **New:** `--sweep` (size and batching, `research/sweep_kernels.py`, NPU plugin compiler), `--intermittent` (energy per burst including power tails), `--baseline desktop` (normal desktop use as the baseline, fixed 10 s after GPU work), `--start-when-idle`, a progress bar, and `scripts/run_kernel_sweep_and_intermittent.sh`. |
| 2026-11-01 | Added the results of the first sweep (`kernel-sweep-20261102T033004Z`, 19 configurations) and intermittent (`kernel-intermittent-20261102T045906Z`) runs, both with `--baseline desktop`. Results: the GPU's total-energy EER is 1.35–6.9× at every size, largest near 1M elements and smallest at the largest sizes. Device only, it is 0.77–2.8×. Burst energy was resolved only for NPU N = 1000, P = 1 s, but bursts did not wake the GPU from P8, and the NPU takes about 12 ms to wake after 20 s idle. Caveats now cover the DRAM rail and the uncore bias. **Fix:** INTERMITTENT.md said its idle windows were taken with the GPU at P8 whatever the baseline; it now says baseline idle windows, and the run was re-analysed (numbers unchanged). |
