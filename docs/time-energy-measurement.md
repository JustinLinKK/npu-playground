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

Results go to `RUN_ROOT/kernel-{validate,timing,energy}-<UTC>/`.

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
```

- **Leave the machine idle during `--validate` and energy runs.** Background load is the main source of energy noise.
- **Locks:** the orchestrator holds `.npu-agent/npu.lock` and `.npu-agent/energy.lock`, so campaign NPU executions
  wait for it.
- **Setup:** the GPU venv also needs `uv pip install "cuda-bench[cu13]"`. See
  `research/time_original_gpu_kernels.py` for the rest.
- **Re-analysis:** `--reanalyse` recomputes an energy run from its `trace.csv` and `windows.json` with the current
  estimator. It writes `ENERGY-reanalysed.md` and `summary-reanalysed.json`, and leaves the run's own report untouched.

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
completion, because the final synchronize or `wait_all` happens inside the measured time. The clock is read only
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
- **Paired difference** for run k: `ΔP_k = P(R_k) − ½·(P(I_before) + P(I_after))`. Averaging the two neighbouring
  idle windows cancels any linear drift exactly.
- **Only settled idle windows are baselines.** `I_before` and `I_after` are the nearest idle windows in which the GPU
  sat at P8 for at least 90% of the samples (`energy_meter.gpu_settled`). Others are skipped, and ENERGY.md states how
  many were used. A run with no settled idle window on one side is dropped, so n can fall below K.
- **NPU runs with a GPU wake-up are dropped.** The GPU should stay at P8 throughout an NPU run. If it leaves P8,
  another program (the desktop, a browser) used it, and that activity also disturbs the package rails. The same 90%
  rule applies, and ENERGY.md reports the count: 1 of 96 NPU runs in the first energy run.
- **Energy per inference** (µJ): `e_k = ΔP_k · t_k`, where `t_k = T_k / N_k` comes from the same window.
- **Gross energy** `P(R_k) · t_k` includes idle power as well. It is dominated by the ~32 W package idle, so it is
  reported but not used as the headline.

What counts for each device:

| Device | Headline (`HEADLINE` in the code) | Breakdown |
| --- | --- | --- |
| NPU | `soc` = ΔPKG · t | `uncore` (NPU and fabric), `host_cores` (PP0, the host thread driving the NPU) |
| GPU | `system` = (ΔPKG + ΔGPU board) · t | `host_cores`, `gpu_board` |

Efficiency is reported as inferences per joule (`1 / e`), shown only when `e` is significant.

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

## Results

These come from the October 21 FP16 campaign (`runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z`). The
NPU driver was 32.0.100.3053 and the NVIDIA driver 617.14. Geometric means are taken over tasks.

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
| NPU sync | 9.8 | 1.95 | 7.8 | ≈0 | 2014 (SoC) | 411 (uncore) | 1588 |
| NPU async | 15.4 | 2.38 | 13.0 | ≈0 | 1672 (SoC) | 264 (uncore) | 1401 |
| GPU sync | 27.4 | −0.59 | 28.0 | 41.5 | 1409 (system) | 848 (board) | 573 |
| GPU pipelined | 27.3 | −0.31 | 27.6 | 41.6 | 477 (system) | 289 (board) | 190 |

Every per-task headline value is significant, with 95% CIs within ±12%.

What the measurements show:

- **The NPU itself draws about 2–2.4 W.** Its uncore increment has CIs of about ±0.05 W. The NPU's whole-package
  energy is dominated by the host thread that drives it, 8–13 W on the CPU cores.
- **The GPU wins on total energy at these shapes.** It draws about 41 W on the board plus about 27 W of host cores,
  but it is 10–16× faster. Per inference it uses 1.4× less energy in sync mode and 3.5× less in pipelined mode.
- **Device-only, they are comparable.** Counting only the NPU's uncore increment and the GPU's board increment, NPU
  async and GPU pipelined are about equal (264 vs 289 µJ), and NPU sync uses about half the energy of GPU sync
  (411 vs 848 µJ).
- **Effect of the correction.** The original analysis, with a 3 s settle and every idle window counted, overstated
  NPU ΔUncore by 0.2–0.3 W and understated GPU ΔBoard by 5–7 W. It moved the headline means by at most 7%.

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
  energy is higher.
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
