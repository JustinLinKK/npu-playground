# Empirical scoring of alternative NPU-correctness verdicts on saved Intel NPU 3720 outputs (gpt-6-astra campaigns)

Scope: local, read-only rescoring of physical-NPU and CPU-plugin outputs already saved by two 40-task campaigns. No device or model calls were made. Every number below was computed by scripts in the session scratchpad. They load each accepted cycle's `contract/manifest.json`, regenerate the 12 test cases with `npu_agent.testcases.generate_cases`, and read `target_execution/<case>.npz` (NPU) and `host/<case>.npz` (OpenVINO CPU plugin). The references are `npu_agent.executors.fp16_reference` (FP16-rounded inputs, FP32+ intermediates, output rounded to FP16; the "FP16 ref") and `npu_agent.oracles.execute_oracle` (the "FP32 ref").

- Run A (FP32 contract, Oct 14 2026): `/home/wentao/coding/npu-playground/runs/intel-npu-baseline-openai-20261014T055513137728490Z/evaluation`
- Run B (prepared FP16 dataset, Oct 21 2026): `/home/wentao/coding/npu-playground/runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z/evaluation`
- Scripts (scratchpad, not in repo): [vlib.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/vlib.py) (loader + verdicts), [score_runs.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/score_runs.py), [mutate.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutate.py), [fp_and_caps.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/fp_and_caps.py), [details.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/details.py). Raw outputs are `scores.json`, `mutations.json` and `ident.json` in the same folder.

## Key question 1: How do the candidate verdicts score the saved NPU (and CPU) outputs at task and case level?

### Takeaway
Under the current verdict (V1), the NPU passes 22/39 accepted tasks in Run A and 27/40 in Run B. Verdicts that model FP16 resolution do much better without becoming permissive: V4 (atol = rtol = 2^-10) gives 35/39 and 39/40, and V9 (magnitude-aware Higham bound + 2^-10 fallback) gives 37–38/39 and 39/40. In Run B the only remaining failure under either is `triton_layer_norm`, at 1.35x the V4 bound. FTZ-awareness alone barely helps (Run B: 27 → 27 tasks), because most V1 failures are sub-ulp rounding residue (≤ 2^-10), not flushed subnormals. The ratio verdicts V7a/V7b and the peak-relative verdict V6 even reject the CPU plugin on some tasks, so they are over-strict in places.

### Cited Findings

**Implementation of each verdict.** A task passes only if all 12 cases pass. Arithmetic is float64 on the saved tensors, a = output and e = reference. A NaN/Inf output where e is finite fails every elementwise verdict, every tensor metric and every Higham verdict. The ratio verdicts count it as one "bad" element. [vlib.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/vlib.py)
- V1 current: |a−e| ≤ 1e-5 + 1e-3·|e|. This reproduces `compare_fp16`: Run A 22/39 tasks and 388/468 cases, Run B 27/40 and 409/480. These match `results/summary.json` in both runs ([Run A summary](/home/wentao/coding/npu-playground/runs/intel-npu-baseline-openai-20261014T055513137728490Z/evaluation/results/summary.json), [Run B summary](/home/wentao/coding/npu-playground/runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z/evaluation/results/summary.json)).
- V2a FTZ-aware: V1, or (|e| < 2^-14 and a == 0). V2b: V1, or (|e| < 2^-14 and |a| ≤ atol). The two give identical results on the NPU data.
- V3: atol = 2^-14, rtol = 1e-3. V4: atol = rtol = 2^-10. V5: atol = rtol = 1e-2.
- V6 peak-relative: |a−e| ≤ 8·2^-11·max|e| + 2^-24, with the peak taken per tensor per case.
- V7 ratio ("dual-thousandth"):
  - V7a: err_i = |a−e|/max(|e|, 2^-24). Element i is bad if err_i > 1e-3; the case passes if bad/n ≤ 1e-3.
  - V7b: same, with ratio ≤ 5e-3.
  - V7c (hybrid): err_i = |a−e| if |e| < 1e-3, else |a−e|/|e|. Threshold 1e-3, ratio ≤ 1e-3.
  - V7d: hybrid with ratio ≤ 5e-3.
  - Consequence: with ratio 1e-3, a tensor of n < 1000 elements tolerates 0 bad elements. Ten tasks have n ≥ 1000: layer_norm 2048, transpose 6144, max_pool_2d 1058, vector_add 4096, attention 1024, softmax 4096, moving_average 4096, conv 1024, tiled_matmul 4096 and gamma 3072. With ratio 5e-3, any n ≥ 200 tolerates at least 1 bad element.
- V8 tensor metrics: cosine ≥ 0.999 / 0.9999; relL2 = ‖a−e‖₂/‖e‖₂ ≤ 1e-3 / 1e-2; SNR = 20·log10(‖e‖/‖a−e‖) ≥ 40 / 60 dB.
  - All-zero reference (‖e‖ = 0, for example the `zeros`, `alternating` and `dynamic_range` cases of matmul): pass iff a is finite and max|a| ≤ 2^-10.
  - Cosine with ‖a‖ = 0 but ‖e‖ > 0 fails.
  - SNR ≥ 40 dB is mathematically identical to relL2 ≤ 1e-2, and SNR ≥ 60 dB to relL2 ≤ 1e-3. The computed counts are identical in every table.
- V9 Higham-style: tol_i = c·u·S_i with u = 2^-11. S_i is the oracle `reference()` evaluated in float64 on |FP16-rounded inputs|, with |scale| for scalar_mul, float64 accumulation for reduce_sum and a float64 output for matmul.
  - Ops covered: reduce_sum, kb_reduce(mean), kb_cumsum (forward and reverse), kb_matmul (plain, transpose_a, transpose_b, gemv, batched), matmul, conv2d, moving_average, kb_pool(average), transpose, scalar_mul, vector_add and kb_diagonal. That is 21 of the 40 tasks.
  - Every other op uses the fallback atol = rtol = 2^-10 (identical to V4): activations, softmax, log_softmax, layer_norm, rms_norm, attention, mse, max/min pool and reduce, gamma.
  - V9a: c = 4. V9b: c = 2 + sqrt(n_i), where n_i is the per-element accumulation length (prefix length for cumsum, min(i+1, window) for moving_average, K for matmul, 9 for conv, the kernel size for pools, the axis length for sum/mean, 2 for vector_add, 1 for the rest). V9c: V9a plus the FTZ allowance of V2a.
  - For the "vs FP32 ref" columns V9 is not applicable, so the script falls back to V4.

**Pass-rate table.** Each cell gives tasks passing all 12 cases, then cases passed. Accepted candidates only: Run A has 39 tasks and 468 cases (triton_layer_norm had no accepted candidate); Run B has 40 tasks and 480 cases. [score_runs.py output / scores.json](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/scores.json)

| Verdict | A: NPU vs FP16 ref | B: NPU vs FP16 ref | A: NPU vs FP32 ref | A: CPU vs FP32 ref | B: CPU vs FP16 ref |
|---|---|---|---|---|---|
| V1 current 1e-5/1e-3 | 22/39 · 388 (82.9%) | 27/40 · 409 (85.2%) | 12 · 349 (74.6%) | 39 · 468 | 40 · 480 |
| V2a/V2b FTZ-aware | 23 · 393 (84.0%) | 27 · 413 (86.0%) | 14 · 354 | 39 · 468 | 40 · 480 |
| V3 atol 2^-14, rtol 1e-3 | 25 · 403 (86.1%) | 29 · 427 (89.0%) | 18 · 367 | 39 · 468 | 40 · 480 |
| V4 atol = rtol = 2^-10 | 35 · 448 (95.7%) | 39 · 476 (99.2%) | 32 · 435 (92.9%) | 39 · 468 | 40 · 480 |
| V5 1e-2/1e-2 (KernelBench) | 39 · 468 (100%) | 40 · 480 (100%) | 39 · 468 | 39 · 468 | 40 · 480 |
| V6 peak-relative 8 half-ulps | 38 · 467 (99.8%) | 39 · 479 (99.8%) | 38 · 467 | **37** · 466 | **38** · 478 |
| V7a ratio rel, 1e-3/1e-3 | 19 · 375 (80.1%) | 20 · 394 (82.1%) | 10 · 324 | **30** · 458 | **34** · 473 |
| V7b ratio rel, 1e-3/5e-3 | 21 · 380 (81.2%) | 22 · 400 (83.3%) | 11 · 329 | **30** · 459 | **34** · 474 |
| V7c ratio hybrid, 1e-3/1e-3 | 24 · 392 (83.8%) | 28 · 414 (86.3%) | 14 · 348 | 39 · 468 | 40 · 480 |
| V7d ratio hybrid, 1e-3/5e-3 | 24 · 395 (84.4%) | 28 · 418 (87.1%) | 16 · 353 | 39 · 468 | 40 · 480 |
| V8a/V8b cosine ≥ 0.999 / 0.9999 | 38 · 467 | 39 · 479 | 37 · 466 | 38 · 467 | 40 · 480 |
| V8c relL2 ≤ 1e-3 (= SNR ≥ 60 dB) | 34 · 453 (96.8%) | 39 · 479 | 33 · 452 | 38 · 467 | 40 · 480 |
| V8d relL2 ≤ 1e-2 (= SNR ≥ 40 dB) | 38 · 467 | 39 · 479 | 37 · 466 | 38 · 467 | 40 · 480 |
| V9a Higham c=4 | 37 · 458 (97.9%) | 39 · 476 (99.2%) | n/a | 39 · 468 (V4 fallback) | 40 · 480 |
| V9b Higham c = 2+√n | 37 · 458 | 39 · 476 | n/a | 39 · 468 | 40 · 480 |
| V9c Higham c=4 + FTZ | 38 · 460 (98.3%) | 39 · 476 | n/a | 39 · 468 | 40 · 480 |

**Planned denominator for Run A** (40 tasks / 480 cases): the task counts are unchanged and the unaccepted `triton_layer_norm` adds 12 failed cases. For example, V1 becomes 22/40 tasks and 388/480 = 80.8% of cases, matching `case_pass_rate.fp16_verdict.of_planned` in [Run A summary](/home/wentao/coding/npu-playground/runs/intel-npu-baseline-openai-20261014T055513137728490Z/evaluation/results/summary.json).

**Failing tasks per verdict, Run B NPU** (cases passed in parentheses):
- V1: kb_026_gelu (4), triton_fused_attention (5), kb_029_softplus (5), kb_048_mean_reduction (5), triton_layer_norm (6), kb_036_rms_norm (6), kb_031_elu (6), hip_moving_average (6), hip_gamma_correction (7), kb_027_selu (7), kb_025_swish (9), kb_028_hard_sigmoid (9), triton_fused_softmax (10).
- V2a: identical task set; swish moves to 10, softmax to 11, attention to 6 and rms_norm to 7.
- V3: 11 tasks fail; it newly passes swish and softmax.
- V4 and V9a/b/c: only triton_layer_norm (8/12).
- V5: none.
- V6 and all V8: only cuda_tiled_matmul (11/12, the `cancellation` case).
- V7a: 20 tasks fail. Besides the V1 set, it adds cumsum, reverse_cumsum, leaky_relu, hip_image_convolution, cuda_tiled_matmul, batched_matmul and sigmoid, each at 11/12.

Source: [score_runs.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/score_runs.py)

**Failing tasks per verdict, Run A NPU:**
- V4: kb_029_softplus (4), kb_089_cumsum (7), kb_091_reverse_cumsum (7), hip_image_convolution (10). This matches the 35/39 and 95.7% recorded in [intel-npu-local.md](/home/wentao/coding/npu-playground/docs/intel-npu-local.md).
- V9a/b: softplus (4) and cuda_tiled_transpose (10/12). Transpose fails because of 2 flushed subnormals; V9c, which adds FTZ, passes it.
- V8c (relL2 ≤ 1e-3): softplus, cumsum, reverse_cumsum, mse_loss and cuda_tiled_matmul.
- V6 and V8a/b/d: only cuda_tiled_matmul.

Source: [score_runs.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/score_runs.py)

**Worst V4 scaled error** (error divided by the bound):
- Run B: triton_layer_norm 1.35, then kb_024_log_softmax 0.82, kb_094_mse_loss 0.80, kb_005_scalar_mul 0.79 and kb_036_rms_norm 0.77.
- Run A: kb_091_reverse_cumsum 4.63, hip_image_convolution 4.0, kb_089_cumsum 3.79 and kb_029_softplus 2.98.

Source: [details.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/details.py)

**Element-level breakdown of V1 failures:**
- Run B: 9,774 failing elements. Of these, 1,721 are flushed subnormals (|e| < 2^-14, e ≠ 0, a = 0) and 7,944 have error ≤ 2^-10. Only 109 exceed 2^-10: 103 in triton_layer_norm and 6 in kb_036_rms_norm.
- Run A: 13,142 failing elements. Of these, 1,722 are flushed, 9,777 have error ≤ 2^-10 and 1,643 exceed 2^-10 (softplus 821, cumsum 369, reverse_cumsum 365, conv 81, rms_norm 6, mse 1).

Source: [score_runs.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/score_runs.py)

**Why tensor metrics and V6 fail cuda_tiled_matmul.** In its `cancellation` case the FP16 reference is entirely tiny: max |e| = 4.77e-7 and ‖e‖ = 2.16e-5, with 2048/4096 subnormal elements. The NPU returns all zeros, so relL2 = 1 and cosine is undefined. V1 passes it because atol 1e-5 exceeds every |e|. [details.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/details.py)

**Run B triton_layer_norm, the only V4/V9 failure.** It fails V1 on normal-0..4 and mixed, and V4 on normal-0, normal-2, normal-3 and mixed. The maximum absolute error is 0.0039 at |e| ≤ 7.5, relL2 ≤ 4.6e-4 and there are no flushes in the failing cases. It passes V5, V6 and V8c. [details.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/details.py)

**CPU-plugin outputs.**
- Every CPU output passes V1, V2, V3, V4, V5, V7c, V7d and V9: Run A against the FP32 ref, Run B against the FP16 ref. For Run A, V9 against the FP32 ref is the V4 fallback.
- V6 rejects the CPU on cuda_reduce_sum and kb_003_batched_matmul (11/12 each) in both runs.
- V7a rejects the CPU on 9 tasks in Run A and 6 in Run B (cumsum, reverse_cumsum, reduce_sum, softplus, conv, batched_matmul, plus avg pools and moving_average in A).
- In Run A, V8a and V8c also reject the CPU on kb_003_batched_matmul (11/12).

Source: [scores.json](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/scores.json)

**Run A NPU against the FP32 reference.** Counting FP16 input-rounding error against the kernel lowers V1 from 22 to 12 tasks and V4 from 35 to 32. [scores.json](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/scores.json)

### Inferences
- Verdicts that reject an FP32-correct CPU result on correct kernels are over-strict for this benchmark's near-zero and cancellation cases. That covers V6 (2 tasks) and V7a/V7b (6–9 tasks); their NPU rejections are not reliable evidence of kernel defects. V7c/V7d's absolute-error branch for |e| < 1e-3 fixes the CPU problem, but they still reject the NPU on 12 tasks in Run B.
- In Run B, V1's failures are dominated by sub-ulp FP16 rounding residue (7,944 elements) rather than FTZ (1,721). This explains why the FTZ fix (V2) changes 0 tasks in Run B, and why moving atol and rtol to one FP16 ulp at unit scale (V4) recovers 12 of the 13 failing tasks.
- V9 (Higham) and V4 give the same Run B task result (39/40). In Run A, V9c recovers 3 more tasks than V4 (cumsum, reverse_cumsum, conv). Its error scale follows the running magnitude of the sum (S_i) rather than |e|, which is the right model for cancellation-prone linear ops.

### Gaps
- V9 covers 21/40 tasks. Non-linear ops (softmax, layer_norm, rms_norm, attention, transcendental activations) use the 2^-10 fallback. No condition-number-based bound was implemented for them.
- Run A's float32 NPU outputs are not exactly FP16-representable for 6 tasks (kb_094_mse_loss, cuda_vector_add, kb_091_reverse_cumsum, hip_image_convolution, kb_005_scalar_mul, kb_027_selu). Some part of the executed graph therefore produced non-FP16 values. The saved data does not show which layer or why.
- The all-zero-reference fallback for tensor metrics (max|a| ≤ 2^-10) is my own choice. A different fallback changes the zero-reference cases, which drive the task-level detection of the +1e-3 bias in Key question 2.

## Key question 2: Do the looser verdicts still reject genuinely wrong kernels (false-positive risk)?

### Takeaway
The repository holds no genuinely wrong kernel outputs. All 20 saved "CPU fail" attempts are mathematically correct `triton_layer_norm` graphs that failed only an ill-conditioned FP32-tolerance case. False-positive risk therefore had to be measured with synthetic mutations.
- V4 and V9 detect every generic mutation at task level, on both easy and noisy tasks. They miss only semantic changes of at most 1 FP16 ulp: gelu-tanh, rms_norm epsilon drop (zero-guarded) and layer_norm epsilon-outside-sqrt.
- V5 (1e-2) misses the +1e-3 bias in 12/12 tasks and the 1.01× scaling in 8/12.
- Cosine similarity misses every pure scaling bug.
- The ratio verdicts miss single-element sign flips and even NaN in large tensors.
- The current V1 detects slightly more sub-ulp formula changes than V4, but it also fails the correct NPU gelu, so it cannot separate them on this device either.

### Cited Findings

**(a) Saved "genuinely wrong" attempts.**
- The only host_execution `numerical_mismatch` attempts in any saved Intel run are triton_layer_norm cycles 1–10 in Run A, plus cycles 1–10 in the earlier run `intel-npu-baseline-openai-20261014T004048655536025Z`. Run B has none. Sources: [Run A attempts.csv](/home/wentao/coding/npu-playground/runs/intel-npu-baseline-openai-20261014T055513137728490Z/evaluation/results/attempts.csv), [Run B attempts.csv](/home/wentao/coding/npu-playground/runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z/evaluation/results/attempts.csv), [fp_and_caps.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/fp_and_caps.py)
- All 20 fail the CPU check only on `operation_boundary`, whose rows are 1 + k·1e-6, with max abs error 1.502e-5 against the manifest tolerance rtol 1e-5 / atol 1e-6. The error is identical in all 10 Run A cycles. Source: validation.json under each cycle, e.g. `runs/intel-npu-baseline-openai-20261014T055513137728490Z/evaluation/cases/4c7234f6ed9056db80d18032/cycle-1/compile/invocation-4tl0znhi/validation.json`.
- The model's code is correct mean/variance/epsilon-inside-sqrt layer normalization. Later cycles switch to balanced pairwise reductions in an attempt to fix the FP32 mean error. Source: `source/model.py` of cycles 1, 2 and 10 in the same case directory.
- Their NPU outputs scored against the FP16 ref:
  - V1, V2 and V3: 6/12. V4 and V9: 8/12 (one attempt 7/12). V7a/b: 5/12. V7c/d: 6/12.
  - V5, V6 and all six V8 metrics: 12/12.
  - So V5, V6 and V8 would "pass" all 20 attempts, but these are not wrong kernels, so that is not a false positive.

  Source: [fp_and_caps.py output](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/fp_and_caps.py)
- Under the FP16 contract (Run B), the same task was accepted on the CPU at the first attempt. [Run B attempts.csv](/home/wentao/coding/npu-playground/runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z/evaluation/results/attempts.csv)

**(b) Synthetic mutation design.**
- Mutations are applied to the saved Run B NPU outputs in float64 and re-rounded to FP16. The element index comes from `default_rng(0).permutation` and is fixed per task.
- Generic mutations:
  - sign flip of 1 element;
  - sign flip of ceil(1%) of elements;
  - off-by-one shift along the last axis (`a[...,1:] = a[...,:-1]`);
  - ×1.01 scaling;
  - +1e-3 bias;
  - zeroing the last column (`a[...,-1] = 0`);
  - NaN in 1 element.
- Task-level detection means the unmutated task passed the verdict on all 12 cases and the mutated task fails at least one; the denominator is the number of tasks passing unmutated. Case-level detection counts (unmutated passes, mutation changed the tensor, mutated fails) over eligible cases.
- SET_G is 12 tasks whose unmutated outputs pass all 20 verdicts: vector_add, scalar_mul, transposed_a/b matmul, tiled_transpose, avg_pool_1d, max_pool_1d, log_softmax, tanh, softsign, diagonal_matmul, max_reduction.

Source: [mutate.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutate.py)

**SET_G detection matrix.** Each cell gives tasks detected out of 12, then cases detected out of eligible cases. [mutations.json](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutations.json)

| Verdict | sign 1 elem | sign 1% | shift | ×1.01 | +1e-3 | zero last col | NaN 1 elem |
|---|---|---|---|---|---|---|---|
| V1 / V2a / V2b | 12 · 126/127 | 12 · 132/132 | 12 · 107/107 | 12 · 132/132 | 12 · 110/126 | 12 · 129/129 | 12 · 144/144 |
| V3 | 12 · 126/127 | 12 · 132 | 12 · 107 | 12 · 132 | 12 · 108/126 | 12 · 129 | 12 · 144 |
| V4 | 12 · 123/127 | 12 · 132 | 12 · 107 | 12 · 132 | 12 · 88/126 | 12 · 126/129 | 12 · 144 |
| V5 1e-2 | 12 · 121/127 | 12 · 132 | 12 · 107 | **4** · 5/132 | **0** · 0/126 | 12 · 126/129 | 12 · 144 |
| V6 peak | 12 · 120/127 | 12 · 132 | 12 · 107 | 12 · 132 | 11 · 13/126 | 12 · 121/129 | 12 · 144 |
| V7a | **10** · 106/127 | 12 · 132 | 12 · 107 | 12 · 132 | 12 · 109/126 | 11 · 119/129 | **10** · 120/144 |
| V7b | **1** · 11/127 | 12 · 128/132 | 12 · 107 | 12 · 132 | 12 · 108/126 | 11 · 119/129 | **1** · 12/144 |
| V7c | **10** · 102/127 | 12 · 132 | 12 · 107 | 12 · 132 | 12 · 108/126 | 11 · 117/129 | **10** · 120/144 |
| V7d | **1** · 11/127 | 12 · 127/132 | 12 · 107 | 12 · 132 | 12 · 107/126 | 11 · 117/129 | **1** · 12/144 |
| V8a cos .999 | 11 · 84/127 | 12 · 130/132 | 12 · 102/107 | **0** · 0/132 | 11 · 12/126 | 11 · 112/129 | 12 · 144 |
| V8b cos .9999 | 12 · 105/127 | 12 · 132 | 12 · 106/107 | **0** · 0/132 | 11 · 13/126 | 12 · 116/129 | 12 · 144 |
| V8c relL2 1e-3 (= SNR 60) | 12 · 119/127 | 12 · 132 | 12 · 107 | 12 · 132 | 11 · 69/126 | 12 · 124/129 | 12 · 144 |
| V8d relL2 1e-2 (= SNR 40) | 12 · 110/127 | 12 · 132 | 12 · 107 | 12 · 47/132 | 11 · 13/126 | 12 · 116/129 | 12 · 144 |
| V9a / V9c | 12 · 125/127 | 12 · 132 | 12 · 107 | 12 · 131/132 | 12 · 78/126 | 12 · 129/129 | 12 · 144 |
| V9b | 12 · 125/127 | 12 · 132 | 12 · 107 | 12 · 131/132 | 12 · 77/126 | 12 · 129 | 12 · 144 |

**Ratio verdicts miss whole tensors.** V7a/V7c miss the single sign flip and the NaN in exactly the two SET_G tasks with n ≥ 1000 (vector_add 4096, transpose 6144). V7b/V7d miss them in every SET_G task except max_reduction (n = 36). [mutate.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutate.py)

**SET_H: 10 noisy tasks.** These are gelu, rms_norm, softmax, mean_reduction, moving_average, cumsum, conv, tiled_matmul, layer_norm and softplus. The task denominator is the number of tasks passing unmutated: V1 3, V3 4, V4 9, V5 10, V6 9, V7a/b 0, V7c/d 3, V8 9, V9 9.
- V4 and V9 detect all 7 generic mutations in 9/9 tasks.
- V5 detects ×1.01 in 3/10 tasks (4/109 cases) and +1e-3 in 0/10 (0/116 cases).
- Cosine (V8a/b) detects ×1.01 in 0/9.
- V6 detects +1e-3 in 8/9 tasks but only 27/115 cases.
- V7c detects NaN in 1/3 tasks; V7d detects it in 0/3.

Source: [mutations.json](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutations.json)

**Tensor metrics and V6 rely on zero-reference cases to catch the bias.** Almost all of their task-level +1e-3 detection comes from all-zero-reference cases: fp16(1e-3) = 1.0004e-3 just exceeds the 2^-10 fallback, or V6's 2^-24 bound when the peak is 0. Their case-level rates are only 12–13/126 for V6, V8a/b and V8d. [mutate.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutate.py)

**Semantic (wrong-formula) mutations.** Each wrong formula is evaluated in float64 on the FP16 inputs and rounded to FP16 (the "pure" variant, no device noise), then compared with the FP16 ref. Each cell is the number of failing cases out of 12; 0 means undetected. "Δ" is the max |wrong − e16| after FP16 rounding. [mutate.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutate.py)

| Mutation (task) | Δ | V1/V2 | V3 | V4 | V5 | V6 | V7a/b | V7c/d | V8a/b cos | V8c/f | V8d/e | V9a-c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gelu: tanh approximation (kb_026) | 9.77e-4 | 7 | 6 | **0** | **0** | **0** | 7 | 6 | **0** | **0** | **0** | **0** |
| rms_norm: eps dropped (NaN on zero rows) | NaN | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| rms_norm: eps dropped, zero-guarded | 9.77e-4 | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| softmax: axis 0 instead of −1 | 0.969 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 10 | 12 | 12 | 12 |
| mean: divide by n−1 (kb_048) | 0.125 | 11 | 11 | 11 | 9 | 11 | 11 | 11 | **0** | 11 | 11 | 11 |
| layer_norm: unbiased variance (N−1) | 0.094 | 9 | 9 | 9 | 2 | 2 | 9 | 9 | **0** | 9 | **0** | 9 |
| layer_norm: eps outside sqrt | 9.77e-4 | **0** | **0** | **0** | **0** | **0** | 2 | **0** | **0** | **0** | **0** | **0** |
| leaky_relu: slope 0.02 | 0.16 | 9 | 9 | 9 | 8 | 9 | 9 | 9 | **0** | 9 | 7 | 9 |
| hard_sigmoid: Keras 0.2x+0.5 | 0.067 | 10 | 10 | 10 | 9 | 9 | 10 | 10 | 2 / 8 | 10 | 9 | 10 |
| cumsum: exclusive vs inclusive | 16 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 9 / 10 | 11 | 11 | 11 |
| attention: missing 1/√d scale | 0.615 | 7 (V2: 6) | 6 | 6 | 6 | 6 | 7 | 6 | 6 | 6 | 6 | 6 |
| avg_pool: count_include_pad=False | 1.2 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| swish: forgot ×x | 15 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 11 | 12 | 12 | 12 |
| tanh: 2σ(x)−1 (= tanh(x/2)) | 0.301 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 7 / 8 | 11 | 11 | 11 |
| conv2d: flipped kernel | 14 | 8 | 8 | 8 | 7 | 7 | 8 | 8 | 6 | 7 | 7 | 7 |
| matmul_b: reshape instead of transpose | 4.1e3 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |

The table covers 16 semantic mutations in total.

**Details behind the semantic results.**
- The exact max difference between erf-GELU and tanh-GELU is 4.12e-4 on the test inputs and 4.73e-4 over [−6, 6] (at x ≈ 2.70). That is below one FP16 ulp at magnitude 1–2.
- Dropping rms_norm's epsilon is caught only because the `zeros` case (560 NaN elements) and the `mixed` case (80 NaN elements, from all-zero channel columns) give 0/0.
- Layer_norm eps-outside-sqrt is caught only by V7a/b, on the `mixed` and `alternating` cases.

Source: [mutate.py / verification run](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutate.py)

**Noisy semantic variant.** The mutation is added to the actual NPU output: fp16(a_npu + (wrong − e16)). V1 "detects" gelu-tanh in 9 cases, but the unmutated, correct NPU gelu already fails V1 in 8/12 cases. V1 therefore cannot separate a correct erf-GELU on this NPU from a tanh-GELU; V4 passes both. [mutations.json](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/mutations.json)

### Inferences
- **V4 and V9.** Of the candidates, V4 (2^-10/2^-10) and V9 (Higham c=4 + FTZ with a 2^-10 fallback) keep the highest pass rate on correct device outputs (Run B 39/40) while still detecting every generic corruption and 13/16 semantic bugs. The three misses change outputs by at most one FP16 ulp (9.77e-4). For FP16 inference they are arguably not observable defects, and no tolerance-based verdict on FP16 outputs can reliably detect them.
- **V5 (KernelBench 1e-2).** It is unsafe for FP16 graphs here: it misses absolute offsets up to 1e-3 and 1% scale errors, and catches layer_norm's N−1 variance bug in only 2/12 cases.
- **Tensor metrics.** Cosine similarity at any threshold is blind to scale bugs: mean/(n−1), N−1 variance, leaky slope and ×1.01 all go undetected. Cosine is also meaningless for the two single-element outputs (mse_loss, reduce_sum), where it reduces to a sign check. relL2 ≤ 1e-3 (= SNR 60 dB) is the only tensor metric with reasonable sensitivity, and it still misses the +1e-3 bias in most non-zero cases (69/126 detected).
- **Ratio verdicts (V7).** They trade outlier tolerance for blindness to localized faults: a single NaN or sign flip passes in large tensors unless NaN/Inf are checked separately. They are also too strict on near-zero references, rejecting correct CPU outputs.
- **Fixed input families limit detection.** Detection depends on the 12 fixed families; the zeros and mixed cases alone expose the epsilon drop. A verdict is only as strong as its inputs, so adding inputs whose reference is far from FP16 resolution would matter more than tightening tolerances.

### Gaps
- No genuinely wrong model-generated kernels exist in the saved runs, so false-positive rates on real model errors could not be measured. The synthetic mutations are author-chosen and may not reflect the distribution of real model bugs.
- Device noise and semantic error can stack. In the noisy variant, V4 flagged the zero-guarded epsilon drop in 2 cases, which the pure variant did not. Detection near the tolerance boundary is therefore partly chance.
- Mutation index choice is seeded (single seed 0); there is no repetition across seeds.

## Key question 3: What does the evidence say about gpt-6-astra's capability on these NPU kernel translation tasks?

### Takeaway
On this 40-task Intel benchmark (OpenVINO graph construction, high-level ops), gpt-6-astra produced CPU-equivalent graphs for 39/40 tasks in Run A and 40/40 in Run B. First-attempt success was 37/40 and 39/40, and Run B used 67,090 tokens in total. Every residual physical-NPU discrepancy is consistent with FP16 device arithmetic rather than translation errors. The one substantive failure, Run A's layer_norm, was an FP32-tolerance artifact.

The caveats are substantial:
- one repetition and one model;
- the model never sees NPU feedback;
- the tasks are graph-level, which is far easier than low-level NPU kernels;
- the repository has no gpt-6-astra XDNA2 (MLIR-AIE) results to contrast with.

### Cited Findings

**Configuration.** gpt-6-astra via the OpenAI Responses API at medium reasoning effort, max_tokens 32,768, one direct arm, one repetition, max 10 cycles and 200,000 tokens per task, corpus of 40 tasks (10 classic + 30 KernelBench adaptations), target intel_npu_3720. [evaluation.config.json](/home/wentao/coding/npu-playground/runs/intel-npu-baseline-openai-20261014T055513137728490Z/evaluation.config.json); [intel-npu-local.md](/home/wentao/coding/npu-playground/docs/intel-npu-local.md)

**Run A.**
- 40 attempted, 39 offline passes, 37 first-attempt passes, 51 cycles and LLM calls.
- 113,000 known tokens: 69,132 input, 43,868 output, 7,774 reasoning. Median 1,473 tokens per task.
- Summed model time 880 s; summed active time 1,173 s.
- triton_layer_norm alone used 49,502 tokens (43.8%) over 10 cycles and ended `budget_exhausted`.

Sources: [Run A summary](/home/wentao/coding/npu-playground/runs/intel-npu-baseline-openai-20261014T055513137728490Z/evaluation/results/summary.json), [Run A tasks.csv](/home/wentao/coding/npu-playground/runs/intel-npu-baseline-openai-20261014T055513137728490Z/evaluation/results/tasks.csv)

**Run B.**
- 40/40 offline passes, 39 first-attempt passes, 41 cycles and LLM calls.
- 67,090 tokens: 34,865 input, 32,225 output, 6,070 reasoning. Median 1,523 tokens per task; max 5,131 (kb_042_max_pool_2d).
- Summed model time 572 s; summed active time 711 s.

Sources: [Run B summary](/home/wentao/coding/npu-playground/runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z/evaluation/results/summary.json), [Run B tasks.csv](/home/wentao/coding/npu-playground/runs/intel-npu-fp16-baseline-openai-20261021T061346915998383Z/evaluation/results/tasks.csv)

**Failure categories** (both runs; all repaired within 1 extra cycle except layer_norm):
- `graph_source` / graph_construction, which are API-usage errors:
  - kb_042_max_pool_2d cycle 1, in both runs: `TypeError: max_pool() got an unexpected keyword argument 'kernel'`.
  - kb_012_diagonal_matmul cycle 1, Run A: `TypeError: data type 'i64' not understood` (passing an OpenVINO type string to `ops.constant`).
- `numerical_mismatch`: only Run A triton_layer_norm, 10 cycles, all on the operation_boundary FP32-tolerance case described in Key question 2.

Source: [fp_and_caps.py output](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/fp_and_caps.py)

**NPU results are observational.** Acceptance and repair feedback use `offline-validated` (target compilation + OpenVINO CPU correctness), and "The repair prompt never receives `target_execution`" ([intel-npu-local.md](/home/wentao/coding/npu-playground/docs/intel-npu-local.md)). This was confirmed in the saved prompts: the cycle-2 repair prompts of triton_layer_norm, kb_042 and kb_012 contain 0 occurrences of "target_execution", "physical_npu" or "fp16". Their evaluator feedback contains only `host_execution` (and `graph_construction`) stages. Source: `calls.code.prompt` in `cases/<id>/report.json` of Run A.

**Physical-NPU verification under the current V1.**
- Run A: 22/39 accepted tasks.
- Run B: 27/40.
- Run B at V4: 39/40. Run B at V9: 39/40.

Source: [scores.json](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/scores.json)

**The model's formulation changed NPU precision between runs.** 23/39 tasks have bit-identical NPU outputs in A and B (after casting A's f32 outputs to f16); 16 differ. [ident.json](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/ident.json)
- Improvements from A to B came from different generated code:
  - kb_089_cumsum: A used a chain of 65 sequential adds (split, running add, concat); B used a matmul with an upper-triangular ones matrix. V1 went from 6/12 to 12/12 cases.
  - kb_029_softplus: A used `ops.softplus(x)` directly, with NPU errors up to 6.8e-3; B used max(x,0) + softplus(−|x|), with errors ≤ 9.8e-4.
  - hip_image_convolution: A used 9 sliced multiply-adds; B used im2col + matmul. V1 went from 5/12 to 12/12 cases.

  The CPU accepted both versions, so the model could not have learned this from feedback. Source: [scratchpad detail run](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/details.py); `source/model.py` of each accepted cycle.
- **Three case-level regressions under V1** (all still pass V4 in both runs), each from different generated code:
  - kb_027_selu, 10 → 7 cases. B uses a polynomial expm1 on [−1, 0] with explicit f32 converts; A used a polynomial on [−0.5, 0]. The failing B elements have error 9.77e-4 at |e| ≈ 0.7–1.2, for example e = −1.22, a = −1.219.
  - kb_028_hard_sigmoid, 11 → 9 cases. B computes (clamp(x, −3, 3) + 3)/6; A computed min(1, max(0, x/6 + 0.5)). In B's cancellation and dynamic_range cases the NPU returns 1.22e-4 (2^-13) where e = 0. That is FP16 residue from the device's lowering, not a formula error.
  - kb_048_mean_reduction, 7 → 5 cases. B computes reduce_sum/17; A used reduce_mean. The differences are 1–2 FP16 ulps, for example e = 0.9263, a = 0.9272.

  Source: [details.py](/tmp/claude-1000/-home-wentao-coding-npu-playground/486254ef-5265-4254-84e4-05b226bcb0c8/scratchpad/details.py)

**XDNA2 / AMD contrast.**
- **No gpt-6-astra XDNA2 campaign exists in the repository.**
  - `runs/xdna2-baseline-20261001` (Qwen route) is referenced in docs but absent from `runs/`. Per the doc, its 40 tasks were all "NOT RUN" because the calibration requests stalled.
  - An Oct 2 OpenRouter Qwen calibration passed add, transpose and reduce at the first attempt; GEMV was cancelled at the 900 s response deadline.
  - The doc states "No kernel-translation accuracy was measured."
  - Even when run, XDNA2 acceptance means compilation + host execution + dataflow simulation, not physical execution.

  Source: [xdna2-baseline.md](/home/wentao/coding/npu-playground/docs/xdna2-baseline.md)
- **Historical case study v2** (model gpt-5.6-terra, prompts including backend examples, 10 classic kernels × 3 reps, offline validation only):

  | Target, baseline arm | Solved | Cycles | Tokens | Active minutes |
  |---|---|---|---|---|
  | AMD XDNA2 | 27/30 | 89 | 1,758,550 | 140.9 |
  | Intel NPU 4000 | 30/30 | 34 | 376,724 | 14.0 |

  AMD cost about 4.7× the tokens and 10× the active time of Intel for the same kernels. [case-study-v2-recovered results.md](/home/wentao/coding/npu-playground/runs/case-study-v2-recovered/results.md); [case-study-v4.md](/home/wentao/coding/npu-playground/docs/case-study-v4.md)
- **Case study v3** (qwen3-coder-30b-a3b-instruct, interrupted): AMD XDNA2 baseline 1/30 and baseline_minimal 0/30; Intel NPU 4000 baseline 8/30 and baseline_minimal 6/30. [case-study-v3-openrouter-recovered results.md](/home/wentao/coding/npu-playground/runs/case-study-v3-openrouter-recovered/results.md)

### Inferences
- **The graphs are correct translations.** On this benchmark gpt-6-astra reliably maps GPU kernels to semantically correct OpenVINO graphs, and the remaining physical-NPU disagreements are device-precision effects (FTZ, FP16 rounding, transcendental approximation, accumulation order).
- **Choice of formulation matters for NPU accuracy.** The model's choice between mathematically equivalent formulations (sequential adds vs matmul, direct softplus vs stable split) changes NPU accuracy by up to roughly 5× the 2^-10 bound. Because NPU results never reach the model, this precision is not something it optimized; it is effectively uncontrolled variation between runs.
- **These are among the easiest NPU translation tasks.** Most tasks map onto one or a few opset nodes (reduce_mean, softplus, matmul, max_pool), and the observed failures were API keyword and dtype mistakes. Both the historical v2 numbers (AMD at about 4.7× the tokens) and v3 (Qwen 1/30 on AMD vs 8/30 on Intel) indicate that low-level MLIR-AIE programming is far harder. The Intel results should not be read as evidence of capability on low-level NPU kernels.
- **The headline "NPU-verified" rate is a verdict choice.** It ranges from 27/40 to 39/40 on identical outputs depending on the verdict. Only verdicts validated by the mutation study (V4, V9) should be used to make a capability claim.

### Gaps
- There are no gpt-6-astra XDNA2 / MLIR-AIE results, so the low-level comparison relies on other models (gpt-5.6-terra, Qwen) and older guided prompts.
- Each run has a single repetition, so run-to-run variance (e.g., the 16/39 tasks whose NPU outputs changed between runs) cannot be separated from the contract change (FP32 → FP16 manifests).
- Dollar cost is not recorded (`known_evaluation_cost_usd: null`), and token counts exclude calibration and preflight.
- The CPU-equivalence acceptance and the 12 fixed input families cannot detect sub-ulp formula differences (Key question 2). "Semantically correct" means correct within these inputs' resolution, not proven equivalent; the corpus records `source_equivalence=not_established` per [case-study-v4.md](/home/wentao/coding/npu-playground/docs/case-study-v4.md).
