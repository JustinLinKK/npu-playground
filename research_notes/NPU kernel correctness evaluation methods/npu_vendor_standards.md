# NPU vendor standards for operator / kernel / compiled-model numerical accuracy

Scope: how NPU vendors themselves define "numerically correct" for operators, kernels and compiled
models — exact metrics, per-dtype thresholds, comparison granularity, near-zero / subnormal handling,
and the tools that implement it. State as of 2026-10-22. Verification status is marked per finding:
**[verified-source]** = read directly from the primary file/page; **[search-summary]** = taken from a
search engine's summary of a primary page that I could not fetch in full (treat as weaker).

---

## Q1. Huawei Ascend (CANN): operator precision standards for FP16/BF16/FP32 (双千/双万, MERE/MARE, msprobe, msit)

### Takeaway

Ascend has the most formalized operator-acceptance regime of any NPU vendor: pass/fail is decided by
**tensor-level statistics plus a ratio-of-failing-elements rule**, never by a bare elementwise
`allclose`. Elements are first partitioned into "normal values" (judged by *relative* error against a
per-dtype `rtol`) and "small values" (judged by *absolute* error against a per-dtype `atol`), and the
verdict is a failing-element *ratio* (双百/双千/双万 = 1%, 0.1%, 0.01%) or a ratio of the NPU's error
count to a same-precision CPU/GPU error count. ULP-based criteria are used for matmul-family ops.

### Cited Findings — msprobe (official Ascend tool, repo `ascend/mstt`)

Tool: `msprobe` offline API precision pre-check (`run_ut`) + `api_precision_compare`. Primary doc read
at repo commit `e416ad2` (2025-09-07).

- **Verdict chain for an API/tensor** (in order): cosine similarity > 0.99 → if ≤ 0.99 mark `error`;
  else max absolute error < 0.001 → if < 0.001 mark `pass`; else fall through to the 双百/双千/双万
  ratio indicators — **[verified-source]** — [msprobe 07.accuracy_checker_PyTorch.md §4.1](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md)
- **双百 / 双千 / 双万 (dual-hundredth / dual-thousandth / dual-ten-thousandth)** definitions, verbatim
  translation: "双千指标 … compare the tensor's elements one by one against the golden data; the
  proportion of elements whose relative error is less than one thousandth. **The pass criterion is that
  the proportion of elements whose relative error is greater than one thousandth is less than one
  thousandth.**" Same construction for 双百 (1% / 1%) and 双万 (0.01% / 0.01%) — **[verified-source]** — [same doc §4](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md)
- **Per-dtype application of those ratios**: for **float16 and bfloat16** — 双百 fail → `error`; 双百
  pass but 双千 fail → `warning`; both pass → `pass`. For **float32 / float64** — 双千 fail → `error`;
  双千 pass but 双万 fail → `warning`; both pass → `pass` — **[verified-source]** — [same doc §4.1](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md)
- **Small-value domain (小值域) thresholds** (doc §4.2): value is classified "small" when
  |golden| ≤ `2**-20` (fp32) or `2**-10` (fp16, bf16); inside that domain the **absolute**-error
  tolerance is `2**-30` (fp32) and `2**-16` (fp16, bf16) — **[verified-source]** — [same doc §4.2](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md)
- Same constants in code: `StandardConfig._small_value = {fp16: 2**-10, bf16: 2**-10, fp32: 2**-20,
  default: 2**-20}`; `_threshold_small_value_atol = {fp16: 2**-16, bf16: 1e-16, fp32: 2**-30}`;
  `_benchmark_small_value_atol = {fp16: 1e-16, bf16: 1e-16, fp32: 2**-30}` — **[verified-source]** — [standard_config.py](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/precision_standard/standard_config.py)
- **Per-dtype relative tolerance** used for "normal" elements: `_rtol = {fp16: 2**-10 (9.766e-4),
  bf16: 2**-8 (3.906e-3), fp32: 2**-20 (9.537e-7), default: 2**-20}` — **[verified-source]** — [standard_config.py](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/precision_standard/standard_config.py)
- **Accumulative-error standard bounds** (for ops with accumulation): `{fp16: 2**-8, bf16: 2**-7,
  fp32: 2**-11}`; minimum-error floor `_minmum_err = {fp16: 2**-11, bf16: 2**-8, fp32: 2**-14}` — **[verified-source]** — [standard_config.py](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/precision_standard/standard_config.py)
- **Error partitioning algorithm**: `small_value_mask = (|bench| ≤ small_value) & both_finite`;
  relative error is computed as `abs_err / (|bench| + eps)` with small-value and inf/nan positions
  *masked out*; `check_small_value()` counts `abs_err > small_value_atol` among small values;
  `check_norm_value()` counts `rel_err > rtol` among normal values. Both return a **ratio**, not a
  boolean — **[verified-source]** — [algorithm.py](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/compare/algorithm.py)
- **Per-operator, per-dtype thresholds** are table-driven. e.g. for `mul` / `add` / `div` / `sub`
  (absolute-threshold standard): fp32 `rtol 1e-6, small_value 1e-6, small_value_atol 1e-9`;
  fp16 `rtol 1e-3, small_value 1e-3, small_value_atol 1e-5`; bf16 `rtol 4e-3, small_value 1e-3,
  small_value_atol 1e-5` — **[verified-source]** — [api_precision_threshold.yaml](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/compare/api_precision_threshold.yaml)
- **"Precision standard by op category" table exists** as `api_precision_standard.yaml`, assigning each
  API one of five comparison algorithms: `AbsoluteThreshStandard` (30 entries: mul/add/div/sub/
  leaky_relu/prelu/reciprocal/rsqrt/square…), `BinaryCompareStandard` (62 entries: abs/ceil/floor/
  gather/logical_*/max_pool3d… — **bit-exact required**), `ULPStandard` (17 entries: matmul/mm/bmm/
  addmm/baddbmm/linear/bilinear…), `ThousandthStandard` (**only `conv1d`, `conv2d`**),
  `AccumulativeErrorStandard` — **[verified-source]** — [api_precision_standard.yaml](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/compare/api_precision_standard.yaml)
- **ULP criterion** (matmul family): ULP error threshold is **1** when the device dtype is float16 or
  bfloat16 and **32** for float32. Pass if *either* "NPU proportion of elements with ULP error above
  threshold < 0.001" (fp16/bf16) or "NPU proportion < GPU proportion". For fp32, pass if *any* of:
  mean ULP error < 64, proportion < 0.05, or NPU proportion < GPU proportion — **[verified-source]** — [07.accuracy_checker_PyTorch.md §5](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md); code constants `_fp32_mean_ulp_err_threshold = 64`, `_fp32_ulp_err_proportion = 0.05`, `_fp16_ulp_err_proportion = 0.001`, `ulp_err_proportion_ratio = 1` — [standard_config.py](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/precision_standard/standard_config.py)
- **ULP computation**: `eb = where(|bench| == 0, 0, floor(log2(|bench|)))`, clamped by `min_eb`
  (fp16 −14, bf16 −126, fp32 −126); `ulp_err = |(device − bench) * 2**(−eb + exponent_num)|` with
  `exponent_num` = mantissa bits (fp16 10, bf16 7, fp32 23) — **[verified-source]** — [algorithm.py / compare_utils.py](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/compare/compare_utils.py)
- **"标杆比对法" (benchmark/baseline comparison method)** — NPU is not judged against an absolute bound
  but against a **GPU run of the same API**, as ratios: small-value error ratio ≤ 2 → pass; RMSE ratio
  ≤ 2 → pass; **max relative error ratio ≤ 10** → pass; **mean relative error ratio ≤ 2** → pass; error
  balance (EB) ratio ≤ 2 → pass (EB excluded from the aggregate verdict) — **[verified-source]** — [07.accuracy_checker_PyTorch.md §5](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md); same constants in `_small_value_threshold/_rmse_threshold/_max_rel_err_threshold/_mean_rel_err_threshold/_eb_threshold` (`error_threshold` 2/2/10/2/2) — [standard_config.py](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/precision_standard/standard_config.py)
- **"绝对阈值法" (absolute-threshold method)** verdict in `api_precision_compare`: inf/nan error rate
  **must equal 0**; relative-error error rate **must equal 0**; absolute-error (small-value) error rate
  **must equal 0** — i.e. for this op class no failing element is tolerated once the small-value /
  normal-value split has been applied — **[verified-source]** — [same doc §5](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md)
- **conv1d/conv2d**: 双千 indicator (fraction of elements with rel err < 1e-3) must be **> 0.999** — **[verified-source]** — [same doc §5](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md)
- Metrics msprobe reports per tensor (the full column set): cosine similarity, max absolute error,
  双百/双千/双万, binary-consistency error rate, error balance (EB), RMSE, small-value-domain error
  proportion, max relative error, mean relative error, inf/nan error rate, relative-error error rate,
  absolute-error error rate, max/mean ULP error, proportion of ULP error above threshold — **[verified-source]** — [same doc §4](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md)
- An older code path in the same repo still carries the pre-2024 constants
  `precision_configs = {fp16: small_value 1e-3 / small_value_atol 1e-5, bf16: 1e-3 / 1e-5,
  fp32: 1e-6 / 1e-9}` — so **two generations of small-value thresholds coexist** (`1e-3/1e-5` in the
  per-op YAML and legacy table vs `2**-10 / 2**-16` in `StandardConfig`) — **[verified-source]** — [compare_utils.py](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/pytorch/api_accuracy_checker/compare/compare_utils.py)

### Cited Findings — CANN "ecosystem operator precision standard" as applied to generated kernels

A 2026 benchmark paper states it adopts Huawei's **ecosystem-operator precision standard** verbatim and
documents the numbers (useful because the standard itself is spread across CANN repos):

- **Metric**: elementwise relative error `e_i = |ŷ_i − y_i| / (|y_i| + 1e-7)`; the tensor passes the
  first screen iff `mean_i e_i < τ_d` **and** `max_i e_i < 10·τ_d` ("mean RE" and "max RE" — the
  MERE/MARE pair) — **[verified-source]** — [CANN Bench §3.1](https://arxiv.org/html/2607.20518v1)
- **Per-dtype τ_d (mean-RE bound) / 10·τ_d (max-RE bound)**: FP16 `2**-10 = 9.77e-4` / `9.77e-3`;
  BF16 `2**-7 = 7.81e-3` / `7.81e-2`; FP32 `2**-13 = 1.22e-4` / `1.22e-3`; HiF32 `2**-11 = 4.88e-4` /
  `4.88e-3`; FP8 E4M3 `2**-3 = 0.125` / `1.25`; FP8 E5M2 `2**-2 = 0.25` / `2.50`. Integer dtypes:
  **exact equality by default** — **[verified-source]** — [CANN Bench Table 4](https://arxiv.org/html/2607.20518v1)
- **Three-region verdict** (this is the near-zero escape hatch): after a first-screen failure the
  checker classifies failing positions as *normal*, *small-value*, or *cancellation*. Only failures
  confined to the two unstable regions are eligible for a fallback; for those the test switches from
  relative error to an **absolute-error count ratio against a same-precision CPU run**:
  `ErrorCount_NPU / max(ErrorCount_CPU, 1) ≤ 2` — **[verified-source]** — [CANN Bench Eq. 3](https://arxiv.org/html/2607.20518v1)
- **Small-value region thresholds** (τ_sv = region boundary, ε_sv = absolute-error tolerance inside it):
  FP16 `2**-11` / `2**-16`; BF16 `2**-8` / `2**-16`; FP32 `2**-14` / `2**-30`; HiF32 `2**-12` / `2**-28`;
  FP8E4M3 `2**-4` / `2**-6`; FP8E5M2 `2**-3` / `2**-5`. A position is small-value when
  `|y_trunc| < τ_sv`, where `y_trunc` is the FP64 golden cast *down to the device dtype and back*
  (the best value representable at device precision) — **[verified-source]** — [CANN Bench Appendix A Table 6](https://arxiv.org/html/2607.20518v1)
- **Cancellation region thresholds**: a position is cancellation-affected when `|ŷ| < τ_cz` and
  `τ_sv ≤ |y| < τ_cb`, with FP16 `τ_cb = τ_cz = 2**-5`; BF16 `2**-3`; FP32 `2**-8`; HiF32 `2**-8`;
  FP8E4M3 `2**-1`; FP8E5M2 `2**0`. Rationale quoted: "A correct kernel can then emit |ŷ_i| ≈ 0 whose
  elementwise relative error against the FP64 Golden is arbitrarily large — not because the arithmetic
  is wrong but because the result's dynamic range no longer matches the reference at the device dtype" — **[verified-source]** — [CANN Bench Appendix A Table 7](https://arxiv.org/html/2607.20518v1)
- **Reference construction**: golden runs on CPU with inputs upcast to **FP64**, and "before e_i is
  computed, the Golden output is cast to the kernel output's dtype, so that the threshold τ_d is
  interpreted at the device dtype rather than at the reference dtype" — **[verified-source]** — [CANN Bench §3.1](https://arxiv.org/html/2607.20518v1)
- **Special values**: NaN positions must match; matching ±inf with the same sign are *excluded* from the
  mean/max-RE statistics; an infinity sign mismatch is a hard failure; one-sided ±inf is replaced by the
  max finite value of the output dtype and then re-gated — **[verified-source]** — [CANN Bench §3.1](https://arxiv.org/html/2607.20518v1)
- **Per-operator relaxation is legitimate and bounded**: an operator may override τ_d per dtype, with a
  recommended cap of **10× the default**, and the effective threshold is recorded so "a relaxed pass
  remains distinguishable from a strict one" — **[verified-source]** — [CANN Bench §3.1](https://arxiv.org/html/2607.20518v1)
- The same paper notes the plan to "migrate from the ecosystem-operator standard now in use to Huawei's
  **commercial operator precision standard** as the latter stabilizes" — i.e. Huawei maintains two
  tiers of operator precision standard — **[verified-source]** — [CANN Bench §3.1](https://arxiv.org/html/2607.20518v1)

### Cited Findings — Ascend model-level / quantization tools

- Ascend **AMCT** "Accuracy-based Automatic Quantization" uses **cosine similarity** between the
  fake-quantized and original outputs and rolls back the layer with the lowest similarity — **[search-summary]** — [hiascend AMCT doc](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/amct/atlasamct_16_0254.html)
- Community/engineering practice treats **cosine similarity > 0.99** as the accept line for
  `msit debug compare` ONNX-vs-OM tensor comparison, and the toolchain reports cosine similarity,
  relative error, max error, relative Euclidean distance and KL divergence; the 0.99 figure appears in
  developer articles rather than in a vendor spec page I could fetch — **[search-summary]** — [Aliyun developer article on offline-inference accuracy analysis](https://developer.aliyun.com/article/1668160); [MindStudio comparison-algorithm list](https://zhuanlan.zhihu.com/p/589651588)

### Inferences

- Ascend's design answer to exactly the problem in our evaluation (FTZ / cancellation / 1-ulp noise) is
  **not a looser global tolerance** but (a) split the tensor into normal / small-value / cancellation
  populations, (b) use absolute error in the unstable populations, and (c) allow a bounded *ratio* of
  failures rather than requiring every element to pass.
- The `rtol` values are deliberately "1 ulp of the dtype" (fp16 `2**-10`, bf16 `2**-8`): a single
  correctly-rounded-differently element is *inside* the tolerance by construction, so ~1-ulp rounding
  differences cannot fail an Ascend-style gate.
- The msprobe chain is strictly cheaper-first: a tensor with max |abs err| < 1e-3 passes immediately
  regardless of relative error — a direct, cheap guard against near-zero reference blow-ups.

### Gaps

- I could not fetch an official hiascend page that states the 双千/双万 criteria in Huawei's own
  commercial-documentation wording (the search engine returned only analyst reports and blogs); the
  authoritative text I *did* verify is Huawei's own `ascend/mstt` repo doc, which is a vendor-owned
  primary source but not the commercial CANN doc portal.
- The `cann-ops` / `ops-math` / `ops-nn` repo file that literally defines the "ecosystem operator
  precision standard" (the source the CANN Bench numbers are drawn from) was not located; mirrors on
  gitcode were the only search hits and I did not confirm the document path.
- No primary source found for default pass thresholds inside `msit compare` itself (the 0.99 cosine
  figure is community practice, and `op_test_frame` / `ascend-op-test` internals were not located).

---

## Q2. Qualcomm (QNN / Qualcomm AI Hub / SNPE / QAIRT accuracy debugger)

### Takeaway

Qualcomm's on-device acceptance metric is **PSNR in dB at tensor level**, with **20 dB as the
machine-enforced scorecard gate** and ">30 dB typically considered good" as the human guidance; there
is no elementwise tolerance anywhere in the public flow. Task-metric regression (≤ 1 percentage point)
is the second gate when a dataset evaluator exists.

### Cited Findings

- `compute_psnr(output_a, output_b, data_range=None, eps=1e-5, eps2=1e-10)` returns
  `20 * log10((data_range_est + eps) / (rmse + eps2))` where `rmse = sqrt(mean((a-b)^2))` and
  `data_range_est = max(|b|)` when not supplied (b = reference) — **[verified-source]** — [qai_hub_models/utils/compare.py](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/utils/compare.py)
- Reported metric set for on-device vs local-CPU-torch comparison: `psnr` (default), `mse`, `mae`,
  `max_abs_diff`, `top-k` match rate, plus `compute_relative_error(expected, actual) =
  |expected − actual| / (|expected| + 1e-20)` as a diagnostic array — **[verified-source]** — [compare.py](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/utils/compare.py)
- Guidance string shipped in the tool: `PSNR_EXPLAINER = "Peak Signal-to-Noise Ratio (PSNR). >30 dB is
  typically considered good."` — **[verified-source]** — [compare.py](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/utils/compare.py)
- Scorecard gate: `--psnr-threshold` **default 20.0** and `--accuracy-threshold` **default 1.0**
  ("Absolute percentage threshold to deem something being low accuracy"); the numerics check passes only
  if `min(PSNR over all outputs) > psnr_threshold`, else the row is labelled "Low PSNR"; a task-metric
  regression greater than 1.0 point vs torch is "On-Target Numerical Failure" (or "Off-Target" for the
  simulator) — **[verified-source]** — [aggregate_scorecard_results.py](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/scripts/aggregate_scorecard_results.py)
- Qualcomm's own validation playbook states the rule as: "**Every output ≥ 20 dB**, matching the
  scorecard's own threshold (`aggregate_scorecard_results.py --psnr-threshold`, default `20.0`). Below
  that, the graph is computing something different on device." It also defines an explicit exemption
  list, `outputs_to_skip_validation`, "precisely for outputs whose PSNR is meaningless (many
  low-confidence values filtered out in postprocessing)" — **[verified-source]** — [plugin/skills/validate-on-device/SKILL.md](https://github.com/quic/ai-hub-models/blob/main/plugin/skills/validate-on-device/SKILL.md)
- `compare_psnr(output_a, output_b, psnr_threshold)` simply asserts `psnr > psnr_threshold`; the unit
  test calls it with `psnr_threshold=30` — **[verified-source]** — [compare.py](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/utils/compare.py), [test_compare.py](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/test/test_utils/test_compare.py)
- AI Hub docs describe the on-device validation workflow as "Verify model accuracy on-device using an
  inference job", suggesting quantitative summaries such as KL divergence or label accuracy; the model
  cards suggest "you can compute like PSNR, relative errors or spot check the output with expected
  output" — **[search-summary]** — [AI Hub "Running Inference" docs](https://app.aihub.qualcomm.com/docs/hub/inference_examples.html)
- The public **accuracy debugger** doc (SNPE/QNN-era, doc 80-62010-1) exposes `--comparator cosine` and
  a one-shot layerwise CSV containing per-layer source/target min, max, median plus a
  "cosine similarity score of the current layer output compared to reference output"; the fetched text
  contains **no default threshold** for any verifier — **[verified-source, negative]** — [Accuracy debugger](https://docs.qualcomm.com/bundle/publicresource/80-62010-1/topics/accuracy_debugger.md)

### Inferences

- Qualcomm has effectively standardized on an **energy-ratio metric** (PSNR), which is scale-free and
  immune to near-zero denominators — a deliberate avoidance of per-element relative error.
- The `outputs_to_skip_validation` mechanism is the vendor acknowledging that some outputs are
  structurally unsuited to numeric comparison; the fix is per-output exemption, not threshold inflation.

### Gaps

- The newer `qairt-accuracy-debugger` verifier list with default parameters (historically `RtolAtol`,
  `AdjustedRtolAtol`, `TopK`, `L1Error`, `CosineSimilarity`, `SQNR`, `MeanIOU`, `ScaledDiff`) and their
  numeric defaults could not be verified — the public page I fetched is truncated/image-heavy and points
  to [general_tools.html#qairt-accuracy-debugger](https://docs.qualcomm.com/doc/80-63442-10/topic/general_tools.html#qairt-accuracy-debugger), which I did not retrieve. Any specific rtol/atol or SQNR dB
  default for QNN should be treated as unverified.

---

## Q3. Apple coremltools: SNR / PSNR thresholds and FP16 validation

### Takeaway

Apple's documented guidance is **tensor-level SNR in dB** ("above 20 dB indicates that the models
perform similarly"), and its own test suite, when the model runs in FP16 or on non-CPU compute units,
**inflates elementwise tolerances to atol 0.5 / rtol 0.05** and additionally permits a configurable
*ratio* of mismatching elements.

### Cited Findings

- Official SNR formula in the conversion docs: `noise_var = sum(noise**2)/len + 1e-9`,
  `signal_energy = sum(y**2)/len + 1e-9`, `SNR = 10*log10(signal_energy/noise_var)`, **with an explicit
  near-zero clamp: `if signal_energy < 1e-10: return 100  # clip max SNR`** — **[verified-source]** — [coremltools docs-guides typed-execution-example](https://apple.github.io/coremltools/docs-guides/source/typed-execution-example.html)
- Acceptance guidance, verbatim: "For most ML applications, **an SNR above 20dB** indicates that the
  models perform similarly. However, a good practice is to verify the error metric not just on a single
  input data point, but over a set of examples" — **[verified-source]** — [same page](https://apple.github.io/coremltools/docs-guides/source/typed-execution-example.html)
- Worked numbers from that page: FP16 Core ML model vs TF reference **SNR ≈ 71.4 dB**; FP32 Core ML
  model vs TF **≈ 95.8 dB**; conclusion "With an SNR of 70dB, the float 16 typed Core ML model is
  sufficient" — **[verified-source]** — [same page](https://apple.github.io/coremltools/docs-guides/source/typed-execution-example.html)
- Library function `compute_snr_and_psnr(x, y)` with `eps = 1e-5`, `eps2 = 1e-10`:
  `snr = 10*log10((signal_energy + eps)/(noise_var + eps2))`,
  `psnr = 10*log10((max_signal_energy + eps)/(noise_var + eps2))` — present twice, in the MIL test
  utilities and in the shipped experimental debugging API — **[verified-source]** — [coremltools/converters/mil/_testing_utils.py](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/_testing_utils.py), [coremltools/models/ml_program/experimental/debugging_utils.py](https://github.com/apple/coremltools/blob/main/coremltools/models/ml_program/experimental/debugging_utils.py)
- **FP16 tolerance inflation in the op test harness** (read at commit `a207b1a`, 2026-09-30): defaults
  are `atol=1e-04, rtol=1e-05`, then
  `if mlmodel.compute_unit != ct.ComputeUnit.CPU_ONLY or (dtype == "fp16"): atol = max(atol*100.0, 5e-1); rtol = max(rtol*100.0, 5e-2)`
  — i.e. **FP16/accelerator comparisons run at atol 0.5, rtol 0.05** — **[verified-source]** — [coremltools/converters/mil/testing_utils.py](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/testing_utils.py)
- The same harness supports a failing-element ratio: `allow_mismatch_ratio` (default `0.0`) — on an
  `assert_allclose` failure it recomputes `mismatch_num/total_num` and only raises if that exceeds the
  ratio; the docstring says it is "Mainly used for comparing compressed models outputs" — **[verified-source]** — [testing_utils.py](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/testing_utils.py)
- Test inputs are generated so they are exactly representable in FP16: "Since this function is
  extensively used as well for the fp16 precision models, we make sure that the numerical value can be
  presented in fp16" (`gen_dtype = np.float16 if dtype == np.float32`), and generated values are kept
  at least `eps_from_int` away from integers — **[verified-source]** — [testing_utils.py](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/testing_utils.py)
- Threshold constants in coremltools' own tests: PyTorch-conversion numeric tests use
  `snr_thresh=20.0, psnr_thresh=23.0`; the legacy TF numeric tests use `snr_thresh=15, psnr_thresh=30`;
  the per-operation debugging-utility tests require `compute_snr_and_psnr(x, y)[1] > 60.0` (PSNR) — **[verified-source]** — [test/optimize/torch/conversion/conversion_utils.py](https://github.com/apple/coremltools/blob/main/coremltools/test/optimize/torch/conversion/conversion_utils.py), [test/neural_network/test_tf_numeric.py](https://github.com/apple/coremltools/blob/main/coremltools/test/neural_network/test_tf_numeric.py), [test/ml_program/experimental/test_debugging_utils.py](https://github.com/apple/coremltools/blob/main/coremltools/test/ml_program/experimental/test_debugging_utils.py)
- The shipped model-debugging guide's example op-level comparator is `np.allclose(reference_output,
  target_output, atol=1e-1)` — i.e. Apple's own suggested per-op screen is atol 0.1 — **[verified-source]** — [docs-guides mlmodel-debugging-perf-utilities](https://github.com/apple/coremltools/blob/main/docs-guides/source/mlmodel-debugging-perf-utilities.md)

### Inferences

- Apple's two-tier practice: tensor-level SNR/PSNR for "is the model equivalent", and a deliberately
  *loose* elementwise check (atol 0.5) only as a crash/garbage detector for FP16 execution. Nobody at
  Apple is running atol 1e-5 against FP16 hardware output.
- The `signal_energy < 1e-10 → return 100` clamp is Apple's explicit statement that an all-near-zero
  reference tensor is a *pass*, not a failure — directly analogous to the problem in our harness.

---

## Q4. AMD Ryzen AI / Vitis AI / XDNA, and NPUEval

### Takeaway

AMD's model-level criterion is a **1 percentage-point absolute drop in the task metric** on the NPU vs a
CPU/GPU baseline (no tensor-level metric at all), while AMD Research's NPU *kernel* benchmark (NPUEval)
uses **flat absolute-error tolerances of 1e-2 / 2e-2 / 3e-2** and removes rounding noise at the source
by programming the NPU's bf16 rounding mode to match the reference model.

### Cited Findings

- Vitis AI (gen2 / Ryzen AI) "Model Accuracy Validation Methodology": metrics are task-standard (mAP,
  Top-1/Top-5, F1) on the full validation set; the **NPU tolerance is fixed at 1%**, defined as "within
  1 percentage point absolute difference" — example given: "For a baseline of 85.0%, acceptable accuracy
  is 84.0% or higher" — **[verified-source]** — [AMD Vitis AI accuracy methodology](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/model_compilation/accuracy_methodology.html)
- Four-step comparison chain: FP32 CPU/GPU baseline → quantized CPU/GPU baseline (skippable for BF16) →
  NPU run via VART/ONNX Runtime compared against the quantized baseline (or the FP32 baseline if BF16) →
  attribution step: "If the CPU baseline passes but NPU fails, the cause is NPU-specific" — **[verified-source]** — [same page](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/model_compilation/accuracy_methodology.html)
- BF16 is "the recommended starting point … no manual conversion or calibration"; for FP16, "a deviation
  over 1% from the original usually indicates an export or conversion problem"; INT8 needs 100–128
  representative calibration samples — **[verified-source]** — [same page](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/model_compilation/accuracy_methodology.html)
- **NPUEval** (AMD Research, 102 kernels for XDNA/AIE): correctness is per-kernel absolute error vs "the
  reference output produced by the Python behavioral model", with "a large **default absolute error
  threshold of 1e-2**" for logical/integer operators, "for **lower precision floating point operations
  the tolerance is set to 2e-2**", and "More complex kernels like **tanh or softmax** activations have
  their tolerances set to **3e-2**" — **[verified-source]** — [NPUEval paper §6.2](https://arxiv.org/html/2507.14403v1)
- NPUEval removes rounding-mode mismatch rather than tolerating it: "The NPU has a **programmable
  rounding mode** for its bfloat16 implementation, which we leverage to ensure the same operations will
  result in identical outputs minimizing compounding rounding errors" — **[verified-source]** — [NPUEval paper](https://arxiv.org/html/2507.14403v1)
- NPUEval computes maximum relative error only as a *supplementary reported metric*, not as a gate; the
  paper gives no relative threshold, no FP32 tolerance, and **no near-zero handling rule** — **[verified-source, negative]** — [NPUEval paper](https://arxiv.org/html/2507.14403v1)

### Inferences

- AMD's kernel-level approach is the opposite of Ascend's: instead of modelling error regions, it uses a
  single generous absolute tolerance (1e-2 to 3e-2) — which incidentally makes subnormal flush-to-zero
  and cancellation irrelevant, because those artifacts are far below 1e-2 in magnitude for bf16 data in
  a [-1,1]-ish range.
- An absolute-only tolerance is only sound when input ranges are controlled by the benchmark; AMD can do
  this because NPUEval generates its own test vectors.

### Gaps

- I did not verify NPUEval's actual comparison code (the paper does not name the function, and the
  GitHub repo clone failed in this environment — `github.com/amd/npueval` was unauthenticated-blocked),
  so whether it uses `np.allclose`-style `atol` alone or `atol + rtol` is unconfirmed.
- No AMD document found that defines a *tensor-level* numeric metric (cosine/PSNR/SNR) for NPU vs CPU
  output comparison; the published methodology is task-metric only.

---

## Q5. Intel: OpenVINO NPU plugin and NNCF

### Takeaway

Intel documents that **NPU hardware computes in FP16** regardless of the model's declared precision, but
publishes **no numeric accuracy criterion for the NPU plugin**; the only quantitative accuracy gate in
the Intel stack is NNCF's accuracy-aware quantization, whose default is a **0.01 absolute drop in the
user's own validation metric**.

### Cited Findings

- OpenVINO NPU device page, verbatim: "The NPU plugin supports the following data types as inference
  precision of internal primitives: Floating-point data types: **F32, F16**; Quantized data types: U8
  (quantized models may be INT8 or mixed FP16-INT8). **Computation precision for the HW is FP16.**" — **[verified-source]** — [OpenVINO 2025 NPU Device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- The NPU plugin exposes `ov::hint::inference_precision` and `ov::hint::execution_mode` as read-write
  properties (so precision is a hint, not a guarantee of FP32 math) — **[verified-source]** — [same page](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- NNCF `quantize_with_accuracy_control`: `max_drop` **default 0.01**, `drop_type` default **ABSOLUTE**
  (alternative: RELATIVE); "NNCF will stop the quantization and report an error if the max_drop value
  can't be reached" — **[search-summary]** — [OpenVINO "Quantizing with Accuracy Control"](https://docs.openvino.ai/2023.3/quantization_w_accuracy_control.html)
- The accuracy-control flow is driven by a **user-supplied validation function and dataset** (task
  metric), i.e. the criterion is end-to-end metric drop, not tensor similarity — **[search-summary]** — [same docs family, e.g. YOLOv8 accuracy-control notebook](https://docs.openvino.ai/2024/notebooks/yolov8-quantization-with-accuracy-control-with-output.html)

### Inferences

- "Computation precision for the HW is FP16" is Intel's own statement that an FP32-declared model will
  still be computed in FP16 on NPU; any reference used for a correctness verdict must therefore be
  FP16-rounded (as our harness does) *and* the verdict must tolerate FP16-level artifacts, because Intel
  publishes no tighter guarantee.

### Gaps

- I found **no OpenVINO document that states a numeric tolerance, SNR, cosine or per-element criterion
  for NPU-plugin accuracy validation**, and no public statement about FP16 subnormal/FTZ behavior on the
  NPU (the 3720's flush-to-zero behavior we observe empirically is not documented in the pages I read).
  The historical `accuracy_checker` / Open Model Zoo validation tooling is metric-based (task metrics),
  not tensor-based, but I did not fetch its current docs in this session.

---

## Q6. MediaTek NeuroPilot, Hailo, Arm Ethos/TOSA/Vela, Google LiteRT

### Takeaway

Arm is the outlier and the most rigorous: the **TOSA specification is a formal numerical conformance
standard** with per-operator ULP bounds, a statistical (variance-based) dot-product test, and explicit
language that **FP16 subnormals may be flushed to zero** — with the checker written to *accept* a
flushed zero. Hailo uses per-layer SNR in dB (> 10 dB rule of thumb), Google's LiteRT debugger uses
`RMSE/scale` with ~0.289 as the healthy value, and MediaTek publishes nothing I could verify.

### Cited Findings — Arm / TOSA (the reference standard behind Ethos-U and Arm NPUs)

- Integer ops: "In a compliant implementation, **individual integer operations must match exactly**. To
  check exact matching, the `tosa_reference_check_fp` function can be used with `num_ulp` set to 0" — **[verified-source]** — [TOSA spec 1.0.2 §1.10 Floating-Point Precision Requirements](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- FP ops are checked against an **fp64 reference** with a per-operator ULP or absolute bound. Verified
  per-op bounds: `ERF` → `tosa_reference_check_fp(..., 5)` (5 ulp); `RECIPROCAL` → 1 ulp; `RSQRT` →
  2 ulp; `SIGMOID` → `calcAbsErrorBound(out_ref, 2*(1+|x|), 0, 1)`; `TANH` →
  `calcAbsErrorBound(out_ref, 4*(exp_err_base + 2*|x|), 0.5, 1)` with
  `exp_err_base = 3 (fp32) : 1 (otherwise)`; `LOG` → `calcAbsErrorBound(out_ref, 5, 0.5, 1)`;
  `SIN`/`COS`/`EXP`/`POW` have their own input-dependent bounds — **[verified-source]** — [TOSA spec 1.0.2 operator sections](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- ULP definition: `tosa_reference_ulp(ref) = max(2**ilog2(|ref|), normal_min) * 2**(-normal_frac)` with
  `normal_frac` = mantissa bits (fp32 23, bf16 7, fp16 10, fp8e4m3 3, fp8e5m2 2) — note the
  `max(..., normal_min)` **floor**, which prevents the bound collapsing to zero near zero — **[verified-source]** — [TOSA spec 1.0.2 §4.5 helpers](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- Absolute error bound helper:
  `calcAbsErrorBound(bound_magnitude, bounds_value, lower_bound, normal_divisor)`:
  `value_bound = max(|bound_magnitude|, normal_min)`; if `lower_bound > 0`,
  `value_bound = max(lower_bound/bounds_value, value_bound)`; then
  `error_bound = 2**(-normal_frac/normal_divisor) * value_bound * bounds_value` — i.e. the standard
  *adds an absolute floor* to a relative bound exactly to cover near-zero references — **[verified-source]** — [TOSA spec 1.0.2 §4.5](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- **FTZ is explicitly legal and explicitly accommodated by the checker.** Number formats: "fp16_t …
  **Subnormal values must either be supported or flushed to sign-preserved zero**." Precision
  requirements: "For data types that allow subnormal values to be flushed to zero, **either all values
  must be flushed to sign-preserved zero, or none of them**." And in `tosa_reference_check_fp_bnd`:
  "Allow subnormal values to be flushed to zero for non-fp8 — `if (test_value == 0) return (ref_min <
  normal_min());`" — i.e. **an implementation output of exactly 0 passes whenever the reference's
  lower error bound dips below the smallest normal** — **[verified-source]** — [TOSA spec 1.0.2 §1.9, §1.10, §4.5](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- Most elementwise ops begin their precision requirements with "**Subnormal bf16_t, fp16_t, and fp32_t
  input values may be flushed to zero before calculation**" — the spec never treats FTZ as an error — **[verified-source]** — [TOSA spec 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- **Dot-product (MATMUL/CONV) accuracy is statistical, not per-element-relative.** With `KS` = kernel
  size (dot-product length) and
  `ksb = ceil(KS / 2**((normal_frac<acc_t> − normal_frac<out_t>)/2)) + (bias present ? 1 : 0)`:
  per-element normalized error is `out_err = (out_imp − out_ref) / max(out_bnd * 2**(-1-normal_frac<out_t>), normal_min<out_t>)`;
  the gates are **`|out_err| ≤ ABS_BOUND = 2*ksb`** per element, **`out_err_sumsq ≤ VARIANCE_ERROR_BOUND * T`**
  (`VARIANCE_ERROR_BOUND = 4*0.4*ksb`, T = number of outputs), and for test sets 3–5 a bias check
  `|out_err_sum| ≤ sqrt(10 * VARIANCE_ERROR_BOUND * T)` ("The factor 10 allows for up to a 4 sigma
  difference of the error sum … assuming errors are normally distributed") — **[verified-source]** — [TOSA spec 1.0.2 "Dot product accuracy requirements"](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- Zero handling in the dot-product check is special-cased: "if `out_bnd_el == 0.0` // All products in the
  dot product are zero → `if (out_ref_el != 0.0 || out_imp_el != 0.0) return false;`"; NaN reference
  requires NaN output; a bound that can overflow within the error bound imposes "no accuracy limit" — **[verified-source]** — [TOSA spec 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- MATMUL's declared accumulate modes include "fp16 with fp16 accumulate" and "fp16 with fp32 accumulate"
  as *separate* supported type rows — the standard treats accumulator width as part of the contract — **[verified-source]** — [TOSA spec 1.0.2 MATMUL](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- Ethos-U **Vela** (int8/int16 only) validates by bit-exactness against the TensorFlow Lite reference
  kernels, versioned per release (Vela 3.10.0 onward vs TensorFlow 2.14) — **[search-summary]** — [Vela commit "MLBEDSW-8240: Document reference comparison point"](https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/5aa9ae24ed834a3aeba4917823437bac75e14ab3%5E%21/); TOSA design goal "integer operators should be implementable in a bit-exact form" — [TOSA wiki](https://developer.mlplatform.org/w/tosa/)

### Cited Findings — Hailo

- Hailo's Dataflow Compiler reports **SNR in dB**, and Hailo community guidance treats **> 10 dB** as
  generally acceptable; the profiler reports two SNRs: one measured at the model output layer(s) with
  only a single layer quantized, and a cumulative per-layer SNR "considering all previous layers have
  been quantized" — **[search-summary]** — [Hailo community: "SNR value not higher than 10"](https://community.hailo.ai/t/snr-value-not-higher-than-10/1931)
- Hailo markets the emulator as "bit exact emulation of chip behavior", while a community thread states
  a HEF cannot be emulated on CPU/GPU and that staged comparison is done through the HAR file's three
  model states — **[search-summary]** — [Hailo software components page](https://hailo.ai/?p=619); [Hailo community: "Hailo Emulation Failing"](https://community.hailo.ai/t/hailo-emulation-failing/14357)

### Cited Findings — Google LiteRT / Edge TPU

- `QuantizationDebugger` emits per-layer `layer_statistics` CSV; the headline derived metric is
  **`rmse/scale` = sqrt(mean_squared_error) / tensor scale**. A value near **1/sqrt(12) ≈ 0.289** means
  "quantized distribution is similar to the original float distribution"; "the larger the value is, it's
  more likely for the layer not being quantized well", and the tutorial's filter is
  **`rmse/scale > 0.7`** (an illustrative cutoff, not a documented constant) — **[search-summary]** — [LiteRT quantization debugger](https://ai.google.dev/edge/litert/models/quantization_debugger)

### Cited Findings — MediaTek

- MediaTek's public NeuroPilot pages describe quantization and TFLite conversion but contain **no
  accuracy metric or threshold**; no cosine-similarity or SNR comparison tool is documented publicly — **[search-summary, negative]** — [NeuroPilot Introduction](https://developer.mediatek.com/ai/6423c8b6c612745b3a4baa2c.html), [NeuroPilot Examples](https://developer.mediatek.com/ai/64254fa5f55b040d6989a99c.html)

### Inferences

- TOSA is the closest thing to an industry *standard* (as opposed to tooling convention) for NPU
  numerical acceptance, and three of its design choices map directly onto our problem: an absolute floor
  (`normal_min`) inside every relative bound; explicit permission for FTZ with the checker accepting a
  zero output; and for reductions, a per-element bound that **grows with reduction length**
  (`ABS_BOUND = 2*ksb`) plus a tensor-level variance test rather than a fixed elementwise tolerance.
- Arm's "either all values flushed or none" rule suggests a useful consistency test: a kernel whose only
  deviation is uniform FTZ is conformant, whereas inconsistent FTZ is not.

### Gaps

- Hailo's official Dataflow Compiler user guide (developer zone, login-gated) was not accessible, so the
  10 dB figure rests on Hailo community posts rather than the SDK manual; no documented per-layer SNR
  pass threshold enforced by the tool was found.
- No primary source for MediaTek NeuroPilot accuracy criteria; likely only in the licensed SDK docs.
- The LiteRT 0.7 cutoff is from a tutorial, and the docs do not define a `mean_error` threshold.

---

## Q7. Cross-vendor: is per-element tolerance or a tensor-level metric the norm? How is near-zero / FTZ handled?

### Takeaway

**No NPU vendor accepts kernels on a flat per-element atol/rtol against a high-precision reference.**
Every vendor uses one of four patterns: (1) tensor-level statistic (PSNR/SNR/cosine) — Qualcomm, Apple,
Hailo, Ascend's first screen; (2) ratio-of-failing-elements — Ascend 双百/双千/双万, coremltools
`allow_mismatch_ratio`; (3) relative error *only on normal-magnitude elements*, absolute error on small
ones — Ascend, TOSA's `calcAbsErrorBound` floor, CANN three-region rule; (4) deliberately coarse
absolute tolerance — NPUEval 1e-2..3e-2. Where a per-element rule survives, the tolerance is set at
roughly **1 ulp of the device dtype** (Ascend fp16 rtol `2**-10`), not at FP32-like 1e-5.

### Cited Findings (consolidated, all verified above)

| Vendor / tool | Granularity | Metric | Threshold | Near-zero / FTZ rule |
|---|---|---|---|---|
| Ascend msprobe (pre-check) | tensor + element-ratio | cosine, max abs err, 双百/双千/双万, ULP | cos > 0.99; max abs err < 1e-3 → pass; fp16/bf16 need rel-err-fail-ratio < 1% (pass) / < 0.1% (strict); fp32 < 0.1% / < 0.01%; ULP thresh 1 (fp16/bf16) with proportion < 0.001 | elements with \|golden\| ≤ 2^-10 (fp16) judged by abs err ≤ 2^-16; excluded from relative-error stats ([doc §4.2](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/07.accuracy_checker_PyTorch.md)) |
| Ascend api_precision_compare (标杆比对法) | tensor, ratio-to-GPU | small-value err ratio, RMSE, max/mean rel err, EB | ≤ 2, ≤ 2, ≤ 10, ≤ 2, ≤ 2 | dtype-intrinsic error cancels because NPU and GPU are both divided by the same CPU-reference error |
| CANN ecosystem operator standard | tensor (mean+max) then region-wise count | mean RE < τ_d, max RE < 10·τ_d | fp16 9.77e-4 / 9.77e-3; bf16 7.81e-3 / 7.81e-2; fp32 1.22e-4 / 1.22e-3 | 3 regions; small-value (\|y_trunc\| < 2^-11 fp16) and cancellation (\|ŷ\| < 2^-5 fp16) fall back to `ErrorCount_NPU / max(ErrorCount_CPU,1) ≤ 2` |
| Qualcomm AI Hub | tensor | PSNR (dB) | ≥ 20 dB enforced; > 30 dB "good"; task metric within 1.0 pt | scale-free metric; per-output exemption list for meaningless outputs |
| Apple coremltools | tensor (SNR) + loose element + ratio | SNR/PSNR dB; `assert_allclose` | SNR > 20 dB guidance; tests use SNR 15–20 dB, PSNR 23–30 dB (60 dB per-op); FP16 element check at atol 0.5 / rtol 0.05 | `if signal_energy < 1e-10: return 100` (near-zero reference ⇒ max SNR) |
| AMD Ryzen AI / Vitis AI | end-to-end task metric | mAP / Top-1 / F1 | within 1 percentage point of CPU baseline | n/a (no tensor metric) |
| AMD NPUEval | per element | absolute error | 1e-2 default; 2e-2 low-precision float; 3e-2 tanh/softmax | handled by construction: NPU bf16 rounding mode programmed to match reference |
| Intel OpenVINO NPU / NNCF | end-to-end task metric | user validation function | `max_drop` 0.01 absolute (NNCF default) | not documented; HW computes FP16 |
| Arm TOSA (Ethos / Arm NPUs) | per element with abs floor + tensor variance | ULP bounds per op; dot-product normalized error | int exact (0 ulp); RECIPROCAL 1 ulp, RSQRT 2 ulp, ERF 5 ulp; dot product \|err\| ≤ 2·ksb and Σerr² ≤ 1.6·ksb·T | `max(..., normal_min)` floor in every bound; FP16 subnormals **may** be flushed (all-or-none), and a zero output passes if `ref_min < normal_min` |
| Google LiteRT debugger | per layer | RMSE / scale | ≈ 0.289 healthy; tutorial flags > 0.7 | metric normalized by quantization scale |
| Hailo | per layer + output | SNR (dB) | > 10 dB rule of thumb | scale-free metric |

### Inferences (directly usable for our FP16 NPU-3720 verdict)

- An industry-aligned verdict for an FP16 NPU would be a **two-stage gate**, which is what both Huawei
  and the CANN ecosystem standard do: (1) cheap tensor-level screen — e.g. `max |abs err| < 1e-3`
  (Ascend) or PSNR ≥ 20 dB (Qualcomm) or SNR ≥ 20 dB (Apple) — and only on failure (2) an
  element-partitioned relative-error test with `rtol ≈ 2**-10` for FP16 on normal-magnitude elements,
  an absolute tolerance (`2**-16`, or 1e-5 in the per-op YAML) on small-magnitude elements, and a
  tolerated failing-element ratio (1e-2 → 1e-3) rather than zero failures.
- Our current `atol 1e-5 + rtol 1e-3` is, in vendor terms, roughly Ascend's fp16 `rtol 1e-3` paired with
  an `atol` that is *tighter than Ascend's small-value tolerance for fp16* (`1e-5` vs Ascend's `1e-5`
  per-op YAML but `2**-16 ≈ 1.5e-5` in `StandardConfig`) — and crucially we apply it **to every element
  with zero tolerated failures**, which is the part no vendor does.
- FP16 subnormal flush-to-zero is explicitly sanctioned by the only formal spec in the field (TOSA),
  which also tells us how to score it: a zero output is acceptable whenever the reference's error
  interval reaches below the smallest *normal* FP16 value (`2**-14 ≈ 6.1e-5`). An `atol` of about
  `6.1e-5` (or the Ascend small-value tolerance of `2**-16`) is therefore the smallest absolute floor
  that is consistent with vendor practice; anything tighter is testing the hardware's subnormal support,
  not the kernel.
- Cancellation-near-zero is handled by the CANN three-region rule, which needs a **same-precision
  reference run** (CPU FP16 emulation) to divide out dtype-intrinsic error; this is the one vendor
  technique that distinguishes "kernel is wrong" from "FP16 cannot represent this result" without
  loosening thresholds globally.

### Gaps

- No vendor source found that quantifies an acceptable *rate* of 1-ulp disagreements for elementwise
  FP16 kernels on NPUs other than Ascend (1e-2/1e-3 ratios) and TOSA (per-op ulp counts); Qualcomm,
  Intel and MediaTek publish nothing at that granularity.
- No vendor document found that states its NPU flushes FP16 subnormals to zero as a specified behavior
  (TOSA permits it as an implementation choice; Intel's NPU docs are silent), so the 3720's FTZ remains
  an empirical observation rather than a documented contract.
