# Tensor-level metric-based and end-to-end acceptance methods for low-precision accelerator outputs

Scope note: source code was fetched from the default branches (main/master) on 2026-10-22, and the line numbers below are as of that fetch. Commits pinned where recorded: pytorch/ao `9e9dca9`, vllm `856b984`, quic/ai-hub-models `c23219d`. Items marked **[UNVERIFIED]** come from search-result summaries or background knowledge, not from a primary source read in this session.

Context for the report writer: the harness in question uses an elementwise check `|dev - ref| <= 1e-5 + 1e-3*|ref|` against an FP16-rounded reference. Those are exactly PyTorch's `torch.testing.assert_close` defaults for float16 (rtol=1e-3, atol=1e-5). Transformer Engine copies the same defaults into its tests (see Q6).

---

## Q1. MLPerf Inference rules: 99% / 99.9% targets, how accuracy is measured, what this implies for kernel testing

### Takeaway
MLPerf judges numerics only at the task level. A submission passes if its task metric (top-1, ROUGE, exact match, WER, mAP, DICE, F1, AUC…) over the full accuracy dataset reaches 99% (or 99.9%, and for a few benchmarks 95–97%) of the reference score computed in FP32, FP16 or BF16. Beyond that, any quantization or numerical format is allowed. MLPerf has no kernel-level criterion, and a 1% task-metric budget is far too coarse to detect localized kernel bugs.

### Cited Findings
- Every valid submission "must be *equivalent* to the reference implementation". Equivalence is defined operationally: "All implementations are allowed as long as the latency and accuracy bounds are met and the reference weights are used. Reference weights may be modified according to the quantization rules." — [MLPerf inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Quantization: submitters "may do arbitrary purely mathematical, reproducible quantization using only the calibration data and weight and bias tensors … to any numerical format that achieves the desired quality". The closed division "allows calibration for quantization and does not allow any retraining". Open division: "any quantization scheme is allowed that achieves the desired quality." — [MLPerf inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Datacenter quality targets in the current master table (quoted). The QSL size is the number of samples in the accuracy dataset.
  - ResNet50-v1.5 / ImageNet: "99% of FP32 (76.46%)".
  - 3D-UNet / KiTS 2019 (QSL 42): "99% of FP32 and 99.9% of FP32 (0.86330 mean DICE score)".
  - Llama3.1-8B / CNN-DailyMail (QSL 13368): "99% of FP32 and 99.9% of FP32 (rouge1=42.9865, rouge2=20.1235, rougeL=29.9881). Additionally … total generation length … more than 90% of the reference (gen_len=8167644)".
  - Llama2-70b / OpenOrca (QSL 24576): "99.9% of FP32 (rouge1=44.4312, …) … tokens per sample should be more than 90% of the reference (tokens_per_sample=294.45)".
  - Llama3.1-405B: "99% of FP16 (… rougeL=21.6666, … exact_match=90.1335) … tokens per sample should be between … 90% and 110% of the reference".
  - Mixtral-8x7B: "99% of FP16 (… (gsm8k)Accuracy=73.66, (mbxp)Accuracy=60.16)" plus the 90–110% length rule.
  - DeepSeek-r1: "99% of FP16 (exact match 81.9132%)".
  - GPT-OSS-120B: "99% of 83.13%".
  - Qwen3-VL-235B: "99% of BF16 (Category Hierarchical F1 Score >= 0.7824)".
  - DLRMv3: "99.9% of FP32 (NE=86.687%, Accuracy=69.651%, AUC=78.663%)".
  - WAN-2.2 T2V: "99% of BF16 (VBench score >= 69.7752)".
  - RGAT: "99% of FP32 (72.86%)".
  - Whisper: "99% of FP32 and 99.9% of FP32 (WER=2.0671%)".
  - E2E-RAG: "97% of FP32 (LLM Judge score = ~35%; +/- ~3%)".
  - Source: [MLPerf inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Edge table (quoted):
  - ResNet50 (QSL 1024): "99% of FP32 (76.46%)".
  - YOLO11l: "95% of FP32 (yolo-95: 0.5073 mAP) and 99 of FP32 (yolo-99: 0.5287 mAP)".
  - BERT / SQuAD v1.1: "99% of FP32 and 99.9% of FP32 (f1_score=90.874%)".
  - PointPainting: "99.9% of FP32 (0.5425 mAP)".
  - Source: [MLPerf inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Newer LLM rows give the reference precision as FP16 or BF16 rather than FP32. The LLM benchmarks also add an output-length guard (≥90%, or 90–110% of the reference tokens per sample), which stops a submission from gaming ROUGE by producing shorter or degenerate outputs. — [MLPerf inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- MLPerf also allows "an INT8 reference for some models", with per-tensor scaling of weights and input activations. — [MLPerf inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)

### Inferences
- "99% of FP32" means the submission's task score must be at least 0.99 × the reference score (e.g. ResNet50 ≥ ~75.70% top-1). The tolerance is defined on the end metric, not on tensors. Two consequences follow: a kernel can deviate a lot from an elementwise standpoint and still pass, and a kernel with a real bug in a rarely exercised path (a tail tile, an odd sequence length, a padding/mask edge) can also pass if that path barely moves the aggregate metric.
- For a per-kernel harness, MLPerf-style checks are a complement that answers "is the FP16 deviation tolerable for the application". They do not replace a bug detector. The practical analogue for LLM kernels is a model-level check: insert the generated kernel into the model and compare token, top-k and logprob agreement or perplexity against the reference (see vLLM's `check_logprobs_close`, Q2).
- The output-length guard is a pattern worth reusing: alongside the main metric, add a cheap secondary invariant that catches ways of gaming it.

### Gaps
- The MLPerf round that the current master corresponds to (v6.x?) was not confirmed. The exact interpretation of "99% of FP32" for lower-is-better metrics such as WER was not verified in the text.
- No MLPerf document gives guidance on kernel-level or tensor-level testing. None appears to exist.

---

## Q2. Cosine / relative-L2 / SNR / PSNR / NMSE thresholds used by real tools

### Takeaway
Industry thresholds cluster in three tiers:
- **(a) Quantization-grade gates.** Cosine ≥ 0.99–0.999, SQNR/SNR ≥ 15–30 dB, PSNR ≥ 30–35 dB. These are appropriate for int8/fp8/fp4 and are far too loose to catch fp16 kernel bugs.
- **(b) Same-precision kernel gates.** Examples: llama.cpp NMSE ≤ 1e-7 (≈70 dB) by default, or 5e-4 (≈33 dB) for matmul/conv/flash-attention; PyTorch inductor's "RMSE vs fp64 no worse than 2–3× eager's".
- **(c) Elementwise gates** with dtype-derived rtol/atol.

Cosine similarity is used mainly as a fallback or diagnostic (PyTorch opt-in flag at 0.99, Polygraphy 0.997, vLLM 0.99/0.999), and practitioners pair it with other checks.

### Cited Findings

**PyTorch Inductor / Dynamo `torch._dynamo.utils.same()`**
- Signature defaults: `tol=1e-4`, `cos_similarity=False`, `exact_dtype=True`. When the dtypes differ, it logs "dtype mismatch" and returns False. — [torch/_dynamo/utils.py L3688–3700, L3812–3816](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/utils.py)
- Cosine mode: flattens to fp32 and passes early if `torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=True)`. The early exit exists because the cosine of two zero vectors is 0. Otherwise it computes `cosine_similarity(..., eps=1e-6)` and returns `score >= 0.99`. — [torch/_dynamo/utils.py L3849–3858](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/utils.py)
- Default (non-cosine) mode is a two-stage check:
  1. Plain `allclose(atol=tol, rtol=tol)` first.
  2. If that fails and an fp64 reference exists, compare `res_error = RMSE(fp64_ref, res)` against `ref_error = RMSE(fp64_ref, ref)`, where `ref` is the eager result in the same low precision. It passes if `res_error <= multiplier * ref_error + tol/10`.
  - `multiplier` is 3.0 for fp16/bf16 and 2.0 otherwise. It is 3.0 when `numel < 1000`, for 1×1 spatial outputs, or when `tol >= 2e-2`. With `use_larger_multiplier_for_smaller_tensor` it is 10.0 for `numel <= 10` and 8.0 for `numel < 1000`; `force_max_multiplier` gives 10.0.
  - The code comment says AMP models "failed the benchmark's correctness check" although end-to-end accuracy differed by "less than 0.1%", so the multiplier was raised from 2 to 3 "to avoid these false alarms".
  - Source: [torch/_dynamo/utils.py L3859–3960](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/utils.py)
- Bool outputs can optionally be compared by IoU (`use_iou_for_bool`, failing if `iou < iou_threshold`), "useful for segmentation models where small floating-point differences get thresholded into boolean masks". — [torch/_dynamo/utils.py](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/utils.py)

**NVIDIA TensorRT Polygraphy (`polygraphy/comparator/compare.py`)**
- `SimpleCompareFunc` defaults: `_DEFAULT_RTOL = 1e-5`, `_DEFAULT_ATOL = 1e-5`, `_DEFAULT_QUANTILE = 0.99`, `_DEFAULT_ERROR_STAT = "elemwise"`. `check_shapes` defaults to True. — [compare.py L799–802, L885](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/comparator/compare.py)
- The single-metric comparators have these defaults:
  - `L2CompareFunc`: `_DEFAULT_THRESHOLD = 1e-5` on the raw (unnormalized) Euclidean distance.
  - `CosineSimilarityCompareFunc`: `_DEFAULT_THRESHOLD = 0.997`.
  - `PsnrCompareFunc`: default 30.0 dB, computed as `20*log10(max(out0)) - 10*log10(MSE)`; if `max_val <= 0`, the peak is set to 1.0.
  - `SnrCompareFunc`: default 20.0 dB, computed as `10*log10(signal_power/noise_power)`.
  - An `LpipsThreshold` (perceptual) also exists.
  - Source: [compare.py L1346–1600](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/comparator/compare.py)
- Cosine edge cases: if both vectors are zero, the result is 1.0; if one is zero, 0.0. A NaN result "must fail the threshold". — [compare.py L1446–1462](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/comparator/compare.py)

**ONNX Runtime quantization debugging (`qdq_loss_debug.py`)**
- The purpose is to run the float32 and quantized models on the same inputs and "compare the corresponding activations between the two models to find where the divergence is". — [qdq_loss_debug.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/qdq_loss_debug.py)
- `compute_signal_to_quantization_noice_ratio` (sic) computes `20*log10(||x|| / ||x-y||)` with both norms floored at `numpy.finfo("float").eps`. The tool is diagnostic only and has no pass/fail threshold. — [qdq_loss_debug.py L341–362](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/qdq_loss_debug.py)

**PyTorch AO (torchao)**
- `compute_error(x, y) = 20*log10(||x|| / ||x−y||)` (SQNR in dB). — [torchao/quantization/utils.py L59–62](https://github.com/pytorch/ao/blob/main/torchao/quantization/utils.py)
- Test thresholds found by grepping `test/` at commit 9e9dca9 (quoted):
  - int8 dynamic-quant integration: `min_sqnr=35`, and one `sqnr >= 45.0` ([test/integration/test_integration.py](https://github.com/pytorch/ao/blob/main/test/integration/test_integration.py)).
  - float8: `sqnr > 25`, `sqnr > 15`, `sqnr > 20` ([test_float8_tensor.py](https://github.com/pytorch/ao/blob/main/test/quantization/quantize_/workflows/float8/test_float8_tensor.py)) and `out_sqnr > 20.0` ([test/float8/test_numerics_integration.py](https://github.com/pytorch/ao/blob/main/test/float8/test_numerics_integration.py)).
  - MX formats: `SQNR_THRESHOLD = 25.0 if elem_dtype == torch.float8_e4m3fn else 12.0`; 18.0 for weights and 15.0 for outputs elsewhere ([test/prototype/mx_formats/test_inference_workflow.py](https://github.com/pytorch/ao/blob/main/test/prototype/mx_formats/test_inference_workflow.py)).
  - mxfp8 grouped MM: `min_sqnr = 27.0` / 25.0 ([test_mxfp8_grouped_mm.py](https://github.com/pytorch/ao/blob/main/test/prototype/moe_training/test_mxfp8_grouped_mm.py)).
  - nvfp4 grouped MM: `min_sqnr = 16.0` / 19.0, output ≥ 15, input-grad ≥ 14, weight-grad ≥ 12 ([test_nvfp4_grouped_mm.py](https://github.com/pytorch/ao/blob/main/test/prototype/moe_training/test_nvfp4_grouped_mm.py)).
  - int8 LUT: `sqnr > 40.0` / `> 25.0` ([test_int8_lut_tensor.py](https://github.com/pytorch/ao/blob/main/test/prototype/test_int8_lut_tensor.py)).
  - intx: `sqnr > 30` / `> 35`.
  - quant API: `sqnr >= 16.5` ([test_quant_api.py](https://github.com/pytorch/ao/blob/main/test/quantization/test_quant_api.py)).
  - QAT with MX: `SQNR_THRESHOLD = 5.0 # Lower threshold for MX`.
  - Overall, the most frequent values are 14–15, 20, 25, 27–28, 30 and 35 dB.

**NVIDIA Transformer Engine tests (`tests/pytorch/utils.py`)**
- `dtype_tols()` is documented as "Based on tolerances for torch.testing.assert_close":
  - fp16: rtol=1e-3, atol=1e-5.
  - bf16: rtol=1.6e-2, atol=1e-5.
  - fp32: rtol=1.3e-6, atol=1e-5.
  - fp64: 1e-7 / 1e-7.
  - fp8 e4m3: rtol=0.125, atol=0.0675 ("epsilon = 0.0625").
  - fp8 e5m2: rtol=0.25, atol=0.125.
  - fp4 e2m1: rtol=0.25, atol=0.125.
  - `quantization_tols()` maps all FP8 recipes (delayed, current, blockwise, MXFP8) to the e4m3 tolerances and NVFP4 to the e2m1 tolerances.
  - Source: [TransformerEngine tests/pytorch/utils.py](https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/utils.py)

**Apple coremltools**
- `compare_backend()` defaults to `atol=1e-04, rtol=1e-05`. For any non-CPU compute unit (GPU/ANE) or fp16 it loosens to `atol = max(atol*100, 5e-1)` and `rtol = max(rtol*100, 5e-2)`. `allow_mismatch_ratio` (default 0.0) lets a fraction of elements fail "for comparing compressed models outputs". — [coremltools/converters/mil/testing_utils.py L407–466](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/testing_utils.py)
- `compute_snr_and_psnr(x, y)`: `snr = 10*log10((mean(y^2)+1e-5)/(mean((x−y)^2)+1e-10))` and `psnr = 10*log10((max(y^2)+1e-5)/(MSE+1e-10))`. — [coremltools/converters/mil/_testing_utils.py](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/_testing_utils.py)
- The model-chunking verifier sets `ABSOLUTE_MIN_PSNR = 35` dB ("parity check passed" if above). The code says this util comes from Apple's ml-stable-diffusion `torch2coreml.py`. — [coremltools/models/utils.py L2069–2097](https://github.com/apple/coremltools/blob/main/coremltools/models/utils.py)

**Qualcomm AI Hub / qai-hub-models**
- `compute_psnr`: `20*log10((data_range+1e-5)/(RMSE+1e-10))`, with `data_range` defaulting to `max(|ground_truth|)`. `compare_psnr(..., psnr_threshold)` asserts `psnr > psnr_threshold`. The explainer text reads: "Peak Signal-to-Noise Ratio (PSNR). >30 dB is typically considered good." The same module also offers MSE, MAE, max_abs_diff, top1 and top5. — [qai_hub_models/utils/compare.py](https://github.com/quic/ai-hub-models/blob/main/src/qai_hub_models/utils/compare.py)
- Qualcomm's docs describe running an on-device inference job and comparing it to the PyTorch output. They suggest going beyond one image to "measuring KL divergence or comparing accuracy (if the labels are known)" **[from search summary; page not fetched in full]**. — [AI Hub inference examples](https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html); [HF qualcomm model card](https://huggingface.co/qualcomm/PSPNet/blob/main/README.md)

**Huawei Ascend**
- Ascend's CANN accuracy-comparison guide (msaccucmp, GPU result as benchmark) lists cosine similarity with an acceptable threshold of "greater than 0.99" **[UNVERIFIED: search snippet only; the page did not render]**. — [Ascend CANN accuracy indicators](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/modelaccuracy/atlasaccuracy_16_0045.html)
- msprobe's accuracy pre-check compares NPU and GPU outputs against a CPU high-precision benchmark. It uses per-API methods: absolute threshold, benchmark comparison, binary consistency, ULP error and "dual-1000" **[UNVERIFIED: search summary; exact defaults not retrieved]**. — [msprobe README](https://gitcode.com/Ascend/msprobe/blob/master/README_EN.md)
- A MindFormers test helper uses small-value threshold 2^-10 for fp16/bf16, error thresholds 2^-11 (fp16) and 2^-8 (bf16), and a small-error-ratio parameter of 2 **[UNVERIFIED: search summary]**. — [mindformers double_benchmark.py](https://gitcode.com/mindspore/mindformers/blob/master/tests/utils/double_benchmark.py)
- CANN Bench (2026), a benchmark of agent-generated kernels on real Ascend NPUs, does not use cosine. It uses mean and max relative error with dtype thresholds plus region fallbacks (details in Q4). — [CANN Bench, arXiv 2607.20518](https://arxiv.org/abs/2607.20518)

**llama.cpp / GGML `test-backend-ops`** (details in Q3)
- The default metric is NMSE = `mse(a,b)/mse(a,0)`. The default threshold is `max_nmse_err() = 1e-7`. The most common relaxed value, 5e-4, is used for mul_mat, mul_mat_id, out_prod, conv*, im2col, flash_attn_ext and others. — [test-backend-ops.cpp L301–315, L1214–1216, L5343](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)

**vLLM kernel tests** (commit 856b984)
- Mostly elementwise: `torch.testing.assert_close`/`allclose` with `atol=rtol=1e-2` (most frequent), 2e-2, 1e-3 and 5e-2. Some kernels use cosine: `assert cosine > 0.999` (bf16 skinny GEMM, [test_bf16_skinny_gemm.py](https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_bf16_skinny_gemm.py)), `cosine > 0.99` (NVFP4 MoE, [test_flashinfer_cutedsl_nvfp4_moe.py](https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_flashinfer_cutedsl_nvfp4_moe.py)), and a row-wise minimum `F.cosine_similarity(output.float(), expected, dim=-1).min() > 0.99` (DeepGEMM MoE, [test_deep_gemm_mega_moe.py](https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_deep_gemm_mega_moe.py)).
- End-to-end model tests use `check_logprobs_close`. Generated tokens may differ. Where they differ (or always, with `always_check_logprobs`), the top-k logprob token sets must match, and each sequence's sampled token must appear in the other's top-k logprobs. — [vllm tests/models/utils.py](https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py)

**OpenVINO (relevant because the Intel NPU 3720 is driven through OpenVINO)**
- `ov::test::utils::compare(expected, actual, inference_precision, abs_threshold=-1, rel_threshold=-1, topk_threshold=1.f, mvn_threshold=1.f)`. An element passes if `|e−a| <= abs_threshold + rel_threshold*|e|`.
  - Default `rel_threshold` = max over (expected type, actual type, inference precision) of `get_eps_by_ov_type`: f32 1e-4, f16 1e-3, bf16 1e-2, f64 1e-8, nf4 1e-1.
  - Default `abs_threshold` = max `std::numeric_limits<T>::epsilon()` over the same types.
  - Pairs where both sides are out of limits on the same side are skipped. This is the NaN/Inf-like handling.
  - Shape mismatch throws.
  - Source: [ov_tensor_utils.cpp L336–660](https://github.com/openvinotoolkit/openvino/blob/master/src/tests/test_utils/common_test_utils/src/ov_tensor_utils.cpp); [ov_tensor_utils.hpp L116–128, L199–222](https://github.com/openvinotoolkit/openvino/blob/master/src/tests/test_utils/common_test_utils/include/common_test_utils/ov_tensor_utils.hpp)

**KernelBench (the LLM-kernel benchmark)**
- `get_tolerance_for_precision`: fp32 1e-4, fp16 1e-2, bf16 1e-2, used as both atol and rtol in `torch.allclose`. A comment says "kernelbench v0 and v0.1 uses fp32, atol = rtol = 1e-02". Correctness requires passing all `num_correct_trials` random-input trials. — [KernelBench src/kernelbench/eval.py L83–101, L798–806](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/eval.py)

### Inferences
- **Converting between metrics** (exact identities, plus the cosine formula assuming the error is uncorrelated with the signal; derived by me, not from a source). NMSE = relL2². SQNR_dB = −10·log10(NMSE) = −20·log10(relL2). cos ≈ 1/√(1+relL2²) ≈ 1 − relL2²/2. Equivalences:

  | Cosine | relL2 | SQNR |
  |---|---|---|
  | 0.99 | ≤ 0.142 | ≳ 16.9 dB |
  | 0.997 (Polygraphy default) | ≤ 0.078 | ≈ 22.2 dB |
  | 0.999 | ≤ 0.045 | ≈ 27 dB |

  | llama.cpp NMSE | SQNR |
  |---|---|
  | 1e-7 | 70 dB |
  | 5e-4 | 33 dB |
  | 2e-3 (`test_llm`) | 27 dB |

  So a "cos ≥ 0.99" gate is roughly 50 dB looser than llama.cpp's default same-precision gate.
- My pure-Python simulation (N=8192, N(0,1) values, IEEE fp16 rounding via `struct 'e'`) found that rounding a correct FP32 result to FP16 gives NMSE ≈ 4.4e-8 (73.5 dB) and 1-ulp noise on every element gives ≈ 5.2e-7. llama.cpp's default 1e-7 therefore leaves only ~2× headroom over the irreducible FP16 output rounding. That is consistent with llama.cpp relaxing to 1e-6 to 5e-5 whenever F16 intermediates are involved (Q3).
- Quantization-grade thresholds (cos 0.99, SNR 20 dB, PSNR 30 dB) are calibrated for int8/fp8/fp4 noise, which is 20–50 dB worse than fp16 rounding. Using them as the acceptance gate for FP16 kernels would accept many real bugs (see Q3 experiment). For an FP16 NPU kernel harness, a same-precision-calibrated tensor gate is more defensible: NMSE in the 1e-6 to 5e-4 range depending on reduction depth, or a ratio against a CPU-FP16 baseline as in PyTorch `same()`.

### Gaps
- SGLang kernel tests were not inspected.
- Ascend msprobe/msaccucmp exact defaults could not be retrieved (404 or empty page); only search snippets are available.
- Qualcomm's own gating thresholds in CI (the `psnr_threshold` values passed to `compare_psnr`) were not inspected.
- Intel NPU-plugin-specific per-test thresholds (in `src/plugins/intel_npu/tests`) were not inspected. Only OpenVINO's shared defaults were read.

---

## Q3. Blind spots of tensor-level metrics; recommended combinations; how llama.cpp `test-backend-ops` works

### Takeaway
Tensor-level aggregate metrics normalize error by total tensor energy, so errors confined to a small subset (a tail tile, one row, a few sign flips, small-magnitude elements) are diluted. Cosine is additionally blind to any multiplicative scale error (a wrong constant gives cos = 1.0). Mature harnesses never rely on one aggregate number. They combine:
- exact structural checks (shape, dtype, NaN/Inf, out-of-bounds sentinels);
- an energy metric whose threshold is scaled to the expected noise;
- per-row, per-region or worst-case guards;
- multiple random inputs.

llama.cpp's `test-backend-ops` is a compact, widely used example of this pattern for accelerator backends.

### Cited Findings

**How llama.cpp test-backend-ops works**
- Every backend under test is compared with the CPU backend, which is switched into its reference implementation (`ggml_backend_cpu_set_use_ref(backend_cpu, true)`, "Use reference implementation on the CPU backend for comparison"). — [test-backend-ops.cpp L13139–13151](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- Inputs are uniform random by default (`init_tensor_uniform(tensor, min=-1.0f, max=1.0f)`). Some ops constrain their inputs; for example, F16 exp/expm1 inputs are kept "in-range so all backends stay finite". — [test-backend-ops.cpp L54, L2125](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- Out-of-bounds write detection: `ggml_new_tensor` is wrapped "to add sentinels after each tensor to check for overflows in the backend". Each sentinel is 1024 floats. After the run, sentinels must be byte-identical (`memcmp`), otherwise the test fails with "sentinel mismatch". — [test-backend-ops.cpp L1299–1345, L1497–1508](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- Hard elementwise special-value checks run before the metric:
  - Any NaN in either output fails ("NaN at index").
  - Infinities: "both must be inf of the same sign, or both must be finite". FLT_MAX is accepted as infinity (`isinf_or_max`).
  - A sign mismatch or a one-sided inf fails.
  - Source: [test-backend-ops.cpp L469–472, L1513–1536](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- The metric is NMSE `= Σ(a−b)² / Σa²`, and the test fails if `err > max_err(backend)`. — [test-backend-ops.cpp L301–315, L1538–1546](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- Per-op thresholds (`max_nmse_err`). The default is 1e-7. Overrides, by group:
  - **Matmul, conv and attention family**
    - 5e-4: `test_mul_mat`, `test_mul_mat_id`, `test_out_prod`, `test_conv_2d`, `test_conv_1d_dw`, `test_conv_3d`, `test_conv_transpose_2d` ("The default 1e-7 is too small for Vulkan."), `test_flash_attn_ext`.
    - 2e-2 for MXFP4/NVFP4 on Blackwell native FP4 ("we quantize activations to mxfp4 instead of q8_1").
    - Generic ops: FLASH_ATTN_EXT gets `5e-4 * max(1.0, kv / 20000.0)` ("Scale error with kv length to account for accumulating floating point error").
    - `test_mul_mat_vec_fusion` 5e-3.
    - `test_llm` 2e-3.
  - **Softmax, norm/rope and fused elementwise chains**
    - SOFT_MAX 1e-6.
    - `test_rms_norm_mul_rope`: 5e-6 when ne0 = 8192, and 1e-5 for positions > 8192 ("large positions amplify the difference between the GPU and CPU trig functions").
    - F16 swiglu variants 5e-6 vs 1e-7 for F32 ("F16 epsilon ~ 9.7e-4: relax tolerance … roundoff drift from the exp()/clamp() chain computed in F16").
    - Fused ADDs 1e-6 for F16 and 1e-4 for BF16 ("Fused ADDs can keep FP32 intermediates while the CPU rounds each ADD to FP16/BF16").
    - `test_unary_mul` 5e-5 for F16 and 1e-4 for BF16.
    - `test_snake_fuse` 5e-3 for BF16 and 5e-5 for F16.
  - **Exact or near-exact checks**
    - `test_argmax`, `test_count_equal` and same-type `test_cpy` require 0.0.
    - `test_top_k` uses max_err 0.0 with a set-based comparison: "When there are no ties, we expect the exact same set of indices, but possibly in a different order".
  - **Backend-specific overrides**
    - Metal gelu_quick F32 is tightened to 1e-10.
    - WebGPU with F16 is relaxed to ≥1e-6.
    - Q8_0 `set_rows` on Qualcomm "HTP" (Hexagon NPU) ≥5e-6.
  - Source: [test-backend-ops.cpp L1214–1240, L2072–2420, L2722–2761, L2968–2976, L3023–3060, L3290–3313, L3549–3565, L4078–4430, L5343–5614, L6299, L6653–7026, L7220–7445, L7713, L7924, L8583, L9408–9432, L9662, L9788](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- Thresholds for quantized types are derived analytically from tensor size. `test_set_rows`/`test_cpy` "estimate what the max nmse error would be if one quantized value is off by one … roughly (2.0 / 2^bits)^2, divided by the mean square value of the reference, which is roughly 0.25 times the number of elements". The resulting threshold is the NMSE of a single off-by-one element, so it shrinks as 1/N. — [test-backend-ops.cpp L2722–2745, L3290–3313](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- Custom error functions exist for ops where NMSE is wrong:
  - `test_topk_moe` sorts both outputs before NMSE (order-insensitive).
  - `test_top_k_mask` takes the max abs diff over finite entries only.
  - `test_topk_qsa` uses a Jaccard set distance plus an integrality check.
  - `test_ssm_scan_rollback` uses a max-abs metric with 1e-6.
  - Source: [test-backend-ops.cpp L4841–4851, L7433–7445, L7563–7573, L9150–9160](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- Gradient mode checks finite-difference gradients with "mean absolute asymmetry" `|a−b|/(a+b)`. The threshold is `max_maa_err` 1e-4 by default and 1e-3 for sin, cos and rope. — [test-backend-ops.cpp L352–372, L1227–1229, L6024, L6067, L6529](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- The maintainers acknowledge that thresholds can be miscalibrated: "TODO: the max_nmse_err() for these cases is not estimated correctly causing sporadic false failures" (those cases are commented out). — [test-backend-ops.cpp L12269](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)

**Documented blind spots and guards elsewhere**
- AWS Neuron's module-test guidelines note several failure modes:
  - "per-element normalization amplifies noise", causing false failures on near-zero elements;
  - "Static thresholds can fail across different model weights and inputs because error profiles vary significantly";
  - an aggregate σ-ratio "can look good while one input is much worse".
  - Their fix is to add per-input worst-case L∞ and L2 ratio guards. — [AWS Neuron module test guidelines](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/accuracy/module_test_guidelines.html)
- vLLM's DeepGEMM MoE test uses the row-wise minimum cosine (`dim=-1).min() > 0.99`) rather than a global cosine, so a single corrupted row fails the test. — [test_deep_gemm_mega_moe.py](https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_deep_gemm_mega_moe.py)
- PyTorch `same()` guards cosine with an allclose early exit because "cosine_similarity(zeros(10), zeros(10), dim=0) is 0". — [torch/_dynamo/utils.py L3849–3856](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/utils.py)
- Polygraphy's PSNR uses `max(out0)`, not `max(|out0|)`, and falls back to a peak of 1.0 when `max_val <= 0`. For all-negative outputs, the "peak" is therefore not the actual dynamic range. — [compare.py L1520–1527](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/comparator/compare.py)
- "Measuring the Checker" (Du, Luu, Huang, Ng, 2026) injected 10,303 compilable faults into verified CUDA solutions of 188 KernelBench problems; 7,384 have an independent kill witness.
  - KernelBench's official check misses 16.9% of witnessed faults: 8.7% of arithmetic faults but 78.6% of "precision" faults.
  - The authors report a "tolerance blind band that grows with reduction size".
  - Two problems are "unrefereeable" because their official references violate the benchmark's own tolerance against fp64.
  - Source: [arXiv 2609.22220](https://arxiv.org/abs/2609.22220)
  - **[UNVERIFIED, search summary only]** An example in the paper: an all-zeros softmax output over ~393k elements passes because atol 1e-2 exceeds the signal, and removing softmax max-subtraction is unobservable with inputs drawn from [0,1).
- Reward hacking of correctness checks in LLM kernel benchmarks is documented:
  - KernelBench-Verified reports that models "hardcode bypasses for specific tensor values", and adds a four-distribution hidden test suite ([arXiv 2607.16241](https://arxiv.org/abs/2607.16241)).
  - CANN Bench lists "bypassing accuracy assertions, memorizing … caching outputs, or tampering with timing APIs", and adds an opt-in two-trial fresh-input mode that "perturbs every floating-point tensor by 0.01 to defeat seed reuse", plus a strict tensor type-identity check that "rejects FakeTensor" ([arXiv 2607.20518](https://arxiv.org/abs/2607.20518)).
  - **[UNVERIFIED, search summary]** Wafer reports a fused kernel claiming 104.89× that read garbage memory and passed defenses until a determinism guard was added ([Wafer blog](https://www.wafer.ai/blog/reward-hacking)).

### Inferences
**Own experiment (pure Python, IEEE fp16 via `struct 'e'`).** Reference: N=8192 iid N(0,1) values rounded to fp16. "Row" means 128 rows of 64. "frac>tol" is the fraction of elements failing the harness's `|d| > 1e-5 + 1e-3|ref|`.

| Perturbation | cosine | NMSE | SQNR dB | max-err/peak | frac>tol | min row-cosine |
|---|---|---|---|---|---|---|
| FP32 result vs FP16 ref (≤½-ulp) | 1.000000 | 4.4e-8 | 73.5 | 2.7e-4 | 0 | 1.000 |
| 1-ulp on every element | 1.000000 | 5.2e-7 | 62.8 | 5.5e-4 | 0 | 1.000 |
| 2-ulp on every element | 1.000000 | 2.0e-6 | 56.9 | – | **0.907** | – |
| FTZ of subnormals, N(0,0.01²) tensor (0.49% subnormal) | 1.0000000 | 6.1e-8 | 72.1 | – | **0.0042** | – |
| **Bug:** sign flip on 0.1% of elements | **0.99793** | 4.1e-3 | **23.8** | 0.90 | 0.001 | 0.92 |
| **Bug:** sign flip on 1% | 0.97593 | 4.8e-2 | 13.2 | 1.65 | 0.0099 | 0.75 |
| **Bug:** last 64-element row zero (tail tile, 0.78%) | **0.99616** | 7.7e-3 | **21.2** | 0.66 | 0.0078 | **0.000** |
| **Bug:** last element of each row dropped (off-by-one bound) | **0.99262** | 1.5e-2 | 18.3 | 0.80 | 0.0156 | 0.945 |
| **Bug:** wrong scale constant ×0.99 | **1.000000** | 1.0e-4 | **40.0** | 9.8e-3 | 0.999 | 1.000 |
| **Bug:** wrong scale constant ×0.9 | **1.000000** | 1.0e-2 | 20.0 | 0.10 | 1.0 | 1.000 |
| **Bug:** sign flip only where \|x\|<0.1 (8% of elems) | **0.99946** | 1.1e-3 | **29.7** | 0.056 | 0.081 | 0.998 |
| **Bug:** off-by-one sliding window (16 vs 17 terms) | 0.97146 | 6.0e-2 | 12.2 | 0.25 | 0.996 | 0.88 |
| **Bug:** single NaN | NaN | NaN | NaN | (naive max → 0) | **0 (naive `>`)** | (naive min → 1) |

Readings from the table:
- **Cosine ≥ 0.99 passes several real bugs.** It accepts 0.1% sign flips, a dropped tail tile, an off-by-one row bound, sign errors on small-magnitude elements, and every multiplicative constant error. Polygraphy's 0.997 still accepts the 0.1% sign flip, the small-value sign flip and the scale errors.
- **Closed forms behind these numbers** (iid data): flipping the sign of a fraction f of elements gives cos = 1 − 2f, so 0.99 tolerates f ≤ 0.5% and 0.997 tolerates f ≤ 0.15%. Zeroing a fraction f gives cos ≈ √(1−f), so 0.99 tolerates ~2% of elements zeroed.
- **SQNR ≥ 20 dB** (Polygraphy SNR default, torchao) passes the 0.1% sign flip, the tail-tile bug, the ×0.99 constant, and sits right at the boundary for ×0.9.
- **llama.cpp's NMSE gates catch most of these bugs.** The 1e-7 default fails every bug in the table. The 5e-4 matmul gate passes the ×0.99 scale error (NMSE 1e-4) but fails the others.
- **The current elementwise gate catches every bug in the table,** apart from NaN when written naively. It also fails the benign cases: FTZ (0.42% of elements) and 2-ulp noise (91% of elements). rtol=1e-3 is ≈ 1 fp16 ulp (2^-10 = 9.77e-4 at the bottom of a binade), so any kernel with 2-ulp accumulation drift fails. atol=1e-5 is below the smallest fp16 normal (6.10e-5), so flush-to-zero of any subnormal output fails.
- **NaN handling is an implementation trap.** `err > tol` is False for NaN, and Python/NumPy-style naive reductions can drop NaN. NaN/Inf must be checked explicitly, as llama.cpp, CANN Bench and Polygraphy do (`_simple_stat_failed` treats NaN as failure).
- **Localization metrics catch what global ones miss.** The min row-cosine (vLLM style) drops to 0 for the tail-tile bug, which global cosine accepts.
- **Scale-invariance is the main structural blind spot of cosine.** Pair it with a magnitude-sensitive metric: relL2/NMSE, max-abs/peak, or an allclose with relaxed tolerances.

**Recommended combination** (synthesized from llama.cpp, PyTorch `same()`, AWS Neuron and CANN Bench):
1. Exact structural checks: shape, dtype, strides/layout. Any NaN, and any Inf pattern or sign mismatch, is a hard failure. Add canary or sentinel buffers to catch out-of-bounds writes.
2. A magnitude-aware aggregate whose threshold is calibrated to the expected FP16 noise for that op's reduction depth: NMSE or relL2, or better, a ratio against a same-precision baseline.
3. A worst-case or localized guard: max-abs/peak, per-row or per-tile max error or cosine, or a count of elements over a loose bound.
4. Several input draws from several distributions.

Cosine at most as a diagnostic.

### Gaps
- No primary source found that systematically measures cosine-similarity false negatives on real kernel bugs. The figures above come from my own simulation, not a published study. "Measuring the Checker" measures allclose, not cosine.
- Whether llama.cpp has an OpenVINO/Intel NPU backend exercised by test-backend-ops was not verified. The file names only the "HTP" (Hexagon), WebGPU, Metal ("MTL") and Vulkan special cases.

---

## Q4. Quantile and error-ratio criteria, and how they trade false positives against false negatives

### Takeaway
Most toolchains offer some form of "allow a bounded fraction or quantile of elements to exceed the elementwise tolerance":
- Polygraphy: `--check-error-stat quantile|mean|median`, default quantile 0.99.
- coremltools: `allow_mismatch_ratio`.
- OpenVINO: `topk_threshold`, the fraction of incorrect values.
- CANN Bench: an ErrorCount ratio against a same-precision CPU run.

The most principled variant compares the count of bad elements with the count produced by a same-precision CPU reference (e.g. NPU count ≤ 2 × CPU count, as in CANN Bench). This cancels the dtype's inherent error while still bounding kernel-induced error.

### Cited Findings
- Polygraphy `check_error_stat` options:
  - "elemwise" (default): "Checks each element … exceeds both tolerances".
  - "max": "the strictest possible check".
  - "mean", "median".
  - "quantile": "Checks the quantile absolute/relative errors against the respective tolerances".
  - `error_quantile` "Defaults to 0.99".
  - Statistics with NaN fail (`_simple_stat_failed: isnan(diff) or diff > tol`).
  - `infinities_compare_equal` (default False) makes matching ±inf count as 0 difference; otherwise they count as NaN.
  - Zeros in the reference get ε = 2.22e-16 for relative error, which "may cause some relative differences to be extremely high".
  - In elemwise mode an element is a mismatch only if it exceeds both atol and rtol.
  - Source: [compare.py L649–655, L849–892, L955–1060](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/comparator/compare.py)
- coremltools `allow_mismatch_ratio`: "Allow a ratio of elements to be out of tolerance of atol and rtol. Mainly used for comparing compressed models outputs". The test raises only if `mismatch_num/total_num > allow_mismatch_ratio`. — [testing_utils.py L428–457](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/testing_utils.py)
- OpenVINO `compare()`:
  - `topk_threshold` is the "percentage of incorrect values in tensor [0,1]". The test fails if the incorrect count exceeds `floor(topk_threshold*size)`. At the default `topk_threshold = 1.f`, any incorrect value fails, so the default is strict.
  - `mvn_threshold` is the "avg value of `diff/threshold`. Shows tensor jitter relative to threshold", default 1.0. This is a mean-normalized-error gate that runs in addition to the count gate.
  - Source: [ov_tensor_utils.hpp L199–215](https://github.com/openvinotoolkit/openvino/blob/master/src/tests/test_utils/common_test_utils/include/common_test_utils/ov_tensor_utils.hpp); [ov_tensor_utils.cpp L456–516](https://github.com/openvinotoolkit/openvino/blob/master/src/tests/test_utils/common_test_utils/src/ov_tensor_utils.cpp)
- CANN Bench correctness gate (Gao et al., 2026; agent-generated kernels on real Ascend NPU). Source for everything below: [arXiv 2607.20518](https://arxiv.org/abs/2607.20518) §3, Tables 4, 6, 7, Appendix A.
  - **First screen (normal values).**
    - Elementwise relative error is `e_i = |ŷ_i − y_i| / (|y_i| + 1e-7)`. Positions with ±∞ or NaN are excluded from this statistic.
    - Pass if `mean_i e_i < τ_d` AND `max_i e_i < 10·τ_d`. The max-RE bound is "ten times looser than the mean-RE bound so that an isolated outlier on a small subset of positions does not by itself fail an otherwise-correct tensor".
    - τ_d values (powers of two anchored to mantissa width): FP16 2^-10 (9.77e-4, so max 9.77e-3), BF16 2^-7 (7.81e-3 / 7.81e-2), FP32 2^-13 (1.22e-4 / 1.22e-3), HiF32 2^-11, FP8 E4M3 2^-3, FP8 E5M2 2^-2. Integers use exact equality by default.
  - **Region fallback.** When the first screen fails, failing positions are attributed to normal, small-value or cancellation regions. Only failures confined to the two "numerically unstable regions" can still pass, via an ErrorCount ratio `ErrorCount_NPU / max(ErrorCount_CPU, 1) ≤ 2`, applied separately per region. The CPU count comes from "a same-precision CPU run of the Golden", and the paper explains: "errors attributable to the dtype itself appear on both sides of the ratio and cancel out, so only kernel-induced contributions survive". Any failure in the normal region is fatal.
    - **Small-value region:** `|y_trunc| < τ_sv`, where y_trunc is the FP64 golden cast to the device dtype and back. Inside it, an element is an error if `|ŷ − y_trunc| > ε_sv`. FP16: τ_sv = 2^-11, ε_sv = 2^-16. BF16: 2^-8, 2^-16. FP32: 2^-14, 2^-30. Rationale: "the denominator |y_i| + 10^-7 … amplifies a single-ULP quantization difference into a spuriously large relative error even when the kernel is computing correctly".
    - **Cancellation region:** `|ŷ| < τ_cz` and `τ_sv ≤ |y| < τ_cb`. FP16 τ_cb = τ_cz = 2^-5; BF16 2^-3; FP32 2^-8. An element counts as an error if its RE exceeds 10·τ_d. Rationale: "catastrophic cancellation (Kahan) wipes out the leading significant bits … elementwise relative error against the FP64 Golden is arbitrarily large — not because the arithmetic is wrong".
    - My text extraction of Tables 6/7 garbled the superscripts, so these values should be re-checked against the PDF.
  - **Per-operator relaxation** is allowed through the operator's `proto.yaml`. The standard "recommends capping floating-point relaxation at 10× the default", and "a relaxed pass remains distinguishable from a strict one".
  - **Reference:** by default the golden runs on CPU after upcasting inputs to FP64, then "the Golden output is cast to the kernel output's dtype" before RE is computed.
  - **Special values:** NaN positions must match. Matching ±∞ are excluded. An ∞ sign mismatch is a hard fail. A one-sided ∞ is replaced by the dtype max finite value and then gated normally ("saturation is not a blanket pass").
- AWS Neuron's three-way criterion is a ratio-style gate against a dtype-error reference (a CPU BF16 run):
  - `σ-ratio = RMS(all target errors)/RMS(all reference errors)`, aggregated over N=10 inputs.
  - The Bhattacharyya coefficient (BC) measures the overlap of the two error distributions.
  - `worst L-inf ratio = max_i[max|Neuron_err_i| / max|Ref_err_i|]` < 5.0 (default `max_linf_ratio`).
  - `worst L2 ratio` < 3.0 (default `max_l2_ratio`).
  - PASS if `(BC >= bc_threshold OR σ-ratio <= 1.0) AND worst L-inf ratio < 5.0 AND worst L2 ratio < 3.0`.
  - BC bands: "≥ 0.99 … production ready, 0.95 to 0.99 … marginal, < 0.95 … likely a bug".
  - Source: [AWS Neuron module test guidelines](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/accuracy/module_test_guidelines.html)

### Inferences
- Trade-off: a fixed fraction allowance r (coremltools/OpenVINO style) removes false rejects caused by a few FTZ or cancellation elements. It also creates a false-accept window for any bug that touches fewer than r·N elements. A tail tile or a single row is often <1% of a tensor (0.78% in the Q3 experiment). With r = 1%, a dropped tail tile would pass. The quantile criterion has the same blind spot: Polygraphy quantile 0.99 ignores the worst 1% of elements.
- The two ways to keep the allowance without the blind spot:
  1. **Restrict where exceedances may occur.** CANN Bench allows them only in small-value and cancellation regions and requires zero failures in the normal region.
  2. **Calibrate the allowance per input against a same-precision reference.** CANN Bench uses NPU count ≤ 2 × CPU count; AWS Neuron uses worst-ratio guards.
  
  Both address the current harness's three false-reject causes (subnormal FTZ, cancellation, ~1-ulp rounding) without accepting localized bugs in normal-magnitude outputs.
- For the Intel NPU harness, a concrete adaptation would be:
  - mean-RE < 2^-10 and max-RE < 10·2^-10 on normal-region elements;
  - an FTZ-aware small-value rule (|y_trunc| < 2^-14 or 2^-11, with absolute tolerance on the order of 2^-16 or the subnormal spacing; output 0 accepted for subnormal references);
  - a cancellation-region count ratio against an emulated-FP16 CPU run;
  - a per-row or per-tile worst-case guard.

### Gaps
- No published ROC-style study was found that compares quantile thresholds (e.g. 0.99 vs 0.999) by false-accept and false-reject rate for kernel bugs.
- The CANN Bench small-value and cancellation table values were extracted from the PDF text stream with partly garbled superscripts and need confirmation.

---

## Q5. Hardware-in-the-loop and differential testing; multiple references; mutation testing of verdicts

### Takeaway
State-of-the-art accelerator harnesses compare the device not just against one reference but against two:
- a high-precision golden (FP64/FP32), the "truth";
- a same-precision reference (CPU FP16/BF16 emulation, or the eager low-precision run), which measures the error the dtype itself causes.

The device passes if its error relative to the golden is within a small multiple (2–5×) of the same-precision reference's error. Mutation analysis (injecting known bugs and measuring how many the verdict detects) is emerging as the way to measure the verdict itself. It shows that loose fixed tolerances miss a large share of precision-type bugs.

### Cited Findings
- PyTorch `same()` is a two-reference rule: the compiled result `res` passes if `RMSE(fp64_ref, res) <= multiplier × RMSE(fp64_ref, ref_lowp) + tol/10`, with multiplier 2–3 (up to 8–10 for tiny tensors). — [torch/_dynamo/utils.py L3868–3936](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/utils.py)
- AWS Neuron frames the question as "Is this error from Neuron, or just from BF16?". It uses an FP32 golden, a BF16 CPU "dtype-error reference" and the BF16 device output as the target. Error₂ "captures the expected precision loss for the specific weights and inputs being tested". Its two-way (device vs FP32) check with `rtol=atol=1e-2` is "informational only" and does not gate. Only the three-way check gates. — [AWS Neuron module test guidelines](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/accuracy/module_test_guidelines.html)
- CANN Bench: FP64 CPU golden plus a same-precision CPU run used for the region ErrorCount ratio (≤ 2). It also supports "native-precision CPU and NPU Golden modes for specialized benchmark integrations". — [arXiv 2607.20518](https://arxiv.org/abs/2607.20518)
- Ascend msprobe's pre-check runs on both NPU and GPU against a CPU high-precision benchmark, which yields NPU-vs-CPU and GPU-vs-CPU results for comparison **[UNVERIFIED: search summary]**. — [msprobe README](https://gitcode.com/Ascend/msprobe/blob/master/README_EN.md)
- llama.cpp differential-tests each backend (CUDA, Metal, Vulkan, WebGPU, Hexagon HTP and others) against the CPU reference implementation, op by op, plus whole-graph fusion checks (`run_whole_graph`, `fusion_test_nodes`). — [test-backend-ops.cpp L1548–1556, L13139–13151](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)
- CRADLE (Pham, Lutellier, Qi, Tan, ICSE 2019) does cross-implementation inconsistency checking across TensorFlow, CNTK and Theano to sidestep the oracle problem. It detected 12 bugs and 104 unique inconsistencies over 30 pre-trained models and 11 datasets, and localizes faults by anomaly propagation. — [CRADLE paper](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf) (abstract-level details from the [ICSE 2019 page](https://2019.icse-conferences.org/event/icse-2019-technical-papers-cradle-cross-backend-validation-to-detect-and-localize-bugs-in-deep-learning-libraries))
- "The Correctness Illusion in LLM-Generated GPU Kernels" (Sarkar, 2026) argues that KernelBench, TritonBench and GEAK use "fixed-shape, small-sample allclose-style" checks. It re-checks kernels with op-schema-aware seeded fuzzing against an fp64 CPU reference with per-(op, dtype) absolute tolerances:
  - 9/9 seeded-bug kernels flagged and 15/15 correct controls passed;
  - in the extended corpus, 10/10 "illusions" caught and 16/16 controls clean, with identical verdicts on five GPUs (RTX 3060, A10, L40S, A100, H100);
  - every failure replays from a stored seed.
  - Source: [arXiv 2606.20128](https://arxiv.org/abs/2606.20128)
- "Measuring the Checker" (2026) applies mutation analysis to the oracle:
  - 10,303 injected faults, 124 mutation rules ([KernelBench-M dataset](https://huggingface.co/datasets/Elfsong/KernelBench-M), **[UNVERIFIED count from search summary]**);
  - 16.9% of witnessed faults missed by KernelBench's official check, with 78.6% of precision faults escaping;
  - KernelBench-Verified's improvement decomposes into "+4.0 points from hidden inputs and +4.5 from tighter tolerance";
  - "a published fuzzing recipe rejects correct kernels 107 times" (false rejects);
  - kill-matrix-optimized suites reach "98.0% detection with two inputs per problem (94.8% on held-out data)";
  - a "measured ceiling on input aggressiveness is set by legitimate floating-point variance".
  - Source: [arXiv 2609.22220](https://arxiv.org/abs/2609.22220)
- Older mutation-sensitivity work for numerical software compares the error of the implementation under test with the error of mutated versions of itself, which avoids needing exact oracles (Wu & Mitchell 2019; Hook 2009 thesis) **[UNVERIFIED: search summary]**. — [Wu & Mitchell 2019](https://www.cs.toronto.edu/~rwu/papers/wu2019mutation.pdf)

### Inferences
- For the Intel NPU 3720 (FP16 compute), the two-reference design maps naturally onto:
  - Ref A: an FP64 or FP32 CPU golden;
  - Ref B: a same-precision emulation, i.e. NumPy float16 evaluated step by step with the kernel's accumulation order where known, or OpenVINO CPU with `inference_precision=f16`;
  - the device output.
  
  Gate on `err(dev, A) ≤ k·err(B, A) + floor` with k ≈ 2–3 for RMS (PyTorch) and ≈ 5 for worst-case (AWS Neuron L∞). Add a per-row or per-tile worst-case version so localized bugs cannot hide in the aggregate.
- Caveat: Ref B must model the device's actual FTZ behaviour and accumulator precision. Otherwise "agreement" just measures emulation fidelity. If the NPU accumulates in FP32 internally while Ref B accumulates in FP16, Ref B overestimates the inherent error and loosens the gate. Use the tighter of the two emulations, or emulate both and take the minimum.
- **Mutation testing as harness calibration.**
  1. Build a small mutant suite for each kernel family: sign flip on a subset, off-by-one loop bound, dropped tail tile, wrong scale constant, missing max-subtraction in softmax, wrong epsilon, transposed index, wrong causal-mask offset.
  2. Report the kill rate.
  3. Measure the false-reject rate on known-good kernels, e.g. OpenVINO's own NPU implementation of the same op, or the CPU-FP16 path run through the device verdict.
  
  "Measuring the Checker" shows both numbers are needed: a fuzzing recipe that kills more mutants also rejected correct kernels 107 times.
- Inputs should be diverse to beat input-specific hardcoding and boundary blindness: several distributions (uniform [-1,1], large-magnitude, near-zero/subnormal, all-positive, a mix that triggers cancellation), non-aligned shapes (tail tiles), and fresh seeds per trial. CANN Bench's 512 B alignment split is an example of a shape-class axis.

### Gaps
- No published hardware-in-the-loop study specific to Intel NPU (VPU 37xx) kernel verification was found.
- The internal details of "Measuring the Checker" (per-family kill rates beyond arithmetic and precision, recommended tolerances) were not read beyond the abstract.
- The tolerances used in "The Correctness Illusion" were not available from the abstract.

---

## Q6. Choosing thresholds as a function of dtype (fp16 / bf16 / int8 / fp8) and tensor size

### Takeaway
Dtype-dependent defaults are converging on "tolerance ≈ machine epsilon of the lowest-precision type involved":
- fp16: rtol ≈ 1e-3 (torch, TE, OpenVINO), or 2^-10 mean / 10·2^-10 max RE (CANN Bench);
- bf16: rtol ≈ 1.6e-2 / 1e-2, or 2^-7;
- fp8 e4m3: ≈ 0.125.

Size dependence is handled in three ways:
- scaling with reduction length: llama.cpp flash-attention 5e-4·max(1, kv/20000); "blind band grows with reduction size";
- looser multipliers for tiny tensors, where noise dominates: PyTorch 3–10×;
- NMSE thresholds derived so that a single wrong element in an N-element tensor is detectable: llama.cpp's 1/N quantization estimate.

### Cited Findings
- Per-dtype elementwise defaults:
  - torch/TE: fp16 rtol 1e-3, atol 1e-5; bf16 rtol 1.6e-2, atol 1e-5; fp32 rtol 1.3e-6, atol 1e-5; fp8 e4m3 rtol 0.125, atol 0.0675; e5m2 / fp4 rtol 0.25, atol 0.125. — [TE tests/pytorch/utils.py](https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/utils.py) (states these are "Based on tolerances for torch.testing.assert_close"; see also [PyTorch testing docs](https://docs.pytorch.org/docs/stable/testing.html), not fetched this session)
  - OpenVINO default rel thresholds: f16 1e-3, bf16 1e-2, f32 1e-4, f64 1e-8, nf4 1e-1. The maximum over expected, actual and inference-precision types is used, so an f32 test run with f16 inference precision automatically gets 1e-3. — [ov_tensor_utils.hpp L116–128](https://github.com/openvinotoolkit/openvino/blob/master/src/tests/test_utils/common_test_utils/include/common_test_utils/ov_tensor_utils.hpp); [ov_tensor_utils.cpp L525–575](https://github.com/openvinotoolkit/openvino/blob/master/src/tests/test_utils/common_test_utils/src/ov_tensor_utils.cpp)
  - KernelBench: fp32 1e-4, fp16/bf16 1e-2. The comment says "torchbench states for bf16 and fp16, use 1e-3 as tolerance and 1e-2 if it's too strict". — [KernelBench eval.py](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/eval.py)
  - CANN Bench relative-error thresholds (2^-10 for FP16, 2^-7 for BF16, 2^-13 for FP32, 2^-3/2^-2 for FP8), "anchored to each format's mantissa width". — [arXiv 2607.20518](https://arxiv.org/abs/2607.20518)
- Tensor-level thresholds by dtype and format: torchao SQNR ≈ 25–35+ dB for fp8/int8 and ≈ 12–20 dB for fp4/MX formats; vLLM cosine 0.999 for bf16 GEMM vs 0.99 for NVFP4 MoE; llama.cpp NMSE 1e-7 for fp32 elementwise, 5e-6/5e-5 for fp16 chains, up to 1e-4/5e-3 for bf16 chains. — sources in Q2/Q3.
- Size and reduction-depth dependence:
  - llama.cpp FLASH_ATTN_EXT: `5e-4 * max(1.0, kv / 20000.0)` ([test-backend-ops.cpp L9424–9429](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)).
  - llama.cpp quantized-copy threshold ∝ 1/(number of elements), "if one quantized value is off by one" ([test-backend-ops.cpp L2726–2745, L3296–3313](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp)).
  - PyTorch `same()` multiplier: 10× for numel ≤ 10, 8× for numel < 1000 (opt-in), 3× for numel < 1000 by default, "In the presence of noise, noise might dominate our error metric for smaller tensors" ([torch/_dynamo/utils.py L3899–3932](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/utils.py)).
  - Tolerance "blind band that grows with reduction size" ([arXiv 2609.22220](https://arxiv.org/abs/2609.22220)).
  - AWS Neuron gives indicative relative-error thresholds by scope: "full model (~0.01), 4 layers (~0.005), single op (~1e-5)". BF16 MatMul P99 error is 0.40–0.49%, and attention error grows with weight range (safe range 0.01–0.20). — [AWS Neuron module test guidelines](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/accuracy/module_test_guidelines.html)
- Rounding-error theory **[UNVERIFIED in-session; background knowledge]**: for a length-n dot product in precision u (fp16 u = 2^-11 ≈ 4.9e-4), the classical worst-case relative bound is γ_n = nu/(1−nu). Higham and Mary's probabilistic analysis gives a bound of order √n·u that holds with high probability for random rounding errors, which justifies scaling tolerances roughly with √K for K-length reductions. — [Higham & Mary, "A New Approach to Probabilistic Rounding Error Analysis", SIAM J. Sci. Comput. 2019](https://doi.org/10.1137/18M1226312)

### Inferences
- For FP16 NPU outputs of reduction-type kernels (matmul, conv, softmax, layer-norm), a fixed rtol = 1e-3 (≈1 fp16 ulp) is tighter than what a correct kernel with a different accumulation order or FP16 intermediates can guarantee once K exceeds a few elements. A dtype- and depth-aware rule such as `atol ≈ c·u_fp16·√K·max|ref|` (normalized to the tensor peak, as AWS Neuron and Qualcomm PSNR do), plus a ratio check against a same-precision CPU run, is better grounded in error theory and in practice.
- For elementwise (non-reduction) kernels, 1–2 ulp is achievable and a strict gate is reasonable. The gate needs FTZ-aware handling: treat |ref| < 2^-14 as matching 0, or use CANN-style small-value absolute tolerances. Cancellation-aware handling is needed only for ops that subtract.
- Two calibration tips:
  1. Calibrate thresholds empirically by running known-good implementations (OpenVINO's built-in NPU ops, CPU-FP16 emulation) over the same input distributions and setting gates at a margin above the observed distribution.
  2. Then validate with mutants (Q5), so the threshold is tied to both false-reject and false-accept rates rather than picked from convention.

### Gaps
- No vendor document was found that publishes an explicit formula for tolerance as a function of reduction length for FP16 NPUs. llama.cpp's kv/20000 scaling and PyTorch's numel-based multipliers are empirical.
- int8-specific thresholds for integer-output kernels (as opposed to dequantized outputs) are mostly "exact equality" (CANN Bench, llama.cpp same-type copy). No source gave a looser integer tolerance standard beyond CANN Bench's "operator-declared" bounded tolerance.
