# Numerical tolerance conventions for low-precision (FP16/BF16) kernel correctness in major ML frameworks, compilers and test suites

Scope and method: almost every number below was read directly from primary source code fetched from GitHub on 2026-10-22, not taken from docs or blogs. Links are pinned to the HEAD commit at fetch time: pytorch `f9a5239`, jax `d3e85eb`, openxla/xla `8476ae4`, apache/tvm `18ec45e` (plus tag v0.27.0 and apache/tvm-tirx `main`), iree `c6882b8`, onnxruntime `fe7700a`, openvino `86fc05b`, openvinotoolkit/npu_compiler `20e7912` (develop, 2026-10-12), NVIDIA/TensorRT `16ed063` (2026-10-09). Line numbers refer to those commits. Notation: `a` = actual/device output, `e` = expected/reference.

## Q0. Cross-framework summary: default tolerances per dtype and the comparison formula

### Takeaway
Most suites use the additive NumPy rule `|a−e| ≤ atol + rtol·|e|`. XLA and Polygraphy instead fail an element only when it exceeds both tolerances, which is the same as `|a−e| ≤ max(atol, rtol·|e|)`. For FP16 the default atol in JAX, IREE, ONNX Runtime, OpenVINO, the NPU layer tests and XLA's half-precision tests is about 1e-3 or larger. PyTorch's `assert_close` (atol 1e-5) and Polygraphy (1e-5/1e-5) are the strict outliers, and those are exactly the defaults our verdict copies.

### Cited Findings

| Framework / tool | fp32 (rtol, atol) | fp16 (rtol, atol) | bf16 (rtol, atol) | Formula / semantics | Source |
|---|---|---|---|---|---|
| PyTorch `torch.testing.assert_close` | 1.3e-6, 1e-5 | 1e-3, 1e-5 | 1.6e-2, 1e-5 | `torch.isclose`: `\|a−e\| ≤ atol + rtol·\|e\|`; quantized types use fp32 tolerances on dequantized values | [_comparison.py L96-112, L1126](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/testing/_comparison.py#L96-L112) |
| PyTorch dynamo/inductor `same()` (benchmark `--accuracy`) | default `tol=1e-4` (used as both rtol and atol) | benchmark sets tol 1e-3 for fp16/AMP (1e-2 or 8e-2 for allow-listed models) | 1e-3, or 1e-2/8e-2 for allow-listed models | First `allclose(tol,tol)`. If that fails, pass when `RMSE(res, fp64) ≤ m·RMSE(eager, fp64) + tol/10`, with m=2 (fp32) or 3 (fp16/bf16) | [utils.py L3688-3960](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/_dynamo/utils.py#L3688-L3960) |
| JAX `jtu.assertAllClose` | 1e-6 (atol = rtol) | 1e-3 | 1e-2 | `np.testing.assert_allclose` (additive), bf16 upcast to fp32 first | [public_test_util.py L46-76, L110-147](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/public_test_util.py#L46-L147) |
| JAX gradient checks (`check_grads`) | 2e-3 | 1e-2 | 1e-1 | atol and rtol are multiplied by the tensor size | [public_test_util.py L88-104, L159-165](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/public_test_util.py#L88-L165) |
| XLA `ErrorSpec` (literal comparison) | no global default; `dot_operation_test` uses `{abs 1e-4, rel 1e-5}` | dot tests on `Eigen::half` use `{1e-3, 3e-3}` or `{1e-4, 1e-3}` | per test | Mismatch only if `abs_err > abs` **and** `rel_err > rel`. Optional ULP mode `within_n_values` | [error_spec.h](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/error_spec.h#L24-L71), [literal_comparison.cc L627-705](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/literal_comparison.cc#L627-L705) |
| TVM `tvm.testing.assert_allclose` (v0.27.0; moved to `tvm_tirx.testing` on main) | 1e-7, 1e-7 (dtype-independent) | same default; fp16 tests usually pass 1e-2/1e-2 | — | `np.testing.assert_allclose` (additive, reference = `desired`) | [v0.27.0 utils.py L113-124](https://github.com/apache/tvm/blob/v0.27.0/python/tvm/testing/utils.py#L113-L124), [tvm-tirx utils.py L34-45](https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/testing/utils.py#L34-L45) |
| IREE `iree-run-module --expected_output` | atol 1e-4, rtol 1e-5 | atol 1e-3 (`--expected_f16_threshold`), rtol 1e-5 | atol 1e-2 (`--expected_bf16_threshold`), rtol 1e-5 | Since May 2026: `\|a−e\| ≤ atol + rtol·\|e\|`, NaN == NaN | [comparison.cc L21-35](https://github.com/iree-org/iree/blob/c6882b84d0afece9ea2d810ccc976997ea0d199b/runtime/src/iree/tooling/comparison.cc#L21-L35) |
| IREE e2e matmul/conv tests | bit-exact first, then `\|a−e\| < 1e-5` | same | same, plus `<10` absolute when \|x\|>127 | Inputs are small integers, so results are exactly representable | [test_utils.c L33-34, L258-294](https://github.com/iree-org/iree/blob/c6882b84d0afece9ea2d810ccc976997ea0d199b/tools/testing/e2e/test_utils.c#L258-L294) |
| ONNX Runtime `OpTester` | rtol 1e-4, atol 1e-5 (training build atol 1e-3; OpenVINO-NPU build atol 5e-3) | rtol 1e-3, atol 2.5e-3 (training 5e-3; DML EP 5e-3) | rtol 1e-2, atol 2e-2 | `EXPECT_NEAR(e, a, atol + rtol·\|e\|)`, NaN must match NaN, Inf must be equal | [checkers.cc L20-123, L495-520](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/unittest_util/checkers.cc#L20-L123) |
| ONNX Runtime `onnx_test_runner` | defaults from `onnx_backend_test_series_overrides.jsonc`: rtol 1e-3, atol 1e-5 (CLI help: 1e-5/1e-5) | same, with per-test overrides (e.g. `test_attention_4d_fp16` atol 5e-4) | — | `diff ≤ atol + rtol·\|e\|`, NaN/Inf must match | [overrides.jsonc](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc#L1-L29), [compare_ortvalue.cc L135-147, L299](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/util/compare_ortvalue.cc#L135-L147) |
| OpenVINO `ov::test::utils::compare` (all plugins) | abs = fp32 machine ε (~1.19e-7), rel 1e-4 | abs = f16 machine ε (2^-10 ≈ 9.77e-4), rel 1e-3 | abs = bf16 ε (2^-7 ≈ 7.8e-3), rel 1e-2 | `\|a−e\| ≤ abs + rel·\|e\|`. Defaults take the max over the expected, actual and `inference_precision` types. Optional `topk_threshold` (fraction of elements allowed to fail) and `mvn_threshold` (mean of diff/threshold) | [ov_tensor_utils.cpp L436-611](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/test_utils/common_test_utils/src/ov_tensor_utils.cpp#L436-L611), [ov_tensor_utils.hpp L116-131](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/test_utils/common_test_utils/include/common_test_utils/ov_tensor_utils.hpp#L116-L131) |
| Intel NPU layer tests (`npu_compiler` `VpuOv2LayerTest`) | base abs 0.01; per-test overrides | base abs 0.01; conv rel 0.02; softmax abs 1e-3 to 1e-2; eltwise NPU3720 SW abs 0.6 | per test | `\|a−e\| ≤ abs + rel·\|e\|`. Threshold value −1 is treated as 0. Inf vs fp16-max overflow is forgiven | [vpu_ov2_layer_test.cpp L36, L303-325](https://github.com/openvinotoolkit/npu_compiler/blob/20e7912fbc230725e1ca7394762348d4f480aa65/tests/functional/shared_tests_instances/vpu_ov2_layer_test.cpp#L36), [tensor_comparison.cpp L66-185](https://github.com/openvinotoolkit/npu_compiler/blob/20e7912fbc230725e1ca7394762348d4f480aa65/tests/functional/common/tensor_comparison.cpp#L66-L185) |
| OpenVINO NPUW accuracy check (`NPUW_ACC_CHECK`) | NRMSE ≤ 0.01 | same | same | `NRMSE = RMSE / max(0.001, range(ref), range(actual))`, computed in fp32 | [comparator.cpp](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/plugins/intel_npu/src/plugin/npuw/accuracy/comparator.cpp#L15-L66), [npuw_option_defs.inc L46-48](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/plugins/intel_npu/src/al/include/intel_npu/config/npuw_option_defs.inc#L46-L48) |
| TensorRT Polygraphy `SimpleCompareFunc` / `polygraphy run` | rtol 1e-5, atol 1e-5 (all dtypes) | same; docs say relax for reduced precision | same | `elemwise` (default): mismatch iff `absdiff > atol` **and** `reldiff > rtol`. Also `max`/`mean`/`median`/`quantile` (default quantile 0.99). Values are upcast before diffing | [compare.py L799-802, L955-1025](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/tools/Polygraphy/polygraphy/comparator/compare.py#L799-L1025) |

- History: PyTorch's `_DTYPE_PRECISIONS` table is identical in v1.10.0 (`torch/testing/_asserts.py`), v2.0.0 and main (2026-10), so it has not changed since 2021 — [v1.10.0 _asserts.py](https://github.com/pytorch/pytorch/blob/v1.10.0/torch/testing/_asserts.py); [main _comparison.py L96-104](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/testing/_comparison.py#L96-L104)
- History: JAX's fp16, bf16 and fp32 defaults (1e-3, 1e-2, 1e-6; gradient 1e-2, 1e-1, 2e-3) are the same at tag jax-v0.4.1 and at main — [jax-v0.4.1 public_test_util.py](https://github.com/jax-ml/jax/blob/jax-v0.4.1/jax/_src/public_test_util.py); [main L46-104](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/public_test_util.py#L46-L104)
- History: IREE PR #24394 (merged 2026-05-08) folded the separate absolute and relative modes into one numpy-style `|a−b| ≤ atol + rtol·|b|` "approximate" mode. The PR text says the old split modes "took atol path by default" and gave false signals in onnx_op tests. PR #19464 (2025-04-30) had added the relative mode, and PR #22547 (2026-04-08) added `--expected_bf16_threshold` — [IREE #24394](https://github.com/iree-org/iree/pull/24394); [#19464](https://github.com/iree-org/iree/pull/19464); [#22547](https://github.com/iree-org/iree/pull/22547)
- History: on TVM main, `python/tvm/testing/utils.py` was removed in the 2026-10-15 refactor "[REFACTOR] Bringup tirx in standalone folder (#20742)". `assert_allclose` now lives in `tvm_tirx.testing` with the same 1e-7/1e-7 defaults — [tvm main testing/__init__.py](https://github.com/apache/tvm/blob/18ec45ed7729b70ae94c340014624b34b6ecf5a7/python/tvm/testing/__init__.py); [tvm-tirx utils.py L34](https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/testing/utils.py#L34-L45)

### Inferences
- Our verdict (`rtol=1e-3, atol=1e-5`) is PyTorch's fp16 `assert_close` default. Of all the suites surveyed it has the tightest fp16 atol: about 100× tighter than JAX, IREE and the OpenVINO defaults, 250× tighter than the ONNX Runtime OpTester and 1000× tighter than the Intel NPU layer-test base.
- The fp16 rtol of about 1e-3 is a strong consensus (PyTorch, JAX, ORT, OpenVINO, ORT backend jsonc). That is roughly one fp16 machine epsilon (2^-10 ≈ 9.77e-4, a standard IEEE binary16 constant not sourced separately here), i.e. about 1–2 ULP depending on where the value sits in its binade.
- The OR rule (XLA, Polygraphy) is slightly looser than the additive rule: an element passes if it meets either the absolute or the relative bound.

### Gaps
- XLA has no global default `ErrorSpec`. Values are chosen per test; the few dot/GEMM test files sampled are listed in Q2. I did not survey XLA's GPU fusion tests exhaustively.
- I did not verify that `std::numeric_limits<ov::float16>::epsilon()` returns exactly 2^-10. The value is inferred from IEEE binary16 semantics.

## Q1. PyTorch: assert_close defaults, OpInfo/device-test overrides, inductor/dynamo `same()`, the fp64 RMSE accuracy checker, and attention-kernel tests

### Takeaway
`assert_close` defaults to fp16 (1e-3, 1e-5) and bf16 (1.6e-2, 1e-5). PyTorch's own test suites rarely rely on these for low-precision kernels:
- **OpInfo tests** loosen them per dtype with `toleranceOverride` or `precisionOverride`.
- **Inductor** computes its reference in fp32 (`reference_in_float=True`).
- **The dynamo benchmark accuracy checker** asks whether the compiled output's RMSE against an fp64 golden run is within 2× (fp32) or 3× (fp16/bf16) of eager's RMSE against the same golden run, plus `tol/10`.

### Cited Findings
- `_DTYPE_PRECISIONS` maps dtype to (rtol, atol): float16 (0.001, 1e-5), bfloat16 (0.016, 1e-5), float32 (1.3e-6, 1e-5), float64 (1e-7, 1e-7), complex32 (0.001, 1e-5). Quantized dtypes reuse the float32 values because they are compared dequantized. When two dtypes are mixed, the looser (max) tolerance wins — [_comparison.py L96-140](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/testing/_comparison.py#L96-L140)
- The closeness check is `torch.isclose(actual, expected, rtol, atol, equal_nan)`, i.e. `|a−e| ≤ atol + rtol·|e|` — [_comparison.py L1115-1128](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/testing/_comparison.py#L1115-L1128)
- `precisionOverride({dtype: precision})` sets the per-dtype atol, stored as `self.precision`. `toleranceOverride({dtype: tol(atol=..., rtol=...)})` sets both and takes priority over `precisionOverride`. Both are applied when a device-generic test is instantiated for a dtype — [common_device_type.py L607-622, L2111-2162](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/testing/_internal/common_device_type.py#L2111-L2162)
- In `TestCase.assertEqual` the overrides can only loosen the tolerance: `self.rtol = max(self.rtol, rtol_override)` and `self.atol = max(self.atol, atol_override)`. `_precision` and `_rel_tol` default to 0 — [common_utils.py L3652-3656, L3822-3823, L5081-5084](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/testing/_internal/common_utils.py#L3652-L3656)
- OpInfo reference tests (`test_ops.py::_ref_test_helper`) first compare with default tolerances. If that fails for a low-precision dtype, they rerun the op in the highest-precision float (float64) and require the reference implementation's summed |error| against that precise result to be ≤ the torch op's summed |error|. A TODO notes that this comparison has no tolerance yet — [test_ops.py L630-721](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/test/test_ops.py#L630-L721)
- Inductor's `check_model` defaults to `reference_in_float=True`: fp16/bf16 inputs are upcast to float32 for the eager reference, which is then compared to the compiled low-precision output. `atol`/`rtol` default to None, which means dtype-based `assertEqual` tolerances — [test_torchinductor.py L715-770](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/test/inductor/test_torchinductor.py#L715-L770)
- `torch._dynamo.utils.same(ref, res, fp64_ref=None, cos_similarity=False, tol=1e-4, ...)` works in stages:
  - Fast path: `torch.allclose(ref, res, atol=tol, rtol=tol)`.
  - If that fails and `fp64_ref` is float64: `ref_error = rmse(fp64_ref, ref)`, `res_error = rmse(fp64_ref, res)`, pass iff `res_error ≤ multiplier·ref_error + tol/10`.
  - Multiplier rules:
    - 3.0 if `res.dtype` is float16/bfloat16, otherwise 2.0. The code comment says 3 avoids AMP false alarms.
    - 3.0 for tensors with numel < 1000, 4-D tensors with 1×1 trailing dims, or `tol ≥ 2e-2`.
    - 10.0 for numel ≤ 10 and 8.0 for numel < 1000 when `use_larger_multiplier_for_smaller_tensor` is set.
    - 10.0 with `force_max_multiplier`.
  - `rmse = sqrt(mean((ref−res)^2))`.
  - Cosine mode: flattens both to fp32 and passes if `allclose(tol)` or cosine similarity ≥ 0.99 (eps 1e-6).
  - There is also a NaN corner case: if ref has NaN but res does not, pass when `allclose(fp64_ref, res, 4e-2, 4e-2)`.
  - [utils.py L3672-3960](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/_dynamo/utils.py#L3672-L3960)
- The benchmark harness (`benchmarks/dynamo/common.py::check_accuracy`) casts a deep copy of the model and inputs to float64 (`cast_to_fp64`), runs it to get `fp64_outputs`, and falls back to cosine mode if fp64 fails. It then calls `same(correct_result, new_result, fp64_outputs, tol=tolerance, cos_similarity=..., use_larger_multiplier_for_smaller_tensor=..., force_max_multiplier=...)`. Two eager runs must match each other with tol 0 (or `tolerance` on ROCm) — [common.py L2280-2313, L2365-2377, L2510-2523](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/benchmarks/dynamo/common.py#L2280-L2523)
- Per-suite `tol` values:
  - **torchbench:** base 1e-4. Under `--float16`/`--amp`: 1e-3, or 1e-2 ("higher_fp16") or 8e-2 ("even_higher") for allow-listed models. Under `--bfloat16`: 1e-2 for allow-listed models. Training on cuda/xpu: 1e-3 — [torchbench.py L435-463](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/benchmarks/dynamo/torchbench.py#L435-L463)
  - **huggingface:** inference 1e-3 (4e-3 or 5e-3 for allow-listed), training 1e-2 or 2e-2 — [huggingface.py L533-553](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/benchmarks/dynamo/huggingface.py#L533-L553)
  - **timm:** 1e-3 base; 8e-2 with freezing (conv-bn fusion); training up to 4e-2, 8e-2 or 16e-2 — [timm_models.py L371-399](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/benchmarks/dynamo/timm_models.py#L371-L399)
- FlexAttention tests (PyTorch inductor) use a three-way check: `golden_out` is computed in float64, `ref_out` is eager in the test dtype, and `compiled_out` is the kernel. They require `mean|golden−compiled| ≤ fudge·mean|golden−ref|` (via `assertEqual(compiled_error, ref_error, rtol=fudge, atol=1e-7)`). The fudge is 1.1 for low-precision dtypes and 10.0 for float32 (20.0 for paged attention). The comment attributes the extra fp32 slack to the online softmax — [test_flex_attention.py L1319-1363](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/test/inductor/test_flex_attention.py#L1319-L1363)
- FlashAttention (Dao-AILab; related evidence, not PyTorch core) asserts `max|out − out_ref| ≤ 2·max|out_pt − out_ref|`. `out_ref` is computed with `upcast=True` (all math in fp32). `out_pt` is the same math in the low-precision dtype (`upcast=False, reorder_ops=True`). Gradients are checked the same way — [test_flash_attn.py L646-714](https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py#L646-L714)

### Inferences
- The inductor accuracy checker, FlexAttention and FlashAttention all test the kernel's error relative to the error a naive same-dtype implementation already makes against a high-precision golden output. They do not use an absolute ULP budget. This scales automatically with reduction length K and with conditioning, because the baseline's own error grows the same way.
- A direct analogue for our NPU harness: compute `ref_fp64` and a "naive fp16" reference (e.g. OpenVINO CPU with `inference_precision=f16`, or numpy fp16 with fp16 accumulation). Accept if `RMSE(npu, ref_fp64) ≤ 3·RMSE(naive_fp16, ref_fp64) + small atol`.

### Gaps
- I did not catalogue concrete `toleranceOverride` values in `common_methods_invocations.py`. They are set per op and per dtype, and no global fp16 table exists.
- `torch.testing.assert_close` has no subnormal-specific handling (searched `_comparison.py`; nothing found).

## Q2. JAX / XLA: default tolerances and ErrorSpec semantics

### Takeaway
JAX uses one value per dtype for both atol and rtol: fp16 1e-3, bf16 1e-2, fp32 1e-6. Gradient checks are looser and multiply the tolerance by the tensor size. XLA's `ErrorSpec{abs, rel}` flags an element only when both bounds are exceeded. It can also express a ULP-distance bound (`within_n_values`) for low-precision floats, in which +0 and −0 are equivalent.

### Cited Findings
- `_default_tolerance` sets bfloat16 1e-2, float16 1e-3, float32 1e-6, float64 1e-15, complex64 1e-6, float8 variants 1e-1, float4_e2m1fn 1e0, and 0 for all integer types — [public_test_util.py L46-76](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/public_test_util.py#L46-L76)
- `default_gradient_tolerance` sets bfloat16 1e-1, float16 1e-2, float32 2e-3, float64 1e-5 — [public_test_util.py L88-104](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/public_test_util.py#L88-L104)
- `JaxTestCase.assertArraysAllClose` takes `atol = max(tolerance(actual.dtype), tolerance(desired.dtype))`, does the same for rtol, and calls `_assert_numpy_allclose`. That function upcasts bf16/fp8/fp4 to float32 and calls `np.testing.assert_allclose(a, b, atol, rtol)`. A falsy (0) atol or rtol is omitted, so NumPy's own default applies — [test_util.py L1318-1325](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/test_util.py#L1318-L1325); [public_test_util.py L110-147](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/public_test_util.py#L110-L147)
- `_assert_numpy_close`, used by `check_close`/`check_jvp`/`check_vjp`/`check_grads`, calls `_assert_numpy_allclose(a, b, atol=atol * a.size, rtol=rtol * b.size)`. The tolerance is explicitly scaled by the number of elements — [public_test_util.py L159-171](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/public_test_util.py#L159-L171)
- `xla::ErrorSpec(abs, rel=0, relaxed_nans=false)` has these fields:
  - `all_nans_are_equivalent=true`.
  - `fewer_infs_ok`: an actual ±inf may stand in for an expected ±max.
  - `allow_integer_rounding_difference`: allows ±1 between fused hardware rounding and truncating references.
  - `low_precision_fp_error_spec{type, within_n_values}`: the "allowable distance in number of representable floats … +/-0 are considered equivalent". When set it overrides abs/rel and is intended for FP8.
  - [error_spec.h L24-71](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/error_spec.h#L24-L71)
- `NearComparator::CompareValues` computes `abs_error = |a−e|` and `rel_error = abs_error/|e|`, with rel_error = ∞ when e == 0. It sets `is_mismatch = (abs_error > spec.abs) && (rel_error > spec.rel)`, unless the value is within `within_n_values` floats. A NaN mismatch counts as infinite error — [literal_comparison.cc L627-705](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/literal_comparison.cc#L627-L705)
- XLA test values:
  - `DotOperationTest` fixture default: `ErrorSpec error_spec_{0.0001, 1e-5}`.
  - For `Eigen::half` it switches to `{0.001, 0.003}` in one test and `{0.0001, 1e-3}` in another — [dot_operation_test.cc L63, L349-353, L831-832](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/tests/dot_operation_test.cc#L349-L353)
  - Tally of ErrorSpec literals in `gemm_rewriter_test.cc` + `dot_operation_test.cc`: `{1e-5,1e-5}` ×42, `{1e-3,1e-3}` ×13, `{1e-4,1e-5}` ×11, `{4e-3,4e-3}` ×8, `{1e-2,7e-2}` ×2, `{0.01,0.01}` ×1 — [gemm_rewriter_test.cc](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/backends/gpu/transforms/gemm_rewriter_test.cc)
- XLA has a `--xla_gpu_ftz` debug flag ("If true, flush-to-zero semantics are enabled in the code generated for GPUs") — [debug_options_flags.cc L1593-1596](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/debug_options_flags.cc#L1593-L1596)

### Inferences
- XLA's per-element mismatch rule is `|a−e| > abs AND |a−e|/|e| > rel`, so an expected 0 always falls back to the abs bound. An FTZ'd subnormal passes whenever `|subnormal| ≤ abs`. With the typical half-precision spec of abs 1e-3 every fp16 subnormal (|x| < 6.1e-5) passes.
- XLA's ULP mode (`within_n_values`) is the only first-party ULP-distance comparator found in this survey. It is meant for FP8, but the same idea can be applied to FP16.

### Gaps
- I could not confirm the default value of `xla_gpu_ftz` from the flags file. The search for its setter in the defaults did not match.
- I found no JAX handling of TPU or denormals in `test_util.py` (searched for denorm/subnormal/ftz; none).

## Q3. TVM: assert_allclose defaults and fp16 test practice

### Takeaway
`tvm.testing.assert_allclose` defaults to rtol = atol = 1e-7 for every dtype, so fp16 tests always pass explicit tolerances, usually 1e-2/1e-2. Some tests sidestep rounding entirely by using small-integer inputs.

### Cited Findings
- `assert_allclose(actual, desired, rtol=1e-7, atol=1e-7)` wraps `np.testing.assert_allclose`, comparing `abs(actual−desired)` with `atol + rtol·abs(desired)`. The docstring: "Since we often allow `desired` to be close to zero, we generally want non-zero `atol`" — [tvm v0.27.0 utils.py L113-124](https://github.com/apache/tvm/blob/v0.27.0/python/tvm/testing/utils.py#L113-L124); [tvm-tirx utils.py L34-45](https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/testing/utils.py#L34-L45)
- In relax BYOC codegen tests on main, fp16 CUTLASS/cuBLAS/cuDNN tests use `rtol=1e-2, atol=1e-2` most often. Some use 1e-3/1e-3, 2.5e-2, 1e-1, 3e-1, or `rtol=1e-2, atol=2e-2` — [test_codegen_cutlass.py](https://github.com/apache/tvm/blob/18ec45ed7729b70ae94c340014624b34b6ecf5a7/tests/python/relax/test_codegen_cutlass.py); [test_codegen_cublas.py](https://github.com/apache/tvm/blob/18ec45ed7729b70ae94c340014624b34b6ecf5a7/tests/python/relax/test_codegen_cublas.py)
- `test_conv2d_offload` (fp16) draws data, weight and bias with `np.random.randint(-1, 1)`, so all values are small integers — [test_codegen_cutlass.py](https://github.com/apache/tvm/blob/18ec45ed7729b70ae94c340014624b34b6ecf5a7/tests/python/relax/test_codegen_cutlass.py)

### Inferences
- TVM sets no dtype-aware default. In practice its fp16 convention (1e-2/1e-2) is about 10× looser than PyTorch's rtol and 1000× looser in atol.

### Gaps
- I did not find a TVM-wide fp16 tolerance policy document. Values are chosen per test.

## Q4. IREE / MLIR: f16 thresholds and ULP checks

### Takeaway
`iree-run-module` and `iree-check-module` defaults are: f16 atol 1e-3, bf16 atol 1e-2, f32/f64 atol 1e-4, plus a shared rtol of 1e-5. The formula is numpy-style `atol + rtol·|expected|`, adopted in May 2026. The e2e matmul/conv tests mostly avoid tolerances: they feed small integers so results are bit-exact, then fall back to a 1e-5 absolute delta.

### Cited Findings
- Flags: `--expected_f16_threshold` 0.001, `--expected_f32_threshold` 0.0001, `--expected_f64_threshold` 0.0001, `--expected_bf16_threshold` 0.01 ("Per-element-type absolute tolerance … in approximate mode"), `--expected_rtol` 1e-5 ("shared across float types"). The `--equality_mode` default is `approximate`, documented as "`abs(actual - expected) <= atol + rtol * abs(expected)` per element with NaN considered equal to NaN"; `exact` is a byte compare — [comparison.cc L21-35](https://github.com/iree-org/iree/blob/c6882b84d0afece9ea2d810ccc976997ea0d199b/runtime/src/iree/tooling/comparison.cc#L21-L35)
- PR #24394 (2026-05-08) says it "collapses both into 'approximate' mode … follows ML frameworks convention (e.g., numpy.isclose, torch.allclose)". It deliberately uses the asymmetric `|a−b| ≤ atol + rtol·|b|` "instead of `|a−b| ≤ max(atol, rtol·|b|)`", and moves `NumpyFuzzyCompare` into `runtime/src/iree/base/internal/math.h` — [IREE PR #24394](https://github.com/iree-org/iree/pull/24394)
- e2e tests (`tools/testing/e2e/test_utils.c`) use `--acceptable_fp_delta` default 1e-5f ("Maximum absolute difference allowed with inexact floating point results"). The code comment reads: "Since we fill buffers with small integers for floating point GEMMs functional testing, we can test for bit-exactness". f16 checks bit equality first, then `fabsf(a−e) < delta`. For bf16 accumulators, `|x| > 127` passes if within 10.0, because "Accumulators grow outside of the narrow range of bf16 exact representation of integers" — [test_utils.c L33-34, L258-294](https://github.com/iree-org/iree/blob/c6882b84d0afece9ea2d810ccc976997ea0d199b/tools/testing/e2e/test_utils.c#L258-L294)
- e2e matmul references accumulate in f32 even for f16-in/f16-out (`acc += f16_to_f32(lhs)*f16_to_f32(rhs)`, then `f32_to_f16(acc)`). That is the same "fp32 accumulate, round once" reference model we use — [iree-e2e-matmul-test.cc L53-120](https://github.com/iree-org/iree/blob/c6882b84d0afece9ea2d810ccc976997ea0d199b/tools/testing/e2e/iree-e2e-matmul-test.cc#L53-L120)

### Inferences
- IREE's integer-input strategy removes accumulation error and subnormals from the correctness question entirely. Products and sums of small integers are exact in fp16 up to 2048. This is a useful complementary test mode for NPU kernels: separate "logic correct" (bit-exact on integer data) from "numerics acceptable" (toleranced on random data).

### Gaps
- I found no ULP-based comparator in IREE's tooling (comparison.cc and test_utils.c use only absolute and relative bounds). I did not inspect MLIR upstream test utilities.

## Q5. ONNX Runtime: OpTester per-EP defaults and onnx_test_runner

### Takeaway
ORT's OpTester defaults are fp16 atol 2.5e-3 / rtol 1e-3 (DML EP: atol 5e-3) and bf16 atol 2e-2 / rtol 1e-2. The numbers were derived empirically: atol is 10× the max fp16 rounding error on U[0,1) data, and rtol is 2× the max relative rounding error. Builds for the OpenVINO EP on NPU loosen the fp32 atol to 5e-3. `onnx_test_runner` uses a per-test override JSON with defaults rtol 1e-3 and atol 1e-5.

### Cited Findings
- `DefaultTolerance` in `checkers.cc`:
  - **double:** atol 1e-5, rtol 1e-5.
  - **float:** atol 1e-5 (1e-3 if `ENABLE_TRAINING`), rtol 1e-4. `get_absolute` returns 0.005 for `kOpenVINOExecutionProvider` under `#ifdef OPENVINO_CONFIG_NPU`; double gets the same NPU bump.
  - **MLFloat16:** atol 0.0025 (training 0.005), rtol 0.001; atol 0.005 for `kDmlExecutionProvider`.
  - **BFloat16:** atol 0.02, rtol 0.01.
  - The code comment gives the derivation: `x = torch.rand(1000,1000); absolute = ((x + 1e-6).to(torch.float16) - x).abs().max() * 10; x[abs(x) < absolute] = absolute; relative = ((x - x.to(torch.float16)) / x).abs().max() * 2`.
  - [checkers.cc L20-101](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/unittest_util/checkers.cc#L20-L101)
- `get_tolerance = atol + rtol·|expected|` ("similar to numpy.isclose"). Explicit `absolute_error`/`relative_error` values on the test replace the defaults. fp16 values are converted to float before comparing. An expected NaN requires a NaN, an expected Inf requires equality, and everything else uses `EXPECT_NEAR` — [checkers.cc L103-123, L495-520](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/unittest_util/checkers.cc#L103-L123)
- Integer outputs: uint8 tolerance is allowed only for NNAPI/DML/XNNPACK EPs (rounding-mode differences). For int8, an absolute error below 1 is equivalent to no tolerance — [checkers.cc L329-389](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/unittest_util/checkers.cc#L329-L389)
- `onnx_test_runner`:
  - CLI help: "-t: Specify custom relative tolerance … default: 1e-5" and "-a: … absolute tolerance … default: 1e-5".
  - Without custom values it reads `testdata/onnx_backend_test_series_overrides.jsonc` (`"rtol_default": 1e-3, "atol_default": 1e-5`), with per-test overrides such as `test_attention_4d_fp16` atol 5e-4, `test_attention_4d_gqa_with_past_and_present_fp16` 6e-4 and `test_attention_4d_causal_fp16` rtol 0.002.
  - If the file is missing: atol 1e-3, rtol 1e-3 (0.017 with CUDA, 0.009 with OpenVINO, both "for resolving … MNIST" mismatches).
  - [main.cc L67-68, L120-131](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/onnx/main.cc#L120-L131); [overrides.jsonc](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc#L1-L29)
- `compare_ortvalue.cc`:
  - `tol = per_sample_tolerance + relative_per_sample_tolerance·|expected|`.
  - `IsResultCloselyMatch`: NaN must pair with NaN, Inf must pair with a same-sign Inf, otherwise `diff ≤ tol`. fp16 is compared after conversion to float.
  - A `post_processing` mode clamps to [0,255] first.
  - Under `MLAS_SBGEMM_AVAILABLE` (bf16 fast-math GEMM), a NaN tolerance switches to a cosine-similarity check with threshold 0.99.
  - [compare_ortvalue.cc L70-96, L135-147, L163, L299](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/util/compare_ortvalue.cc#L135-L147)
- Runtime FTZ knob `session.set_denormal_as_zero`: "If a value is '1', flush-to-zero and denormal-as-zero are applied. The default is '0'". The comment warns that it "may hurt model accuracy" and recommends exporting models without denormals instead — [onnxruntime_session_options_config_keys.h L57-65](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L57-L65)

### Inferences
- ORT's derivation recipe is a principled template we could reuse for the NPU: atol = k × (max fp16 representation error on the test input distribution), with ref clamped away from zero when estimating rtol. With k=10 the result is 2.5e-3, about 40× the largest fp16 subnormal.
- ORT already treats "OpenVINO EP on NPU" as needing a 500× looser fp32 atol (5e-3 vs 1e-5).

### Gaps
- QNN EP fp16 thresholds: the QNN tests live in `onnxruntime/test/providers/qnn/` and set per-test tolerances. I did not survey them.
- I did not confirm which onnx_test_runner flag letter (`-e` or `-t`) maps to relative tolerance in the current parser; the help text says `-t` and the parser has a `case 'e'`.

## Q6. OpenVINO: functional test comparison, inference_precision, GPU plugin and Intel NPU tests

### Takeaway
OpenVINO's shared `compare()` uses the additive formula, but its default thresholds are precision-aware:
- **abs:** the machine epsilon of the lowest of the expected, actual and `inference_precision` types (fp16 → about 9.77e-4).
- **rel:** a per-type table (f32 1e-4, f16 1e-3, bf16 1e-2).

Two outlier tolerances are built in: `topk_threshold` (fraction of elements allowed to fail) and `mvn_threshold`. The Intel NPU layer tests in `npu_compiler` use base abs 0.01 and loosen per op and per platform (e.g. conv rel 0.02; eltwise on NPU3720 abs 0.6). NPUW's runtime accuracy check uses range-normalized RMSE ≤ 0.01.

### Cited Findings
- `get_eps_by_ov_type`: f64 1e-8, f32 1e-4, f16 1e-3, bf16 1e-2, nf4 1e-1, otherwise 0. This is used as the default rel threshold — [ov_tensor_utils.hpp L116-131](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/test_utils/common_test_utils/include/common_test_utils/ov_tensor_utils.hpp#L116-L131)
- `calculate_default_abs_threshold(expected_type, actual_type, inference_precision)` = max of `std::numeric_limits<T>::epsilon()` over bf16/f16/f32/f64 among the three types. `calculate_default_rel_threshold` = max of `get_eps_by_ov_type` over the same three. `calculate_threshold = abs + rel·|ref|` — [ov_tensor_utils.cpp L521-575](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/test_utils/common_test_utils/src/ov_tensor_utils.cpp#L521-L575)
- `compare(expected, actual, inference_precision, abs=-1, rel=-1, topk=1.0, mvn=1.0)`:
  - −1 means "use the defaults".
  - The header documents `topk_threshold` as the "percentage of incorrect values in tensor [0,1]" and `mvn_threshold` as "avg value of diff/threshold … tensor jitter relative to threshold".
  - `Error::update` adds `diff/threshold` to `mvn_results` and records a mismatch if `diff > threshold`.
  - `check_results` fails if (any mismatch and topk == 1) or `#mismatch > floor(topk·N)`, or if `mean(diff/threshold) > mvn`.
  - Pairs where both values are NaN/±Inf, or both sit at or beyond the type max/lowest, are skipped as "not suitable for comparison".
  - [ov_tensor_utils.hpp L198-215](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/test_utils/common_test_utils/include/common_test_utils/ov_tensor_utils.hpp#L198-L215); [ov_tensor_utils.cpp L436-519, L583-650](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/test_utils/common_test_utils/src/ov_tensor_utils.cpp#L436-L519)
- `SubgraphBaseTest` reads `inference_precision = compiledModel.get_property(ov::hint::inference_precision)` after compiling. `init_thresholds()` fills unset thresholds from `calculate_thresholds_by_model(function, functionRefs, inference_precision)`, which takes the max over every op's input element types and every output's (type, ref type, inference precision). A `custom_op_thresholds` map exists but is empty by default — [ov_subgraph.cpp L307, L519-527](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/functional/base_func_tests/src/base/ov_subgraph.cpp#L519-L527); [calculate_thresholds.cpp L19-95](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/functional/base_func_tests/src/base/utils/calculate_thresholds.cpp#L19-L95); [calculate_thresholds.hpp L29](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/tests/functional/base_func_tests/include/shared_test_classes/base/utils/calculate_thresholds.hpp#L29)
- GPU plugin functional tests (98 files under `intel_gpu/tests/functional/{single_layer_tests,subgraph_tests}`) contain explicit overrides from `abs_threshold = 1e-4f` up to `4.0f`. Common values are 1e-4, 5e-3, 2.5e-2, 0.1 and 1.0; rel values run from 5e-3 to 0.2, most often 0.01 and 5e-3 (my tally of the source tree) — [intel_gpu functional tests](https://github.com/openvinotoolkit/openvino/tree/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/plugins/intel_gpu/tests/functional)
- The OpenVINO repo's own `intel_npu/tests/functional` holds mostly behavior/NPUW tests. Per-layer accuracy tests for NPU are in `openvinotoolkit/npu_compiler` (public, not archived, default branch `develop`) — [npu_compiler repo](https://github.com/openvinotoolkit/npu_compiler)
- `VpuOv2LayerTest` constructor sets `abs_threshold = 0.01`. Its `compare()` calls `init_thresholds()` and then `ov::test::utils::compareTensors(expected, actual, abs_threshold, rel_threshold)` for each output — [vpu_ov2_layer_test.cpp L36, L303-325](https://github.com/openvinotoolkit/npu_compiler/blob/20e7912fbc230725e1ca7394762348d4f480aa65/tests/functional/shared_tests_instances/vpu_ov2_layer_test.cpp#L303-L325)
- NPU `compareTensorsTyped` (`tests/functional/common/tensor_comparison.cpp`, updated 2026-09-02 in "UD2026.38"):
  - A threshold of −1 means "not set" and is treated as 0.
  - Both NaN = match; one NaN = mismatch.
  - Inf matches only same-sign Inf, except that overflow "at type boundary" is skipped: one value ≥ T::max and the other +Inf. The comment says this "is expected for low-precision types (e.g. fp16) where hardware arithmetic may overflow slightly before or after the reference".
  - Otherwise mismatch if `|e−a| > abs + rel·|e|`.
  - It reports the top-N worst elements and warns if the expected tensor contains NaN/Inf ("reference data may be corrupt").
  - [tensor_comparison.cpp L66-185, L414-435](https://github.com/openvinotoolkit/npu_compiler/blob/20e7912fbc230725e1ca7394762348d4f480aa65/tests/functional/common/tensor_comparison.cpp#L66-L185)
- NPU per-test overrides:
  - Convolution: `rel_threshold = 0.02` for NPU3720/4000/5010/5020, HW and SW variants.
  - Softmax: `abs_threshold` 0.01 or 1e-3.
  - Eltwise on NPU3720 SW: `abs_threshold = 0.6`.
  - Tally over the 94 layer/subgraph test files: rel 0.02 ×25, rel 0.01 ×21, abs 1e-3 ×17, abs 0.6 ×15, abs 0.02 ×12, rel 1e-3 ×8, abs 0.0056 ×8.
  - [convolution.cpp L81-120](https://github.com/openvinotoolkit/npu_compiler/blob/20e7912fbc230725e1ca7394762348d4f480aa65/tests/functional/shared_tests_instances/single_layer_tests/convolution.cpp#L81-L120); [eltwise.cpp L236-237](https://github.com/openvinotoolkit/npu_compiler/blob/20e7912fbc230725e1ca7394762348d4f480aa65/tests/functional/shared_tests_instances/single_layer_tests/eltwise.cpp#L236-L237); [softmax.cpp L46-60](https://github.com/openvinotoolkit/npu_compiler/blob/20e7912fbc230725e1ca7394762348d4f480aa65/tests/functional/shared_tests_instances/single_layer_tests/softmax.cpp#L46-L60)
- NPUW accuracy check (`NPUW_ACC_CHECK` default false, `NPUW_ACC_THRESH` default 0.01, `NPUW_ACC_DEVICE` for the reference device). Both tensors are converted to fp32, then `rmse = sqrt(Σ(a−r)²/N)` and `den = max(0.001, max(0,max_ref) − min(0,min_ref), same for actual)`. It passes if `rmse/den ≤ threshold`, or if `Σ(a−r)² ≤ DBL_EPSILON` — [npuw_option_defs.inc L46-48](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/plugins/intel_npu/src/al/include/intel_npu/config/npuw_option_defs.inc#L46-L48); [comparator.cpp L15-66](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/plugins/intel_npu/src/plugin/npuw/accuracy/comparator.cpp#L15-L66)
- The OpenVINO CPU plugin exposes `ov::intel_cpu::denormals_optimization` ("CPU_DENORMALS_OPTIMIZATION"). The docs say FTZ/DAZ "could significantly improve the performance, but it does not comply with IEEE standard. In most cases, this behavior has little impact on model accuracy" — [intel_cpu/properties.hpp L84-102](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/inference/include/openvino/runtime/intel_cpu/properties.hpp#L84-L102)

### Inferences
- For an fp16 NPU test with default thresholds, OpenVINO's shared comparator gives roughly `|a−e| ≤ 9.77e-4 + 1e-3·|e|`. The NPU layer-test base gives `0.01 + rel·|e|`; rel comes from `init_thresholds()` (likely 1e-3 for an f16 graph, but this depends on the test's model types and is not directly confirmed). Either atol absorbs every FTZ'd fp16 subnormal (≤ 6.1e-5) by a factor of 16–160.
- Intel's own NPU3720 tests treat errors of 2% relative (conv) or 0.6 absolute (SW eltwise) as acceptable for some kernels. Our 1e-3/1e-5 verdict is far stricter than the vendor's own test suite.
- `topk_threshold` (e.g. 0.01) is OpenVINO's native mechanism for tolerating a small fraction of outliers such as FTZ or cancellation sites without loosening the bound for every element.

### Gaps
- OpenVINO Accuracy Checker / `accuracy_checker` (Open Model Zoo tooling) works at the dataset-metric level (top-1, mAP). I did not investigate it here because it does not define per-element kernel tolerances.
- I did not verify which `inference_precision` the NPU plugin reports in `VpuOv2LayerTest` flows, so the exact default rel_threshold for NPU tests is inferred.

## Q7. TensorRT / Polygraphy: defaults, check-error-stat options, quantile/median outlier tolerance

### Takeaway
Polygraphy defaults to atol = rtol = 1e-5 with an `elemwise` check that fails only when an element exceeds both tolerances. Its docs say these defaults suit FP32 and must be relaxed for reduced precision; the example uses `--atol 1e-3 --rtol 1e-3 --check-error-stat median`. The `median`/`mean`/`quantile` stats compare a tensor-level statistic of |diff| and |diff|/|ref| instead of every element, so a minority of outliers cannot cause a failure.

### Cited Findings
- `SimpleCompareFunc`:
  - `_DEFAULT_RTOL = 1e-5`, `_DEFAULT_ATOL = 1e-5`, `_DEFAULT_QUANTILE = 0.99`, `_DEFAULT_ERROR_STAT = "elemwise"`.
  - Valid stats are `["max", "mean", "median", "elemwise", "quantile"]`.
  - rtol is "a percentage of the second set of output values", i.e. the reference.
  - The docstring describes `max` as "the strictest possible check".
  - [compare.py L799-802, L826-888, L947](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/tools/Polygraphy/polygraphy/comparator/compare.py#L799-L888)
- Diff computation:
  - `absdiff = |cast_up(out0) − cast_up(out1)|`, i.e. computed after upcasting.
  - Zeros in the reference get `EPSILON = 2.220446049250313e-16` added before `reldiff = absdiff/|out1|`, with a warning that "this may cause some relative differences to be extremely high".
  - `elemwise` verdict: `mismatches = (absdiff > atol | isnan(absdiff)) & (reldiff > rtol | isnan(reldiff))`.
  - [compare.py L955-990, L1022-1025](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/tools/Polygraphy/polygraphy/comparator/compare.py#L955-L1025)
- Scalar-stat verdict: `pass = not (stat_absdiff > atol and stat_reldiff > rtol)`, where stat is max, mean, median or the q-quantile of the per-element arrays — [compare.py L97-139](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/tools/Polygraphy/polygraphy/comparator/compare.py#L97-L139)
- CLI options:
  - Per-output `--rtol/--rel-tol` and `--atol/--abs-tol` use the `[<out_name>:]<value>` syntax, e.g. `--rtol 1e-5 out0:1e-4 out1:1e-3`.
  - `--check-error-stat [<out>:]<stat>` and `--error-quantile` (float in [0,1], e.g. `0.95 out0:0.8`).
  - Metric thresholds: `--l2-tolerance`, `--cosine-similarity-threshold`, `--psnr-tolerance`, `--snr-tolerance`, `--lpips-threshold`, `--index-tolerance` (top-k).
  - `--infinities-compare-equal`.
  - [args/comparator/compare.py L63-134, L220-449](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/tools/Polygraphy/polygraphy/tools/args/comparator/compare.py#L63-L134)
- Official example (01_comparing_frameworks README):
  - "The default tolerances used by `run` are usually appropriate for FP32 precision but may not be appropriate for reduced precisions."
  - It explains that elemwise checks `absdiff[i] > atol and reldiff[i] > rtol` per index, which "is less stringent than … `--check-error-stat max`" because max can fail on different indices i and j.
  - Example command: `polygraphy run … --trt --tf32 --onnxrt --atol 0.001 --rtol 0.001 --check-error-stat median`.
  - The same `--atol 0.001 --rtol 0.001 --check-error-stat median` appears in the 02_comparing_across_runs example.
  - [01_comparing_frameworks/README.md](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/tools/Polygraphy/examples/cli/run/01_comparing_frameworks/README.md); [02_comparing_across_runs/README.md](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/tools/Polygraphy/examples/cli/run/02_comparing_across_runs/README.md)

### Inferences
- How quantile/median checks tolerate outliers: `quantile` with q=0.99 passes as long as fewer than about 1% of elements have both abs and rel error above tolerance (approximately, since abs and rel are thresholded as separate statistics). `median` tolerates up to about 50%. A handful of FTZ or cancellation sites therefore cannot fail a median or q≤0.99 check. This is a statistical analogue of OpenVINO's `topk_threshold`.
- Because of the epsilon hack, Polygraphy's reldiff is enormous wherever the reference is exactly 0, so those elements must pass on atol alone. The same holds when the reference is subnormal and the device flushes it.

### Gaps
- I did not find an NVIDIA-official per-dtype fp16 tolerance table for Polygraphy; the docs only say "relax". `how-to/work_with_reduced_precision.md` exists but I found no specific tolerance numbers in it.

## Q8. Explicit handling of subnormal flush-to-zero, near-zero cancellation, and reduction-length (K) scaling

### Takeaway
None of the comparators surveyed special-cases subnormals or treats |ref| below the smallest normal as zero. They absorb FTZ implicitly through an fp16 atol (≥ about 1e-3) that dwarfs the largest fp16 subnormal (about 6.1e-5). Near-zero values rely on atol in every tool. No comparator found scales tolerance with reduction length K; JAX's gradient checker scaling by total tensor size is the closest. K-dependence is handled indirectly instead:
- reference-relative error metrics (dynamo fp64 RMSE ×2–3, FlashAttention 2×, FlexAttention 1.1×);
- integer-valued inputs that make GEMMs exact (IREE, TVM);
- aggregate metrics (NRMSE, cosine, median/quantile, topk).

### Cited Findings
- FTZ / subnormals:
  - Searches of the comparator sources for "denorm|subnormal|ftz|flush" found nothing in PyTorch `_comparison.py`, XLA `literal_comparison.cc`, JAX `test_util.py`, OpenVINO `ov_tensor_utils.cpp`, the NPU `tensor_comparison.cpp`, ORT `checkers.cc`/`compare_ortvalue.cc`, Polygraphy `compare.py` or IREE `comparison.cc`/`test_utils.c` (negative result of my source greps).
  - The FTZ/denormal knobs that do exist are runtime execution flags, not test-tolerance rules: ORT `session.set_denormal_as_zero` (default "0"), OpenVINO CPU `denormals_optimization`, XLA `--xla_gpu_ftz` — [ORT config keys L57-65](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L57-L65); [OV CPU properties L84-102](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/inference/include/openvino/runtime/intel_cpu/properties.hpp#L84-L102); [XLA flags L1593](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/debug_options_flags.cc#L1593-L1596)
- The closest thing to an explicit low-precision edge-case rule is the NPU comparator forgiving fp16 overflow at the type boundary (finite ≥ max vs Inf). This is the overflow analogue of FTZ — [npu_compiler tensor_comparison.cpp L141-160](https://github.com/openvinotoolkit/npu_compiler/blob/20e7912fbc230725e1ca7394762348d4f480aa65/tests/functional/common/tensor_comparison.cpp#L141-L160)
- XLA's ULP mode treats +0 and −0 as equivalent. Its relative error is defined as ∞ when expected == 0, so zeros always fall back to the abs bound — [error_spec.h L60-71](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/error_spec.h#L60-L71); [literal_comparison.cc L672-684](https://github.com/openxla/xla/blob/8476ae48fd5f490a969d32ec91ff8d29b51e14fb/xla/literal_comparison.cc#L672-L684)
- Near-zero:
  - ORT's derivation clamps `x[abs(x) < absolute] = absolute` before estimating rtol — [checkers.cc L71-75](https://github.com/microsoft/onnxruntime/blob/fe7700a7e700c7966a822c41ae781ef3523a2538/onnxruntime/test/unittest_util/checkers.cc#L67-L80).
  - Polygraphy adds 2.2e-16 to zero references — [compare.py L975-990](https://github.com/NVIDIA/TensorRT/blob/16ed0636df41e4a27d46f82d9028fc50fadc7200/tools/Polygraphy/polygraphy/comparator/compare.py#L975-L990).
  - The TVM docstring warns that "we often allow `desired` to be close to zero, we generally want non-zero `atol`" — [tvm-tirx utils.py L34-41](https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/testing/utils.py#L34-L41).
  - NPUW normalizes RMSE by the tensor's value range, with a floor of 0.001 — [comparator.cpp L60-66](https://github.com/openvinotoolkit/openvino/blob/86fc05b3b26ee1aeed2fce0aa233983d897b5603/src/plugins/intel_npu/src/plugin/npuw/accuracy/comparator.cpp#L15-L66).
- Size/K scaling:
  - JAX `_assert_numpy_close` uses `atol * a.size` and `rtol * b.size` — [public_test_util.py L164](https://github.com/jax-ml/jax/blob/d3e85eb23e712dddc78935b33a12d580e0198985/jax/_src/public_test_util.py#L159-L165).
  - Dynamo's multiplier depends on tensor numel (larger multipliers for numel < 1000 or ≤ 10, i.e. small tensors where noise dominates RMSE), not on K — [utils.py L3899-3932](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/_dynamo/utils.py#L3899-L3932).
  - A grep of `test/test_matmul_cuda.py` for K- or sqrt(K)-scaled tolerances found none (negative result) — [test_matmul_cuda.py](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/test/test_matmul_cuda.py).
- Exactness by construction: the IREE e2e GEMM/conv tests ("we fill buffers with small integers … test for bit-exactness") and TVM's CUTLASS conv test (`randint(-1, 1)`) — [IREE test_utils.c L258-262](https://github.com/iree-org/iree/blob/c6882b84d0afece9ea2d810ccc976997ea0d199b/tools/testing/e2e/test_utils.c#L258-L262); [TVM test_codegen_cutlass.py](https://github.com/apache/tvm/blob/18ec45ed7729b70ae94c340014624b34b6ecf5a7/tests/python/relax/test_codegen_cutlass.py)
- Reference-relative bounds that automatically absorb K-dependent accumulation error:
  - dynamo `res_rmse ≤ 3·eager_rmse + tol/10` for fp16/bf16 — [utils.py L3936](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/torch/_dynamo/utils.py#L3936).
  - FlashAttention `max_err ≤ 2·max_err(pt same-dtype)` — [test_flash_attn.py L705](https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py#L705).
  - FlexAttention `mean_err ≤ 1.1·mean_err(eager same-dtype)` against an fp64 golden — [test_flex_attention.py L1335-1363](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/test/inductor/test_flex_attention.py#L1335-L1363).
  - PyTorch OpInfo `ref_distance ≤ torch_distance` against float64 — [test_ops.py L704-721](https://github.com/pytorch/pytorch/blob/f9a523939b1243f1d84038313bc132ab32c40f1c/test/test_ops.py#L704-L721).

### Inferences
- **Why our failures happen.** These use standard IEEE binary16 constants, not separately sourced here: smallest normal 2^-14 ≈ 6.10e-5, largest subnormal ≈ 6.10e-5, smallest subnormal 2^-24 ≈ 5.96e-8. Under `|a−e| ≤ 1e-5 + 1e-3·|e|`, an FTZ'd reference value v fails whenever |v| > about 1.001e-5, i.e. every reference value in roughly (1.0e-5, 6.1e-5). Each other suite's fp16 atol covers the whole subnormal range: JAX 1e-3, IREE 1e-3, OpenVINO ~9.77e-4, ORT 2.5e-3, NPU tests 1e-2, XLA half dot tests 1e-3.
- **Two ways to model FTZ, both consistent with established practice.**
  - (a) Raise fp16 atol to at least 2^-14 (6.1e-5), or to the consensus value of about 1e-3.
  - (b) Keep a tight atol, but flush the reference first: set elements with |ref| < 2^-14 to 0 in the fp16-rounded reference before comparing, so FTZ is "the expected semantics of the target". No surveyed framework does (b), but it is the explicit form of what large atols do implicitly.
- **For cancellation and K-long reductions**, the precedents point to a two-part verdict:
  - an elementwise check with a modest atol (1e-3) and rtol (1e-3 to 2e-3) plus an outlier allowance (OpenVINO topk, e.g. ≤ 0.1–1% of elements, or a Polygraphy-style median/0.99-quantile);
  - an aggregate check relative to a naive same-precision baseline against an fp64 reference (dynamo-style `RMSE_npu ≤ 3·RMSE_baseline + tol/10`), or an NPUW-style NRMSE ≤ 0.01.
- **Optional integer-input pass.** Adding an integer-valued input pass (IREE/TVM style) would separate logic bugs (which must be bit-exact) from numerics (toleranced).

### Gaps
- I found no framework-level documentation or code that scales tolerance as a function of the reduction length K. If such scaling is wanted (e.g. atol ∝ sqrt(K)·eps·max|x||w|, from standard floating-point error analysis), it would have to come from the numerical-analysis literature, not from these test suites. That literature is outside this note's scope.
- I could not check the internal Intel NPU compiler (VPUX) CI thresholds that are not in the public `npu_compiler` repo.
