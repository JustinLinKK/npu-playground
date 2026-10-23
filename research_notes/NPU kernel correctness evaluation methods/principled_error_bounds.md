# Principled Error Bounds for Low-Precision (FP16/BF16) Correctness Verdicts

Scope: numerically principled alternatives to a fixed elementwise `|npu − ref| ≤ 1e-5 + 1e-3·|ref|`, for an Intel NPU 3720 (FP16 compute, FTZ on FP16 subnormals, approximate transcendentals) compared against an FP32-intermediate reference rounded once to FP16. Failure classes used throughout:
(a) subnormal results returned as 0; (b) near-zero outputs from cancellation of larger terms; (c) accumulation error growing with length (sum/mean/cumsum/conv/matmul); (d) 2–13 ulp errors from multi-op chains and approximate transcendentals (softplus, erf/gelu, expm1 for elu/selu); (e) fused-op residues such as `x·fp16(1/6)+0.5` giving 2^-13 where the exact answer is 0.

Numeric constants used below (IEEE binary16 / bfloat16; verified locally with numpy `finfo` where noted):
- FP16: unit roundoff u = 2^-11 ≈ 4.883e-4; machine epsilon ε = 2^-10 ≈ 9.766e-4 (numpy `finfo(float16).eps = 0.000977`); smallest normal 2^-14 ≈ 6.104e-5 (`tiny`); smallest subnormal 2^-24 ≈ 5.96e-8; max finite 65504. For a normal x in [2^e, 2^(e+1)), ulp(x) = 2^(e−10), so ulp(x)/|x| ∈ (2^-11, 2^-10].
- BF16: u = 2^-8 ≈ 3.906e-3; smallest normal 2^-126 ≈ 1.18e-38 (same exponent range as FP32).
- FP32: u = 2^-24 ≈ 5.96e-8.
- What the current criterion means in ulps: for normal |ref|, 1e-3·|ref| is between 1e-3/2^-10 ≈ 1.02 and 1e-3/2^-11 ≈ 2.05 ulp(ref). So it is effectively a "1–2 ulp" test. The atol 1e-5 is about 168 FP16 subnormal steps (1e-5/2^-24 ≈ 167.8) and below the FP16 smallest normal. A flushed zero passes only if |ref| ≤ 1e-5/(1−1e-3) ≈ 1.001e-5, so every reference in (1.0e-5, 6.1e-5) that the device flushes is a spurious failure (computed locally).

---

## Q1. ULP-based accuracy specifications for half precision (OpenCL, Vulkan/SPIR-V, CUDA, DirectX/HLSL, TOSA) and how conformance tests treat FTZ

### Takeaway
None of the major APIs requires FP16 transcendentals to be correctly rounded. OpenCL Full Profile allows 2 ulp for exp/log/tanh, 4 ulp for erf, 1.5 ulp for sqrt and 1 ulp for division. Vulkan allows (1+2|x|) ulp for half exp and an *absolute* 2^-7 for log near 1 and for sin/cos. TOSA 1.0, the most ML-relevant spec, gives FP16 exp a relative bound of (1+2|x|)·2^-10, gives erf 5 ulp, and puts absolute error floors on LOG and TANH outputs near zero. Every spec allows FP16 subnormals to be flushed. The conformance checkers (OpenCL CTS, TOSA `tosa_reference_check_fp_bnd`) accept a zero result when the precise reference is subnormal, and the CTS also accepts results computed from flushed inputs.

### Cited Findings

**OpenCL C 3.0 (unified), Table 69 "ULP Values for Half-Precision Floating-Point Arithmetic Operations" (Full Profile / Embedded Profile)** — [OpenCL C spec §7.4](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html)
- x+y, x−y, x*y, fma: correctly rounded. 1.0/x: ≤1 ulp. x/y: ≤1 ulp. sqrt: ≤1.5 ulp. rsqrt: ≤1 ulp.
- exp, exp2, exp10, expm1: ≤2 ulp (Embedded: ≤3). log, log2, log10, log1p: ≤2 ulp (Embedded: ≤3).
- tanh: ≤2 ulp (Embedded ≤3). sinh/cosh: ≤2 (≤3). erf, erfc: ≤4 ulp (both profiles). pow/pown/powr/rootn: ≤4 ulp (≤5). sin, cos, sinpi, cospi: ≤2 ulp. tan: ≤2.5 ulp (≤3). atan, asin, acos, etc.: ≤2 ulp (≤3). tgamma: ≤4 ulp. mad: implementation-defined.
- Length-dependent *absolute* tolerances for reductions-in-a-builtin. `dot`: "absolute error tolerance of 'max * max * (2n − 1) * HALF_EPSILON', for vector width n and maximum input operand magnitude max". `cross`: "max * max * (3 * HALF_EPSILON)". `length`: ≤ 0.25 + 0.5n ulp. `distance`: ≤ 2n ulp. `normalize`: ≤ 1 + n ulp.
- `HALF_EPSILON` is defined as `0x1.0p-10h` (= 2^-10).
- The reference is "the infinitely precise result". The ULP definition is taken from Muller (INRIA RR-5504): ulp(x) = |b − a| for x strictly between consecutive floats a and b. Otherwise it is the distance between the two non-equal finite floats nearest x.
- If `cl_khr_fp16` is supported, half add, sub, mul and fma "are required to be correctly rounded using the default rounding mode."
- Denormals: "Support for denormal values is optional for half_ functions." The half_ functions "may return any result allowed by Edge Case Behavior, even when -cl-denorms-are-zero ... is not in force". Denormal support is implementation-defined for native_ functions. These notes cover the `half_`-prefixed reduced-accuracy float functions, which are distinct from the `half` data type.
- Half conversions: "Denormalized numbers for the half data type which may be generated when converting a float to a half using vstore_half and converting a half to a float using vload_half cannot be flushed to zero." Separately, for image `CL_HALF_FLOAT` channels, half denormals produced by float→half conversion "may be flushed to zero".

**OpenCL C §7.5.3 "Edge Case Behavior in Flush to Zero Mode" (verbatim structure)** — [OpenCL C spec](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html)
- "If denormals are flushed to zero, then a function may return one of four results:
  (1) Any conforming result for non-flush-to-zero mode;
  (2) If the result given by 1. is a sub-normal before rounding, it may be flushed to zero;
  (3) Any non-flushed conforming result for the function if one or more of its sub-normal operands are flushed to zero;
  (4) If the result of 3. is a sub-normal before rounding, the result may be flushed to zero."
- "In each of the above cases, if an operand or result is flushed to zero, the sign of the zero is undefined."
- "A non-zero number is said to be sub-normal before rounding if after normalization, its radix-2 exponent is less than (TYPE_MIN_EXP − 1)."

**OpenCL CTS math_brute_force acceptance rules** — [OpenCL-CTS math_brute_force README](https://github.com/KhronosGroup/OpenCL-CTS/blob/main/test_conformance/math_brute_force/README.txt)
- The device is compared "against results from a higher precision reference function". A result is correct if it is "(1) bitwise identical to the output of the reference function, rounded to the appropriate precision" or "(2) within the allowed ulp error tolerance of the infinitely precise result (as estimated by the reference function)". Rule (3): if the reference is NaN, any NaN is correct.
- "(4) if the devices is running in FTZ mode, then the result is also correct if the infinitely precise result (as estimated by the reference function) is subnormal, and the returned result is a zero."
- "(5) if the devices is running in FTZ mode, then we also calculate the estimate of the infinitely precise result with the reference function with subnormal inputs flushed to +- zero. If any of those results are within the error tolerance of the returned result, then it is deemed correct."
- "(6) half_func functions may flush per 4&5 above, even if the device is not in FTZ mode."
- Rules (7) and (8): premature overflow to infinity, or premature underflow, is allowed "so long as the estimated infinitely precise result is within the stated ulp error limit" of the max finite value or the min normal value.

**Vulkan "Precision and Operation of SPIR-V Instructions"** — [Vulkan spec, spirvenv appendix](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html); table source in [Vulkan-Docs spirvenv.adoc](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/appendices/spirvenv.adoc)
- Half-precision column, GLSL.std.450:
  - exp/exp2: **1 + 2·|x| ULP** (single precision: 3 + 2·|x| ULP).
  - log/log2: "3 ULP outside the range [0.5, 2.0]. Absolute error < 2^-7 inside the range [0.5, 2.0]" (single precision: 2^-21).
  - sin/cos: "Absolute error ≤ 2^-7 inside the range [−π, π]" (single: 2^-11).
  - inversesqrt: 2 ULP. sqrt: "Inherited from 1.0 / inversesqrt()".
  - atan/atan2: 5 ULP for half (4096 ULP for single).
  - pow: inherited from exp2(y × log2(x)). tanh: inherited from sinh()/cosh(), and sinh is inherited from (exp(x) − exp(−x))·0.5.
  - fma(): "Inherited from OpFMul followed by OpFAdd."
- Core ops: OpFAdd, OpFSub, OpFMul and OpFmaKHR are correctly rounded. OpFDiv is "2.5 ULP for |y| = 0 or |y| in the range [2^-14, 2^14]" for half. OpDot and the matrix products are "Inherited from Σ x_i × y_i".
- "Inherited from" semantics: the result must be within E = max(|x − F_min|, |x − F_max|) of the exact x. F_min and F_max range over the values permitted when the per-op rules are applied to the formula and to any associativity/commutativity/distributivity rearrangement of it.
- Denormals: "By default any denormalized floating-point value input into a shader or potentially generated by any instruction ... may be flushed to zero." Under DenormFlushToZero, "the denormalized result must be flushed to zero and the denormalized operands may be flushed to zero."
- Key FTZ boundary rule: "When denormal values are being flushed, the result of an operation may be considered denormal whenever the infinitely precise result is non-zero and of smaller magnitude than the smallest normal value, even if rounding would otherwise give a normal result."
- Reassociation: "Implementations may rearrange floating-point operations using any of the mathematical properties governing the expressions in precise arithmetic ... associativity and distributivity, and may involve a different number of rounding steps". Example: "implementations are allowed to implement a + b − a ... as b" unless the operation is decorated NoContraction.

**DirectX / HLSL** — [Direct3D 11 floating-point rules](https://learn.microsoft.com/en-us/windows/win32/direct3d11/floating-point-rules)
- 16-bit rules: "Unfused operations on 16-bit floating-point numbers produce a result that is the nearest representable value to an infinitely-precise result". "16-bit floating-point rules adhere to 0.5 ULP for unfused operations, and 0.6 ULP for fused operations". "16-bit floating-point numbers preserve denorms."
- 32-bit rules: "Denorms are flushed to sign-preserved zero on input and output of any floating-point mathematical operation."
- Fused-op principle: "Fused operations (such as mad, dp3) produce results that are no less accurate than the worst possible serial ordering of evaluation of the unfused expansion of the operation ... The individual steps in the unfused expansion are each allowed 1 ULP tolerance".
- 32-bit sqrt and rcp have 1 ULP tolerance.
- min16float (from search snippet only, page not fetched): minimum precision is a floor, and "the graphics driver might execute arithmetic on a min16float value at full 32-bit precision" — [Using HLSL minimum precision](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/using-hlsl-minimum-precision) (unverified excerpt).

**CUDA half intrinsics**
- The CUDA Math API "Half Math Functions" page describes `hexp` etc. as computing "in round-to-nearest-even mode" and lists special cases. I found **no published ulp bound** for the half functions — [CUDA Math API Half Math Functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html).
- An NVIDIA developer-forum thread asks exactly this. It notes that float and double have ULP tables in the Programming Guide but half does not, and the visible snippet contains no official answer — [NVIDIA forum](https://forums.developer.nvidia.com/t/does-the-ulp-error-standard-for-half-precision-floating-point-mathematical-functions-seem-to-be-missing/315885).

**TOSA 1.0.2 (Arm/MLPlatform tensor operator spec; directly applicable to NPUs)** — [TOSA 1.0.2 spec](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html)
- fp16_t/bf16_t/fp32_t: "Subnormal values must either be supported or flushed to sign-preserved zero."
- Consistency rule: "For data types that allow subnormal values to be flushed to zero, either all values must be flushed to sign-preserved zero, or none of them."
- Per-op text: "Subnormal bf16_t, fp16_t, and fp32_t input values may be flushed to zero before calculation". For some ops it adds that result values "may be flushed to zero of the appropriate sign after the calculation".
- "These requirements allow fp16_t operations to be implemented using the fp32_t datatype."
- Reference: "out_ref be the result calculated using fp64_t arithmetic". "The operation order must follow that specified in the operator pseudocode".
- Helper `calcAbsErrorBound<in_t>(mag, bounds_value, lower_bound, normal_divisor)`: value_bound = max(|mag|, normal_min). If lower_bound > 0, value_bound = max(lower_bound/bounds_value, value_bound). The bound is 2^(−normal_frac/normal_divisor) · value_bound · bounds_value, with normal_frac(fp16) = 10, bf16 = 7, fp32 = 23.
- FP16 instantiations:
  - **EXP**: err_bnd = 2^-10·max(|out_ref|, nm)·(err_base + 2|x|), with err_base = 3 for fp32 and 1 otherwise.
  - **LOG**: calcAbsErrorBound(out_ref, 5, 0.5, 1), i.e. 5·2^-10·|out_ref| with an absolute floor of 0.5·2^-10 = 2^-11.
  - **TANH**: calcAbsErrorBound(out_ref, 4·(exp_err_base + 2|x|), 0.5, 1). The informational note says this is "derived from the error bound for the exp operator based on implementation of TANH as (exp(x) − exp(−x)) / (exp(x) + exp(−x))". "The absolute lower bound of 0.5 ulp relative to 1.0 is due to exp(x) − exp(−x) having an error of this magnitude when x is close to 0."
  - **SIGMOID**: calcAbsErrorBound(out_ref, 2·(1+|x|), 0, 1).
  - **ERF**: `tosa_reference_check_fp(out_imp, out_ref, 5)`, i.e. 5 ulp.
  - **RECIPROCAL**: 1 ulp. **RSQRT**: 2 ulp. **ADD/SUB/MUL/CAST/FLOOR/CEIL**: 0.5 ulp.
  - **SIN**: max(calcAbsErrorBound(x, 2, 0, 2), π·normal_min). **COS**: calcAbsErrorBound(1+|x|, 1, 0, 2). normal_divisor = 2 means a 2^-5 factor for fp16.
  - **POW**: composed from the EXP, MUL and LOG bounds.
  - **REDUCE_PRODUCT**: max(|out_ref|, nm)·((1 + 2^-(frac+1))^n − 1).
- `tosa_reference_ulp`: ulp = max(2^⌊log2|ref|⌋, normal_min)·2^-frac (for fp16, ulp of a subnormal ref = 2^-24). `tosa_reference_check_fp_bnd` accepts `test_value == 0` iff `ref_min < normal_min` (with ref_min = |ref| − err_bnd). This is explicitly the rule "Allow subnormal values to be flushed to zero".
- "Operator sequence precision requirement": an implementation of a sequence A[0..n−1] is compliant if it "gives the same result as a sequence of implementations M[0] to M[n−1] such that each M[k] implements A[k] with same or higher precision datatypes" and meets A[k]'s accuracy, "where the M[k] output is converted to A[k] output precision using round to nearest".

### Inferences
- Implication for (d): a 1–2 ulp test is stricter than every published FP16 spec for transcendentals. OpenCL allows erf 4 ulp; TOSA allows erf 5 ulp, EXP (1+2|x|)·2^-10 relative (≈ 2(1+2|x|) ulp at worst), and TANH 4(1+2|x|)·2^-10 relative. Vulkan allows half exp (1+2|x|) ulp. A composite like softplus = log(1+exp(x)) or gelu = 0.5·x·(1+erf(x/√2)) should be judged as an *operator sequence* (TOSA's rule): each primitive's tolerance is propagated through the chain. A flat ulp budget is not appropriate. Observed 2–13 ulp errors on chains are within what an "inherited from formula" (Vulkan) or "operator sequence" (TOSA) analysis would permit, and should be checked against that composed bound.
- Implication for (a): every relevant spec (OpenCL §7.5.3, CTS rules 4–6, Vulkan default, TOSA `check_fp_bnd`) accepts 0 for a subnormal-before-rounding exact result. Vulkan explicitly extends this to results whose exact value is < 2^-14 even if they would round up to 2^-14. A harness can therefore accept `npu == ±0` whenever |ref_exact| < 2^-14 (+ the op's error bound), ignoring the sign of zero.
- CUDA cannot be used as a source of FP16 ulp numbers. Use OpenCL Table 69, Vulkan's half column, or TOSA.
- Which spec to adopt: TOSA's per-operator bounds are the closest ready-made "spec sheet" for an NPU FP16 verdict. They are expressed with `normal_min` floors and an fp64 reference, which matches this harness's structure (FP32+ reference, compared at FP16).

### Gaps
- Vulkan's exact precision rule for `RelaxedPrecision`-decorated 32-bit ops was not found in the fetched appendix. The table header says "Single precision, unless decorated with RelaxedPrecision", which suggests relaxed ops fall back to the half column (unverified).
- No official CUDA half-precision ulp table was found (as of the pages fetched, Oct 2026).
- The exact `min16float` precision and denorm guarantees were not confirmed from a fetched primary page.
- Intel NPU 3720 / OpenVINO NPU-plugin documented accuracy per op: not researched here (out of scope; see sibling notes).

---

## Q2. Forward error bounds for summation, dot products, matmul and convolution, and turning them into per-element tolerances

### Takeaway
The principled tolerance for any linear reduction is `|ŷ_i − y_i| ≤ c(n)·u·M_i`, where M_i is the *same operator applied to absolute values* (|A||B| for matmul, conv(|x|,|w|)+|b| for conv, Σ|x| for sum/mean/cumsum). c(n) is ~n (worst case, γ_n), ~λ√n (probabilistic, Higham–Mary) or ~O(1) (blocked or FP32 accumulation, mean-zero data). Two production test suites implement exactly this: the reference BLAS test ratio `|C − Ĉ|/(eps·(|α||A||B| + |β||C|))` with threshold 16, and TOSA's `tosa_reference_check_dotproduct`, which adds statistical mean and variance checks. oneDNN benchdnn and CUTLASS avoid the issue by using integer data that makes results exact.

### Cited Findings

**Deterministic and probabilistic theory**
- Classical analysis uses γ_n = nu/(1 − nu). Higham & Mary state that these bounds "can offer little useful information for large-scale or low-precision computations". Their probabilistic analysis (independent, mean-zero rounding errors) replaces γ_n with a relaxed constant γ̃_n proportional to √(n log n)·u (or λ√n·u), which holds with probability bounded below independently of n. Published in SIAM J. Sci. Comput. 41(5):A2815–A2835 (2019) — [Higham & Mary, MIMS eprint](https://eprints.maths.manchester.ac.uk/2731/); [HAL](https://hal.archives-ouvertes.fr/hal-02311269) (abstract via search snippet).
- Hallman & Ipsen's restatement: max_k |∏_{j>k}(1+δ_j) − 1| ≤ γ̃_n(δ) = exp((λ(δ)√n·u + n·u²)/(1−u)) − 1 "with failure probability at most δ". Their refined summation bound is |E_n| ≤ u‖s_n‖·λ(δ/2)·(1+γ̃_n(δ/2)). The bounds "hold to all orders" and are useful while λ√n·u < 1.
- Fp16 specifics from the same paper: u ≈ 5×10⁻⁴. With λ ≈ 9, λ√n·u ≈ 1 at n ≈ 5.2×10⁴ for fp16 and n ≈ 810 for bfloat16. Beyond that point recursive summation has "little relative accuracy in practice" — [Hallman & Ipsen, arXiv:2104.06531](https://arxiv.org/abs/2104.06531).
- Mary's slides give the rule of thumb: if the normalized error is ≈ N(0, u), then |error| ≤ λ√n·u with high probability, e.g. 99.7% for λ = 3 — [Mary, CSE19 slides](https://www-pequan.lip6.fr/~tmary/doc/CSE19ProbErrAna.pdf) (search snippet).
- Random data: "for data with zero or small mean, the existing probabilistic bounds of order √n·u can be relaxed to much sharper bounds of order u, independent of n". The fundamental result is for summation, extended to basic linear algebra kernels — [Higham & Mary, "Sharper Probabilistic Backward Error Analysis for Basic Linear Algebra Kernels with Random Data"](https://eprints.maths.manchester.ac.uk/2778) (search snippet).
- Blocking: "a blocked algorithm has an error bound about a factor of b smaller than that for the corresponding unblocked algorithm"; practical block sizes are 128 or 256. With FABsum "the error constant is further reduced to bu + O(u²)". Combined with the probabilistic analysis, blocked inner products have "a probabilistic bound of order √(n/b)·u" — [Higham blog, 2021](https://nhigham.com/2021/09/14/can-we-solve-linear-algebra-problems-at-extreme-scale-and-low-precisions/).
- FABsum (fast summation inside blocks, accurate summation across blocks) has a backward error bound of (b+1)u + O(u²), independent of n to first order — [Blanchard, Higham & Mary, SIAM J. Sci. Comput. 2020](https://epubs.siam.org/doi/pdf/10.1137/19M1257780); [HAL](https://hal.archives-ouvertes.fr/hal-02491081) (via search snippet).
- Mixed-precision block FMA (Blanchard, Higham, Lopez, Mary, Pranesh, SIAM J. Sci. Comput. 42(3):C124–C141, 2020, DOI 10.1137/19M1289546): the framework generalizes scalar FMA and tensor units. On V100, "the TC32 variant is much more accurate than the TC16 one" — [HAL hal-02491076](https://hal.archives-ouvertes.fr/hal-02491076) (abstract via search snippet).
- Mary's slides on tensor cores: Volta uses block 4×4×4 with fp16 inputs and fp32 internal precision. Tensor cores "only support round-to-zero", so the worst-case n·u_32 accumulation bound is attained, whereas round-to-nearest usually gives ≈ √n·u_32 — [Mary, SIAM LA21 slides](https://www-pequan.lip6.fr/~tmary/doc/SIAM_LA21_multiword.pdf) (search snippet).

**Test suites that use an |A||B| "absolute-value reference"**
- **Reference BLAS (dblat3.f, DMMCH)**:
  - Computes G(I) = Σ_k |A(I,K)|·|B(K,J)|, then G(I) = |ALPHA|·G(I) + |BETA|·|C(I,J)|.
  - Then ERRI = |CT(I) − CC(I,J)| / EPS, divided by G(I) if G(I) ≠ 0, and ERR = max ERRI.
  - Raises "FATAL ERROR - COMPUTED RESULT IS LESS THAN HALF ACCURATE" if ERR·√EPS ≥ 1.
  - The routine passes if ERRMAX < THRESH. The default input file comment is "16.0 THRESHOLD VALUE OF TEST RATIO".
  - Source: [Reference-LAPACK BLAS/TESTING/dblat3.f](https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/TESTING/dblat3.f).
- **TOSA `tosa_reference_check_dotproduct`**. Applies to CONV2D/3D, DEPTHWISE, TRANSPOSE_CONV, MATMUL, FFT, and REDUCE_SUM/AVG_POOL2D, which are expressed as dot products with weights 1 or 1/KS — [TOSA 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html):
  - `input_abs = max(|input|, normal_min)`, likewise for weight and bias. If `local_bound` is false (the default, to allow Winograd/FFT fast convolution), every input_abs element is replaced by the global max.
  - `output_ref = reference_fp64(OP, input, weight, bias)`; `output_bnd = reference_fp64(OP, input_abs, weight_abs, bias_abs)`.
  - `ksb = ceil(KS / 2^((normal_frac(acc_t) − normal_frac(out_t))/2)) + (bias ? 1 : 0)`.
  - Per element: `out_err_bnd = max(out_bnd·2^(−1−normal_frac(out_t)), normal_min(out_t))`, i.e. u_out·Σ|a||w|; `out_err = (imp − ref)/out_err_bnd`; fail if `|out_err| > ABS_BOUND = 2·ksb` ("Allow one rounding error for each accumulation. The 2 factor allows for different rounding modes").
  - Special cases: if out_bnd == 0 (all products zero), both ref and imp must be exactly 0. If the bound overflows, there is no limit. A NaN ref requires a NaN imp.
  - Statistical checks over the T outputs: Σ out_err² ≤ VARIANCE_ERROR_BOUND·T with VARIANCE_ERROR_BOUND = 4·0.4·ksb ("0.4 factor is derived from the uniform [−1,1] distribution variance of 1/3 by rounding up"). For test sets S = 3..5, a bias check |Σ out_err| ≤ √(10·VARIANCE_ERROR_BOUND·T) ("up to a 4 sigma difference").
  - Data sets: MIN_DOT_PRODUCTS = 1000 outputs. Generators are S=0 (products with zero give zero), S=1 (large exponents, scaled by B/√(KS+1) with B = 255.875 for fp16→fp16 and 65504 for fp16→fp32), S=2/S=3 (small values accumulated onto a large first value, of similar or varying magnitude), S=4 (mix of zero and non-zero products) and S=5 (signed inputs of large range). KS definitions: CONV2D KS = KW·KH·IC, MATMUL KS = C, REDUCE_SUM KS = shape[axis], AVG_POOL2D KS = KX·KY with weights 1/KS.
- **oneDNN benchdnn**:
  - Convolution fill draws integers from `uniform_int_distribution(range_min, range_max)`. FP16 ranges are e.g. {−4,4}, {−2,2} or {−8,8} per tensor kind.
  - A Bernoulli "density" of nonzeros is lowered so that the accumulation chain is "safe": `safe_n_acc = 2^safe_digits / (max|src|·max|wei|)` with `safe_digits = min(digits(dst dt), digits(accumulation mode))`. f16 accumulation mode uses digits_dt(f16), and "relaxed/any" modes assume f16 because "In practice, the minimum accumulation type is f16".
  - Direct convolution then sets threshold `trh = 0` (exact match). Only Winograd uses `trh = 7e-3` (f16) or `2e-5`, with an empirical K-growth factor for backward-weights.
  - The comparator rounds the f32 reference to the output dt first (`exp = round_to_nearest_representable(dt, exp_f32)`). It passes if |exp| ≤ 1e-5 and diff < eps_f32, else if (|exp| > 1e-5 ? rel_diff : abs_diff) ≤ trh.
  - A "zero trust percent" check flags MISTRUSTED results when too many outputs are zero (matmul uses 90%).
  - Sources: [benchdnn conv.cpp](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/conv/conv.cpp), [conv/cfg.cpp](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/conv/cfg.cpp), [utils/cfg.hpp](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/utils/cfg.hpp), [utils/compare.cpp](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/utils/compare.cpp), [matmul.cpp](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/matmul/matmul.cpp), [benchdnn_general_info.md](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/benchdnn_general_info.md).
- oneDNN defaults to f32 accumulation for floating-point primitives. The `accumulation_mode` attribute can select f16 accumulators, and "relaxed" mode "allows some partial accumulators to be rounded to src/dst datatype in memory" — [oneDNN accumulation-mode docs](https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2024-1/primitive-attributes-accumulation-mode.html) (search snippet).
- **CUTLASS GEMM device testbed**:
  - Uniform fill scope is [−5,5] when the output is 16-bit, [−8,8] otherwise, and [−1,1] for ≤8-bit inputs. It calls `TensorFillRandomUniform(view, seed, scope_max, scope_min, 0)`, where the last argument is the number of fractional bits, so values are integers.
  - It asserts the reference norm is > 0 and then requires `TensorEquals(reference_D, tensor_D)`, i.e. bit-exact equality — [CUTLASS test/unit/gemm/device/testbed.h](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/testbed.h).
  - The CUTLASS utilities doc says restricting to "integers or values with exact fixed-point representations" "simplifies functional comparisons" — [CUTLASS utilities doc](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/utilities.html) (search snippet).
- OpenCL's half `dot` builtin tolerance is max²·(2n−1)·2^-10: an input-magnitude-based, length-scaled absolute bound — [OpenCL C Table 69](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html).

### Inferences
- Textbook constants. These are standard results (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Ch. 3–4) that I did **not** re-verify from the primary text in this session:
  - Recursive summation: |ŝ − s| ≤ γ_{n−1}·Σ|x_i|.
  - Inner product: |fl(xᵀy) − xᵀy| ≤ γ_n·|x|ᵀ|y|.
  - Matmul: |Ĉ − C| ≤ γ_n·|A||B|.
  - Pairwise summation: γ_{⌈log2 n⌉}·Σ|x_i|.
  - Blocked with block b: ≈ γ_{b+n/b−1}.
  - Compensated (Kahan): (2u + O(nu²))·Σ|x_i|.
  - TOSA's ABS_BOUND = 2·KS·u_out·Σ|a||w| for fp16 accumulation is a 2γ_KS-type deterministic bound, consistent with this theory.
- Per-element tolerance recipe for (c):
  1. Compute `ref = op(x, w)` in FP64/FP32.
  2. Compute `M = op(|x|, |w|) (+|b|)` with the same operator. For sum/mean/cumsum, M = sum/mean/cumsum of |x|. For conv, use conv with |w|. For softmax-denominators, use Σ exp. For `mean`, M = Σ|x_i|/n.
  3. Set `tol_i = c·u16·M_i + k·ulp16(ref_i)`. The ulp term covers the final rounding to FP16.
  4. Choose c by accumulation model: c ≈ 2·n for FP16 sequential accumulation (TOSA's 2·ksb with ksb = KS); c ≈ 2·⌈n/90.5⌉ when accumulating in FP32 and rounding to FP16 (TOSA's ksb formula with frac 23 vs 10 gives 2^6.5 ≈ 90.5); or a probabilistic c ≈ λ√n with λ ≈ 3–9 for RN accumulation. Use the deterministic c ≈ n if the device may use round-to-zero, since RZ attains the worst case per Mary's tensor-core observation.
  5. Record the normalized error e_i = (npu_i − ref_i)/(u16·M_i). A per-element check of max|e_i| plus TOSA-style RMS and mean checks distinguish benign growth (RMS ∝ √n, mean ≈ 0) from systematic bias (truncation/RZ accumulation, mean ≠ 0).
- Why this fixes (b) for reductions: when the true sum is near zero because of cancellation, M_i = Σ|terms| stays large. The tolerance is then set by the resolution of the large terms, not of the small output. This is precisely what the BLAS ratio and TOSA's out_bnd do.
- Inferring the NPU's accumulator: fit the observed RMS of e_i against n. Slope ≈ 0.5 in log–log suggests RN accumulation (√n). Slope ≈ 1 suggests RZ or a deterministic worst case. A plateau suggests FP32 or blocked accumulation. That tells you which c(n) to use.
- Exact-data alternative: choose integer inputs so every partial sum is exactly representable, then require bit-exactness (benchdnn trh = 0, CUTLASS TensorEquals). For FP16 accumulation this means Σ|a||w| ≤ 2^11 = 2048 (FP16 integers are exact up to 2048). Examples: values in [−4,4] with KS·16 ≤ 2048, i.e. KS ≤ 128, or use sparsity/density as benchdnn does. For FP32 accumulation with FP16 output, only |final| ≤ 2048 is needed. This cleanly separates "wrong indexing or logic" bugs from rounding.

### Gaps
- The exact statement of Blanchard et al.'s block-FMA theorem was not retrieved (HAL and lip6 PDFs were blocked or failed). From memory, the bound has the form |C − Ĉ| ≤ (2u_low + u_low² + n·u_high + O(u_high²))·|A||B| for fp32 inputs converted to fp16 with fp32 accumulation (TC32). The 2u_low term disappears if inputs are already fp16. **Unverified; do not quote without checking the paper.**
- The precise Higham–Mary (2019) Theorem 2.4 probability expression was not retrieved. From memory it is P(λ) = 1 − 2exp(−λ²(1−u)²/2) per product, with λ√n·u + O(u²) first order. **Unverified.**
- The venue and year of the "Sharper" paper were not verified in this session.
- No source was found describing the Intel NPU 3720's actual accumulator precision or rounding mode for conv/matmul/reduce.
- MAGMA and cuBLAS internal test tolerances were not researched.

---

## Q3. Condition-number-based tolerance for elementwise functions

### Takeaway
For y = f(x) evaluated on an already-rounded FP16 input, the best achievable error is about κ_f(x)·u_in·|y| + 0.5·ulp(y), where κ_f(x) = |x·f'(x)/f(x)|. Boost.Math's accuracy envelope uses max(0.5, ½·κ) ulps and notes that κ is unbounded at roots. That is why near-zero outputs need an *absolute* term proportional to |x·f'(x)|·u rather than a relative ulp test. TOSA hard-codes such floors for LOG/TANH and an |x|-proportional term for EXP.

### Cited Findings
- Boost.Math derives the envelope by modelling the input as x̂ = x(1+ε), |ε| < μ. A first-order Taylor expansion gives "the best accuracy we can hope for". The envelope is "the maximum of 0.5 and half the condition number of function evaluation at each point", and "the condition number of function evaluation becomes unbounded at the roots" — [Boost.Math ULPs plots](https://www.boost.org/latest/libs/math/doc/html/math_toolkit/ulps_plots.html).
- A university lecture on numeric testing recommends that test tolerances depend on both ULP and the function's condition number, against a high-precision (e.g. GMP) reference — [Liverpool COMP285 lecture 11](https://www2.csc.liv.ac.uk/~coopes/comp285/notes/lecture011_numeric_testing.pdf) (search snippet; exact formula there not verified).
- CPython's `test_math` passes a value if *either* the absolute error *or* the ulp error is within tolerance (a combined criterion) — [CPython test_math.py](https://android.googlesource.com/platform/external/python/cpython3/+/cfca4a6774/Lib/test/test_math.py) (search snippet).
- TOSA encodes condition-number-like terms directly — [TOSA 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html):
  - EXP's relative bound grows as (1 + 2|x|)·2^-10. κ_exp(x) = |x|, so this is κ·(2ε) plus a base term.
  - TANH has relative 4(1+2|x|)·2^-10 with an absolute floor of 0.5·2^-10 near x = 0.
  - LOG has relative 5·2^-10 with an absolute floor of 0.5·2^-10. κ_log(x) = 1/|log x| blows up at x = 1, where log x → 0.
  - SIN's bound is proportional to |x|.
- Vulkan uses *absolute* error for half log on [0.5, 2] (< 2^-7) and half sin/cos on [−π, π] (≤ 2^-7), instead of ULP, in exactly the regions where the outputs cross zero — [Vulkan spirvenv.adoc](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/appendices/spirvenv.adoc).

### Inferences
- Recipe for scalar f:
  `tol(x) = κ_f(x)·u16·|f(x)| + k·ulp16(f(x))`.
  Equivalently, in absolute form, which stays finite at roots of f:
  `tol(x) = |x·f'(x)|·u16 + k·ulp16(f(x)) + floor`,
  where k is the spec ulp budget (e.g. 2 for exp/log/tanh per OpenCL, 4–5 for erf) and floor = 0 or ~2^-24.
- κ for the failing ops (derived analytically; not from a cited source):
  - softplus(x) = log(1+eˣ): κ = |x|·σ(x)/softplus(x), which tends to |x|·eˣ/eˣ = 1 for x → −∞.
  - elu/selu negative branch α(eˣ−1): κ = |x|eˣ/|eˣ−1| → 1 as x → 0⁻.
  - gelu(x) = x·Φ(x): the derivative is Φ(x) + xφ(x), so κ = |1 + xφ/Φ|.
  - erf(x): κ = |x·(2/√π)e^{−x²}/erf(x)| → 1 as x → 0.
  - The absolute form avoids division by f(x) ≈ 0.
- For multi-op chains (d) and fused residues (e), generalize to first-order (linearized) error propagation over the dataflow graph. For each intermediate z_j that is rounded (with unit roundoff u_j) or approximated (with error budget e_j·u_j), add |∂y/∂z_j|·|z_j|·(u_j or e_j·u_j). Constants that are themselves rounded to FP16 (e.g. 1/6, √2, 1/√2, SELU α/λ) count as intermediates.
  - This is the scalar analogue of the |A||B| reference, and the same as Vulkan's "inherited from formula" idea.
  - Computation: evaluate ∂y/∂z_j with forward-mode AD in float64, or pass |·| through the linear parts.
- Worked example for (e), verified numerically with numpy in the project venv:
  - fp16(1/6) = 0.1666259765625 = 1365·2^-13, with representation error −4.07e-5.
  - For x = −3, x·fp16(1/6) + 0.5 = 1.2207e-4 = exactly 2^-13 (computed in FP32), while the exact answer is 0.
  - Propagated constant-rounding term: |x|·|c|·u16 = 3·(1/6)·2^-11 ≈ 2.44e-4 ≥ 2^-13, so the result passes with k = 1. Equivalently, the |terms| magnitude is |x·c| + 0.5 = 1.0 and u16·1.0 = 4.88e-4.
  - A relative or ulp test against ref = 0 can never pass, and an atol of 1e-5 is 12× too small.
- Worked logic for (b), elementwise cancellation (e.g. x − y, x·σ(x) near 0, 1 − tanh², sub in layernorm): use M = |x| + |y| (the sum of magnitudes of the terms being cancelled), giving tol = c·u16·M. This is the same principle as Q2.

### Gaps
- I found no published ML/NPU test suite that computes κ_f(x) per element at runtime. Boost uses it only for plots, and TOSA bakes κ-like terms into fixed formulas. The per-element κ/AD approach above is a synthesis, not an established standard.
- The Liverpool lecture's exact formula (reported in a snippet as "Cn × ULP × f(x)/x") was not verified and may be misquoted.

---

## Q4. Device-emulating references (chop, CPFloat, MPFR, QPyTorch, numpy float16 step-by-step), FTZ-aware references, and the order/fusion trade-off

### Takeaway
Simulators such as CPFloat (C) and chop (MATLAB) round higher-precision results to an arbitrary target format, with flags for subnormal support and rounding mode. They can build a reference that mimics FP16 storage, FTZ and per-op rounding. But an emulated reference is only "tighter" if it reproduces the device's operation order, fusion and accumulator precision, which specs explicitly allow implementations to change (Vulkan reassociation, TOSA's "same or higher precision" sequence rule). The robust design is to keep an exact (FP64) reference with principled bounds, and use emulation for diagnosis: bracketing several plausible device models.

### Cited Findings
- CPFloat is a C library "for simulating low-precision floating-point arithmetics" with routines for rounding, arithmetic and math functions. A target format is specified by precision, exponent range and "a Boolean variable σ, set to true if subnormal are supported and to false otherwise". Reference: Fasi & Mikaitis, ACM TOMS 49(2), Article 18, 2023 — [CPFloat README](https://github.com/north-numerical-computing/cpfloat); [DOI 10.1145/3585515](https://doi.org/10.1145/3585515).
- The CPFloat paper notes that subnormals, underflow and overflow need special handling, and reports roughly an order-of-magnitude speedup from bit-level manipulation — [Manchester preprint](https://eprints.maths.manchester.ac.uk/2785/) (search snippet).
- Higham & Pranesh, "Simulating Low Precision Floating-Point Arithmetic", SIAM J. Sci. Comput. 41:C585–C602 (2019), provides MATLAB `chop` — [DOI 10.1137/19M1251308](https://doi.org/10.1137/19M1251308); [chop repo](https://github.com/higham/chop). It can simulate "fp16, bfloat16, and other low precision arithmetics, with or without the representation of subnormal numbers and with the options of round to nearest, directed rounding, stochastic rounding, and random bit flips in the significand" — [ECP mixed-precision survey, arXiv:2007.06674](https://arxiv.org/abs/2007.06674).
- Specs permit order and fusion changes:
  - Vulkan allows rearrangement by associativity and distributivity with "a different number of rounding steps" unless NoContraction — [Vulkan spirvenv](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).
  - TOSA allows each op in a sequence to be implemented "with same or higher precision datatypes" and allows fast convolution (local_bound = false) — [TOSA 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html).
  - D3D defines fused-op accuracy relative to "the worst possible serial ordering" — [D3D11 floating-point rules](https://learn.microsoft.com/en-us/windows/win32/direct3d11/floating-point-rules).
- TOSA's fp64 reference "must follow that specified in the operator pseudocode" for operation order — [TOSA 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html).
- TOSA's test-set note for S=2 and S=3: "If the implementation changes the order of the sum, then the test data must also be reordered so that the largest values occur first in the sum." This shows that bounds tight enough to catch accumulation bugs depend on the summation order — [TOSA 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html).
- oneDNN benchdnn rounds its f32 reference to the output dt before comparing. This is single-rounding, the same structure as this harness's "FP32+ then round once" reference — [benchdnn compare.cpp](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/utils/compare.cpp).

### Inferences
- FTZ-aware reference recipe (mirrors OpenCL CTS rules 4–5 and OpenCL §7.5.3):
  - Compute up to three references: R0 = exact (FP64 on the FP16 inputs); R1 = exact on inputs with FP16 subnormals flushed to ±0; R2 = R0 or R1 with any |result| < 2^-14 replaced by 0.
  - Accept if the device output is within tolerance of any of them.
  - This is cheap (two forward passes) and exactly matches what conformance suites allow.
- Device-model bracketing for diagnosis, not the verdict:
  - Emulate the op step by step in FP16, e.g. with numpy float16 or CPFloat (σ = false for FTZ), under a few plausible models: FP16 sequential accumulation; FP32 accumulation then a single round; pairwise or blocked accumulation; constants rounded to FP16; transcendentals approximated by exp/log compositions.
  - If the device matches one model bit-exactly or within 1 ulp, the failure is explained as "different but legal" rounding. If it matches none and also violates the Q2/Q3 bound, it is a real bug.
- Trade-off: a device-emulating reference with the wrong order can be *further* from the device than the exact reference. Use emulation to choose the c(n) constant and the accumulator model in the bound, not as the pass/fail reference.
- For (e): emulating "constant rounded to FP16" (fp16(1/6)) reproduces the 2^-13 residue exactly. This confirms the residue comes from constant quantization (legal under TOSA's "operator sequence" model, where the constant is an FP16 tensor) rather than from a kernel bug.

### Gaps
- QPyTorch, MPFR/gmpy2-based references and FloatX were not researched (no sources fetched).
- No source was found on how OpenVINO or Intel NPU lowers HSigmoid/HSwish, i.e. whether it uses the constant fp16(1/6) or a division by 6. That would confirm the (e) mechanism.

---

## Q5. Treatment of subnormals: IEEE semantics, FTZ/DAZ, acceptance of flushed results, and implications for absolute tolerances

### Takeaway
All relevant FP16 specs allow FTZ: OpenCL, Vulkan, TOSA, and the D3D 32-bit rules. D3D's 16-bit rules are the exception and require denorms to be preserved. Conformance checkers accept a zero output when the exact result is below the smallest normal (2^-14 ≈ 6.1e-5 for FP16). TOSA's checker implements this as `test == 0 → pass iff (|ref| − err_bnd) < normal_min`. A global atol of 1e-5 is therefore inconsistent with FTZ semantics. The right fix is an FTZ acceptance rule, not a bigger atol, because a global atol of 6.1e-5 would also loosen every small *normal* result.

### Cited Findings
- TOSA: fp16/bf16/fp32 subnormals "must either be supported or flushed to sign-preserved zero", and FTZ must be all-or-nothing. `tosa_reference_check_fp_bnd` contains `if (test_value == 0) return (ref_min < normal_min());` under the comment "Allow subnormal values to be flushed to zero for non-fp8". Dot-product error bounds have the floor `max(out_bnd·2^(−1−frac), normal_min(out_t))` — [TOSA 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html).
- OpenCL CTS: "(4) if the devices is running in FTZ mode, then the result is also correct if the infinitely precise result ... is subnormal, and the returned result is a zero". Rule (5) re-evaluates with subnormal inputs flushed. "(6) half_func functions may flush per 4&5 above, even if the device is not in FTZ mode" — [OpenCL-CTS README](https://github.com/KhronosGroup/OpenCL-CTS/blob/main/test_conformance/math_brute_force/README.txt).
- OpenCL C §7.5.3: four permitted results under FTZ, and "if an operand or result is flushed to zero, the sign of the zero is undefined". Subnormal is defined as "0 < x < TYPE_MIN and −TYPE_MIN < x < −0" — [OpenCL C spec](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html).
- Vulkan: by default denormals in inputs or results "may be flushed to zero". When flushing, a result "may be considered denormal whenever the infinitely precise result is non-zero and of smaller magnitude than the smallest normal value, even if rounding would otherwise give a normal result" — [Vulkan spirvenv](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).
- D3D11: 32-bit "Denorms are flushed to sign-preserved zero on input and output". "16-bit floating-point numbers preserve denorms" — [D3D11 floating-point rules](https://learn.microsoft.com/en-us/windows/win32/direct3d11/floating-point-rules).
- PyTorch's default `assert_close` tolerances: float16 (rtol, atol) = (1e-3, 1e-5); bfloat16 (1.6e-2, 1e-5); float32 (1.3e-6, 1e-5). The harness's current criterion is exactly PyTorch's FP16 default — [torch/testing/_comparison.py](https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py).
- benchdnn treats tiny values specially: values with |exp| ≤ 1e-5 pass if diff < eps_f32. The comment says "Discard tiny values very close to each other. It's impossible to compare them reliably and fit into any criterion" — [benchdnn compare.cpp](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/utils/compare.cpp).

### Inferences
- FTZ rule for (a):
  - Output side: pass if npu == ±0 and |ref| < 2^-14 + tol(ref). Ignore the sign of zero (OpenCL: sign undefined; TOSA: sign-preserved, so a stricter harness can check the sign).
  - Input side: also accept results computed from inputs with subnormals flushed (CTS rule 5).
  - The npu may also return a subnormal value. Per TOSA, either all subnormals flush or none do, so a mix of flushed and preserved values in one tensor is a finding worth logging.
- Absolute-tolerance implications:
  - In FP16 the smallest representable step near zero is 2^-24 ≈ 6e-8 (with subnormals) or 2^-14 ≈ 6.1e-5 (with FTZ).
  - Any atol for FP16 outputs should be expressed in units of the *input or term magnitude* (u16·M, Q2/Q3), not as a fixed 1e-5. The fixed 1e-5 is simultaneously too strict for FTZ (< 2^-14) and for cancellation residues (2^-13 ≈ 1.22e-4 in (e)), and meaningless for large-magnitude terms.
- For BF16 the smallest normal is ~1.18e-38, so FTZ almost never matters in practice. The dominant issue is u = 2^-8: relative 1e-3 is *below* 1 ulp for BF16, which is why PyTorch uses rtol = 1.6e-2 (≈ 4u, i.e. 2 ulp in the worst case).

### Gaps
- The Intel NPU 3720's exact FTZ behaviour (output-only FTZ vs DAZ on inputs, sign of zero) was not found in any primary source. It should be characterized empirically with a sweep of subnormal inputs and outputs.

---

## Q6. Statistical, backward-error and metamorphic testing approaches for numerical kernels

### Takeaway
Beyond per-element bounds, three established techniques make verdicts robust:
1. Statistical checks on the normalized error distribution. TOSA requires Σe² ≤ 1.6·ksb·T and a bias bound on Σe, which catches systematic rounding bias that per-element bounds miss.
2. Backward-error style ratios normalized by |A||B| (the BLAS test ratio with threshold 16).
3. Exact-arithmetic test data (integer inputs with bounded accumulation), which turns the check into bit-exact comparison (benchdnn, CUTLASS).

Probabilistic theory (Higham–Mary) explains why RMS error grows like √n·u for RN and is O(u) for mean-zero data.

### Cited Findings
- TOSA statistical compliance:
  - The normalized error e = (imp − ref)/(u_out·Σ|a||w|) must satisfy |e| ≤ 2·ksb per element and Σe² ≤ 4·0.4·ksb·T over T ≥ 1000 outputs. For S = 3..5, |Σe| ≤ √(10·VARIANCE_ERROR_BOUND·T), "assuming errors are normally distributed".
  - Six structured data sets target zero products, large exponents, small-onto-large accumulation, mixed zeros and wide signed range.
  - Source: [TOSA 1.0.2](https://www.mlplatform.org/tosa/tosa_spec_1_0_2.html).
- BLAS testing ratio: max |CT − CC|/(EPS·G), with G = |α||A||B| + |β||C|. Pass threshold 16, and fatal if ERR·√EPS ≥ 1 ("less than half accurate") — [dblat3.f](https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/TESTING/dblat3.f).
- Exact-data testing:
  - benchdnn uses integer fills with density reduced so that accumulation is exact ("safe_n_acc"), then expects trh = 0 for direct conv. It flags results as MISTRUSTED when the fraction of zero outputs is too high, citing "improper filling of input data for a given driver" — [benchdnn conv.cpp](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/conv/conv.cpp), [benchdnn_general_info.md](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/benchdnn_general_info.md).
  - CUTLASS uses integer uniform fills and `TensorEquals` — [CUTLASS testbed.h](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/testbed.h).
- Probabilistic and mean-zero behaviour:
  - Error ∝ λ√n·u w.h.p. for independent mean-zero rounding errors — [Higham & Mary 2019](https://eprints.maths.manchester.ac.uk/2731/); [Hallman & Ipsen](https://arxiv.org/abs/2104.06531).
  - O(u), independent of n, for zero-mean data — [Higham & Mary "Sharper"](https://eprints.maths.manchester.ac.uk/2778).
  - Round-to-zero accumulation attains the worst case n·u — [Mary slides](https://www-pequan.lip6.fr/~tmary/doc/SIAM_LA21_multiword.pdf).

### Inferences
- Suggested harness architecture, combining all of the above:
  1. **Exact tier.** Integer or dyadic inputs sized so that all intermediate results are FP16-exact. Require bit-exact output, with FTZ allowed. Catches logic, indexing and layout bugs with zero false positives.
  2. **Bound tier.** Random realistic inputs. Per-element `|npu − ref| ≤ c(op, n)·u16·M + k_op·ulp16(ref)`, plus the FTZ rule. M is the |·|-operator reference for linear ops and the |∂y/∂z|·|z| sum for elementwise chains. k_op comes from the TOSA or OpenCL ulp tables.
  3. **Statistical tier.** Over ≥ 1000 outputs, compute normalized errors e_i. Require RMS(e) ≤ √(1.6·ksb) and |mean(e)| ≤ √(16·ksb/T) (TOSA's bias test rearranged). Bias detects RZ/truncation or a systematically wrong approximation even when each element is in-bound.
  4. **Metamorphic checks**, cheap and order-independent:
     - Scaling by powers of two: f(2^k·x) = 2^k·f(x) exactly for linear ops, absent overflow/underflow.
     - Permutation invariance of reductions: the result should be within the bound, not identical.
     - sum(x) + sum(−x) ≈ 0 within u·Σ|x|.
     - Odd/even symmetry: tanh(−x) = −tanh(x), erf(−x) = −erf(x), which should be exact if the implementation is symmetric.
     - mean(x + c) = mean(x) + c within the bound.
- Mapping to failure classes:
  - (a) → FTZ acceptance rule (Q5).
  - (b) → |terms| magnitude M (Q2, Q3).
  - (c) → c(n)·u·M with c chosen by the accumulation model, plus RMS/bias statistics.
  - (d) → per-primitive ulp budgets (OpenCL/TOSA/Vulkan) propagated through the chain (TOSA operator-sequence rule; Vulkan "inherited from").
  - (e) → include rounding of FP16 constants as an error source in the linearized propagation, or use |terms| magnitude.
- A "well-conditioned inputs only" filter is a cheaper but weaker alternative. Exclude points with κ_f(x) > threshold or M_i/|ref_i| > threshold and use ulp tests elsewhere. It hides exactly the regions where (b) and (e) appear, so prefer the magnitude-based tolerance.

### Gaps
- No NPU-vendor (Intel, Qualcomm, Arm Ethos) documentation was found that publishes a statistical acceptance test like TOSA's. Whether Intel's NPU compiler validation uses TOSA-style checks is unknown.
- Backward-error checks for non-linear-algebra ML ops (e.g. "find x̃ near x with f(x̃) = npu output") were not found in any test suite. This is plausible for monotone elementwise functions (invert via the inverse function and measure |x̃ − x|/ulp(x)) but no source applies it.
