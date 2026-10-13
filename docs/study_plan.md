# Training-Free Strategies for CUDA→NPU Kernel Translation, and Using Bucketed Length Requirements as Generation Features

## TL;DR

- **Your JSON-summary pivot is the wrong intermediate**: the strongest 2024–2026 evidence says the pivot should be *executable and verifiable* (NumPy/PyTorch/Triton/IRON-Python reference → NKC-IR you already execute), not prose/JSON, and the single largest training-free lever is an **agentic compile–test–profile repair loop**, which lifts AMD-AIE functional correctness from 21.6% to 73.5% in NPUEval and raises Triton agent correctness from a 7.07%–16.85% baseline to 54.89% (TritonBench-revised) and 63.33% (ROCm) in GEAK — far more than RAG gave you. On AMD AIE, RAG actually *hurt* functional correctness in NPUEval — its root cause being that "the RAG database contained kernels with Chess compiler-specific optimization pragmas that were ignored by the LLVM-AIE (Peano) compiler used in evaluation" (arXiv 2507.14403) — while only modestly helping vectorization, which explains your marginal gains.
- **Fix retrieval by keying on op-class + schedule/shape features and compiler-error→fix pairs rather than text similarity, add template/skeleton-based generation (plan-select + LLM fills the vector micro-kernel), constrained/grammar decoding for the target DSL, and a self-growing verified-kernel skill library** — these are the cheapest high-payoff changes and each plugs directly into ReNPU's planner and knowledge base.
- **Length buckets are a rich, free feature source**: a static-shape bucket set (256/512/1024/…), padding ratio, prefill-vs-decode phase, KV length, batch, head-dim and dtype determine whether a kernel is GEMM- vs GEMV-bound, what tiling/`rTile` sizes are valid, and whether you need valid-length masks so padded tokens don't corrupt softmax/norm reductions. Encode these as a **"shape/length contract" field in NKC-IR**, generate one parameterized kernel template and specialize per bucket, and pick per-bucket schedules with an analytic cost model (Roller/Vortex-style) or nearest-neighbor retrieval over a (shape, schedule, latency) table — all doable training-free.

---

## Part 1 — Training-free directions (ranked by expected payoff vs effort)

A note on your baseline: NPUEval (AMD Research, arXiv 2507.14403) is the most direct evidence for your setting. Frontier LLMs average only ~10% vectorization across its 102 AMD-AIE kernel tasks even with compiler feedback and RAG — per CANN Bench (arXiv 2607.20518), "NPUEval evaluates frontier LLMs on 102 AMD-NPU kernel tasks and reports an average vectorization score of around 10%"; the best model's average vectorization is ~9.7% (DeepSeek R1, read from the paper's figures).\[1\] Crucially, **compiler feedback raised functional correctness dramatically (Claude 3.7 Sonnet 21.6%→73.5% over ≤5 recompiles) while RAG generally *lowered* correctness** (Qwen2.5-Coder 68.6%→13.7%, GPT-4o-Mini 66.7%→35.3%) and only helped vectorization. This is the empirical backdrop for the ranking below: the loop is the big lever; naive RAG is a trap.

### Direction 1 (highest payoff, cheap) — Replace the JSON pivot with an *executable, verifiable* intermediate, and make NKC-IR the oracle

**Mechanism.** Instead of CUDA → prose/JSON summary → target, translate CUDA → a *runnable* reference in a high-resource language the LLM knows well (NumPy/PyTorch, or a tensor IR), which doubles as the correctness oracle; then lower that to the target DSL. This is exactly the verified-lifting pattern: Tenspiler (ECOOP 2024) lifts sequential C++/Python into a small unified tensor IR (Tensir) and proves equivalence\[2\] with an SMT solver, generating code for six tensor backends;\[3\] LLMLift (NeurIPS 2024) shows an LLM can emit the IR as plain Python (because Python is high-resource in pretraining)\[4\] and that this beats direct DSL generation, which is "infeasible" to do reliably for low-resource DSLs. AscendCraft (arXiv 2601.22760) is the clearest NPU instance: a *lightweight DSL pivot* plus multi-pass, constraint-driven LLM lowering achieves 98.1% compilation and 90.4% functional correctness on MultiKernelBench, with 46.2% of kernels matching/beating PyTorch eager — a massive jump over direct AscendC generation, which scores <5% compilation (Comp@1) and <5% functional correctness (Pass@1) in AscendCraft's own comparison.

**Why it fixes the JSON+RAG weakness.** A prose/JSON summary is lossy and *unverifiable* — nothing executes it, so errors propagate silently into target code. An executable reference (a) is checkable against the source CUDA on random/adversarial/boundary inputs, (b) keeps the LLM in a high-resource language for the hard semantic step, and (c) isolates the *low-resource* step to a narrow, mechanical lowering. This is why LLMLift and AscendCraft outperform summary-then-generate.

**Plug into ReNPU.** You already specify NKC-IR as *executable* with a numerical contract and hidden tests — lean into it: make NKC-IR (or a NumPy shadow of it) the literal reference implementation that the contract-validation stage runs. Treat the JSON fields as *metadata attached to executable IR*, not as the pivot itself.

**Cheap first experiment.** On a MultiKernelBench/NPUEval subset, compare three pipelines at equal token budget: (i) your CUDA→JSON→target; (ii) CUDA→NumPy/PyTorch reference→target; (iii) CUDA→NKC-IR(executable)→target. Measure compile rate, pass@1, vectorization. Expect (ii)/(iii) to dominate.

### Direction 2 (highest payoff, moderate) — Agentic compile–test–profile loop with execution feedback

**Mechanism.** Wrap generation in a Generator→Evaluator→Reflector→Optimizer loop that feeds back compiler diagnostics, test failures, and profiler counters (vectorization %, DMA stalls), with iterative repair and reflection. GEAK (AMD, arXiv 2507.23194) uses exactly this four-module Reflexion-style loop and raises correct-kernel rate from a 7.07%–16.85% baseline (direct prompting) to 54.89% on TritonBench-revised and 63.33% on the ROCm benchmark, with up to 2.59× average speedups (and one kernel surpassing a human-written expert kernel by 2.26×) — all *inference-time*, no fine-tuning. NPUEval shows the same loop mechanism works on AMD AIE (21.6%→73.5% correctness with recompilation). AutoKernel (arXiv 2603.21331) shows a single well-instructed iterative loop (no multi-agent machinery) is competitive,\[5\] and AgentKernelArena (AMD ROCm) benchmarks such agents directly.

**Why it fixes your weakness.** Your pipeline is essentially one-shot after retrieval. The compiler and the hardware are far more reliable "critics" than a static knowledge bank; most NPU failures are mechanical (hallucinated headers like the nonexistent `adf.h`, wrong lane widths, wrong pragmas) and are exactly what a diagnostic-driven loop fixes.

**Plug into ReNPU.** This is your "compile-test-profile feedback loop" — make the profiler's *vectorization score and DMA/ObjectFIFO stall counters* first-class reflection signals, not just pass/fail, so the loop optimizes quality (the NPUEval weak point), not just correctness.

**Cheap first experiment.** Add a ≤5-iteration recompile+repair loop around your *current* pipeline and measure the lift before changing anything else. NPUEval predicts most of the gain arrives by iteration 5.

### Direction 3 (high payoff, cheap) — Template/skeleton generation: plan-select + LLM fills the vector micro-kernel

**Mechanism.** Decompose into (a) selecting a *parameterized schedule/skeleton* per op-class from a template library (tiling, local-memory partition, DMA/ObjectFIFO double-buffering, layout, precision), and (b) having the LLM fill only the compute micro-kernel (the `aie::mmul`/`aie::load_v` body). Pair with an autotuner or analytic cost model to pick template parameters. This mirrors how Tenspiler and Tenet-style synthesis use user-supplied sketches, and how production AIE kernels are structured (IRON `design.py` describes ObjectFIFO dataflow; a small C++ compute kernel does the vector MACs).

**Why it fixes your weakness.** NPUEval's failure taxonomy shows LLMs *can* write scalar logic but fail at (i) data movement/tiling and (ii) using vector intrinsics instead of scalar loops. A template fixes the data-movement/tiling scaffold deterministically and restricts the LLM to the part it can do, directly attacking the ~10% vectorization ceiling.

**Plug into ReNPU.** This *is* your "generation-specific planner emitting an explicit schedule record" — formalize the schedule record as a template instantiation and store templates per op-class in the knowledge base, keyed on NKC-IR features.

**Cheap first experiment.** For 2–3 op-classes (elementwise, RMSNorm, GEMV), hand-write one parameterized IRON/MLIR-AIE skeleton each, let the LLM fill only the inner vector loop, and compare vectorization/correctness vs free-form generation.

### Direction 4 (high payoff, cheap) — Fix retrieval: key on op-class/schedule/shape features and compiler-error→fix pairs, not text similarity

**Mechanism.** Replace text-embedding similarity over kernels with retrieval keyed on *structured* features: op-class, dtype, reduction axes, launch geometry, and (Part 2) shape/bucket features — plus a separate **error-driven** store of (compiler-diagnostic → fix) pairs and verified skeleton kernels. Documentation-grounded generation for low-resource DSLs supports this: DocCGen retrieves the right library/schema then constrains generation; the low-resource-codegen survey (arXiv 2410.03981) shows documentation + retrieval + decoding constraints are what move the needle for unseen APIs.

**Why it fixes your weakness.** NPUEval found naive RAG over "similar kernels" *reduced* correctness (it injected Chess-compiler pragmas incompatible with the LLVM-AIE/Peano toolchain, and off-target examples).\[1\] Text similarity retrieves superficially-similar but semantically/toolchain-wrong code. Feature-keyed and error-keyed retrieval returns *actionable, target-correct* context.

**Plug into ReNPU.** Your "versioned hardware knowledge base with retrieval conditioned on target profile + NKC-IR features" is exactly right — the fix is to (a) add compiler/driver/firmware version as a hard filter so you never retrieve Chess pragmas for a Peano target, and (b) add an error→fix table populated by the Direction-2 loop.

**Cheap first experiment.** A/B your current text-similarity RAG vs feature-keyed retrieval with a strict toolchain-version filter; also measure "RAG off" (NPUEval suggests off may beat naive-on for correctness).

### Direction 5 (high payoff, moderate) — Constrained/grammar-guided decoding for the target DSL/API

**Mechanism.** Constrain decoding to the target DSL grammar and the *valid API surface* so the model cannot emit hallucinated functions or headers. DocCGen (arXiv 2406.11925) does document-driven constrained decoding for DSLs; Autogrammar (arXiv 2608.05493) learns a CFG from docs+execution logs for grammar-constrained decoding; Decode-Time Grammars / GPROJ (arXiv 2607.18357) instantiate grammar fragments from the runtime environment to eliminate\[6\] "ghost references" (undeclared buffers, nonexistent functions) — and explicitly name *tensor-kernel DSLs such as TileLang and Triton* as the strongest fit.

**Why it fixes your weakness.** NPUEval's #2 failure mode is hallucinated APIs (`adf.h`, nonexistent intrinsics). Grammar/type/environment constraints make that class of error *impossible by construction*, which a JSON summary and RAG cannot guarantee.

**Plug into ReNPU.** Add an API/grammar constraint layer at the final lowering stage (NKC-IR→target), with the valid symbol set drawn from the versioned knowledge base for the exact compiler/driver.

**Cheap first experiment.** Start with cheap grammar enforcement (e.g., a verifier that rejects non-whitelisted AIE symbols and triggers repair) before full constrained decoding; measure reduction in compile failures due to hallucination.

### Direction 6 (medium payoff, cheap) — Self-growing verified-kernel skill library (experience without training)

**Mechanism.** Accumulate *verified* kernels, lessons-learned, and a failure taxonomy into a persistent, retrievable memory that grows with each successful translation — Voyager-style skill libraries, and coding-specific CODESKILL (arXiv 2605.25430) / SkillCommit (arXiv 2608.15165). Critically, admit skills to the library *only after verification*: SoK on agentic skills (arXiv 2602.20867) reports SkillsBench found self-generated skills averaged −1.3pp without iterative verification — verification at admission is the key control.

**Why it fixes your weakness.** Your knowledge bank is static; a verified, self-growing library compounds across tasks and captures target-specific fixes the base model lacks (the data-scarcity problem AgenticCANN and AscendKernelGen identify for NPU DSLs).

**Plug into ReNPU.** Feed the Direction-2 loop's *successful* kernels and *distilled failure fixes* back into the knowledge base, gated by your contract validation so only verified skills are admitted.

**Cheap first experiment.** Persist every contract-passing kernel keyed by NKC-IR features; measure pass@1 improvement on later tasks drawn from the same op-classes.

### Direction 7 (medium payoff, moderate) — Differential/metamorphic testing with the CUDA source as oracle, and test-time search

**Mechanism.** Use the source CUDA kernel (or its executable NKC-IR) as a differential oracle: generate N candidates (best-of-N / beam / evolutionary search à la AlphaEvolve/OpenEvolve, used training-free in EvoEngineer and KernelFoundry), and select by passing hidden randomized/adversarial/boundary + metamorphic tests and profiling. Symbolic equivalence (Tenspiler-style SMT) covers affine parts.

**Why it fixes your weakness.** One-shot generation can't escape local failures; test-time scaling + a strong oracle converts extra compute into correctness/performance without training. GEAK/AutoKernel show parallel+iterative scaling is where most inference-time gains come from.

**Plug into ReNPU.** This operationalizes your "contract validation via execution + hidden tests + metamorphic + symbolic equivalence" as a *selection* signal over a candidate population.

**Cheap first experiment.** Add best-of-N (N=4–8) with contract-based selection on top of Direction 2; compare to N=1.

### Target-specific notes that change strategy

- **AMD XDNA2/AIE2P**: the target is IRON/MLIR-AIE Python describing ObjectFIFO dataflow + a C++ compute kernel using `aie::vector`, `aie::load_v/store_v`, `aie::mmul` (2×2 blocked 8×8×8 MAC). LLMs get lane widths wrong (AIE2 vs AIE2P differ: e.g., bf16 lanes 16 vs 32)\[7\] and emit scalar loops — so Direction 3 templates + Direction 5 constraints matter most here. The mlir-aie kernel library already contains per-arch reference kernels (flash-attention prefill, dwconv1d, RMSNorm, GEMM) you can retrieve as skeletons.
- **Intel NPU (OpenVINO)**: there is *no public CUDA-like kernel language* — the NPU path is graph/compiled-model (OpenVINO IR, static shapes). So "translation to Intel NPU" is realistically **CUDA kernel → OpenVINO subgraph composition** (map the kernel's semantics onto existing OpenVINO ops / custom-op extensions), not kernel codegen. NITRO (arXiv 2412.11053) documents that the NPU requires static shapes and that LLM attention must be reworked for this. Your generation strategy for Intel should therefore target *op-graph assembly*, with correctness checked against the CUDA oracle — a different generator than the AMD/AIE path.

---

## Part 2 — Length/bucket features for predicting & generating the target NPU kernel

**Why this is well-founded.** NPUs compile per static shape; serving systems bucket lengths and pad to the nearest bucket. This is documented across vendors: Intel OpenVINO GenAI NPU uses static shapes with `MAX_PROMPT_LEN`/`MIN_RESPONSE_LEN` (default prompt 1024) and the NPU plugin "only supports static models" (NITRO); AWS Neuron's `transformers-neuronx` "automatically uses bucketing… without requiring support for dynamic shapes," with separate **prefill vs decode** bucket lists; Intel Gaudi auto-buckets/pads dynamic shapes; mobile `llm.npu` (ASPLOS'25) uses fixed-size **chunked prefill** (chunk 256) with shared vs non-shared subgraphs; ShadowNPU buckets compute graphs by a scale factor + shape. The LENS paper (arXiv 2606.18042) formalizes the key consequence: **NPU latency is a step function — flat within a bucket, jumping at boundaries** —\[8\] so length/bucket identity is a first-class predictor of both latency and the right kernel.

### Proposed feature list (extractable training-free, mostly from the CUDA source + serving config)

| Feature | Source | What it tells the generator |
|---|---|---|
| **Bucket set** {256,512,1024,…} and the assigned bucket | serving config (e.g., Neuron bucket lists; OpenVINO `MAX_PROMPT_LEN`) | Which static shapes to specialize; number of binaries to emit |
| **Real request-length distribution / weights** | workload trace | Which buckets deserve specialized kernels vs a shared one (DietCode weights shapes) |
| **Padding ratio** = real_len / bucket_len | derived | Whether valid-length masking is needed; expected wasted compute (step-function waste) |
| **Phase: prefill vs decode** | serving state | GEMM-bound (prefill, large seq tile) vs GEMV-bound (decode, seq=1) → different tiling/template entirely |
| **KV-cache length** (context so far) | serving state | Attention inner-loop bound; masking of padded KV positions |
| **Batch size** | serving config | Parallelism/tiling across the array; shared-kernel viability |
| **Head dim, #heads, hidden dim** | model config / CUDA launch | Fixed tile dimensions; `rTile`/MMUL geometry alignment |
| **dtype** (INT8/FP16/BF16/FP32) | NKC-IR numerical contract | Vector lane count (AIE2 vs AIE2P), accumulator width, MMUL variant |
| **Launch geometry** (gridDim/blockDim) | CUDA source | Maps to NPU tile/loop-nest structure (T2T's Unified Parallelism Abstraction) |
| **Loop bounds & guards** like `if (idx < n)` | CUDA source AST | Precisely identify the valid-length boundary to turn into an NPU-side mask |
| **Reduction axes** (softmax/norm) | NKC-IR | Where padded tokens must be excluded from reductions |

### How each feature influences kernel design

- **Prefill vs decode → GEMM vs GEMV kernel.** Prefill processes the whole prompt in parallel (matmul-heavy, large sequence tile); decode generates one token (matrix-vector, memory-bound). These need *different* schedule templates, not one parameterized by length. Neuron keeps separate prefill/decode buckets precisely for this; `llm.npu`/sd.npu switch to a "decoding-optimized block graph." So the first branch in your planner should be *phase*, then bucket.
- **Padding ratio + guards → in-kernel valid-length masking.** The dangerous correctness bug on NPUs: padded tokens corrupting softmax/LayerNorm/RMSNorm *reductions* (sum/max over a padded axis). Extract the CUDA guard (`if (idx < n)`) and the reduction axis from NKC-IR, and emit a valid-length mask so padded lanes contribute identity elements (0 for sum, −inf for max). This is the shape contract's most important job.
- **Bucket set + distribution → which buckets get specialized kernels.** DietCode's lesson: a single dynamic-shape program is ≥5% slower than per-shape static ones, but one kernel per shape is infeasible — so bucket, and use *local padding* (pad the local/shared workspace, not global tensors) to bound waste. Weight specialization by request-length frequency.
- **dtype + head dim → vector/MMUL geometry.** AIE2 vs AIE2P differ in lanes (bf16 16 vs 32) and bfp16 MMUL support; the tile sizes must align to these. This is where an rTile-style alignment constraint prevents the scalar-fallback failure.

### Candidate methods to turn features into a kernel (training-free)

1. **One parameterized template, specialized per bucket.** Generate a single shape-generic kernel (Direction 3) whose tile sizes are functions of the bucket; instantiate per bucket at compile time. This matches DietCode's shape-generic micro-kernels and FTuner's `uKernel` composition, and is the cleanest fit for static-shape NPUs.
2. **Analytic cost model to pick per-bucket schedules — no autotuning.** Roller (OSDI'22) constructs kernels by aligning `rTile` to hardware features and uses a *white-box micro-performance model* (no on-device search), compiling in seconds; Vortex (arXiv 2409.01075) and MikPoly do sample-free/micro-kernel-polymerization dynamic-shape optimization with analytic models. Use such a model to choose tiling per bucket from features alone — fully training-free.
3. **Nearest-neighbor retrieval over a (shape, schedule, latency) database.** Build a small table of measured (bucket-shape, schedule, latency) tuples; for a new bucket, retrieve the nearest known shape's schedule. This is the retrieval analogue of a cost model and reuses your knowledge base. LENS shows just **two E2E measurements per bucket** suffice to predict latency to a 2.15% mean error (1.69% median, with 89.9% of cases under 5% error), validated across Inferentia2 and TPU v4/v5e/v6e — so a tiny per-bucket probe set is enough to rank candidate schedules.
4. **Latency-aware bucket/schedule co-selection.** Because latency is a step function (LENS), choosing the bucket *set* and the per-bucket schedule jointly minimizes padding waste at boundaries. Feed the step-function model into the planner to decide specialization granularity.

### Integrating into ReNPU's NKC-IR and planner

- **Add a "shape/length contract" field to NKC-IR** capturing: dynamic axes and their symbolic bounds, the bucket set, valid-length semantics (which axes are padded, which reductions must mask padding), per-dtype lane/MMUL alignment constraints, and the extracted CUDA guards. This makes padding-correctness and tile-legality *checkable* by your contract-validation stage (run the executable NKC-IR at the real length and at the padded length; outputs must match on the valid region).
- **Planner uses the contract to branch**: phase (prefill/decode) → bucket → template instantiation → analytic/NN schedule selection → emit per-bucket schedule records. The feasibility gate can *reject* a bucket whose padding waste or tile misalignment exceeds a threshold, or route it to a shared kernel.
- **Retrieval conditioning**: add shape/bucket features to the retrieval key (Direction 4) so you fetch skeletons and error→fix pairs that match the phase and shape regime.

### Benchmarks / datasets for shape-varying NPU kernels

- **MultiKernelBench** (arXiv 2507.17773):\[9\] 285 tasks, 14 categories, CUDA/AscendC/TileLang-Ascend/Pallas; now includes 15 attention tasks (MQA/GQA) — the practical substrate for shape-varying attention.
- **NPUEval** (arXiv 2507.14403):\[10\] 102 AMD-AIE kernels with reference + canonical vectorized implementations; note it currently targets single-output 1-in/2-in kernels on AIE2/AIE2P.
- **CANN Bench** (arXiv 2607.20518): 53 operators / 1060 test cases, four difficulty tiers incl. FlashAttention and MoE on Ascend;\[11\] notes a *static-shape track* with a single canonical input configuration — relevant to your bucketing question.
- **LENS** (arXiv 2606.18042): methodology + measurements for bucket-induced step-function latency across multiple vendors' NPUs — the reference for the latency side of Part 2.
- Dynamic-shape compiler baselines with shape sets: **DietCode**, **Nimble**, **BladeDISC/DISC**, **MikPoly**, **Vortex**, **Roller** — useful as cost-model/schedule sources even though they are compilers, not LLM systems.

---

## Recommended first experiments (ordered)

1. **Add the ≤5-iteration compile–test–profile repair loop around your current pipeline** (Direction 2). Cheapest, largest expected lift; NPUEval predicts correctness jumps of ~50 points. Measure before any other change.
2. **Swap the JSON pivot for an executable reference** (NumPy/PyTorch or executable NKC-IR) and re-run the A/B (Direction 1). Expect the AscendCraft-style gains (compile/correctness up sharply).
3. **Turn RAG off, then turn on feature-keyed + toolchain-version-filtered retrieval** (Direction 4). NPUEval suggests naive-on can be worse than off; verify on your data.
4. **Hand-write parameterized skeletons for 2–3 op-classes and let the LLM fill only the vector micro-kernel** (Direction 3), targeting the vectorization ceiling.
5. **Implement the NKC-IR shape/length contract + valid-length masking, and generate one parameterized template specialized per bucket; pick tiling with an analytic (Roller-style) cost model** (Part 2, methods 1–2). Validate padding-correctness by running NKC-IR at real vs padded length.
6. **Add best-of-N with contract-based selection** (Direction 7) once the loop and oracle are solid.

**Caveats.** (1) NPUEval's per-model vectorization numbers live in figures, so the ~9.7% best-average is read from plots, not prose — treat as approximate. (2) Many agent results (GEAK, AutoKernel, AscendCraft) are on Triton/AMD-GPU/Ascend, not AMD-AIE or Intel-NPU specifically — the *mechanisms* transfer but absolute numbers may not. (3) Intel's NPU has no public kernel language, so "kernel translation" there is subgraph composition; plan a separate generator. (4) Several cited 2026 works (CANN Bench, AscendKernelGen, AscendOptimizer) are very recent; AscendKernelGen's headline gains come from fine-tuning/RL and are *out of scope* for your training-free constraint — use it only for its failure analysis and benchmark (NPUKernelBench).

## Sources

1. [NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers](https://www.alphaxiv.org/abs/2507.14403)
2. [arxiv.org](https://arxiv.org/html/2404.18249v1)
3. [Tenspiler: A Verified Lifting-Based Compiler for Tensor Operations (Extended Version) — AI for Code](https://awesomepapers.io/ai-for-code/papers/2404.18249)
4. [Verified Code Transpilation with LLMs](https://arxiv.org/pdf/2406.03003)
5. [AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search](https://arxiv.org/html/2603.21331v1)
6. [Decode-Time Grammars: Constrained LLM Generation over a Refinement Order of Grammar Fragments](https://arxiv.org/pdf/2607.18357)
7. [Single-core Kernel Optimizations and Tooling by hunhoffe · Pull Request #3801 · Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie/pull/3801)
8. [Latency Prediction for LLM Inference on NPU Systems](https://arxiv.org/pdf/2606.18042)
9. [MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation](https://arxiv.org/pdf/2507.17773)
10. [AMD XDNA NPUs: Architecture & Optimization](https://www.emergentmind.com/topics/amd-xdna-npus)
11. [CANN Bench: Benchmarking Agent Generated Kernels against Real NPU and Algorithmic Limits](https://arxiv.org/html/2607.20518v1)
