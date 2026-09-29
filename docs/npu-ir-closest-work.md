# Closest-work comparison for the IR study

Reviewed 2026-09-17 against the primary sources below. This is a comparison of the five works named in the research plan, not an exhaustive novelty search or a reproduction of their experiments. Versions are pinned where the paper provides one. Repository documentation is a dated observation.

| Work | Existing mechanism and evidence | Consequence for this study |
| --- | --- | --- |
| [LLMLift, arXiv v1](https://arxiv.org/html/2406.03003v1), especially Sections 2–3 | LLMs produce a program summary and proof annotations in a Python representation of DSL semantics. An automated theorem prover checks equivalence, and rewrite rules produce target syntax. | LLM lifting through an IR followed by checked lowering is established prior work. Our finite oracle tests do not provide its proof-based guarantee. A contribution must be measured porting behavior or a specific new mechanism, not the existence of the intermediate stage. |
| [Tenspiler, arXiv v3](https://arxiv.org/html/2404.18249v3), especially Sections 3–4 | TensIR supports tensor-oriented synthesis, equivalence checking and pattern-based generation across multiple backends. Its intermediate separates functional tensor semantics from concrete target APIs. | Target-independent tensor semantics and multi-backend lowering are established. Measure whether retaining an intermediate reduces total preparation and translation cost across AMD and Intel; do not claim that portability alone is new. |
| [AscendCraft, arXiv v1](https://arxiv.org/html/2601.22760v1), especially Sections 3–5 | An Ascend-specific DSL exposes host planning and on-chip computation. Category-specific expert examples guide generation; constrained LLM passes lower the result to AscendC. Evaluation includes functional correctness and kernel execution performance. | Separate our target-independent semantic representation from hardware-specific plans. Retain a guided direct baseline and charge guidance/preparation costs. A pure-LLM comparison cannot establish superiority over an expert-guided method. Offline translation timing is not comparable to device kernel speedup. |
| [QiMeng-Xpiler, arXiv v1](https://arxiv.org/html/2505.02146v1), especially Sections 4–5 | Multiple transformation passes combine LLM annotation and generation with bug localization and SMT-based local repair. Programming-manual retrieval and platform examples inform transformations; hierarchical tuning explores transformation choices. | Localized repair, semantic annotations, guidance and multi-platform translation already have close precedents. Isolate their contributions with ablations and disclose supported subsets. Do not describe ordinary compiler feedback plus retries as symbolic repair or formal verification. |
| [KernelBench-Verified repository](https://github.com/facebookresearch/kernel_bench_verified), README evaluation sections | Its hidden suite tests four input distributions, including scale and sign changes. Input-blind generation strips test construction from prompts for selected susceptible problems. Hidden correctness gates reported results separately from standard evaluation. | Hidden final inputs and generation/input separation are established evaluation practices. Our final-input path is methodological infrastructure, not a novelty claim. Freeze the private suite before generation and prevent final feedback from entering repairs; also retain distinct held-out kernel families. |

## Proposed contribution boundary

The defensible research question is whether validated semantic work earns its construction and checking overhead under a controlled NPU-porting protocol. The planned contribution is an empirical account of when that happens, supplemented by a checked-lowering method only if the diagnostics justify and evaluate it. This positioning is our inference from the mechanisms above, not a claim that the cited authors evaluated our particular design.

The experiments must distinguish three separate endpoints:

1. Producing a correct intermediate under a declared semantic contract.
2. Producing backend artifacts that satisfy the offline compiler, numerical and dataflow acceptance contract.
3. Executing those artifacts correctly and efficiently on physical target hardware.

Only the first two are currently part of the pilot. The separately recorded source-layer-normalization discrepancy also prevents treating an oracle-valid intermediate as established source equivalence for that case. Successful finite testing, theorem-prover equivalence and physical NPU execution must not share a single undifferentiated correctness label.

## Controls required by this comparison

- Keep direct, hinted and executable-IR arms under matched source, target, model, sampling, budgets and backend acceptance.
- Treat API references and worked examples as explicit guidance conditions. Preserve the pure comparison and the stronger guided direct control.
- Separate semantic validation, retry reuse, checked transport and cross-target reuse. Attribute improvement to the entire evaluated method until ablations support a narrower explanation.
- Report total and amortized multi-target cost, including shared preparation, invalidation, failed calls and unsuccessful translations. Reference intermediates prepared outside the workflow are diagnostic controls.
- Keep hidden-input evaluation separate from held-out-kernel generalization. A private test suite cannot substitute for independent kernel families.
- Preserve negative and inconclusive results. Do not compare published success percentages or kernel speedups directly across different task sets, assistance levels, correctness contracts and hardware.

No cited system has been reproduced in this repository. A future quantitative comparison would require runnable artifacts, matched task/target support and disclosed changes; this review supplies positioning and experimental requirements only. No novelty or superiority conclusion is established while the pilot and diagnostic gates remain open.
