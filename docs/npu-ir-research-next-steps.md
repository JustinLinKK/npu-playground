# Next steps: structured IR for LLM kernel porting

For the committed evidence, current status, and cloud review scope, start with the
[cloud analysis handoff](cloud-analysis-handoff.md). Execution notes below retain
historical states; the combined Phase 1 report includes the later AMD reruns.

## Research objective and paper framing

Determine **when executable semantic IR, validation and reuse reduce the cost of correct GPU-to-NPU translation**. Measure correctness, total tokens, active translation time and repair work together. Treat an improvement, a regression and insufficient evidence as legitimate outcomes.

Working paper title:

> When Does Structured IR Help LLM Kernel Porting? A Controlled Study Across NPU Backends

The intended paper combines a controlled explanation of the observed failures with a targeted system improvement, if the diagnostic experiments justify one. The claim should identify which mechanisms help, under which conditions, and at what cost.

The [v4 protocol and historical audit](case-study-v4.md) provide the implemented experimental foundation. Audited-IR diagnostics, checked backend plans, cross-target reuse, additional guidance controls and hidden final evaluation described below are proposed research extensions. Keep them separately versioned from the existing v4 baseline.

## Evidence motivating the plan

| Observation | Research implication |
| --- | --- |
| The successful Terra IR revisions also changed backend guidance, numerical guidance and repair behavior while the baseline remained frozen. | The original improvement does not isolate an IR effect. |
| Extended Terra direct and structured IR each solved 57/60 cases, while structured IR consumed 35.3% more tokens. | Intermediate construction and validation can cost more than they save. |
| Qwen produced an executable IR passing the semantic tests in 18/20 structured cases, but only seven completed backend translation. | Semantic understanding and backend implementation need separate diagnosis. |
| Qwen repeatedly returned unchanged failing code and sometimes fed very large rejected responses back into subsequent calls. | Repair quality and feedback accounting can dominate the apparent effect of the representation. |

These are historical observations, not confirmatory results. The Qwen campaign is incomplete, and its recovered and backup directories overlap. See the [detailed forensic audit](../runs/analysis/qwen-terra-20260917/REPORT.md) and [historical comparison](../runs/analysis/v4-design-audit-20260917/historical_outcomes.png).

The working hypothesis is that IR earns its overhead when validated semantic work can be preserved and translated reliably, avoiding enough downstream repair to offset intermediate generation and checking. Evaluate this separately for tokens and time; fewer outer cycles alone do not establish either saving.

## Research questions

1. **End-to-end benefit:** Does structured IR improve correctness under equal resource budgets, or reduce cost at comparable correctness, relative to direct and hinted translation?
2. **Mechanism:** What do representation constraints, semantic validation, intermediate reuse and backend lowering each contribute?
3. **Conditions:** How do model capability, reasoning mode, backend API guidance and target programming model change the result?
4. **Portability:** Can one validated semantic representation reduce the total cost of translating to multiple NPU backends?

## Phase 1: establish the v4 measurement baseline

**Results:** [Read the combined Phase 1 report, including AMD reruns and all charts](../runs/analysis/research-next-steps-v1/pilot-retry1/PHASE1_REPORT.md).

**Purpose:** Confirm that the served model, evaluator, accounting and repair process are usable before committing to a large study.

- [x] Inspect the [pilot configuration](../configs/case-study-v4.json) and archive the intended protocol/configuration identity.
- [x] Run the offline plan command and inspect the selected cases and resource limits.
- [x] Run the provider/compiler preflight, retaining its separate usage records.
- [x] Run the existing 18-case pilot without changing its prompts, arms or acceptance criteria during the campaign (two HTTP 429 interruptions retained; user-authorized parallel continuation disclosed).
- [x] Inspect every failure and reconcile provider calls, usage, active time, backend attempts and interruption status (all known charges reconciled; missing historical records remain explicitly unresolved).
- [x] Produce a pilot report with stage-level failure counts and cost distributions.

The pilot contains vector addition, transpose and layer normalization, both AMD and Intel targets, and the three primary arms: direct, hinted IR and structured IR. The configured model is reasoning-enabled DeepSeek-V3.2 through a pinned OpenRouter endpoint. Endpoint availability must be checked at execution time.

Execution record: `runs/case-study-v4-deepseek-pilot-config-key` archives the frozen campaign identity, repository/corpus snapshots, offline `plan.json`, and successful compiler and three-schema provider preflight. It stopped with two budget-exhausted cases, one infrastructure-blocked case and 15 unstarted cases after a request returned without usage or generation identity. The spending guard retained `unknown_spending=true` and initially stopped further paid calls pending reconciliation or explicit continuation authorization (recorded below). Recorded spending is $0.1870210641; aggregate key usage agrees within rounding, but does not reconstruct the missing request. See `runs/analysis/research-next-steps-v1/billing-reconciliation.json`. No diagnostic or confirmatory claim follows. The earlier `runs/case-study-v4-deepseek-pilot` is preserved separately with its failed preflight and unknown usage. An environment key overrode `config.yaml` in that attempt; the funded launch used `env -u OPENROUTER_API_KEY`. Do not pool failed preflights with translation outcomes or omit them from spending reconciliation.

From the repository root, inspect the plan without model calls:

```bash
.venv/bin/python scripts/case_study_v4.py plan
```

The initial OpenRouter budget was **$30 total across this research work**, including preflight and failed calls. The user subsequently authorized continuation under their provider-side billing limit and instructed stopping on billing-limit errors. The running campaign retains its local $30 known-spend backstop; do not restart after a billing-limit error. Use the shared spending ledger for every subsequent campaign; do not reset it for diagnostics or controls. The guard stops after a response reaches the limit, preserving that response and its cost; one in-flight request can overshoot. Unknown billed cost also stops further paid calls. Terra handles live experiment monitoring.

After billing reconciliation or explicit authorization to continue with recorded historical billing gaps, use a new campaign directory with the config key and spending guard. A small provider-adapter fix now preserves HTTP status codes on future failures; the original frozen campaign must remain unchanged and cannot resume with modified source:

```bash
env -u OPENROUTER_API_KEY .venv/bin/python research/run_budgeted_study.py \
  --ledger runs/research-next-steps-budget.json --limit-usd 30 \
  --config configs/case-study-v4.json \
  --output runs/case-study-v4-deepseek-pilot-retry1
```

The spending guard was introduced at a graceful pause during the first case after the user specified the budget. Its initial ledger imports all returned preflight and translation costs ($0.0723847332). It preserves the runner, prompts, cases and acceptance rules; guard bookkeeping contributes to subsequent active time. Guarded-run provenance is archived in the campaign directory. Generate the separate stage/cost audit without model calls using `PYTHONPATH=src .venv/bin/python research/audit_v1.py --campaign runs/case-study-v4-deepseek-pilot-config-key --output runs/analysis/research-next-steps-v1/pilot`.

Classify failures as semantic errors, unsupported IR constructs, backend API errors, buffer/dataflow errors, numerical discrepancies, model-output failures or infrastructure failures. Trace where tokens and active time accumulate.

**Decision point:** Proceed when accounting is complete and the results contain enough successful and failed translations to investigate differences. If all methods fail on a backend, record that limitation and investigate model/backend viability. Define any replacement-model capability criterion before selection; do not choose a model based on which produces the largest IR advantage. Preserve the first pilot and start a new campaign for changed settings.

## Phase 2: isolate the semantic-to-backend bottleneck

**Purpose:** Test whether validated semantic IR becomes useful once backend construction is reliable, and distinguish semantic, backend API, repair and evaluation-contract failures.

The [combined Phase 1 report](../runs/analysis/research-next-steps-v1/pilot-retry1/PHASE1_REPORT.md) records five offline successes, twelve budget failures and one remaining interruption after the AMD reruns. Backend construction dominated the failures; Intel structured vector addition failed despite validated IR, while direct and hinted translation succeeded. Layer normalization exposed a source/oracle tolerance discrepancy and repeated filename-contract failures. These findings motivate controlled diagnostics; they do not establish that IR is generally worse.

### 2.1 Resolve entry conditions and evaluation contracts

- [ ] Preserve Phase 1 configurations, outcomes, interrupted attempts and costs. Implement any evaluator or protocol changes in a separately versioned diagnostic campaign.
- [ ] Resolve the accounting entry gate explicitly. Recover missing provider evidence where possible. If the five historical billing gaps cannot be reconstructed, archive a prospective protocol amendment defining whether an independently accounted diagnostic may proceed, while retaining Phase 1's incomplete-accounting status. The existing runner gate remains in force until that decision and its implementation are recorded; continuation authorization alone is not complete accounting.
- [ ] Compare original-source, reference-IR and backend layer-normalization outputs on the failing boundary input. Define and justify the numerical contract before changing tolerances; do not relax acceptance merely to pass observed candidates. Revalidate source and reference evidence under any new contract, and retain the old failures.
- [ ] Address harmless generated-path differences such as `./model.py` in the new evaluator, with explicit normalization and feedback applied equally to every arm. Continue rejecting unsafe paths. Record this change as an evaluator condition, separate from any IR or lowering improvement.
- [ ] Freeze a bounded provider-recovery policy before execution: retain each request/error and known charge, distinguish infrastructure interruptions from candidate failures, and preserve checkpoints without regenerating saved responses. Keep historical and new missing usage visible. Billing-limit errors stop further requests.

The original Triton layer-normalization source passed 11/12 tests but failed `operation_boundary` with 1,280 mismatches and maximum absolute error about `1.502e-5`; seven Intel backend validations showed the same error metrics. This agreement does not prove source equivalence. Evidence remains in `runs/analysis/research-next-steps-v1/source-layer-norm/report.json` and the counterexample [PNG](../runs/analysis/research-next-steps-v1/source-layer-norm/counterexample.png)/[SVG](../runs/analysis/research-next-steps-v1/source-layer-norm/counterexample.svg).

### 2.2 Freeze the diagnostic cases and reference evidence

Start with the following four kernel/backend pairs, retaining both layer-normalization pairs as **contract-unresolved** in coverage reporting. Add their executable comparisons after resolving the contract; do not silently drop them or count unresolved cases as successful controls.

| Diagnostic pair | Phase 1 evidence | Role |
| --- | --- | --- |
| Intel vector addition | Direct and hinted passed; structured failed during backend construction despite valid IR | Separate semantic success from backend API reliability |
| Intel transpose | All three arms passed | Successful control for regressions and overhead |
| AMD vector addition | All three arms completed without an accepted translation | Investigate backend construction when no workflow succeeds |
| AMD transpose | Repeated construction failures; direct remains interrupted after rerun | Test lowering while retaining the unresolved infrastructure outcome |

- [ ] Version the execution selection before observing alternative-lowering outcomes. Preserve the existing six-pair selection and record why four pairs are initially executable; declare repetitions and stopping rules prospectively.
- [ ] Audit each reference against source semantics and the independent validation contract; bind source, manifest, IR and test evidence by hash.
- [ ] Disclose reviewer identity and reference-preparation effort, including unknown preparation time/tokens. Agent inspection must not be described as human review.

The [existing diagnostic selection](../configs/research-next-steps-v1/diagnostic-selection-retry1.json) already binds agent-inspected vector-add and transpose IR references and original CUDA source-execution reports. Both original CUDA sources and their IR references pass the twelve finite tests. This is useful starting evidence, not a proof for every input. The selection/configuration and diagnostic runner still require the prospective changes above; this plan edit does not itself satisfy their execution gates.

### 2.3 Compare two lowering paths from the same audited IR

The first experimental milestone is **four kernel/backend pairs × two lowering paths**, with repetitions fixed before execution.

| Lowering path | Controlled treatment | Question answered |
| --- | --- | --- |
| Current LLM backend generator | Supply the audited IR to the existing backend-generation and repair loop | Does backend generation fail even when intermediate construction is removed? |
| Constrained or mechanical lowering | Lower the exact same IR through a declared supported subset, generating repetitive interfaces and transport where feasible | Can checked construction remove API, interface and data-movement failures? |

- [ ] Declare supported operations, shapes, layouts and buffer/dataflow interfaces before evaluation. Preserve unsupported cases, disclose fallback, and charge fallback work.
- [ ] Hold source, reference IR, target/compiler identity, acceptance checks and per-case budgets fixed within each paired comparison. Hold model route, reasoning and sampling fixed for all model calls.
- [ ] Use the same independent end-to-end backend validation for both paths. A generated artifact or successful compilation alone is insufficient.
- [ ] Disclose and account for reference preparation, lowering implementation, validation and repair costs. A mechanical path's lack of model calls does not make its preparation free.

This comparison diagnoses the semantic-to-backend boundary. It does **not** by itself establish deployable end-to-end IR efficiency: audited intermediates bypass work required by the complete workflow, and mechanical lowering changes the implementation method. Keep the frozen direct/hinted/structured results as context; any new end-to-end comparison must run its controls under the same diagnostic conditions.

### 2.4 Separate API guidance from representation and reuse

Bring a small instance of the Phase 4 API-guidance control forward because API errors dominated Phase 1. Keep it separate from the two-path lowering comparison.

- [ ] Preregister guided and unguided direct, hinted and structured comparisons on the diagnostic tasks. Give every guided arm the same reference for the pinned backend version, without kernel solutions, hidden inputs or oracle outputs. Keep all other settings and budgets fixed, and charge reference preparation and prompt tokens.
- [ ] Establish reproducible successful translations before spending on mechanism ablations. Then disable semantic validation and intermediate reuse individually, retaining final backend validation in every condition.
- [ ] Do not choose a replacement model based on an observed IR advantage. If AMD remains unusable, follow the existing [model-viability policy](../configs/research-next-steps-v1/model-viability-policy.json), with candidate ordering fixed before evaluation and model-specific results reported separately.

### 2.5 Report the evidence and decide what follows

- [ ] Produce one diagnostic report with correctness under fixed budgets, total tokens and billed cost, model calls, backend attempts, semantic failures, repeated candidates, and per-stage active time. Include failed, unsupported and interrupted work; keep successful-only comparisons secondary.
- [ ] Record concurrency and compiler limits for every condition. Parallel execution may reduce campaign wall time; use comparable scheduling for paired conditions and report elapsed wall time separately from summed active time.
- [ ] Publish coverage, source/IR/compiler identities, preparation costs, unresolved accounting, the guidance condition and ablation settings alongside results. Label offline validation separately from physical NPU execution.

| Finding | Decision |
| --- | --- |
| Audited IR fails with generated lowering but passes with constrained/mechanical lowering | Advance to Phase 3's checked-lowering prototype; attribute the diagnostic gain to lowering until end-to-end controls and ablations establish more |
| Equal API guidance helps direct, hinted and structured similarly | Attribute that improvement to guidance and reassess IR's additional value |
| Valid IR still produces numerical or interface/dataflow failures | Investigate the semantic contract and lowering correctness before scaling |
| IR validation rejects semantically appropriate translations | Investigate language expressiveness, numerical contracts and gate behavior |
| Structured workflows remain more expensive at comparable correctness | Retain the negative result; investigate task complexity or cross-target reuse as later conditions |
| No viable AMD baseline emerges | Report the backend/model limitation and apply the preregistered viability policy; do not claim an IR efficiency advantage |

**First milestone:** resolve or explicitly disposition the entry conditions, then complete the four-pair audited-IR/lowering comparison. A broad repeat of the unchanged Phase 1 pilot is not the next experiment.

## Phase 3: prototype checked lowering if the diagnostics justify it

**Purpose:** Preserve validated semantic work through backend construction.

Proposed pipeline:

```text
GPU source
    -> executable semantic IR
    -> checked backend plan
    -> backend code
    -> independent end-to-end validation
```

- [ ] Define the supported backend-plan subset before evaluation, including buffer signatures, shapes, layouts and data movement.
- [ ] Generate repetitive interfaces and transport mechanically where feasible; constrain the model's remaining decisions.
- [ ] Check buffer count, ordering, shape and layout requirements before expensive compilation.
- [ ] Route feedback to the failing stage and preserve validated intermediates during backend repairs.
- [ ] Invalidate retained evidence and dependent artifacts when a new counterexample requires changing the intermediate.
- [ ] Preserve unsupported cases in the reported coverage; disclose any fallback and charge its work.
- [ ] Compare the new method with the frozen v4 workflows and the relevant ablations.

The observed vector-add failure is a concrete acceptance case: a two-input/one-output kernel must not repeatedly reach backend compilation with a two-buffer runtime interface.

Any measured improvement belongs to the complete tested method, such as **IR plus checked lowering and localized repair**. Ablations must establish which components contribute; an improvement from generated transport alone is not evidence that structured reasoning alone saves tokens.

**Decision point:** Advance the prototype only if it improves reproducible correctness or cost on development cases without relying on hidden oracle information or retrospective exclusions. Otherwise preserve v4 and pursue an empirical explanation of the limitations.

## Phase 4: test guidance, reasoning and reuse as separate conditions

The pure-LLM comparison remains primary. Additional controls address practical applicability and possible alternative explanations.

| Control | Required design |
| --- | --- |
| Backend API guidance | Give every arm the same API reference without kernel solutions. Run this as a separate guidance condition. |
| Reasoning mode | Compare enabled/disabled reasoning with the same model weights, endpoint, quantization, sampling and budgets. |
| Model robustness | Use a second open-source model configuration after freezing the primary design. Keep cost comparisons within each model. |
| Retry reuse | Give hinted and structured intermediates equivalent opportunities to be reused. |
| Cross-target reuse | Generate a target-independent intermediate once and reuse it for AMD and Intel; permit equivalent reuse in the hinted control. |

- [ ] Preregister the selected controls and their interpretation before running them.
- [ ] Retain the stronger guided direct baseline alongside the pure baseline in the paper.
- [ ] Report both single-target cost and total/amortized cost for multi-target porting.
- [ ] Record all shared preparation, validation and cache invalidation costs.

The existing Terra–Qwen comparison cannot isolate reasoning. Open-source weights and a pinned OpenRouter route improve transparency but do not expose an immutable checksum of the provider's served weights or make service latency constant.

## Phase 5: prepare a confirmatory evaluation

**Purpose:** Test the frozen research claims on data that did not drive method development.

- [ ] Define the task population and kernel-family grouping before selecting the final benchmark.
- [ ] Prioritize independent kernels and new compositions over additional repetitions of the ten development examples.
- [ ] Use pilot variability and the available budget to justify sample size; repetitions do not replace independent families.
- [ ] Construct a reviewed held-out corpus and source-execution evidence satisfying the [v4 confirmatory requirements](case-study-v4.md#development-versus-confirmatory-runs).
- [ ] Separate feedback inputs from hidden final inputs; implement the final evaluation path before starting the confirmatory study.
- [ ] Keep hidden final outcomes unavailable to the repair loop and to prompt/model tuning.
- [ ] Cover numerical boundaries and input distributions within the declared contract; vary shapes/layouts across tasks where appropriate.
- [ ] Freeze models, routes, prompts, acceptance criteria, budgets, exclusions and statistical analysis.
- [ ] Run paired cases and retain failures, unsupported cases, interruptions and all known spending.
- [ ] Report AMD and Intel separately, accounting for their different programming models.

Held-out kernels test generalization beyond development examples. Hidden final inputs test whether iterative repairs overfit the feedback tests. Both are necessary for the intended claim.

The separately versioned [hidden-evaluation path](../research/README.md#hidden-final-inputs) implements prospective input commitments and terminal-campaign-only evaluation. It has synthetic fixture coverage; a reviewed held-out corpus, private input suite and confirmatory execution remain outstanding. It does not change v4's existing paired-analysis rules or establish a research result.

Primary reporting should include correctness under token/time budgets, total expenditure, success-budget curves and failure-penalized efficiency measures. Supplement these with model-call counts, backend attempts, semantic failures, unchanged repair outputs and stage-level active time. Keep successful-only comparisons explicitly secondary.

Use the current v4 paired-analysis rules for its frozen study. Any revision to those rules must be specified before the new confirmatory data are observed. Negative or inconclusive results must remain visible.

## Phase 6: construct the paper and release artifacts

Proposed paper structure:

1. **Problem and research questions:** The cost of obtaining correct NPU translations, with semantic reasoning and backend implementation as distinct challenges.
2. **Motivating evidence:** Historical confounds and the controlled pilot's failure taxonomy. Historical development outcomes motivate hypotheses rather than establish the main claim.
3. **Method:** Executable IR and, if supported by Phase 2, checked lowering and localized repair. Define the supported language, numerical contract and limitations.
4. **Evaluation design:** Pure and guided controls, hinted IR, mechanism ablations, model configurations, held-out kernels, hidden inputs and complete cost accounting.
5. **Results:** End-to-end effects first, followed by mechanism, backend/model conditions and cross-target reuse.
6. **Limitations and reproducibility:** Source equivalence, finite test coverage, unsupported constructs, provider drift, hardware evidence and development effort.

Target contributions are a controlled characterization of IR's costs and benefits, a justified method for preserving validated work during lowering, and a reproducible evaluation artifact. Do not present a proposed method as a contribution until it has been implemented and evaluated.

Prepare the following figures and tables:

- Correctness versus total token and active-time budgets for every primary arm.
- Stage-level failure and cost breakdowns by backend.
- Ablation effects for validation, reuse and lowering.
- Single-target versus multi-target translation cost.
- Representative repair traces showing how a diagnosed failure was resolved or remained unresolved.

Archive configurations, prompt/schema/response records, source/compiler identities, case-selection provenance, generated artifacts, analysis scripts and complete usage records. Use publication-ready PNG/SVG figures.

Use **validated** for the current finite-test approach. Physical NPU correctness and runtime performance require device execution; translation time and generated-kernel execution time are different endpoints. If physical evaluation is unavailable, scope the paper explicitly to offline-validated porting.

If structured IR continues to lose under the controlled study, an empirical paper should explain reproducible conditions and mechanisms behind that result. It should not substitute another tuned development result for the confirmatory evaluation.

## Related work to position against

The [closest-work comparison](npu-ir-closest-work.md) records a primary-source review of these five works and its implications for controls and claim scope. This remains a positioning list, not an exhaustive novelty review or a reproduction.

| Work | Relevance to the proposed contribution |
| --- | --- |
| [LLMLift: Verified Code Transpilation with LLMs](https://arxiv.org/abs/2406.03003) | Combines LLM-based lifting with formal correctness proofs. Distinguish our finite validation and measured porting costs. |
| [Tenspiler](https://arxiv.org/abs/2404.18249) | Uses a tensor intermediate language for lifting, verification and multiple backends. Generic IR-based portability is established prior work. |
| [AscendCraft](https://arxiv.org/abs/2601.22760) | Uses a DSL, expert examples and constrained LLM lowering for Ascend kernels. Compare guidance assumptions and lowering mechanisms explicitly. |
| [QiMeng-Xpiler](https://arxiv.org/abs/2505.02146) | Combines LLM transformations and symbolic repair across accelerator platforms. Position any repair and multi-backend contribution carefully. |
| [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) | Uses additional input distributions and input-blind generation controls, motivating hidden final evaluation. |

The paper needs a specific, measured contribution beyond inserting an intermediate representation into the prompt sequence. Complete the closest-work comparison before finalizing novelty claims or scaling the proposed method.

## Immediate priorities

1. Preserve the completed Phase 1 pilot and AMD reruns; resolve or explicitly disposition the accounting and evaluation-contract entry conditions in Phase 2.1.
2. Freeze the four-pair, two-lowering-path diagnostic in Phase 2.2–2.3 before execution, retaining layer normalization as contract-unresolved.
3. Use those results to choose between a checked-lowering prototype and a focused empirical study of IR overhead and limitations.
4. Freeze the selected method and analysis before investing in the confirmatory benchmark.

Continuation authorization (2026-09-17): the user authorized continuation and reported configuring API-key billing limits. A read-only key check returned `limit=null`; the local $30 known-spend guard remains active as a backstop. The retry campaign uses `--acknowledge-unknown-spending`, preserves the original incomplete campaign and billing gap, and pauses on any new unknown-cost request. This authorization does not establish complete accounting for the original pilot.

A subsequent key check confirmed the newly applied **$30 provider-side limit**, no reset period, and **$29.267515988 remaining**. Retry campaign `runs/case-study-v4-deepseek-pilot-retry1` is running with Terra monitoring and archived guard/authorization provenance. Provider billing-limit errors stop execution; historical missing usage remains incomplete evidence.

Additional diagnostic evidence: the unchanged vector-add and transpose CUDA sources each passed all 12 manifest cases on RTX 5090 through the separate `research/validate_cuda_references.py` adapter. Reports, CUBINs, actual outputs and environment hashes are under `runs/analysis/research-next-steps-v1/source-cuda/`. All 24 archived comparisons were independently recomputed. The retry diagnostic selection binds these reports; no pilot acceptance or source file changed. This establishes finite source-runtime agreement for these references, not universal equivalence or physical NPU correctness.

Retry interruption: the first hinted-IR AMD transpose case stopped in cycle 5 on HTTP 429 with absent usage. Its preceding attempts and blocked status remain intact. Under the user's continuation instruction, `--continue-past-rate-limit --acknowledge-unknown-spending` advances to remaining scheduled cases without retrying or relabeling that case. Billing-limit errors still stop execution. See the campaign's `rate-limit-continuation.json`; this preserves coverage and evidence but does not establish complete pilot accounting or satisfy the diagnostic gate.

The prospective [model viability policy](../configs/research-next-steps-v1/model-viability-policy.json) defines a replacement-model capability criterion before any replacement is selected: successful compiler/schema preflight plus at least one budget-qualified direct result on each backend among the two source-validated CUDA tasks. It retains the complete pilot coverage, requires candidate ordering to be frozen before paid evaluation, and never selects by an observed IR advantage. Infrastructure interruptions are unresolved evidence, not proof of model incapability. No replacement model has been selected or launched.

Guidance preparation: `research/capture_backend_api.py` captured task-independent API signatures from both exact pilot compiler images, with no model calls or kernel solutions. The successful inventory and provenance are under `runs/analysis/research-next-steps-v1/backend-api-reference-v2`; the first capture's entrypoint-related failure is retained separately. This is an incomplete reference inventory (generic/unavailable signatures still need documentation), not an executed guidance control. No API guidance has been added to the active pilot.

Phase 1 parallel continuation (2026-09-17): the user limited the current scope to Phase 1 and requested higher concurrency. The serial process paused cleanly; its six unfinished Intel cases resumed through `research/run_parallel_study.py` with up to eight case workers and four compiler jobs. Completed case records and the two recorded HTTP 429 interruptions are retained. Prompts, models, per-case budgets and acceptance checks remain fixed. This changes the serial execution protocol: the pilot report must disclose mixed execution modes and must not interpret summed active time as campaign wall time or make an unqualified serial-versus-parallel time comparison. Campaign events and archived continuation scripts bind the affected cases and execution limits. Billing-limit errors stop further requests while in-flight responses are saved. No later phase is authorized in the current process.

Phase 1 finished: the [final pilot report](../runs/analysis/research-next-steps-v1/pilot-retry1/PHASE1_REPORT.md) records five offline successes, eleven budget-exhausted failures and two retained HTTP 429 interruptions across all 18 cases. Every case and cycle has a hash-bound review. Known shared spending reconciles exactly to $0.9146858058 across 174 returned generation records; four historical calls across initial attempts and the retry still lack complete billing records. Complete accounting remains false, and the Phase 2 gate is not satisfied. The six-case parallel segment took 686.9 seconds; this is observed elapsed time, not a measured speedup against a serial replay. Current work stops at Phase 1 as requested.

Requested follow-up: the two interrupted AMD transpose cases were rerun with unchanged per-case settings in a separate campaign. [Rerun results](../runs/analysis/research-next-steps-v1/amd-interruption-rerun/RERUN_REPORT.md): hinted IR failed after ten backend-construction attempts; direct encountered another HTTP 429 after four compiler attempts. The replacement outcome view is 5/18 offline successes, 12 budget failures and one unresolved interruption. Original pilot records and all costs remain retained. A fresh-database initialization race in the parallel wrapper was fixed by initializing SQLite before workers start; 22 focused tests passed. This follow-up does not satisfy complete accounting or advance Phase 2.
