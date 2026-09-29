# Case study v4: testing whether structured IR earns its overhead

The current evidence does **not** establish that structured IR generally translates kernels faster or with fewer tokens. V4 tests that proposition, including the possibility of a negative or inconclusive result. It preserves the historical experiments and their evaluators rather than rewriting their results.

## Audit of the existing studies

| Study | Direct | Structured IR | Hinted IR | Interpretation |
| --- | --- | --- | --- | --- |
| Original Terra, one trial per kernel/target | 9/20; 2.052M tokens | 7/20; 3.991M | 9/20; 3.899M | The original structured pipeline underperformed. |
| Terra IR revision 2 | Frozen 9/20; 2.052M | 20/20; 0.792M | Frozen 9/20; 3.899M | IR changes, backend examples, numerical guidance, feedback compaction and reuse changed together. Baselines were not rerun. |
| Extended Terra v2, three repetitions | 57/60; 2.135M | 57/60; 2.888M | 60/60; 2.444M | Under the improved common backend guidance, structured IR did not save tokens. |
| Qwen v3 frozen interruption backup, one recorded repetition | Minimal 6/20; guided 9/20 | 7/20 | 6/20 | Six cases have lost in-flight responses. AMD backend generation and repeated unsuccessful repairs dominate the failures. |

Counts are offline validation outcomes, not physical NPU execution. The Qwen totals include unresolved cases in the *recorded-case* count; they are not final campaign accuracy estimates. Two additional planned repetitions did not run in that backup.

[All historical pipeline outcomes (PNG)](../runs/analysis/v4-design-audit-20260917/historical_outcomes.png) · [SVG](../runs/analysis/v4-design-audit-20260917/historical_outcomes.svg) · [Reproducible aggregation](../runs/analysis/v4-design-audit-20260917/analyze.py). The aggregation retains overlapping recovery directories separately and verifies that the source CSV hashes remain unchanged.

The relevant records are:

- [IR evolution results](../runs/ir-evolution-20260911/results.md), including original, first revision and second revision. Revision 1 reached 18/20, spending 2.141M tokens. The final revision used 36 cycles versus baseline's 125, but on the nine jointly solved cases used **24.2% more tokens and 26.8% more active time**. Both revision rounds together cost 2.933M tokens; 0.792M is the final fresh run, not the development cost.
- [Extended Terra results](../runs/case-study-v2-recovered/results.md). AMD direct and structured each solve 27/30; Intel each solves 30/30. Combined structured IR spends **35.3% more tokens**, 119 versus 123 cycles, and approximately 168.3 versus 154.9 summed active minutes. Hinted solves all 60 in 91 cycles. AMD `structured_ir_no_validation` solves 30/30 and costs less than validated IR; this is evidence to investigate gate overhead/rejections, not permission to drop end-to-end validation.
- [Detailed Qwen versus Terra forensic audit](../runs/analysis/qwen-terra-20260917/REPORT.md). The three shared arms' initial prompt text, schema and schema hash match in **180/180 comparisons**. Qwen constructs a valid executable IR in **18/20** structured cases, but only seven finish the backend pipeline. Its successive backend bundles are identical in **179/309** comparisons, versus **0/152** for Terra. Thirteen output-limit calls account for **42.5% of known Qwen output tokens**; raw rejected output fed into later prompts expands some from about 2K to about 190K characters.
- `runs/case-study-v2` is the interrupted predecessor of the recovered Terra run: 100 provider-blocked records, not a separate negative replication. The recovery retained saved responses and narrowly repaired the disabled-code-mode diagnostic guard. Count the recovered campaign once.
- `runs/case-study-v3-openrouter` and `runs/case-study-v3-openrouter-interrupted-backup-20260917T031835Z` contain the same 80-case outcomes (28 solved, 46 exhausted, six blocked). Count this evidence once.
- `runs/case-study-v3-openrouter-recovered` reuses 74 previous cases and replaces six interrupted cases. At this audit its CSV has 28 solved, 47 exhausted and five unfinished rows. It is still an overlapping, incomplete recovery, not a clean extra repetition. Original lost usage remains unrecoverable.
- `runs/case-study-v3-spark-preflight` contains a provider/account compatibility failure before a kernel study, not a translation result.

The baseline in extended Terra **did** receive backend/API examples and shared layer-normalization guidance. Global knowledge retrieval was disabled, but that does not mean the prompt contained no knowledge. Providing the same backend guidance to all arms was a useful control for the IR comparison; it answers a different question from an unaided baseline. V4 explicitly tests the latter.

Qwen's non-reasoning setup is a plausible contributor, not an established cause. Weights, sampling, serving backend, quantization, output cap and provider instructions also differ from Terra. Correct intermediate math did not resolve invalid OpenVINO calls, malformed MLIR-AIE interfaces, or repeated unchanged repair code. A reasoning hypothesis needs reasoning on/off with the **same weights, route, quantization and sampling**, in separate preregistered cohorts.

All prior active-time totals sum concurrent case clocks. They are not campaign elapsed times. The six lost Qwen cases additionally under-record their case clocks after resume, despite retained provider timings. They cannot support a clean speed comparison.

## Primary comparison

| Control | Direct LLM | Hinted IR | Structured IR |
| --- | --- | --- | --- |
| Original GPU source in every backend prompt | Yes | Yes | Yes |
| Common tensor/scalar interface, tolerances, target and submission contract | Yes | Yes | Yes |
| Worked backend examples, per-kernel solution recipes, retrieval, external LLM tools | None | None | None |
| Intermediate generation | None | Free-form, at most 16,000 characters | Executable KernelIR 2.0 |
| Intermediate correctness gate | None | None | Existing semantic interpreter and independent oracle |
| Preserve intermediate during backend repair | N/A | Yes | Yes after validation |
| Backend/compiler/numerical/dataflow acceptance | Same | Same | Same |

“Direct LLM” means direct generation without an intermediate stage, retrieved knowledge, curated examples or external model tools. It still needs a task/interface contract and receives bounded evaluator feedback on retries. Its first-cycle outcome is the zero-feedback result. It is not a claim that a pretrained model has no prior knowledge.

The IR language reference defines operation semantics, including numerical defaults. It supplies no worked layer-normalization decomposition or backend code. That language and semantic validation are parts of the **structured-IR system treatment**; the primary comparison does not isolate JSON syntax alone. Optional `structured_ir_no_validation`, `structured_ir_no_reuse`, and `hinted_ir_no_reuse` arms isolate mechanisms with the same backend acceptance. These are secondary ablations, not additional primary winners to select after seeing results.

The model-facing manifest omits oracle operation/parameters, test seed, provenance and source paths. V4 also isolates candidate Python construction/emission: its container sees a sanitized contract, its entry script, and an empty framework mount. It cannot obtain the oracle through `build_model(manifest)` or read the project's evaluator source in those stages. Trusted compiler/validation stages retain the full oracle contract. Installed SDKs remain available. This is a scoped experimental boundary, not a general proof against adversarial benchmark exploitation.

## Model cohorts

The shipped **pilot** configuration selects `deepseek/deepseek-v3.2`, reasoning enabled, OpenRouter `alibaba/fp8`, upstream `Alibaba`, temperature 0.6, top-p 0.95, and 16,384 maximum completion tokens. Its [model license is MIT](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/LICENSE). Selection is based on an open license and an available endpoint advertising reasoning and strict structured output; no superiority claim has been made or measured.

The adapter checks live endpoint capabilities and output limits, records context limits and quantization, and checks the exact returned model/provider, generation ID and usage. It sets `only`, `allow_fallbacks=false`, `require_parameters=true`, and strict JSON schema. [OpenRouter's structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs) describes this routing requirement. The capability snapshot from design time is [saved here](../runs/analysis/v4-design-audit-20260917/openrouter-endpoint.json); run preflight checks it again because availability changes.

Reasoning is explicitly enabled/disabled, and counts are required for hybrid cohorts. The reasoning-enabled preflight must actually return reasoning tokens. Reasoning text is excluded from saved responses, while its billed tokens remain counted inside completion tokens. Do **not** add reasoning tokens to `input + output` a second time. [OpenRouter reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) explains these controls. The total includes cached input tokens; their counts are retained separately.

For a same-weights reasoning diagnostic, copy the config into two separate files and set both to `qwen/qwen3-32b`, endpoint `deepinfra/fp8`, upstream `DeepInfra`, with identical sampling/budgets and `reasoning=true` versus `false`. [Qwen3-32B supports thinking and non-thinking modes under Apache 2.0](https://huggingface.co/Qwen/Qwen3-32B). The runner rechecks endpoint support before spending. The original `qwen/qwen3-coder-30b-a3b-instruct` is allowed only with reasoning disabled as a separately named historical-configuration diagnostic; it cannot establish a reasoning-on/off contrast.

Provider pins do not expose an immutable weight checksum or server implementation. Record the date and route snapshot and limit the conclusion to the tested served configuration. Cross-model token counts are not directly comparable measures of algorithmic efficiency; the primary contrasts stay within one model configuration.

## Budgets, schedule and stopping

- One case at a time and one validation job at a time. Repetition/kernel/target blocks contain every arm. A seeded randomized base order rotates arm positions across repetitions. This removes queue competition and reduces provider-time/order confounding; it cannot make remote service latency constant.
- Ten outer cycles maximum; direct uses at most one model call per cycle, intermediate arms at most two. Reports separately count model calls, backend attempts, semantic failures, unchanged backend pairs and output-limit finishes. An IR rejection consumes a cycle.
- 100,000 total tokens and 1,800 active seconds are **stopping thresholds checked between stages**, not hard spending or elapsed-time caps. A synchronous generation/validation stage can overshoot. The socket timeout is not a whole-request deadline, and compiler timeouts apply per container stage. All actual returned usage and elapsed active time are charged. A success beyond a threshold is an offline pass but not a budget-qualified success.
- Stop at the first complete offline pass. No best-of selection, successful-case substitution, unrecorded retry, free malformed response, free IR generation, or free failed validation.
- Keep only bounded evaluator feedback and parsed previous candidates in repair prompts. Never replay a rejected raw response. Preserve full returned raw output on disk for audit. A neutral note identifies repeated identical failed bundles; there is no method-specific early-stop heuristic or free compile cache.
- Preflight tests the selected oracles, known-good compiler fixtures and all three actual response schemas before any benchmark calls. Three paid schema probes are recorded separately from translation metrics and must be included in expenditure reconciliation. They are not tuning trials on benchmark kernels.

The default config selects vector addition, transpose and layer normalization, both targets, three primary arms, one repetition: **18 pilot cases, at most 303 model calls including preflight**. These expose the observed transport, API and numerical failure modes without authorizing a large campaign. Changing to all ten classic kernels and three repetitions gives 180 development cases; it is still development data.

## Correctness and statistical decision

The primary endpoint is a complete `offline-validated` pass within the cycle/token/time limits. AMD requires target compilation, host numerical execution and dataflow simulation; Intel requires target compilation and CPU graph equivalence. All use the existing independent reference/test contract. Physical target execution, target latency/power and correctness for every possible input remain untested. The pilot's builtin references are not evidence that the original GPU program was executed; the existing compiler explicitly records `source_equivalence=not_established`.

`cases.csv` retains every scheduled case. `results.md` distinguishes finished/planned/blocked cases, known spending and incomplete accounting. `summary.json` and PNG/SVG plots report successes versus token/time/cycle budgets. Resource curves replay observed first successes under this fixed repair policy; they are not experiments reoptimizing the policy at every smaller budget.

For each target, pair structured IR against direct and hinted using the same kernel/repetition. Repetitions and multiple kernels in one reviewed family remain inside the resampled family cluster. Report family-weighted paired differences for success, and normalized success costs in tokens/time/cycles; unsuccessful cases receive the full budget in each penalized cost. These penalized costs are statistical endpoints, **not actual expenditure**. Actual expenditure includes all failures and any overshoot.

Use 20,000 family-cluster bootstrap draws and family-wise alpha 0.05 divided across two comparators, the selected targets, and four endpoints. A positive claim requires complete matched evidence, a confirmatory holdout, at least ten reviewed families, a lower success-difference bound of at least zero, and upper cost-difference bounds below zero for **all three** cost metrics. This is deliberately strict: a cycle improvement alone does not establish lower tokens or time. The joint claim across targets/comparators requires every primary contrast to pass. Ablations remain descriptive.

Possible verdicts are `supported`, `structured_ir_reduces_success`, `joint_efficiency_claim_not_supported`, or `inconclusive`. No population noninferiority claim is made from an all-tied success bootstrap; that result is shown as observed equality with uncertainty. The report also exposes `observed_joint_benefit` for the finite benchmark without presenting it as population proof. Small clusters, floor/ceiling effects, narrow task families and noisy serving latency can leave the study inconclusive. Repetition is not a substitute for independent kernels or a prospective power analysis; inspect pilot variability before preregistering holdout sample size.

## Development versus confirmatory runs

Do not select the model, prompts, budget, kernel exclusions or success criterion after inspecting confirmatory outcomes. Complete the development pilot, freeze these choices, then conduct one prospective holdout study. If a methodological bug requires a change, preserve the old campaign and start a new protocol/configuration; do not silently patch and pool it.

A confirmatory config requires `stage="confirmatory"`, at least five repetitions, and `holdout_manifest` pointing to a JSON document containing:

```json
{
  "reviewed_by": "reviewer identity",
  "review_notes": "How kernels were selected before results, prior exposure, and family grouping",
  "development_source_hashes": ["SHA256 of every source used in development"],
  "families": {"new_kernel_name": "independent_family"},
  "source_validation": {
    "new_kernel_name": {
      "manifest_sha256": "SHA256 of the exact manifest",
      "report": "path/to/source-runtime-validation/report.json"
    }
  }
}
```

At least ten families are required. Existing classic names and exact source hashes, including renamed copies, are rejected. A human review must also catch semantically duplicated kernels and unlisted prior exposure; hashing cannot do that. Every source-runtime report must pass, identify `representation="original_gpu_source"`, and match `details.source_sha256`. Its content/hash and the manifest binding are frozen. The existing optional `scripts/validate_source.py` has adapters for only some CUDA/Triton operations; other held-out kernels need an appropriate source-execution adapter/evidence before a confirmatory run. This implementation does not manufacture unseen kernels or source-equivalence evidence.

The runner snapshots code, tool scripts, compiler fixtures, lockfiles, corpus, model/configuration, package versions, targets, provider capabilities and compiler identities. Resume rejects changed inputs. It records generated prompts/schemas/responses and per-stage checkpoints. Output directories are locked against concurrent launchers. Legacy v1–v3 files and runner behavior remain unchanged.

## Commands and interruption

Inspect the plan without model or Docker calls:

```bash
.venv/bin/python scripts/case_study_v4.py plan
```

When ready to spend on the 18-case pilot, use the direct Python launcher (the same command resumes):

```bash
bash scripts/run_case_study_v4.sh run \
  --config configs/case-study-v4.json \
  --output runs/case-study-v4-deepseek-pilot
```

`preflight` instead of `run` performs compiler/oracle checks and three paid schema probes, then stops. Credentials come from `OPENROUTER_API_KEY` or only the `openrouter.api_key` value in the existing root `config.yaml`; legacy model settings in that secret file do not override v4.

Ctrl+C or SIGTERM requests a pause: the current synchronous stage finishes, its result/usage/time is checkpointed, and no new stage starts. Repeated signals also drain; this can take as long as the current stage. The launcher executes `.venv/bin/python` directly, avoiding the prior `uv` process-group double-interrupt problem. Do not SIGKILL or close the session if the response must be retained. After a hard interruption, an in-flight case becomes blocked with unknown timing/usage rather than silently replaying it. That campaign cannot establish an efficiency claim; a replacement must retain this history and use a separate output directory.

Regenerate reports offline:

```bash
.venv/bin/python scripts/case_study_v4.py report --output runs/case-study-v4-deepseek-pilot
```

Implementation validation uses fake provider/compiler state-machine tests, mocked HTTP responses, real repeated SIGINT delivery, and bounded real AMD/Intel positive fixtures. No paid v4 model calls or campaign runs were executed while implementing this protocol.
