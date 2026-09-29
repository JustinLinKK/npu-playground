# Structured IR evolution: completed results

Two revision rounds reached **20/20 offline-validated translations** on the existing ten kernels and two targets.
The final revision used **36 programming cycles and 792,024 tokens**, versus **125 cycles and 2,052,011 tokens**
for the frozen direct baseline, which solved 9/20. That is **71.2% fewer total cycles** and
**61.4% fewer total tokens**, including unsuccessful baseline attempts.
All final cases solved by cycle six; twelve solved on cycle one. This meets the requested improvement target,
so the evolution stopped after two rounds.

[HTML report](comparison/report.html) · [Overview PNG](comparison/overview.png) ·
[Wall-time PNG](comparison/wall_time.png) · [Paired efficiency PNG](comparison/paired_efficiency.png) ·
[All case outcomes](comparison/case_cycles.png) ·
[CSV with all 100 historical and new case records](comparison/cases.csv) · [Audited summary JSON](comparison/summary.json)

| Pipeline | Success | AMD | Intel | All cycles spent | All tokens spent | Active case time | LLM calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 9/20 | 0/10 | 9/10 | 125 | 2,052,011 | 111.7 min | 125 |
| Original structured IR | 7/20 | 0/10 | 7/10 | 149 | 3,991,279 | 145.7 min | 256 |
| Frozen hinted IR | 9/20 | 0/10 | 9/10 | 123 | 3,898,546 | 164.1 min | 246 |
| IR revision 1 | 18/20 | 9/10 | 9/10 | 56 | 2,140,821 | 82.1 min | 108 |
| IR revision 2 | 20/20 | 10/10 | 10/10 | 36 | 792,024 | 40.5 min | 56 |

## Wall-clock timing

Each case's `end_to_end_seconds` is a wall clock around model generation and validation. The active time below
sums those clocks across 20 cases. It is useful for comparing pipeline work, but it is not the experiment's
elapsed campaign clock because 20 workers ran cases concurrently.

| Pipeline | Summed active case time | Mean per case | Median per case | Versus baseline |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline | 111.7 min | 5.6 min | 6.3 min | +0.0% |
| Original structured IR | 145.7 min | 7.3 min | 9.8 min | +30.4% |
| Frozen hinted IR | 164.1 min | 8.2 min | 11.3 min | +47.0% |
| IR revision 1 | 82.1 min | 4.1 min | 1.2 min | -26.5% |
| IR revision 2 | 40.5 min | 2.0 min | 1.2 min | -63.7% |

The frozen baseline, original structured IR, and hinted IR were interleaved in one run, whose single campaign
clock was **32.3 minutes for all 60 cases**. The saved data cannot split that concurrent clock into
three exact pipeline-specific campaign clocks. The structured-only campaign clocks were **26.3 minutes**
for revision one and **9.9 minutes** for revision two.

On the same nine cases solved by baseline, hinted IR, and both revisions, baseline used
**4.8 active minutes**, hinted IR used
**8.8 minutes** (**81.9% more**), revision one used
**5.1 minutes**, and revision two used
**6.1 minutes** (**26.8% more** than baseline).
Thus the revised IR reduced total active time across the full corpus by solving failures sooner, but did not beat
direct baseline wall time on the already-solvable paired subset. These are one observation per case, not repeated
timing trials, and model-service latency and concurrent load were not controlled.

## What changed and why it helped

**Revision one (`ir-repair-v1`) repaired the existing pipeline.** Experimental generation now requests
KernelIR schema 2.0 explicitly, while the historical reader keeps its 1.0 default. The prompt explains executable
reduction metadata, transpose axes, operation defaults, and macro semantics. Repair context retains the last
backend candidate and its diagnostics across intervening IR failures. Shared backend requirements now show a
container-verified MLIR transport example and the correct OpenVINO `node.output(0).get_tensor()` naming API.
This raised success from the original structured IR's 7/20 to 18/20. It still regenerated IR on every retry,
and four layer-normalization IR attempts regressed numerically. AMD matmul and Intel layer normalization exhausted
all ten cycles. AMD attention succeeded on cycle ten.

**Revision two (`validated-ir-v2`) made numerical intent executable and preserved validated work.** The new
`center` operation casts to its declared floating-point accumulation dtype, computes `d = x - take(x, [0], axis)`,
then returns `d - mean(d, axis, keepdims=True)`. Normalization composes that result with squared values, reduction,
variance scaling, epsilon, square root, and the affine transform. Backend generation receives the exact operation
order. This avoids the loss of small variations caused by subtracting a rounded mean near a common offset.
The LLM generated this decomposition independently for both targets.

On Intel layer normalization, the frozen baseline's first numerical execution had **1,280 mismatches** and
**1.5020370483398438e-5 maximum absolute error** on `operation_boundary`. Revision two's first candidate had
**zero mismatches and zero maximum absolute error on that case**, passed all twelve numerical cases, and passed
target compilation. This is direct evidence that the numerical representation mattered for this failure.
It is not a proof of correctness for every possible input or of physical NPU execution.

After successful IR validation reaches backend testing, backend repairs now reuse that exact IR. Reuse records
the original cycle, zero newly executed IR cases, zero validation duration, and no intermediate model call.
Invalid IR still consumes a cycle and must be regenerated. The hint-based scenario continues regenerating its
intermediate. Revision two performed **20 actual IR validations, all passing, and 16 reuses**. Its **56 model calls**
comprise 20 intermediate calls and 36 backend calls. There were no rejected provider responses or blocked cases.

Repair prompts retain failing numerical comparisons, compiler diagnostics, previous code, and IR, while dropping
container commands and artifact hashes. Full logs remain persisted. Replaying this formatting change on thirty
recorded revision-one repair contexts reduced their character count by **65.2%**; the measured experiment token
totals above are the authoritative cost results. Shared backend requirements also document the installed
`npu_numeric.h` scalar ABI, which passed separate AMD half-precision matmul and attention checks.

## The efficiency claim has a boundary

The total-token improvement includes eleven cases on which the frozen baseline spent its budget without solving
the translation. On the **nine jointly solved cases (9/20 coverage)**, revision two used **10 cycles versus 15**,
but **219,922 tokens versus 177,136**: **24.2% more tokens**. Therefore the revision improves overall
coverage and total corpus cost, and reduces paired cycles; it does **not** establish lower token cost on already
solvable cases. The paired plot now includes hinted IR and the active wall-time comparison.

Both evolution rounds together consumed **2,932,845 translation-model tokens** (2,140,821 + 792,024).
The 792,024 figure is the final revision's fresh twenty-case run, not the entire development search cost.
The second round started every case from scratch. No successful first-round candidates were substituted into it.

Shared backend prompt fixes were applied alongside IR changes. The old baseline did not receive those improved
requirements, so this is a comparison of complete pipeline revisions against the frozen baseline. It does not
isolate an IR-only causal effect. This is one trial per kernel/target on the same corpus used for development;
research novelty and held-out generalization remain untested.

## Verification and preservation

- `uv run pytest -o addopts='' -q`: **138 passed, 23 skipped**. The skipped tests require explicit real-container execution.
- **12 separate real-container positive checks** passed: eight existing fixtures, the exact MLIR prompt example,
  the anchored Intel normalization graph, and two AMD half-precision kernels.
- Python compilation and `git diff --check` passed. The main implementation matches the frozen second-round source.
- Both completed live runs resumed without repeating a model call or backend test: revision one remained at
  **108 calls / 52 backend tests**; revision two remained at **56 calls / 36 backend tests**. Costs and cycles stayed unchanged.
- Every new backend invocation matched the frozen baseline's **manifest and deterministic input hashes**.
  Image IDs, installed toolchains, target profiles, numerical tolerances, and backend evaluator implementation matched.
  Only the scenario/experiment/prompt code and the additive IR operation changed.
- Knowledge retrieval, seeding and lesson reuse remain disabled. Existing database contents and unrelated
  uncommitted container work were preserved. Experimental calls remain isolated Terra `xhigh` calls with external
  tools and delegation disabled.
- Numerical host validation and target compilation passed for every final case, including AMD dataflow validation.
  **Physical target execution, target-binary correctness, latency and power remain unknown.**

The next research step should be a baseline rerun with the same improved backend contract, followed by untuned
kernels and repeated trials. Ablate validated-IR reuse, compact feedback, and numerical primitives separately.
If backend cycles remain the bottleneck, a deterministic DMA/FIFO resource checker or planner is a more focused
next change than further semantic-IR regeneration. AMD attention still needed six cycles, and moving average five.

## Artifacts and commands

- Frozen baseline: `runs/20260910T234814Z`.
- Revision one: `round1/20260911T032541Z`; its source snapshot is `round1/source`.
- Revision two: `round2/20260911T034836Z`; its frozen runtime source is `round2/source`.
- [Campaign metadata and source hashes](campaign.json), [incremental implementation diff](implementation.diff),
  [verification record](verification.json), and per-round `resume-verification.json` preserve the audit trail.
- New structured-only experiment from the current worktree:

```bash
uv run python scripts/run_experiment.py --scenarios structured_ir --max-cycles 10 \
  --case-workers 20 --max-compiler-jobs 4 --compiler-cpus 6 --compiler-memory 8g
```

The completed final run can be resumed safely (this performs no new model work):

```bash
uv run python scripts/run_experiment.py --resume runs/ir-evolution-20260911/round2/20260911T034836Z
```

Rebuild the comparison from the saved records:

```bash
uv run python runs/ir-evolution-20260911/analyze_evolution.py
uv run python runs/ir-evolution-20260911/write_findings.py
```
