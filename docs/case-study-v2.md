# Case study v2: controlled validated-IR translation

## Question and evidence boundary

Does an executable intermediate representation improve the reliability and cost
of translating GPU kernels to AMD XDNA2 and Intel NPU 4000, when all methods use
the same improved backend requirements and numerical guidance?

The earlier 20/20 result is development evidence: IR, repair, and shared backend
prompts changed together. This campaign reruns every arm from scratch under a
new `case-study-v2` protocol. Historical `validated-ir-v2` runs remain readable
and resumable under their original protocol; they are not comparison arms here.

Success means independent reference checks, backend host/graph numerical checks,
AMD dataflow checks where applicable, and target compilation all pass. It does
not establish execution, correctness, latency, or power on a physical NPU.

## Controlled campaign

| Arm | Intermediate | Executable IR gate | Reuse after a backend failure |
| --- | --- | --- | --- |
| `baseline` | None; direct source translation | None | None |
| `hinted_ir` | Freely chosen text | None | Regenerate each cycle |
| `structured_ir` | Schema-2.0 KernelIR | Yes | Reuse passing IR |
| `structured_ir_no_validation` | Same KernelIR schema | Disabled | Reuse schema-valid, unchecked IR |
| `structured_ir_no_reuse` | Same KernelIR schema | Yes | Regenerate and validate each cycle |

The unchecked arm retains schema validation and every backend acceptance check;
it never records a skipped IR check as a passing semantic validation. It keeps
the same reuse policy as the full method except that there is no numerical IR
gate. The no-reuse arm changes automatic reuse only: existing repair context and
instructions to preserve correct semantics remain available to the model.

All arms receive the same backend contract, compact backend diagnostics, target
profile, source manifest, numerical tolerances, and deterministic twelve-input
test suite. For layer normalization, all backend prompts include the same
stable-centering algorithm. Structured arms additionally receive the IR schema
and operation meanings; this extra representation context is part of the method.
This study holds compact feedback and numerical guidance constant; it does not
claim to isolate their individual effects or to completely separate planning
from representation and validation.

- Corpus: the existing ten development kernels, each on both targets.
- Model: `gpt-5.6-terra`, `xhigh`, through the isolated experiment Codex provider.
- Knowledge retrieval, cross-case lessons, MCTS, and optimization: disabled.
- Repetitions: three fresh experiments, with fresh databases and no candidate reuse
  between cases, targets, or repetitions. Repeat numbers are not model RNG seeds.
- Budget: ten cycles per case, including the initial attempt; stop at first pass.
- Schedule: all five arms in each repetition, shuffled with recorded scheduling
  seeds 20260916, 20260917, and 20260918. Repetitions run sequentially.
- Default concurrency: 20 cases, four container validations, six CPUs and 8 GiB
  per container. This is the existing 32-thread / 40-GiB host configuration.

Default size: **300 cases**, at most **3,000 cycles** and **5,400 model calls**
(10 direct + 20 for each of the four intermediate arms, times 60 kernel/target/repetition combinations).
The full IR arm can consume two calls in cycles that do not produce a testable
backend candidate; reuse reduces this conservative bound when it is possible.
There is no hard token or dollar ceiling. Failed translations can consume the
entire cycle budget. The previous final-run cost is not a campaign cost estimate.

## Measurements and decisions

Primary: per-target fraction solved within ten cycles, including all requested
cases in the denominator. Report blocked infrastructure and unsupported candidates
separately. Stop subsequent repetitions on blocked cases, unexpected failures,
or incomplete token telemetry; these are not evidence of model inferiority.

Secondary: first-cycle success, cycles spent, model calls, input/output/total
tokens, and summed active case time, including unsuccessful attempts. Per-trial
reports expose cycle curves and token costs. Campaign `summary.json` adds:

- Per-target/arm totals and success counts for each repetition.
- Success at cumulative token thresholds 10k, 25k, 50k, 100k, 200k, and 400k.
  These replay recorded first-success costs within ten cycles, not hard token
  caps or a token-budget-controlled generation experiment. Unknown costs do not
  count as known successes at a threshold; coverage is reported.
- Paired full-IR comparisons with all four other arms. Success differences average
  repetitions within each kernel, then average kernels equally. Exploratory 95%
  intervals bootstrap 2,000 samples of whole kernels, keeping repetitions together.
  Incomplete/blocked kernel clusters are excluded from these intervals and their
  coverage is explicit. Ten development kernels provide limited uncertainty evidence.
- Token and active-time sums on jointly solved pairs, with pair coverage. These
  conditional comparisons must not replace the full-corpus result.

Summed concurrent case time is work across pipelines, not campaign elapsed time.
The saved elapsed clock includes downtime between interruption and resume. Model
service latency is uncontrolled; no translation-time result is NPU execution speed.

Decision: if full IR retains a coverage or cost advantage against the updated
baseline, freeze this implementation and evaluate holdout. If the gap closes,
report that and narrow the contribution to supported failure mechanisms. Compare
full IR with each ablation to identify whether validation or automatic reuse
explains the difference. A null ablation result is useful, especially if the
current corpus rarely produces invalid IR. No success threshold is tuned afterward.

## Commands and recovery

Install the existing environment with `uv sync --extra dev` if needed. Before a
paid campaign, authenticate the Codex CLI and check the existing compiler images:

```bash
uv run npu-agent env smoke
bash scripts/run_case_study_v2.sh plan
bash scripts/run_case_study_v2.sh run --output runs/case-study-v2
```

The launcher defaults to `plan`, which reads the protocol and corpus without
creating a campaign, contacting a model, or starting Docker. `run` explicitly
starts the live experiment. Run from any working directory; relative paths are
resolved from the repository root. For a smaller machine, set `--case-workers`,
`--max-compiler-jobs`, `--compiler-cpus`, and `--compiler-memory` before starting.
`--repetitions`, `--kernels`, and `--targets` support a separately named pilot;
such a pilot is not the default 300-case study.

Reissue the exact same `run` command to resume; a completed campaign makes no
model calls. Ctrl+C/SIGTERM lets the existing runner checkpoint in-flight work.
Lost provider responses remain explicitly blocked with incomplete cost rather
than being silently replayed. Terminal blocked cases are not automatically retried;
inspect the evidence, fix infrastructure, and use a new campaign directory when
a fresh run is necessary. The launcher stops before paying for later repetitions.

Campaign settings, source/corpus hashes, provider version, package versions, and
compiler identities are frozen. Changing them requires a new output directory.
The snapshot includes the dirty working-tree source actually used, not just HEAD.
The campaign lock prevents two launchers from using the same directory.

```bash
bash scripts/run_case_study_v2.sh report --output runs/case-study-v2
```

Outputs: `campaign.json`, `snapshot/`, `cases.csv`, `summary.json`, `results.md`,
and `repeat-NNN/<experiment-id>/` with original checkpoints, audit logs,
`report.html`, and PNG/SVG figures for every arm.

## Separate held-out stage

Prepare and review a new corpus before seeing its translation results. Aim for
10–20 previously unused kernels, including new operation combinations where the
independent evaluator supports them, plus new shapes and numerical edge cases.
Each directory needs the existing source and `manifest.json` contract. Do not
rename the development kernels and call that generalization. New oracle/operator
support requires development and a newly frozen controlled campaign first.

The held-out stage runs the three main arms, three repetitions, both targets,
with the same frozen implementation. For ten kernels this is 180 cases. It
requires a completed controlled campaign and rejects overlapping kernel names or
identical source hashes. These guards cannot prove semantic novelty; review the
new workload selection independently. Keep the held-out corpus outside the
development corpus and do not tune prompts or implementations on its outcomes.

```bash
bash scripts/run_case_study_v2.sh plan --stage heldout \
  --corpus /absolute/path/to/heldout-kernels \
  --development-campaign runs/case-study-v2 --output runs/case-study-v2-heldout
bash scripts/run_case_study_v2.sh run --stage heldout \
  --corpus /absolute/path/to/heldout-kernels \
  --development-campaign runs/case-study-v2 --output runs/case-study-v2-heldout
```

Physical NPU execution is a later, separate study using saved artifacts and a
verified device runner. This offline launcher does not implement that stage.
