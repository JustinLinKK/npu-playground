# Part 1: training-free translation experiments

Implements the seven directions in [the study plan](../docs/study_plan.md) as
isolated ablations. Each direction has its own folder and comparison arms:

| Direction | Folder | Arms |
| --- | --- | --- |
| 1. Executable intermediate | [executable_ir](executable_ir/README.md) | `direct`, `json_summary`, `executable_ir` |
| 2. Compile/test/profile repair | [feedback_loop](feedback_loop/README.md) | `one_shot`, `repair` |
| 3. Fixed transport skeleton | [template_generation](template_generation/README.md) | `free_form`, `skeleton` |
| 4. Feature/error retrieval | [feature_retrieval](feature_retrieval/README.md) | `off`, `text`, `features` |
| 5. API constraints | [api_constraints](api_constraints/README.md) | `off`, `checked` |
| 6. Verified skill library | [verified_skills](verified_skills/README.md) | `off`, `reuse` |
| 7. Differential/metamorphic search | [candidate_search](candidate_search/README.md) | `single`, `best_of_n` |

`runner.py` shares budget accounting, the existing independent numerical oracle,
container validation, prompts, provenance, and result recording. Existing
`src/npu_agent`, historical experiments, manifests, tolerances and toolchains are
unchanged. Part 2's length contracts, masking, bucket specialization and latency
models are not part of this implementation.

## Plan review

The checkout already has executable KernelIR 2.0 and iterative repair
(`src/npu_agent/scenarios.py`, `study_v4.py`, `ir.py`). Its structured intermediate
is executable JSON, not just a descriptive summary. These experiments reuse the
interpreter, independent oracle, and restricted `StudyCompiler` rather than
assuming a one-shot baseline. "NKC-IR" in the plan maps to the repository's
`ExecutableKernelIR`; it is not a new schema.

[NPUEval Table 1](https://arxiv.org/html/2507.14403v1#S6.T1) supports the cited
21.6% to 73.5% result for one model after five **recompilations**. That means one
initial attempt plus up to five repairs, which is the convention here. Its
Appendix D relates Chess pragmas to one model's vectorization decline; it does
not establish that pragmas caused every RAG correctness regression. These are
motivations for controlled measurements, not expected gains for this checkout.

## Run

From the repository root, using the existing environment:

```bash
.venv/bin/python -m bucketed_length list
.venv/bin/python -m bucketed_length run --help
.venv/bin/python -m pytest bucketed_length/tests
```

Example launch (makes model calls once the local compiler is available):

```bash
.venv/bin/python -m bucketed_length run \
  --direction feedback_loop --arm repair \
  --source examples/classic/01_cuda_vector_add/kernel.cu \
  --manifest tests/fixtures/backends/amd_xdna2_npu2/add/manifest.json \
  --target amd_xdna2_npu2 --provider codex-cli --model gpt-5.6-terra \
  --max-repairs 5 --token-budget 100000 \
  --output runs/bucketed-length/feedback-repair
```

Change `--direction`, `--arm`, and `--output` to run another cell from the table.
The small add manifest uses 64 elements and is compatible with the CUDA source's
`n` parameter. Record it as a shape-specialized study input, not the original
4096-element classic-corpus case. Intel uses `--target intel_npu_4000`; the AMD
transport-skeleton arm deliberately rejects Intel.

Codex uses the existing isolated, tool-disabled provider pinned to
`gpt-5.6-terra`/`xhigh`. Other existing providers are accepted: `openai`,
and `openrouter`. OpenRouter uses the root configuration and
requires a matching model; credentials never enter the report. This study does
not install dependencies, pull/build images, or execute hardware binaries.
Compiler availability is checked before the first model request. A missing Docker
server or pinned image blocks the case without spending generation tokens.

Outputs must be new directories. A run records the source and manifest, config,
Git SHA, implementation snapshots and hashes, compiler/evaluator identity,
library snapshot, every prompt/response and usage record, each attempted bundle,
full compiler artifacts, `report.json`, and `evidence.sha256.json`. There is no
automatic resume: interrupted or unaccounted requests must not silently become
free retries. Hashes detect accidental changes, not malicious replacement of a
report and its manifest together.

Exit codes: `0` completed, `1` failed/budget exhausted, `3` blocked/interrupted.
Argument and filesystem errors are errors, not correctness results.

Create a comparison JSON without model calls; saved evidence is verified first:

```bash
.venv/bin/python -m bucketed_length report \
  runs/bucketed-length/feedback-one-shot runs/bucketed-length/feedback-repair \
  --output runs/bucketed-length/feedback-comparison.json
```

## Run all seven directions in parallel

This launches **all 16 comparison arms**, with one process per active direction,
then renders the combined results automatically:

```bash
.venv/bin/python -m bucketed_length.run_all \
  --workers 7 --repetitions 1 --token-budget 200000 \
  --provider codex-cli --model gpt-5.6-terra \
  --output runs/bucketed-length/all-directions
```

The default input is the same 64-element CUDA addition case used above. Each
direction runs its arms/repetitions sequentially; the seven directions overlap.
Use `--source PATH --manifest PATH --target TARGET` for another case. The launcher
retains unsupported combinations (such as the AMD skeleton on Intel) as failed or
blocked records. It does not silently remove an arm. A run directory must be new.

`--max-repairs`, `--population`, `--token-budget`, `--selection-seed`, `--docker`
and `--compiler-timeout` pass through to each child. The token cap is **per case**:
the default is 16 × 100,000 accounted tokens, with the existing one-call overrun
rule. Seven workers allow up to seven compiler containers, each capped at 2 CPUs
and 4 GiB, plus process overhead. Reduce `--workers` for a smaller machine.
The example explicitly uses 200,000 tokens to leave room for four independent
IR-plus-code search candidates; the CLI default remains 100,000. Inspect actual
population coverage in the report instead of assuming every candidate ran.

Add `--library PATH` to seed retrieval with previously verified examples. Inputs
are frozen and each case gets a separate copy of that library; concurrent runs
cannot publish new skills into each other's prompts or mutate the original
library. Without a seed library, retrieval remains empty and the report explicitly
says that populated retrieval and cross-task skill growth were not measured.

Preview the exact commands and render an all-NOT-RUN report without model/compiler
calls (choose a separate output directory):

```bash
.venv/bin/python -m bucketed_length.run_all --dry-run \
  --output runs/bucketed-length/all-directions-preview
```

`suite.json` records every planned command and child exit code. Per-case logs are
in `logs/`; individual evidence bundles are in `cases/`. When the suite finishes,
`results/` contains:

- `REPORT.md`: comparison tables, limitations, and failure/log links.
- `cases.csv` and `summary.json`: all planned cases, including missing, failed,
  blocked and invalid-evidence rows.
- `outcomes.png` / `.svg`: accepted/failed/blocked counts for every arm.
- `resources.png` / `.svg`: recorded tokens and summed active case time, with
  actual suite wall time labeled separately.

Rerender existing results without restarting experiments:

```bash
.venv/bin/python -m bucketed_length.run_all --render-only \
  --output runs/bucketed-length/all-directions
```

Rendering verifies each case's evidence hashes and suite input/settings bindings.
Ctrl+C/SIGTERM stops new cases, lets active cases finish, then renders a partial
report; active cases may still make their remaining bounded model calls. There is
no automatic retry or resume. Exit code `0` means all cases were accepted (or a
dry-run/render-only command succeeded), `1` means one or more cases did not pass,
and `130` means the suite was interrupted. Rendering preserves the individual
raw reports and replaces only derived files under `results/`.

## Measurement contract

The `guided-breadth-v2` revision supplies the existing pinned API/transport
guidance to every free-form arm and keeps compact earlier-failure history during
repairs. Best-of-N generates all initial candidates before starting repairs,
then repairs surviving candidates in round-robin order. Validated IR is reused
within each candidate. Reports include population coverage and retrieval hits
so an unexercised treatment is visible.

The subsequent `guided-compute-v3` revision also tells skeleton generation which
vector APIs the evaluator supports and uses scalar pointer loops for unsupported
multiply/dot operations. The study preserves its frozen v2 results and reports
the four-case GEMV development follow-up separately, with unchanged acceptance
checks. Read the [combined report](../runs/bucketed-length/guided-study-20261001/results/REPORT.md).
The repository includes the derived report, summary, protocol, audit, and chart;
links to raw per-case bundles require the original local run directory.

The recovery study in `runs/bucketed-length/guided-study-20261001/` freezes three
verified seed examples and evaluates larger addition, reduction, and GEMV shapes,
with two repetitions per arm and a 200,000-token ceiling. Its `protocol.json`,
input hashes, and per-suite commands document the exact setup. Regenerate the
combined explanatory report without model calls:

```bash
.venv/bin/python -m bucketed_length.report_study \
  runs/bucketed-length/guided-study-20261001
```

- Use the same source/manifest, target image/evaluator fingerprint, model/settings,
  token cap, repair cap, and frozen library for paired arms. Repeat cases in a
  predeclared order; the CLI runs one case/arm at a time and makes no significance
  or benchmark-performance claims.
- The cap counts input plus output tokens for **all** calls, including invalid
  responses, intermediate generation and every population member. Missing usage
  blocks the run. A response crossing the cap is charged but not evaluated. The
  cap is checked after a request and can overrun by one call; it is not a hard
  provider-spending limit. A previously verified candidate is retained if a later
  search/optimization call exhausts the cap; the overrun remains visible and
  disqualifies that run from automatic skill admission. Different pivots can use
  different call counts.
- Compilation and numerical correctness are distinct outcomes. Acceptance
  requires the existing `offline-validated` policy, with source/manifest/toolchain
  bindings, independent reference checks, host execution, and AMD dataflow checks.
  A JSON schema pass, source symbol, or unavailable compiler is never success.
- Validated IR is checked against the independent built-in reference; generated
  IR never replaces that reference unchecked. No CUDA execution or formal source
  equivalence is claimed. Random/adversarial tests establish only their tested
  numerical contract, and the old layer-normalization discrepancy remains open.
- Physical latency, vectorization and stalls are `null` unless a trusted evaluator
  supplies validated physical-device evidence. Host time, source intrinsics and
  static metrics are not device-performance measurements. The current executor
  supplies no vectorization/stall counters; no new device profiler is implemented.
- The search arm's private tests are **selection data**, not an untouched final
  benchmark. Their outputs/diagnostics never enter a repair prompt. Use a separate
  reviewed holdout for a later confirmatory experiment.

## Verification scope

Focused tests cover all arms using explicitly synthetic providers and compiler
records, IR rejection/reuse, repair and token limits, infrastructure blocking,
toolchain filtering, API rejection/repair, verified admission and tampering,
private-input selection, metamorphic relations, and native C++ microkernels with
ASan/UBSan on all twelve existing input families. Those tests do not measure a
model's translation quality, target compilation, or physical NPU execution.
