# XDNA2 direct baseline

## Replay on a Ryzen AI Ultra machine

This checkout contains the forty-task corpus, launchers, provider configurations,
and `examples/kernelbench/source-validation-report.json`. The report records the
earlier CUDA/upstream/oracle source check. Evaluation verifies its source hashes;
using it on an NPU-only machine does not claim a fresh CUDA check there. A new
CUDA validation report can be passed as the launcher's second argument.

Install Python 3.12 and project dependencies with `uv sync --python 3.12 --extra dev`.
Build the pinned compiler image:

```bash
docker build -t npu-playground-amd-xdna2:v1.4.2 -f playground/dockerfiles/Dockerfile.xdna2-offline .
```

Set the provider key in ignored `config.yaml` (`openrouter.api_key` for Qwen or
`openai.api_key` for the direct route), then run
`bash scripts/run_xdna2_baseline_openrouter_budget.sh` for the configured Qwen
route or `bash scripts/run_xdna2_baseline_openai.sh` for the model in
`configs/xdna2-baseline-openai.json`. These commands make paid model calls and
print a fresh campaign root. Keep that root for reports and physical replay.
The OpenRouter route relies on account-side dollar limits; the direct OpenAI
route has no local dollar cap.

For physical NPU correctness, install the Ryzen AI driver/XRT and MLIR-AIE
1.4.2 in the **host** Python environment. Follow the
[MLIR-AIE setup guide](https://github.com/Xilinx/mlir-aie) and confirm
`xrt-smi examine` sees an NPU2 device. `pyxrt` must import in the same Python
used below. The pinned container still compiles artifacts, while this step runs
them through XRT on the local Ultra NPU. Linux or WSL must expose the NPU to
the host environment. First smoke-check one task that the offline report marks
accepted, then check all available accepted artifacts:

```bash
.venv/bin/python scripts/replay_xdna2_local_npu.py RUN_ROOT/evaluation \
  --output RUN_ROOT/local-npu-smoke --kernel ACCEPTED_TASK_NAME
.venv/bin/python scripts/replay_xdna2_local_npu.py RUN_ROOT/evaluation \
  --output RUN_ROOT/local-npu-all
```

Each new output directory contains `summary.json` and hash-bound execution
requests. The runner verifies saved inputs and binaries, executes `final.xclbin`
with `insts.bin` through XRT, and compares physical outputs with the independent
oracle. It does not rewrite the frozen offline result or measure hardware
latency. [MLIR-AIE 1.4.2 documents this artifact-loading API](https://xilinx.github.io/mlir-aie/1.4.2/programming_guide/compilation_stages/#consuming-external-artifacts-bring-your-own).

`xdna2-baseline-v1` runs one direct translation arm on XDNA2 only, with one
repetition per task. It uses the classic ten tasks unchanged and thirty pinned,
reduced-shape KernelBench CUDA adaptations. The new protocol is separate from the
historical v4 three-arm comparison. It supplies no intermediate representation,
retrieved implementation, candidate population, or arithmetic solution.

The current prompt is revision 2 (unguided): only the source, public interface,
target metadata and container invocation/submission contract are supplied. It
contains no IR syntax examples, backend API lessons, transport templates, tiling
advice or numerical implementation recipes. Required command flags and toolchain
identifiers remain part of the container contract. Failed attempts still receive
evaluator diagnostics and the model's own previous candidate, up to the existing
attempt/token/time limits; first-attempt and eventual acceptance are separate metrics.

Earlier frozen revision-1 campaigns used explicit MLIR/IRON guidance and must be
reported as guided results. Run revision 2 in a fresh campaign directory with fresh
calibration; never resume or overwrite those historical runs with the new prompt.

The configuration is `configs/xdna2-baseline-v1.json`. Qwen3.8-27B is pinned to
OpenRouter's `parasail/fp8` route with reasoning enabled and routing fallback
disabled. The previous `deepinfra/bf16` route was unavailable on October 8, 2026;
the user selected an available FP8 route. Comparisons with earlier BF16 campaigns
therefore change provider/precision as well as prompt guidance.
The launcher removes `OPENROUTER_API_KEY` from child environments so the
provider reads `openrouter.api_key` from the repository's `config.yaml`. That file
is not included in frozen snapshots or logs.

## Preparation

The committed corpus records upstream revision, license, source hash, constructor
arguments, shapes and tolerances. `scripts/prepare_kernelbench_corpus.py` reproduces
it into a new directory; it refuses to overwrite an existing corpus.

Execute all thirty CUDA sources against pinned upstream PyTorch and independent
NumPy references, using an existing CUDA-enabled torch environment:

```bash
PYTHONPATH=src python \
  scripts/validate_kernelbench_sources.py \
  --output runs/my-source-validation \
  --nvrtc /path/to/libnvrtc.so
```

The source-check output must be new. Every task checks twelve deterministic input
families and saves CUDA, upstream and independent outputs plus three comparisons.
Source or manifest changes invalidate this evidence. This is finite numerical
validation, not a proof of universal source equivalence. Existing classic source
contracts retain their historical limitations.

## Calibration and evaluation

```bash
.venv/bin/python research/run_xdna2_baseline.py calibrate --revision 1
.venv/bin/python research/run_xdna2_baseline.py evaluate --revision 1 \
  --source-report examples/kernelbench/source-validation-report.json
.venv/bin/python research/run_xdna2_baseline.py report
```

Calibration uses four separately authored small tasks and at most two attempts per
task. The unguided gate requires a completed independent container smoke test with
the offline contract met, and four terminal calibration tasks without unresolved
infrastructure, reference or provider failures. Model-generated API errors,
unsupported designs and compilation/numerical failures remain recorded outcomes;
they do not trigger prompt tutoring or prevent the forty-task evaluation. Candidate
acceptance still requires all existing validation stages. The earlier guided gate
also rejected API/simulator failures; its frozen campaigns retain that policy.
Evaluation checks all thirty source reports, the calibration evidence, and exact
prompt/evaluator source hashes before paid requests. Freeze guidance after this
gate; a changed evaluator requires a separate revision and fresh calibration.

Both phases share `runs/xdna2-baseline-20261001/budget.json`. The $50 threshold
stops new requests; up to four already outstanding calls can settle afterward.
Unknown billing or lost responses stop further spending. They are not silently
replayed. Qwen3.8 requests enforce a 1,800-second total response deadline even if the service
sends whitespace heartbeats, and retain the `X-Generation-Id` header when available
for billing reconciliation. Each evaluation task stops at its first pass, ten total attempts, or
200,000 tokens. A stage can cross its token threshold and is still charged.
The five-hour per-case time guard can stop a case before ten attempts when provider
responses are slow. Four case workers share two container slots, each using two
CPUs, 4 GiB and a 600-second timeout. Completed cases are not rerun on resume.

### OpenRouter-side dollar limits

To delegate dollar-budget enforcement entirely to OpenRouter, run:

```bash
LC_ALL=C.utf8 bash scripts/run_xdna2_baseline_openrouter_budget.sh
```

This creates a fresh timestamped campaign, runs calibration, and starts the forty
evaluation tasks only after the preparation gates pass. It uses the key in
`config.yaml` and the existing source-validation report. It neither reads nor
modifies the old spending ledger. Returned costs remain in per-attempt reports,
but there is no local dollar ceiling or stop for a missing cost field. Check
total billing, including calibration and preflight, on OpenRouter.

The ten-attempt, 200,000-token and time limits, four case workers, two compiler
jobs, and validation requirements still apply. Provider errors (including HTTP
402) still stop new work; lost responses are retained and never silently replayed.
The script prints its output directory. To resume the same campaign, pass that
directory as the first argument. An optional second argument selects a different
source-validation report. Frozen identity and blocked-response checks still apply.

For individual phases, use `research/run_xdna2_baseline.py` with
`--budget-on-openrouter --root NEW_RUN_ROOT`. The original launcher invocation
without this flag retains its $50 guard. Budget modes cannot be switched within
an existing campaign.

### Direct OpenAI

Fill in the `openai.api_key` value in the ignored repository-root `config.yaml`:

```yaml
openai:
  api_key: "YOUR_OPENAI_API_KEY"
```

The experiment settings are in `configs/xdna2-baseline-openai.json`, including the
configured model ID `gpt-6-astra`, medium reasoning effort, a 32,768-output-token limit
and an 1,800-second response deadline. The model in this JSON file is authoritative;
`config.yaml` supplies only the key. The JSON has no temperature or top-p because
GPT-6 Astra rejects them. For other OpenAI models, `provider.temperature` and
`provider.top_p` are sent only when set, and configuration validation allows them
only with `"reasoning": false` (effort `none`). Preflight confirms that the exact
model accepts them before any task runs. Astra has no reasoning-disabled mode.
The request uses the Responses API, strict JSON schema, no tools, no stored
conversation and no automatic retries. Keys are excluded from snapshots and logs.
The request format follows the [official OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
The simple OpenAI schema probe may report zero reasoning tokens even at medium
effort. Preflight accepts a reported zero and requires reasoning-token telemetry
to be present; schema validation and complete input/output accounting still apply.

```bash
LC_ALL=C.utf8 bash scripts/run_xdna2_baseline_openai.sh
```

The launcher prints a fresh campaign directory, checks access to the exact model
before any generation request, runs the four calibration tasks and evaluates all
forty tasks only after the preparation gates pass. The access check makes no generation calls;
an unavailable model fails with no model fallback.
The supported direct OpenAI IDs also include `gpt-6-sol` and `gpt-6-astra`;
choose one explicitly in `provider.model` in the experiment JSON.
If changing the model after a failed launch, start a fresh campaign using the
launcher without a `RUN_ROOT` argument. Existing campaigns retain their frozen model.

This launcher has **no local dollar cap**. OpenAI Responses usage supplies token
counts, not billed dollar amounts; reports mark dollar billing unavailable. Check
spending in your OpenAI account. The ten-attempt, 200,000-token, five-hour per-task,
four-worker and two-compiler-job limits remain unchanged. Provider errors stop new
work; unsuccessful or lost responses are preserved without automatic replay.

To resume, pass the printed campaign directory as the first argument. An optional
second argument selects the source-validation report, as with the OpenRouter script.
The default report is `examples/kernelbench/source-validation-report.json`.
The launcher clears `OPENAI_API_KEY` in child environments so it uses `config.yaml`.
Use a fresh campaign for the new provider; historical Qwen campaigns remain frozen.
Results are saved to `RUN_ROOT/evaluation/results/REPORT.md`.

For individual phases:

```bash
.venv/bin/python research/run_xdna2_baseline.py calibrate \
  --config configs/xdna2-baseline-openai.json --budget-on-openai --root runs/xdna2-openai-01
.venv/bin/python research/run_xdna2_baseline.py evaluate \
  --config configs/xdna2-baseline-openai.json --budget-on-openai --root runs/xdna2-openai-01 \
  --source-report examples/kernelbench/source-validation-report.json
```

The October 2 campaign `xdna2-baseline-openrouter-budget-20261002T090858388326470Z`
passed add, transpose and reduce on their first attempts. GEMV was cancelled at
the old 900-second response deadline before returning a candidate. OpenRouter's
generation record reports 26,551 completion tokens and $0.050185425 for that
cancelled request; the original report's zero recorded usage is not zero billing.
The record is saved as `gemv-generation-metadata.json` beside `calibration-v1`.
This is a provider timeout, not a compiler or numerical failure. The new deadline
requires a fresh campaign: do not resume the frozen 900-second campaign or clear
its blocked state. The locale override above avoids the unavailable `en_US.UTF-8`
setting on this WSL installation; that warning did not cause the calibration failure.

## Evidence and reporting

The report is `evaluation/results/REPORT.md`, with task, attempt and exclusion CSVs
and PNG/SVG plots. Every planned task remains in the denominator. Confirmed kernel
errors include generated target code/verifier, arithmetic, ABI, memory and dataflow
defects. Emitter invocation failures, simulator coverage limitations, compiler
crashes and infrastructure faults are separately disclosed. Unknown diagnostics
remain unclassified. A later unsupported simulator stage does not erase an
earlier target source failure. Physical-execution and prerequisite blocks do not
create extra failed attempts.

Acceptance requires independent reference validation, target compilation, host
execution and dataflow simulation. It does not establish physical XDNA2 execution,
latency, throughput, vectorization quality or power. Summed concurrent case time
is not campaign elapsed time.

Focused checks:

```bash
.venv/bin/python -m pytest tests/test_xdna2_baseline.py tests/test_study_v4.py \
  tests/test_study_v4_provider.py tests/test_research_budget.py tests/test_research_parallel.py
NPU_AGENT_CONTAINER_TESTS=1 .venv/bin/python -m pytest \
  tests/test_xdna2_baseline.py -k scalar_container_probe
```

## October 1 execution status

The local source checks passed all 360 cases, and six representative real XDNA2
container probes passed. The schema probe completed, but the four calibration
requests started at `2026-10-02T04:36:53Z` returned no translations while their
connections continued receiving small packets. They exceeded the intended total
deadline and were locally stopped. That exposed the inactivity-timeout issue now
covered by a regression test and the total-deadline implementation above.

`runs/xdna2-baseline-20261001/evaluation/results/REPORT.md` retains all forty tasks
as **NOT RUN**. The calibration artifacts, original in-flight reports, and ledger
are preserved. The known $0.00040665 schema charge is a lower bound: four pending
requests have unreconciled billing and no locally captured generation IDs.
Continuing under the original local guard requires provider-side reconciliation
of those requests and a fresh calibration revision. The explicitly selected
OpenRouter-budget mode above starts a separate campaign without changing or
reconciling these historical records. No kernel-translation accuracy was measured.
