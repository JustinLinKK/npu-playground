# NPU Agent Playground

For the current research results and next-step decision, start with the
[cloud analysis handoff](docs/cloud-analysis-handoff.md).

This repository contains an agentic CUDA, Triton, and HIP-to-NPU translation
framework. It uses a hardware-neutral semantic IR, LangGraph orchestration,
SQLite persistence, backend-specific code generation, compiler-guided debugging,
and Monte Carlo tree search (MCTS).

The repository also provides hardware-free compiler environments for:

- AMD Ryzen AI XDNA2 (`npu2`): IRON/MLIR-AIE, Peano kernels, `xclbin`, and
  instruction-stream generation.
- Intel NPU 4000: OpenVINO graph execution on CPU plus Compiler-in-Plugin
  offline NPU blob generation.

These environments validate independent reference math, Intel CPU graphs, and a
bounded AMD C++/dataflow subset, then compile real target artifacts. They do not
execute those target binaries or predict device latency. A result is only marked as
hardware-measured when a real-device runner is explicitly configured.

## Install

Python 3.12 or newer, `uv`, Docker, and at least one configured model provider
are required.

```bash
uv sync --extra dev
uv run npu-agent --help
```

The OpenAI API provider is selected by default and requires both
`OPENAI_API_KEY` and either `--model` or `NPU_AGENT_OPENAI_MODEL`. Two
noninteractive CLI providers are also supported:

```bash
uv run npu-agent translate kernel.cu \
  --manifest manifest.json \
  --provider codex-cli

uv run npu-agent translate kernel.cu \
  --manifest manifest.json \
  --provider claude-cli \
  --model sonnet
```

Codex runs with an ephemeral session, a read-only sandbox, and a JSON output
schema. Claude runs in noninteractive plan mode with a JSON schema and no
session persistence. Generated source is returned as data; providers do not
write candidate files directly.

## Compiler environments

Build and verify both pinned offline compiler images:

```bash
uv run npu-agent env build
uv run npu-agent env smoke
uv run npu-agent env capabilities --json
uv run npu-agent validate-suite tests/fixtures/backends --validation-policy offline-validated
```

Build or test only one profile with `--targets amd_xdna2_npu2` or
`--targets intel_npu_4000`.

Generated programs are compiled in network-disabled, read-only containers with
dropped Linux capabilities and bounded CPU, memory, and process counts. Only a
dedicated output directory and temporary work/cache locations are writable.

The CLI defaults to `--validation-policy offline-validated`. `compile-only` retains
compilation-only acceptance; `target-executed` returns exit 3 without a validated
executor. See [the validation guide](docs/validation.md) for the supported surface,
corpus reproduction, stage evidence, exit codes, and execution adapter contract.
The [verified upgrade results](docs/validation-results.md) record exact image/tool
identities, test coverage, and saved evidence from the local acceptance run.

## Translate a kernel

Every kernel requires a JSON manifest describing its exact signature, static
shapes, dtypes, aliasing, numeric tolerance, and reference oracle.

```bash
uv run npu-agent translate \
  examples/classic/01_cuda_vector_add/kernel.cu \
  --manifest examples/classic/01_cuda_vector_add/manifest.json \
  --targets amd_xdna2_npu2,intel_npu_4000 \
  --provider codex-cli \
  --search-rounds 3 \
  --branching-factor 3 \
  --debug-retries 2
```

The default search evaluates one draft and three proposals in each of three
MCTS expansion rounds for each target. Every candidate receives a recorded
compile attempt. Compiler failures enter the bounded debug loop.

Run the complete ten-kernel corpus with the real Codex CLI:

```bash
uv run npu-agent suite examples/classic \
  --provider codex-cli \
  --targets amd_xdna2_npu2,intel_npu_4000 \
  --search-rounds 3 \
  --branching-factor 3 \
  --debug-retries 2
```

Live runs are intentionally expensive: the full command searches both backends
for all ten examples and may invoke the configured model many times.

## Three-scenario experiment

The version-2 experiment compares three translation workflows using Codex CLI,
`gpt-5.6-terra`, and explicit `xhigh` reasoning on every call:

| Scenario | One programming cycle |
| --- | --- |
| `baseline` | GPU source → NPU code → container validation |
| `structured_ir` | GPU source → project KernelIR → executable IR validation → NPU code → container validation; backend repairs reuse validated IR |
| `hinted_ir` | GPU source → freely chosen intermediate text → NPU code → container validation |

Both intermediate workflows initially use separate intermediate and backend model
calls. Structured IR is regenerated after IR failure and reused during backend
repairs; the hint-based workflow regenerates its intermediate on every retry.
Only `structured_ir` receives the project IR schema and executable
IR feedback. This compares the full validated IR pipeline, including its additional
feedback. All scenarios share backend requirements, twelve deterministic input
cases, numeric tolerances, and the `offline-validated` evaluator. They use no MCTS,
optimization, knowledge retrieval, or cross-case lessons.

A case stops at its first passing translation or after **10 cycles**, including the
initial attempt. A malformed response or invalid IR consumes a cycle without
running later stages. Failed numerical checks, compilation, and unsupported
candidate code return feedback for repair. Unavailable providers, infrastructure,
or independent references are recorded as blocked. These are offline validation
results; they do not establish execution of the emitted NPU binary.

Run the ten kernels × two targets × three scenarios (60 cases):

```bash
uv run python scripts/run_experiment.py --max-cycles 10 --runs-dir runs
```

For a 9950X3D with 32 logical CPUs and at least 40 GiB available to Linux/Docker,
overlap 20 independent cases across all three scenarios while allowing four local
validation jobs, each limited to 6 CPUs and 8 GiB:

```bash
uv run python scripts/run_experiment.py --max-cycles 10 --runs-dir runs \
  --case-workers 20 --max-compiler-jobs 4 --compiler-cpus 6 --compiler-memory 8g
```

This still runs all 60 cases. Model calls overlap while waiting for remote Terra
responses; actual throughput also depends on provider capacity and rate limits.
The launcher defaults host OpenMP/BLAS pools to one thread unless overridden in
the environment. The validation limit covers the entire container pipeline,
including image metadata checks, across all workers. Defaults remain one case
worker, two validation slots, and 4 CPUs / 8 GiB per container.

For a focused run, use `--kernels cuda_vector_add --targets intel_npu_4000`.
Provider, model, and validation-policy arguments remain accepted but conflicting
values are rejected. Experimental Codex calls run in temporary directories with
user configuration, repository instructions, tools, memory, plugins, and delegation
disabled; authentication remains available. Generated code is tested by the supervisor.

Resume an interrupted experiment with its recorded budget and completed stages:

```bash
uv run python scripts/run_experiment.py --resume EXPERIMENT_ID --runs-dir runs
```

To evaluate an IR revision against an existing baseline, run just the twenty
structured cases and keep the baseline experiment directory unchanged:

```bash
uv run python scripts/run_experiment.py --scenarios structured_ir --max-cycles 10 \
  --case-workers 20 --max-compiler-jobs 4 --compiler-cpus 6 --compiler-memory 8g
```

New runs record protocol revision `validated-ir-v2`. It pins generated executable
IR to schema 2.0, explains operation contracts, and preserves the last backend
candidate and diagnostics across intervening IR failures. After a validated IR
reaches backend testing, subsequent backend repairs reuse that exact IR and its
validation evidence; the reused stage records its origin cycle, zero executed
cases, zero validation duration, and no new model call. Invalid IR still consumes
a cycle and must be regenerated. The hint-based workflow still regenerates its
intermediate on every cycle. Repair prompts retain failed numerical comparisons
and compiler diagnostics while omitting container commands and artifact hashes;
full validation records remain persisted. Shared backend prompts include verified
MLIR transport syntax, portable scalar numeric types, and OpenVINO tensor naming.
Comparisons with older runs therefore measure the combined pipeline revision;
they do not isolate an IR-only effect. Historical reports remain readable, but
resuming checkpoints from another protocol revision is rejected. Scenario
selection is recorded and cannot change on resume.

Concurrency and container resource settings are recorded and restored on resume;
explicit flags can override them. Ctrl+C or SIGTERM stops scheduling new work,
waits for current model calls and validation jobs to checkpoint, then exits.
The experiment directory is printed at startup for use with `--resume`.

Every case checkpoints model responses, intermediate representations, validation
results, and feedback in its `report.json`; completed stages are reused. If an
in-flight provider response was lost before checkpointing, the case is blocked
with incomplete usage rather than silently repeating a potentially paid call.
Source/manifest changes, target-profile changes, and compiler/evaluator fingerprint
changes are rejected. Historical version-1 reports remain readable, but cannot be
resumed using the new protocol.

Each experiment owns an isolated `state.sqlite`. Outputs include `experiment.json`,
`metrics.jsonl`, `metrics.csv`, `summary.json`, `report.html`, per-case checkpoints
and artifacts, provider/compile audit logs, and PNG/SVG charts. Primary measurements
are cycles to first success and total tokens across **all** calls, including both
intermediate stages, rejected responses, and repairs. Cached input and reasoning
tokens are breakdowns, not extra tokens added to the total. Missing usage leaves
the exact total null and exposes a known subtotal and completeness flag.

Unsolved cases retain cycles and tokens spent; their cycles-to-success is null.
Paired efficiency uses jointly solved cases, with explicit coverage and a separate
complete-usage requirement for token comparisons. Each target is reported separately.

The explicitly named Python API `translate_one_shot` and the legacy `npu-agent baseline`
command remain available for reproducing a single-call baseline; they are separate
from the version-2 experiment's iterative `baseline` scenario.

## IR and backend contracts

The versioned `KernelIR` represents logical inputs, outputs, iteration domains,
index expressions, operations, reductions, side effects, preconditions, and
numeric behavior. Hardware scheduling terms such as CUDA blocks or AMD tiles
are rejected from semantic fields. The IR is executed with NumPy and compared
with an independently implemented manifest oracle on twelve deterministic cases
before backend generation begins. Primitive IR 2.0 supports composed operations;
prose metadata is descriptive and unsupported executable contracts are rejected.
The `center` primitive casts to its declared floating-point accumulation dtype,
subtracts the first element along one axis, then subtracts the mean of those
differences with dimensions retained. This executable order reduces cancellation
when composing normalization in float32; backend generation receives the same
operation definition. It does not change the oracle or correctness tolerances.

AMD candidates must contain exactly `design.py` and `kernel.cc`. The design must
support the standard IRON `--dev npu2`, `--emit-mlir`, `--xclbin-path`, and
`--insts-path` arguments. The supervisor emits once and compiles the saved NPU2 MLIR and
exact `kernel.cc` with a trusted Peano/aiecc driver. Offline acceptance additionally
requires source-coupled host/dataflow validation and independent comparison.

Intel candidates must contain exactly `model.py` with a side-effect-free
`build_model(manifest=None)` function returning an `openvino.Model`. Input and output tensor names must explicitly match the manifest. Construction,
serialized-graph execution, and target compilation run in separate processes.
The graph is checked against the oracle through OpenVINO CPU before it is compiled for
`NPU_PLATFORM=4000` and exported as a blob.

## Persistence and memory

Runtime data is stored under `.npu-agent/` by default:

- `state.sqlite` contains LangGraph checkpoints, runs, candidates, evaluations,
  compile attempts, provider evidence, agent knowledge, and compiler lessons.
- `runs/<run-id>/` contains generated sources and content-hashed compiler
  artifacts.

Automatic knowledge seeding, retrieval, and learned-lesson storage/reuse are
**disabled globally** for this experiment. Existing catalog data and database tables
are preserved, and manual catalog-maintenance code remains available. Active prompts
contain no knowledge or lesson sections; run, candidate, provider, and validation
logging continue normally.

Inspect existing records without activating retrieval:

```bash
uv run npu-agent runs show RUN_ID
uv run npu-agent memory list --hardware npu2
```

The Python API exposes `build_workflow(settings)`, asynchronous
`translate(request, settings)`, and asynchronous
`translate_one_shot(request, settings)`. `Settings.role_providers` and
`Settings.role_models` can override the provider or model independently for
the analysis, coding, optimization, and debug roles.

## Development checks

```bash
uv run pytest
uv run python -m compileall -q src tests playground/tools
git diff --check
```

The example corpus is documented in
[`examples/classic/THIRD_PARTY_NOTICES.md`](examples/classic/THIRD_PARTY_NOTICES.md).
