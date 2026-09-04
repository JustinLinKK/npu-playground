# NPU Agent Playground

This repository contains an agentic CUDA, Triton, and HIP-to-NPU translation
framework. It uses a hardware-neutral semantic IR, LangGraph orchestration,
SQLite persistence, backend-specific code generation, compiler-guided debugging,
and Monte Carlo tree search (MCTS).

The repository also provides hardware-free compiler environments for:

- AMD Ryzen AI XDNA2 (`npu2`): IRON/MLIR-AIE, Peano kernels, `xclbin`, and
  instruction-stream generation.
- Intel NPU 4000: OpenVINO graph execution on CPU plus Compiler-in-Plugin
  offline NPU blob generation.

These environments validate reference semantics and compilation. They do not
emulate NPU execution or predict device latency. A result is only marked as
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
```

Build or test only one profile with `--targets amd_xdna2_npu2` or
`--targets intel_npu_4000`.

Generated programs are compiled in network-disabled, read-only containers with
dropped Linux capabilities and bounded CPU, memory, and process counts. Only a
dedicated output directory is writable.

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

## One-shot baseline and experiment

The baseline makes exactly one structured provider call and one compile attempt
per kernel-target pair. It does not create semantic IR, invoke LangGraph, query
memory, optimize, run MCTS, or repair a failed result.

```bash
uv run npu-agent baseline \
  examples/classic/01_cuda_vector_add/kernel.cu \
  --manifest examples/classic/01_cuda_vector_add/manifest.json \
  --targets amd_xdna2_npu2,intel_npu_4000 \
  --provider codex-cli
```

Run the reproducible baseline-versus-agentic experiment with all ten kernels
and both targets:

```bash
uv run python scripts/run_experiment.py \
  --provider codex-cli \
  --runs-dir runs
```

Use comma-separated `--kernels` or `--targets` for a focused run, and resume an
interrupted experiment without repeating completed cases:

```bash
uv run python scripts/run_experiment.py --resume EXPERIMENT_ID --runs-dir runs
```

Each experiment owns an isolated `state.sqlite`. Baselines run first; agentic
cases then see the committed catalog plus only verified lessons accumulated in
that experiment. Failures are recorded and later cases continue. Generated
outputs include `experiment.json`, `metrics.jsonl`, `metrics.csv`,
`summary.json`, method-specific artifacts, PNG/SVG charts, per-target search
trees, `logs/provider_calls.jsonl`, `logs/compile_attempts.jsonl`, and
`report.html`. Intel CPU p50/p95 is labeled as a host diagnostic;
target-NPU latency and speedup charts are emitted only when a hardware runner
provides real measurements.

## IR and backend contracts

The versioned `KernelIR` represents logical inputs, outputs, iteration domains,
index expressions, operations, reductions, side effects, preconditions, and
numeric behavior. Hardware scheduling terms such as CUDA blocks or AMD tiles
are rejected from semantic fields. The IR is executed with NumPy and compared
with the manifest oracle before backend generation begins.

AMD candidates must contain exactly `design.py` and `kernel.cc`. The design must
support the standard IRON `--dev npu2`, `--emit-mlir`, `--xclbin-path`, and
`--insts-path` arguments. Successful offline evidence requires NPU2 MLIR and
nonempty, inspectable `xclbin` and instruction files.

Intel candidates must contain exactly `model.py` with a side-effect-free
`build_model(manifest=None)` function returning an `openvino.Model`. The graph
is checked against the oracle through OpenVINO CPU before it is compiled for
`NPU_PLATFORM=4000` and exported as a blob.

## Persistence and memory

Runtime data is stored under `.npu-agent/` by default:

- `state.sqlite` contains LangGraph checkpoints, runs, candidates, evaluations,
  compile attempts, provider evidence, agent knowledge, and compiler lessons.
- `runs/<run-id>/` contains generated sources and content-hashed compiler
  artifacts.

Inspect a run or memory:

```bash
uv run npu-agent runs show RUN_ID
uv run npu-agent memory list --hardware npu2
uv run npu-agent memory seed
uv run npu-agent memory search "matmul tiling float32 accumulation" \
  --role optimize --target amd_xdna2_npu2
```

Import future role-scoped knowledge:

```bash
uv run npu-agent memory import notes.md \
  --role optimize \
  --title "XDNA2 matrix multiplication notes" \
  --tags "matmul tiling" \
  --backend amd_xdna2 \
  --hardware npu2
```

Verified compiler repairs are retrieved only for the same backend, hardware,
and compiler fingerprint. Unresolved lessons remain auditable but are excluded
from optimization prompts.

New databases are seeded idempotently from
`src/npu_agent/data/knowledge.json`. Entries are role- and operation-tagged and
exactly scoped by target ID, vendor, backend, hardware, and compiler
fingerprint. The AMD material covers pinned MLIR-AIE 1.4.2 IRON, ObjectFifo,
Peano ABI, data movement, reductions, tiled GEMM, transpose, softmax, and
convolution patterns. The Intel material covers OpenVINO model construction,
the ten graph patterns, CPU equivalence, NPU 4000 Compiler-in-Plugin blob
export, capability checks, and performance properties. Intel's supported
interface is graph compilation; this project does not claim an equivalent
arbitrary low-level kernel SDK.

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
