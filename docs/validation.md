# Offline validation

The supported milestone is independent math, host/source execution, AMD dataflow
modeling, and actual target compilation. **Neither image executes the emitted NPU
binary.** `target_execution` remains `blocked: NO_VALIDATED_TARGET_EXECUTOR`, with
unknown correctness and null hardware latency. No NPU hardware, host NPU driver,
firmware change, GPU pass-through, or provider credentials are needed here.
See [the verified results](validation-results.md) for the completed local acceptance
run, exact identities, coverage, and saved evidence.

## Reproduce

From this repository, using Python 3.12+ and Docker:

```bash
uv sync --extra dev
uv run npu-agent env build
uv run npu-agent env capabilities --json
uv run npu-agent env smoke
uv run npu-agent validate-suite tests/fixtures/backends \
  --validation-policy offline-validated --report-dir runs/container-validation
NPU_AGENT_CONTAINER_TESTS=1 uv run pytest tests/test_container_integration.py
uv run python scripts/prepare_validation_corpus.py
uv run npu-agent validate-suite runs/validation-corpus \
  --validation-policy offline-validated --report-dir runs/corpus-validation
```

`prepare_validation_corpus.py` creates audited scalar C++ and OpenVINO graph
baselines for the ten existing static manifests plus sigmoid. This is a
provider-free evaluator integration test, not a baseline-versus-agentic research
experiment or proof of translation from original GPU source. The simulator
executes those actual C++ bodies; it never dispatches to an oracle operation.

A pre-generated bundle can be checked without a provider:

```bash
uv run npu-agent validate-candidate tests/fixtures/backends/amd_xdna2_npu2/add \
  --manifest tests/fixtures/backends/amd_xdna2_npu2/add/manifest.json \
  --target amd_xdna2_npu2 --validation-policy offline-validated
```

Replacing the policy with `target-executed` must return **exit 3**, with a saved
blocked execution stage. Exit 0 means the requested policy passed; 1 means a
validation failure; 2 is invalid CLI/configuration; 3 is blocked or unsupported.
Mixed suites retain every result and fail if any required candidate fails.
Counts always include requested, passed, failed, blocked, and unsupported
candidates, plus input cases executed at each stage.

## Policies and evidence

- `compile-only`: inspected target artifacts suffice; it makes no execution claim.
- `offline-validated`: independent golden/oracle checks, host numerical validation,
  target compilation, and AMD dataflow validation are required.
- `target-executed`: the exact artifacts must execute and pass independent numerical
  comparison through an explicitly configured executor. It never downgrades itself.

New CLI translations, baselines, and experiments default to `offline-validated`.
Python request defaults and historical requests retain `compile-only` for API and
resume compatibility; pass `TranslationRequest(validation_policy=...)` explicitly.
The three version-2 experiment scenarios use the same evaluator and twelve input
cases, with up to ten programming cycles and explicit Terra `xhigh` calls. Only the
structured-IR scenario adds executable IR validation. The separate legacy one-shot
API still makes one generation call and one target compilation attempt.

Stages are independent, not a single fidelity ladder. A CPU pass survives a
subsequent compile failure. Failed arithmetic does not erase a successful compile.
Compatibility booleans/tier fields remain available. SQLite migration 7 adds
versioned evidence to historical compile/evaluation payloads without inventing
oracle, dataflow, or target-execution passes. Old reports and checkpoints remain
readable. Automatic compiler-lesson storage and retrieval are disabled. Experiment attempts
record the resolved image/tool/evaluator/flag fingerprint; source token counts and
artifact size do not earn performance rewards.

Each fresh `invocation-*` directory contains immutable source snapshots, the public
manifest, identical serialized input bytes, source/ABI/model/artifact hashes, full
logs, exact commands, partial/final `validation.json`, actual output arrays, and
compiler intermediates. Expected tensors are computed by the supervisor and never
mounted in candidate construction or execution containers. Intel graph construction
runs in its own container; trusted CPU and NPU compilation processes consume the
same serialized graph. Output ports require explicit tensor names, shapes and dtypes.

## Supported surface

| Area | Supported | Explicitly unsupported |
| --- | --- | --- |
| Inputs | Static, contiguous row-major tensors; 5 normal seeds and 7 boundary families | Arbitrary strides, aliasing, inout, undeclared numerical parameters |
| References | Ten original builtin contracts; scalar multiply, sigmoid, SiLU; independent golden vectors | Arbitrary executable candidate-supplied oracles; prose as executable semantics |
| Primitive IR 2.0 | Constants, arithmetic, exp/sqrt, min/max, compare/select, reshape/cast, structured reductions | Arbitrary index expressions, ordered/nonstandard reductions, side effects |
| AMD C++ | Ordinary pointer/scalar loops; approved portable numerical types; ASan/UBSan, immutable input checks, compiler-checked ABI | Host-only preprocessor branches, unknown includes/intrinsics, nonmatching ABI |
| AMD vector API | Audited load/store, lane addition, same-type reduction | Unaudited accumulator modes and other vector intrinsics |
| AMD topology | Explicit NPU2 tiles, finite workers, acquire windows, broadcast lifetimes, static DMA start/await/free, contiguous split/join links with exact element coverage | Logical/unplaced tiles, infinite workers, custom locks, dynamic DMA/control flow, nontrivial stream layouts/repeat |
| Intel | Built-in serialized OpenVINO graphs, CPU execution, NPU 4000 Compiler-in-Plugin export | Candidate custom libraries, unnamed/ambiguous ports, multiple outputs, CPU fallback labeled as NPU execution |
| Timing | Independently recorded stage durations | Treating CPU or simulator time, source size, or blob size as target speed |

AMD `host_execution` runs the exact source body using the extracted topology under
one schedule. `dataflow_simulation` additionally checks three seeded schedules; all
returned schedules are compared outside the containers. Both depend on the same
bounded topology model. The C++ process persists across changed-input invocations
to detect stale state. A scheduler deadlock includes blocked actors and FIFO needs;
a step/time bound is inconclusive, never evidence of deadlock for all schedules.

The full tensor baselines use one worker; attention and layer norm join two inputs
on a memory tile to stay within compute DMA channel limits. Native `_Float16` is
unsupported by this Peano target. `npu_numeric.h` supplies an identical binary16
storage/conversion type and scalar math for host and Peano, with round-to-nearest,
ties-to-even conversion tests. These implementations establish correctness for the
tested contracts, not optimized target performance. Layer norm calculates its shared
square root before its output loop to avoid a pinned Peano code-generation crash.

Tail manifests are separate static compilations (`T-1`, `T`, `T+1`, `2T+3`); shapes
are never changed after compilation. Unsupported optimized designs cannot satisfy
an offline policy or outrank a validated candidate. See the saved corpus report for
operation, shape, dtype, input identity, and each stage's actual outcome.

## Reproducibility and isolation

`playground/toolchains.lock.json` records real vendor archive/wheel checksums,
resolved source commits, pinned numerical dependencies, and the Ubuntu amd64 base
digest. The selected versions remain MLIR-AIE 1.4.2, its matching Peano requirement,
IRON commit `0d9ebcdd72f5eb85858ddfb0060cf463bedccf02`, and OpenVINO 2026.3.1 archive
build `22476.56d9685302d`. Its runtime reports its own distribution build separately.
This pins the top-level artifacts; apt and transitive dependency resolution are not
a complete snapshot. Image metadata records installed packages and actual tool
hashes/version output. Saved validation reports include the immutable image ID and
full fingerprint inputs, not only a configured version string.

Every compiler/evaluator invocation uses a unique output directory and fresh
`/work/cache`, network disabled, a read-only root, dropped capabilities, arbitrary
non-root UID, and separate read-only source/tools/input mounts. `/work` is a writable
1777 tmpfs. Defaults are four CPUs, 8 GB, 256 processes, and at most two concurrent
compiler containers per orchestrator. Resource options precede the subcommand:

```bash
uv run npu-agent --compiler-cpus 4 --compiler-memory 8g \
  --compiler-timeout-seconds 1200 env smoke
```

Timeouts forcibly remove the named container, preserve earlier evidence and logs,
and never accept an older invocation's artifacts. CI runs CPU tests and two vendor
container jobs. Setting `NPU_AGENT_CONTAINER_TESTS=1` makes unavailable Docker an
error. Local unit-only runs can skip those integration tests; no `requires_npu`
test is claimed to pass in an offline run.

## Optional execution

`TargetProfile.hardware_runner` remains the command configuration point, now using
`ExecutionRequest`/`ExecutionResult` schema 1.0. The request includes target/profile,
manifest/ABI identities, exact artifact hashes, named tensor bytes and hashes, cases,
timeout, and timing scope. Responses must return tensor bytes, artifact hashes,
executor kind, target/runtime identity and compatibility confirmation. The supervisor
checks outputs; a legacy `correct: true` response cannot confer target evidence.
Timing is optional, but provided samples/counts/units/provenance must be consistent
and finite. Only physical execution timing can populate hardware latency; simulator
wall time is never hardware latency.

An optional original-source harness covers CUDA vector add and **recursive complete
reduction** with a user-provided CuPy/CUDA runtime, and Triton softmax/layer norm with
a user-provided PyTorch/Triton CUDA runtime:

```bash
uv run python scripts/validate_source.py examples/classic/02_cuda_reduce_sum/kernel.cu \
  --manifest examples/classic/02_cuda_reduce_sum/manifest.json
```

It is separate from the offline suite and was not required for it. HIP and other
unimplemented source launch contracts report unsupported. A successful source run
checks the original GPU implementation against the independent manifest reference;
it does not execute either NPU artifact.
