# Container upgrade validation results

The hardware-free milestone is implemented and verified. Actual execution of the
emitted AMD XDNA2 and Intel NPU binaries remains **blocked** for both targets.
Neither host execution nor target compilation establishes target-binary correctness.

Validation used repository revision `d3efa60a08e5807c54b40832f3dcb2549f44ea00` plus
this uncommitted implementation. No provider calls or original GPU-source runs were
made. The desktop needed no hardware, firmware, or host NPU driver changes; Docker
Desktop was started and both images were rebuilt locally.

## Changes by subsystem

- `models.py`, `validation.py`, `executors.py`, `database.py`: versioned independent
  stage evidence, policies, strict tensor/execution protocol, blocked execution and
  conservative legacy migration.
- `oracles.py`, `testcases.py`, `ir.py`: independent golden references, twelve input
  cases, strict numerical/layout contracts and primitive IR composition.
- `compilers.py`, `config.py`, Dockerfiles, `playground/tools/`, toolchain lock:
  immutable image identities, trusted compilation, isolated Intel construction and
  execution, resource limits, fresh artifacts, timeout cleanup and partial reports.
- `sim/` and `host_compat/`: exact supported AMD C++ execution with ABI/sanitizer
  checks and emitted-MLIR FIFO/DMA modeling under multiple schedules.
- Workflow, baseline, MCTS, CLI, experiment and reporting modules: common acceptance
  policies, honest ranking, scoped compiler lessons, coverage and blocked outcomes.
- Backend fixtures, unit/container tests, corpus/source scripts, CI and documentation:
  reproducible validation independent of model generation.

## Verified outcomes

| Evidence | AMD XDNA2/NPU2 | Intel NPU 4000 |
| --- | --- | --- |
| Independent math | 11/11 contracts, 132 input cases | 11/11 contracts, 132 input cases |
| Generated implementation on host | 11/11, 132 cases | 11/11, 132 cases through serialized OpenVINO graphs |
| Source/dataflow simulation | 11/11, 132 cases across 3 schedules each | Not requested |
| Actual target compilation | 11/11 inspected xclbin/instruction outputs | 11/11 exported NPU blobs |
| Actual target binary execution | 0 executed; 11 blocked | 0 executed; 11 blocked |
| Hardware latency | Unknown (`null`) | Unknown (`null`) |

The corpus contains all ten existing static contracts plus sigmoid. All 22
candidates passed `offline-validated`; zero were dropped, failed or unsupported at
that policy. These are deterministic scalar C++ and OpenVINO baselines generated
from the public manifests, not a provider-driven translation experiment or original
CUDA/HIP/Triton execution. AMD dataflow comparisons cover 396 case/schedule pairs.

All eight checked-in positive backend fixtures passed, including streaming add,
scalar multiply, a two-worker pipeline, transpose, and a four-tile reduction with
an acquire/release window of four objects. Separate tail tests compile shapes
`15`, `16`, `17`, and `35` for each target.

The full regression run passed **98 tests**, including **23 real container tests**.
After the final aliasing guard, the CPU suite passed **76 tests** (23 container tests
deselected), 47 focused boundary checks passed, and both the complete fixture and
22-candidate corpus suites passed again with the final fingerprints below.

Negative checks cover arithmetic, DMA bounds, missing release/deadlock, memory
bounds and immutable inputs, target mismatch, invalid platform, source syntax,
unaudited intrinsic rejection, stale artifacts, timeout cleanup, verdict spoofing,
malformed results, executor provenance/timing, ranking and migration. A CPU-valid
Intel `Unique` plus inverse-index `Gather` graph was specifically rejected by this
pinned NPU compiler with `String attribute index_element_type is not supported`;
its CPU pass survived in the stage report.

Both `target-executed` CLI requests returned **exit 3**, retaining their offline
passes and unknown target correctness. Capability probes and production-isolation
smokes passed for both images. `compileall`, `uv lock --check`, `uv build`, and
`git diff --check` also passed. CI was added but has not been run remotely.

## Exact identities

| Target | Local immutable image ID | Evaluator/toolchain fingerprint |
| --- | --- | --- |
| AMD | `sha256:fd31211a92209c950c993f647351ae657eb3ab8be1158b92f5da849a1d41a026` | `83ea5a2cc99cb715b16f56673294b87da90aec44b18e53371a8a55a51134f704` |
| Intel | `sha256:d1278b0741322f84979488ddee1c8524bbb7fec0cb446951a75df4a22f69a2cc` | `b439eb1b6a589bd0524f47426350f1a6fcb52dc1f6d05b75c825d71a73ec6e4d` |

AMD uses MLIR-AIE 1.4.2 at `760932a4abf084bfc7abdec851703b27a03c01ba`,
IRON `0d9ebcdd72f5eb85858ddfb0060cf463bedccf02`, and Peano
`21.0.0.2026080301+c9c5ecb7` (compiler commit
`c9c5ecb725fc8c765e4b687356e6ec1e54da7a0e`). Host GCC is 13.3.0.
Intel uses archive `2026.3.1.22476.56d9685302d`; the installed runtime reports
`2026.3.1-22476-759c5a6ab8c-releases/2026/3`.

Archive/wheel/base-image hashes are in
[`toolchains.lock.json`](../playground/toolchains.lock.json). The saved identities
include actual binary/library hashes, installed packages, evaluator/header hashes,
profiles and flags. The supervisor used Python 3.13.12, NumPy 2.5.2 and Pydantic
2.13.5; the containers use Python 3.12 and Pydantic 2.12.5. Top-level vendor artifacts
are pinned; apt/transitive package resolution is not a complete snapshot, so fresh
builds may have different image IDs. Images have not been published.

## Saved evidence and reproduction

- [Final summary and identities](../runs/container-upgrade-validation/summary.json)
- [Final corpus report](../runs/container-upgrade-validation/corpus-final/report.json)
- [Fixture report](../runs/container-upgrade-validation/fixtures/report.json)
- [Full test log](../runs/container-upgrade-validation/pytest.log) and
  [final CPU test log](../runs/container-upgrade-validation/cpu-tests.log)
- [Target-execution exit-code checks](../runs/container-upgrade-validation/target-execution-exit-codes.json)

These local reports are under ignored `runs/`; each invocation also retains its
source snapshot, contract, input/output bytes, actual commands, complete logs,
compiler intermediates and artifact identities. Reproduce from the repository:

```bash
uv sync --frozen --extra dev
uv run npu-agent env build
uv run npu-agent env capabilities --json
uv run npu-agent env smoke
NPU_AGENT_CONTAINER_TESTS=1 uv run pytest
uv run npu-agent validate-suite tests/fixtures/backends \
  --validation-policy offline-validated --report-dir runs/container-validation
uv run python scripts/prepare_validation_corpus.py
uv run npu-agent validate-suite runs/validation-corpus \
  --validation-policy offline-validated --report-dir runs/corpus-validation
```

The [validation guide](validation.md) gives the complete supported surface and
executor contract. Unsupported MLIR/intrinsics, aliasing/inout, and unimplemented
source launch contracts stay explicit. Connecting and validating a real target
executor, adding more target profiles, GPU-source runs, and provider experiments
remain optional follow-on work. The original all-three milestone stays blocked
until both exact emitted target binaries execute and pass independent comparison.
