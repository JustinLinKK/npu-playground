# Launch contract: feasibility boundary

This is a source audit, not an established device launch protocol. Paths below
refer to the repository snapshot saved in each evidence bundle. No cold/reset or
reusable-device protocol is validated, so neither is admitted for scoring.

| State | Source evidence | Per invocation / persistence conclusion |
| --- | --- | --- |
| Configuration | `playground/tools/amd_compile.py:main` requests xclbin, instruction buffer and intermediates | Emission is established; when configuration is loaded or reset on a device is unknown. |
| Host allocation and binding | `src/npu_agent/sim/amd_dataflow.py:simulate` copies inputs and allocates outputs at entry | New simulator arrays each invocation. Physical allocation identity, address reuse and rebinding are unknown. |
| Address patching | Compile driver requests `--get-input-with-addresses`; retained `target/intermediates/input_with_addresses.mlir` can be inventoried | No version-matched instruction decoder has been validated. MLIR addresses alone do not establish physical host bindings, patch timing, endianness or encoded stride units. |
| Descriptor initialization/start | `tests/fixtures/backends/amd_xdna2_npu2/add/design.py` configures and starts x, y and output tasks | Source-level ordering only. The simulator creates new transfer records each call; binary descriptor state across launches is unknown. |
| Completion/free | Same fixture awaits output then frees input and output tasks; `amd_dataflow.py:simulate` requires tasks done/freed and output coverage | Simulator completion semantics are explicit; equivalence to runtime events is not established. Freeing a compiler task must not be assumed to reset a physical channel. |
| FIFO contents / ownership | `amd_dataflow.py:Fifo` and `simulate` construct new FIFOs and track acquire/release | Slot generations are local to a simulator call, not a device input-generation counter. No persistent-device FIFO observation exists. |
| Locks | `amd_mlir_frontend.py:parse_mlir` accepts a bounded objectFIFO subset; lowered artifacts can contain `aie.lock` / `aie.use_lock` | Actual core lock instructions are not decoded or checked. Core synchronization remains an assumption. |
| Worker program counters | `add/design.py` and `pipeline/design.py` have finite loops ending in `aie.end`; `simulate` starts every worker PC at zero | Repeating host DMA is not established as legal without restarting/reconfiguring finite workers. Device restart behavior is unknown. |
| Reset/rearm | `src/npu_agent/executors.py:execute_target` delegates to an external command and validates returned artifact/output identities | No built-in persistent AMD session or documented device recovery/rearm protocol is provided by this adapter. An alive host object is insufficient evidence of persistence. |
| Next launch | `tests/test_amd_simulation.py:test_two_worker_transfers_and_changed_invocations` calls `simulate` separately for changed payloads | Fresh model state each call. This is useful simulation coverage, not a continuously retained device session. |

Launch number, input-data generation, and allocation identity must be three
separate fields in any future model. Updating data in allocation A does not
constitute rebinding to allocation B. None of those device identities is inferred
from the simulator's slot counter.

The unmodified feasibility fixture computes float32 addition. It is an artifact
availability probe, not T1's asymmetric int32 workload, and cannot count as one of
the eight clean cases. Archived successful validation is historical evidence;
it is not a new compile or a device observation.

Gate A requires a fresh pinned build, trustworthy extraction and a sufficiently
specified launch contract. Until then: selected artifact slice = none;
runtime lifecycle = unknown; checker/model/schedule exploration = not run.
