# NPU Transport Correctness Early Case Study Plan

Prepared 30 September 2026. Repository: [JustinLinKK/npu-playground](https://github.com/JustinLinKK/npu-playground). Inspected baseline: `c1d246f4c09606d9b0ad5388ddb98c3d2a60bee7`.

## Start here

Build a small, independent checker that asks: **Will the compiled NPU program move the right data, wait for the right events, and remain correct when launched again under its documented runtime contract?**

This is a feasibility study, not an instruction to build a complete verifier or an agent framework. First establish whether this catches a meaningful class of mistakes that the repository simulator and existing compiler checks miss. Novelty is a hypothesis, not an established fact.

Timebox: 10 working days, approximately 40 to 60 engineering hours. Default mode is offline, with no paid LLM calls and no device execution. Stop at the decision gates below if the evidence does not justify more implementation.

### Instruction to the coding agent

> Work in the supplied npu-playground checkout. Read this entire plan and the applicable AGENTS.md instructions. Inspect the actual checkout and report differences from the pinned baseline before changing anything. Preserve unrelated changes and historical experiment outputs. Implement only the bounded early study below, starting with the feasibility gate. Do not push, open a PR, use paid APIs, upgrade the frozen toolchain, or run mutated artifacts on hardware without explicit authorization. Return a reproducible evidence bundle and an honest CONTINUE, NARROW, or STOP recommendation. A negative result is a valid outcome.

## 1 Research question and limits

The proposed contribution is independent checking of the relationship between a tensor-level transport contract and the compiled transport/control artifacts, including the state left for subsequent launches.

Test three questions:

1. **Artifact gap:** Can an error in emitted transport/control instructions escape checks of the source or emitted MLIR but be detected independently at the artifact boundary?
2. **Lifecycle gap:** Are there meaningful failures that require reasoning about state across launches, beyond a straightforward replay or reset check?
3. **Incremental value:** Does the proposed analysis detect those failures beyond upstream checks, the existing simulator, and cheap deterministic checks?

Do not assume any of these gaps exists. A hand-injected binary mutation demonstrates sensitivity to that mutation; it does not demonstrate that the compiler naturally emits the defect.

### In scope

- AMD XDNA2 / NPU2, static shapes, finite control, bounded integer kernels.
- Declared host buffers, tensor bindings, DMA descriptors, transfer extents and layouts.
- Ordering, completion, descriptor lifetime, and the synchronization state actually represented in the inspected artifacts.
- Cold launches and documented reusable-session launches.
- Short, replayable counterexample traces.

### Out of scope

- Full AIE instruction emulation, numerical algorithm verification, performance optimization, arbitrary dynamic shapes, full driver correctness, and universal deadlock freedom.
- Intel target support, a large kernel benchmark, a new LLM orchestration framework, or repairing all earlier experiments.
- Claims that no similar work exists or that this is already a publishable contribution.

The integer kernel body is trusted unless separately checked. If device-core synchronization is not decoded from the artifact, treat it as an explicit assumption, not something verified by the checker.

## 2 Repository starting point

Read these files first. Paths refer to the inspected baseline; locate their current equivalents if the checkout differs.

| Area | Files | Why it matters |
| --- | --- | --- |
| Research context | `docs/cloud-analysis-handoff.md`, `docs/npu-ir-research-next-steps.md`, `docs/npu-ir-closest-work.md` | Preserve existing conclusions and unresolved issues |
| Recent results | `runs/analysis/research-next-steps-v1/pilot-retry1/PHASE1_REPORT.md`, `combined-data.json` in the same directory | Do not assume structured IR has already beaten direct generation |
| Simulator | `src/npu_agent/sim/amd_mlir_frontend.py`, `amd_dataflow.py`, `amd_host.py` | Establish exactly what is already checked |
| Compile pipeline | `playground/tools/amd_compile.py`, `playground/toolchains.lock.json` | Locate real emitted artifacts and pinned versions |
| Execution boundary | `src/npu_agent/executors.py`, `src/npu_agent/validation.py` | Retain existing correctness checks and distinguish unavailable execution |
| Fixtures and tests | `tests/fixtures/backends/amd_xdna2_npu2/`, `tests/test_amd_simulation.py`, `tests/test_executors.py` | Reuse infrastructure without changing historical fixtures |

Important observations to confirm locally:

- The current AMD simulator uses emitted design MLIR and host-compiled C++ kernel bodies. It does not execute the emitted NPU binaries.
- The dataflow simulator already checks many transfer, ownership, and completion conditions. Rediscovering these is not incremental value.
- A fresh simulator invocation creates fresh state. The existing changed-input invocation test is not evidence of a continuously retained device session.
- The compile path requests `insts.bin`, `final.xclbin`, input-with-addresses information, and intermediates. Check which are actually retained and readable.
- Some fixture workers are finite. Repeating host DMA without legitimately restarting or reconfiguring those workers is not necessarily a valid runtime use.
- The recent small DeepSeek pilot did not establish an IR advantage. The earlier layer-normalization tolerance issue is separate: do not loosen that reference to make this study pass.

Record the repository SHA, dirty-file list, image digest, actual compiler versions, toolchain lock hash, and artifact hashes. A container tag alone is insufficient provenance.

## 3 Day 1 feasibility gate

Spend no more than one working day answering these questions before building the analysis.

### A Obtain one real artifact bundle

Compile an unmodified legal fixture with the existing pinned toolchain. Save the exact source, emitted MLIR, core object or ELF when available, instruction buffer, xclbin, compile commands, and logs. Hash each file.

If the toolchain is unavailable, report the blocker. You may implement unit tests against explicitly labeled parser fixtures, but these do not satisfy the real-artifact gate.

### B Map the launch contract

Produce `launch_contract.md` with a table of each state component and its lifecycle:

- one-time configuration versus per-launch transactions;
- buffer allocation, binding, address patching, and reuse;
- descriptor initialization, start, completion, and permitted reconfiguration;
- locks, FIFO contents, worker program counters, and core restart;
- documented reset or rearm steps;
- what persists and what resets between launches.

Support each entry with a source location, generated artifact field, or runtime observation. Mark unknowns. Do not infer persistence merely because a host object remains alive.

Keep these concepts distinct: **launch number**, **input-data generation**, and **buffer allocation identity**. Reusing one buffer address with new data is not the same as rebinding a descriptor to a new buffer.

### C Establish a readable artifact boundary

Use official format definitions or a version-matched decoder for syntax. Identify relevant operations and address-patching information. Record decoder versions and dependencies.

Possible outcomes:

- **Full selected-transport slice:** relevant host instructions, configuration, bindings, and core synchronization effects are represented.
- **Host or shim subset:** only instruction-buffer effects are represented; core/configuration effects are assumptions.
- **No trustworthy extraction:** unsupported format or missing state prevents meaningful checking.

Using an upstream disassembler is acceptable, but shared code is a common-mode risk. An encode/decode round trip tests consistency, not semantic correctness. Do not construct the expected contract by copying the decoded program.

### Gate A

Proceed with the artifact study only if at least one real compiled fixture is readable and the launch contract is sufficiently specified. Otherwise deliver a narrow feasibility report and stop artifact-level claims.

A host/shim-only study may proceed under that label. It must not be described as verification of the complete NPU binary.

## 4 Minimal checker design

Separate the expected contract from the implementation being checked.

### Inputs

1. A hand-specified tensor contract: input/output identities, shapes, integer types, index mapping, and allowed launch/reset behavior.
2. A hashed compiled-artifact bundle.
3. A launch script specifying allocation, binding, input generation, launch, completion, and allowed reset/rearm actions.
4. Explicit assumptions for effects outside the decoded slice.

### State and obligations

Model only state needed by the selected slice: buffer identities and generations, descriptor lifecycle, in-flight transfers, completion events, locks or FIFO state where available, and relevant actor positions.

Check:

- Every transfer address is in its declared allocation.
- Extents, strides, and element units match the version-specific descriptor semantics.
- Each consumer receives the intended element or region from the intended input generation.
- Output coverage and ordering match the tensor contract.
- A descriptor or buffer is not illegally changed or reused while active.
- A completion event does not claim that an output is ready before its required writes.
- The actual documented next-launch preconditions hold after completion and permitted cleanup.

A useful specification shape is:

```text
Assuming Inv(state) and valid input bindings:
    execute the selected artifact under the documented launch protocol
    check TransportMatches(contract, input_generation)
    check Inv(next_state) after the protocol's permitted cleanup
```

This is a goal, not a proof already achieved. Bounded exploration establishes only what the explored model covers. Termination and fairness require separate arguments.

Use small exact integer values and a separate CPU oracle for end-to-end expected outputs. For transport identity, track symbolic provenance such as tensor, generation, and logical index. Keep arithmetic and transport claims separate.

### Exploration

Start with deterministic event replay. Add bounded exploration only where different enabled event orders can affect an obligation.

For the tiny fixtures, allow at most 50,000 explored states or 30 seconds per case initially. Record the actual bounds. A state bound, unknown operation, or ambiguous runtime behavior yields `INCONCLUSIVE` or `UNSUPPORTED`, never `PASS`.

A deadlock report needs a reachable blocked state and a stated environment/fairness model. A timeout alone is not proof of deadlock.

Return one of:

```text
PASS_BOUNDED | FAIL_WITH_WITNESS | INCONCLUSIVE | UNSUPPORTED | NOT_RUN
```

Each failure must include the first violated obligation and a short replay trace tied to artifact offsets or decoded operation IDs.

## 5 Frozen experiment matrix

Build the smallest legal versions first. Suggested sizes below are starting points, not assertions of backend legality. Adjust for documented alignment or allocation constraints during preflight, explain the change, then freeze the manifest before scoring results.

### Eight clean workloads

Use four topologies and two legal sizes per topology.

| ID | Computation | Initial sizes | Main transport stress |
| --- | --- | --- | --- |
| T1 | Two inputs, `out = x + 2*z` | 64 and 128 elements, tile 16 | Distinct input bindings and output completion |
| T2 | Out-of-place integer transpose | 8 by 8 and 8 by 16 | Stride, index mapping, and complete output coverage |
| T3 | Two-worker pipeline: T1 followed by multiplication by 2 | 64 and 128 elements, tile 16 | Intermediate ownership and worker ordering |
| T4 | Split input into two branches, then join in a declared order | 64 and 128 elements, even split | Region identity, overlap, holes, and branch completion |

Use int32 and input magnitudes small enough that all calculations stay within int32. The asymmetric T1 prevents a swapped-input bug from being hidden by commutative addition.

If T4 cannot be supported within two working days, mark it unsupported and reduce scope; do not invent a successful case. Do not introduce padding or remainder semantics silently.

For every supported workload:

- Test a cold/reset session and a documented reusable session, if the latter exists.
- Offline launch lengths: 1, 2, and 8.
- Use three fixed schedule seeds for existing randomized simulation. These are repetitions, not independent workloads.
- Change the payload every launch, including index-distinguishing patterns, alternating signs, and a fixed-seed random pattern.
- Separately test same-allocation/new-data and, where legally supported, new-allocation/rebinding.
- Never let all-zero, constant, or identical consecutive inputs be the only test.

Run the existing simulator fairly. Do not call it defective merely because it was never designed to retain device state.

### Sixteen planned mutant cases

Create two applicable instances of each of these eight defect classes, preferably on different topologies or sizes. Keep the manifest's full planned denominator, including unsupported instances.

| Class | Controlled defect | Important qualification |
| --- | --- | --- |
| M1 | Wrong but in-bounds stride or offset | Distinguish semantic corruption from ordinary bounds checking |
| M2 | Wrong input binding | Use an asymmetric computation |
| M3 | Split/join region overlap or hole | Existing simulator may already catch this |
| M4 | Missing required start, wait, or release | Positive control for existing checks, not presumed novelty |
| M5 | Descriptor rewritten or reused before legal completion | Model actual lifecycle semantics |
| M6 | Stale binding/address patch on a later launch | Requires a genuinely supported rebinding protocol |
| M7 | Missing required rearm/restart action | Only a defect relative to the documented launch contract |
| M8 | Completion exposed before all required writes | Require a concrete ordering witness |

Assign the injection layer in advance: source/design mutation or structured post-lowering artifact mutation. Include both layers overall. Keep each mutant paired with its clean parent and modify one cause at a time.

Artifact mutants must remain syntactically valid and decodable. Preserve unrelated fields and show the decoded before/after change. Random byte corruption and parse failures do not demonstrate semantic verification.

Do not require an impossible mutation just to fill the matrix. Mark it `NOT_APPLICABLE` with a reason and keep it visible. Do not change mutation labels after seeing the checker result.

### Held-out check

After freezing the checker and mutation templates, run one additional legal size and one new payload seed per supported topology. Report these separately. If you change the implementation after seeing a held-out failure, label the subsequent run as development, not untouched holdout evidence.

## 6 Baselines and ablations

Use the same expected contract and applicable launch protocol wherever an adapter permits. Record missing observability instead of converting it into a failure.

| Baseline | What to run | What it tests |
| --- | --- | --- |
| B0 | Existing compile/validation path and applicable upstream checks | Whether the error is already rejected |
| B1 | Current emitted-MLIR simulator | Whether this is already a simulator-detectable problem |
| B2 | Cheap deterministic checks: bounds, extents, bindings, event replay, and lifecycle bookkeeping | Whether a simple engineering patch explains the whole gain |
| B3 | PEQC-MLIR on compatible source-level cases | Overlap with existing equivalence and DMA/lock reasoning |
| P1 | Proposed artifact checker, one launch | Added value from the artifact boundary |
| P2 | Same checker, multi-launch state | Added value from lifecycle reasoning |
| P3 | Same checker with bounded schedule exploration | Added value beyond deterministic replay |

Give B2 retained state under the same documented reset rules. Comparing a persistent checker only against a deliberately fresh-state baseline would be unfair.

For upstream comparison, inspect the version-matched checks and also current upstream documentation. In particular, inspect `aie-verify-runtime-rearm` and instruction-buffer validation tooling. A missing check in the frozen version may already be fixed upstream.

Timebox optional PEQC installation/integration to four hours. If unavailable or incompatible, report the exact attempted version and reason as `NOT_RUN` or `UNSUPPORTED`. Do not claim to beat a tool you did not run. Read its assumptions before deciding a repeated-launch case is outside its scope; finite launches may be expressible by unrolling.

An artifact-level finding counts as incremental only if it survives comparison with all applicable, actually run controls. Missing a baseline leaves the corresponding comparison unresolved.

### Optional strong-model challenge

Do not run this during the default zero-cost study. If requested later with an approved budget:

- Give the same strong coding model, documentation, artifacts, execution access, and repair tasks to both arms.
- Arm A may use existing tools and create helpers; Arm B additionally receives checker witnesses.
- Compare a fixed token budget and a larger, predeclared budget for Arm A. Record all setup, retry, and repair tokens.
- Evaluate on held-out cases and report correctness, not just compilation.
- Keep the oracle hidden from both arms and distinguish the cost of writing the checker from the cost of using it.

Beating one model at one budget is not the novelty claim. The defensible question is whether the checker supplies an independently testable guarantee or witness that remains useful regardless of who wrote the program. More tokens are a serious baseline, not something to dismiss.

## 7 Implementation boundaries

Prefer a small isolated package, for example:

```text
research/transport_case_study/
    README.md
    manifest.json
    contracts/
    fixtures/
    artifacts.py
    decode.py
    model.py
    replay.py
    explore.py
    mutations.py
    baselines.py
    report.py
    cli.py
tests/test_transport_case_study.py
```

These are proposed new paths, not existing interfaces. Reuse current compile and fixture helpers where practical. Add only minimal artifact-export hooks if needed. Do not broadly refactor the agent, compiler wrappers, or simulator.

Provide a CLI with the following capabilities; choose final command spelling and document it:

```text
preflight     identify tools, versions, supported artifact slice, and launch contract
build         compile clean parents and generate labeled mutants
evaluate      run explicitly selected offline baselines and checker variants
report        regenerate tables and conclusions from immutable raw records
```

Device execution must require a separate explicit opt-in flag and a configured supported executor. No endpoint discovery, credential extraction, automatic billing, or automatic device resets.

Required tests:

- unknown/truncated operations and invalid descriptor fields fail closed;
- version, endianness, stride units, and address patch interpretation are tested against real format evidence;
- independent expected contracts are not overwritten from decoded artifacts;
- clean valid parents are not rejected for harmless scheduling differences;
- distinct launch generations cannot silently alias in the model;
- descriptor and completion state follows the documented lifecycle;
- witness replay reproduces each claimed failure;
- timeout, unavailable tool, and unsupported operation remain distinct from detected bugs;
- report totals reconcile with the frozen manifest, including skipped cases.

A toy JSON representation is useful internally, but is not a replacement for demonstrating extraction from real emitted artifacts.

## 8 Hardware validation and safety

Hardware is an optional second stage, not a prerequisite for parser and model work.

Before running, confirm explicit authorization for an isolated device, a validated executor, and a supported recovery procedure. Retain exact artifact hashes and allocation/binding logs.

For clean supported workloads, compare cold execution with a reusable session of 1, 2, and 32 launches, changing inputs every time. If only 8 launches are feasible, record the deviation. The adapter must preserve the actual intended device/session state; separate fresh process runs are not automatically equivalent.

Start with clean artifacts. Do not execute raw post-lowering mutants on hardware by default. A mutant needs separate approval and a safety review of addresses, transfer bounds, device state, and recovery behavior. Offline failures are sufficient for the initial mutation study.

A host timeout or killed process does not establish that the device is idle or reset. Stop after a hang or unclear device state; do not automatically reboot, invoke privileged resets, or keep launching work.

Use exact integer outputs. Record whether failure is wrong output, stale generation, executor error, timeout, or inability to validate device state. Never relabel a timeout as proven deadlock.

If hardware is unavailable, say so prominently. The study can support bounded-model and artifact-extraction conclusions only, not claims of observed silicon bugs.

## 9 Evidence and reporting

Write new outputs under a unique run directory, for example:

```text
runs/analysis/transport-case-study/<run-id>/
    REPORT.md
    environment.json
    manifest.frozen.json
    launch_contract.md
    support_matrix.csv
    results.jsonl
    metrics.csv
    artifact_manifest.json
    witnesses/
    logs/
    reproductions/
```

Do not overwrite earlier studies. Store large binary artifacts according to the repository policy, with hashes and reproducible retrieval/build instructions; do not commit them automatically.

Each result must include:

```text
case_id, clean_parent_id, topology, shape, mutation_class, injection_layer
repo_sha, toolchain_identity, artifact_hashes, decoder_version
launch_protocol, launch_count, input_seed, schedule_seed
baseline_or_checker, supported_slice, assumptions
status, violated_obligation, witness_path
elapsed_seconds, explored_states, exploration_complete_within_bound
hardware_executed, raw_log_path
```

Report:

- detection by canonical defect class, with raw case counts;
- additional witnessed findings beyond each applicable baseline;
- clean-case false rejections, unsupported cases, and inconclusive cases;
- one-shot versus repeated-launch differences;
- deterministic replay versus exploration differences;
- extraction/modeling effort and checking cost;
- real-program findings versus deliberately injected mutants;
- artifact-level mismatches versus source-level errors already visible before compilation.

Do not count 32 failing launches, three seeds, or several duplicate mutants as 32 or three independent discoveries. Do not report throughput speedup from simulator steps.

For every claimed new finding, produce a one-command reproduction plus:

1. Clean intended contract.
2. Exact artifact and launch protocol.
3. Earliest violating event.
4. Why applicable baselines miss it.
5. Whether the original program, runtime usage, compiler output, or checker model is at fault.
6. Minimal repair and a clean rerun.
7. Limits and assumptions.

Have a human inspect the most important witness before treating it as a research result. The checker is also software and can be wrong.

## 10 Schedule and decision gates

| Time | Deliverable | Decision |
| --- | --- | --- |
| Day 1 | One real artifact, launch contract, support boundary | Stop or narrow if the boundary is unreadable |
| Days 2 to 3 | Two clean topologies, independent contract, deterministic replay, B0 to B2 | Stop if useful analysis requires a full ISA emulator |
| Days 4 to 5 | Frozen clean/mutant matrix and initial witnesses | Check whether existing tools explain all wins |
| Days 6 to 7 | Multi-launch model and bounded exploration where justified | Continue only if they add measurable value |
| Day 8 | Held-out tests and optional approved clean hardware runs | Separate model claims from hardware observations |
| Days 9 to 10 | Reproduction bundle, baseline comparison, prior-art update, verdict | Decide whether to invest in a paper-scale study |

Default limits: no paid calls; at most two concurrent compiles; initially 180 seconds per tiny-fixture compile and 30 seconds or 50,000 states per check. Set practical CPU/memory limits for the available machine during preflight. One documented adjustment before freezing is acceptable; unexplained per-case tuning is not.

### CONTINUE to a larger research study

All of the following should hold:

- Real artifact extraction works for at least T1 and T3, with a precisely declared coverage boundary.
- There are no confirmed false rejections among supported clean cases.
- At least two distinct, replayable failure mechanisms add value beyond applicable controls, including one genuinely temporal mechanism if repeated-launch reasoning is the proposed contribution.
- The gain is not just missing bounds checks, a forgotten documented reset, or a fresh-versus-persistent baseline mismatch.
- At least one finding comes from an unmodified real program or a documented historical failure, with corroboration appropriate to the claim. It is not solely an invented mutant.
- The closest related tools do not already provide the same capability under equivalent assumptions.

These are investment gates, not publication guarantees. A small study cannot establish absence of prior work or broad statistical performance.

### NARROW and validate further

Choose this if the checker works but only synthetic mutations show value, relevant baselines remain unavailable, artifact coverage is only a small host/shim slice, or hardware evidence is still needed for the central claim.

State the single next experiment that would resolve the uncertainty. Do not scale up kernel counts just to make the result look larger.

### STOP or reposition as an engineering contribution

Choose this if simple deterministic checks catch everything, existing upstream tooling already covers the capability, the proposed lifecycle bugs disappear under the real reset protocol, or the extraction cost is disproportionate to the demonstrated benefit.

Improved regression tests may still be worth merging later. That is different from a strong novel-paper direction.

## 11 Related work check

Recheck these sources before making novelty claims. Record exact tool versions and access dates. This list bounds an initial search; it is not proof of novelty.

- [PEQC-MLIR paper](https://arxiv.org/abs/2605.01124) and [implementation](https://github.com/axolotls73/PEQC-MLIR): closest formal-equivalence comparison; inspect supported AIE DMA/lock semantics and static-control assumptions.
- [MLIR-AIE AIEX passes](https://xilinx.github.io/mlir-aie/dev/AIEXPasses/): inspect the experimental runtime-rearm check and its documented limitations.
- [AMD aie-codegen tools](https://github.com/Xilinx/aie-codegen/blob/main/tools/README.md) and [driver examples](https://github.com/Xilinx/aie-codegen/blob/main/driver/examples/README.md): instruction/control-code validation and transaction tooling. Public visibility may not cover every internal validation suite.
- [LLMLift](https://arxiv.org/abs/2406.03003) and [Tenspiler](https://arxiv.org/abs/2404.18249): verified lifting/translation is not a new category by itself.
- [AscendCraft](https://arxiv.org/abs/2601.22760) and [QiMeng-Xpiler](https://arxiv.org/abs/2505.02146): constrained generation and neural-symbolic compiler repair overlap with generic framework proposals.
- [NPUEval](https://arxiv.org/abs/2507.14403): relevant hardware-grounded NPU generation evaluation.

Search specifically for compiled accelerator control-program verification, DMA descriptor lifecycle checking, repeated invocation/reset invariants, translation validation for AIE, and asynchronous dataflow equivalence. For each close match, compare artifact boundary, supported state, proof assumptions, lifecycle scope, and counterexample generation.

Do not claim that PEQC cannot handle repeated launches simply because it operates at MLIR level. Do not claim an upstream check is absent because it was not enabled in the frozen repository toolchain.

## 12 Required final response from the coding agent

Return this short summary first, followed by links to the evidence:

```text
Verdict: CONTINUE / NARROW / STOP

Implemented:
Exact artifact and lifecycle coverage:
Clean cases: passed / failed / inconclusive / unsupported
Mutant cases: detected / missed / inconclusive / not applicable
Real-program findings:
Incremental findings beyond existing controls:
Hardware actually executed:
Most important counterexample:
Closest prior-art overlap:
Main reason this may not deserve a paper:
One next experiment:
Reproduction command:
```

The success condition is not “we built a checker.” It is “we learned, reproducibly, whether independent artifact and lifecycle reasoning provides a research-worthy capability beyond existing checks and a strong coding model with more effort.”

