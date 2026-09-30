# NPU translation progress — speaker notes

## Slide 1 — Progress at a glance

The measured claim is that the evaluated one-shot baseline is unreliable, not that all current LLMs cannot translate NPU code. In the completed historical experiment, baseline completed 14 of 20 kernel-target tasks and agentic completed 19 of 20. AMD completion meant compilation only; Intel completion included historical CPU equivalence and NPU compilation. The newer 22-case suite verifies the upgraded evaluator using deterministic candidates and zero provider calls. It does not demonstrate a new 100% LLM translation rate. No exact NPU binary execution or NPU latency/power measurement is recorded.

Sources:

- E1: runs/20260904T011639Z/{experiment.json, metrics.jsonl, summary.json}
- E2: runs/container-upgrade-validation/{summary.json, corpus-final/report.json, fixtures/report.json}
- E3: docs/validation-results.md and docs/container-simulation-vs-hardware-report.md

## Slide 2 — One-shot baseline

The table is derived from metrics.jsonl and cross-checked against the completed experiment. AMD baseline passes: reduce sum, tiled matmul, tiled transpose, gamma correction, image convolution, fused softmax (6/10). Intel baseline fails image convolution and moving average (8/10 pass). PASS is historical task completion and must not be described as physical NPU correctness. The provider metadata names codex-cli 0.144.5 with model ‘CLI default’; a precise model version is not pinned in the experiment metadata. This limits broader claims about contemporary LLMs.

Sources:

- E1: runs/20260904T011639Z/{experiment.json, metrics.jsonl, summary.json}

## Slide 3 — Agentic results and cost

The native chart uses final case completion, not the count of successful intermediate compile attempts. Agentic completes 9/10 AMD and 10/10 Intel tasks, compared with baseline 6/10 and 8/10. Five paired baseline failures become agentic successes; AMD fused attention remains unsuccessful with IR mismatch {'output': 0.5390625} after three analysis attempts. Agentic uses 127 provider calls and 216 compile attempts versus 20 and 20 for baseline. Search configuration: three rounds, branching factor three, up to two repair retries per candidate. Therefore the observed +25 percentage points is a workflow result at different budgets, not an isolated causal estimate of agent architecture. Intel host equivalence uses the historical evaluator, not the upgraded independent 12-input suite. AMD historical results contain no source or device correctness proof.

Sources:

- E1: runs/20260904T011639Z/{experiment.json, metrics.jsonl, summary.json}

## Slide 4 — Correctness acceptance gates

The upgraded suite contains 11 supported static contracts per target: the ten classic manifests plus sigmoid. Each candidate runs on five normal seeds plus seven boundary input families. Across the targets there are 264 host input cases. AMD executes 132 unique dataflow input cases under three schedules, yielding 396 case/schedule comparisons; those are not 396 independent inputs. All 22 deterministic candidate-target cases pass the offline policy; eight additional positive fixtures also pass. Negative checks exercise arithmetic mutation, DMA bounds, missing FIFO release/deadlock and other boundaries. This supports scoped source/graph and modeled-dataflow correctness, not a proof for arbitrary kernels, original CUDA/HIP/Triton execution, or final device binaries. AMD host and dataflow checks share a topology model. The current suite has zero provider calls and is separate from the historical LLM experiment. Current exact-binary target execution remains blocked on both vendors.

Sources:

- E2: runs/container-upgrade-validation/{summary.json, corpus-final/report.json, fixtures/report.json}
- E3: docs/validation-results.md and docs/container-simulation-vs-hardware-report.md

## Slide 5 — Vendor-specific evidence

AMD: the upgraded path executes supported candidate C++ through host compatibility support, models emitted MLIR topology and bounded FIFO/DMA behavior, and compiles with the pinned MLIR-AIE/Peano toolchain. Unsupported intrinsics or topology are rejected or marked unsupported rather than treated as verified. This is source/dataflow validation, not an AIE2P instruction-set or cycle-accurate binary simulator. Intel: a candidate graph executes through OpenVINO CPU and also undergoes offline NPU compilation. CPU graph correctness is separate from correctness of the compiled NPU blob. Both profiles have no hardware runner, and every target_execution stage is blocked with NO_VALIDATED_TARGET_EXECUTOR; target-executed validation returns exit code 3. A container can host a real device runner when compatible hardware and drivers are passed through, but these current offline paths do not provide that execution. Vendor documentation is supporting context; the counts and executor status come from local reports.

Sources:

- E2: runs/container-upgrade-validation/{summary.json, corpus-final/report.json, fixtures/report.json}
- E3: docs/validation-results.md and docs/container-simulation-vs-hardware-report.md
- P1: https://xilinx.github.io/mlir-aie/1.4.2/skills/aie-dataflow-presim/SKILL/
- P2: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html

## Slide 6 — Performance and power remain open

No NPU latency, throughput, power or energy result is available from the current validation. Missing latency is null, not zero; power and energy were not collected. Functional simulations do not model the device’s actual clock behavior, runtime overhead, memory traffic timing, thermal state or physical power draw with validated accuracy. Hardware measurements should report warmup policy, repeated timings (including p50 and p95), throughput, average power and energy per completed inference. Energy per inference equals measured energy over the measurement window divided by completed inferences. Keep the workload, precision, shape, batch, transfer boundary and measurement boundary explicit. A power rating or CPU runtime is not a substitute for a measured device workload. MLPerf Tiny includes quality, latency and optional energy; MLPerf Edge explains whole-system power measurement. These sources motivate the proposed measurements; our project has no MLPerf submission or performance result.

Sources:

- E2: runs/container-upgrade-validation/{summary.json, corpus-final/report.json, fixtures/report.json}
- E3: docs/validation-results.md and docs/container-simulation-vs-hardware-report.md
- P3: https://mlcommons.org/benchmarks/inference-tiny/
- P4: https://mlcommons.org/benchmarks/inference-edge/

## Slide 7 — Next milestones

These are proposed next steps, not completed work. First rerun the baseline and agentic translation experiment under the new offline acceptance policy, with an explicitly pinned model and a matched-budget comparison in addition to the existing workflow comparison. Second validate a device runner on each target and execute the exact compiled artifacts, verifying outputs before collecting performance. Third benchmark fixed workloads and precision under documented warmup and steady-state conditions, measuring latency, throughput, power and energy. State whether host transfers, loading and compilation are included. A useful result is a set of candidates that preserve device correctness while improving measured latency or energy, with the measurement boundaries disclosed.

Sources:

- E1: runs/20260904T011639Z/{experiment.json, metrics.jsonl, summary.json}
- E2: runs/container-upgrade-validation/{summary.json, corpus-final/report.json, fixtures/report.json}
- E3: docs/validation-results.md and docs/container-simulation-vs-hardware-report.md

## Slide 8 — Evidence and interpretation

Evidence snapshot: 2026-09-09. Historical experiment ID 20260904T011639Z completed at 2026-09-04T06:42:44 UTC. The historical revision is 74f8ae752b2396cfb3f5e44226df93f2d8eb277d. The upgraded evaluator was validated on base d3efa60a08e5807c54b40832f3dcb2549f44ea00 plus uncommitted implementation changes. Image identities and compiler fingerprints, original per-case metrics, aggregate counts and SHA-256 hashes of source reports are preserved in the colocated evidence.json. Do not compare 19/20 and 22/22 as a before/after translation accuracy experiment: the candidates, evaluation criteria and use of LLMs differ. The presentation wording intentionally limits the first claim to the measured baseline and limits correctness to the representations actually checked. Full test run before the final alias guard: 98 passed including 23 container tests. Final CPU run: 76 passed, 23 deselected; final boundary checks: 47 passed. The corpus and fixtures were rerun after the guard. Those software checks do not provide hardware performance evidence.

Sources:

- E1: runs/20260904T011639Z/{experiment.json, metrics.jsonl, summary.json}
- E2: runs/container-upgrade-validation/{summary.json, corpus-final/report.json, fixtures/report.json}
- E3: docs/validation-results.md and docs/container-simulation-vs-hardware-report.md
- P1: AMD MLIR-AIE dataflow pre-simulation — https://xilinx.github.io/mlir-aie/1.4.2/skills/aie-dataflow-presim/SKILL/
- P2: Intel OpenVINO NPU device documentation — https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- P3: MLPerf Tiny: quality, latency and optional energy — https://mlcommons.org/benchmarks/inference-tiny/
- P4: MLPerf Inference Edge: power measurement — https://mlcommons.org/benchmarks/inference-edge/
