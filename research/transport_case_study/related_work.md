# Feasibility source audit

Accessed 2026-09-30. These are source inspections, not executed baselines. The
offline CLI does not download or install tools. Recheck moving documentation
before resuming the study; absence of an executed comparison leaves it unresolved.

| Source / version | Observed overlap and limits | Study status |
| --- | --- | --- |
| [MLIR-AIE rearm implementation at the locked commit](https://github.com/Xilinx/mlir-aie/blob/760932a4abf084bfc7abdec851703b27a03c01ba/lib/Dialect/AIEX/Transforms/AIEVerifyRuntimeRearm.cpp), lines 8–26; matching `include/aie/Dialect/AIEX/Transforms/AIEXPasses.td`, lines 423–426 | The frozen source already defines an opt-in runtime-rearm check. It checks missing lock rearming after channel reset, with documented gaps including queue restart. Presence in source does not establish its inclusion or execution in the installed wheel. | B0 unresolved until the pinned executable and actual pipeline are inspected. |
| [Current AIEX passes](https://xilinx.github.io/mlir-aie/dev/AIEXPasses/) | Documents runtime rearm and optional unsafe descriptor reuse diagnostics; also describes queue-depth handling. These overlap the proposed lifecycle checks. | Source inspection only; no claim that a new checker improves them. |
| [aie-codegen tools](https://github.com/Xilinx/aie-codegen/blob/efed7deb33147e6c1df710b64aba1065aafa1194/tools/README.md) and [driver examples](https://github.com/Xilinx/aie-codegen/blob/efed7deb33147e6c1df710b64aba1065aafa1194/driver/examples/README.md) | Assembly validator checks referenced payload labels and lengths. Driver examples include transaction reserialization and an instruction-buffer API test. Their documented scope is not a general tensor-transport equivalence proof. | Not run; neither a round trip nor a tool name establishes semantic validation. |
| [PEQC-MLIR README](https://github.com/axolotls73/PEQC-MLIR/blob/9309f01b4583f074af7b66b4240f54fa4ca73623/README.md), supported operations; [paper v1](https://arxiv.org/abs/2605.01124v1) | MLIR equivalence under statically interpretable control flow; supports AIE DMA and locks with stated restrictions. Finite repeated launches may be expressible by unrolling; this was not tested. | B3 NOT_RUN. No installation attempted after Gate A stopped; inspected revision is not an executed version. Current instructions require LLVM 23 and associated dependencies. |
| [AccelSync v1](https://arxiv.org/abs/2605.07881v1), abstract | Additional close lead from the search: checks ordering of cross-unit memory accesses under accelerator-specific event semantics. The paper describes Ascend/Cambricon instances. | Abstract inspected only; binary extraction, lifecycle coverage and witness comparison remain unresolved. |

The search covered accelerator control-program verification, DMA descriptor
lifecycle, repeated invocation/reset invariants, AIE translation validation, and
asynchronous dataflow equivalence. This bounded review does not establish novelty.
The existing repository simulator already checks DMA bounds, transfer lengths,
FIFO ownership/release, completion and output coverage. Any future artifact gain
must be compared with those checks and a persistent deterministic B2 baseline.

Recommendation at this gate: STOP artifact-level implementation until a fresh
pinned build and an evidenced launch protocol are available. This is an
infrastructure/coverage decision, not a negative experimental result about the
research hypothesis. No mutant or naturally occurring compiler bug was detected.
