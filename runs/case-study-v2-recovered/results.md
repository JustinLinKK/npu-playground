# Case study v2

Stage: controlled. Status: completed.

Offline validation only; no physical NPU correctness or performance claim.

| Target | Arm | Solved / requested | Blocked | Cycles | Tokens | Active minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| amd_xdna2_npu2 | baseline | 27/30 | 0 | 89 | 1,758,550 | 140.9 |
| amd_xdna2_npu2 | structured_ir | 27/30 | 0 | 87 | 2,141,898 | 144.0 |
| amd_xdna2_npu2 | hinted_ir | 30/30 | 0 | 56 | 1,665,963 | 99.0 |
| amd_xdna2_npu2 | structured_ir_no_validation | 30/30 | 0 | 60 | 1,431,629 | 87.5 |
| amd_xdna2_npu2 | structured_ir_no_reuse | 28/30 | 0 | 84 | 3,126,524 | 155.4 |
| intel_npu_4000 | baseline | 30/30 | 0 | 34 | 376,724 | 14.0 |
| intel_npu_4000 | structured_ir | 30/30 | 0 | 32 | 746,094 | 24.3 |
| intel_npu_4000 | hinted_ir | 30/30 | 0 | 35 | 777,609 | 24.8 |
| intel_npu_4000 | structured_ir_no_validation | 30/30 | 0 | 32 | 740,257 | 23.6 |
| intel_npu_4000 | structured_ir_no_reuse | 30/30 | 0 | 32 | 778,453 | 24.4 |

Active time sums concurrent case clocks; campaign elapsed clocks include any resume pauses.
Token-budget curves in summary.json replay first-success costs within the ten-cycle cap; they are not hard spending caps.
Unresolved cases remain in requested denominators; paired intervals use only complete kernel clusters, with coverage reported.
Bootstrap intervals are exploratory, resampling kernels while keeping their repetitions together.
Per-repetition report.html files contain all-arm PNG/SVG charts and detailed evidence.
