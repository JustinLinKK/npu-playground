# Case study v3: prompt guidance and IR

Model: qwen/qwen3-coder-30b-a3b-instruct. Provider: openrouter.

Stage: controlled. Status: interrupted.

Offline validation only; no physical NPU correctness or performance claim.

| Target | Arm | Solved / requested | Blocked | Cycles | Tokens | Active minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| amd_xdna2_npu2 | baseline_minimal | 0/30 | 0 | 100 | unknown (known: 484,302) | 86.6 |
| amd_xdna2_npu2 | baseline | 1/30 | 0 | 91 | unknown (known: 479,738) | 64.0 |
| amd_xdna2_npu2 | structured_ir | 1/30 | 0 | 92 | unknown (known: 684,049) | 92.6 |
| amd_xdna2_npu2 | hinted_ir | 1/30 | 0 | 79 | unknown (known: 693,395) | 84.5 |
| intel_npu_4000 | baseline_minimal | 6/30 | 0 | 59 | unknown (known: 142,829) | 16.0 |
| intel_npu_4000 | baseline | 8/30 | 0 | 36 | unknown (known: 88,950) | 10.9 |
| intel_npu_4000 | structured_ir | 6/30 | 0 | 60 | unknown (known: 231,724) | 18.7 |
| intel_npu_4000 | hinted_ir | 5/30 | 0 | 44 | unknown (known: 262,335) | 51.8 |

Active time sums concurrent case clocks; campaign elapsed clocks include any resume pauses.
Token-budget curves in summary.json replay first-success costs within the ten-cycle cap; they are not hard spending caps.
Unresolved cases remain in requested denominators; paired intervals use only complete kernel clusters, with coverage reported.
Bootstrap intervals are exploratory, resampling kernels while keeping their repetitions together.
Per-repetition report.html files contain all-arm PNG/SVG charts and detailed evidence.
