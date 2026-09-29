> **Start here:** [Combined Phase 1 report with all results and charts](PHASE1_REPORT.md). This file retains the detailed historical audit.

# Research pilot audit v1

Execution changed from serial to parallel during this campaign. Case active time includes compiler queueing; summed active time is not campaign wall time. Timing comparisons are confounded by execution mode. See parallel_events in audit.json for affected cases and worker limits.

Campaign status: completed.
Preflight status: completed. Accounting complete: False.

Known tokens: 987,525 (2,680 preflight; 984,845 translation). Unknown-usage calls: 2 (0 currently in flight).

Bound terminal manual reviews applied: 18. Unreviewed stage failure labels are triage suggestions, not causal diagnoses. Counts are stage events, not unique failed cases; downstream prerequisites are omitted.
Distributions include all started cases, including failures and interruptions; pending cases are not zero-cost trials.
Known stage durations omit unrecorded orchestration and interrupted work; do not equate their sum with case active time.
Repeated IR evidence is charged zero validation time when the runner records reuse. Reasoning tokens are already included in output tokens.

| Category | Stage events |
| --- | ---: |
| backend_api | 176 |
| buffer_dataflow | 2 |
| infrastructure | 2 |
| model_output | 20 |
| numerical | 7 |
| semantic | 6 |
| unsupported_ir | 3 |

Full evidence, input hashes, cost quantiles, case status and per-stage traces: `audit.json`, `cases.csv`, `stages.csv`.
This is offline validation evidence, not physical NPU correctness or kernel runtime performance.
