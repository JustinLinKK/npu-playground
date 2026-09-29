# Research pilot audit v1

Execution changed from serial to parallel during this campaign. Case active time includes compiler queueing; summed active time is not campaign wall time. Timing comparisons are confounded by execution mode. See parallel_events in audit.json for affected cases and worker limits.

Campaign status: completed.
Preflight status: completed. Accounting complete: False.

Known tokens: 151,117 (3,089 preflight; 148,028 translation). Unknown-usage calls: 1 (0 currently in flight).

Bound terminal manual reviews applied: 2. Unreviewed stage failure labels are triage suggestions, not causal diagnoses. Counts are stage events, not unique failed cases; downstream prerequisites are omitted.
Distributions include all started cases, including failures and interruptions; pending cases are not zero-cost trials.
Known stage durations omit unrecorded orchestration and interrupted work; do not equate their sum with case active time.
Repeated IR evidence is charged zero validation time when the runner records reuse. Reasoning tokens are already included in output tokens.

| Category | Stage events |
| --- | ---: |
| backend_api | 28 |
| infrastructure | 1 |

Full evidence, input hashes, cost quantiles, case status and per-stage traces: `audit.json`, `cases.csv`, `stages.csv`.
This is offline validation evidence, not physical NPU correctness or kernel runtime performance.
