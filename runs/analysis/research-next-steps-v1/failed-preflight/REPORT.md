# Research pilot audit v1

Campaign status: blocked.
Preflight status: blocked. Accounting complete: False.

Known tokens: 0 (0 preflight; 0 translation). Unknown-usage calls: 1.

Stage failure labels are triage suggestions requiring manual review, not causal diagnoses. Counts are stage events, not unique failed cases; downstream prerequisites are omitted.
Distributions include all started cases, including failures and interruptions; pending cases are not zero-cost trials.
Known stage durations omit unrecorded orchestration and interrupted work; do not equate their sum with case active time.
Repeated IR evidence is charged zero validation time when the runner records reuse. Reasoning tokens are already included in output tokens.

| Category | Stage events |
| --- | ---: |
| infrastructure | 1 |

Full evidence, input hashes, cost quantiles, case status and per-stage traces: `audit.json`, `cases.csv`, `stages.csv`.
This is offline validation evidence, not physical NPU correctness or kernel runtime performance.
