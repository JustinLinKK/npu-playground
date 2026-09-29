> **Start here:** [Combined Phase 1 report with all results and charts](../pilot-retry1/PHASE1_REPORT.md). This file retains the detailed historical audit.

# AMD interruption rerun

The two original HTTP 429 interruptions were rerun as fresh attempts with unchanged model, prompts, source, validation, and per-case budgets. Original reports were preserved and their hashes rechecked. Both cases started concurrently. A new HTTP 429 blocked the direct case; the hinted response drained and its checkpoint resumed without regenerating saved work. Their results supplement rather than overwrite the original pilot.

| Arm | Original outcome | Rerun outcome | Rerun tokens | Rerun active seconds | Backend attempts |
| --- | --- | --- | ---: | ---: | ---: |
| hinted_ir | HTTP 429 interrupted | failed | 97,167 | 1134.4 | 10 |
| direct | HTTP 429 interrupted | blocked | 50,861 | 617.7 | 4 |

Using these replacement attempts for the two interrupted slots gives **5/18 offline successes**, 12 budget-exhausted failures, and 1 unresolved slots. This is a post-interruption result view; it is not an uninterrupted, equal-total-budget rerun of all 18 cases. The original pilot remains five successes, eleven budget failures, and two interruptions.

The replacement attempts have complete usage/timing records: **False**. Additional known spending for this rerun request, including both preflight launches, is **$0.1256507031**. Shared recorded spending is **$1.0403365089**. Historical missing billing records remain unresolved and must not be treated as zero or erased by successful request delivery.

The first launch stopped at preflight because the model returned an invalid IR schema. The second launch passed preflight but initially encountered a local SQLite WAL-initialization race before model generation. The runner now initializes the database before starting worker threads; the clean checkpoint resumed without repeating preflight. All launch records and costs are retained.

Detailed failures are in [the audited report](REPORT.md) and `case-reviews/`. The [resolved case table](resolved-pilot-cases.csv) records original and rerun outcomes separately and charges both attempts in `all_attempts_known_tokens`. No physical NPU correctness or performance claim follows from this offline experiment.
