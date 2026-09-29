# Cloud analysis handoff

This checkout packages the current implementation and selected saved evidence for
analysis as of 2026-09-28. Phase 1 and the requested AMD interruption reruns have
finished; no Phase 2 experiment was started for this handoff. Older running-state
and billing-limit notes are historical records, not current launch instructions.

## Read in this order

1. [Combined Phase 1 report](../runs/analysis/research-next-steps-v1/pilot-retry1/PHASE1_REPORT.md): outcomes, charts, costs, failure analysis, and limitations.
2. [Combined data](../runs/analysis/research-next-steps-v1/pilot-retry1/combined-data.json) and [case table](../runs/analysis/research-next-steps-v1/amd-interruption-rerun/resolved-pilot-cases.csv): original and replacement attempts remain separate.
3. [Pilot audit](../runs/analysis/research-next-steps-v1/pilot-retry1/audit.json), [rerun audit](../runs/analysis/research-next-steps-v1/amd-interruption-rerun/audit.json), and their adjacent `case-reviews/`: case metrics and reviewed stage failures.
4. [V4 protocol](case-study-v4.md), [frozen configuration](../configs/case-study-v4.json), and [research plan](npu-ir-research-next-steps.md#phase-2-isolate-the-semantic-to-backend-bottleneck): distinguish implemented behavior from proposed experiments.

## Current result

| Item | Saved evidence |
| --- | --- |
| Latest outcome across 18 slots | 5 offline successes, 12 budget failures, 1 HTTP 429 interruption |
| Original pilot | 5 successes, 11 budget failures, 2 interruptions |
| Intel direct / hinted / structured | 2/3, 2/3, 1/3 successes |
| AMD | No offline success; direct transpose remains interrupted after replacement |
| Known spending across associated attempts and preflights | $1.0403365089; 195 recorded generation charges |
| Accounting | Five historical requests lack complete billing records; complete accounting remains false |

The two AMD replacements received fresh per-case budgets. Their outcomes cannot
be presented as one uninterrupted equal-total-budget trial. Charge original and
replacement work together. Active case time sums concurrent work and includes a
mixed serial/parallel schedule; it is not campaign elapsed time or device latency.
Offline acceptance does not establish execution on physical NPU hardware.

Backend construction is the main observed obstacle. Intel structured vector
addition failed after valid IR, while direct and hinted passed. Layer normalization
also has an unresolved source/oracle discrepancy: its original Triton source
failed one of twelve finite tests, matching the error metrics of seven Intel host
validations. These observations motivate diagnostics, not a general IR advantage
or disadvantage.

## Recommended next decision

Review and disposition the Phase 2 entry conditions before launching another
campaign. Recover the missing billing evidence if possible; otherwise propose an
explicit prospective accounting amendment while preserving Phase 1's incomplete
status. The existing diagnostic runner's complete-accounting gate remains active.
Also resolve the layer-normalization numerical contract, specify safe filename
normalization equally across arms, and freeze provider-interruption handling in a
separately versioned protocol. Do not change the frozen pilot's outcomes or relax
its tolerance to pass observed candidates.

Then preregister four pairs (vector addition and transpose on Intel and AMD),
comparing the same audited IR through the current model backend generator and a
declared constrained/mechanical lowering path. Keep layer normalization visible as
contract-unresolved. Fix repetitions, supported operations, compiler identities,
acceptance, budgets, concurrency, and preparation-cost accounting before execution.
The existing [diagnostic selection](../configs/research-next-steps-v1/diagnostic-selection-retry1.json)
still contains six pairs and requires prospective versioning; the alternative
lowering is a proposal, not an implemented or validated treatment.

The cloud review should return a decision with evidence citations, unresolved
entry conditions, and the smallest proposed implementation/configuration changes.
A broad repeat of the unchanged pilot is not the recommended next experiment.
This handoff does not authorize paid runs or implement the proposed Phase 2 work.

Useful supporting evidence:

- [CUDA vector-add report](../runs/analysis/research-next-steps-v1/source-cuda/cuda_vector_add/report.json) and [transpose report](../runs/analysis/research-next-steps-v1/source-cuda/cuda_tiled_transpose/report.json): each original source passed twelve finite tests.
- [Layer-normalization source report](../runs/analysis/research-next-steps-v1/source-layer-norm/report.json) and [counterexample](../runs/analysis/research-next-steps-v1/source-layer-norm/counterexample.png): preserve the unresolved contract.
- [Backend API reference v4](../runs/analysis/research-next-steps-v1/backend-api-reference-v4/reference.json): captured signatures and image/module identities; not an executed guidance control or a complete semantics reference.
- [Historical audit](../runs/analysis/qwen-terra-20260917/REPORT.md) and [historical comparison](../runs/analysis/v4-design-audit-20260917/historical-summary.json): context only; overlapping recovery runs are not independent repetitions.

## What is portable

The commit includes source, compiler tools, fixtures, tests, versioned configs,
research scripts, documentation, and an explicit selection of reports, JSON/CSV
aggregates, terminal reviews, and figures. [The evidence checksum list](cloud-evidence.sha256)
identifies every included saved artifact. Their original bytes and paths are
preserved; `/runs/*` remains ignored for future outputs.

Raw campaign checkpoints, provider responses, SQLite databases, the live spending
ledger, tensor archives, compiled binaries, credentials, local `config.yaml`, and
the older presentation bundle remain local. Absolute paths in saved JSON are
provenance from the original workstation. Raw-record links in historical reports
may therefore be unavailable in a cloud checkout. Aggregate analysis is portable;
full raw-record re-auditing, report regeneration, and campaign resumption require
the original local artifacts. In particular, `research/build_phase1_report.py`
requires those raw records and the original ledger. Do not create an empty ledger
to resume a historical campaign.

Verify the selected evidence without model calls or hardware:

```bash
sha256sum --check --quiet docs/cloud-evidence.sha256
uv sync --frozen --extra dev
uv run pytest -m 'not container and not requires_npu'
```

At handoff preparation, the focused research checks passed (81 passed, 2 skipped),
the full non-container/non-hardware suite passed (257 passed, 25 deselected), and
`uv lock --check --offline` and `uv build --offline` passed. Container, CUDA, and
physical-NPU checks were not rerun; source-execution claims refer to the archived
reports. No new model calls were made.

An export containing only the staged files independently passed the same 257
offline tests and all 105 evidence checksums. The 18 selected outcomes, 20 terminal
review hashes, 195 recorded charges, and five unresolved billing gaps were checked
against the local source records before committing. The source diff check passed;
the unrestricted whitespace check flags existing generated SVG/CSV formatting,
Markdown hard breaks, and one fixture's trailing blank line, preserved unchanged.
