# Versioned research extensions

These tools are separate from the frozen v4 implementation. The active research plan is [../docs/npu-ir-research-next-steps.md](../docs/npu-ir-research-next-steps.md).

## OpenRouter spending

The initial authorization was **$30 total**, shared across pilot, diagnostics and later controls. The user subsequently authorized continuation under their OpenRouter-side billing limit and instructed stopping on billing-limit errors. The current process retains its local $30 known-spend backstop. Always reuse `runs/research-next-steps-budget.json` to preserve cumulative spending and historical gaps; do not restart after a billing-limit error.

```bash
env -u OPENROUTER_API_KEY .venv/bin/python research/run_budgeted_study.py \
  --ledger runs/research-next-steps-budget.json --limit-usd 30 \
  --config configs/case-study-v4.json \
  --output runs/case-study-v4-deepseek-pilot-retry1 --acknowledge-unknown-spending
```

The environment override is removed for this command because this experiment uses the verified key in `config.yaml`. No credential is stored in the ledger. Returned OpenRouter `usage.cost` is charged, including rejected responses and preflight. A ledger lock prevents concurrent guarded launchers. Existing campaign records are imported idempotently by generation ID.

The guard records a pending request before sending it. Missing costs, duplicate generation identities or lost responses stop further spending. Do not clear a pending/unknown-spending flag without independent billing reconciliation. With explicit user authorization, `--acknowledge-unknown-spending` permits historical billing gaps while preserving their call records and the unknown flag. It does not waive pending requests, the known-spend limit, or new billing gaps. Known spending remains a lower bound; this option does not verify a provider-side cap. Each subsequent launch also checks campaigns already represented in the ledger.

The response that reaches $30 is saved before the runner pauses; an in-flight request can exceed the threshold. No further paid request starts. Budget bookkeeping contributes to measured active time. Preserve the archived guard version and instrumentation notes when interpreting the pilot.

## Offline audit

```bash
PYTHONPATH=src .venv/bin/python research/audit_v1.py \
  --campaign runs/case-study-v4-deepseek-pilot-config-key \
  --output runs/analysis/research-next-steps-v1/pilot
```

Outputs include source-record hashes, call/stage and case CSVs, cost quantiles, provisional failure categories, and PNG/SVG figures. The audit never writes into the campaign. Pending cases are not zero-cost trials, and in-flight calls are not classified as failures. Categories require manual review; stage-event counts are not independent case failures. Physical target execution is outside the offline acceptance contract.

The diagnostic selection and reference IRs are under `configs/research-next-steps-v1/`. Reference preparation is agent-inspected, not human-reviewed. Original layer normalization fails one frozen source-validation input; that discrepancy remains visible rather than being resolved by changing pilot acceptance.

## Audited-reference diagnostic

`diagnostic_v1.py` uses the unchanged structured-IR backend prompt and repair loop with an externally prepared reference checkpoint. It makes no intermediate-generation call. Binding checks and semantic validation are charged as active preparation time; unknown original preparation effort remains disclosed. Backend failures reuse the validated reference. Unresolved references remain explicit coverage rows.

For the authorized retry, use `configs/research-next-steps-v1/diagnostic-selection-retry1.json`; it retains the original six-case selection and reference hashes. Pass `--acknowledge-unknown-spending` only under the recorded continuation authorization. This permits historical campaign billing gaps without relaxing the selected pilot audit gate.

Execution is gated on a complete, reconciled pilot and a review JSON at `<pilot-directory>-review.json`. The review must contain `decision="proceed_to_diagnostic"`, `reviewer`, substantive `review_notes`, and `pilot_hashes` equal to the offline audit's `input_sha256`. Inspect failures and the source-contract discrepancy before recording that decision; this gate is not satisfied by tests alone. Provider, compiler, source and reference identities are checked before spending.

```bash
env -u OPENROUTER_API_KEY .venv/bin/python research/diagnostic_v1.py \
  --selection configs/research-next-steps-v1/diagnostic-selection.json \
  --output runs/audited-ir-diagnostic-v1 \
  --ledger runs/research-next-steps-budget.json
```

This is a diagnostic control, not a free-preparation competitor. Alternative lowering and confirmatory analysis remain gated on the research decisions in the plan.

## Hidden final inputs

`hidden_evaluation.py` is a separate, serial, model-free evaluation process. It cannot run on the development pilot. For a new confirmatory campaign, use the spending guard with `--command preflight`, then seal the reviewed private suite before running any benchmark case:

```bash
PYTHONPATH=src .venv/bin/python research/hidden_evaluation.py seal \
  --campaign runs/CONFIRMATORY_CAMPAIGN --suite /private/final-inputs/suite.json \
  --commitment /private/final-inputs/commitment.json
```

The suite is JSON with `protocol="hidden-inputs-v1"`, `reviewed_by`, `review_notes`, and `kernels`. Each kernel maps to its exact `manifest_sha256` and a nonempty `cases` list; each case has a unique safe `id`, an NPZ `path` relative to the suite, and its `sha256`. Archives contain exactly the declared input tensors with matching shapes/dtypes and numerical domain. Scalars retain their manifest values. Duplicate inputs and exact copies of feedback cases are rejected. Distribution coverage and reviewer identity require actual review; this tool does not manufacture either.

After the campaign completes, evaluate with the same suite and commitment:

```bash
PYTHONPATH=src .venv/bin/python research/hidden_evaluation.py evaluate \
  --campaign runs/CONFIRMATORY_CAMPAIGN --suite /private/final-inputs/suite.json \
  --commitment /private/final-inputs/commitment.json --output /private/final-results
```

The output must be new and outside the campaign. Frozen corpus, evaluator, compiler and private-input identities must still match. Selection is fixed at sealing: evaluate the first offline-passing bundle, otherwise the last submitted bundle, retaining absent/unsupported cases. A final success requires both budget-qualified feedback success and hidden-input success. Outputs never enter the repair runner; no model calls occur. Keep private files out of prompts and model-tuning work. This does not establish source equivalence or replace the required reviewed held-out kernel corpus. Current validation uses synthetic fixtures, not a confirmatory study.

Focused verification:

```bash
.venv/bin/python -m pytest tests/test_research_budget.py tests/test_research_audit.py tests/test_hidden_evaluation.py tests/test_research_diagnostic.py
```

## Original CUDA reference validation

`validate_cuda_references.py` separately compiles and executes the selected, hash-bound vector-add and transpose source files using NVRTC and the CUDA driver API. It uses an existing CUDA-enabled torch environment for allocations, requires an explicit NVRTC library path, and installs no dependencies. Fixed launch geometry applies only to these two selected manifests. Output buffers begin as NaNs to expose unwritten elements. Existing output directories are rejected.

```bash
PYTHONPATH=src /home/justin/GXG-opt/.venv-gpt2-v2/bin/python research/validate_cuda_references.py \
  --output runs/analysis/research-next-steps-v1/source-cuda \
  --nvrtc /home/justin/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib/libnvrtc.so.13
```

The archived execution passed 12/12 cases for each source on RTX 5090. All 24 saved NPZ outputs were independently re-compared against the manifest oracle and matched the recorded comparisons exactly. Reports bind source, manifest and CUBIN hashes; environment metadata binds the adapter and NVRTC library. `diagnostic-selection-retry1.json` links this evidence without changing the original reference IR or case selection. Layer normalization's recorded source-contract failure remains unresolved.

A recorded HTTP 429 interruption may be retained while continuing the remaining pilot schedule with `--continue-past-rate-limit --acknowledge-unknown-spending`. This changes only the outer loop's view of already blocked 429 cases. Their reports, attempts and missing usage remain untouched; newly encountered errors still pause, and HTTP 402 is never skipped. The campaign reaching the end of its schedule does not imply every case completed or accounting is complete. The audit still includes blocked cases and refuses complete-accounting claims. Recovery provenance is archived separately in the campaign.

Use `audit_v1.py --reviews PATH` to apply completed manual reviews to ambiguous stage labels. Each JSON must have `status="terminal_review"`, `reviewer`, `source_report`, its exact `source_report_sha256`, and `cycles` containing `cycle` and `failure_category`. Provisional reviews are ignored. The tool rejects stale records, other campaigns, nonterminal cases, duplicate terminal reviews and unknown categories. It only replaces `needs_review` labels; it preserves automatic counts, hashes the review files, and does not change accounting or acceptance status. The retry audit uses `runs/analysis/research-next-steps-v1/pilot-retry1/case-reviews`.

## Backend API guidance preparation

`capture_backend_api.py` captures selected task-independent Python API signatures from the exact image IDs in a completed pilot preflight. It uses the image's normal entrypoint, network-disabled read-only containers, no candidate code, and no test or oracle data. It archives its implementation, module hashes, image IDs and preparation durations. Existing outputs are never overwritten.

```bash
.venv/bin/python research/capture_backend_api.py \
  --campaign runs/case-study-v4-deepseek-pilot-retry1 \
  --output runs/analysis/research-next-steps-v1/backend-api-reference-v2
```

The successful capture contains 16 AMD symbols and 19 Intel symbols; one Intel signature is unavailable through Python introspection. Generic decorator signatures also need additional documentation before a complete guidance condition is frozen. This inventory is preparation, not an executed guidance control or evidence that signatures alone solve translation. The earlier `backend-api-reference` directory preserves an Intel capture failure caused by bypassing its image entrypoint; include both attempts' preparation time. Any later guidance condition must provide the same target-specific reference to all arms and remain separate from the pure pilot.

The expanded capture in `backend-api-reference-v4` adds AMD `Device`/`Core` constructor signatures, the `aie.dialects.aiex.runtime_sequence` decorator, and documented overload signatures when Python introspection is generic or unavailable. Its presence checks passed for those AMD interfaces and OpenVINO `Model`/`save_model`. Earlier inventories remain archived. These signatures identify callable interfaces, not full operation semantics, legal dataflow designs, or complete guidance coverage. Capture durations exclude unmetered agent inspection effort; do not present that preparation as free.

### Parallel Phase 1 continuation

At the user's request, the serial pilot was gracefully paused and its six remaining Intel cases resumed concurrently. Use the separate runner to preserve the frozen per-case implementation:

```bash
env -u OPENROUTER_API_KEY .venv/bin/python research/run_parallel_study.py \
  --config configs/case-study-v4.json \
  --output runs/case-study-v4-deepseek-pilot-retry1 \
  --ledger runs/research-next-steps-budget.json \
  --workers 8 --compiler-jobs 4 \
  --acknowledge-unknown-spending --continue-past-rate-limit
```

The runner exclusively locks both campaign and ledger, verifies frozen identities, retains completed/failed cases and explicitly retained HTTP 429 interruptions, and resumes only unfinished cases. Each model request gets a persistent pending entry; ledger updates are synchronized. Provider errors stop new calls while already-sent responses drain into their case records. Up to one request per active worker can already be in flight when a spending threshold or billing error is observed. Unresolved pending entries block both serial and parallel restarts.

Compiler jobs retain the configured per-job CPU/memory limits. The campaign archives the runner, guard, affected case IDs, concurrency limits and transition timestamp. Reports identify mixed serial/parallel execution: queueing contributes to case active time, summed active time differs from campaign wall time, and timing comparisons across execution modes are confounded. This operational change does not alter prompts, model settings, per-case budgets or acceptance criteria. Current scope ends after Phase 1.
