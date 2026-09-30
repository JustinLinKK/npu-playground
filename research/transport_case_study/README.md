# Transport correctness feasibility study

Implements the Day 1 gate in
[the study plan](../../docs/NPU_TRANSPORT_EARLY_CASE_STUDY.md). The current result
is **STOP at Gate A**: Docker's WSL integration and Windows Linux-engine pipe were
unavailable on 2026-09-30, no native compiler was found, and neither binary decoding
nor a reusable-device launch contract was validated. This is not evidence that the
research idea fails. Historical binaries are retained separately from new builds.

From the repository root, collect a new offline evidence bundle:

```bash
.venv/bin/python -m research.transport_case_study.cli build \
  --output runs/analysis/transport-case-study/first-feasibility
```

`preflight` uses the same options and only probes availability; `build` additionally
attempts the existing unmodified `add` fixture when Docker is available. It checks
the installed compiler package versions and AIE commit against the lock, resolves
the local image ID, then uses that immutable identity with the existing restricted
compiler wrapper (2 CPUs, 4 GiB, one job, 180 seconds per stage). It never pulls or
rebuilds an image. The fixture is float32 addition, not a scored int32 T1 workload.

To preserve an existing invocation's complete artifact/log bundle, optionally add:

```text
--historical-invocation runs/case-study-v4-amd-interruption-rerun-retry1/preflight-compiler/amd_xdna2_npu2/invocation-udua7nbv
```

This is an optional machine-local archive, not a dependency of preflight. Its
source must match the existing feasibility fixture; the snapshot records its
image/tool metadata and lock-hash comparison without promoting historical compile
or simulation success to a new result. All copied files are hashed. Missing or
unsafe archives are errors. No historical file is modified.

Exit codes: **2** means the feasibility gate stopped after evidence was written;
**1** means evidence collection/verification failed; **0** means report regeneration
succeeded. None is a transport correctness PASS. A successful compile still cannot
open Gate A without a validated decoder and sufficiently specified launch protocol.

Regenerate tables from immutable raw records without Docker or the old invocation:

```bash
.venv/bin/python -m research.transport_case_study.cli report \
  --run runs/analysis/transport-case-study/first-feasibility
```

Each run includes `REPORT.md`, `environment.json`, source snapshots,
`manifest.frozen.json`, `launch_contract.md`, `related_work.md`, `results.jsonl`,
`artifact_manifest.json`, `support_matrix.csv`, `metrics.csv` and probe logs.
`report` verifies hashes before regenerating the three derived report/table files;
the hash manifest excludes those derived files and itself. Hashes detect accidental
changes, not malicious replacement of both the evidence and hash manifest.
The 8 clean and 16 mutant slots remain visible across all 7 controls (168 NOT_RUN
rows); these are denominator records, not executed seeds or launches. Mutation
parents/layers are assigned provisionally, and the manifest explicitly says it is
not frozen for scoring until legality is established.

Outputs must be new directories. `runs/` is already ignored by repository policy;
large binaries are not committed automatically. Keep the whole run directory to
preserve reproducibility. The current source implementation remains isolated here;
the compiler, simulator, executors and historical numerical tolerances are unchanged.

No `evaluate` command, binary parser, transport model, mutation generator or schedule
explorer is implemented: the plan explicitly stops that work when Gate A fails.
Accordingly no syntax/stride/lifecycle/witness tests are represented as completed.
There are no device, paid-provider, installation, reset or endpoint-discovery paths.
The next experiment is a fresh pinned fixture build plus an independently evidenced
decoder and launch/reset protocol. Review [the launch audit](launch_contract.md)
and [prior-art overlap](related_work.md) before extending scope.

Focused checks:

```bash
.venv/bin/python -m pytest tests/test_transport_case_study.py
```
