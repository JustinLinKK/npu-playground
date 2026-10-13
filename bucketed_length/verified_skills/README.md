# Direction 6: verified kernel skills

Admit a completed run after evidence verification:

```bash
.venv/bin/python -m bucketed_length admit \
  --run runs/bucketed-length/feedback-repair \
  --library runs/bucketed-length/library
```

Compare `--direction verified_skills --arm off` against `--arm reuse --library
runs/bucketed-length/library`. `reuse` snapshots prior entries before generation
and admits its own successful result only after the run is finished. Run a
predeclared sequence of distinct tasks to measure later-task first-attempt
acceptance. Use separate library copies/cohorts for a fair fixed-library control.

Admission rejects synthetic, failed, incomplete-usage and over-budget runs,
changed evidence, unbound candidate files, incorrect/legacy validation, and
unresolved toolchains. Content-addressed entries include the contract, bundle,
compiler evidence, source/report hashes, features and observed error-to-repair
pairs. Retrieval rechecks integrity and validation; both target and compiler
fingerprint must match. Entries never gain a physical-execution label from a
host-only pass. These are trusted local records, not signed attestations.

The library stores verified code and observed repairs, not unverified
model-authored claims. A prior error followed by a passing bundle records an
association, not proof that a particular textual edit caused the repair.
