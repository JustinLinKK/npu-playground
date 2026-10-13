# Direction 4: feature and diagnostic retrieval

Compare `--direction feature_retrieval` arms `off`, `text`, and `features` against
one frozen `--library runs/bucketed-length/library`. Populate it using
[verified admission](../verified_skills/README.md), not unverified scraped kernels.

Feature keys contain operation class, dtypes, tensor ranks/shapes, reduction axes,
layouts, target and the actual compiler/evaluator fingerprint. Target and
fingerprint are hard filters, including for the lexical text-overlap control.
The feature arm also filters operation/dtypes/ranks/reduction/layout and ranks
matching diagnostic-to-fix records ahead of logarithmic shape distance. This is a
deterministic training-free comparison; the text arm is not embedding-based RAG.

The fingerprint includes image identity, installed tools, profile and evaluator
hashes. Compiler mismatch never falls back to a different backend's examples.
No firmware/driver claim is possible in the offline evaluator. Every selected
entry ID is recorded, and each run snapshots its library before generation.
An empty or mismatched library produces no retrieved examples, not invented ones.

The operation class comes from the declared contract, not inferred source AST
analysis. Bucket/phase fields and launch-geometry extraction belong to Part 2.
