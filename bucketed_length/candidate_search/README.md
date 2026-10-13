# Direction 7: differential/metamorphic candidate search

Compare `--direction candidate_search --arm single` and `--arm best_of_n
--population 4` (up to 8). Each member gets an independent bounded repair chain
using validated executable KernelIR. The whole population shares one accounted
token cap, including IR and repair calls. Chains stop after their first feedback
pass; failures on selection inputs never trigger additional repairs.

Before generation, the runner commits five unseen random input cases using
`--selection-seed`, disjoint from feedback random seeds. The twelve feedback
cases already include boundary and adversarial families. Where applicable it
adds a paired metamorphic case: commutativity for addition, or linearity for
sum reduction, matmul/GEMV, transpose and scalar multiplication.

Passing candidates are compiled/evaluated again with these private inputs. Actual
backend outputs must agree with both the executable IR and independent reference;
the selected metamorphic relation is checked against actual output archives.
Missing archives cannot pass. Other operations explicitly record metamorphic
coverage as unsupported and retain differential checking.

The runner chooses the first fully passing member, or lowest measured physical
latency if every eligible profile has comparable provenance and timing scope.
When latency is absent, comparable vectorization or stall counters use the same
declared ordering as the [feedback loop](../feedback_loop/README.md).
It does not rank CPU timing or invented vectorization scores. Saved selection
inputs/diagnostics are never appended to a model prompt. The serial evaluator
uses the repository's existing private-input injection pattern; do not share it
across concurrent threads.

These inputs select candidates and are not a final untouched benchmark. This
bounded implementation makes no formal-equivalence, universal-correctness or
best-of-N improvement claim without a separately controlled experiment.
