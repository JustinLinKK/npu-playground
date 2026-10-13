# Direction 2: evaluator-guided repair

Compare `--direction feedback_loop --arm one_shot` against `--arm repair`.
`--max-repairs 5` means at most six generation/evaluation attempts including the
initial attempt. Stop at the first accepted candidate. Compiler diagnostics,
failed numerical stages, prior code and available measured profiles become
bounded reflection/repair context. Provider/infrastructure failures stop the run;
they do not become purported program failures.

`profile()` accepts latency and optional vectorization/DMA/ObjectFIFO counters
only from a physical-NPU execution/performance record with provenance. Missing
counters stay `null`; source-level intrinsics are never scored as vectorization.
For a separately validated profiling evaluator, the Python `RunConfig` option
`optimize_profile=True` continues bounded optimization after correctness and
selects a passing candidate only when profile scopes/provenance match: lowest
latency, otherwise highest vectorization, otherwise lowest DMA/FIFO stalls (in
that order, using a metric available for every passing candidate).
The ordinary CLI uses offline validation and cannot collect those measurements.

Use [the shared commands](../README.md#run) to launch and summarize runs.
