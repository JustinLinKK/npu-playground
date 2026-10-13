# Direction 1: executable intermediate

Run `--direction executable_ir` with `--arm direct`, `json_summary`, or
`executable_ir`; see the [shared launch command](../README.md#run).

The summary control requests operation/indexing/numeric-behavior metadata.
The executable arm requests existing KernelIR 2.0 and runs its NumPy interpreter
against the independent oracle on all twelve feedback cases before backend
generation. Invalid IR consumes an attempt and gets repair feedback; validated
IR is reused when only the backend fails. Direct generation has no pivot call.
All arms keep source, interface, backend requirements and the accounted token cap.

This chooses the plan's executable-IR option. It never executes arbitrary Python
returned by a model, changes the repository's IR schema, or makes a generated
reference the sole judge of its own correctness. Inspect `semantic_validation`,
calls, tokens, compilation and offline acceptance in each run's `report.json`.
