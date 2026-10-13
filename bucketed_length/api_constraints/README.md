# Direction 5: API subset check and repair

Compare `--direction api_constraints --arm off` and `--arm checked`.
The checked arm includes the declared repository-supported API subset in its
prompt, then checks Python imports/qualified symbols, selected dynamic-resolution
constructs, C++ includes/directives and `aie::` intrinsics before compilation.
Rejections consume an attempt and feed diagnostics into bounded repair.

The subset is intentionally conservative: it includes the AIE host validator's
supported intrinsics and selected OpenVINO/IRON symbols, not the complete vendor
API. It rejects `adf.h`, unknown qualified functions, wildcard imports, and common
dynamic lookup escapes. It is tied to the run's recorded compiler fingerprint and
its own snapshotted source. Compiler checks still validate overloads, lane widths,
types, buffer bindings, and emitted transport.

This implements the plan's proposed cheap verifier experiment, **not** full
token-level grammar decoding, a complete Python/C++ type checker, or a security
sandbox. Instance methods, aliases and embedded MLIR are not fully resolved.
Whitelist acceptance is never correctness evidence; the existing restricted
container and numerical checks remain mandatory.
