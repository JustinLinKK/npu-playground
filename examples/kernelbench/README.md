# KernelBench CUDA adaptations

Thirty distinct Level-1 tasks pinned to `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`.
The upstream MIT sources are retained beside each adaptation. Static float32 shapes are reduced for
offline correctness evaluation; these are not original KernelBench performance results. Manifests
record source provenance and frozen tolerances. Use scripts/validate_kernelbench_sources.py to
compare CUDA, upstream PyTorch, and the independent NumPy references on all twelve input families.
