# Direction 3: fixed transport, generated compute

Compare `--direction template_generation --arm free_form` with `--arm skeleton`.
The skeleton arm selects an explicit schedule from the contract and requests only
a C++ `compute` body. `design.py`, buffer order, FIFO allocation, DMA, completion
and the function signature come from the existing trusted single-worker scaffold.
The model cannot submit a replacement transport file through `MicroKernel`.

The initial supported templates are float32 rank-one elementwise addition,
rank-one sum reduction with a retained singleton output, and rank-two GEMV
`[M,K] × [K,1] → [M,1]`. The schedule records shapes, element counts, a lane hint,
tail handling, FIFO depth, and scaffold hash. It allows one finite launch with
at most 16 KiB total tensor storage. Other shapes, layouts and targets are rejected
explicitly. The generic single-worker graph is reused; historical fixtures are
not edited.

Reduction is the third supported operation class alongside elementwise and GEMV;
the plan's suggested RMSNorm example requires a new independent oracle, which the
existing manifest schema does not provide. RMSNorm, double buffering, multi-tile
autotuning and intrinsic families unsupported by the existing simulator are not
claimed. The lane hint is not measured vector utilization or a legal guarantee
for arbitrary vector intrinsics; the existing compiler remains the final judge.

Native tests execute all three microkernel bodies, including a 17-element vector
tail, under ASan/UBSan against independent randomized/adversarial references.
Transport and NPU compilation still require the pinned container. Use the small
64-element manifest in [the shared example](../README.md#run) for the first run.
