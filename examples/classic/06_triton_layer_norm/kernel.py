# Adapted from Triton's layer-normalization tutorial (MIT).
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(output, input, weight, bias, columns: tl.constexpr, epsilon: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    x = tl.load(input + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / columns
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    normalized = centered * tl.rsqrt(variance + epsilon)
    w = tl.load(weight + offsets, mask=mask)
    b = tl.load(bias + offsets, mask=mask)
    tl.store(output + row * columns + offsets, normalized * w + b, mask=mask)

