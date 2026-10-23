# Adapted from Triton's fused-softmax tutorial (MIT).
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output, input, input_stride, output_stride, rows: tl.constexpr, cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(input + row * input_stride + offsets, mask=offsets < cols, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(output + row * output_stride + offsets, result, mask=offsets < cols)

