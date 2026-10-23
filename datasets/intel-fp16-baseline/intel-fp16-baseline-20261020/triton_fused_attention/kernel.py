# Reduced forward kernel based on Triton's fused-attention tutorial (MIT).
import triton
import triton.language as tl

@triton.jit
def attention_kernel(output, query, key, value, sequence: tl.constexpr, head_dim: tl.constexpr, scale: tl.constexpr):
    query_block = tl.load(query + tl.arange(0, sequence)[:, None] * head_dim + tl.arange(0, head_dim)[None, :])
    key_block = tl.load(key + tl.arange(0, sequence)[:, None] * head_dim + tl.arange(0, head_dim)[None, :])
    scores = tl.dot(query_block, tl.trans(key_block)) * scale
    probabilities = tl.exp(scores - tl.max(scores, axis=1)[:, None])
    probabilities = probabilities / tl.sum(probabilities, axis=1)[:, None]
    value_block = tl.load(value + tl.arange(0, sequence)[:, None] * head_dim + tl.arange(0, head_dim)[None, :])
    tl.store(output, tl.dot(probabilities.to(tl.float16), value_block))

