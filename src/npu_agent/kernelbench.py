"""Independent NumPy/math references for the thirty pinned KernelBench adaptations."""
from __future__ import annotations

import math

import numpy as np


def reference(operation, args, p):
    x = args[0].astype(np.float64)
    if operation == 'kb_matmul':
        a, b = x, args[1].astype(np.float64)
        if p.get('transpose_a'):
            a = a.swapaxes(-1, -2)
        if p.get('transpose_b'):
            b = b.swapaxes(-1, -2)
        y = np.einsum('...ik,...kj->...ij', a, b, optimize=False)
    elif operation == 'kb_diagonal':
        y = x[:, None] * args[1]
    elif operation == 'kb_activation':
        kind = p['kind']
        if kind == 'relu':
            y = np.maximum(x, 0)
        elif kind == 'leaky_relu':
            y = np.where(x >= 0, x, .01 * x)
        elif kind in ('sigmoid', 'swish'):
            e = np.exp(-np.abs(x))
            y = np.where(x >= 0, 1 / (1 + e), e / (1 + e))
            if kind == 'swish':
                y *= x
        elif kind == 'tanh':
            y = np.tanh(x)
        elif kind == 'log_softmax':
            shifted = x - np.max(x, axis=1, keepdims=True)
            y = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        elif kind == 'gelu':
            erf = np.array([math.erf(float(v) / math.sqrt(2)) for v in x.flat]).reshape(x.shape)
            y = .5 * x * (1 + erf)
        elif kind == 'selu':
            y = 1.0507009873554805 * np.where(x > 0, x, 1.6732632423543772 * np.expm1(np.minimum(x, 0)))
        elif kind == 'hard_sigmoid':
            y = np.clip(x / 6 + .5, 0, 1)
        elif kind == 'softplus':
            y = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
        elif kind == 'softsign':
            y = x / (1 + np.abs(x))
        elif kind == 'elu':
            y = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        elif kind == 'hard_tanh':
            y = np.clip(x, -1, 1)
        else:
            raise ValueError(f'unknown KernelBench activation: {kind}')
    elif operation == 'kb_rms_norm':
        y = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + p['epsilon'])
    elif operation == 'kb_pool':
        spatial = x.shape[2:]
        k, stride, pad, dilation = (p[name] for name in ('kernel', 'stride', 'padding', 'dilation'))
        shape = tuple((s + 2 * pad - dilation * (k - 1) - 1) // stride + 1 for s in spatial)
        y = np.empty(x.shape[:2] + shape, dtype=np.float64)
        for index in np.ndindex(y.shape):
            values = []
            for offset in np.ndindex((k,) * len(spatial)):
                pos = tuple(i * stride - pad + o * dilation for i, o in zip(index[2:], offset))
                if all(0 <= v < size for v, size in zip(pos, spatial)):
                    values.append(float(x[index[:2] + pos]))
                elif p['kind'] == 'average':
                    values.append(0.)
            y[index] = max(values) if p['kind'] == 'max' else math.fsum(values) / len(values)
    elif operation == 'kb_reduce':
        reducer = {'mean': np.mean, 'max': np.max, 'min': np.min}[p['kind']]
        y = reducer(x, axis=p['axis'])
    elif operation == 'kb_cumsum':
        axis = p['axis']
        values = np.flip(x, axis) if p['reverse'] else x
        y = np.cumsum(values, axis=axis)
        if p['reverse']:
            y = np.flip(y, axis)
    elif operation == 'kb_mse':
        y = np.array([np.mean((x - args[1].astype(np.float64)) ** 2)])
    else:
        raise ValueError(f'unknown KernelBench reference: {operation}')
    return np.asarray(y, dtype=np.float32)


GOLDENS = [
    ('kb_matmul', [[[1, 2], [3, 4]], [[2], [3]]], {}, [[8], [18]]),
    ('kb_matmul', [[[1, 3], [2, 4]], [[2], [3]]], {'transpose_a': True}, [[8], [18]]),
    ('kb_diagonal', [[2, 3], [[1, 2], [3, 4]]], {}, [[2, 4], [9, 12]]),
    ('kb_rms_norm', [[[3, 4]]], {'epsilon': 3.5}, [[.75, 1.]]),
    ('kb_pool', [[[[1, 3, 2, 4]]]], {'kind': 'max', 'kernel': 2, 'stride': 1, 'padding': 0, 'dilation': 1}, [[[3, 3, 4]]]),
    ('kb_pool', [[[[2, 4]]]], {'kind': 'average', 'kernel': 2, 'stride': 1, 'padding': 1, 'dilation': 1}, [[[1, 3, 2]]]),
    ('kb_reduce', [[[1, 3], [5, 7]]], {'kind': 'mean', 'axis': 1}, [2, 6]),
    ('kb_reduce', [[[1, 3], [5, 7]]], {'kind': 'max', 'axis': 1}, [3, 7]),
    ('kb_reduce', [[[1, 3], [5, 7]]], {'kind': 'min', 'axis': 1}, [1, 5]),
    ('kb_cumsum', [[[1, 2, 3]]], {'axis': 1, 'reverse': False}, [[1, 3, 6]]),
    ('kb_cumsum', [[[1, 2, 3]]], {'axis': 1, 'reverse': True}, [[6, 5, 3]]),
    ('kb_mse', [[1, 3], [2, 1]], {}, [2.5]),
]
for kind, inputs, output in [
    ('relu', [-2, 0, 2], [0, 0, 2]), ('leaky_relu', [-2, 0, 2], [-.02, 0, 2]),
    ('sigmoid', [0], [.5]), ('tanh', [0], [0]), ('log_softmax', [[0, 0]], [[-math.log(2), -math.log(2)]]),
    ('swish', [0], [0]), ('gelu', [0], [0]), ('selu', [0, 1], [0, 1.0507009873554805]),
    ('hard_sigmoid', [-4, 0, 4], [0, .5, 1]), ('softplus', [0], [math.log(2)]),
    ('softsign', [-1, 0, 1], [-.5, 0, .5]), ('elu', [0, 1], [0, 1]), ('hard_tanh', [-2, 0, 2], [-1, 0, 1]),
]:
    GOLDENS.append(('kb_activation', [inputs], {'kind': kind}, output))
