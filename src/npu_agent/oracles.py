"""Independent manifest references. This module must never import the IR evaluator."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import KernelManifest


def reference(operation: str, args: list[np.ndarray], parameters: dict[str, Any]) -> np.ndarray:
    x = args[0]
    p = parameters
    if operation == 'vector_add':
        return np.add(x, args[1])
    if operation == 'scalar_mul':
        return x * p['scale']
    if operation == 'reduce_sum':
        return np.add.reduce(x, axis=p.get('axis'), dtype=p.get('accumulation_dtype', 'float32'),
                             keepdims=p.get('keepdims', False))
    if operation == 'transpose':
        axes = p.get('axes', list(reversed(range(x.ndim))))
        out = np.empty(tuple(x.shape[i] for i in axes), dtype=x.dtype)
        for index in np.ndindex(x.shape):
            out[tuple(index[i] for i in axes)] = x[index]
        return out
    if operation == 'matmul':
        return np.einsum('...ik,...kj->...ij', x.astype(np.float32), args[1].astype(np.float32),
                         optimize=False).astype(p.get('output_dtype', 'float16'))
    if operation == 'softmax':
        axis = p.get('axis', -1)
        rows = np.moveaxis(x, axis, -1)
        out = np.empty_like(rows)
        for index in np.ndindex(rows.shape[:-1]):
            row = rows[index]
            largest = float(max(row))
            weights = [math.exp(float(item) - largest) for item in row]
            total = math.fsum(weights)
            out[index] = [item / total for item in weights]
        return np.moveaxis(out, -1, axis)
    if operation == 'layer_norm':
        axis = p.get('axis', -1)
        rows = np.moveaxis(x, axis, -1)
        normalized = np.empty(rows.shape, dtype=np.float64)
        for index in np.ndindex(rows.shape[:-1]):
            row = [float(item) for item in rows[index]]
            mean = math.fsum(row) / len(row)
            var = math.fsum((item - mean) ** 2 for item in row) / len(row)
            normalized[index] = [(item - mean) / math.sqrt(var + p.get('epsilon', 1e-5)) for item in row]
        return (np.moveaxis(normalized, -1, axis) * args[1] + args[2]).astype(x.dtype)
    if operation == 'attention':
        scores = np.einsum('...ik,...jk->...ij', x.astype(np.float32), args[1].astype(np.float32))
        scores *= p.get('scale', x.shape[-1] ** -0.5)
        probs = reference('softmax', [scores], {'axis': -1})
        return np.einsum('...ij,...jk->...ik', probs, args[2].astype(np.float32)).astype(x.dtype)
    if operation == 'moving_average':
        return np.array([math.fsum(float(v) for v in x[max(0, i - p['window'] + 1):i + 1]) /
                         min(i + 1, p['window']) for i in range(len(x))], dtype=x.dtype)
    if operation == 'conv2d':
        w = args[1]
        pad = p.get('padding', 0)
        h, width = x.shape[2] + 2 * pad - w.shape[2] + 1, x.shape[3] + 2 * pad - w.shape[3] + 1
        if x.shape[1] != w.shape[1] or min(h, width) <= 0:
            raise ValueError('invalid convolution dimensions')
        padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        out = np.zeros((x.shape[0], w.shape[0], h, width), dtype=np.float64)
        for c in range(w.shape[1]):
            for r in range(w.shape[2]):
                for s in range(w.shape[3]):
                    out += padded[:, c:c + 1, r:r + h, s:s + width].astype(np.float64) * w[None, :, c, r, s, None, None]
        return out.astype(x.dtype)
    if operation == 'gamma_correction':
        return np.array([min(1.0, max(0.0, float(v))) ** p['gamma'] for v in x.flat], dtype=x.dtype).reshape(x.shape)
    if operation in ('sigmoid', 'silu'):
        values = []
        for v in x.flat:
            v = float(v)
            s = 1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v))
            values.append(s if operation == 'sigmoid' else v * s)
        return np.array(values, dtype=x.dtype).reshape(x.shape)
    raise ValueError(f'unknown independent reference: {operation}')


def execute_oracle(manifest: KernelManifest, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    args = [inputs[t.name] for t in manifest.tensors if t.direction != 'output']
    output = next(t for t in manifest.tensors if t.direction == 'output')
    value = np.asarray(reference(manifest.oracle.operation, args, manifest.oracle.parameters), dtype=output.dtype)
    if value.shape != output.shape:
        raise ValueError(f'oracle output shape {value.shape} disagrees with {output.shape}')
    return {output.name: value}


# Small hand-computed vectors exercise every builtin independently of manifests/IR.
GOLDENS = [
    ('vector_add', [[1., -2.], [3., 2.]], {}, [4., 0.]),
    ('scalar_mul', [[1., -2.]], {'scale': 3}, [3., -6.]),
    ('reduce_sum', [[1., -2., 4.]], {'axis': 0, 'keepdims': True}, [3.]),
    ('matmul', [[[1., 2.]], [[3.], [4.]]], {'output_dtype': 'float32'}, [[11.]]),
    ('transpose', [[[1., 2., 3.], [4., 5., 6.]]], {'axes': [1, 0]}, [[1., 4.], [2., 5.], [3., 6.]]),
    ('softmax', [[[0., 0.]]], {}, [[.5, .5]]),
    ('softmax', [[[0., math.log(3)]]], {}, [[.25, .75]]),
    ('layer_norm', [[[2., 2.]], [3., 3.], [1., -1.]], {}, [[1., -1.]]),
    ('layer_norm', [[[1., 3.]], [1., 2.], [0., 1.]], {'epsilon': 3}, [[-.5, 2.]]),
    ('attention', [[[[0.]]], [[[1.], [2.]]], [[[2.], [4.]]]], {}, [[[3.]]]),
    ('moving_average', [[2., 4., 9.]], {'window': 2}, [2., 3., 6.5]),
    ('conv2d', [[[[[1., 2.], [3., 4.]]]], [[[[2.]]]]], {'padding': 0}, [[[[2., 4.], [6., 8.]]]]),
    ('gamma_correction', [[-1., 0., .5, 1., 2.]], {'gamma': 2}, [0., 0., .25, 1., 1.]),
    ('sigmoid', [[-1000., 0., 1000.]], {}, [0., .5, 1.]),
    ('silu', [[-1000., 0., 1000.]], {}, [0., 0., 1000.]),
]


def validate_goldens(operation: str) -> None:
    found = False
    for op, args, parameters, expected in GOLDENS:
        if op != operation:
            continue
        found = True
        actual = reference(op, [np.array(v, dtype=np.float32) for v in args], parameters)
        if not np.allclose(actual, np.array(expected), rtol=1e-6, atol=1e-7):
            raise ValueError(f'independent oracle golden failed: {operation}')
    if not found:
        raise ValueError(f'no trusted golden for {operation}')
