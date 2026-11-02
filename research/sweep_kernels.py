"""Synthetic FP16 kernels for the size and batching sweep of research/benchmark_kernel_energy.py (--sweep).

Each configuration is one operation at one shape, built three ways from the same spec: an OpenVINO graph of a single
native op for the NPU (the best graph a generator could emit), the matching torch op on the GPU (cuBLAS or a torch
kernel), and a float32 numpy reference rounded to FP16. Two axes:
  size   one instance grows (vector length, matrix size, softmax row count and length)
  batch  B independent copies of a dataset task's shape in one call (kb_019_relu [4, 65], cuda_tiled_matmul [64, 64],
         triton_fused_softmax [32, 128]); B = 1 is the dataset shape itself
Only numpy is needed at import time: the WSL orchestrator imports torch, the Windows worker imports openvino.
Method: docs/time-energy-measurement.md.
"""
from __future__ import annotations

import math

import numpy as np


def config(name: str, op: str, axis: str, shapes: dict[str, list[int]], instances: int = 1) -> dict:
    output = output_shape(op, shapes)
    flops = {'add': 1, 'relu': 1, 'softmax': 5}.get(op, 0) * math.prod(output)
    if op == 'matmul':
        a, b = shapes['a'], shapes['b']
        flops = 2 * math.prod(a[:-1]) * a[-1] * b[-1]
    moved = 2 * (sum(math.prod(s) for s in shapes.values()) + math.prod(output))  # FP16 bytes in and out
    return {'name': name, 'op': op, 'axis': axis, 'inputs': shapes, 'output': output, 'instances': instances,
            'elements': math.prod(output), 'flops': flops, 'bytes': moved}


def output_shape(op: str, shapes: dict[str, list[int]]) -> list[int]:
    if op == 'matmul':
        return [*shapes['a'][:-1], shapes['b'][-1]]
    return list(next(iter(shapes.values())))


def _sweep() -> list[dict]:
    found = []
    for n in (4096, 1 << 16, 1 << 20, 1 << 24):
        found.append(config(f'add_n{n}', 'add', 'size', {'x': [n], 'y': [n]}))
    for n in (64, 256, 1024, 2048):
        found.append(config(f'matmul_{n}', 'matmul', 'size', {'a': [n, n], 'b': [n, n]}))
    for rows, cols in ((32, 128), (256, 1024), (1024, 4096)):
        found.append(config(f'softmax_{rows}x{cols}', 'softmax', 'size', {'x': [rows, cols]}))
    for b in (1, 64, 4096, 65536):
        found.append(config(f'relu_b{b}', 'relu', 'batch', {'x': [b, 4, 65]}, b))
    for b in (16, 256):
        found.append(config(f'matmul_b{b}', 'matmul', 'batch', {'a': [b, 64, 64], 'b': [b, 64, 64]}, b))
        found.append(config(f'softmax_b{b}', 'softmax', 'batch', {'x': [b, 32, 128]}, b))
    return found


SWEEP = {c['name']: c for c in _sweep()}
# B = 1 of the matmul and softmax batch axes is the size axis's first point.
BATCH_BASE = {'matmul_b16': 'matmul_64', 'matmul_b256': 'matmul_64', 'softmax_b16': 'softmax_32x128',
              'softmax_b256': 'softmax_32x128'}


def inputs(spec: dict, seed: int = 20261101) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape, dtype=np.float32).astype(np.float16) for name, shape in spec['inputs'].items()}


def reference(spec: dict, feed: dict[str, np.ndarray]) -> np.ndarray:
    x = {name: value.astype(np.float32) for name, value in feed.items()}
    if spec['op'] == 'add':
        out = x['x'] + x['y']
    elif spec['op'] == 'relu':
        out = np.maximum(x['x'], 0)
    elif spec['op'] == 'softmax':
        shifted = np.exp(x['x'] - x['x'].max(axis=-1, keepdims=True))
        out = shifted / shifted.sum(axis=-1, keepdims=True)
    else:
        out = np.matmul(x['a'], x['b'])
    return out.astype(np.float16)


def check(spec: dict, got: np.ndarray, want: np.ndarray, tolerance: float = 2 ** -10) -> dict:
    """The FP16 verdict used for the campaign (atol = rtol = 2^-10), reported, not gated: large reductions on FP16
    hardware can exceed it."""
    got, want = np.asarray(got, np.float32).reshape(want.shape), want.astype(np.float32)
    error = np.abs(got - want)
    within = error <= tolerance + tolerance * np.abs(want)
    return {'within_tolerance': float(within.mean()), 'max_abs_error': float(error.max()),
            'max_rel_error': float((error / np.maximum(np.abs(want), 1e-3)).max())}


def openvino_model(spec: dict):
    import openvino as ov
    from openvino import opset13 as ops

    params = {name: ops.parameter(shape, ov.Type.f16, name=name) for name, shape in spec['inputs'].items()}
    for name, param in params.items():
        param.output(0).get_tensor().set_names({name})
    if spec['op'] == 'add':
        node = ops.add(params['x'], params['y'])
    elif spec['op'] == 'relu':
        node = ops.relu(params['x'])
    elif spec['op'] == 'softmax':
        node = ops.softmax(params['x'], -1)
    else:
        node = ops.matmul(params['a'], params['b'], False, False)
    node.output(0).get_tensor().set_names({'out'})
    return ov.Model([ops.result(node)], list(params.values()), spec['name'])


def torch_launch(spec: dict, tensors: dict):
    """A closure that enqueues the op on the current stream and returns the output tensor (outputs are preallocated
    where torch accepts out=, so steady-state calls do not allocate)."""
    import torch

    op = spec['op']
    if op == 'softmax':
        return lambda: torch.softmax(tensors['x'], -1)
    out = torch.empty(spec['output'], dtype=torch.float16, device=next(iter(tensors.values())).device)
    if op == 'add':
        return lambda: torch.add(tensors['x'], tensors['y'], out=out)
    if op == 'relu':
        return lambda: torch.clamp(tensors['x'], min=0, out=out)
    return lambda: torch.matmul(tensors['a'], tensors['b'], out=out)
