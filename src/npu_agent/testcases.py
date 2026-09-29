from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import KernelManifest
from .validation import check_contract, sha256, write_json


@dataclass
class TestCase:
    id: str
    inputs: dict[str, Any]


def generate_cases(manifest: KernelManifest) -> list[TestCase]:
    check_contract(manifest)
    cases = []
    families = [f'normal-{manifest.seed + i}' for i in range(5)] + [
        'zeros', 'positive', 'mixed', 'alternating', 'cancellation', 'dynamic_range', 'operation_boundary',
    ]
    for family in families:
        rng = np.random.default_rng(int(family.split('-')[1]) if family.startswith('normal-') else manifest.seed)
        inputs = {}
        for t_index, tensor in enumerate(manifest.tensors):
            if tensor.direction == 'output':
                continue
            count = int(np.prod(tensor.shape))
            ramp = np.arange(count)
            if family.startswith('normal-'):
                values = rng.normal(0, .5, count) if tensor.dtype.startswith('float') else rng.integers(-3, 4, count)
            elif family == 'zeros':
                values = np.zeros(count)
            elif family == 'positive':
                values = np.full(count, 2)
            elif family == 'mixed':
                values = (ramp + t_index) % 7 - 3
            elif family == 'alternating':
                values = np.where((ramp + t_index) % 2, -1, 1)
            elif family == 'cancellation':
                values = np.resize([4., .0001, -4., -.0001], count)
            elif family == 'dynamic_range':
                values = np.resize([16., .0625, -16., -.0625], count)
            else:
                values = (ramp % 31) / 16
                op = manifest.oracle.operation
                if op == 'transpose':
                    values = ramp
                elif op == 'softmax':
                    values = np.where(ramp % tensor.shape[-1] == 0, 40., -20.)
                elif op == 'layer_norm':
                    values = 1 + (ramp % 3) * 1e-6
                elif op == 'conv2d':
                    values = np.zeros(count)
                    values[count // 2] = 1
                elif op == 'gamma_correction':
                    values = np.resize([-1., 0., .5, 1., 2.], count)
                elif op in ('sigmoid', 'silu'):
                    values = np.resize([-1000., -100., 0., 100., 1000.], count)
            inputs[tensor.name] = np.asarray(values, dtype=tensor.dtype).reshape(tensor.shape)
        inputs.update({s.name: s.value for s in manifest.scalars})
        cases.append(TestCase(family, inputs))
    return cases


def serialize_cases(cases: list[TestCase], destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for case in cases:
        path = destination / f'{case.id}.npz'
        np.savez(path, **{k: v for k, v in case.inputs.items() if isinstance(v, np.ndarray)})
        hashes[case.id] = sha256(path)
    write_json(destination / 'cases.json', hashes)
    return hashes


def tail_manifests(manifest: KernelManifest, width: int) -> list[KernelManifest]:
    if width < 2 or manifest.oracle.operation not in ('vector_add', 'scalar_mul', 'reduce_sum', 'sigmoid', 'silu'):
        raise ValueError('tail variants require a one-dimensional elementwise or reduction contract')
    result = []
    for size in (width - 1, width, width + 1, 2 * width + 3):
        payload = manifest.model_dump(mode='json')
        payload['name'] += f'_tail_{size}'
        for tensor in payload['tensors']:
            if len(tensor['shape']) != 1:
                raise ValueError('tail variants require rank one')
            if tensor['direction'] == 'input' or manifest.oracle.operation != 'reduce_sum':
                tensor['shape'] = [size]
        for scalar in payload['scalars']:
            if scalar['name'] == 'n':
                scalar['value'] = size
        result.append(KernelManifest.model_validate(payload))
    return result
