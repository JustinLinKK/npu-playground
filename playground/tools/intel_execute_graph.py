"""Trusted graph execution. Expected tensors are never mounted in this process."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import openvino as ov

from npu_agent.models import KernelManifest
from npu_agent.validation import sha256, write_json


def check_ports(model, manifest):
    mappings = []
    for direction, ports in [('input', model.inputs), ('output', model.outputs)]:
        args = {t.name: t for t in manifest.tensors if t.direction == direction}
        if len(ports) != len(args):
            raise ValueError(f'{direction} port count mismatch')
        mapping = {}
        for port in ports:
            names = set(port.get_names()) & set(args)
            if len(names) != 1:
                raise ValueError(f'ambiguous or unnamed {direction} port: {port.get_names()}')
            name = names.pop()
            tensor = args[name]
            if name in mapping:
                raise ValueError('duplicate port identity')
            if tuple(port.get_shape()) != tensor.shape or np.dtype(port.get_element_type().to_dtype()) != np.dtype(tensor.dtype):
                raise ValueError(f'port shape/dtype mismatch: {name}')
            mapping[name] = port
        mappings.append(mapping)
    return mappings


def read_model():
    # Core is fresh, with only the distribution's built-in plugins/extensions.
    core = ov.Core()
    model = core.read_model('/graph/model.xml', '/graph/model.bin')
    if any(op.get_type_name() in ('FrameworkNode', 'PyOp', 'Custom') for op in model.get_ops()):
        raise ValueError('custom graph extensions are unsupported')
    return core, model


def main() -> None:
    started = time.perf_counter()
    manifest = KernelManifest.model_validate_json(Path('/contract/manifest.json').read_text())
    core, model = read_model()
    check_ports(model, manifest)
    compiled = core.compile_model(model, 'CPU', {'INFERENCE_PRECISION_HINT': 'f32'})
    input_ports, output_ports = check_ports(compiled, manifest)
    cases = json.loads(Path('/inputs/cases.json').read_text())
    for case in cases:
        with np.load(f'/inputs/{case}.npz', allow_pickle=False) as archive:
            inputs = {name: archive[name] for name in input_ports}
        outputs = compiled(inputs)
        np.savez(f'/output/{case}.npz', **{name: np.asarray(outputs[port]) for name, port in output_ports.items()})
    write_json(Path('/output/execution.json'), {'model_sha256': sha256(Path('/graph/model.xml')),
        'weights_sha256': sha256(Path('/graph/model.bin')), 'cases_run': len(cases),
        'duration_seconds': time.perf_counter() - started, 'openvino_version': ov.__version__})


if __name__ == '__main__':
    main()
