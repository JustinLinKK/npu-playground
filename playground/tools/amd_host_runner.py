from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from npu_agent.sim.amd_mlir_frontend import DataflowProgram, UnsupportedMLIR
from npu_agent.sim.amd_dataflow import simulate
from npu_agent.sim.amd_host import HostKernel, build_host
from npu_agent.validation import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['host', 'dataflow'], required=True)
    args = parser.parse_args()
    program = DataflowProgram.model_validate_json(Path('/program/program.json').read_text())
    source = Path('/candidate/kernel.cc')
    if sha256(source) != program.source_sha256 or sha256(Path('/design/design.mlir')) != program.mlir_sha256:
        raise ValueError('source/design identity mismatch')
    manifest = json.loads(Path('/contract/manifest.json').read_text())
    cases = json.loads(Path('/inputs/cases.json').read_text())
    binary = build_host(source, program.kernels, Path('/output'), Path('/tools/host_compat'))
    kernel = HostKernel(binary, program.kernels, Path('/output/host-sanitizer.log'))
    traces = []
    try:
        for case in cases:
            with np.load(f'/inputs/{case}.npz', allow_pickle=False) as archive:
                inputs = {name: archive[name] for name in archive.files}
            for seed in ([0] if args.mode == 'host' else [1, 7, 31]):
                outputs, trace = simulate(program, inputs, manifest['tensors'], kernel, seed=seed)
                traces.append({'case': case, **trace})
                # Save every tested schedule for independent supervisor comparison.
                suffix = '' if seed in (0, 1) else f'-schedule-{seed}'
                np.savez(f'/output/{case}{suffix}.npz', **outputs)
        write_json(Path('/output/execution.json'), {'source_sha256': program.source_sha256,
            'mlir_sha256': program.mlir_sha256, 'traces': traces, 'representation': 'kernel_source',
            'topology': 'emitted_mlir', 'binary_execution': False})
    finally:
        kernel.close()


if __name__ == '__main__':
    try:
        main()
    except UnsupportedMLIR as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)
