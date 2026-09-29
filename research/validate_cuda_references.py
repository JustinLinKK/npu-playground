"""Execute the two selected CUDA references without modifying the frozen source adapter.

Uses NVIDIA's NVRTC CUBIN workflow: https://docs.nvidia.com/cuda/archive/13.0.0/nvrtc/index.html
Requires an existing CUDA-enabled torch installation and an explicit NVRTC library path.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path
import time

import numpy as np

from npu_agent.models import KernelManifest
from npu_agent.oracles import execute_oracle, validate_goldens
from npu_agent.testcases import generate_cases
from npu_agent.validation import compare_outputs, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def checked(library, name, types, *args):
    function = getattr(library, name)
    function.argtypes = types
    function.restype = C.c_int
    status = function(*args)
    if status:
        raise RuntimeError(f'{name} failed with status {status}')


def validate(selection: Path, output: Path, nvrtc_path: Path) -> None:
    import torch

    if output.exists():
        raise ValueError('use a new output directory to preserve previous evidence')
    if not torch.cuda.is_available():
        raise ValueError('CUDA device unavailable')
    references = json.loads(selection.read_text())['references']
    selected = {}
    for path in (ROOT / 'examples/classic').glob('*/manifest.json'):
        manifest = KernelManifest.model_validate_json(path.read_text())
        selected[manifest.name] = (path.parent / manifest.source_file, path, manifest)
    for reference in references:
        name = reference['kernel']
        if name not in ('cuda_vector_add', 'cuda_tiled_transpose'):
            raise ValueError('unsupported source launch adapter')
        source, manifest_path, _ = selected[name]
        if sha256(source) != reference['source_sha256'] or sha256(manifest_path) != reference['manifest_sha256']:
            raise ValueError('source or manifest differs from the selected reference')
    torch.cuda.init()
    torch.empty(1, device='cuda')  # Establish torch's primary context before driver module loading.
    driver = C.CDLL('libcuda.so.1')
    nvrtc = C.CDLL(str(nvrtc_path.resolve()))
    pointer = C.c_void_p
    major, minor = torch.cuda.get_device_capability()
    options = [b'--std=c++17', f'--gpu-architecture=sm_{major}{minor}'.encode()]
    option_array = (C.c_char_p * len(options))(*options)
    output.mkdir(parents=True)
    write_json(output / 'environment.json', {
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'nvrtc_library': str(nvrtc_path.resolve()), 'nvrtc_sha256': sha256(nvrtc_path),
        'adapter_sha256': sha256(Path(__file__)), 'selection_sha256': sha256(selection),
        'compiler_options': [item.decode() for item in options],
        'evidence_scope': 'Original GPU source on finite manifest inputs; no NPU execution or universal equivalence claim.'})
    for reference in references:
        started = time.perf_counter()
        source, manifest_path, _ = selected[reference['kernel']]
        manifest = KernelManifest.model_validate_json(manifest_path.read_text())
        destination = output / reference['kernel']
        destination.mkdir()
        program, module, function = pointer(), pointer(), pointer()
        comparisons = []
        try:
            checked(nvrtc, 'nvrtcCreateProgram', [C.POINTER(pointer), C.c_char_p, C.c_char_p, C.c_int, pointer, pointer],
                    C.byref(program), source.read_bytes(), source.name.encode(), 0, None, None)
            try:
                checked(nvrtc, 'nvrtcCompileProgram', [pointer, C.c_int, C.POINTER(C.c_char_p)], program, len(options), option_array)
            finally:
                size = C.c_size_t()
                checked(nvrtc, 'nvrtcGetProgramLogSize', [pointer, C.POINTER(C.c_size_t)], program, C.byref(size))
                log = C.create_string_buffer(size.value)
                checked(nvrtc, 'nvrtcGetProgramLog', [pointer, pointer], program, log)
                (destination / 'compiler.log').write_bytes(log.value)
            size = C.c_size_t()
            checked(nvrtc, 'nvrtcGetCUBINSize', [pointer, C.POINTER(C.c_size_t)], program, C.byref(size))
            cubin = C.create_string_buffer(size.value)
            checked(nvrtc, 'nvrtcGetCUBIN', [pointer, pointer], program, cubin)
            (destination / 'source.cubin').write_bytes(cubin.raw)
            checked(driver, 'cuModuleLoadData', [C.POINTER(pointer), pointer], C.byref(module), cubin)
            checked(driver, 'cuModuleGetFunction', [C.POINTER(pointer), pointer, C.c_char_p],
                    C.byref(function), module, manifest.entrypoint.encode())
            validate_goldens(manifest.oracle.operation)
            output_spec = next(t for t in manifest.tensors if t.direction == 'output')
            for case in generate_cases(manifest):
                inputs = [torch.from_numpy(case.inputs[t.name]).cuda() for t in manifest.tensors if t.direction == 'input']
                result = torch.full(output_spec.shape, float('nan'), dtype=torch.float32, device='cuda')
                values = [pointer(t.data_ptr()) for t in inputs + [result]]
                if reference['kernel'] == 'cuda_vector_add':
                    values.append(C.c_int(case.inputs['n']))
                    grid, block = (16, 1, 1), (256, 1, 1)
                else:
                    grid, block = (3, 2, 1), (32, 32, 1)
                arguments = (pointer * len(values))(*(C.cast(C.byref(value), pointer) for value in values))
                torch.cuda.synchronize()
                checked(driver, 'cuLaunchKernel', [pointer] + [C.c_uint] * 7 + [pointer, C.POINTER(pointer), pointer],
                        function, *grid, *block, 0, None, arguments, None)
                checked(driver, 'cuCtxSynchronize', [])
                actual = {output_spec.name: result.cpu().numpy()}
                np.savez(destination / f'{case.id}.npz', **actual)
                comparisons.append(compare_outputs(actual, execute_oracle(manifest, case.inputs), manifest.tolerance, case.id))
            correct = all(item['correct'] for item in comparisons)
            write_json(destination / 'report.json', {
                'status': 'passed' if correct else 'failed', 'correct': correct,
                'representation': 'original_gpu_source', 'engine': 'nvrtc-cuda-driver-torch',
                'cases_run': len(comparisons), 'duration_seconds': time.perf_counter() - started,
                'details': {'source_sha256': sha256(source), 'manifest_sha256': sha256(manifest_path),
                            'cubin_sha256': sha256(destination / 'source.cubin'), 'comparisons': comparisons}})
        except Exception as exc:
            write_json(destination / 'report.json', {'status': 'blocked', 'representation': 'original_gpu_source',
                'message': str(exc), 'cases_run': len(comparisons), 'duration_seconds': time.perf_counter() - started,
                'details': {'source_sha256': sha256(source), 'manifest_sha256': sha256(manifest_path)}})
            raise
        finally:
            if module.value:
                checked(driver, 'cuModuleUnload', [pointer], module)
            if program.value:
                checked(nvrtc, 'nvrtcDestroyProgram', [C.POINTER(pointer)], C.byref(program))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=ROOT / 'configs/research-next-steps-v1/diagnostic-selection.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nvrtc', type=Path, required=True)
    args = parser.parse_args()
    validate(args.selection, args.output, args.nvrtc)
