"""Compare the exact adapted CUDA, pinned PyTorch and independent oracle on a CUDA GPU."""
from __future__ import annotations

import argparse
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.validate_cuda_references import checked
from npu_agent.models import KernelManifest
from npu_agent.oracles import execute_oracle, validate_goldens
from npu_agent.testcases import generate_cases
from npu_agent.validation import compare_outputs, sha256, write_json


def validate(corpus, output, nvrtc_path):
    import torch

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.init()
    torch.empty(1, device='cuda')
    driver, nvrtc, pointer = C.CDLL('libcuda.so.1'), C.CDLL(str(nvrtc_path)), C.c_void_p
    major, minor = torch.cuda.get_device_capability()
    options = [b'--std=c++17', b'--fmad=false', f'--gpu-architecture=sm_{major}{minor}'.encode()]
    option_array = (C.c_char_p * len(options))(*options)
    report = {'status': 'running', 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'nvrtc_sha256': sha256(nvrtc_path), 'adapter_sha256': sha256(Path(__file__)),
              'compiler_options': [item.decode() for item in options], 'tasks': []}
    write_json(output / 'report.json', report)
    for manifest_path in sorted(corpus.rglob('manifest.json')):
        manifest = KernelManifest.model_validate_json(manifest_path.read_text())
        name, source = manifest.name, manifest_path.parent / manifest.source_file
        destination = output / name
        destination.mkdir()
        upstream = source.parent / 'upstream.py'
        row = {'kernel': name, 'status': 'running', 'source_sha256': sha256(source),
               'manifest_sha256': sha256(manifest_path), 'upstream_sha256': sha256(upstream), 'cases': []}
        report['tasks'].append(row)
        program, module, function = pointer(), pointer(), pointer()
        try:
            if row['upstream_sha256'] != manifest.provenance['upstream_sha256']:
                raise ValueError('pinned upstream source hash mismatch')
            validate_goldens(manifest.oracle.operation)
            spec = importlib.util.spec_from_file_location(name, upstream)
            reference_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference_module)
            model = reference_module.Model(*json.loads(manifest.provenance['upstream_init'])).eval()
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
            checked(driver, 'cuModuleGetFunction', [C.POINTER(pointer), pointer, C.c_char_p], C.byref(function), module, b'kernel')
            out = next(t for t in manifest.tensors if t.direction == 'output')
            for case in generate_cases(manifest):
                cpu = [torch.from_numpy(case.inputs[t.name].copy()) for t in manifest.tensors if t.direction == 'input']
                inputs = [value.cuda() for value in cpu]
                result = torch.full(out.shape, float('nan'), device='cuda', dtype=torch.float32)
                values = [pointer(t.data_ptr()) for t in inputs + [result]]
                arguments = (pointer * len(values))(*(C.cast(C.byref(value), pointer) for value in values))
                torch.cuda.synchronize()
                checked(driver, 'cuLaunchKernel', [pointer] + [C.c_uint] * 7 + [pointer, C.POINTER(pointer), pointer],
                        function, (result.numel()+127)//128, 1, 1, 128, 1, 1, 0, None, arguments, None)
                checked(driver, 'cuCtxSynchronize', [])
                upstream_args = cpu + ([manifest.oracle.parameters['scale']] if manifest.oracle.operation == 'scalar_mul' else [])
                with torch.no_grad():
                    expected = model(*upstream_args).numpy().reshape(out.shape)
                actual = result.cpu().numpy()
                independent = execute_oracle(manifest, case.inputs)
                comparisons = {
                    'cuda_vs_upstream': compare_outputs({out.name: actual}, {out.name: expected}, manifest.tolerance, case.id),
                    'oracle_vs_upstream': compare_outputs(independent, {out.name: expected}, manifest.tolerance, case.id),
                    'cuda_vs_oracle': compare_outputs({out.name: actual}, independent, manifest.tolerance, case.id),
                }
                np.savez(destination / f'{case.id}.npz', cuda=actual, upstream=expected, oracle=independent[out.name])
                row['cases'].append({'id': case.id, 'passed': all(v['correct'] for v in comparisons.values()), 'comparisons': comparisons})
            row['status'] = 'passed' if all(case['passed'] for case in row['cases']) else 'failed'
            row['cubin_sha256'] = sha256(destination / 'source.cubin')
        except Exception as exc:
            row.update(status='blocked', error=str(exc))
        finally:
            if module.value:
                checked(driver, 'cuModuleUnload', [pointer], module)
            if program.value:
                checked(nvrtc, 'nvrtcDestroyProgram', [C.POINTER(pointer)], C.byref(program))
        write_json(destination / 'report.json', row)
        write_json(output / 'report.json', report)
        print(name, row['status'], row.get('error', ''), flush=True)
    report['status'] = 'passed' if len(report['tasks']) == 30 and all(r['status'] == 'passed' for r in report['tasks']) else 'failed'
    write_json(output / 'report.json', report)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'examples/kernelbench')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nvrtc', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(validate(args.corpus, args.output, args.nvrtc))
