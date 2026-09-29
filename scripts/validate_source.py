"""Optional original CUDA/Triton execution; never required by the offline suite."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np

from npu_agent.models import KernelManifest, StageResult
from npu_agent.oracles import execute_oracle, validate_goldens
from npu_agent.testcases import generate_cases
from npu_agent.validation import compare_outputs, sha256, write_json


def validate_source(source: Path, manifest: KernelManifest, output: Path) -> StageResult:
    operation = manifest.oracle.operation
    supported = {'cuda': {'vector_add', 'reduce_sum'}, 'triton': {'softmax', 'layer_norm'}}
    if operation not in supported.get(manifest.dialect.value, set()):
        return StageResult(status='unsupported', reason_code='UNSUPPORTED_SOURCE_RUNTIME',
                           message=f'{manifest.dialect.value}/{operation} has no original-source launch adapter')
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if manifest.dialect.value == 'cuda':
            import cupy as cp
            if not cp.cuda.runtime.getDeviceCount():
                raise ImportError('no CUDA device')
            module = cp.RawModule(code=source.read_text(), options=('--std=c++17',))
            function = module.get_function(manifest.entrypoint)
            engine = f'cuda-cupy-{cp.__version__}'
            def run(inputs):
                arrays = [cp.asarray(inputs[t.name]) for t in manifest.tensors if t.direction == 'input']
                if operation == 'vector_add':
                    result = cp.empty_like(arrays[0])
                    n = arrays[0].size
                    function(((n + 255) // 256,), (256,), (*arrays, result, np.int32(n)))
                else:
                    current = arrays[0]
                    # The manifest requires the complete reduction, not block partials.
                    while current.size > 1:
                        blocks = (current.size + 511) // 512
                        result = cp.empty(blocks, dtype=cp.float32)
                        function((blocks,), (256,), (current, result, np.int32(current.size)), shared_mem=256 * 4)
                        current = result
                    result = current
                cp.cuda.runtime.deviceSynchronize()
                return cp.asnumpy(result)
        else:
            import torch
            import triton
            if not torch.cuda.is_available():
                raise ImportError('no CUDA device')
            spec = importlib.util.spec_from_file_location('original_kernel', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            function = getattr(module, manifest.entrypoint)
            engine = f'triton-{triton.__version__}'
            def run(inputs):
                arrays = [torch.from_numpy(inputs[t.name]).cuda() for t in manifest.tensors if t.direction == 'input']
                x = arrays[0]
                if x.ndim != 2 or x.dtype != torch.float32:
                    raise ValueError('Triton source harness supports rank-two float32 inputs')
                rows, cols = x.shape
                result = torch.empty_like(x)
                if operation == 'softmax':
                    function[(rows,)](result, x, x.stride(0), result.stride(0), rows, cols, triton.next_power_of_2(cols))
                else:
                    function[(rows,)](result, x, arrays[1], arrays[2], cols, manifest.oracle.parameters['epsilon'], triton.next_power_of_2(cols))
                torch.cuda.synchronize()
                return result.cpu().numpy()
        validate_goldens(operation)
        comparisons = []
        name = next(t.name for t in manifest.tensors if t.direction == 'output')
        for case in generate_cases(manifest):
            actual = {name: run(case.inputs)}
            np.savez(output / f'{case.id}.npz', **actual)
            comparisons.append(compare_outputs(actual, execute_oracle(manifest, case.inputs), manifest.tolerance, case.id))
        correct = all(c['correct'] for c in comparisons)
        return StageResult(status='passed' if correct else 'failed', correct=correct, engine=engine,
            representation='original_gpu_source', cases_run=len(comparisons), duration_seconds=time.perf_counter() - started,
            error_category=None if correct else 'numerical_mismatch', details={'source_sha256': sha256(source),
            'reference_origin': 'source_runtime', 'comparisons': comparisons})
    except ImportError as exc:
        return StageResult(status='blocked', reason_code='SOURCE_RUNTIME_UNAVAILABLE', message=str(exc),
                           duration_seconds=time.perf_counter() - started)
    except Exception as exc:
        return StageResult(status='failed', error_category='source_runtime', message=str(exc),
                           duration_seconds=time.perf_counter() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/source-validation'))
    args = parser.parse_args()
    result = validate_source(args.source, KernelManifest.model_validate_json(args.manifest.read_text()), args.output)
    write_json(args.output / 'report.json', result.model_dump(mode='json'))
    print(result.model_dump_json(indent=2))
    raise SystemExit(0 if result.status == 'passed' else 3 if result.status in ('blocked', 'unsupported') else 1)
