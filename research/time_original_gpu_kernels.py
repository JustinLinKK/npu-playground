"""Time a prepared Intel FP16 dataset's original CUDA, HIP and Triton sources on a local NVIDIA GPU and compare them
with the physical-NPU latencies of scripts/replay_intel_local_npu.py --timing.

Sources run unmodified at their original dtypes (float32, except the tiled matmul and fused attention, which are
float16) on the dataset's stored inputs cast to those dtypes, and must match the stored references on all twelve
cases. HIP sources are compiled as CUDA by NVRTC. Each kernel is timed on its first case, like the NPU, three ways:
CUDA events around the launch (device), host time of launch plus synchronize, and host time of input upload, launch
and output download (end to end, the closest analogue of the NPU's synchronous InferRequest.infer).

Needs a CUDA environment separate from the project venv, for example:
    uv venv --python 3.13 ~/.cache/npu-playground-gpu-venv
    uv pip install --python ~/.cache/npu-playground-gpu-venv/bin/python torch cupy-cuda13x numpy ziglang
Triton builds a C launcher on first use; without a system compiler, point CC at a `python -m ziglang cc` wrapper.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from types import SimpleNamespace

import numpy as np

BLOCK = 256
TOLERANCE = 2.0 ** -10  # the FP16 verdict's rtol = atol, applied to the stored FP16-contract references
# Grid and block of the classic sources, from the shapes their indexing assumes; KernelBench adaptations use one
# thread per output element (if(i<N)) and BLOCK threads per block.
CLASSIC_LAUNCH = {'cuda_vector_add': ((16,), (256,)), 'cuda_tiled_transpose': ((3, 2), (32, 32)),
                  'cuda_tiled_matmul': ((4, 4), (16, 16)), 'hip_gamma_correction': ((12,), (256,)),
                  'hip_image_convolution': ((2, 2), (16, 16)), 'hip_moving_average': ((16,), (256,))}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def stats(samples: list[float]) -> dict:
    return {'p50_us': float(np.percentile(samples, 50)) * 1e3, 'p95_us': float(np.percentile(samples, 95)) * 1e3,
            'min_us': float(min(samples)) * 1e3}


def cuda_task(folder: Path, manifest: dict, inputs: list):
    import cupy as cp

    code = (folder / manifest['source_file']).read_text()
    if 'half' in code and 'cuda_fp16.h' not in code:
        code = '#include <cuda_fp16.h>\n' + code  # nvcc includes it implicitly; NVRTC does not
    function = cp.RawModule(code=code, options=('--std=c++17',)).get_function(manifest['entrypoint'])
    scalars = [np.dtype(s['dtype']).type(s['value']) for s in manifest.get('scalars', [])]
    output = next(t for t in manifest['tensors'] if t['direction'] == 'output')
    if manifest['name'] == 'cuda_reduce_sum':
        # Block partials are reduced again until one value remains, as in scripts/validate_source.py.
        sizes = [inputs[0].size]
        while sizes[-1] > 1:
            sizes.append((sizes[-1] + 2 * BLOCK - 1) // (2 * BLOCK))
        partials = [cp.empty(size, dtype=cp.float32) for size in sizes[1:]]

        def launch():
            current = inputs[0]
            for buffer in partials:
                function((buffer.size,), (BLOCK,), (current, buffer, np.int32(current.size)), shared_mem=BLOCK * 4)
                current = buffer
            return current

        return launch, f'{len(partials)} launches, block {BLOCK}'
    result = cp.empty(output['shape'], dtype=output['dtype'])
    if manifest['name'] in CLASSIC_LAUNCH:
        grid, block = CLASSIC_LAUNCH[manifest['name']]
    else:
        count = int(re.search(r'if\(i<(\d+)\)', code).group(1))
        grid, block = ((count + BLOCK - 1) // BLOCK,), (BLOCK,)

    def launch():
        function(grid, block, (*inputs, result, *scalars))
        return result

    return launch, f'grid {grid}, block {block}'


def triton_task(folder: Path, manifest: dict, inputs: list):
    import torch
    import triton

    spec = importlib.util.spec_from_file_location('original_' + manifest['name'], folder / manifest['source_file'])
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True  # a __pycache__ beside the source would break the dataset's file inventory
    spec.loader.exec_module(module)
    function = getattr(module, manifest['entrypoint'])
    operation, parameters = manifest['oracle']['operation'], manifest['oracle']['parameters']
    result = torch.empty_like(inputs[0])
    if operation == 'attention':
        # The reduced kernel handles one (sequence, head_dim) head and has no program id: one launch per head.
        batch, heads, sequence, head_dim = inputs[0].shape
        views = [[tensor[b, h] for tensor in (result, *inputs)] for b in range(batch) for h in range(heads)]

        def launch():
            for view in views:
                function[(1,)](*view, sequence, head_dim, parameters['scale'])
            return result

        return launch, f'{len(views)} launches, grid (1,)'
    rows, columns = inputs[0].shape
    block = triton.next_power_of_2(columns)
    if operation == 'softmax':
        arguments = (result, inputs[0], inputs[0].stride(0), result.stride(0), rows, columns, block)
    else:
        arguments = (result, *inputs, columns, parameters['epsilon'], block)

    def launch():
        function[(rows,)](*arguments)
        return result

    return launch, f'grid ({rows},), BLOCK {block}'


def prepare(folder: Path, cases: list[str]) -> SimpleNamespace:
    """Compile an original source, check it on every stored case and leave the first case's inputs on the device.
    launch() enqueues the kernel on the current stream (CuPy's, or torch's for Triton) and returns its output."""
    import cupy as cp
    import torch

    manifest = json.loads((folder / 'original-manifest.json').read_text())
    tensors = [t for t in manifest['tensors'] if t['direction'] == 'input']
    output = next(t['name'] for t in manifest['tensors'] if t['direction'] == 'output')
    triton = manifest['dialect'] == 'triton'
    if triton:
        device = [torch.empty(t['shape'], dtype=getattr(torch, t['dtype']), device='cuda') for t in tensors]
        upload = lambda arrays, host: [d.copy_(torch.from_numpy(h)) for d, h in zip(arrays, host)]
        download, synchronize = (lambda value: value.cpu().numpy()), torch.cuda.synchronize
        event = lambda: torch.cuda.Event(enable_timing=True)
        elapsed = lambda start, end: start.elapsed_time(end)
        launch, launch_shape = triton_task(folder, manifest, device)
    else:
        device = [cp.empty(t['shape'], dtype=t['dtype']) for t in tensors]
        upload = lambda arrays, host: [d.set(h) for d, h in zip(arrays, host)]
        download, synchronize = (lambda value: value.get()), cp.cuda.Device().synchronize
        event = cp.cuda.Event
        elapsed = cp.cuda.get_elapsed_time
        launch, launch_shape = cuda_task(folder, manifest, device)

    def host_inputs(case):
        stored = load(folder / 'inputs' / f'{case}.npz')
        return [np.ascontiguousarray(stored[t['name']].astype(t['dtype'])) for t in tensors]

    worst, passed = 0.0, 0
    for case in cases:
        upload(device, host_inputs(case))
        actual = download(launch()).astype(np.float64)
        expected = load(folder / 'expected' / f'{case}.npz')[output].astype(np.float64)
        error = np.abs(actual.reshape(expected.shape) - expected)
        passed += bool(np.all(error <= TOLERANCE + TOLERANCE * np.abs(expected)))
        worst = max(worst, float(np.max(error / (TOLERANCE + TOLERANCE * np.abs(expected)))))
    first = host_inputs(cases[0])
    upload(device, first)
    synchronize()
    return SimpleNamespace(manifest=manifest, triton=triton, launch=launch, launch_shape=launch_shape,
                           upload=lambda host: upload(device, host), download=download, synchronize=synchronize,
                           event=event, elapsed=elapsed, first=first, passed=passed, cases_run=len(cases), worst=worst)


def measure(folder: Path, cases: list[str], warmup: int, runs: int) -> dict:
    task = prepare(folder, cases)

    def on_device():
        start, end = task.event(), task.event()
        start.record()
        task.launch()
        end.record()
        end.synchronize()
        return task.elapsed(start, end)

    def launch_and_sync():
        started = time.perf_counter()
        task.launch()
        task.synchronize()
        return (time.perf_counter() - started) * 1e3

    def end_to_end():
        started = time.perf_counter()
        task.upload(task.first)
        task.download(task.launch())
        return (time.perf_counter() - started) * 1e3

    timings = {}
    for name, step in (('device', on_device), ('launch_sync', launch_and_sync), ('end_to_end', end_to_end)):
        for _ in range(warmup):
            step()
        timings[name] = stats([step() for _ in range(runs)])
    manifest = task.manifest
    dtypes = sorted({t['dtype'] for t in manifest['tensors']})
    return {'kernel': folder.name, 'dialect': manifest['dialect'], 'dtypes': dtypes, 'launch': task.launch_shape,
            'cases_passed': task.passed, 'cases_run': task.cases_run, 'correct': task.passed == task.cases_run,
            'max_error_over_bound': task.worst, **timings}


def environment() -> dict:
    import cupy as cp
    import torch
    import triton

    return {'gpu': torch.cuda.get_device_name(0), 'capability': '.'.join(map(str, torch.cuda.get_device_capability(0))),
            'cuda_driver': cp.cuda.runtime.driverGetVersion(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'cupy': cp.__version__, 'triton': triton.__version__, 'platform': platform.platform()}


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def compare(rows: list[dict], npu: dict) -> list[dict]:
    tasks = {task['kernel']: task for task in npu['tasks']}
    joined = []
    for row in rows:
        task = tasks.get(row['kernel'], {})
        p50 = task.get('latency_p50_ms')
        npu_us = p50 * 1e3 if p50 is not None else None
        gpu = {name: row[name]['p50_us'] if name in row else None for name in ('end_to_end', 'launch_sync', 'device')}
        joined.append({'kernel': row['kernel'], 'dialect': row['dialect'], 'original_dtypes': '/'.join(row['dtypes']),
                       'gpu_correct': row['correct'], 'gpu_error': row.get('error'), 'npu_fp16_correct': task.get('fp16_correct'),
                       'npu_p50_us': npu_us, 'gpu_end_to_end_p50_us': gpu['end_to_end'],
                       'gpu_launch_sync_p50_us': gpu['launch_sync'], 'gpu_device_p50_us': gpu['device'],
                       'npu_over_gpu_end_to_end': npu_us / gpu['end_to_end'] if npu_us and gpu['end_to_end'] else None,
                       'npu_over_gpu_device': npu_us / gpu['device'] if npu_us and gpu['device'] else None})
    return joined


def write_report(path: Path, joined: list[dict], env: dict, npu: dict, warmup: int, runs: int) -> dict:
    both = [row for row in joined if row['gpu_correct'] and row['npu_fp16_correct'] and row['npu_p50_us']]
    summary = {'tasks_compared': len(both), 'tasks': len(joined),
               'geomean_npu_p50_us': geomean([row['npu_p50_us'] for row in both]),
               'geomean_gpu_end_to_end_p50_us': geomean([row['gpu_end_to_end_p50_us'] for row in both]),
               'geomean_gpu_launch_sync_p50_us': geomean([row['gpu_launch_sync_p50_us'] for row in both]),
               'geomean_gpu_device_p50_us': geomean([row['gpu_device_p50_us'] for row in both]),
               'geomean_npu_over_gpu_end_to_end': geomean([row['npu_over_gpu_end_to_end'] for row in both]),
               'geomean_npu_over_gpu_device': geomean([row['npu_over_gpu_device'] for row in both])}

    def number(value, digits=1, unit=''):
        return 'n/a' if value is None else f'{value:.{digits}f}{unit}'

    timing = npu.get('timing') or {}
    lines = ['# Original GPU kernels vs generated NPU kernels', '',
             f'**Over the {summary["tasks_compared"]}/{summary["tasks"]} tasks correct on both devices, the NPU kernel\'s '
             f'p50 latency is {number(summary["geomean_npu_over_gpu_end_to_end"])}× the original GPU kernel\'s end-to-end '
             f'latency and {number(summary["geomean_npu_over_gpu_device"], 0)}× its device time (geometric means).**', '',
             f'GPU: {env["gpu"]} (sm_{env["capability"].replace(".", "")}), torch {env["torch"]}, CuPy {env["cupy"]}, '
             f'Triton {env["triton"]}, under WSL2. NPU: Intel NPU 3720 through native Windows OpenVINO, from '
             f'`{npu["campaign"]}`.', '',
             '## What each number measures', '',
             f'Both devices ran {timing.get("warmup_runs", warmup)} warmup and {timing.get("timed_runs", runs)} timed runs '
             'of the first input case at the dataset\'s reduced shapes; tables show medians in microseconds.', '',
             '- **NPU p50**: host `perf_counter` around a synchronous `InferRequest.infer(feed)` of the generated '
             'OpenVINO blob: input copies, dispatch, execution and output copies.',
             '- **GPU end to end**: host time to upload the inputs, launch the original kernel and download the output. '
             'This is the closest analogue of the NPU number.',
             '- **GPU launch+sync**: host time to launch and synchronize, with data already on the device.',
             '- **GPU device**: CUDA events around the launch, i.e. kernel execution only. The NPU exposes no comparable '
             'device-only timer for these blobs.', '',
             '**Read with care.** At these sizes (tens to a few thousand elements) both devices are dominated by fixed '
             'dispatch cost, not arithmetic, so the ratios compare software stacks and submission paths more than compute. '
             'The original sources compute in float32 (FP16 for the tiled matmul and fused attention), while the NPU '
             'kernels compute in FP16. The GPU is a discrete 250 W card; host-side GPU numbers include WSL2 '
             'paravirtualization overhead and Python dispatch (CuPy, or Triton\'s launcher, which is slower). Multi-launch '
             'tasks (reduce_sum, fused attention) pay one launch per pass or head. The NPU starts each executor process '
             'in a low-power state and reaches steady latency only after a few hundred inferences: with 20 warmups its '
             'p50 was about twice the steady value, so compare replays with at least 1000 warmups.', '',
             '| Task | Source | Original dtype | NPU p50 | GPU end to end | GPU launch+sync | GPU device | NPU / GPU e2e | NPU / GPU device |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in sorted(joined, key=lambda row: row['kernel']):
        flags = ('' if row['gpu_correct'] else ' (original does not run)' if row['gpu_error'] else ' (GPU incorrect)') + \
                ('' if row['npu_fp16_correct'] else ' (NPU FP16 fail)')
        lines.append(f'| {row["kernel"]}{flags} | {row["dialect"]} | {row["original_dtypes"]} | {number(row["npu_p50_us"])} | '
                     f'{number(row["gpu_end_to_end_p50_us"])} | {number(row["gpu_launch_sync_p50_us"])} | '
                     f'{number(row["gpu_device_p50_us"], 2)} | {number(row["npu_over_gpu_end_to_end"], 1, "×")} | '
                     f'{number(row["npu_over_gpu_device"], 0, "×")} |')
    lines += ['', f'| Geometric mean over {summary["tasks_compared"]} tasks | | | {number(summary["geomean_npu_p50_us"])} | '
              f'{number(summary["geomean_gpu_end_to_end_p50_us"])} | {number(summary["geomean_gpu_launch_sync_p50_us"])} | '
              f'{number(summary["geomean_gpu_device_p50_us"], 2)} | {number(summary["geomean_npu_over_gpu_end_to_end"], 1, "×")} | '
              f'{number(summary["geomean_npu_over_gpu_device"], 0, "×")} |', '',
              'Data: [comparison.csv](comparison.csv), [gpu-original-timing.json](gpu-original-timing.json).', '']
    errors = [row for row in joined if row['gpu_error']]
    if errors:
        lines[-2:-2] = ['**Original sources that do not run as written** (excluded; not patched):', '',
                        *[f'- `{row["kernel"]}`: {row["gpu_error"]}' for row in errors], '']
    path.write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/intel-fp16-baseline/intel-fp16-baseline-20261020'))
    parser.add_argument('--npu-replay', type=Path, required=True, help='summary.json of a replay run with --timing')
    parser.add_argument('--output', type=Path, required=True, help='new directory')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--runs', type=int, default=200)
    args = parser.parse_args()
    npu = json.loads(args.npu_replay.read_text())
    if not npu.get('timing'):
        parser.error('the NPU replay was run without --timing')
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = json.loads((args.dataset / 'dataset.json').read_text())
    rows = []
    for task in dataset['tasks']:
        folder = args.dataset / task['kernel']
        try:
            row = measure(folder, task['cases'], args.warmup, args.runs)
        except Exception as exc:  # an original source that does not compile or launch is reported, never patched
            manifest = json.loads((folder / 'original-manifest.json').read_text())
            row = {'kernel': task['kernel'], 'dialect': manifest['dialect'], 'dtypes': sorted({t['dtype'] for t in manifest['tensors']}),
                   'correct': False, 'error': f'{type(exc).__name__}: {str(exc).strip().splitlines()[-1]}'}
            print(f"{row['kernel']:28s} ERR {row['error']}")
        else:
            print(f"{row['kernel']:28s} {'ok ' if row['correct'] else 'BAD'} device {row['device']['p50_us']:7.2f} us  "
                  f"launch+sync {row['launch_sync']['p50_us']:7.1f} us  end-to-end {row['end_to_end']['p50_us']:7.1f} us")
        rows.append(row)
    env = environment()
    (args.output / 'gpu-original-timing.json').write_text(json.dumps(
        {'environment': env, 'dataset': str(args.dataset.resolve()), 'warmup_runs': args.warmup, 'timed_runs': args.runs,
         'tolerance': {'rtol': TOLERANCE, 'atol': TOLERANCE}, 'tasks': rows}, indent=2))
    joined = compare(rows, npu)
    with (args.output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    summary = write_report(args.output / 'COMPARISON.md', joined, env, npu, args.warmup, args.runs)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
