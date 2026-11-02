"""Time the dataset's original GPU kernels with NVIDIA NVBench (PyPI cuda-bench) in the GPU venv.

NVBench reports a cold GPU time per launch (each launch held behind a blocking kernel, stopping on relative noise)
and a batch GPU time (back-to-back launches until a target amount of GPU time; total time / launches). Kernels are
the same launch closures as research/time_original_gpu_kernels.py, run on NVBench's stream; a task registers only
when it matches the stored references on all cases. Arguments after the known ones go to NVBench, for example:
    python research/nvbench_original_kernels.py --tasks kb_019_relu,cuda_tiled_matmul --json out.json
Method: docs/time-energy-measurement.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))

import time_original_gpu_kernels as gpu_kernels  # noqa: E402

SUMMARIES = {'nv/cold/time/gpu/median': 'cold_gpu_median_s', 'nv/cold/time/gpu/mean': 'cold_gpu_mean_s',
             'nv/cold/time/cpu/median': 'cold_cpu_median_s', 'nv/cold/time/cpu/mean': 'cold_cpu_mean_s',
             'nv/cold/sample_size': 'cold_samples', 'nv/batch/time/gpu/mean': 'batch_gpu_mean_s',
             'nv/batch/sample_size': 'batch_samples'}


def register(kernel: str, folder: Path, cases: list[str]) -> bool:
    import cupy as cp
    import cuda.bench as bench
    import torch

    task = gpu_kernels.prepare(folder, cases)
    if task.passed != task.cases_run:
        print(f'{kernel}: skipped, {task.passed}/{task.cases_run} cases match the references', file=sys.stderr)
        return False

    def benchmark(state):
        handle = state.get_stream()
        stream = cp.cuda.Stream.from_external(handle)

        def launcher(launch):
            if launch.get_stream().addressof() != handle.addressof():
                raise RuntimeError('NVBench launched on a stream other than the state stream')
            task.launch()

        # The current stream is set once around exec: entering a stream context per launch adds microseconds of
        # Python overhead, which batch (back-to-back) times would include.
        with torch.cuda.stream(torch.cuda.ExternalStream(stream.ptr)) if task.triton else stream:
            state.exec(launcher)

    benchmark.__name__ = benchmark.__qualname__ = kernel
    bench.register(benchmark)
    return True


def parse(report: dict) -> dict[str, dict]:
    """NVBench JSON output -> {benchmark name: {summary key: value}} for the first state of each benchmark."""
    results = {}
    for benchmark in report['benchmarks']:
        values = {}
        for summary in benchmark['states'][0]['summaries']:
            if summary['tag'] in SUMMARIES:
                value = next(item['value'] for item in summary['data'] if item['name'] == 'value')
                values[SUMMARIES[summary['tag']]] = float(value)
        results[benchmark['name']] = values
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/intel-fp16-baseline/intel-fp16-baseline-20261020')
    parser.add_argument('--tasks', required=True, help='comma-separated task names')
    args, rest = parser.parse_known_args()
    cases = {task['kernel']: task['cases'] for task in json.loads((args.dataset / 'dataset.json').read_text())['tasks']}
    registered = [kernel for kernel in args.tasks.split(',') if register(kernel, args.dataset / kernel, cases[kernel])]
    if registered:
        import cuda.bench as bench

        bench.run_all_benchmarks([sys.argv[0], *rest])


if __name__ == '__main__':
    main()
