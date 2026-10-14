"""Rescore saved physical-NPU outputs of an Intel NPU baseline campaign under an FP16 atol/rtol choice.

No device or model calls: it reads the outputs saved beside each attempt (target_execution/<case>.npz),
or those written by scripts/replay_intel_local_npu.py when --outputs points at a replay directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from npu_agent.executors import FP16_ATOL, FP16_RTOL, compare_fp16, fp16_reference
from npu_agent.models import KernelManifest
from npu_agent.testcases import generate_cases
from npu_agent.validation import write_json


def rescore(evaluation: Path, rtol: float, atol: float, outputs: Path | None = None) -> dict:
    campaign = json.loads((evaluation / 'campaign.json').read_text())
    tasks = []
    for case in campaign['cases']:
        report = evaluation / 'cases' / case['id'] / 'report.json'
        cycles = json.loads(report.read_text())['cycles'] if report.exists() else []
        accepted = next((cycle for cycle in cycles if cycle['status'] == 'passed'), None)
        if accepted is None:
            continue
        run = Path(accepted['compile_result']['artifacts']['validation.json']).parent
        saved = (outputs / case['id'] / f'cycle-{accepted["number"]}') if outputs else run / 'target_execution'
        manifest = KernelManifest.model_validate_json((run / 'contract' / 'manifest.json').read_text())
        passed, worst, flushed = 0, 0.0, 0
        for test in generate_cases(manifest):
            inputs = {**test.inputs, **{s.name: s.value for s in manifest.scalars}}
            with np.load(saved / f'{test.id}.npz', allow_pickle=False) as archive:
                actual = {name: archive[name] for name in archive.files}
            result = compare_fp16(actual, fp16_reference(manifest, inputs), manifest.tolerance, test.id, rtol, atol)
            passed += bool(result['correct'])
            for output in result['outputs'].values():
                worst = max(worst, output['max_scaled_error'] if output['max_scaled_error'] is not None else float('inf'))
                flushed += output['flushed_subnormals']
        tasks.append({'kernel': case['kernel'], 'cases_passed': passed, 'cases_run': 12,
                      'max_scaled_error': worst, 'flushed_subnormals': flushed})
    run_cases = sum(task['cases_run'] for task in tasks)
    return {'campaign': str(evaluation), 'rtol': rtol, 'atol': atol, 'accepted_tasks': len(tasks),
            'tasks_passing': sum(task['cases_passed'] == task['cases_run'] for task in tasks),
            'case_pass_rate': sum(task['cases_passed'] for task in tasks) / run_cases if run_cases else None,
            'flushed_subnormals': sum(task['flushed_subnormals'] for task in tasks), 'tasks': tasks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path, help='RUN_ROOT/evaluation')
    parser.add_argument('--rtol', type=float, default=FP16_RTOL)
    parser.add_argument('--atol', type=float, default=FP16_ATOL)
    parser.add_argument('--outputs', type=Path, help='replay directory, for campaigns recorded before outputs were saved')
    parser.add_argument('--output', type=Path, help='write the full result as JSON')
    args = parser.parse_args()
    result = rescore(args.evaluation.resolve(), args.rtol, args.atol, args.outputs.resolve() if args.outputs else None)
    for task in sorted(result['tasks'], key=lambda task: task['cases_passed']):
        if task['cases_passed'] < task['cases_run']:
            print(f"{task['kernel']:28s} {task['cases_passed']:2d}/12 cases  worst error/bound {task['max_scaled_error']:.1f}")
    print(f"rtol {args.rtol:.3g}, atol {args.atol:.3g}: {result['tasks_passing']}/{result['accepted_tasks']} accepted tasks, "
          f"case pass rate {result['case_pass_rate']:.1%}, flushed subnormals {result['flushed_subnormals']}")
    if args.output:
        write_json(args.output, result)
