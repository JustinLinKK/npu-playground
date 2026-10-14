"""Re-run an Intel NPU baseline campaign's compiled blobs on the local NPU, saving outputs and both verdicts.

Model-free: it replays the saved execution requests of every compiled attempt. The campaign is never rewritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from npu_agent.executors import FP16_VERDICT, execute_target
from npu_agent.models import ExecutionRequest
from npu_agent.validation import sha256, write_json


def inside(path: Path, root: Path) -> Path:
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f'missing or unsafe campaign artifact: {path}')
    return path


def half_ulps(stage) -> float | None:
    values = [output['max_error_half_ulps'] for comparison in stage.details.get('fp16_comparisons', [])
              for output in comparison.get('outputs', {}).values() if output.get('max_error_half_ulps') is not None]
    return max(values) if values else None


def replay(evaluation: Path, output: Path, runner: list[str], accepted_only: bool = False) -> dict:
    campaign = json.loads((evaluation / 'campaign.json').read_text())
    if campaign['config']['protocol'] != 'intel-npu-baseline-v1':
        raise ValueError('Intel NPU baseline campaign required')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in campaign['cases']:
        report_path = evaluation / 'cases' / case['id'] / 'report.json'
        cycles = json.loads(report_path.read_text())['cycles'] if report_path.exists() else []
        for cycle in cycles:
            compiled = cycle.get('compile_result') or {}
            validation = compiled.get('validation') or {}
            stages = validation.get('stages', {})
            if stages.get('target_compile', {}).get('status') != 'passed' or (accepted_only and cycle['status'] != 'passed'):
                continue
            run = inside(Path(compiled['artifacts']['validation.json']), evaluation).parent
            request = ExecutionRequest.model_validate_json(inside(run / 'execution-request.json', evaluation).read_text())
            blob = inside(Path(request.artifacts['compiled.blob']), evaluation)
            if request.target.id != campaign['config']['targets'][0] or not (
                    sha256(blob) == request.artifact_sha256['compiled.blob'] == stages['target_compile']['artifacts']['compiled.blob']):
                raise ValueError(f'compiled blob or target changed since compilation: {case["kernel"]} cycle {cycle["number"]}')
            stage = execute_target(request, runner, output / case['id'] / f'cycle-{cycle["number"]}')
            recorded = stages.get('target_execution', {})
            rows.append({'kernel': case['kernel'], 'case_id': case['id'], 'cycle': cycle['number'],
                'accepted': cycle['status'] == 'passed', 'host_correct': stages.get('host_execution', {}).get('correct'),
                'status': stage.status, 'engine': stage.engine, 'error_category': stage.error_category,
                'message': stage.message[-500:], 'cases_run': stage.cases_run,
                'manifest_cases_passed': stage.details.get('cases_passed'), 'manifest_correct': stage.correct,
                'recorded_manifest_cases_passed': sum(c.get('correct') is True for c in recorded.get('details', {}).get('comparisons', [])),
                'fp16_cases_passed': stage.details.get('fp16_cases_passed'), 'fp16_correct': stage.details.get('fp16_correct'),
                'fp16_max_half_ulps': half_ulps(stage), 'stage': stage.model_dump(mode='json')})
            write_json(output / 'rows.json', rows)
    executed = [row for row in rows if row['engine'] == 'physical_npu']
    accepted = [row for row in executed if row['accepted']]
    cases = sum(row['cases_run'] for row in accepted)
    summary = {'campaign': str(evaluation), 'runner': runner, 'fp16_verdict': FP16_VERDICT,
        'attempts_replayed': len(rows), 'execution_errors': len(rows) - len(executed),
        'reproduces_recorded_manifest_counts': sum(row['manifest_cases_passed'] == row['recorded_manifest_cases_passed'] for row in executed),
        'accepted_tasks': len(accepted),
        'accepted_tasks_manifest_correct': sum(row['manifest_correct'] is True for row in accepted),
        'accepted_tasks_fp16_correct': sum(row['fp16_correct'] is True for row in accepted),
        'accepted_cases_run': cases,
        'accepted_case_pass_rate_manifest': sum(row['manifest_cases_passed'] or 0 for row in accepted) / cases if cases else None,
        'accepted_case_pass_rate_fp16': sum(row['fp16_cases_passed'] or 0 for row in accepted) / cases if cases else None,
        'cpu_vs_fp16_npu': {key: sum(f"cpu_{'pass' if row['host_correct'] else 'fail'}/npu_fp16_{'pass' if row['fp16_correct'] else 'fail'}" == key
                                     for row in executed)
                            for key in sorted({f"cpu_{'pass' if row['host_correct'] else 'fail'}/npu_fp16_{'pass' if row['fp16_correct'] else 'fail'}"
                                               for row in executed})},
        'tasks': [{key: row[key] for key in ('kernel', 'cycle', 'manifest_cases_passed', 'fp16_cases_passed', 'cases_run',
                                              'fp16_max_half_ulps')} for row in accepted]}
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path, help='RUN_ROOT/evaluation')
    parser.add_argument('--output', type=Path, required=True, help='new directory for outputs and summary.json')
    parser.add_argument('--runner', default=os.environ.get('NPU_AGENT_INTEL_NPU_3720_RUNNER'),
                        help='executor command; defaults to NPU_AGENT_INTEL_NPU_3720_RUNNER')
    parser.add_argument('--accepted-only', action='store_true', help='replay only each task\'s accepted attempt')
    args = parser.parse_args()
    if not args.runner:
        parser.error('set NPU_AGENT_INTEL_NPU_3720_RUNNER or pass --runner')
    result = replay(args.evaluation.resolve(), args.output.resolve(), shlex.split(args.runner), args.accepted_only)
    for task in result['tasks']:
        print(f"{task['kernel']:28s} manifest {task['manifest_cases_passed']:>2}/{task['cases_run']}  "
              f"fp16 {task['fp16_cases_passed']:>2}/{task['cases_run']}  worst {task['fp16_max_half_ulps'] or 0:.1f} half-ulps")
    print(json.dumps({key: value for key, value in result.items() if key not in ('tasks', 'runner', 'fp16_verdict')}, indent=2))
    raise SystemExit(0 if result['execution_errors'] == 0 else 1)
