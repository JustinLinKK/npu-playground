"""Prepare/calibrate/run the 40-task XDNA2 or Intel NPU baseline through OpenRouter or direct OpenAI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from npu_agent.experiment import discover_kernels
from npu_agent.models import KernelManifest
from npu_agent.study_v4 import StudyConfig
from npu_agent.validation import sha256, write_json
from npu_agent.xdna2_baseline import classify_attempt
from scripts.prepare_kernelbench_corpus import cuda_body


def calibration_inputs(root):
    corpus = root / 'calibration-inputs'
    if corpus.exists():
        return corpus
    definitions = [
        ('add', [(7,), (7,)], (7,), 'vector_add', {}, 'out[i]=x[i]+b[i];'),
        ('reduce', [(17,)], (1,), 'reduce_sum', {'keepdims': True}, 'float s=0;for(int t=0;t<17;++t)s+=x[t];out[i]=s;'),
        ('transpose', [(3, 5)], (5, 3), 'transpose', {'axes': [1, 0]}, 'out[i]=x[(i%3)*5+i/3];'),
        ('gemv', [(3, 7), (7, 1)], (3, 1), 'kb_matmul', {}, None),
    ]
    for name, shapes, out_shape, op, params, body in definitions:
        folder = corpus / name
        folder.mkdir(parents=True)
        tensors = [dict(name='x' if i == 0 else 'b', direction='input', dtype='float32', shape=list(s)) for i, s in enumerate(shapes)]
        tensors.append(dict(name='out', direction='output', dtype='float32', shape=list(out_shape)))
        manifest = KernelManifest(name='calibration_' + name, source_file='kernel.cu', dialect='cuda', entrypoint='kernel',
                                  tensors=tensors, seed=20261021, oracle={'operation': op, 'parameters': params})
        if body is None:
            source = cuda_body(op, shapes, params, out_shape)
        else:
            count = 1
            for n in out_shape:
                count *= n
            args = 'const float *x, ' + ('const float *b, ' if len(shapes) == 2 else '') + 'float *out'
            source = f'extern "C" __global__ void kernel({args}) {{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<{count}){{{body}}}}}\n'
        (folder / 'kernel.cu').write_text(source)
        (folder / 'manifest.json').write_text(manifest.model_dump_json(indent=2) + '\n')
    return corpus


def calibration_gate(campaign):
    metadata = json.loads((campaign / 'campaign.json').read_text())
    preflight_path = campaign / 'preflight.json'
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    target = metadata.get('config', {}).get('targets', ['amd_xdna2_npu2'])[0]
    evaluator_ready = (preflight.get('status') == 'completed' and
                       preflight.get('compiler_smokes', {}).get(target, {})
                       .get('validation', {}).get('offline_contract_met') is True)
    rows = []
    for case in metadata['cases']:
        path = campaign / 'cases' / case['id'] / 'report.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        cycles = state.get('cycles', [])
        last = cycles[-1] if cycles else {}
        failure = classify_attempt(last)
        clean = (evaluator_ready and bool(cycles) and state.get('status') in ('completed', 'failed') and
                 failure['category'] not in ('infrastructure', 'reference_failure', 'provider_or_interruption', 'not_evaluated'))
        rows.append({'kernel': case['kernel'], 'clean': clean, 'last_failure': failure,
                     'report_sha256': sha256(path) if path.exists() else None, 'report': str(path)})
    result = {'status': 'passed' if len(rows) == 4 and all(row['clean'] for row in rows) else 'failed',
              'criterion': 'independent_evaluator_preflight', 'evaluator_ready': evaluator_ready,
              'preflight_sha256': sha256(preflight_path) if preflight_path.exists() else None,
              'tasks': rows, 'source_hashes': metadata['identity']['source_hashes'],
              'campaign_sha256': sha256(campaign / 'campaign.json')}
    write_json(campaign / 'calibration-gate.json', result)
    return result


def check_preparation(root, source_report, revision):
    report = json.loads(source_report.read_text())
    expected = discover_kernels(ROOT / 'examples/kernelbench')
    if report['status'] != 'passed' or len(report['tasks']) != 30 or set(r['kernel'] for r in report['tasks']) != set(expected):
        raise ValueError('all thirty CUDA/upstream/oracle validations must pass')
    for row in report['tasks']:
        source, manifest, _ = expected[row['kernel']]
        if (row['status'] != 'passed' or len(row['cases']) != 12 or not all(c['passed'] for c in row['cases']) or
                row['source_sha256'] != sha256(source) or row['manifest_sha256'] != sha256(manifest) or
                row['upstream_sha256'] != sha256(source.parent / 'upstream.py')):
            raise ValueError('source validation is incomplete or stale: ' + row['kernel'])
    campaign = root / f'calibration-v{revision}'
    gate = calibration_gate(campaign)
    if gate['status'] != 'passed':
        raise ValueError('calibration has unavailable evaluator, incomplete tasks or infrastructure/reference/provider failures')
    for relative, digest in gate['source_hashes'].items():
        if relative.startswith(('src/', 'playground/')) and sha256(ROOT / relative) != digest:
            raise ValueError('prompt/evaluator changed after calibration: ' + relative)
    return {'source_report': str(source_report.resolve()), 'source_report_sha256': sha256(source_report),
            'calibration_gate': str(campaign / 'calibration-gate.json'),
            'calibration_gate_sha256': sha256(campaign / 'calibration-gate.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('calibrate', 'evaluate', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/xdna2-baseline-20261001')
    parser.add_argument('--revision', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--source-report', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/xdna2-baseline-v1.json')
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--budget-on-openrouter', action='store_true',
                        help='Delegate dollar limits to OpenRouter; keep attempt/token limits and validation gates')
    budget.add_argument('--budget-on-openai', action='store_true',
                       help='Use direct OpenAI with token/time limits and no local dollar ledger; check billing on OpenAI')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    if args.phase == 'report':
        from npu_agent.xdna2_baseline import build_report

        print(json.dumps(build_report(root / 'evaluation'), indent=2))
        return 0
    StudyConfig.model_validate(config)
    if args.budget_on_openai != config['provider']['model'].startswith('gpt-'):
        parser.error('the direct OpenAI configuration requires --budget-on-openai')
    if args.phase == 'calibrate':
        config.update(corpus=str(calibration_inputs(root)), max_cycles=2)
        campaign = root / f'calibration-v{args.revision}'
    else:
        if args.source_report is None:
            parser.error('evaluate requires --source-report')
        calibration = json.loads((root / f'calibration-v{args.revision}' / 'campaign.json').read_text())
        if calibration['config']['provider'] != StudyConfig.model_validate(config).provider.model_dump():
            raise ValueError('provider configuration changed after calibration; use a new campaign')
        evidence = check_preparation(root, args.source_report, args.revision)
        if len(discover_kernels(ROOT / config['corpus'])) != 40:
            raise ValueError('evaluation requires exactly forty distinct tasks')
        campaign = root / 'evaluation'
        campaign.mkdir(exist_ok=True)
        write_json(campaign / 'preparation.json', evidence)
    StudyConfig.model_validate(config)
    config_path = root / (campaign.name + '.config.json')
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('campaign configuration changed; preserve the previous campaign')
    write_json(config_path, config)
    environment = os.environ.copy()
    environment.pop('OPENROUTER_API_KEY', None)
    if args.budget_on_openai:
        environment.pop('OPENAI_API_KEY', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[name] = '1'
    budget_args = (['--budget-on-openai'] if args.budget_on_openai else
                   ['--budget-on-openrouter'] if args.budget_on_openrouter else ['--ledger', str(root / 'budget.json')])
    if not (args.budget_on_openrouter or args.budget_on_openai):
        command = [sys.executable, str(ROOT / 'research/run_budgeted_study.py'), '--command', 'preflight',
                   '--ledger', str(root / 'budget.json'), '--limit-usd', '50', '--config', str(config_path), '--output', str(campaign)]
        status = subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode
        if status:
            return status
    status = subprocess.run([sys.executable, str(ROOT / 'research/run_parallel_study.py'), '--config', str(config_path),
        '--output', str(campaign), *budget_args, '--workers', '4', '--compiler-jobs', '2'],
        cwd=ROOT, env=environment, check=False).returncode
    if args.phase == 'calibrate':
        gate = calibration_gate(campaign)
        print('Calibration gate:', gate['status'], flush=True)
        return status or (0 if gate['status'] == 'passed' else 1)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
