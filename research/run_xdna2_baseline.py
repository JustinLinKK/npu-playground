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

FP16_CONTRACT = 'intel-fp16-v1'
# Carried by every intel-fp16-v1 manifest; the original GPU source still declares its float32 interface.
FP16_CONSTRAINT = ('Intel FP16 variant: use the manifest tensor dtypes and scalar values even when the original GPU source '
                   'uses float32. The reference evaluates the supplied FP16 inputs with FP32 or higher intermediates and '
                   'rounds the final output to FP16.')


def prepared_dataset(corpus):
    """Identity of a prepared dataset corpus (dataset.json at its root), or None for a source corpus."""
    path = corpus / 'dataset.json'
    if not path.is_file():
        return None
    identity = json.loads(path.read_text())['identity']
    if identity.get('precision_contract') != FP16_CONTRACT:
        raise ValueError('unsupported dataset precision contract: ' + str(identity.get('precision_contract')))
    return identity


def calibration_folder(dataset):
    return 'calibration-inputs-fp16' if dataset else 'calibration-inputs'


def calibration_inputs(root, dataset=None):
    """Four toy tasks; with an FP16 dataset they take its dtypes and tolerance so calibration qualifies that contract."""
    corpus = root / calibration_folder(dataset)
    if corpus.exists():
        return corpus
    dtype = 'float16' if dataset else 'float32'
    precision = {'constraints': [FP16_CONSTRAINT], 'tolerance': dataset['tolerance']} if dataset else {}
    definitions = [
        ('add', [(7,), (7,)], (7,), 'vector_add', {}, 'out[i]=x[i]+b[i];'),
        ('reduce', [(17,)], (1,), 'reduce_sum', {'keepdims': True}, 'float s=0;for(int t=0;t<17;++t)s+=x[t];out[i]=s;'),
        ('transpose', [(3, 5)], (5, 3), 'transpose', {'axes': [1, 0]}, 'out[i]=x[(i%3)*5+i/3];'),
        ('gemv', [(3, 7), (7, 1)], (3, 1), 'kb_matmul', {}, None),
    ]
    for name, shapes, out_shape, op, params, body in definitions:
        folder = corpus / name
        folder.mkdir(parents=True)
        tensors = [dict(name='x' if i == 0 else 'b', direction='input', dtype=dtype, shape=list(s)) for i, s in enumerate(shapes)]
        tensors.append(dict(name='out', direction='output', dtype=dtype, shape=list(out_shape)))
        manifest = KernelManifest(name='calibration_' + name, source_file='kernel.cu', dialect='cuda', entrypoint='kernel',
                                  tensors=tensors, seed=20261021, oracle={'operation': op, 'parameters': params}, **precision)
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


def check_dataset(corpus):
    """Bind an evaluation to a prepared dataset: its listed files, the original sources it was derived from, and
    bit-exact agreement of the pipeline's generated inputs and oracle references with the stored ones."""
    import numpy as np

    from npu_agent.oracles import execute_oracle
    from npu_agent.testcases import generate_cases

    report = json.loads((corpus / 'dataset.json').read_text())
    identity = prepared_dataset(corpus)
    files = {str(path.relative_to(corpus)) for path in corpus.rglob('*') if path.is_file()} - {'dataset.json'}
    if files != set(report['artifacts']) or any(sha256(corpus / name) != digest for name, digest in report['artifacts'].items()):
        raise ValueError('prepared dataset files are missing, unlisted or changed')
    for relative, digest in identity['sources'].items():
        if sha256(ROOT / 'examples' / relative) != digest:
            raise ValueError('dataset was prepared from different original sources: ' + relative)
    kernels = discover_kernels(corpus)
    if sorted(kernels) != sorted(task['kernel'] for task in report['tasks']):
        raise ValueError('dataset task inventory disagrees with its manifests')

    def same(left, right):
        return set(left) == set(right) and all(
            left[k].dtype == right[k].dtype and left[k].shape == right[k].shape and np.array_equal(left[k], right[k], equal_nan=True)
            for k in left)

    def load(path):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    cases = 0
    for task in report['tasks']:
        _, manifest_path, manifest = kernels[task['kernel']]
        if manifest.tolerance.model_dump(include={'rtol', 'atol'}) != identity['tolerance']:
            raise ValueError('manifest tolerance differs from the dataset tolerance: ' + task['kernel'])
        tests = generate_cases(manifest)
        if [test.id for test in tests] != task['cases']:
            raise ValueError('dataset case inventory disagrees with generated cases: ' + task['kernel'])
        for test in tests:
            inputs = {name: value for name, value in test.inputs.items() if isinstance(value, np.ndarray)}
            reference = execute_oracle(manifest, {**test.inputs, **{s.name: s.value for s in manifest.scalars}})
            if (not same(inputs, load(manifest_path.parent / 'inputs' / f'{test.id}.npz')) or
                    not same(reference, load(manifest_path.parent / 'expected' / f'{test.id}.npz'))):
                raise ValueError(f'pipeline inputs or references differ from the prepared dataset: {task["kernel"]} {test.id}')
            cases += 1
    return {'dataset': str(corpus.resolve()), 'dataset_sha256': sha256(corpus / 'dataset.json'),
            'precision_contract': identity['precision_contract'], 'tolerance': identity['tolerance'],
            'tasks': len(report['tasks']), 'cases': cases}


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
    dataset = prepared_dataset(ROOT / config['corpus'])
    if args.phase == 'calibrate':
        config.update(corpus=str(calibration_inputs(root, dataset)), max_cycles=2)
        campaign = root / f'calibration-v{args.revision}'
    else:
        if args.source_report is None:
            parser.error('evaluate requires --source-report')
        calibration = json.loads((root / f'calibration-v{args.revision}' / 'campaign.json').read_text())
        if calibration['config']['provider'] != StudyConfig.model_validate(config).provider.model_dump():
            raise ValueError('provider configuration changed after calibration; use a new campaign')
        if Path(calibration['config']['corpus']).name != calibration_folder(dataset):
            raise ValueError('calibration precision differs from the evaluation corpus; use a fresh RUN_ROOT')
        evidence = check_preparation(root, args.source_report, args.revision)
        if len(discover_kernels(ROOT / config['corpus'])) != 40:
            raise ValueError('evaluation requires exactly forty distinct tasks')
        if dataset:
            evidence['dataset'] = check_dataset(ROOT / config['corpus'])
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
