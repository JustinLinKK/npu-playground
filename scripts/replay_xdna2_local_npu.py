"""Check the offline-accepted XDNA2 baseline artifacts on a local Ryzen AI NPU2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from npu_agent.config import AMD_XDNA2_NPU2
from npu_agent.executors import decode_tensor, encode_tensor, execute_target
from npu_agent.models import ExecutionRequest, KernelManifest
from npu_agent.testcases import generate_cases
from npu_agent.validation import json_digest, sha256, write_json


def replay(evaluation: Path, output: Path, kernel: str | None = None) -> dict:
    campaign = json.loads((evaluation / 'campaign.json').read_text())
    if campaign['config']['protocol'] != 'xdna2-baseline-v1':
        raise ValueError('XDNA2 baseline campaign required')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in campaign['cases']:
        if kernel is not None and case['kernel'] != kernel:
            continue
        report_path = evaluation / 'cases' / case['id'] / 'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        cycle = next((item for item in report.get('cycles', []) if item.get('status') == 'passed'), None)
        row = {'kernel': case['kernel'], 'case_id': case['id'], 'status': 'not_offline_accepted'}
        if cycle is None:
            rows.append(row)
            continue
        compiled = cycle['compile_result']
        if compiled['validation']['offline_contract_met'] is not True:
            raise ValueError(f'offline validation missing: {case["kernel"]}')
        artifacts = compiled['artifacts']
        binary_paths = {name: Path(artifacts[name]) for name in ('final.xclbin', 'insts.bin')}
        abi_path = Path(artifacts['abi.json'])
        manifest_path = Path(artifacts['contract/manifest.json'])
        for path in (*binary_paths.values(), abi_path, manifest_path):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(evaluation.resolve()):
                raise ValueError(f'missing or unsafe campaign artifact: {path}')
        compiled_hashes = compiled['validation']['stages']['target_compile']['artifacts']
        for name, path in {**binary_paths, 'abi.json': abi_path}.items():
            if sha256(path) != compiled_hashes[name]:
                raise ValueError(f'compiled artifact changed: {name}')
        if not json.loads(abi_path.read_text())['device'].startswith('npu2'):
            raise ValueError('compiled artifact does not target NPU2')
        manifest = KernelManifest.model_validate_json(manifest_path.read_text())
        if manifest.name != case['kernel'] or json_digest(manifest.model_dump(mode='json')) != compiled['validation']['manifest_sha256']:
            raise ValueError('campaign manifest mismatch')
        cases = {item.id: {name: encode_tensor(value) for name, value in item.inputs.items()
                 if isinstance(value, np.ndarray)} for item in generate_cases(manifest)}
        saved_inputs = compiled['validation']['input_sha256']
        if set(saved_inputs) != set(cases):
            raise ValueError('saved case list mismatch')
        for case_id, digest in saved_inputs.items():
            path = Path(artifacts[f'inputs/{case_id}.npz'])
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(evaluation.resolve()) or sha256(path) != digest:
                raise ValueError(f'saved input mismatch: {case_id}')
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(cases[case_id]) or any(
                    not np.array_equal(saved[name], decode_tensor(payload), equal_nan=True)
                    for name, payload in cases[case_id].items()):
                    raise ValueError(f'generated input mismatch: {case_id}')
        request = ExecutionRequest(run_id=case['id'], target=AMD_XDNA2_NPU2,
            artifacts={name: str(path) for name, path in binary_paths.items()},
            artifact_sha256={name: sha256(path) for name, path in binary_paths.items()},
            manifest=manifest, manifest_sha256=json_digest(manifest.model_dump(mode='json')),
            abi_sha256=sha256(abi_path), cases=cases,
            compiler_fingerprint=compiled['compiler_fingerprint'], timeout_seconds=600)
        write_json(output / f'{case["id"]}.request.json', request.model_dump(mode='json'))
        result = execute_target(request, [sys.executable, str(ROOT / 'scripts/xdna2_local_npu_executor.py')])
        row.update(status=result.status, correct=result.correct, stage=result.model_dump(mode='json'))
        rows.append(row)
        write_json(output / 'summary.json', {'campaign': str(evaluation), 'rows': rows})
    if kernel is not None and not rows:
        raise ValueError(f'unknown kernel: {kernel}')
    summary = {'campaign': str(evaluation), 'requested': len(rows),
        'offline_accepted': sum(row['status'] != 'not_offline_accepted' for row in rows),
        'physical_passes': sum(row['status'] == 'passed' and row.get('correct') is True for row in rows),
        'rows': rows}
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path, help='RUN_ROOT/evaluation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel', help='Run one named task first as a hardware smoke check')
    args = parser.parse_args()
    result = replay(args.evaluation.resolve(), args.output.resolve(), args.kernel)
    print(f'Physical NPU passes: {result["physical_passes"]}/{result["requested"]}')
    raise SystemExit(0 if result['physical_passes'] == result['offline_accepted'] else 1)
