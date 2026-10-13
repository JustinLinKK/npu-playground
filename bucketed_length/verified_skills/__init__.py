"""Content-addressed, verification-gated kernels and observed diagnostic/fix pairs."""
import json
from pathlib import Path

from npu_agent.database import normalize_error
from npu_agent.artifacts import ALLOWED_FILES
from npu_agent.baseline import _validate_bundle
from npu_agent.models import CodeBundle, CompileResult, KernelManifest, TargetProfile
from npu_agent.validation import json_digest, sha256

from ..feature_retrieval import features


def verify_entry(entry):
    from ..runner import checked_result

    payload = {key: value for key, value in entry.items() if key != 'id'}
    if entry['id'] != json_digest(payload):
        raise ValueError('skill content hash changed')
    manifest = KernelManifest.model_validate(entry['manifest'])
    target = TargetProfile.model_validate(entry['target'])
    bundle = CodeBundle.model_validate(entry['bundle'])
    _validate_bundle(bundle, target)
    if bundle.unsupported_operations or {f.relative_path for f in bundle.files} != ALLOWED_FILES[target.backend]:
        raise ValueError('skill requires a complete supported bundle')
    import hashlib

    hashes = {file.relative_path: hashlib.sha256(file.content.encode()).hexdigest() for file in bundle.files}
    result = CompileResult.model_validate(entry['compile_result'])
    if entry['features'] != features(manifest, target, result.compiler_fingerprint):
        raise ValueError('skill features disagree with its validated contract')
    if not checked_result(result, manifest, target, result.compiler_fingerprint, hashes):
        raise ValueError('skill has not passed independent offline validation')
    return entry


def load(directory):
    if directory is None or not Path(directory).exists():
        return []
    result = []
    for path in sorted(Path(directory).glob('*.json')):
        if path.is_symlink():
            raise ValueError('skill files must not be symlinks')
        entry = verify_entry(json.loads(path.read_text()))
        if path.stem != entry['id']:
            raise ValueError('skill filename does not match its content hash')
        result.append(entry)
    return result


def verify_evidence(run):
    run = Path(run).resolve()
    hashes = json.loads((run / 'evidence.sha256.json').read_text())
    if 'report.json' not in hashes:
        raise ValueError('missing report evidence hash')
    for name, digest in hashes.items():
        path = run / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(run) or not path.is_file() or sha256(path) != digest):
            raise ValueError(f'evidence hash or path changed: {name}')
    return hashes


def admit(run, directory):
    run = Path(run).resolve()
    hashes = verify_evidence(run)
    state = json.loads((run / 'report.json').read_text())
    if (state['protocol'] != 'bucketed-length-part1-v1' or state['status'] != 'completed'
            or not state['usage_complete'] or state['synthetic']
            or state['tokens_used'] > state['config']['token_budget']):
        raise ValueError('only completed, budget-qualified, nonsynthetic runs may become skills')
    winner = next(item for item in state['attempts'] if item['number'] == state['winner'])
    if winner['status'] != 'passed':
        raise ValueError('winner is not verified')
    if state['config']['direction'] == 'candidate_search' and winner.get('selection', {}).get('status') != 'passed':
        raise ValueError('winner failed selection validation')
    for name, digest in winner['source_hashes'].items():
        relative = f'attempt-{winner["number"]}/candidate/{name}'
        if hashes.get(relative) != digest:
            raise ValueError('candidate source is not bound to saved evidence')
    fixes = {}
    for previous in state['attempts']:
        if previous['number'] >= winner['number'] or previous['status'] != 'failed':
            continue
        diagnostic = previous.get('compile_result', {}).get('stderr') or previous.get('error', '')
        if diagnostic:
            fixes[normalize_error(diagnostic)] = {'diagnostic': diagnostic[-4000:],
                'failed_source_hashes': previous.get('source_hashes', {}),
                'description': 'The subsequent stored bundle passed validation; this is an observed repair, not proof of causality.'}
    payload = {'schema': 'verified-kernel-v1', 'features': state['features'], 'bundle': winner['bundle'],
               'manifest': state['manifest'], 'target': state['target'], 'compile_result': winner['compile_result'],
               'evidence_sha256': sha256(run / 'evidence.sha256.json'), 'report_sha256': hashes['report.json'],
               'fixes': fixes, 'evidence_tier': 'offline-validated', 'source_equivalence': 'not_established'}
    entry = verify_entry({'id': json_digest(payload), **payload})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{entry["id"]}.json'
    if path.exists():
        if json.loads(path.read_text()) != entry:
            raise ValueError('existing skill differs from admitted evidence')
    else:
        with path.open('x') as handle:
            json.dump(entry, handle, indent=2, allow_nan=False)
    return path
