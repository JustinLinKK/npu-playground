"""Feature and diagnostic retrieval with a mandatory evaluator/toolchain filter."""
import json
import math
import re

from npu_agent.database import normalize_error


def features(manifest, target, fingerprint):
    if not fingerprint or fingerprint == 'unresolved':
        raise ValueError('retrieval requires a resolved compiler/evaluator fingerprint')
    return {'target': target.id, 'fingerprint': fingerprint, 'operation': manifest.oracle.operation,
            'dtypes': [t.dtype for t in manifest.tensors], 'shapes': [list(t.shape) for t in manifest.tensors],
            'reduction_axes': [manifest.oracle.parameters.get('axis')] if manifest.oracle.operation in
                              ('reduce_sum', 'softmax', 'layer_norm') else [],
            'layout': [t.layout for t in manifest.tensors]}


def retrieve(entries, key, *, mode='features', diagnostic='', query='', limit=2):
    if mode not in ('off', 'text', 'features'):
        raise ValueError('unknown retrieval mode')
    if mode == 'off':
        return []
    candidates = []
    for entry in entries:
        other = entry['features']
        # Even the text control cannot inject a different compiler or target's code.
        if any(other.get(k) != key[k] for k in ('target', 'fingerprint')):
            continue
        if mode == 'features':
            if any(other.get(k) != key[k] for k in ('operation', 'dtypes', 'reduction_axes', 'layout')):
                continue
            if [len(s) for s in other['shapes']] != [len(s) for s in key['shapes']]:
                continue
            distance = sum(abs(math.log2(a / b)) for x, y in zip(other['shapes'], key['shapes'])
                           for a, b in zip(x, y))
            exact_fix = bool(diagnostic) and normalize_error(diagnostic) in entry.get('fixes', {})
            score = (not exact_fix, distance, entry['id'])
        else:
            words = set(re.findall(r'\w+', query.lower()))
            found = set(re.findall(r'\w+', json.dumps(entry['bundle']).lower()))
            score = (-len(words & found), entry['id'])
        candidates.append((score, entry))
    return [entry for _, entry in sorted(candidates, key=lambda item: item[0])[:limit]]
