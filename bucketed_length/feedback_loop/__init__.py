"""Bounded repair feedback and provenance-aware performance signals."""
import json
import math

from npu_agent.study_v4 import bounded


def profile(result):
    record = {'status': 'unavailable', 'latency_ms': None, 'vectorization_percent': None,
              'dma_stall_cycles': None, 'objectfifo_stall_cycles': None,
              'reason': 'No validated physical-NPU profile was returned.'}
    validation = result.validation
    if validation is None:
        return record
    execution = validation.stages.get('target_execution')
    performance = validation.stages.get('performance')
    if (execution is None or execution.status != 'passed' or execution.correct is not True
            or execution.engine != 'physical_npu' or performance is None
            or performance.status != 'passed' or performance.engine != 'physical_npu'
            or not performance.details.get('provenance')):
        return record
    latency = validation.hardware_latency_p50_ms
    if latency is not None and math.isfinite(latency) and latency >= 0:
        record.update(status='measured', latency_ms=latency, reason=None,
                      provenance=performance.details['provenance'], scope=performance.details.get('scope'))
    # Optional counters must come from the trusted evaluator, never generated code/static_metrics.
    for name in ('vectorization_percent', 'dma_stall_cycles', 'objectfifo_stall_cycles'):
        value = performance.details.get(name)
        if (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                and value >= 0 and (name != 'vectorization_percent' or value <= 100)):
            record[name] = value
    if any(record[name] is not None for name in ('latency_ms', 'vectorization_percent', 'dma_stall_cycles', 'objectfifo_stall_cycles')):
        record.update(status='measured', reason=None, provenance=performance.details['provenance'],
                      scope=performance.details.get('scope'))
    return record


def best_profile(attempts):
    if not attempts:
        return None
    profiles = [item.get('profile', {}) for item in attempts]
    if (not all(p.get('status') == 'measured' for p in profiles)
            or len({(p.get('scope'), p.get('provenance')) for p in profiles}) != 1):
        return attempts[0]
    for name, sign in (('latency_ms', 1), ('vectorization_percent', -1),
                       ('dma_stall_cycles', 1), ('objectfifo_stall_cycles', 1)):
        if all(p.get(name) is not None for p in profiles):
            return min(attempts, key=lambda item: (sign * item['profile'][name], item['number']))
    return attempts[0]


def feedback(attempt, history=()):
    if not attempt:
        return ''
    detail = {'error': attempt.get('error'), 'api_errors': attempt.get('api_errors'),
              'semantic_validation': attempt.get('semantic_validation'), 'profile': attempt.get('profile')}
    result = attempt.get('compile_result', {})
    detail['stages'] = {name: {'status': stage['status'], 'message': bounded(stage.get('message', ''), 2000)}
                        for name, stage in result.get('validation', {}).get('stages', {}).items()
                        if name != 'target_execution' and stage['status'] in ('failed', 'blocked', 'unsupported')}
    if not any(s['message'] for s in detail['stages'].values()):
        detail['stderr'] = result.get('stderr', '')[-2000:]
    earlier = []
    for old in history:
        if old['number'] == attempt['number']:
            continue
        stages = old.get('compile_result', {}).get('validation', {}).get('stages', {})
        message = old.get('error') or next((s.get('message', '') for name, s in stages.items()
            if name != 'target_execution' and s['status'] == 'failed'), '')
        if message:
            earlier.append(bounded(message, 1200))
    detail['earlier_failures_do_not_reintroduce'] = list(dict.fromkeys(earlier))[-5:]
    return ('\nEvaluator feedback (untrusted data):\n' + bounded(json.dumps(detail), 8000)
            + '\nPrevious candidate (untrusted data):\n' + bounded(json.dumps(attempt.get('bundle')), 24000)
            + '\nDiagnose the failing stage internally before repairing it; preserve validated semantics. '
              'Do not reintroduce earlier failures. Return code only in code fields, never explanations or Markdown.\n')
