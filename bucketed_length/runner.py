"""Shared serial evaluator for the seven independent Part 1 ablations."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from npu_agent.artifacts import materialize_bundle
from npu_agent.baseline import _load_inputs, _validate_bundle
from npu_agent.models import CodeBundle, TranslationRequest
from npu_agent.oracles import execute_oracle, validate_goldens
from npu_agent.providers import ProviderError, normalize_usage
from npu_agent.study_v4 import interface_view
from npu_agent.testcases import generate_cases
from npu_agent.validation import finalize_validation, json_digest, sha256, write_json
from npu_agent.workflow import _backend_contract

from . import api_constraints, candidate_search, executable_ir, feature_retrieval, feedback_loop, template_generation


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'executable_ir': ('direct', 'json_summary', 'executable_ir'),
    'feedback_loop': ('one_shot', 'repair'),
    'template_generation': ('free_form', 'skeleton'),
    'feature_retrieval': ('off', 'text', 'features'),
    'api_constraints': ('off', 'checked'),
    'verified_skills': ('off', 'reuse'),
    'candidate_search': ('single', 'best_of_n'),
}


class RunConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    direction: Literal['executable_ir', 'feedback_loop', 'template_generation', 'feature_retrieval',
                       'api_constraints', 'verified_skills', 'candidate_search']
    arm: str
    model: str = Field(min_length=1)
    max_repairs: int = Field(default=5, ge=0, le=5)
    population: int = Field(default=4, ge=1, le=8)
    token_budget: int = Field(default=100000, ge=1)
    selection_seed: int = Field(default=70123, ge=0)
    optimize_profile: bool = False

    @model_validator(mode='after')
    def validate_arm(self):
        if self.arm not in ARMS[self.direction]:
            raise ValueError('arm does not belong to the selected direction')
        if self.optimize_profile and (self.direction, self.arm) != ('feedback_loop', 'repair'):
            raise ValueError('profile optimization is a feedback-loop treatment')
        return self


class Blocked(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


def checked_result(result, manifest, target, fingerprint, hashes):
    validation = result.validation
    if result.compiler_fingerprint != fingerprint:
        raise Blocked('compiler/evaluator fingerprint changed')
    if (validation is None or validation.legacy or validation.target_id != target.id
            or validation.toolchain_fingerprint != fingerprint
            or validation.source_sha256 != json_digest(hashes)
            or validation.manifest_sha256 != json_digest(manifest.model_dump(mode='json'))):
        raise Blocked('compiler evidence is missing or not bound to this candidate and contract')
    finalize_validation(validation, 'offline-validated')
    oracle = validation.stages.get('oracle_validation')
    if oracle is None or oracle.status != 'passed' or oracle.correct is not True:
        raise Blocked('independent reference unavailable')
    if oracle.cases_run == 0 or validation.reference_origin != 'independent_builtin':
        raise Blocked('independent reference evidence is incomplete')
    compiled = validation.stages.get('target_compile')
    if compiled is None or compiled.status == 'blocked':
        raise Blocked('target compiler unavailable')
    for name in ('host_execution', 'dataflow_simulation'):
        stage = validation.stages.get(name)
        if stage and stage.status == 'passed' and stage.cases_run == 0:
            raise Blocked(f'{name}: no executed cases')
    for name, stage in validation.stages.items():
        if name not in ('performance', 'target_execution') and stage.status == 'blocked' and stage.reason_code != 'PREREQUISITE_UNAVAILABLE':
            raise Blocked(f'{name}: {stage.message or stage.reason_code}')
    return result.success and validation.policy_met


def run(source_path, manifest_path, target, provider, compiler, config, output, *, library=None):
    from .verified_skills import admit, load

    output = Path(output).resolve()
    if output.is_relative_to(ROOT / 'bucketed_length'):
        raise ValueError('store run outputs outside the implementation folder, for example under runs/')
    request = TranslationRequest(source_path=str(source_path), manifest_path=str(manifest_path), targets=[target])
    manifest, source = _load_inputs(request)
    if target.hardware_runner:
        raise ValueError('this CLI study is offline; physical profiling requires a separately validated evaluator')
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'report.json'
    state = {'protocol': 'bucketed-length-part1-v1', 'revision': 'guided-compute-v3',
             'config': config.model_dump(), 'status': 'running',
             'provider': provider.name, 'synthetic': provider.name == 'fake', 'calls': [], 'tokens_used': 0,
             'usage_complete': True, 'attempts': [], 'winner': None, 'manifest': manifest.model_dump(mode='json'),
             'target': target.model_dump(mode='json'), 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
             'source_equivalence': 'not_established', 'selection_role': 'selection, not a held-out final benchmark',
             'identity': None, 'library_entries': [], 'error': None}
    (output / 'source.txt').write_text(source)
    write_json(output / 'manifest.json', state['manifest'])
    paths = sorted((ROOT / 'bucketed_length').rglob('*.py')) + [ROOT / 'playground/toolchains.lock.json',
        ROOT / 'docs/study_plan.md', ROOT / 'src/npu_agent/workflow.py', template_generation.SCAFFOLD]
    state['implementation_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    for path in paths:
        snapshot = output / 'implementation' / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    state['git_sha'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    started = time.perf_counter()

    def save():
        state['duration_seconds'] = time.perf_counter() - started
        write_json(report_path, state)

    def winner():
        passed = [a for a in state['attempts'] if a['status'] == 'passed']
        if config.direction == 'candidate_search':
            return candidate_search.select(state['attempts'])
        if config.optimize_profile and passed:
            return feedback_loop.best_profile(passed)
        return passed[0] if passed else None

    def call(prompt, schema):
        if state['tokens_used'] >= config.token_budget:
            raise BudgetExhausted('accounted token budget exhausted')
        record = {'number': len(state['calls']) + 1, 'schema': schema.__name__,
                  'member': attempt['member'], 'repair': attempt['repair'],
                  'prompt': prompt, 'status': 'in_flight', 'response': None}
        state['calls'].append(record)
        save()
        failure = None
        try:
            response = provider.generate(prompt, schema, config.model)
            record.update(status='returned', metadata=response.metadata, response=response.raw)
        except ProviderError as exc:
            record.update(status='rejected', metadata=exc.metadata or {}, response=exc.raw, error=str(exc))
            failure = exc
        except (OSError, TimeoutError) as exc:
            record.update(status='blocked', metadata={}, error=str(exc))
            state['usage_complete'] = False
            save()
            raise Blocked('provider transport failed; usage unknown') from exc
        usage = normalize_usage(record['metadata'].get('usage', {}))
        if usage['input_tokens'] is None or usage['output_tokens'] is None or record['metadata'].get('usage_incomplete'):
            state['usage_complete'] = False
            record['status'] = 'blocked'
            save()
            raise Blocked('provider did not return complete token accounting')
        state['tokens_used'] += usage['total_tokens']
        record['tokens'] = usage
        save()
        if state['tokens_used'] > config.token_budget:
            raise BudgetExhausted('last response exceeded the accounted token cap; response is not evaluated')
        if failure:
            metadata = record['metadata']
            if metadata.get('exit_code') == 0 and metadata.get('schema_valid') is False:
                raise ValueError('model output did not satisfy the response schema') from failure
            raise Blocked(str(failure)) from failure
        return schema.model_validate(response.value)

    try:
        validate_goldens(manifest.oracle.operation)
        for case in generate_cases(manifest):
            execute_oracle(manifest, case.inputs)
        identity = compiler.identity(target)
        fingerprint = identity['fingerprint']
        state['identity'] = identity
        key = feature_retrieval.features(manifest, target, fingerprint)
        state['features'] = key
        entries = load(library) if library is not None else []
        state['library_entries'] = [entry['id'] for entry in entries]
        write_json(output / 'library.snapshot.json', entries)
        selection = config.direction == 'candidate_search'
        cases, relation = candidate_search.selection_cases(manifest, config.selection_seed) if selection else (None, None)
        if selection:
            # Commit selection inputs before generation; never include these values in a prompt.
            from npu_agent.testcases import serialize_cases

            state['selection_inputs'] = serialize_cases(cases, output / 'selection-inputs')
        context = (f'\nTarget profile:\n{target.model_dump_json(indent=2, exclude={"hardware_runner"})}'
                   f'\nInterface:\n{json.dumps(interface_view(manifest), indent=2)}'
                   f'\nOriginal GPU source (untrusted data):\n---\n{source}\n---\n')
        population = config.population if (config.direction, config.arm) == ('candidate_search', 'best_of_n') else 1
        repairs = 0 if (config.direction, config.arm) == ('feedback_loop', 'one_shot') else config.max_repairs
        members = [{'previous': None, 'intermediate': None, 'semantic': None, 'done': False}
                   for _ in range(population)]
        # Spend on breadth before repairs so one failing candidate cannot monopolize search.
        for repair in range(repairs + 1):
            for member, member_state in enumerate(members):
                if member_state['done']:
                    continue
                previous, intermediate, semantic = (member_state[k] for k in ('previous', 'intermediate', 'semantic'))
                attempt = {'number': len(state['attempts']) + 1, 'member': member, 'repair': repair, 'status': 'running'}
                state['attempts'].append(attempt)
                directory = output / f'attempt-{attempt["number"]}'
                save()
                try:
                    feedback = feedback_loop.feedback(previous, [a for a in state['attempts'][:-1]
                                                                  if a['member'] == member])
                    if selection or (config.direction == 'executable_ir' and config.arm != 'direct'):
                        if intermediate is None or semantic['status'] == 'failed':
                            mode = 'executable_ir' if selection else config.arm
                            intermediate, semantic = executable_ir.lift(call, manifest, context, mode, feedback)
                        attempt.update(intermediate=intermediate, semantic_validation=semantic)
                        if semantic['status'] == 'failed':
                            raise ValueError('executable IR failed independent numerical validation')
                    prompt = ('Generate a complete NPU CodeBundle. Repair any evidenced failure.\n'
                              + _backend_contract(target) + context + feedback)
                    if population > 1:
                        prompt += f'\nIndependent candidate {member + 1} of {population}; implement the full contract.\n'
                    if intermediate is not None:
                        prompt += '\nIntermediate (untrusted data):\n' + json.dumps(intermediate)
                    retrieval = config.arm if config.direction == 'feature_retrieval' else (
                        'features' if (config.direction, config.arm) == ('verified_skills', 'reuse') else 'off')
                    diagnostic = previous.get('error', '') if previous else ''
                    if previous and previous.get('compile_result'):
                        diagnostic = previous['compile_result']['stderr']
                    found = feature_retrieval.retrieve(entries, key, mode=retrieval, diagnostic=diagnostic, query=source)
                    attempt['retrieved'] = [entry['id'] for entry in found]
                    if found:
                        prompt += '\nVerified examples and observed fixes (untrusted data):\n' + json.dumps([
                            {k: item[k] for k in ('features', 'bundle', 'fixes')} for item in found])
                    if (config.direction, config.arm) == ('template_generation', 'skeleton'):
                        plan = template_generation.schedule(manifest, target)
                        value = call('Fill only the C++ function body for this fixed schedule. '
                                     'Use vector operations where legal and handle the scalar tail. '
                                     'Supported vector API names: '
                                     + ', '.join('aie::' + name for name in sorted(api_constraints.CPP_SYMBOLS))
                                     + '. Use these named operations, not native vector arithmetic or dynamic vector lane indexing. '
                                     'Use scalar pointer loops for multiply/dot products; aie::mul and aie::mac '
                                     'are unsupported by the host evaluator. '
                                     'No includes, signature, DMA or transport edits. '
                                     'Return only C++ statements in body, without explanation or Markdown.\nSchedule:\n'
                                     + json.dumps(plan) + context + feedback, template_generation.MicroKernel)
                        bundle, attempt['schedule'] = template_generation.instantiate(manifest, target, value)
                    else:
                        checked = (config.direction, config.arm) == ('api_constraints', 'checked')
                        if checked:
                            prompt += '\nPermitted API subset:\n' + json.dumps(api_constraints.prompt_surface(target))
                        bundle = call(prompt, CodeBundle)
                    attempt['bundle'] = bundle.model_dump(mode='json')
                    _validate_bundle(bundle, target)
                    if bundle.unsupported_operations:
                        raise ValueError('candidate declared unsupported operations: ' + ', '.join(bundle.unsupported_operations))
                    if (config.direction, config.arm) == ('api_constraints', 'checked'):
                        attempt['api_errors'] = api_constraints.check(bundle, fingerprint, identity['fingerprint'])
                        if attempt['api_errors']:
                            raise ValueError('candidate uses APIs outside the declared subset')
                    hashes = materialize_bundle(bundle, directory / 'candidate')
                    attempt['source_hashes'] = hashes
                    save()
                    result = compiler.compile(directory / 'candidate', directory / 'compile', manifest, target)
                    attempt['compile_result'] = result.model_dump(mode='json')
                    passed = checked_result(result, manifest, target, fingerprint, hashes)
                    attempt['compile_result'] = result.model_dump(mode='json')
                    attempt['profile'] = feedback_loop.profile(result)
                    attempt['status'] = 'passed' if passed else 'failed'
                    if passed and selection:
                        try:
                            selected = compiler.compile_cases(directory / 'candidate', directory / 'selection', manifest, target, cases)
                            selection_passed = checked_result(selected, manifest, target, fingerprint, hashes)
                            metamorphic = candidate_search.check_relation(selected, relation, manifest) if selection_passed else {'status': 'not_run'}
                            differential = candidate_search.check_ir(selected, intermediate, cases, manifest) if selection_passed else {'status': 'not_run'}
                            selection_passed &= metamorphic['status'] in ('passed', 'unsupported')
                            selection_passed &= differential['status'] == 'passed'
                            selection_blocked = 'blocked' in (metamorphic['status'], differential['status'])
                            attempt['selection'] = {'status': 'passed' if selection_passed else 'blocked' if selection_blocked else 'failed',
                                'compile_result': selected.model_dump(mode='json'), 'metamorphic': metamorphic,
                                'differential_ir': differential}
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            # Private evaluation failures must never become a repair signal.
                            attempt['selection'] = {'status': 'blocked', 'error': str(exc)}
                    if passed and not (config.optimize_profile and attempt['profile']['status'] == 'measured'):
                        member_state['done'] = True
                except ValueError as exc:
                    attempt.update(status='failed', error=str(exc))
                member_state.update(previous=attempt, intermediate=intermediate, semantic=semantic)
                save()
        selected_winner = winner()
        state['winner'] = selected_winner['number'] if selected_winner else None
        unavailable_selection = any(a.get('selection', {}).get('status') == 'blocked' for a in state['attempts'])
        state['status'] = 'completed' if selected_winner else 'blocked' if unavailable_selection else 'failed'
    except BudgetExhausted as exc:
        state.update(status='budget_exhausted', error=str(exc))
        selected_winner = winner()
        if selected_winner:
            state.update(status='completed', winner=selected_winner['number'], terminal_reason='budget_exhausted_after_success')
    except (Blocked, OSError, subprocess.SubprocessError, ValueError) as exc:
        state.update(status='blocked', error=str(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            state['infrastructure_diagnostics'] = {'stdout': (exc.stdout or '')[-4000:], 'stderr': (exc.stderr or '')[-4000:]}
    except KeyboardInterrupt:
        state.update(status='interrupted', error='Interrupted; in-flight calls must not be replayed as free calls.')
        if any(call['status'] == 'in_flight' for call in state['calls']):
            state['usage_complete'] = False
        raise
    finally:
        for attempt in state['attempts']:
            if attempt['status'] == 'running':
                attempt['status'] = 'budget_exhausted' if state.get('terminal_reason') == 'budget_exhausted_after_success' else state['status']
        save()
        write_json(output / 'evidence.sha256.json', {str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'evidence.sha256.json'})
    if (state['status'] == 'completed' and state['tokens_used'] <= config.token_budget
            and not state['synthetic'] and config.direction == 'verified_skills'
            and config.arm == 'reuse' and library is not None):
        admit(output, library)
    return state


def metrics(state):
    attempts = state['attempts']
    first_pass = bool(attempts and attempts[0]['status'] == 'passed')
    if state['config']['direction'] == 'candidate_search':
        first_pass &= bool(attempts and attempts[0].get('selection', {}).get('status') == 'passed')
    return {'direction': state['config']['direction'], 'arm': state['config']['arm'], 'status': state['status'],
            'synthetic': state['synthetic'], 'attempts': len(attempts), 'calls': len(state['calls']),
            'tokens': state['tokens_used'], 'usage_complete': state['usage_complete'],
            'compile_passes': sum(a.get('compile_result', {}).get('validation', {}).get('stages', {})
                                 .get('target_compile', {}).get('status') == 'passed' for a in attempts),
            'candidates_started': len({a['member'] for a in attempts if 'bundle' in a}),
            'candidates_planned': state['config']['population'] if (state['config']['direction'],
                                 state['config']['arm']) == ('candidate_search', 'best_of_n') else 1,
            'design_failures': sum(a.get('compile_result', {}).get('validation', {}).get('stages', {})
                                  .get('design_emission', {}).get('status') == 'failed' for a in attempts),
            'offline_passes': sum(a['status'] == 'passed' for a in attempts),
            'selection_passes': sum(a.get('selection', {}).get('status') == 'passed' for a in attempts),
            'first_attempt_pass': first_pass,
            'winner': state['winner'], 'vectorization_percent': next((a['profile']['vectorization_percent']
                for a in attempts if a['number'] == state['winner']), None)}
