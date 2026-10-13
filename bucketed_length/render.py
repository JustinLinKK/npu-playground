"""Render verified per-case evidence while retaining missing and blocked cases."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from npu_agent.reporting import _plot_module, _save_figure
from npu_agent.validation import json_digest, write_json

from .runner import ARMS, metrics
from .verified_skills import verify_evidence


COLORS = {'completed': '#34875b', 'failed': '#c9514e', 'budget_exhausted': '#d39036',
          'blocked': '#73869b', 'interrupted': '#b27eaa', 'process_error': '#8c5a4a',
          'invalid_evidence': '#7c5bac', 'incomplete': '#c4ad69', 'not_run': '#d7dce2'}


def collect(output):
    suite = json.loads((output / 'suite.json').read_text())
    if suite['protocol'] != 'bucketed-length-parallel-v1':
        raise ValueError('unsupported suite protocol')
    expected = {(direction, arm, repetition) for direction, arms in ARMS.items() for arm in arms
                for repetition in range(1, suite['config']['repetitions'] + 1)}
    actual = [(c['direction'], c['arm'], c['repetition']) for c in suite['cases']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('suite does not contain every planned arm exactly once per repetition')
    rows = []
    for case in suite['cases']:
        case_id = f'{case["direction"]}--{case["arm"]}--r{case["repetition"]}'
        if case['id'] != case_id:
            raise ValueError('case ID does not match its planned arm')
        row = {key: case[key] for key in ('id', 'direction', 'arm', 'repetition', 'returncode', 'elapsed_seconds')}
        row.update(status='not_run', tokens=None, usage_complete=False, first_attempt_pass=None, winner=None,
                   compile_passes=0, offline_passes=0, selection_passes=0, calls=0, attempts=0,
                   candidates_started=0, candidates_planned=suite['config']['population'] if
                   (case['direction'], case['arm']) == ('candidate_search', 'best_of_n') else 1,
                   design_failures=0,
                   synthetic=False, fingerprint=None, retrieval_hits=0, vectorization_percent=None, reason='')
        directory = output / 'cases' / case_id
        if not (directory / 'report.json').is_file():
            if case['process_status'] == 'not_run':
                row['reason'] = 'Case was not started.'
            elif case['process_status'] == 'running':
                row.update(status='incomplete', reason='No terminal report; saved process state is running, live state unverified.')
            else:
                row.update(status='process_error', reason=case.get('error') or 'Process exited without a case report; inspect its log.')
        else:
            try:
                verify_evidence(directory)
                state = json.loads((directory / 'report.json').read_text())
                if (state['protocol'] != 'bucketed-length-part1-v1'
                        or state['config']['direction'] != case['direction'] or state['config']['arm'] != case['arm']
                        or any(state['config'][key] != suite['config'][key] for key in
                               ('model', 'token_budget', 'max_repairs', 'population', 'selection_seed'))
                        or (not state['synthetic'] and state['provider'] != suite['config']['provider'])
                        or state['library_entries'] != (suite['seed_library_ids'] if case['direction'] in
                                                       ('feature_retrieval', 'verified_skills') else [])
                        or state['source_sha256'] != suite['source_sha256']
                        or json_digest(state['manifest']) != suite['manifest_sha256']
                        or state['target']['id'] != suite['config']['target']):
                    raise ValueError('case evidence does not match the frozen suite inputs/settings')
                row.update(metrics(state), reason=state.get('error') or '',
                           fingerprint=(state['identity'] or {}).get('fingerprint'),
                           retrieval_hits=sum(len(a.get('retrieved', [])) for a in state['attempts']))
                if not row['reason'] and row['status'] != 'completed' and state['attempts']:
                    last = state['attempts'][-1]
                    stages = last.get('compile_result', {}).get('validation', {}).get('stages', {})
                    row['reason'] = last.get('error') or next((name + ' (' + s['status'] + '): ' + s.get('message', '')
                        for name, s in stages.items() if name != 'target_execution' and s['status'] in ('failed', 'unsupported')),
                        'No candidate passed all acceptance checks; inspect the case report.')
                if row['status'] not in COLORS:
                    row.update(status='incomplete', reason='Case has no terminal outcome.')
                expected_exit = 0 if row['status'] == 'completed' else 3 if row['status'] in ('blocked', 'interrupted') else 1
                if case['process_status'] == 'finished' and case['returncode'] != expected_exit:
                    row.update(status='process_error', first_attempt_pass=None, winner=None,
                               reason=f'Child exit {case["returncode"]} disagrees with its report; inspect log.')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status='invalid_evidence', reason=str(exc))
        rows.append(row)
    return suite, rows


def render(output):
    output = Path(output).resolve()
    suite, rows = collect(output)
    destination = output / 'results'
    destination.mkdir(exist_ok=True)
    groups = []
    for direction, arms in ARMS.items():
        for arm in arms:
            cases = [r for r in rows if r['direction'] == direction and r['arm'] == arm]
            groups.append({'direction': direction, 'arm': arm, 'planned': len(cases),
                'statuses': dict(Counter(r['status'] for r in cases)),
                'known_tokens': sum(r['tokens'] or 0 for r in cases),
                'unknown_usage_cases': sum(not r['usage_complete'] for r in cases),
                'first_attempt_passes': sum(r['first_attempt_pass'] is True for r in cases),
                'active_seconds': sum(r['elapsed_seconds'] or 0 for r in cases)})
    summary = {'suite_status': suite['status'], 'wall_seconds': suite['wall_seconds'],
               'cases': rows, 'groups': groups, 'note': 'Offline outcomes; summed active time is not suite wall time or NPU latency.'}
    write_json(destination / 'summary.json', summary)
    with (destination / 'cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt = _plot_module()
    labels = [g['direction'].replace('_', ' ') + ' / ' + g['arm'] for g in groups]
    evidence_label = (' — SYNTHETIC TEST DATA' if any(r['synthetic'] for r in rows) else
                      ' — DRY RUN' if suite['status'] == 'dry_run' else '')
    figure, axis = plt.subplots(figsize=(12, 9))
    left = [0] * len(groups)
    for status, color in COLORS.items():
        counts = [g['statuses'].get(status, 0) for g in groups]
        axis.barh(labels, counts, left=left, color=color, label=status.replace('_', ' '))
        left = [a + b for a, b in zip(left, counts)]
    axis.invert_yaxis()
    axis.set_xlabel('Cases (every planned repetition retained)')
    axis.set_title('Part 1: outcomes across all seven directions' + evidence_label)
    axis.set_xlim(0, max(1, suite['config']['repetitions']))
    axis.xaxis.get_major_locator().set_params(integer=True)
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.09), ncol=3, frameon=False)
    axis.spines[['top', 'right']].set_visible(False)
    _save_figure(figure, destination, 'outcomes')
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    for axis, key, title, unit in ((axes[0], 'known_tokens', 'Recorded model tokens', 'Tokens (input + output)'),
                                   (axes[1], 'active_seconds', 'Summed active case time', 'Seconds; not NPU latency')):
        values = [g[key] for g in groups]
        axis.barh(labels, values, color='#4d7da8')
        axis.set_title(title)
        axis.set_xlabel(unit)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_xlim(0, max(values, default=0) * 1.3 or 1)
        for index, value in enumerate(values):
            text = f'{value:,.0f}' if key == 'known_tokens' else f'{value:,.1f}'
            if key == 'known_tokens' and groups[index]['unknown_usage_cases']:
                text += ' *'
            axis.text(value, index, ' ' + text, va='center', fontsize=8)
    axes[0].invert_yaxis()
    figure.suptitle(f'Suite wall time: {suite["wall_seconds"]:,.1f}s | * Usage missing or case not run' + evidence_label, fontsize=12)
    _save_figure(figure, destination, 'resources')
    plt.close(figure)
    counts = Counter(r['status'] for r in rows)
    lines = ['# Seven-direction experiment results', '',
        f'Suite status: **{suite["status"]}**. Planned cases: **{len(rows)}**. '
        f'Accepted cases: **{counts["completed"]}**. Wall time: **{suite["wall_seconds"]:.1f} s**.', '',
        f'Target: `{suite["config"]["target"]}`. Model: `{suite["config"]["model"]}`. '
        f'Parallel directions: {suite["config"]["workers"]}.', '',
        'Acceptance means offline validation, not physical NPU correctness or performance. '
        'Unrun/blocked cases remain in every denominator. Active time sums overlapping child processes; '
        'it includes startup and differs from suite wall time. Missing token usage is not zero cost.', '']
    if not suite['seed_library_ids']:
        lines += ['**The seed library was empty.** Retrieval arms had no prior verified examples; '
                  'these runs do not measure the benefit of populated retrieval or cross-task skill growth.', '']
    if any(r['synthetic'] for r in rows):
        lines += ['**Synthetic test evidence is present; do not interpret these as model or compiler benchmark results.**', '']
    if len({r['fingerprint'] for r in rows if r['fingerprint']}) > 1:
        lines += ['**Compiler/evaluator fingerprints differ across cases; comparisons are confounded.**', '']
    lines += ['## What the run exercised', '',
        f'- Recorded model usage: **{sum(r["tokens"] or 0 for r in rows):,} input + output tokens** '
        '(cached inputs included; this is not a dollar cost).',
        f'- Budget-exhausted cases: **{counts["budget_exhausted"]}**. '
        'A response crossing the per-case cap is charged but not evaluated.',
        f'- Attempts failing design emission: **{sum(r["design_failures"] for r in rows)}**.',
        f'- Retrieved-example occurrences: **{sum(r["retrieval_hits"] for r in rows)}**.', '']
    for row in rows:
        if row['direction'] == 'candidate_search':
            lines.append(f'- `{row["id"]}`: generated {row["candidates_started"]}/{row["candidates_planned"]} '
                         f'candidate bundles; {row["selection_passes"]} passed private selection checks.')
    lines.append('')
    lines += ['![Outcomes](outcomes.png)', '', '![Resources](resources.png)', '',
              '| Direction | Arm | Accepted / planned | First-attempt passes | Known tokens | Active seconds |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for group in groups:
        token_text = f'{group["known_tokens"]:,}' + (' *' if group['unknown_usage_cases'] else '')
        lines.append(f'| {group["direction"]} | {group["arm"]} | {group["statuses"].get("completed", 0)} / '
                     f'{group["planned"]} | {group["first_attempt_passes"]} | {token_text} | {group["active_seconds"]:.1f} |')
    lines += ['', '* Token usage missing or case not run; shown values are recorded totals only.', '',
              'Rows, token counts and failure reasons: [CSV](cases.csv), [JSON](summary.json). '
              'Vectorization remains unknown unless physical profiling was supplied. '
              'These are descriptive results, not a causal or statistically significant improvement claim.', '', '## Cases needing attention', '']
    for row in rows:
        if row['status'] != 'completed':
            reason = row['reason'].replace('\n', ' ').replace('|', '/')[:400]
            log_link = f' [Log](../logs/{row["id"]}.log)' if (output / 'logs' / f'{row["id"]}.log').is_file() else ''
            lines.append(f'- `{row["id"]}`: **{row["status"]}**. {reason}{log_link}')
    if all(row['status'] == 'completed' for row in rows):
        lines.append('All planned cases returned accepted, hash-verified results.')
    (destination / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument('output', type=Path)
    render(argument_parser.parse_args().output)
