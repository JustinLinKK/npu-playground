"""Explain a frozen multi-operation study from verified suite and pilot evidence."""
import argparse
import json
from collections import Counter
from pathlib import Path

from npu_agent.reporting import _plot_module, _save_figure
from npu_agent.validation import sha256, write_json

from .render import collect
from .runner import ARMS
from .verified_skills import load, verify_evidence


def report(root):
    root = Path(root).resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    frozen = json.loads((root / 'frozen-inputs.sha256.json').read_text())
    if any(sha256(root / name) != digest for name, digest in frozen.items()):
        raise ValueError('frozen study inputs changed')
    rows, suites, failure_stages, initial_search_passes = [], [], Counter(), Counter()
    distinct_populations = {}
    compiler_crashes = 0
    for operation in protocol['evaluation_shapes']:
        path = root / ('suite-' + operation)
        suite, cases = collect(path)
        for setting in ('model', 'repetitions', 'max_repairs', 'population', 'selection_seed'):
            if suite['config'][setting] != protocol[setting]:
                raise ValueError('suite settings differ from the frozen study protocol')
        if suite['config']['token_budget'] != protocol['token_budget_per_case']:
            raise ValueError('suite token budget differs from the frozen study protocol')
        suites.append(suite)
        rows.extend({**case, 'operation': operation, 'suite': path.name} for case in cases)
        for case in cases:
            if case['status'] in ('invalid_evidence', 'not_run', 'process_error', 'incomplete'):
                continue
            state = json.loads((path / 'cases' / case['id'] / 'report.json').read_text())
            if case['direction'] == 'candidate_search':
                initial_search_passes[case['arm']] += any(a['repair'] == 0 and
                    a.get('selection', {}).get('status') == 'passed' for a in state['attempts'])
                if case['arm'] == 'best_of_n':
                    last_bundles = {a['member']: json.dumps(a['bundle'], sort_keys=True)
                                    for a in state['attempts'] if 'bundle' in a}
                    distinct_populations[operation + '/' + case['id']] = len(set(last_bundles.values()))
            for attempt in state['attempts']:
                if attempt['status'] != 'failed':
                    continue
                stages = attempt.get('compile_result', {}).get('validation', {}).get('stages', {})
                failed = next((name + (' (unsupported)' if stage['status'] == 'unsupported' else '')
                    for name, stage in stages.items() if name != 'target_execution'
                    and stage['status'] in ('failed', 'unsupported')), 'generation/IR/API')
                failure_stages[failed] += 1
                compiler_crashes += 'frontend command failed' in stages.get('target_compile', {}).get('message', '')
    expected = protocol['planned_evaluation_cases']
    if len(rows) != expected:
        raise ValueError('study case count disagrees with its frozen protocol')
    seeds = load(root / 'seed-library')
    if any(s['seed_library_ids'] != [e['id'] for e in seeds] for s in suites):
        raise ValueError('suite seed library differs from the frozen study library')
    pilots = []
    for path in sorted(root.glob('*/report.json')):
        verify_evidence(path.parent)
        state = json.loads(path.read_text())
        pilots.append({'name': path.parent.name, 'status': state['status'],
                       'tokens': state['tokens_used'], 'error': state.get('error'),
                       'usage_complete': state['usage_complete']})
    followup = []
    followup_root = root / 'gemv-capability-followup'
    if (followup_root / 'plan.json').exists():
        plan = json.loads((followup_root / 'plan.json').read_text())
        for case in plan['cases']:
            row = {'id': case['id'], 'arm': case['arm'], 'status': 'not_run', 'tokens': None,
                   'first_pass': False, 'attempts': 0, 'usage_complete': False}
            path = followup_root / 'cases' / case['id']
            if (path / 'report.json').exists():
                try:
                    verify_evidence(path)
                    state = json.loads((path / 'report.json').read_text())
                    if (state['revision'] != plan['revision'] or state['source_sha256'] != plan['source_sha256']
                            or state['config']['direction'] != 'template_generation'
                            or state['config']['arm'] != case['arm']
                            or any(state['config'][key] != plan[key] for key in ('model', 'token_budget', 'max_repairs'))
                            or state['target']['id'] != plan['target'] or state['synthetic']
                            or state['manifest'] != json.loads((root / 'inputs/evaluation/gemv/manifest.json').read_text())):
                        raise ValueError('follow-up evidence differs from its declared inputs/settings')
                    row.update(status=state['status'], tokens=state['tokens_used'], attempts=len(state['attempts']),
                               usage_complete=state['usage_complete'],
                               first_pass=bool(state['attempts'] and state['attempts'][0]['status'] == 'passed'))
                    expected_exit = 0 if state['status'] == 'completed' else 3 if state['status'] in ('blocked', 'interrupted') else 1
                    if case['returncode'] != expected_exit:
                        row['status'] = 'process_error'
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row.update(status='invalid_evidence', error=str(exc))
            followup.append(row)
    counts = Counter(r['status'] for r in rows)
    groups = []
    for direction, arms in ARMS.items():
        for arm in arms:
            selected = [r for r in rows if (r['direction'], r['arm']) == (direction, arm)]
            groups.append({'direction': direction, 'arm': arm, 'planned': len(selected),
                'accepted': sum(r['status'] == 'completed' for r in selected),
                'first_passes': sum(r['first_attempt_pass'] is True for r in selected),
                'tokens': sum(r['tokens'] or 0 for r in selected),
                'active_seconds': sum(r['elapsed_seconds'] or 0 for r in selected),
                'by_operation': {op: sum(r['status'] == 'completed' and r['operation'] == op for r in selected)
                                 for op in protocol['evaluation_shapes']}})
    summary = {'protocol': protocol, 'statuses': dict(counts), 'groups': groups, 'cases': rows,
               'failed_attempts_by_first_failing_stage': dict(failure_stages),
               'search_cases_passing_before_repairs': dict(initial_search_passes),
               'distinct_final_bundles_per_population': distinct_populations,
               'compiler_frontend_crash_attempts': compiler_crashes,
               'pilots': pilots, 'evaluation_tokens': sum(r['tokens'] or 0 for r in rows),
               'pilot_tokens': sum(p['tokens'] for p in pilots),
               'followup_cases': followup, 'followup_tokens': sum(r['tokens'] or 0 for r in followup),
               'wall_seconds': sum(s['wall_seconds'] for s in suites),
               'unknown_usage_cases': sum(not r['usage_complete'] for r in rows)}
    destination = root / 'results'
    destination.mkdir(exist_ok=True)
    write_json(destination / 'summary.json', summary)
    plt = _plot_module()
    labels = [g['direction'].replace('_', ' ') + ' / ' + g['arm'] for g in groups]
    figure, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    axes[0].barh(labels, [g['accepted'] for g in groups], color='#34875b', label='Accepted after allowed attempts')
    axes[0].barh(labels, [g['first_passes'] for g in groups], color='#245379', height=.35, label='First attempt')
    axes[0].set_xlim(0, max(g['planned'] for g in groups))
    axes[0].set_xlabel(f'Accepted / {groups[0]["planned"]} planned cases per arm')
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.08), fontsize=8)
    axes[1].barh(labels, [g['tokens'] / 1000 for g in groups], color='#527fa2')
    axes[1].set_xlabel('Total tokens (thousands); all failures included')
    for index, group in enumerate(groups):
        axes[1].text(group['tokens'] / 1000, index, f' {group["tokens"] / 1000:.1f}', va='center', fontsize=8)
    axes[1].set_xlim(0, max(g['tokens'] for g in groups) / 1000 * 1.2 or 1)
    axes[0].invert_yaxis()
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Three-operation exploratory study: offline correctness and model cost')
    _save_figure(figure, destination, 'comparison')
    plt.close(figure)
    by_arm = {(g['direction'], g['arm']): g for g in groups}
    repair, one_shot = (by_arm['feedback_loop', arm] for arm in ('repair', 'one_shot'))
    single, population = (by_arm['candidate_search', arm] for arm in ('single', 'best_of_n'))
    search_cost = f'{population["tokens"] / single["tokens"]:.2f}x' if single['tokens'] else 'unknown'
    skeleton = by_arm['template_generation', 'skeleton']
    lines = ['# Seven-direction study: results and interpretation', '',
        f'**{counts["completed"]}/{expected} planned evaluation cases passed offline validation.** '
        f'{counts["budget_exhausted"]} exhausted the token budget. '
        f'The evaluation recorded **{summary["evaluation_tokens"]:,} tokens**; '
        f'new pilot/seed work recorded another **{summary["pilot_tokens"]:,} tokens**.', '',
        'A pass means the emitted artifact compiled for AMD NPU2 and its supported host/dataflow execution '
        'matched an independent numerical oracle. Physical NPU execution, speed, vectorization, and power were not measured.', '',
        f'- **Repair versus one-shot:** {repair["accepted"]}/{repair["planned"]} versus '
        f'{one_shot["accepted"]}/{one_shot["planned"]} accepted cases.',
        f'- **Four-candidate search versus one candidate:** {population["accepted"]}/{population["planned"]} '
        f'versus {single["accepted"]}/{single["planned"]}, using **{search_cost}** the recorded model tokens.',
        f'- **Original skeleton prompt:** {skeleton["accepted"]}/{skeleton["planned"]} accepted. '
        'Read the GEMV capability follow-up separately; it does not replace the frozen results.', '',
        '## Why this rerun differs from the original', '',
        'The original add-only run passed 1/16 arms; 14 exhausted 100,000 tokens. Its dominant failure '
        'was incompatible IRON API usage. This revision supplies the existing pinned-toolchain transport/API '
        'guidance to every free-form arm, retains earlier errors in compact repair feedback, and generates '
        'all initial search candidates before repairing them. Numerical tolerances and acceptance gates were not relaxed.', '',
        f'The ceiling is now {protocol["token_budget_per_case"]:,} tokens per case, equally across arms, to accommodate four independent '
        'IR-plus-code candidates. Cached input and reasoning tokens remain included in input + output usage '
        'without double counting. A call can cross the ceiling; that response is charged and not evaluated. '
        'These counts are not a dollar bill. Changes to guidance, budget, tasks, and libraries make the '
        'old/new comparison a recovery check, not a causal estimate of any single fix.', '',
        '## Frozen evaluation design', '',
        '| Operation | Verified seed shape | Evaluation shape | Repetitions per arm |',
        '| --- | --- | --- | ---: |']
    for op, shape in protocol['evaluation_shapes'].items():
        lines.append(f'| {op} | {protocol["seed_shapes"][op]} | {shape} | {protocol["repetitions"]} |')
    lines += ['', 'Model: `gpt-5.6-terra`, reasoning `xhigh`; up to five repairs; best-of-N has four candidates. '
        'Every arm sees the same evaluation input, compiler, numerical tolerance, and token ceiling for its task. '
        'The three operation suites run sequentially, with seven directions concurrent within each suite. '
        'The reduction and addition sources come from the classic corpus with smaller static shapes. '
        'GEMV uses the study-authored float32 CUDA loop stored with the inputs.', '',
        'The evaluator uses the existing twelve public input families per backend attempt. Search also '
        'checks five private random inputs plus a metamorphic input against both the executable IR and '
        'the independent reference. Repetitions vary model generation; they use the same frozen numerical '
        'test seeds and are not additional independent benchmark tasks.', '',
        f'The frozen library contains **{len(seeds)} verified seed entries**, from smaller shapes in the same '
        'operation families. Each case receives a private copy; evaluation results cannot leak into later '
        'cases. The earlier add skeleton is reused as one seed, with its original evidence preserved. '
        'Seed construction cost is separated from evaluation cost.', '',
        '## Results by arm', '', '![Correctness and token comparison](comparison.png)', '',
        '| Direction / arm | Accepted / planned | First attempt | Add | Reduction | GEMV | Total tokens |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in groups:
        cells = ' | '.join(f'{g["by_operation"][op]}/{protocol["repetitions"]}' for op in protocol['evaluation_shapes'])
        lines.append(f'| {g["direction"]} / {g["arm"]} | {g["accepted"]}/{g["planned"]} | '
                     f'{g["first_passes"]}/{g["planned"]} | {cells} | {g["tokens"]:,} |')
    controls = {('executable_ir', 'direct'), ('feedback_loop', 'repair'),
                ('template_generation', 'free_form'), ('feature_retrieval', 'off'),
                ('api_constraints', 'off'), ('verified_skills', 'off')}
    baseline = [r for r in rows if (r['direction'], r['arm']) in controls]
    lines += ['', 'The six controls `executable_ir/direct`, `feedback_loop/repair`, '
        '`template_generation/free_form`, `feature_retrieval/off`, `api_constraints/off`, and '
        '`verified_skills/off` use the same guided repair algorithm. Their individual differences '
        'are repeated-run variation, not six distinct methods. Pooled descriptively, these controls accepted '
        f'**{sum(r["status"] == "completed" for r in baseline)}/{len(baseline)}** cases, with '
        f'**{sum(r["first_attempt_pass"] is True for r in baseline)}** first-attempt passes.']
    complete = [g for g in groups if g['accepted'] == g['planned']]
    if complete:
        cheapest = min(complete, key=lambda g: g['tokens'])
        lines += ['', f'Among arms passing every planned case, **{cheapest["direction"]} / {cheapest["arm"]}** '
                  f'used the fewest recorded tokens ({cheapest["tokens"]:,}). '
                  + ('This is a repeated guided-repair control; its low sample cost does not identify a separate better algorithm.'
                     if (cheapest['direction'], cheapest['arm']) in controls else
                     'The small cohort does not establish a general winner.')]
    lines += ['', '## What each comparison can tell us', '']
    explanations = {
        'executable_ir': 'Compare semantic-pivot success and its extra token cost. A passing IR is not sufficient for backend acceptance.',
        'feedback_loop': 'The observed acceptance difference favors bounded repair on these tasks. The two arms use separate model generations; this is not a matched replay of identical first drafts.',
        'template_generation': 'Compare fixed transport against generated transport. This is an offline correctness/cost comparison; it cannot establish a vectorization advantage.',
        'feature_retrieval': 'Compare off, text, and feature selection from the same nonempty frozen library. With only three seed entries, broad retrieval superiority remains untested.',
        'api_constraints': 'Compare import/symbol checking with no checking under shared API guidance. This is post-generation lint, not full grammar-constrained decoding.',
        'verified_skills': 'Compare reuse of earlier verified kernels with no reuse on new shapes. Reuse uses the same inference-time selector as feature_retrieval/features; their differences are repeated-run variation. This measures transfer from verified experience; the fixed library does not measure a long-term growth curve.',
        'candidate_search': 'Compare one candidate with four independently generated candidates using private numerical, differential-IR, and metamorphic selection. Private tests are selection data, not a final untouched benchmark.'}
    for direction in ARMS:
        selected = [g for g in groups if g['direction'] == direction]
        result = '; '.join(f'`{g["arm"]}` {g["accepted"]}/{g["planned"]} accepted, '
                           f'{g["first_passes"]} first-attempt, {g["tokens"]:,} tokens' for g in selected)
        control = selected[0]
        differences = []
        for treatment in selected[1:]:
            delta = treatment['accepted'] - control['accepted']
            change = f'{delta} more' if delta > 0 else f'{-delta} fewer' if delta < 0 else 'the same number of'
            cost = f'{treatment["tokens"] / control["tokens"]:.2f} times the recorded tokens' if control['tokens'] else 'unknown relative token cost'
            differences.append(f'Against `{control["arm"]}`, `{treatment["arm"]}` accepted {change} cases '
                               f'and used {cost}.')
        lines += [f'### {direction.replace("_", " ").capitalize()}', '', result + '.', '',
                  ' '.join(differences), '', explanations[direction], '']
    search = [r for r in rows if r['arm'] == 'best_of_n']
    retrieved = [r for r in rows if r['direction'] in ('feature_retrieval', 'verified_skills') and r['arm'] != 'off']
    lines += ['## Coverage and unresolved failures', '',
        f'- Status counts: `{dict(counts)}`; all {expected} planned cases remain in the denominator.',
        f'- Best-of-N cases generating all four bundles: {sum(r["candidates_started"] == 4 for r in search)}/{len(search)}.',
        f'- Best-of-N cases with four distinct final code bundles: {sum(n == 4 for n in distinct_populations.values())}/{len(search)} '
        '(textual diversity, not proof of distinct performance strategies).',
        f'- Search cases with at least one private-selection pass before repairs: '
        f'single {initial_search_passes["single"]}/{len(search)}; best-of-N {initial_search_passes["best_of_n"]}/{len(search)}. '
        'The first-attempt column above counts only the first candidate, not the entire initial population.',
        f'- Retrieval/reuse cases with at least one retrieved example: {sum(r["retrieval_hits"] > 0 for r in retrieved)}/{len(retrieved)}.',
        f'- Attempts failing design emission: {sum(r["design_failures"] for r in rows)}.',
        f'- Cases exceeding the accounted token ceiling: {sum((r["tokens"] or 0) > protocol["token_budget_per_case"] for r in rows)}.',
        f'- Cases with complete accounting using at most the original 100,000-token ceiling: '
        f'{sum(r["usage_complete"] and r["tokens"] is not None and r["tokens"] <= 100000 for r in rows)}/{expected}. '
        'This is recorded consumption, not a replay under the old stopping rule.',
        f'- Missing token accounting: {summary["unknown_usage_cases"]} cases.',
        f'- Sum of sequential suite wall times: {summary["wall_seconds"] / 60:.1f} minutes. '
        'Overlapping child active times are not added to this clock; pilot and setup time are separate.', '']
    lines += ['Failed attempts, counted once at their first reported failed stage (including attempts later repaired):', '',
              '| First failing stage | Attempts |', '| --- | ---: |']
    lines.extend(f'| {stage} | {count} |' for stage, count in failure_stages.most_common())
    lines.append('')
    if any('unsupported' in stage for stage in failure_stages):
        lines += ['Some target-compiled candidates could not be evaluated by the supported host simulator. '
                  'They remain unaccepted; unsupported validation is not proof that their arithmetic is wrong. '
                  'In GEMV, generated `aie::mul`/`aie::mac` code exposes this coverage limit.', '']
    if compiler_crashes:
        lines += [f'The target compiler frontend crashed on **{compiler_crashes} generated candidates**. '
                  'These are compiler failures on the recorded source, not numerical-test failures. '
                  'The pinned toolchain and acceptance policy were retained.', '']
    for row in rows:
        if row['status'] != 'completed':
            reason = row['reason'].strip()
            if '\n' in reason:
                diagnostics = [line.strip() for line in reason.splitlines()
                               if 'error:' in line.lower() and not line.startswith('subprocess.')]
                reason = reason.partition(':')[0] + ': ' + (diagnostics[-1] if diagnostics else reason.splitlines()[-1])
            reason = reason.replace('|', '/')[:350]
            lines.append(f'- [{row["operation"]}/{row["id"]}](../{row["suite"]}/cases/{row["id"]}/report.json): '
                         f'**{row["status"]}**. {reason}')
    if len({r['fingerprint'] for r in rows if r['fingerprint']}) > 1:
        lines += ['**Compiler/evaluator fingerprints differ; comparisons are confounded.**', '']
    if any(r['synthetic'] for r in rows):
        lines += ['**SYNTHETIC TEST DATA: these outcomes are not model/compiler experiment evidence.**', '']
    lines += ['', '## Pilot and seed ledger', '', '| Run | Outcome | Recorded tokens |', '| --- | --- | ---: |']
    for pilot in pilots:
        lines.append(f'| [{pilot["name"]}](../{pilot["name"]}/report.json) | {pilot["status"]} | {pilot["tokens"]:,} |')
    lines += ['', 'The initial GEMV seed attempt stopped before generation because the oracle does not bind '
        'arbitrary rows/cols scalars. Its corrected input uses explicit static dimensions; the blocked record '
        'is retained. Historical add-seed usage belongs to the original run and is not charged again here.', '']
    if followup:
        lines += ['## Separate GEMV capability-guidance follow-up', '',
            'The frozen study exposed a gap in the skeleton prompt: it encouraged vector operations '
            'without listing the host evaluator\'s supported compute subset. Revision `guided-compute-v3` '
            'adds that subset and requests scalar pointer loops for unsupported multiply/dot operations, '
            'avoiding nonportable vector operators and dynamic lane indexing. The compiler, numerical '
            'tolerances, reference, and acceptance policy stay the same.', '',
            '**This is a development replay on the same GEMV shape, not a new held-out benchmark.** '
            'Its four predeclared cases are separate from the 96-case table; original failures are retained.', '',
            '| Arm / repetition | Outcome | First attempt | Attempts | Tokens |',
            '| --- | --- | --- | ---: | ---: |']
        for row in followup:
            token_text = f'{row["tokens"]:,}' if row['tokens'] is not None else 'unknown'
            token_text += '' if row['usage_complete'] else ' *'
            lines.append(f'| [{row["id"]}](../gemv-capability-followup/cases/{row["id"]}/report.json) | '
                         f'{row["status"]} | {"yes" if row["first_pass"] else "no"} | {row["attempts"]} | {token_text} |')
        lines += ['', f'Follow-up total: **{sum(r["status"] == "completed" for r in followup)}/{len(followup)} accepted**, '
                  f'**{summary["followup_tokens"]:,} additional recorded tokens**. '
                  'A passing scalar fallback establishes offline correctness, not vectorization or device speed.', '',
                  '* Incomplete usage: displayed token counts are known usage only.', '',
                  '[Follow-up plan and commands](../gemv-capability-followup/plan.json).', '']
        corrected = [r for r in followup if r['arm'] == 'skeleton']
        original = [r for r in rows if r['operation'] == 'gemv' and
                    (r['direction'], r['arm']) == ('template_generation', 'skeleton')]
        lines += [f'For GEMV skeletons specifically, the original prompt accepted '
                  f'**{sum(r["status"] == "completed" for r in original)}/{len(original)}** cases using '
                  f'**{sum(r["tokens"] or 0 for r in original):,} tokens**. The corrected prompt accepted '
                  f'**{sum(r["status"] == "completed" for r in corrected)}/{len(corrected)}**, with '
                  f'**{sum(r["first_pass"] for r in corrected)} first-attempt passes** and '
                  f'**{sum(r["tokens"] or 0 for r in corrected):,} tokens**. '
                  'Treat this as evidence for the targeted portability fix on the replayed shape.', '']
    lines += ['## How to use these results', '',
        'Keep guided repair as the default for this task family. Four-candidate search added no final '
        'accepted cases here, so reserve that extra token cost for harder cases. Use the corrected '
        'compute-capability guidance when trying skeletons; do not interpret a scalar fallback as a '
        'vectorization or device-performance improvement. Feature retrieval is promising for first-draft '
        'reliability in its six runs, but the identical skill-reuse selector varied and the library is small.', '',
        'Use accepted count first, then first-attempt count and total tokens to choose the next exploratory '
        'pipeline. Equal acceptance with more tokens is no observed correctness benefit on this cohort. '
        'There are only three small operation families and two repetitions per arm: report descriptive '
        'differences, not statistical significance or general NPU superiority. Larger workloads, different '
        'operations, physical-device performance, and direct CUDA equivalence remain untested. '
        'Skeleton selection and feature retrieval use the reviewed manifest operation labels; this study '
        'does not measure automatic operation classification from arbitrary CUDA source.', '',
        '## Evidence and reproduction', '',
        '- [Frozen protocol](../protocol.json) and [machine-readable summary](summary.json).']
    if (root / 'validation-audit.json').exists():
        lines.append('- [Evidence audit](../validation-audit.json): frozen code, compiler identity, inputs, and complete token accounting.')
    for op in protocol['evaluation_shapes']:
        lines.append(f'- [{op} detailed report](../suite-{op}/results/REPORT.md), including every child log and evidence bundle.')
    lines += ['', 'Regenerate this report without model calls:', '', '```bash',
              f'.venv/bin/python -m bucketed_length.report_study {root}', '```', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    report(args.root)
