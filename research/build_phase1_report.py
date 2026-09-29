"""Build one Phase 1 report and figures from the pilot and interruption rerun."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/analysis/research-next-steps-v1'
OUT = BASE / 'pilot-retry1'
FIG = OUT / 'combined-figures'
FIG.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text())


pilot = read(OUT / 'audit.json')
rerun = read(BASE / 'amd-interruption-rerun/audit.json')
resolved = read(BASE / 'amd-interruption-rerun/resolved-pilot-cases.json')
ledger = read(ROOT / 'runs/research-next-steps-budget.json')
config = read(ROOT / 'configs/case-study-v4.json')
assert len(resolved) == 18 and len({r['case_id'] for r in resolved}) == 18
assert not ledger.get('pending_call') and not ledger.get('pending_calls')
for audit in (pilot, rerun):
    assert audit['in_flight_calls'] == 0
    for path, expected in audit['manual_reviews'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        review = read(Path(path))
        assert hashlib.sha256((ROOT / review['source_report']).read_bytes()).hexdigest() == review['source_report_sha256']
assert len(pilot['manual_reviews']) + len(rerun['manual_reviews']) == 20
original_by_id = {c['id']: c for c in pilot['cases']}
rerun_by_id = {c['id']: c for c in rerun['cases']}
for row in resolved:
    selected = rerun_by_id[row['case_id']] if row['rerun_status'] else original_by_id[row['case_id']]
    assert row['selected_status'] == selected['status']
    row['selected_tokens'] = selected['known_tokens']
    row['active_seconds'] = original_by_id[row['case_id']]['active_seconds'] + (selected['active_seconds'] if row['rerun_status'] else 0)
    row['backend_attempts'] = original_by_id[row['case_id']]['backend_attempts'] + (selected['backend_attempts'] if row['rerun_status'] else 0)

arms = ['direct', 'hinted_ir', 'structured_ir']
arm_labels = {'direct': 'Direct', 'hinted_ir': 'Hinted IR', 'structured_ir': 'Structured IR'}
targets = ['amd_xdna2_npu2', 'intel_npu_4000']
target_labels = dict(zip(targets, ['AMD', 'Intel']))
groups = [(t, a) for t in targets for a in arms]
labels = [f'{target_labels[t]} / {arm_labels[a]}' for t, a in groups]
colors = {'completed': '#23856b', 'failed': '#c25445', 'blocked': '#d89b2d'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    for extension in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 4.5))
left = np.zeros(6)
for status, label in [('completed', 'Offline pass'), ('failed', 'Budget failure'), ('blocked', 'HTTP 429 unresolved')]:
    counts = np.array([sum(r['target'] == t and r['arm'] == a and r['selected_status'] == status for r in resolved) for t, a in groups])
    ax.barh(labels, counts, left=left, color=colors[status], label=label)
    for i, count in enumerate(counts):
        if count: ax.text(left[i] + count / 2, i, str(count), ha='center', va='center', color='white', fontweight='bold')
    left += counts
ax.invert_yaxis(); ax.set_xticks([0, 1, 2, 3]); ax.set_xlabel('Cases (3 planned per backend and arm)')
ax.set_title('Latest outcome view: 5 passes, 12 failures, 1 interruption', loc='left', pad=18)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3, frameon=False)
fig.text(.5, -.16, 'Only the two interrupted AMD slots use fresh rerun outcomes; original interruptions remain in the record.', ha='center', fontsize=9)
save(fig, 'outcomes')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, metric, scale, title in [(axes[0], 'known_tokens', 1000, 'Known translation tokens (thousands)'), (axes[1], 'active_seconds', 60, 'Summed recorded active minutes')]:
    first = [sum(c[metric] for c in pilot['cases'] if c['target'] == t and c['arm'] == a)/scale for t,a in groups]
    second = [sum(c[metric] for c in rerun['cases'] if c['target'] == t and c['arm'] == a)/scale for t,a in groups]
    ax.barh(labels, first, color='#4875a5', label='Original 18-case pilot')
    ax.barh(labels, second, left=first, color='#aa79ae', label='Two AMD rerun attempts')
    ax.set_xlabel(title); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    for i,(x,y) in enumerate(zip(first,second)):
        ax.text(x+y+max(first)*.02, i, f'{x+y:,.1f}', va='center', fontsize=9)
    ax.set_xlim(0,max(x+y for x,y in zip(first,second))*1.2)
axes[0].invert_yaxis(); axes[0].legend(loc='upper center', bbox_to_anchor=(.55,-.16), ncol=2, frameon=False)
fig.suptitle('Charge all attempts, including failures and original interruptions', x=.12, ha='left')
fig.text(.5,-.09,'Preflight excluded from these panels. Missing usage is not zero. Active minutes are not wall time; execution mixed serial and parallel.',ha='center',fontsize=9)
fig.tight_layout(); save(fig,'work')

failures = Counter(pilot['failure_counts']) + Counter(rerun['failure_counts'])
fig, ax = plt.subplots(figsize=(10,4.5))
names = {'backend_api':'Backend API / syntax', 'model_output':'Model output / bundle contract', 'semantic':'Semantic validation', 'unsupported_ir':'Unsupported IR', 'numerical':'Numerical mismatch', 'buffer_dataflow':'Interface / dataflow', 'infrastructure':'Infrastructure interruption'}
items = failures.most_common()
ax.barh([names[k] for k,v in items],[v for k,v in items],color='#4875a5')
for i,(_,v) in enumerate(items): ax.text(v+1,i,str(v),va='center')
ax.invert_yaxis(); ax.set_xlim(0,max(failures.values())*1.12)
ax.set_xlabel('Failed stage events across original pilot + reruns')
ax.set_title('Backend construction dominates recorded failures',loc='left',pad=18)
fig.text(.5,-.04,'Not unique cases: one candidate can produce multiple failed stages. Preflight failures are described separately.',ha='center',fontsize=9)
save(fig,'failures')

costs = {}
for call in ledger['calls'].values():
    root = str(Path(call['campaign']).relative_to(ROOT))
    costs[root] = costs.get(root,Decimal(0)) + Decimal(call['cost_usd'])
assert sum(costs.values(),Decimal(0)) == Decimal(ledger['known_spent_usd'])
root_names = {
 'runs/case-study-v4-deepseek-pilot-config-key':'Initial pilot attempt (stopped early)',
 'runs/analysis/research-next-steps-v1/route-health-20260917':'Route-health probe',
 'runs/case-study-v4-deepseek-pilot-retry1':'Main 18-case pilot, including preflight',
 'runs/case-study-v4-amd-interruption-rerun':'AMD rerun launch: failed schema preflight',
 'runs/case-study-v4-amd-interruption-rerun-retry1':'AMD rerun: passed preflight and two attempts'}
unknown = []
roots = {ROOT / p for p in costs} | {ROOT / 'runs/case-study-v4-deepseek-pilot'}
observed = {}
for root in sorted(roots):
    for p in [root / 'preflight.json', *sorted((root / 'cases').glob('*/report.json'))]:
        if not p.exists(): continue
        data = read(p); calls = list(data.get('calls',{}).values())
        for cycle in data.get('cycles',[]): calls.extend(cycle['calls'].values())
        for call in calls:
            m=call.get('metadata',{}); generation=m.get('generation_id'); cost=m.get('usage',{}).get('cost')
            if not generation or cost is None:
                unknown.append({'report':str(p.relative_to(ROOT)), 'http_status':m.get('http_status')})
            else:
                assert generation not in observed
                observed[generation]={'cost_usd':str(Decimal(str(cost))), 'campaign':str(root)}
assert observed == ledger['calls']
assert len(unknown) == 5
status_words = {'completed':'Pass','failed':'Fail','blocked':'Interrupted'}
lookup = {(r['target'],r['kernel'],r['arm']):r for r in resolved}
kernels = [('cuda_vector_add','Vector addition'),('cuda_tiled_transpose','Transpose'),('triton_layer_norm','Layer normalization')]
lines = ['# Phase 1 — combined pilot and AMD rerun report','',
 '**Latest result: 5 offline successes, 12 budget-exhausted failures, and 1 unresolved HTTP 429 interruption across 18 case slots. Known spending for all associated attempts and preflights is $1.0403365089. All experiment processes have stopped.**','',
 'This is the single reading view for the original pilot, its parallel continuation, and both AMD replacement attempts. The charts and tables below include the reruns; no other report is needed to follow the findings. Original evidence is retained for auditing.','',
 '## What we found','',
 '1. **Backend code construction is the main obstacle.** Most failures were unavailable APIs, invalid Python/MLIR, or incompatible constructor arguments. Validated semantic IR did not reliably produce a working backend implementation.',
 '2. **Structured IR showed no advantage in this pilot.** Intel direct and hinted each passed 2/3 tasks; structured passed 1/3 while using more tokens. With only three families and one repetition, this does not establish that IR is generally worse.',
 '3. **Layer normalization has a source/acceptance-contract discrepancy.** Seven Intel host validations failed only the numerical boundary case, with the same error metrics observed when executing the original Triton source. The failures remain failures under the frozen tolerance.',
 '4. **Repair did not always act on feedback.** Hinted Intel layer normalization repeated the disallowed filename `./model.py` for ten cycles without reaching compilation.',
 '5. **The AMD reruns did not add a success.** Hinted IR completed ten cycles of backend-construction failures. Direct hit another HTTP 429 after four compiler attempts and remains unresolved.','',
 '## Results, including the requested reruns','',
 '![Latest outcomes](combined-figures/outcomes.png)','',
 '| Backend | Kernel | Direct | Hinted IR | Structured IR |','| --- | --- | --- | --- | --- |']
for target in targets:
    for kernel,label in kernels:
        cells=[]
        for arm in arms:
            row=lookup[target,kernel,arm];cells.append(status_words[row['selected_status']] + (' (rerun)' if row['rerun_status'] else ''))
        lines.append(f"| {target_labels[target]} | {label} | {' | '.join(cells)} |")
lines += ['', '**How to read this:** “Pass” means target compilation and the configured offline correctness checks passed. It does not mean execution on physical NPU hardware. “Fail” means the per-case repair/token budget was exhausted. “Interrupted” is a provider failure, not evidence of candidate correctness or incapability.', '',
 'The original pilot ended with **5 passes, 11 failures, 2 interruptions**. Only the two interrupted AMD transpose slots were replaced in the latest outcome view. Those replacements are fresh attempts with fresh per-case budgets; this is not an uninterrupted equal-total-budget experiment. Original attempts remain included in work and spending below.', '',
 '## Tokens, active time, and repair work','', '![Work including original attempts and reruns](combined-figures/work.png)','',
 '| Backend / arm | Latest passes | Original pilot tokens | Added rerun tokens | Total known translation tokens | Summed active minutes | Backend attempts |',
 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for t,a in groups:
    rows=[r for r in resolved if r['target']==t and r['arm']==a]
    lines.append(f"| {target_labels[t]} / {arm_labels[a]} | {sum(r['selected_success'] for r in rows)}/3 | {sum(r['original_known_tokens'] for r in rows):,} | {sum(r['rerun_known_tokens'] for r in rows):,} | {sum(r['all_attempts_known_tokens'] for r in rows):,} | {sum(r['active_seconds'] for r in rows)/60:.2f} | {sum(r['backend_attempts'] for r in rows)} |")
lines += ['', 'These totals include all original pilot cases and both AMD reruns, including failed and interrupted work. They exclude preflight and the earlier aborted pilot attempt; all known monetary charges are included in the next section. Missing request usage is not zero. Reasoning tokens are already included in output tokens.', '',
 '**Timing:** the six-case Intel parallel segment took **11.45 minutes of elapsed wall time**, using eight available workers and four compiler slots. That is not a measured speedup against a serial replay. Case active time includes compiler queueing, sums concurrent work, and spans mixed serial/parallel execution; it must not be interpreted as campaign wall time or a clean arm-level speed comparison.', '',
 '## Complete spending history and accounting limits','',
 '| Component | Known billed cost (USD) |','| --- | ---: |']
for root,cost in costs.items(): lines.append(f'| {root_names[root]} | ${cost} |')
lines += [f"| **Total known spending** | **${ledger['known_spent_usd']}** |",'',
 f"All **{len(observed)}** recorded generation charges were reconciled exactly against saved provider records. The AMD rerun request added **$0.1256507031**, including its failed preflight launch. No request remains pending.", '',
 '**Five historical requests lack complete billing identity/cost:** one from the initial wrong-key preflight, one from the early stopped pilot, two HTTP 429 calls in the main pilot, and one new HTTP 429 during the AMD direct rerun. Their cost remains unknown. Consequently, complete accounting is still false; rerunning does not repair the missing historical records. No billing-limit error was established by these HTTP 429 records.', '',
 '## Failure analysis','', '![Failure categories across both sets of attempts](combined-figures/failures.png)','',
 'Failure counts above are **stage events**, not unique cases: a single invalid graph can cause both graph-construction and target-compilation failures. All 18 original case records and both rerun records have hash-bound cycle reviews.', '',
 '| Area | Observed failure | Interpretation |','| --- | --- | --- |',
 '| AMD backend generation | Missing Python modules, unsupported MLIR operations, invalid syntax and DMA/interface declarations | Primarily backend construction, often before numerical execution |',
 '| Intel structured vector addition | Valid IR followed by repeated OpenVINO import/parameter errors | Semantic validation alone did not resolve backend API usage |',
 '| Intel layer norm, hinted | `./model.py` rejected instead of required `model.py`, ten times | Model-output/evaluator filename contract, not NPU numerical failure |',
 '| Intel layer norm, direct/structured | Seven host runs failed only `operation_boundary` | Numerical contract needs investigation; preserve frozen outcomes |',
 '| HTTP 429 cases | Missing generation identity and usage | Infrastructure interruption; not a zero-cost or successful translation |', '',
 'The seven boundary failures each recorded **1,280 mismatches**, maximum absolute error **1.5020370483398438e-5**, and maximum scaled error **1.365878858816452**. The other eleven inputs passed. These metrics match the independently executed original Triton source discrepancy; matching finite-test metrics does not prove whole-domain equivalence.', '',
 '## Rerun and execution history','',
 '1. The initial pilot attempt stopped early with missing provider accounting. Its costs and failures were retained.',
 '2. The main 18-case pilot ran with fixed model/prompt/acceptance settings. Two AMD transpose cases encountered HTTP 429. At the user’s request, six remaining Intel cases ran in parallel.',
 '3. The user requested fresh attempts for the two interrupted AMD slots. The first launch failed the model’s IR-schema preflight before either case started; its cost was retained.',
 '4. The next launch passed preflight but exposed a fresh-database SQLite initialization race. The parallel runner was fixed to initialize WAL/schema before worker threads; clean checkpoints resumed without repeating preflight.',
 '5. Both AMD cases started concurrently. Direct hit another HTTP 429 on cycle 5. The hinted response was saved, and that case resumed to its ten-cycle limit without regenerating saved work.', '',
 '## All 18 case slots','',
 'The selected attempt supplies the outcome; “all-attempt tokens” also charges the original interrupted attempt. Preflight is accounted separately above.','',
 '| Backend | Kernel | Arm | Original → latest | Selected-attempt tokens | All-attempt tokens |',
 '| --- | --- | --- | --- | ---: | ---: |']
for t in targets:
    for kernel,label in kernels:
        for a in arms:
            r=lookup[t,kernel,a];status=status_words[r['selected_status']]
            if r['rerun_status']:status=status_words[r['original_status']]+' → '+status
            lines.append(f"| {target_labels[t]} | {label} | {arm_labels[a]} | {status} | {r['selected_tokens']:,} | {r['all_attempts_known_tokens']:,} |")
lines += ['', '## Scope, verification, and conclusion','',
 f"Configuration: **{config['provider']['model']}**, pinned **{config['provider']['endpoint']}** route, reasoning enabled; three kernel families, two backends, three arms, one repetition. Per-case limits remained {config['max_cycles']} cycles, {config['token_budget']:,} tokens and {config['active_seconds_budget']:,} active seconds, checked at stage boundaries. Prompts, sources, model settings and acceptance criteria were preserved across the replacement attempts.", '',
 'Verification completed previously: the Phase 1 full suite passed **256 tests with 25 skipped**. After the fresh-database fix, **36 focused tests passed**. For this consolidated report, source review hashes, selected outcomes, ledger charges, absence of pending requests, and chart/table totals were rechecked. No new model calls were made to build this report.', '',
 '**Conclusion:** this pilot identifies backend API reliability, repair behavior and numerical contracts as the main issues to address. It does not establish a general IR advantage, generalization to held-out families, or physical NPU performance. One AMD slot remains interrupted and historical accounting is incomplete. Phase 2’s complete-accounting gate remains unsatisfied; no later research phase was started.', '',
 '### Reproducibility files (optional)','',
 'This report contains the full reading view. For auditing only: [original audit](audit.json), [rerun audit](../amd-interruption-rerun/audit.json), [combined case table](../amd-interruption-rerun/resolved-pilot-cases.csv), and [consolidated report data](combined-data.json). Rebuild with `.venv/bin/python research/build_phase1_report.py`. Each chart is also saved as SVG beside its PNG.','']
(OUT/'PHASE1_REPORT.md').write_text('\n'.join(lines))
(OUT/'combined-data.json').write_text(json.dumps({'cases':resolved,'known_spending_usd':ledger['known_spent_usd'],'cost_by_campaign':{k:str(v) for k,v in costs.items()},'unknown_billing_records':unknown,'failure_stage_counts':dict(failures),'reviewed_attempts':20},indent=2)+'\n')
print('Wrote', OUT/'PHASE1_REPORT.md')
print('Validated 18 case slots, 20 reviewed attempts,',len(observed),'known charges,',len(unknown),'unresolved billing records.')
