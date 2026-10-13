"""Guidance and evidence classification for the separately frozen XDNA2 baseline."""
from __future__ import annotations

import json


def api_guidance(target) -> str:
    return """Container invocation and submission contract (revision 2, unguided):
Return exactly design.py and kernel.cc.
The emitter runs exactly: python design.py --dev npu2 --emit-mlir
--xclbin-path and --insts-path must NOT be required in emission mode. Accept them optionally.
Print only the target design to stdout. The harness compiles it and kernel.cc inside the pinned container;
do not invoke compilers or access a device from design.py. Resolve candidate files relative to its directory.
kernel.cc is compiled to kernel.o. Its entrypoints must use C linkage and match the submitted design.
Runtime buffer order, shapes and types must match the interface exactly.
The harness checks target compilation, host execution and dataflow simulation against an independent reference.
Evaluator diagnostics and your previous candidate are returned on failed attempts. No physical device is used.
"""


def classify_attempt(cycle: dict) -> dict:
    """Count one primary failure; prerequisite and physical-execution events are not failures."""
    if cycle.get("status") == "passed":
        return {"category": "passed", "kernel_failure": False, "noise": False, "stage": ""}
    validation = cycle.get("compile_result", {}).get("validation") or cycle.get("feedback", {}).get("validation", {})
    stages = validation.get("stages", {})
    for name in ("oracle_validation", "design_emission", "target_compile", "host_execution", "dataflow_simulation"):
        stage = stages.get(name, {})
        if stage.get("status") not in ("failed", "unsupported", "blocked"):
            continue
        if stage.get("reason_code") == "PREREQUISITE_UNAVAILABLE":
            continue
        diagnostic = json.dumps(stage).lower()
        category, kernel, noise = "unclassified", False, False
        if name == "oracle_validation":
            category, noise = "reference_failure", True
        elif stage.get("status") == "blocked":
            category, noise = "infrastructure", True
        elif name == "design_emission" and any(s in diagnostic for s in (
                "required", "unrecognized arguments", "importerror", "modulenotfounderror", "attributeerror", "typeerror")):
            category, noise = "emitter_api_contract", True
        elif name == "target_compile" and any(s in diagnostic for s in (
                "please submit a bug report", "unable to legalize", "llvm error", "segmentation fault",
                "clang frontend command failed with exit code 70", "clang frontend command failed with exit code 134")):
            category, noise = "toolchain_crash", True
        elif stage.get("status") == "unsupported":
            category, noise = "simulator_coverage", True
        elif name == "target_compile":
            category, kernel = "target_source_or_verifier", True
        elif stage.get("error_category") == "numerical_mismatch":
            category, kernel = "numerical_mismatch", True
        elif name in ("host_execution", "dataflow_simulation") and any(s in diagnostic for s in (
                "addresssanitizer", "runtime error:", "immutable input", "simulation_deadlock", "dma bounds", "abi",
                "dma token count or element size mismatch")):
            category, kernel = "memory_abi_or_dataflow", True
        elif name == "design_emission":
            category, kernel = "design_source", True
        return {"category": category, "kernel_failure": kernel, "noise": noise, "stage": name}
    calls = list(cycle.get("calls", {}).values())
    if any(call.get("status") in ("blocked", "in_flight") for call in calls):
        return {"category": "provider_or_interruption", "kernel_failure": False, "noise": True, "stage": "provider"}
    if cycle.get("status") == "running" and cycle.get("bundle") and not cycle.get("compile_result"):
        return {"category": "not_evaluated", "kernel_failure": False, "noise": True, "stage": "validation"}
    return {"category": "model_output_or_unclassified", "kernel_failure": False, "noise": False, "stage": "generation"}


def build_report(root):
    import csv
    from collections import Counter

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .study_v4 import StudyConfig
    from .study_v4_reporting import case_metrics
    from .validation import write_json

    campaign = json.loads((root / 'campaign.json').read_text())
    config = StudyConfig.model_validate(campaign['config'])
    direct_openai = config.provider.model.startswith('gpt-')
    cases, attempts = [], []
    for case in campaign['cases']:
        path = root / 'cases' / case['id'] / 'report.json'
        state = json.loads(path.read_text()) if path.exists() else None
        row = case_metrics(case, state, config)
        row.update(first_attempt=False, cost_usd=0., cost_complete=state is not None,
                   report=f'cases/{case["id"]}/report.json', terminal_reason=state.get('terminal_reason') if state else None)
        if state is None and campaign['status'] == 'blocked':
            row.update(status='not_run', terminal_reason='campaign prerequisite blocked')
        if state:
            row['first_attempt'] = bool(state['cycles'] and state['cycles'][0].get('status') == 'passed')
            for cycle in state['cycles']:
                category = classify_attempt(cycle)
                costs = [c.get('metadata', {}).get('usage', {}).get('cost') for c in cycle['calls'].values()]
                cost_complete = bool(costs) and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in costs)
                cost = sum(c for c in costs if isinstance(c, (int, float)) and not isinstance(c, bool))
                row['cost_usd'] += cost
                row['cost_complete'] &= cost_complete
                attempts.append({'kernel': case['kernel'], 'attempt': cycle['number'], 'status': cycle['status'],
                                 **category, 'cost_usd': None if direct_openai else cost, 'cost_complete': cost_complete, 'report': row['report']})
        if direct_openai:
            row.update(cost_usd=None, cost_complete=False)
        cases.append(row)
    directory = root / 'results'
    directory.mkdir(exist_ok=True)
    for name, rows in [('tasks', cases), ('attempts', attempts), ('exclusions', [a for a in attempts if a['noise']])]:
        with (directory / f'{name}.csv').open('w', newline='') as stream:
            fields = list(rows[0]) if rows else ['kernel', 'attempt', 'category', 'report']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    counts = Counter(row['status'] for row in cases)
    failures = Counter(a['category'] for a in attempts if a['kernel_failure'])
    exclusions = Counter(a['category'] for a in attempts if a['noise'])
    external_budget = campaign.get('budget_enforcement') in ('openrouter', 'openai')
    ledger_path = root.parent / 'budget.json'
    ledger = json.loads(ledger_path.read_text()) if not external_budget and ledger_path.exists() else {}
    summary = {'protocol': config.protocol, 'status': campaign['status'], 'planned': len(cases),
               'budget_enforcement': campaign.get('budget_enforcement', 'local'),
               'blocker': campaign.get('blocker'),
               'attempted_tasks': sum(row['cycles'] > 0 for row in cases),
               'first_attempt_passes': sum(row['first_attempt'] for row in cases),
               'offline_passes': sum(row['offline_pass'] for row in cases),
               'budget_successes': sum(row['success'] for row in cases), 'statuses': dict(counts),
               'known_evaluation_tokens': sum(row['known_tokens'] for row in cases),
               'known_evaluation_cost_usd': None if direct_openai else sum(row['cost_usd'] for row in cases),
               'known_total_spend_usd': ledger.get('known_spent_usd'),
               'billing_complete': None if external_budget else not ledger.get('unknown_spending', True) and not ledger.get('pending_calls') and not ledger.get('pending_call'),
               'kernel_failure_attempts': dict(failures), 'excluded_attempts': dict(exclusions),
               'unclassified_attempts': sum(a['category'] in ('unclassified', 'model_output_or_unclassified') for a in attempts)}
    write_json(root / 'summary.json', summary)
    write_json(directory / 'summary.json', summary)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    xs = list(range(1, config.max_cycles + 1))
    axes[0, 0].step(xs, [sum(r['success'] and r['cycles'] <= n for r in cases) for n in xs], where='post')
    axes[0, 0].set(title='Offline success by attempt', xlabel='Maximum attempts', ylabel=f'Accepted / {len(cases)}', ylim=(0, len(cases)))
    axes[0, 1].bar(list(counts), list(counts.values()), color='#497fa8')
    axes[0, 1].set(title='All planned task outcomes', ylabel='Tasks')
    axes[1, 0].barh(list(failures) or ['No confirmed kernel failures'], list(failures.values()) or [0], color='#b86955')
    axes[1, 0].set(title='Confirmed kernel failures', xlabel='Attempts, including later repairs')
    if direct_openai:
        axes[1, 1].text(.5, .5, 'Billed dollars unavailable in OpenAI responses\nCheck billing on OpenAI',
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=13)
        axes[1, 1].set_axis_off()
    else:
        axes[1, 1].scatter([r['known_tokens'] for r in cases], [r['cost_usd'] for r in cases],
                           c=['#3a8562' if r['success'] else '#b86955' for r in cases])
        axes[1, 1].set(title='Recorded evaluation usage', xlabel='Tokens per task', ylabel='Recorded dollars per task')
    model_label = config.provider.model if direct_openai else 'Qwen3.8-27B'
    if not summary['attempted_tasks']:
        for axis, message in [(axes[0, 0], 'Success rate not measured\nNo evaluation attempts'),
                              (axes[1, 0], 'No kernel failure evidence\nNo evaluation attempts'),
                              (axes[1, 1], 'No evaluation usage\nCalibration billing is reported separately')]:
            axis.clear()
            axis.text(.5, .5, message, ha='center', va='center', transform=axis.transAxes, fontsize=13)
            axis.set_axis_off()
        figure.suptitle(f'{model_label} → XDNA2 | {len(cases)} tasks prepared | Evaluation NOT RUN')
    else:
        figure.suptitle(f'{model_label} → XDNA2 | {summary["budget_successes"]}/{len(cases)} accepted | {campaign["status"]}')
    figure.tight_layout()
    for extension in ('png', 'svg'):
        figure.savefig(directory / f'baseline.{extension}', dpi=180)
    plt.close(figure)
    lines = ['# XDNA2 direct baseline', '',
             ('**NOT RUN: calibration/provider prerequisites blocked every evaluation task.**' if not attempts and campaign['status'] == 'blocked'
              else f'**{summary["budget_successes"]}/{len(cases)} planned tasks accepted**, '
                   f'{summary["first_attempt_passes"]} on the first attempt. Campaign: **{campaign["status"]}**.'), '',
             campaign.get('blocker', ''), '',
             f'Model `{config.provider.model}`, route `{config.provider.endpoint}`, reasoning enabled. '
             f'At most {config.max_cycles} attempts and {config.token_budget:,} tokens per task.', '',
             f'Recorded evaluation: **{summary["known_evaluation_tokens"]:,} tokens**, ' +
             ('billed dollars unavailable in OpenAI responses; check billing, including calibration/preflight, on OpenAI.'
              if direct_openai else f'**${summary["known_evaluation_cost_usd"]:.4f}**. ' +
             ('No shared spending ledger is enforced; total billing, including calibration/preflight, must be checked on OpenRouter.'
              if external_budget else
              f'Shared ledger (including calibration/preflight): **${summary["known_total_spend_usd"]}**. '
              f'Billing complete: **{summary["billing_complete"]}**.')), '',
             ('Direct OpenAI has no local dollar cap; attempt, token and time limits remain enforced. '
              if direct_openai else
              'Dollar-budget enforcement is delegated to OpenRouter; no local dollar threshold or missing-cost stop is applied. '
              if external_budget else 'The $50 threshold stops new requests; up to four outstanding requests can settle afterward. ') +
             'Pending, blocked and excluded cases remain in the planned denominator. '
             'Token ceilings can end a case before attempt ten.', '',
             'Acceptance means target compilation plus independent host/dataflow correctness on twelve inputs. '
             'The emitted physical NPU binary was not executed; NPU performance is unmeasured. '
             'KernelBench inputs are CUDA adaptations with reduced shapes, not original performance workloads. '
             'Classic sources/manifests remain unchanged; their original source-equivalence limitations remain.', '',
             '![Baseline results](baseline.png)', '',
             '## Kernel-only failure analysis', '',
             'Counts describe failed attempts, including attempts subsequently repaired. Kernel failures include '
             'generated C++, target source/verifier errors, numerical mismatches, and evidenced memory/dataflow defects. '
             'Exclusion never converts an attempt into a pass or refunds its cost.', '',
             '| Category | Attempts |', '| --- | ---: |']
    lines += [f'| {name} | {count} |' for name, count in failures.items()]
    lines += ['', '## Excluded and unresolved evidence', '', '| Category | Attempts |', '| --- | ---: |']
    lines += [f'| {name} | {count} |' for name, count in exclusions.items()]
    lines += [f'| Unclassified/model-output failures | {summary["unclassified_attempts"]} |', '',
              '`not_evaluated` records contain generated code without a compiler verdict; they are not kernel failures. '
              'Physical-execution blocking and downstream prerequisite events are not additional failed attempts. '
              'Raw diagnostics remain linked in [attempts.csv](attempts.csv) and [exclusions.csv](exclusions.csv).', '',
              '## All tasks', '', '| Task | Status | Attempts | Tokens | Cost | Evidence |', '| --- | --- | ---: | ---: | ---: | --- |']
    for row in cases:
        cost_label = 'unavailable' if direct_openai else f'${row["cost_usd"]:.4f}'
        lines.append(f'| {row["kernel"]} | {row["status"]} | {row["cycles"]} | {row["known_tokens"]:,} | '
                     f'{cost_label} | [report](../{row["report"]}) |')
    lines += ['', 'Full data: [tasks.csv](tasks.csv), [attempts.csv](attempts.csv), [summary.json](summary.json).',
              'Calibration and source checks are separate directories beside the evaluation campaign. '
              'Per-case active time includes queueing and is not campaign wall time under concurrency.', '']
    (directory / 'REPORT.md').write_text('\n'.join(lines))
    return summary
