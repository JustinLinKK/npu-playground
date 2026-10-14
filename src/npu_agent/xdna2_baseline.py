"""Guidance and evidence classification for the separately frozen XDNA2 and Intel NPU baselines."""
from __future__ import annotations

import json

TARGET_LABELS = {"amd_xdna2_npu2": "XDNA2", "intel_npu_3720": "Intel NPU 3720"}
CASES_PER_TASK = 12  # five seeded normal cases plus seven boundary families (testcases.generate_cases)
# OpenVINO port checks in the trusted Intel graph tools (playground/tools/intel_execute_graph.py).
GRAPH_INTERFACE_ERRORS = ("port count mismatch", "ambiguous or unnamed", "port shape/dtype mismatch",
                          "duplicate port identity", "custom graph extensions are unsupported")


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
    for name in ("oracle_validation", "design_emission", "graph_construction", "target_compile", "host_execution",
                 "dataflow_simulation"):
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
        elif any(s in diagnostic for s in GRAPH_INTERFACE_ERRORS):
            category, kernel = "graph_interface", True
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
        elif name == "graph_construction":
            category, kernel = "graph_source", True
        return {"category": category, "kernel_failure": kernel, "noise": noise, "stage": name}
    calls = list(cycle.get("calls", {}).values())
    if any(call.get("status") in ("blocked", "in_flight") for call in calls):
        return {"category": "provider_or_interruption", "kernel_failure": False, "noise": True, "stage": "provider"}
    if cycle.get("status") == "running" and cycle.get("bundle") and not cycle.get("compile_result"):
        return {"category": "not_evaluated", "kernel_failure": False, "noise": True, "stage": "validation"}
    return {"category": "model_output_or_unclassified", "kernel_failure": False, "noise": False, "stage": "generation"}


def fp16_from_outputs(stage: dict, compiled: dict) -> dict | None:
    """Score saved physical-NPU outputs with the current FP16 tolerances; None when they were not saved."""
    from pathlib import Path

    import numpy as np

    from .executors import compare_fp16, fp16_reference
    from .models import KernelManifest
    from .testcases import generate_cases
    from .validation import sha256

    path = compiled.get("artifacts", {}).get("validation.json")
    if not path:
        return None
    run = Path(path).parent
    manifest_path = run / "contract" / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = KernelManifest.model_validate_json(manifest_path.read_text())
    tests = generate_cases(manifest)
    passed, worst, flushed = 0, 0.0, 0
    for test in tests:
        name = f"target_execution/{test.id}.npz"
        output = run / name
        # Only the exact tensors hashed at execution time count as evidence.
        if not output.is_file() or stage.get("artifacts", {}).get(name) != sha256(output):
            return None
        with np.load(output, allow_pickle=False) as archive:
            actual = {key: archive[key] for key in archive.files}
        inputs = {**test.inputs, **{scalar.name: scalar.value for scalar in manifest.scalars}}
        result = compare_fp16(actual, fp16_reference(manifest, inputs), manifest.tolerance, test.id)
        passed += bool(result["correct"])
        for item in result["outputs"].values():
            worst = max(worst, item["max_scaled_error"] if item["max_scaled_error"] is not None else float("inf"))
            flushed += item["flushed_subnormals"]
    return {"npu_fp16_correct": passed == len(tests), "npu_fp16_cases_passed": passed,
            "npu_fp16_max_scaled_error": worst, "npu_flushed_subnormals": flushed}


def npu_observation(cycle: dict | None) -> dict:
    """Physical-NPU outcome recorded beside an attempt; it never contributes to acceptance."""
    from .executors import FP16_VERDICT

    compiled = (cycle or {}).get("compile_result") or {}
    stage = (compiled.get("validation") or {}).get("stages", {}).get("target_execution")
    empty = {"npu_correct": None, "npu_max_abs_err": None, "npu_cases_run": 0, "npu_cases_passed": None,
             "npu_fp16_correct": None, "npu_fp16_cases_passed": None, "npu_fp16_max_scaled_error": None,
             "npu_flushed_subnormals": None}
    if not stage or stage.get("status") in ("blocked", "not_requested"):
        return {"npu_status": "not_run", **empty}
    details = stage.get("details", {})
    physical = stage.get("engine") == "physical_npu"
    errors = [output["max_absolute_error"] for comparison in details.get("comparisons", [])
              for output in comparison.get("outputs", {}).values() if output.get("max_absolute_error") is not None]
    row = {**empty, "npu_status": stage["status"] if physical else "execution_error",
           "npu_correct": stage.get("correct"), "npu_max_abs_err": max(errors) if errors else None,
           "npu_cases_run": stage.get("cases_run", 0),
           "npu_cases_passed": sum(c.get("correct") is True for c in details.get("comparisons", [])) if errors else None}
    # Prefer rescoring the saved outputs with today's tolerances; accept a recorded verdict only if it used them.
    fp16 = fp16_from_outputs(stage, compiled) if physical else None
    if fp16 is None and physical and details.get("fp16_verdict") == FP16_VERDICT:
        fp16 = {"npu_fp16_correct": details.get("fp16_correct"), "npu_fp16_cases_passed": details.get("fp16_cases_passed")}
    return {**row, **(fp16 or {})}


def npu_headline(summary: dict, planned: int) -> str:
    tasks = summary['physical_npu']['tasks']
    verdict = summary['physical_npu']['fp16_verdict']
    if tasks['npu_verified_fp16'] is None:
        return f'Physical NPU: the FP16 verdict was not recorded; {tasks["npu_verified_manifest"]}/{planned} accepted tasks passed the manifest tolerance.'
    return (f'**{tasks["npu_verified_fp16"]}/{planned} planned tasks accepted and verified on the physical NPU** '
            f'(FP16 verdict, rtol {verdict["rtol"]:g}, atol {verdict["atol"]:g}); '
            f'{tasks["npu_verified_manifest"]}/{planned} also meet the float32 manifest tolerance on the NPU.')


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
    target = config.targets[0]
    label = TARGET_LABELS.get(target, target)
    # Physical-NPU columns exist only for campaigns frozen with a configured hardware runner.
    observed = bool(campaign.get('identity', {}).get('targets', {}).get(target, {}).get('hardware_runner'))
    cases, attempts = [], []
    for case in campaign['cases']:
        path = root / 'cases' / case['id'] / 'report.json'
        state = json.loads(path.read_text()) if path.exists() else None
        row = case_metrics(case, state, config)
        row.update(first_attempt=False, cost_usd=0., cost_complete=state is not None,
                   report=f'cases/{case["id"]}/report.json', terminal_reason=state.get('terminal_reason') if state else None)
        if state is None and campaign['status'] == 'blocked':
            row.update(status='not_run', terminal_reason='campaign prerequisite blocked')
        if observed:
            accepted = next((c for c in state['cycles'] if c.get('status') == 'passed'), None) if state else None
            row.update(npu_observation(accepted))
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
                if observed:
                    host = cycle.get('compile_result', {}).get('validation', {}).get('stages', {}).get('host_execution', {})
                    attempts[-1].update(npu_observation(cycle), host_correct=host.get('correct'))
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
    if observed:
        from .executors import FP16_VERDICT

        executed = [a for a in attempts if a['npu_status'] in ('passed', 'failed')]
        accepted = [row for row in cases if row['status'] == 'completed']
        measured = [row for row in accepted if row['npu_status'] in ('passed', 'failed')]
        fp16_known = all(row['npu_fp16_cases_passed'] is not None for row in measured)

        def rate(passed, total):
            return {'passed': passed, 'total': total, 'rate': passed / total if total else None}

        # "Of planned": a task or case without an accepted, NPU-executed candidate counts as a failure.
        planned_cases, accepted_cases = CASES_PER_TASK * len(cases), sum(row['npu_cases_run'] for row in measured)
        manifest_cases = sum(row['npu_cases_passed'] or 0 for row in measured)
        fp16_cases = sum(row['npu_fp16_cases_passed'] or 0 for row in measured) if fp16_known else None
        summary['physical_npu'] = {
            'fp16_verdict': FP16_VERDICT,
            'executed_attempts': len(executed),
            'execution_errors': sum(a['npu_status'] == 'execution_error' for a in attempts),
            'tasks': {'planned': len(cases), 'accepted': len(accepted), 'accepted_and_executed': len(measured),
                      'npu_verified_manifest': sum(row['npu_correct'] is True for row in measured),
                      'npu_verified_fp16': sum(row['npu_fp16_correct'] is True for row in measured) if fp16_known else None},
            'case_pass_rate': {
                'manifest_tolerance': {'of_planned': rate(manifest_cases, planned_cases), 'of_accepted': rate(manifest_cases, accepted_cases)},
                'fp16_verdict': ({'of_planned': rate(fp16_cases, planned_cases), 'of_accepted': rate(fp16_cases, accepted_cases)}
                                 if fp16_known else None)},
            'flushed_subnormals': sum(row['npu_flushed_subnormals'] or 0 for row in measured),
            'cpu_npu_agreement': dict(Counter(f"cpu_{'pass' if a['host_correct'] else 'fail'}/npu_{'pass' if a['npu_correct'] else 'fail'}"
                                              for a in executed)),
            'cpu_npu_fp16_agreement': dict(Counter(f"cpu_{'pass' if a['host_correct'] else 'fail'}/npu_fp16_{'pass' if a['npu_fp16_correct'] else 'fail'}"
                                                   for a in executed)) if fp16_known else None}
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
        figure.suptitle(f'{model_label} → {label} | {len(cases)} tasks prepared | Evaluation NOT RUN')
    else:
        verified = (f' | {summary["physical_npu"]["tasks"]["npu_verified_fp16"]} also NPU-verified (FP16)'
                    if observed and summary['physical_npu']['tasks']['npu_verified_fp16'] is not None else '')
        figure.suptitle(f'{model_label} → {label} | {summary["budget_successes"]}/{len(cases)} accepted{verified} | {campaign["status"]}')
    figure.tight_layout()
    for extension in ('png', 'svg'):
        figure.savefig(directory / f'baseline.{extension}', dpi=180)
    plt.close(figure)
    lines = [f'# {label} direct baseline', '',
             ('**NOT RUN: calibration/provider prerequisites blocked every evaluation task.**' if not attempts and campaign['status'] == 'blocked'
              else f'**{summary["budget_successes"]}/{len(cases)} planned tasks accepted**, '
                   f'{summary["first_attempt_passes"]} on the first attempt. Campaign: **{campaign["status"]}**.'), '',
             npu_headline(summary, len(cases)) if observed and attempts else '', '',
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
             'Acceptance means target compilation plus independent ' +
             ('host/dataflow correctness' if target.startswith('amd_') else 'OpenVINO CPU graph correctness') +
             ' on twelve inputs. ' +
             ('Every compiled attempt was also executed on the physical NPU without affecting acceptance; see Physical NPU verification. '
              if observed else 'The emitted physical NPU binary was not executed; NPU performance is unmeasured. ') +
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
              'Raw diagnostics remain linked in [attempts.csv](attempts.csv) and [exclusions.csv](exclusions.csv).', '']
    if observed:
        npu = summary['physical_npu']
        tasks, rates, verdict = npu['tasks'], npu['case_pass_rate'], npu['fp16_verdict']

        def cell(value):
            if value is None or value['total'] == 0:
                return 'not recorded'
            return f"{value['passed']}/{value['total']} ({value['rate']:.1%})"

        def task_cell(count, total):
            return 'not recorded' if count is None else cell({'passed': count, 'total': total, 'rate': count / total if total else None})

        lines += ['## Physical NPU verification', '',
                  '**How to read this section.** Acceptance above is decided offline: target compilation plus independent '
                  'OpenVINO CPU float32 correctness. It drives the repair loop and is unchanged by the NPU. Every compiled '
                  'attempt also ran its exact exported blob on the physical NPU. A task is **NPU-verified** when it was '
                  'accepted and its accepted candidate passes on the NPU. "Of planned" counts tasks or cases without an '
                  'accepted candidate as failures; "of accepted" covers accepted candidates only.', '',
                  '**Two NPU verdicts.** The *manifest* verdict reuses the float32 tolerance; the NPU computes in FP16, so '
                  'it is expected to fail on most inputs. The *FP16* verdict compares the NPU with the independent oracle '
                  'evaluated on FP16-rounded inputs, with its result rounded to FP16, elementwise as '
                  f"|NPU − reference| ≤ {verdict['atol']:g} + {verdict['rtol']:g}·|reference|. "
                  'It is rescored from the saved NPU outputs whenever this report is built.', '',
                  f'| Measure | Of planned ({tasks["planned"]}) | Of accepted ({tasks["accepted"]}) |', '| --- | ---: | ---: |',
                  f'| Tasks accepted offline | {task_cell(tasks["accepted"], tasks["planned"])} | — |',
                  f'| **Tasks NPU-verified, FP16 verdict** | {task_cell(tasks["npu_verified_fp16"], tasks["planned"])} | '
                  f'{task_cell(tasks["npu_verified_fp16"], tasks["accepted"])} |',
                  f'| Tasks NPU-verified, manifest tolerance | {task_cell(tasks["npu_verified_manifest"], tasks["planned"])} | '
                  f'{task_cell(tasks["npu_verified_manifest"], tasks["accepted"])} |']
        for name, key in (('FP16 verdict', 'fp16_verdict'), ('manifest tolerance', 'manifest_tolerance')):
            value = rates[key] or {}
            lines.append(f'| Input cases passing on the NPU, {name} | {cell(value.get("of_planned"))} | {cell(value.get("of_accepted"))} |')
        lines += ['', '| Diagnostic | Count |', '| --- | ---: |',
                  f'| Attempts executed on the NPU | {npu["executed_attempts"]} |',
                  f'| NPU execution errors (executor or device, not kernel) | {npu["execution_errors"]} |',
                  f'| FP16 subnormal reference values the NPU returned as zero (accepted candidates) | {npu["flushed_subnormals"]} |']
        for key, count in sorted((npu['cpu_npu_fp16_agreement'] or npu['cpu_npu_agreement']).items()):
            cpu, device = key.split('/')
            lines.append(f'| Executed attempts: CPU {cpu.split("_")[-1]}, NPU {device.split("_")[-1]}'
                         f'{" (FP16)" if npu["cpu_npu_fp16_agreement"] else " (manifest)"} | {count} |')
        lines.append('')
    lines += ['## All tasks', '', '| Task | Status | Attempts | Tokens | Cost | ' +
              ('NPU cases passed: FP16 / manifest | ' if observed else '') +
              'Evidence |', '| --- | --- | ---: | ---: | ---: | ' + ('--- | ' if observed else '') + '--- |']
    for row in cases:
        cost_label = 'unavailable' if direct_openai else f'${row["cost_usd"]:.4f}'
        npu_label = ''
        if observed:
            run = row['npu_cases_run']
            if row['status'] != 'completed':
                counts = 'not accepted'
            elif row['npu_cases_passed'] is None:
                counts = row['npu_status']
            else:
                fp16 = f'{row["npu_fp16_cases_passed"]}/{run}' if row['npu_fp16_cases_passed'] is not None else 'n/a'
                counts = f'{fp16} / {row["npu_cases_passed"]}/{run}'
            npu_label = counts + ' | '
        lines.append(f'| {row["kernel"]} | {row["status"]} | {row["cycles"]} | {row["known_tokens"]:,} | '
                     f'{cost_label} | {npu_label}[report](../{row["report"]}) |')
    lines += ['', 'Full data: [tasks.csv](tasks.csv), [attempts.csv](attempts.csv), [summary.json](summary.json).',
              'Calibration and source checks are separate directories beside the evaluation campaign. '
              'Per-case active time includes queueing and is not campaign wall time under concurrency.', '']
    (directory / 'REPORT.md').write_text('\n'.join(lines))
    return summary
