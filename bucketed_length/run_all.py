"""Run all seven directions in separate processes, then render every planned result."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock

from npu_agent.baseline import _load_inputs
from npu_agent.config import TARGETS
from npu_agent.models import TranslationRequest
from npu_agent.validation import json_digest, write_json

from .runner import ARMS, ROOT, RunConfig
from .verified_skills import load


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--source', type=Path, default=ROOT / 'examples/classic/01_cuda_vector_add/kernel.cu')
    result.add_argument('--manifest', type=Path, default=ROOT / 'tests/fixtures/backends/amd_xdna2_npu2/add/manifest.json')
    result.add_argument('--target', choices=TARGETS, default='amd_xdna2_npu2')
    result.add_argument('--provider', choices=('codex-cli', 'openai', 'openrouter'), default='codex-cli')
    result.add_argument('--model', default='gpt-5.6-terra')
    result.add_argument('--workers', type=int, default=7, help='parallel directions; one child process per active direction')
    result.add_argument('--repetitions', type=int, default=1)
    result.add_argument('--library', type=Path, help='verified seed library, copied independently for every case')
    result.add_argument('--max-repairs', type=int, default=5)
    result.add_argument('--population', type=int, default=4)
    result.add_argument('--token-budget', type=int, default=100000, help='per case, including the whole search population')
    result.add_argument('--selection-seed', type=int, default=70123)
    result.add_argument('--compiler-timeout', type=int, default=180)
    result.add_argument('--docker', default='docker')
    result.add_argument('--dry-run', action='store_true', help='write commands and render NOT RUN rows without launching children')
    result.add_argument('--render-only', action='store_true', help='regenerate reports for an existing suite; no experiments')
    return result


def prepare(args):
    if not 1 <= args.workers <= 7 or args.repetitions < 1 or args.compiler_timeout < 1:
        raise ValueError('workers must be 1..7; repetitions and compiler timeout must be positive')
    RunConfig(direction='candidate_search', arm='best_of_n', model=args.model, max_repairs=args.max_repairs,
              population=args.population, token_budget=args.token_budget, selection_seed=args.selection_seed)
    if args.provider == 'codex-cli' and args.model != 'gpt-5.6-terra':
        raise ValueError('the isolated Codex provider is pinned to gpt-5.6-terra')
    output = args.output.resolve()
    if output.is_relative_to(ROOT / 'bucketed_length'):
        raise ValueError('store suite outputs outside bucketed_length/, for example under runs/')
    request = TranslationRequest(source_path=str(args.source), manifest_path=str(args.manifest), targets=[TARGETS[args.target]])
    manifest, source = _load_inputs(request)
    if args.library is not None and not args.library.is_dir():
        raise ValueError('--library must be an existing verified library directory; omit it to start empty')
    entries = load(args.library)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'
    inputs.mkdir()
    source_path = inputs / ('kernel' + args.source.suffix)
    source_path.write_text(source, encoding='utf-8')
    manifest_path = inputs / 'manifest.json'
    write_json(manifest_path, manifest.model_dump(mode='json'))
    (output / 'logs').mkdir()
    config = {key: value for key, value in vars(args).items() if key not in ('output', 'source', 'manifest', 'library')}
    suite = {'protocol': 'bucketed-length-parallel-v1', 'created_at': datetime.now(UTC).isoformat(),
             'config': config, 'status': 'planned', 'wall_seconds': 0.0, 'cases': [],
             'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
             'manifest_sha256': json_digest(manifest.model_dump(mode='json')),
             'seed_library_ids': [entry['id'] for entry in entries],
             'case_token_cap_sum': sum(map(len, ARMS.values())) * args.repetitions * args.token_budget}
    for direction, arms in ARMS.items():
        for repetition in range(1, args.repetitions + 1):
            for arm in arms:
                case_id = f'{direction}--{arm}--r{repetition}'
                library = output / 'libraries' / case_id
                library.mkdir(parents=True)
                for entry in entries:
                    write_json(library / f'{entry["id"]}.json', entry)
                command = [sys.executable, '-m', 'bucketed_length', 'run', '--direction', direction, '--arm', arm,
                    '--source', str(source_path), '--manifest', str(manifest_path), '--target', args.target,
                    '--provider', args.provider, '--model', args.model, '--output', str(output / 'cases' / case_id),
                    '--max-repairs', str(args.max_repairs), '--population', str(args.population),
                    '--token-budget', str(args.token_budget), '--selection-seed', str(args.selection_seed),
                    '--compiler-timeout', str(args.compiler_timeout), '--docker', args.docker]
                if direction in ('feature_retrieval', 'verified_skills'):
                    command += ['--library', str(library)]
                suite['cases'].append({'id': case_id, 'direction': direction, 'arm': arm, 'repetition': repetition,
                    'command': command, 'process_status': 'not_run', 'returncode': None, 'elapsed_seconds': None})
    write_json(output / 'suite.json', suite)
    return output, suite


def execute_case(case, output):
    started = time.perf_counter()
    environment = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment.setdefault(name, '1')
    try:
        with (output / 'logs' / f'{case["id"]}.log').open('w') as log:
            # Process isolation is required by the private-input evaluator's scoped patch.
            completed = subprocess.run(case['command'], cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, check=False)
        result = {'process_status': 'finished', 'returncode': completed.returncode}
    except OSError as exc:
        result = {'process_status': 'process_error', 'returncode': None, 'error': str(exc)}
    return {**result, 'elapsed_seconds': time.perf_counter() - started}


def launch(output, suite, stop=None):
    stop = stop if stop is not None else Event()
    lock = Lock()
    started = time.perf_counter()
    suite['status'] = 'running'
    write_json(output / 'suite.json', suite)

    def direction_worker(direction):
        for case in suite['cases']:
            if case['direction'] != direction:
                continue
            with lock:
                if stop.is_set():
                    break
                case['process_status'] = 'running'
                write_json(output / 'suite.json', suite)
                print(f'Starting {case["id"]}', flush=True)
            result = execute_case(case, output)
            with lock:
                case.update(result)
                suite['wall_seconds'] = time.perf_counter() - started
                write_json(output / 'suite.json', suite)
                print(f'Finished {case["id"]}: exit {case["returncode"]}', flush=True)

    try:
        with ThreadPoolExecutor(max_workers=suite['config']['workers']) as pool:
            futures = [pool.submit(direction_worker, direction) for direction in ARMS]
            try:
                for future in as_completed(futures):
                    future.result()
            except Exception as exc:
                suite['error'] = str(exc)
                stop.set()
                raise
    finally:
        suite['status'] = 'error' if suite.get('error') else 'interrupted' if stop.is_set() else 'finished'
        suite['wall_seconds'] = time.perf_counter() - started
        write_json(output / 'suite.json', suite)
    return suite


def main(argv=None):
    args = parser().parse_args(argv)
    from .render import render

    if args.render_only:
        render(args.output.resolve())
        print(args.output.resolve() / 'results/REPORT.md')
        return 0
    output, suite = prepare(args)
    print(f'Suite: {output}\n{len(suite["cases"])} cases; {args.workers} parallel directions; '
          f'{args.token_budget:,} accounted tokens per case.', flush=True)
    if not suite['seed_library_ids']:
        print('Seed library is empty; retrieval arms will have no prior examples.', flush=True)
    if args.dry_run:
        suite['status'] = 'dry_run'
        write_json(output / 'suite.json', suite)
        summary = render(output)
    else:
        stop = Event()

        def request_stop(signum, frame):
            stop.set()
            print('Stopping new cases; active cases will finish before partial results are rendered.', flush=True)

        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            launch(output, suite, stop)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            summary = render(output)
    print(output / 'results/REPORT.md', flush=True)
    if args.dry_run:
        return 0
    if suite['status'] == 'interrupted':
        return 130
    return 0 if all(row['status'] == 'completed' for row in summary['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
