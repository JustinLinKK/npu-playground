"""Run isolated Part 1 ablations; no provider request occurs without a run command."""
import argparse
import json
from pathlib import Path

from npu_agent.config import Settings, TARGETS
from npu_agent.providers import ExperimentCodexProvider, ExperimentOpenRouterProvider, create_provider
from npu_agent.validation import write_json

from .candidate_search import SelectionCompiler
from .runner import ARMS, ROOT, RunConfig, metrics, run
from .verified_skills import admit, verify_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='list the seven directions and comparison arms')
    launch = commands.add_parser('run', help='run one arm using the existing container evaluator')
    launch.add_argument('--direction', required=True, choices=ARMS)
    launch.add_argument('--arm', required=True)
    launch.add_argument('--source', type=Path, required=True)
    launch.add_argument('--manifest', type=Path, required=True)
    launch.add_argument('--target', choices=TARGETS, default='amd_xdna2_npu2')
    launch.add_argument('--provider', choices=('codex-cli', 'openai', 'openrouter'), required=True)
    launch.add_argument('--model', required=True)
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('--library', type=Path)
    launch.add_argument('--max-repairs', type=int, default=5)
    launch.add_argument('--population', type=int, default=4)
    launch.add_argument('--token-budget', type=int, default=100000)
    launch.add_argument('--selection-seed', type=int, default=70123)
    launch.add_argument('--compiler-timeout', type=int, default=180)
    launch.add_argument('--docker', default='docker')
    summarize = commands.add_parser('report', help='derive a comparison table from saved runs')
    summarize.add_argument('runs', nargs='+', type=Path)
    summarize.add_argument('--output', type=Path, required=True)
    admission = commands.add_parser('admit', help='admit a verified run to the local skill library')
    admission.add_argument('--run', type=Path, required=True)
    admission.add_argument('--library', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'list':
        print(json.dumps(ARMS, indent=2))
        return 0
    if args.command == 'admit':
        print(admit(args.run, args.library))
        return 0
    if args.command == 'report':
        if args.output.exists():
            parser.error('report output must be new')
        for path in args.runs:
            verify_evidence(path)
        states = [json.loads((path / 'report.json').read_text()) for path in args.runs]
        # Different kernels may form a cohort; per-case identities stay in each row.
        rows = [{**metrics(state), 'source_sha256': state['source_sha256'],
                 'model': state['config']['model'], 'token_budget': state['config']['token_budget'],
                 'fingerprint': (state['identity'] or {}).get('fingerprint')} for state in states]
        write_json(args.output, {'rows': rows, 'note': 'Raw per-run outcomes; not a controlled causal estimate.'})
        print(args.output)
        return 0
    config = RunConfig(direction=args.direction, arm=args.arm, model=args.model, max_repairs=args.max_repairs,
                       population=args.population, token_budget=args.token_budget, selection_seed=args.selection_seed)
    if args.compiler_timeout <= 0:
        parser.error('--compiler-timeout must be positive')
    if args.provider == 'codex-cli' and args.model != 'gpt-5.6-terra':
        parser.error('the isolated repository Codex provider is pinned to gpt-5.6-terra')
    if args.direction in ('feature_retrieval', 'verified_skills') and args.arm != 'off' and args.library is None:
        parser.error('retrieval arms require --library (an initially empty directory is allowed for verified_skills)')
    settings = Settings(repository_path=ROOT, runs_path=args.output, validation_policy='offline-validated',
                        compiler_timeout_seconds=args.compiler_timeout, compiler_cpus=2, compiler_memory='4g',
                        max_compiler_jobs=1, docker_executable=args.docker)
    if args.provider == 'codex-cli':
        provider = ExperimentCodexProvider(ROOT)
    elif args.provider == 'openrouter':
        provider = ExperimentOpenRouterProvider(ROOT)
        if args.model != provider.config.model:
            parser.error('--model must match the existing OpenRouter configuration')
    else:
        provider = create_provider(args.provider, ROOT, args.model, settings.provider_timeout_seconds)
    state = run(args.source, args.manifest, TARGETS[args.target], provider, SelectionCompiler(settings),
                config, args.output, library=args.library)
    print(json.dumps(metrics(state), indent=2))
    return 0 if state['status'] == 'completed' else 3 if state['status'] in ('blocked', 'interrupted') else 1


if __name__ == '__main__':
    raise SystemExit(main())
