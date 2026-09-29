# Case study v3: prompt guidance and intermediate representations

This campaign separates curated prompt guidance from the IR workflow. It uses
`qwen/qwen3-coder-30b-a3b-instruct` through OpenRouter for every generation call,
with the `siliconflow/fp8` endpoint pinned and provider fallback disabled.
The existing v2 launcher retains its Terra protocol and five-arm defaults.

| Arm | Initial backend context | Intermediate workflow |
| --- | --- | --- |
| `baseline_minimal` | Task and submission requirements only | Direct source translation |
| `baseline` | Existing v2 backend and numerical guidance | Direct source translation |
| `structured_ir` | Same guidance as `baseline` | Latest schema-2.0 executable IR; validate and reuse passing IR |
| `hinted_ir` | Same guidance as `baseline` | Free-form intermediate; regenerate after failure |

The minimal baseline retains source, manifest, target/compiler versions, tensor
contracts, tolerances, required filenames/entrypoints and evaluator-supported
constructs. It omits API usage recipes, transport examples, portability helpers
and the layer-normalization stable-centering recipe. Repair calls still receive
the same compact evaluator diagnostics and their own previous candidate. Thus it
measures an agent without curated initial guidance, not a knowledge-free model.

The other three arms retain their v2 prompts and workflows. Structured IR also
receives its schema and executable operation meanings, which are part of that
method. Its validation and automatic reuse remain enabled. No cross-case memory,
retrieval, tools, or prior campaign candidates are supplied to any arm.

Default size: ten existing development kernels x two targets x four arms x three
independent repetitions = **240 cases**. Each case has at most ten cycles,
including the first attempt: at most 2,400 cycles and 3,600 generation calls.
Equal cycle budgets do not imply equal token budgets; report calls, tokens,
first-cycle success, final success and active time including failed searches.
There is no hard token-cost cap. Infrastructure failures are not model failures.

Primary comparisons are guided direct versus minimal direct (guidance), structured
IR versus guided direct (structured workflow), hinted IR versus guided direct
(free-form planning), and structured IR versus hinted IR. The campaign summary
includes paired comparisons and exploratory kernel-cluster bootstrap intervals.
The comparisons between guided IR and minimal direct combine both interventions.
Three repetitions of ten kernels do not constitute thirty independent kernels.

Keep the existing offline acceptance checks, compiler images, deterministic inputs,
feedback, concurrency and fresh per-case state. These results cannot establish
physical NPU correctness or performance. Prompt guidance was developed on this
corpus; this rerun is development evidence, not a held-out generalization result.
Changing the model/provider means historical Terra-to-Qwen differences cannot isolate a
prompt effect. Make claims from comparisons within this new campaign.

Qwen3-Coder-30B-A3B-Instruct is an Apache-2.0 coding model with 30.5B total and
3.3B active parameters. It is selected for coding specialization, affordable
repetitions and a plausible opportunity for explicit decomposition to help.
No NPU evidence establishes that structured IR will improve this model. Freeze
this choice before observing results and report every arm, including null or
negative outcomes. Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct),
[OpenRouter model](https://openrouter.ai/qwen/qwen3-coder-30b-a3b-instruct).

Set `openrouter.api_key` in the repository root `config.yaml`. This local file is
ignored by Git and excluded from snapshots. A new checkout can create it as:

```yaml
openrouter:
  api_key: ""
  model: "qwen/qwen3-coder-30b-a3b-instruct"
  endpoint: "siliconflow/fp8"
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  max_tokens: 32768
```

Only non-secret settings enter campaign provenance. Each request contains one
fresh user prompt and its output schema; no tools or conversation history are
provided. The model has no separate thinking mode, so `xhigh` is not sent and
reasoning effort is recorded as `none`. Sampling parameters follow the Qwen
recommendations where supported; repetition penalty is omitted because this
endpoint does not advertise it. Every arm has the same per-call output limit;
the additional IR calls still count toward total token usage.

Structured-output support is required, and response model/provider identities,
usage, schema and tool-call absence are checked. API failures stop cases as
infrastructure failures; malformed generated JSON consumes a cycle and retains
its billed usage. No invisible API retries or model fallback are performed.
Endpoint settings are frozen across repetitions and resume. The hosted service
does not expose an immutable weights snapshot, so endpoint pinning is not proof
of an unchanged deployment. Provider generation IDs and returned model/provider
names are retained for auditing.

```bash
bash scripts/run_case_study_v3.sh plan
bash scripts/run_case_study_v3.sh run
bash scripts/run_case_study_v3.sh report
```

`run` checks model access with one small structured-response call before starting
benchmark cases. The response and usage are saved in `model-preflight.json` and
excluded from benchmark totals. Access failure stops the campaign before any
kernel trials; it never substitutes a different model. Once access is restored,
repeat the same command. The default output is `runs/case-study-v3-openrouter`.
Source snapshots, environment checks, locking, resume and reporting follow v2.

For a separate held-out stage, provide a reviewed, previously unused corpus and a
completed v3 development campaign:

```bash
bash scripts/run_case_study_v3.sh run --stage heldout \
  --corpus /absolute/path/to/heldout-kernels \
  --development-campaign runs/case-study-v3-openrouter \
  --output runs/case-study-v3-openrouter-heldout
```

All four arms, the frozen model/protocol and code are retained. The held-out corpus
is not authored here. Do not adjust prompts after observing its translation results.
