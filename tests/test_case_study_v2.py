import importlib.util
import json
from pathlib import Path

import pytest

from npu_agent.scenarios import ABLATIONS, SCENARIOS
from test_experiment import CorpusCompiler, CorpusProvider


spec = importlib.util.spec_from_file_location("case_study_v2", Path("scripts/case_study_v2.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.fixture
def setup_campaign(tmp_path, monkeypatch):
    provider, compiler = CorpusProvider(), CorpusCompiler()
    config = {"stage": "controlled", "corpus": str(Path("examples/classic").resolve()),
              "kernels": ["cuda_vector_add"], "targets": ["intel_npu_4000"],
              "development_campaign": None, "repetitions": 3, "order_seed": 50,
              "scenarios": list(SCENARIOS + ABLATIONS),
              "execution": {"case_workers": 1, "max_compiler_jobs": 1, "compiler_cpus": 1, "compiler_memory": "1g"}}
    original = campaign.run_experiment
    monkeypatch.setattr(campaign, "run_experiment", lambda **kwargs: original(**kwargs, compiler=compiler))
    monkeypatch.setattr(campaign, "environment", lambda targets, protocol_revision: (
        {"toolchains": {t: {"fingerprint": f"fake:{t}"} for t in targets}}, provider))
    monkeypatch.setattr("npu_agent.reporting.generate_charts", lambda *args: [])
    return tmp_path, config, provider, compiler


def test_campaign_repeats_reports_and_refuses_drift(setup_campaign):
    output, config, provider, compiler = setup_campaign
    campaign.run_campaign(output, config)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert len(summary["groups"]) == 5
    assert all(g["solved"] == g["requested"] == 3 for g in summary["groups"])
    assert all(g["successes_by_repetition"] == [1, 1, 1] for g in summary["groups"])
    assert all(g["success_within_token_budget"]["10000"] == 1 for g in summary["groups"])
    assert all(p["jointly_solved"] == 3 for p in summary["paired"])
    assert len(provider.calls) == 27
    assert compiler.calls == 15
    assert (output / "snapshot/code/src/npu_agent/scenarios.py").exists()
    metadata = [json.loads(p.read_text()) for p in sorted(output.glob("repeat-*/*/experiment.json"))]
    assert [m["order_seed"] for m in metadata] == [50, 51, 52]
    assert len({c["run_id"] for m in metadata for c in m["cases"].values()}) == 15
    campaign.run_campaign(output, config)
    assert len(provider.calls) == 27
    with pytest.raises(ValueError, match="configuration or source/corpus changed"):
        campaign.run_campaign(output, dict(config, repetitions=4))


def test_campaign_resumes_interruption_without_repeating_calls(setup_campaign, monkeypatch):
    output, config, provider, compiler = setup_campaign
    original = compiler.compile
    interrupted = False

    def compile(*args):
        nonlocal interrupted
        if compiler.calls == 5 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original(*args)

    monkeypatch.setattr(compiler, "compile", compile)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_campaign(output, config)
    assert json.loads((output / "campaign.json").read_text())["status"] == "interrupted"
    first_path = next((output / "repeat-001").glob("*/experiment.json"))
    first = first_path.read_bytes()
    campaign.run_campaign(output, config)
    assert first_path.read_bytes() == first
    assert len(provider.calls) == 27
    assert compiler.calls == 15
    assert len(list(output.glob("repeat-*/*/experiment.json"))) == 3


def test_campaign_stops_on_blocked_evidence(setup_campaign):
    from npu_agent.providers import ProviderError

    output, config, provider, compiler = setup_campaign
    def unavailable(*args):
        raise ProviderError("test unavailable")
    provider.generate = unavailable
    with pytest.raises(ValueError, match="blocked, unexpected, or incomplete-usage"):
        campaign.run_campaign(output, config)
    assert not (output / "repeat-002").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert all(g["blocked"] == 1 and g["requested"] == 3 for g in summary["groups"])
    assert all(p["complete_kernel_clusters"] == 0 for p in summary["paired"])


def test_heldout_requires_frozen_disjoint_corpus(setup_campaign):
    output, config, _, _ = setup_campaign
    campaign.run_campaign(output, config)
    heldout = dict(config, stage="heldout", development_campaign=str(output))
    identity = campaign.inputs(Path(config["corpus"]), config["kernels"])
    with pytest.raises(ValueError, match="names overlap"):
        campaign.check_heldout(heldout, identity)
    identity["corpus"] = {"renamed": next(iter(identity["corpus"].values()))}
    with pytest.raises(ValueError, match="source duplicates"):
        campaign.check_heldout(heldout, identity)
    identity["corpus"]["renamed"]["source"] = "new-source"
    campaign.check_heldout(heldout, identity)
    identity["code"]["src/npu_agent/scenarios.py"] = "changed"
    with pytest.raises(ValueError, match="frozen"):
        campaign.check_heldout(heldout, identity)


def test_v3_campaign_preflight_four_arms_and_paired_guidance(setup_campaign):
    from npu_agent.scenarios import GUIDANCE_MODEL, GUIDANCE_PROTOCOL, GUIDANCE_SCENARIOS

    output, config, provider, compiler = setup_campaign
    provider.model = GUIDANCE_MODEL
    config.update(model=GUIDANCE_MODEL, provider="openrouter", protocol_revision=GUIDANCE_PROTOCOL, scenarios=list(GUIDANCE_SCENARIOS))
    campaign.run_campaign(output, config)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert len(summary["groups"]) == 4
    assert all(g["solved"] == g["requested"] == 3 for g in summary["groups"])
    pairs = {(p["method"], p["reference"]) for p in summary["paired"]}
    assert {("baseline", "baseline_minimal"), ("structured_ir", "baseline"),
            ("hinted_ir", "baseline"), ("structured_ir", "hinted_ir")} <= pairs
    assert len(provider.calls) == 19  # One access probe, then six calls per repetition.
    assert compiler.calls == 12
    assert json.loads((output / "model-preflight.json").read_text())["status"] == "passed"
    assert "Case study v3" in (output / "results.md").read_text()
    assert not (output / "snapshot/code/config.yaml").exists()
    campaign.run_campaign(output, config)
    assert len(provider.calls) == 19
    heldout = dict(config, stage="heldout", development_campaign=str(output), model="gpt-5.6-terra")
    with pytest.raises(ValueError, match="preserve model"):
        campaign.check_heldout(heldout, campaign.inputs(Path(config["corpus"]), config["kernels"]))


def test_v3_model_access_failure_stops_before_benchmark_cases(setup_campaign):
    from npu_agent.providers import ProviderError
    from npu_agent.scenarios import GUIDANCE_MODEL, GUIDANCE_PROTOCOL, GUIDANCE_SCENARIOS

    output, config, provider, compiler = setup_campaign
    provider.model = GUIDANCE_MODEL
    config.update(model=GUIDANCE_MODEL, provider="openrouter", protocol_revision=GUIDANCE_PROTOCOL, scenarios=list(GUIDANCE_SCENARIOS))
    original = provider.generate
    def unavailable(*args):
        raise ProviderError("model not supported", {"model": GUIDANCE_MODEL}, {})
    provider.generate = unavailable
    with pytest.raises(ValueError, match="model access preflight failed"):
        campaign.run_campaign(output, config)
    assert not list(output.glob("repeat-*"))
    assert compiler.calls == 0
    assert json.loads((output / "model-preflight.json").read_text())["status"] == "blocked"
    assert "no benchmark cases were started" in (output / "results.md").read_text()
    provider.generate = original
    campaign.run_campaign(output, config)
    assert compiler.calls == 12
    assert "blocked_reason" not in json.loads((output / "campaign.json").read_text())


def test_v3_campaign_refuses_changed_openrouter_sampling_on_resume(setup_campaign, monkeypatch):
    from npu_agent.scenarios import GUIDANCE_MODEL, GUIDANCE_PROTOCOL, GUIDANCE_SCENARIOS

    output, config, provider, compiler = setup_campaign
    provider.model = GUIDANCE_MODEL
    config.update(model=GUIDANCE_MODEL, provider="openrouter", protocol_revision=GUIDANCE_PROTOCOL, scenarios=list(GUIDANCE_SCENARIOS))
    original_environment = campaign.environment
    temperature = 0.7
    def environment(*args):
        identity, selected = original_environment(*args)
        identity["provider_settings"] = {"temperature": temperature}
        return identity, selected
    monkeypatch.setattr(campaign, "environment", environment)
    original_compile = compiler.compile
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(compiler, "compile", interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_campaign(output, config)
    calls_before = len(provider.calls)
    temperature = 0.2
    monkeypatch.setattr(compiler, "compile", original_compile)
    with pytest.raises(ValueError, match="identity changed"):
        campaign.run_campaign(output, config)
    assert len(provider.calls) == calls_before
