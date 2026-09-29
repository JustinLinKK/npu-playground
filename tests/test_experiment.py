from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import pytest

from npu_agent.experiment import run_experiment
from npu_agent.models import (
    ArgumentDirection,
    Backend,
    CodeBundle,
    CompileResult,
    DebugResponse,
    GeneratedFile,
    IROperation,
    IterationDomain,
    KernelIR,
    KernelManifest,
    OperationAttributes,
    OptimizationProposal,
    ProposalSet,
)
from npu_agent.providers import ProviderResponse
from npu_agent.scenarios import IntermediateRepresentation
from npu_agent.models import StageResult, ValidationResult


class CorpusProvider:
    name = "fake"

    def __init__(self, model="gpt-5.6-terra") -> None:
        self.calls: list[str] = []
        self.model = model

    @staticmethod
    def _manifest(prompt: str) -> KernelManifest:
        marker = "Manifest:\n"
        start = prompt.index(marker) + len(marker)
        value, _ = json.JSONDecoder().raw_decode(prompt[start:])
        return KernelManifest.model_validate(value)

    @staticmethod
    def _bundle(prompt: str, suffix: str = "") -> CodeBundle:
        backend = Backend.AMD_XDNA2 if '"backend": "amd_xdna2"' in prompt else Backend.INTEL_OPENVINO
        files = (
            [
                GeneratedFile(relative_path="design.py", content=f"# design {suffix}"),
                GeneratedFile(relative_path="kernel.cc", content=f"// kernel {suffix}"),
            ]
            if backend == Backend.AMD_XDNA2
            else [GeneratedFile(relative_path="model.py", content=f"# model {suffix}")]
        )
        return CodeBundle(backend=backend, files=files)

    def generate(self, prompt: str, response_model: type[BaseModel], model: str | None = None) -> ProviderResponse:
        assert model == self.model
        self.calls.append(response_model.__name__)
        if issubclass(response_model, KernelIR):
            manifest = self._manifest(prompt)
            inputs = [item for item in manifest.tensors if item.direction != ArgumentDirection.OUTPUT]
            outputs = [
                item
                for item in manifest.tensors
                if item.direction in (ArgumentDirection.OUTPUT, ArgumentDirection.INOUT)
            ]
            operation = "add" if manifest.oracle.operation == "vector_add" else manifest.oracle.operation
            value = response_model(
                name=manifest.name,
                inputs=inputs,
                scalars=manifest.scalars,
                outputs=outputs,
                iteration_domains=[IterationDomain(variables=["i"], bounds=["logical output domain"])],
                operations=[
                    IROperation(
                        id="result",
                        op=operation,
                        inputs=[item.name for item in inputs],
                        output=outputs[0].name,
                        attributes=OperationAttributes.model_validate(manifest.oracle.parameters),
                    )
                ],
                numeric_behavior=["match the manifest oracle"],
            )
        elif response_model is IntermediateRepresentation:
            value = IntermediateRepresentation(intermediate="Compute the manifest operation with its declared inputs and outputs.")
        elif response_model is CodeBundle:
            value = self._bundle(prompt)
        elif response_model is ProposalSet:
            value = ProposalSet(
                proposals=[
                    OptimizationProposal(
                        label=f"proposal-{index}",
                        rationale=f"fake proposal {index}",
                        expected_benefit="test coverage",
                        bundle=self._bundle(prompt, str(index)),
                    )
                    for index in range(3)
                ]
            )
        elif response_model is DebugResponse:
            raise AssertionError("successful fake compiles must not enter debugging")
        else:
            raise AssertionError(response_model)
        raw = value.model_dump(mode="json")
        metadata: dict[str, Any] = {
            "provider": "fake",
            "model": "fake",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(response_model.__name__.encode()).hexdigest(),
            "exit_code": 0,
            "schema_valid": True,
            "usage": {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 5},
            "telemetry": [{"type": "turn.completed"}],
            "duration_seconds": 0.002,
        }
        return ProviderResponse(value=value, metadata=metadata, raw=raw)


class CorpusCompiler:
    def __init__(self) -> None:
        self.calls = 0

    def compile(self, candidate_dir: Path, output_dir: Path, manifest, target) -> CompileResult:
        del candidate_dir, output_dir, manifest
        self.calls += 1
        return CompileResult(
            success=True,
            exit_code=0,
            compiler_fingerprint=f"fake:{target.id}",
            duration_seconds=0.003,
            evaluation_duration_seconds=0.001 if target.backend == Backend.INTEL_OPENVINO else 0,
            host_correct=True if target.backend == Backend.INTEL_OPENVINO else None,
            cpu_latency_p50_ms=0.4 if target.backend == Backend.INTEL_OPENVINO else None,
            cpu_latency_p95_ms=0.6 if target.backend == Backend.INTEL_OPENVINO else None,
            static_metrics={"source_bytes": 20, "artifact_bytes": 30, "vectorization_signals": 1},
            validation=ValidationResult(candidate_id="fake", target_id=target.id, stages={
                "oracle_validation": StageResult(status="passed", correct=True, cases_run=12),
                "target_compile": StageResult(status="passed"),
                "host_execution": StageResult(status="passed", correct=True, cases_run=12),
                "dataflow_simulation": StageResult(status="passed", correct=True, cases_run=12) if target.vendor == "amd" else StageResult(status="not_requested"),
                "target_execution": StageResult(status="blocked"),
            }),
        )


@pytest.mark.parametrize("workers", [1, 20])
def test_full_fake_experiment_reports_all_cases_and_resumes(tmp_path: Path, workers: int) -> None:
    lock = threading.Lock()
    first_batch = threading.Barrier(workers)

    class ParallelProvider(CorpusProvider):
        active = 0
        peak = 0
        started = 0
        initial_stages = []

        def generate(self, prompt, response_model, model=None):
            with lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.started += 1
                initial = self.started <= workers
                if initial:
                    self.initial_stages.append(response_model.__name__)
            if initial:
                first_batch.wait(timeout=30)
            try:
                return super().generate(prompt, response_model, model)
            finally:
                with lock:
                    self.active -= 1

    class ParallelCompiler(CorpusCompiler):
        active = 0
        peak = 0

        def compile(self, *args):
            with lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(0.01)
                with lock:
                    return super().compile(*args)
            finally:
                with lock:
                    self.active -= 1

    provider = ParallelProvider()
    compiler = ParallelCompiler()
    root = run_experiment(
        repository=Path.cwd(),
        corpus=Path("examples/classic"),
        runs_dir=tmp_path / "runs",
        provider_name="codex-cli",
        provider=provider,
        compiler=compiler,
        case_workers=workers,
        max_compiler_jobs=4,
        compiler_cpus=6,
        compiler_memory="8g",
    )
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert experiment["status"] == "completed"
    assert experiment["schema_version"] == "2.0"
    assert experiment["model"] == "gpt-5.6-terra"
    assert experiment["reasoning_effort"] == "xhigh"
    assert experiment["knowledge_enabled"] is False
    assert len(experiment["cases"]) == 60
    assert all(case["status"] == "completed" for case in experiment["cases"].values())
    assert len(metrics) == 60
    assert all(item["success"] for item in metrics)
    assert all(item["target_npu_p50_ms"] is None for item in metrics)
    assert all(
        item["amd_compile_valid"] is True and item["intel_cpu_equivalent"] is None
        for item in metrics
        if item["backend"] == "amd_xdna2"
    )
    assert all(
        item["amd_compile_valid"] is None and item["intel_cpu_equivalent"] is True
        for item in metrics
        if item["backend"] == "intel_openvino"
    )
    assert all(item["target_npu_correct"] is None for item in metrics)
    assert all(item["provider_seconds"] > 0 for item in metrics)
    assert all(item["compiler_seconds"] > 0 for item in metrics)
    assert all(item["evaluation_seconds"] > 0 for item in metrics)
    assert provider.peak == workers
    assert 1 <= compiler.peak <= min(workers, 4)
    if workers == 1:
        assert provider.calls[:20] == ["CodeBundle"] * 20
        assert provider.calls[20] == "ExecutableKernelIR"
    else:
        assert compiler.peak > 1
        assert set(provider.initial_stages) == {"CodeBundle", "ExecutableKernelIR", "IntermediateRepresentation"}
    assert experiment["execution"] == {"case_workers": workers, "max_compiler_jobs": 4,
                                        "compiler_cpus": 6, "compiler_memory": "8g"}
    assert compiler.calls == 60
    assert (root / "metrics.csv").is_file()
    assert (root / "summary.json").is_file()
    assert (root / "report.html").is_file()
    assert len((root / "logs" / "provider_calls.jsonl").read_text(encoding="utf-8").splitlines()) == 100
    assert len((root / "logs" / "compile_attempts.jsonl").read_text(encoding="utf-8").splitlines()) == 60
    assert not list((root / "charts").glob("search_tree_*.png"))
    assert len(list((root / "charts").glob("*.png"))) == 7
    assert all(item["cycles_to_success"] == 1 and item["backend_test_attempts"] == 1 for item in metrics)
    assert all(item["total_tokens"] == (15 if item["method"] == "baseline" else 30) for item in metrics)
    assert list((root / "charts").glob("*.svg"))
    svg_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "charts").glob("*.svg")
    }
    calls_before = len(provider.calls)
    compiles_before = compiler.calls
    resumed = run_experiment(
        repository=Path.cwd(),
        corpus=Path("examples/classic"),
        runs_dir=tmp_path / "runs",
        resume=root.name,
        provider=provider,
        compiler=compiler,
    )
    assert resumed == root
    assert len(provider.calls) == calls_before
    assert compiler.calls == compiles_before
    assert json.loads((root / "experiment.json").read_text())["execution"] == experiment["execution"]
    assert svg_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "charts").glob("*.svg")
    }


@pytest.mark.parametrize("options", [{"provider_name": "claude-cli"}, {"model": "other"},
    {"validation_policy": "compile-only"}, {"max_cycles": 0}, {"max_cycles": -1}, {"max_cycles": True},
    {"case_workers": 0}, {"case_workers": True}, {"case_workers": 1.5}, {"max_compiler_jobs": -1},
    {"compiler_cpus": 0}, {"compiler_memory": ""}])
def test_conflicting_experiment_settings_are_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        run_experiment(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path, **options)
    assert not list(tmp_path.iterdir())


def test_historical_experiment_cannot_be_resumed_under_new_protocol(tmp_path):
    (tmp_path / "experiment.json").write_text(json.dumps({"schema_version": "1.0"}))
    with pytest.raises(ValueError, match="historical"):
        run_experiment(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path, resume=str(tmp_path))


def test_case_study_v2_all_arms_share_guidance_and_resume(tmp_path):
    from npu_agent.scenarios import ABLATIONS, CASE_STUDY_PROTOCOL, SCENARIOS

    class PromptProvider(CorpusProvider):
        prompts = []

        def generate(self, prompt, response_model, model=None):
            if response_model is CodeBundle:
                self.prompts.append(prompt)
            return super().generate(prompt, response_model, model)

    provider, compiler = PromptProvider(), CorpusCompiler()
    options = dict(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path,
                   kernel_names=["triton_layer_norm"], target_ids=["intel_npu_4000"],
                   provider=provider, compiler=compiler, protocol_revision=CASE_STUDY_PROTOCOL, order_seed=42)
    root = run_experiment(**options, scenarios=list(SCENARIOS + ABLATIONS))
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 5
    assert all(r["success"] for r in records)
    assert len(provider.prompts) == 5
    assert all("Shared numerical guidance for every case-study arm" in prompt for prompt in provider.prompts)
    assert len(json.loads((root / "summary.json").read_text())["paired_efficiency"]) == 4
    chart = (root / "charts/success_by_cycle_intel_npu_4000.svg").read_text()
    assert all(method in chart for method in SCENARIOS + ABLATIONS)
    before = len(provider.calls)
    run_experiment(**options, resume=str(root))
    assert len(provider.calls) == before
    with pytest.raises(ValueError, match="order_seed"):
        run_experiment(**dict(options, order_seed=43), resume=str(root))
    with pytest.raises(ValueError, match="protocol revision"):
        run_experiment(**dict(options, protocol_revision="validated-ir-v2"), resume=str(root))


def test_v3_openrouter_four_arms_isolate_guidance_and_pin_resume(tmp_path):
    from npu_agent.scenarios import GUIDANCE_MODEL, GUIDANCE_PROTOCOL, GUIDANCE_SCENARIOS

    provider, compiler = CorpusProvider(GUIDANCE_MODEL), CorpusCompiler()
    options = dict(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path,
                   kernel_names=["triton_layer_norm"], provider=provider, compiler=compiler,
                   protocol_revision=GUIDANCE_PROTOCOL)
    root = run_experiment(**options)
    metadata = json.loads((root / "experiment.json").read_text())
    assert metadata["model"] == GUIDANCE_MODEL
    assert metadata["provider"] == "openrouter"
    assert metadata["reasoning_effort"] == "none"
    assert metadata["scenarios"] == list(GUIDANCE_SCENARIOS)
    assert len(metadata["cases"]) == 8
    for case in metadata["cases"].values():
        report = json.loads((root / case["result_path"]).read_text())
        assert report["terminal_reason"] == "solved"
        prompt = report["cycles"][0]["calls"]["code"]["prompt"]
        guided = case["method"] != "baseline_minimal"
        assert ("Shared numerical guidance" in prompt) is guided
        assert ("c=d-mean(d" in prompt) is guided
        if case["target_id"] == "amd_xdna2_npu2":
            assert ("aie.device(npu2_1col)" in prompt) is guided
            assert ("Runtime(seq_fn, fn_args)" in prompt) is guided
            assert ("npu::float16" in prompt) is guided
            assert "--emit-mlir" in prompt and "Runtime buffer order" in prompt
        else:
            assert ("node.output(0).get_tensor()" in prompt) is guided
            assert "build_model(manifest=None)" in prompt
        assert "compiler_version" in prompt and "Manifest:" in prompt
    chart = (root / "charts/success_by_cycle_intel_npu_4000.svg").read_text()
    assert all(method in chart for method in GUIDANCE_SCENARIOS)
    historical = run_experiment(**dict(options, protocol_revision="case-study-v2",
                                      provider=CorpusProvider()), scenarios=["baseline", "structured_ir", "hinted_ir"])
    old_cases = json.loads((historical / "experiment.json").read_text())["cases"]
    for key, case in old_cases.items():
        old = json.loads((historical / case["result_path"]).read_text())
        new = json.loads((root / metadata["cases"][key]["result_path"]).read_text())
        assert {s: c["prompt"] for s, c in new["cycles"][0]["calls"].items()} == {
            s: c["prompt"] for s, c in old["cycles"][0]["calls"].items()}
    before = len(provider.calls)
    run_experiment(**dict(options, protocol_revision=None), resume=str(root))
    assert len(provider.calls) == before
    with pytest.raises(ValueError, match="model cannot change"):
        run_experiment(**options, resume=str(root), model="gpt-5.6-terra")
    with pytest.raises(ValueError, match="requires qwen/qwen3-coder"):
        run_experiment(**options, model="gpt-5.6-terra")
    with pytest.raises(ValueError, match="baseline_minimal requires"):
        run_experiment(**dict(options, protocol_revision="case-study-v2"), scenarios=["baseline_minimal"])
    with pytest.raises(ValueError, match="provider cannot change"):
        run_experiment(**options, resume=str(root), provider_name="codex-cli")


def test_structured_only_experiment_reports_twenty_cases_and_preserves_revision(tmp_path):
    provider, compiler = CorpusProvider(), CorpusCompiler()
    options = dict(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path,
                   provider=provider, compiler=compiler)
    root = run_experiment(**options, scenarios=["structured_ir"], case_workers=20)
    experiment = json.loads((root / "experiment.json").read_text())
    assert experiment["scenarios"] == ["structured_ir"]
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 20
    assert all(record["success"] for record in records)
    run_experiment(**options, resume=str(root))
    assert len(provider.calls) == 40
    assert compiler.calls == 20
    with pytest.raises(ValueError, match="scenarios cannot change"):
        run_experiment(**options, resume=str(root), scenarios=["baseline"])
    del experiment["protocol_revision"]
    (root / "experiment.json").write_text(json.dumps(experiment))
    with pytest.raises(ValueError, match="protocol revision changed"):
        run_experiment(**options, resume=str(root))


def test_unpinned_live_provider_cannot_bypass_experiment_settings(tmp_path):
    from npu_agent.providers import CodexCLIProvider

    with pytest.raises(ValueError, match="pinned"):
        run_experiment(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path,
                       provider=object.__new__(CodexCLIProvider))
    assert not list(tmp_path.iterdir())


def test_interrupted_experiment_resumes_same_case_and_budget(tmp_path):
    class InterruptedCompiler(CorpusCompiler):
        def compile(self, *args):
            if self.calls == 0:
                self.calls += 1
                raise KeyboardInterrupt
            return super().compile(*args)

    provider, compiler = CorpusProvider(), InterruptedCompiler()
    options = dict(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path,
                   kernel_names=["cuda_vector_add"], target_ids=["intel_npu_4000"], provider=provider, compiler=compiler)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(**options, max_cycles=3)
    root = next(tmp_path.iterdir())
    before = json.loads((root / "experiment.json").read_text())
    assert before["status"] == "interrupted"
    with pytest.raises(ValueError, match="max_cycles"):
        run_experiment(**options, resume=str(root), max_cycles=4)
    run_experiment(**options, resume=str(root))
    after = json.loads((root / "experiment.json").read_text())
    assert after["max_cycles"] == 3
    assert after["status"] == "completed"
    assert len(provider.calls) == 5
    assert compiler.calls == 4
    key = "baseline:cuda_vector_add:intel_npu_4000"
    assert before["cases"][key]["run_id"] == after["cases"][key]["run_id"]


def test_parallel_stop_checkpoints_responses_and_resumes_without_repeating_calls(tmp_path, monkeypatch):
    import npu_agent.experiment as runner

    lock = threading.Lock()
    entered = threading.Event()
    thread_state = threading.local()
    real_translate, real_wait = runner.translate_scenario, runner.wait
    instances = []
    compiler_calls = []
    stopping = True

    def translate(*args, **kwargs):
        thread_state.stop_event = kwargs["stop_event"]
        return real_translate(*args, **kwargs)

    class InFlightProvider(CorpusProvider):
        started = 0

        def generate(self, *args):
            if stopping:
                with lock:
                    self.started += 1
                    if self.started == 3:
                        entered.set()
                assert thread_state.stop_event.wait(10)
            return super().generate(*args)

    class WorkerCompiler(CorpusCompiler):
        def __init__(self, settings):
            super().__init__()
            self.owner = threading.get_ident()
            assert settings.compiler_cpus == 6 and settings.compiler_memory == "8g"
            assert settings.max_compiler_jobs == 4
            instances.append(self)

        def compile(self, *args):
            assert threading.get_ident() == self.owner
            with lock:
                compiler_calls.append(self.owner)
            return super().compile(*args)

    def interrupt_after_calls_start(*args, **kwargs):
        if stopping:
            assert entered.wait(10)
            raise KeyboardInterrupt
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(runner, "translate_scenario", translate)
    monkeypatch.setattr(runner, "DockerCompiler", WorkerCompiler)
    monkeypatch.setattr(runner, "wait", interrupt_after_calls_start)
    provider = InFlightProvider()
    options = dict(repository=Path.cwd(), corpus=Path("examples/classic"), runs_dir=tmp_path,
                   kernel_names=["cuda_vector_add"], target_ids=["intel_npu_4000"], provider=provider)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(**options, case_workers=3, max_compiler_jobs=4, compiler_cpus=6)
    root = next(tmp_path.iterdir())
    before = json.loads((root / "experiment.json").read_text())
    assert before["status"] == "interrupted"
    assert len(provider.calls) == 3 and not compiler_calls
    assert len(instances) == 3 and len({item.owner for item in instances}) == 3
    for case in before["cases"].values():
        checkpoint = json.loads((root / case["method"] / case["run_id"] / "report.json").read_text())
        assert checkpoint["backend_test_attempts"] == 0
        assert all(call["status"] == "completed" for call in checkpoint["cycles"][0]["calls"].values())
    stopping = False
    run_experiment(**options, resume=str(root))
    after = json.loads((root / "experiment.json").read_text())
    assert after["status"] == "completed"
    assert after["execution"] == before["execution"]
    assert len(provider.calls) == 5 and len(compiler_calls) == 3
    assert [case["run_id"] for case in before["cases"].values()] == [case["run_id"] for case in after["cases"].values()]
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert all(row["total_tokens"] == (15 if row["method"] == "baseline" else 30) for row in metrics)
    assert all(row["backend_test_attempts"] == 1 for row in metrics)
    run_experiment(**options, resume=str(root), case_workers=1)
    assert json.loads((root / "experiment.json").read_text())["execution"]["case_workers"] == 1
    assert len(provider.calls) == 5 and len(compiler_calls) == 3
