from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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


class CorpusProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

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
        del model
        self.calls.append(response_model.__name__)
        if response_model is KernelIR:
            manifest = self._manifest(prompt)
            inputs = [item for item in manifest.tensors if item.direction != ArgumentDirection.OUTPUT]
            outputs = [
                item
                for item in manifest.tensors
                if item.direction in (ArgumentDirection.OUTPUT, ArgumentDirection.INOUT)
            ]
            operation = "add" if manifest.oracle.operation == "vector_add" else manifest.oracle.operation
            value = KernelIR(
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
        )


def test_full_fake_experiment_reports_all_cases_and_resumes(tmp_path: Path) -> None:
    provider = CorpusProvider()
    compiler = CorpusCompiler()
    root = run_experiment(
        repository=Path.cwd(),
        corpus=Path("examples/classic"),
        runs_dir=tmp_path / "runs",
        provider_name="codex-cli",
        provider=provider,
        compiler=compiler,
    )
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert experiment["status"] == "completed"
    assert len(experiment["cases"]) == 40
    assert all(case["status"] == "completed" for case in experiment["cases"].values())
    assert len(metrics) == 40
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
    assert provider.calls[:20] == ["CodeBundle"] * 20
    assert provider.calls[20] == "KernelIR"
    assert compiler.calls == 220
    assert (root / "metrics.csv").is_file()
    assert (root / "summary.json").is_file()
    assert (root / "report.html").is_file()
    assert len((root / "logs" / "provider_calls.jsonl").read_text(encoding="utf-8").splitlines()) == 120
    assert len((root / "logs" / "compile_attempts.jsonl").read_text(encoding="utf-8").splitlines()) == 220
    assert len(list((root / "charts").glob("search_tree_*.png"))) == 20
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
    assert svg_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "charts").glob("*.svg")
    }
