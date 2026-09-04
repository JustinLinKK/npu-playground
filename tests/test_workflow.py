from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from npu_agent.config import AMD_XDNA2_NPU2, INTEL_NPU_4000, Settings
from npu_agent.baseline import translate_one_shot
from npu_agent.ir import generate_inputs
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
    OptimizationProposal,
    ProposalSet,
    TranslationRequest,
)
from npu_agent.providers import ProviderResponse
from npu_agent.providers import ProviderError
from npu_agent.workflow import build_workflow, resume_run, translate


class FakeProvider:
    name = "fake"

    def __init__(self, manifest: KernelManifest) -> None:
        self.manifest = manifest
        self.calls: list[str] = []

    def metadata(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": "fake",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(str(schema).encode()).hexdigest(),
            "exit_code": 0,
            "usage": {},
        }

    def generate(self, prompt: str, response_model: type[BaseModel], model: str | None = None) -> ProviderResponse:
        del model
        self.calls.append(response_model.__name__)
        inputs = [item for item in self.manifest.tensors if item.direction != ArgumentDirection.OUTPUT]
        outputs = [item for item in self.manifest.tensors if item.direction == ArgumentDirection.OUTPUT]
        if response_model is KernelIR:
            value = KernelIR(
                name=self.manifest.name,
                inputs=inputs,
                scalars=self.manifest.scalars,
                outputs=outputs,
                iteration_domains=[IterationDomain(variables=["i"], bounds=["0 <= i < element_count"])],
                operations=[
                    IROperation(id="add", op="add", inputs=[item.name for item in inputs], output=outputs[0].name)
                ],
                numeric_behavior=["float32 elementwise addition"],
            )
        elif response_model is CodeBundle:
            backend = Backend.AMD_XDNA2 if '"backend": "amd_xdna2"' in prompt else Backend.INTEL_OPENVINO
            files = (
                [GeneratedFile(relative_path="design.py", content="# design"), GeneratedFile(relative_path="kernel.cc", content="// kernel")]
                if backend == Backend.AMD_XDNA2
                else [GeneratedFile(relative_path="model.py", content="# model")]
            )
            value = CodeBundle(backend=backend, files=files)
        elif response_model is ProposalSet:
            backend = Backend.AMD_XDNA2 if '"backend": "amd_xdna2"' in prompt else Backend.INTEL_OPENVINO
            files_for = lambda index: (
                [GeneratedFile(relative_path="design.py", content=f"# design {index}"), GeneratedFile(relative_path="kernel.cc", content=f"// kernel {index}")]
                if backend == Backend.AMD_XDNA2
                else [GeneratedFile(relative_path="model.py", content=f"# model {index}")]
            )
            value = ProposalSet(
                proposals=[
                    OptimizationProposal(
                        label=f"proposal-{index}",
                        rationale=f"rationale {index}",
                        expected_benefit="test",
                        bundle=CodeBundle(backend=backend, files=files_for(index)),
                    )
                    for index in range(3)
                ]
            )
        elif response_model is DebugResponse:
            raise AssertionError("debug should not be called for successful fake compilation")
        else:
            raise AssertionError(response_model)
        raw = value.model_dump(mode="json")
        return ProviderResponse(value, self.metadata(prompt, response_model.model_json_schema()), raw)


class FakeCompiler:
    def compile(self, candidate_dir: Path, output_dir: Path, manifest, target) -> CompileResult:
        del candidate_dir, output_dir, manifest
        return CompileResult(
            success=True,
            exit_code=0,
            compiler_fingerprint=f"fake:{target.id}",
            host_correct=True if target.backend == Backend.INTEL_OPENVINO else None,
            static_metrics={"source_bytes": 20, "artifact_bytes": 30, "vectorization_signals": 1},
        )


class CountingCompiler(FakeCompiler):
    def __init__(self) -> None:
        self.calls = 0

    def compile(self, candidate_dir: Path, output_dir: Path, manifest, target) -> CompileResult:
        self.calls += 1
        return super().compile(candidate_dir, output_dir, manifest, target)


class RepairProvider(FakeProvider):
    def generate(self, prompt: str, response_model: type[BaseModel], model: str | None = None) -> ProviderResponse:
        if response_model is DebugResponse:
            backend = Backend.AMD_XDNA2 if '"backend": "amd_xdna2"' in prompt else Backend.INTEL_OPENVINO
            files = (
                [GeneratedFile(relative_path="design.py", content="# fixed"), GeneratedFile(relative_path="kernel.cc", content="// fixed")]
                if backend == Backend.AMD_XDNA2
                else [GeneratedFile(relative_path="model.py", content="# fixed")]
            )
            value = DebugResponse(
                diagnosis="the generated model contained invalid syntax",
                fix_summary="replace the broken model with a valid graph",
                bundle=CodeBundle(backend=backend, files=files),
            )
            raw = value.model_dump(mode="json")
            return ProviderResponse(value, self.metadata(prompt, response_model.model_json_schema()), raw)
        response = super().generate(prompt, response_model, model)
        if response_model is CodeBundle:
            response.value.files[0].content = "# broken"
            return ProviderResponse(
                response.value,
                response.metadata,
                response.value.model_dump(mode="json"),
            )
        return response


class RepairCompiler:
    def compile(self, candidate_dir: Path, output_dir: Path, manifest, target) -> CompileResult:
        del output_dir, manifest
        content = (candidate_dir / "model.py").read_text(encoding="utf-8")
        success = "fixed" in content
        return CompileResult(
            success=success,
            exit_code=0 if success else 1,
            stderr="syntax error at /tmp/generated/model.py:12345" if not success else "",
            compiler_fingerprint=f"{target.id}:{target.compiler_version}",
            host_correct=True if success else None,
            static_metrics={"source_bytes": 20, "artifact_bytes": 30},
        )


class PartialCompiler(FakeCompiler):
    def compile(self, candidate_dir: Path, output_dir: Path, manifest, target) -> CompileResult:
        if target.backend == Backend.AMD_XDNA2:
            return CompileResult(
                success=False,
                exit_code=1,
                stderr="persistent AIE compiler error",
                compiler_fingerprint=f"{target.id}:{target.compiler_version}",
            )
        return super().compile(candidate_dir, output_dir, manifest, target)


class AnalysisRepairProvider(FakeProvider):
    rejected = False

    def generate(self, prompt: str, response_model: type[BaseModel], model: str | None = None) -> ProviderResponse:
        response = super().generate(prompt, response_model, model)
        if response_model is KernelIR and not self.rejected:
            self.rejected = True
            response.value.operations[0].op = "transpose"
            response.value.operations[0].inputs = ["x"]
            return ProviderResponse(
                response.value,
                response.metadata,
                response.value.model_dump(mode="json"),
            )
        return response


class InterruptedAnalysisProvider(FakeProvider):
    failures = 3

    def generate(self, prompt: str, response_model: type[BaseModel], model: str | None = None) -> ProviderResponse:
        if response_model is KernelIR and self.failures:
            self.failures -= 1
            raise ProviderError("temporary provider interruption")
        return super().generate(prompt, response_model, model)


def test_full_graph_fans_out_targets_and_evaluates_all_proposals(tmp_path: Path) -> None:
    manifest_path = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()
    source_path = manifest_path.parent / "kernel.cu"
    manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    provider = FakeProvider(manifest)
    settings = Settings(
        database_path=tmp_path / "state.sqlite",
        runs_path=tmp_path / "runs",
        repository_path=Path.cwd(),
        provider="codex-cli",
    )
    request = TranslationRequest(
        source_path=str(source_path),
        manifest_path=str(manifest_path),
        targets=[AMD_XDNA2_NPU2, INTEL_NPU_4000],
        provider="codex-cli",
        search_rounds=3,
        branching_factor=3,
        debug_retries=2,
    )
    result = asyncio.run(translate(request, settings, provider=provider, compiler=FakeCompiler()))
    assert result.status == "completed"
    assert {item.target.id for item in result.targets} == {"amd_xdna2_npu2", "intel_npu_4000"}
    assert all(item.candidates_evaluated == 10 for item in result.targets)
    assert provider.calls.count("KernelIR") == 1
    assert provider.calls.count("CodeBundle") == 2
    assert provider.calls.count("ProposalSet") == 6
    assert Path(result.report_path).is_file()


def test_one_shot_has_one_call_and_compile_per_target_without_memory(tmp_path: Path, monkeypatch) -> None:
    from npu_agent.database import Database

    manifest_path = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()
    manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    provider = FakeProvider(manifest)
    compiler = CountingCompiler()
    monkeypatch.setattr(
        Database,
        "retrieve_knowledge",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("baseline queried knowledge")),
    )
    settings = Settings(
        database_path=tmp_path / "state.sqlite",
        runs_path=tmp_path / "runs",
        repository_path=Path.cwd(),
        provider="codex-cli",
    )
    request = TranslationRequest(
        source_path=str(manifest_path.parent / "kernel.cu"),
        manifest_path=str(manifest_path),
        targets=[AMD_XDNA2_NPU2, INTEL_NPU_4000],
        provider="codex-cli",
    )
    result = asyncio.run(translate_one_shot(request, settings, provider=provider, compiler=compiler))
    assert result.status == "completed"
    assert provider.calls == ["CodeBundle", "CodeBundle"]
    assert compiler.calls == 2
    database = Database(settings.database_path)
    roles = database.connection.execute(
        "SELECT role, COUNT(*) FROM agent_calls WHERE run_id=? GROUP BY role", (result.run_id,)
    ).fetchall()
    assert [tuple(row) for row in roles] == [("baseline", 2)]
    assert database.connection.execute(
        "SELECT COUNT(*) FROM candidates WHERE run_id=?", (result.run_id,)
    ).fetchone()[0] == 2
    assert database.connection.execute(
        """SELECT COUNT(*) FROM compile_attempts ca JOIN candidates c ON c.id=ca.candidate_id
           WHERE c.run_id=?""",
        (result.run_id,),
    ).fetchone()[0] == 2
    assert database.connection.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 0


def test_one_shot_does_not_retry_invalid_provider_output(tmp_path: Path) -> None:
    from npu_agent.database import Database

    manifest_path = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()

    class InvalidProvider:
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, response_model, model=None):
            del prompt, response_model, model
            self.calls += 1
            raise ProviderError(
                "invalid schema",
                {
                    "provider": "fake",
                    "model": "fake",
                    "prompt_sha256": "prompt",
                    "schema_sha256": "schema",
                    "exit_code": 0,
                    "schema_valid": False,
                },
                {"invalid": True},
            )

    provider = InvalidProvider()
    compiler = CountingCompiler()
    settings = Settings(
        database_path=tmp_path / "state.sqlite",
        runs_path=tmp_path / "runs",
        repository_path=Path.cwd(),
        provider="codex-cli",
    )
    request = TranslationRequest(
        source_path=str(manifest_path.parent / "kernel.cu"),
        manifest_path=str(manifest_path),
        targets=[AMD_XDNA2_NPU2],
        provider="codex-cli",
    )
    result = asyncio.run(translate_one_shot(request, settings, provider=provider, compiler=compiler))
    assert result.status == "failed"
    assert provider.calls == 1
    assert compiler.calls == 0
    database = Database(settings.database_path)
    assert database.connection.execute(
        "SELECT COUNT(*) FROM compile_attempts ca JOIN candidates c ON c.id=ca.candidate_id WHERE c.run_id=?",
        (result.run_id,),
    ).fetchone()[0] == 1
    call = database.connection.execute(
        "SELECT schema_valid, response_json FROM agent_calls WHERE run_id=?", (result.run_id,)
    ).fetchone()
    assert call["schema_valid"] == 0
    assert json.loads(call["response_json"]) == {"invalid": True}


def test_compile_failure_is_debugged_and_verified_lesson_is_retrievable(tmp_path: Path, monkeypatch) -> None:
    from npu_agent.database import Database

    queries: list[tuple[str, str]] = []
    original_retrieve = Database.retrieve_knowledge

    def recording_retrieve(self, role, query, *args, **kwargs):
        queries.append((role, query))
        return original_retrieve(self, role, query, *args, **kwargs)

    monkeypatch.setattr(Database, "retrieve_knowledge", recording_retrieve)
    manifest_path = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()
    manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    settings = Settings(
        database_path=tmp_path / "state.sqlite",
        runs_path=tmp_path / "runs",
        repository_path=Path.cwd(),
        provider="codex-cli",
    )
    request = TranslationRequest(
        source_path=str(manifest_path.parent / "kernel.cu"),
        manifest_path=str(manifest_path),
        targets=[INTEL_NPU_4000],
        provider="codex-cli",
        search_rounds=0,
        debug_retries=2,
    )
    result = asyncio.run(
        translate(request, settings, provider=RepairProvider(manifest), compiler=RepairCompiler())
    )
    assert result.status == "completed"
    assert result.targets[0].candidates_evaluated == 2
    assert result.targets[0].winner.debug_attempt == 1
    database = Database(settings.database_path)
    lessons = database.retrieve_lessons(INTEL_NPU_4000, "add syntax")
    assert len(lessons) == 1
    assert lessons[0]["verified"] == 1
    assert any(role == "debug" and "syntax error" in query for role, query in queries)


def test_one_backend_failure_produces_partial_result(tmp_path: Path) -> None:
    manifest_path = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()
    manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    provider = RepairProvider(manifest)
    settings = Settings(
        database_path=tmp_path / "state.sqlite",
        runs_path=tmp_path / "runs",
        repository_path=Path.cwd(),
        provider="codex-cli",
    )
    request = TranslationRequest(
        source_path=str(manifest_path.parent / "kernel.cu"),
        manifest_path=str(manifest_path),
        targets=[AMD_XDNA2_NPU2, INTEL_NPU_4000],
        provider="codex-cli",
        search_rounds=0,
        debug_retries=1,
    )
    result = asyncio.run(translate(request, settings, provider=provider, compiler=PartialCompiler()))
    assert result.status == "partial"
    assert {item.target.id: item.status for item in result.targets} == {
        "amd_xdna2_npu2": "failed",
        "intel_npu_4000": "completed",
    }


def test_analysis_mismatch_is_repaired(tmp_path: Path) -> None:
    manifest_path = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()
    manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    provider = AnalysisRepairProvider(manifest)
    settings = Settings(
        database_path=tmp_path / "state.sqlite",
        runs_path=tmp_path / "runs",
        repository_path=Path.cwd(),
        provider="codex-cli",
    )
    request = TranslationRequest(
        source_path=str(manifest_path.parent / "kernel.cu"),
        manifest_path=str(manifest_path),
        targets=[INTEL_NPU_4000],
        provider="codex-cli",
        search_rounds=0,
    )
    result = asyncio.run(translate(request, settings, provider=provider, compiler=FakeCompiler()))
    assert result.status == "completed"
    assert provider.calls.count("KernelIR") == 2


def test_failed_node_resumes_from_sqlite_checkpoint(tmp_path: Path) -> None:
    manifest_path = Path("examples/classic/01_cuda_vector_add/manifest.json").resolve()
    manifest = KernelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    request = TranslationRequest(
        source_path=str(manifest_path.parent / "kernel.cu"),
        manifest_path=str(manifest_path),
        targets=[INTEL_NPU_4000],
        provider="codex-cli",
        search_rounds=0,
    )
    settings = Settings(
        database_path=tmp_path / "state.sqlite",
        runs_path=tmp_path / "runs",
        repository_path=Path.cwd(),
        provider="codex-cli",
    )
    provider = InterruptedAnalysisProvider(manifest)
    graph = build_workflow(settings, provider=provider, compiler=FakeCompiler())
    run_id = "checkpoint-resume"
    initial = {
        "run_id": run_id,
        "request": request.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json"),
        "source": Path(request.source_path).read_text(encoding="utf-8"),
        "target_results": [],
    }
    try:
        graph.invoke(initial, {"configurable": {"thread_id": run_id}})
    except RuntimeError as exc:
        assert "temporary provider interruption" in str(exc)
    else:
        raise AssertionError("the initial run should have been interrupted")
    result = resume_run(graph, run_id)
    assert result.status == "completed"
