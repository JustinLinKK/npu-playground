import importlib.util
import json
from pathlib import Path
from threading import Event

import pytest

from npu_agent.scenarios import ExecutableKernelIR
from npu_agent.study_v4 import run_case
from npu_agent.study_v4_reporting import case_metrics
from npu_agent.validation import sha256
from test_study_v4 import Provider, RepairCompiler, setup_case

spec = importlib.util.spec_from_file_location("diagnostic", Path(__file__).resolve().parents[1] / "research/diagnostic_v1.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_reference_seed_reuses_backend_loop_without_synthetic_model_call(tmp_path):
    request, settings, config, case = setup_case(tmp_path, "structured_ir")
    ir_path = Path("configs/research-next-steps-v1/cuda_vector_add.json")
    reference = {"ir_sha256": sha256(ir_path), "source_sha256": sha256(Path(request.source_path)),
                 "manifest_sha256": sha256(Path(request.manifest_path)), "reviewer": "test"}
    case.update(treatment="audited_ir_current_backend", reference_sha256=reference["ir_sha256"])
    diagnostic.seed_reference(request, settings, config, case, reference, ir_path)

    class BackendOnly(Provider):
        def generate(self, prompt, response_model, model=None):
            assert response_model is not ExecutableKernelIR
            return super().generate(prompt, response_model, model)

    provider, compiler = BackendOnly(), RepairCompiler()
    state = run_case(request, settings, config, case, provider, compiler, Event())
    assert state["status"] == "completed" and compiler.calls == 2
    metrics = case_metrics(case, state, config)
    assert metrics["llm_calls"] == 2 and metrics["known_tokens"] == 30
    assert "intermediate" not in state["cycles"][0]["calls"]
    assert state["cycles"][1]["intermediate_reused_from"] == 1
    assert state["cycles"][0]["semantic_validation"]["duration_seconds"] > 0
    assert state["reference_preparation"] == reference
    original = (settings.runs_path / case["id"] / "report.json").read_bytes()
    diagnostic.seed_reference(request, settings, config, case, reference, ir_path)
    assert (settings.runs_path / case["id"] / "report.json").read_bytes() == original
    with pytest.raises(ValueError, match="binding changed"):
        diagnostic.seed_reference(request, settings, config, case, dict(reference, ir_sha256="changed"), ir_path)


@pytest.mark.parametrize("acknowledge_unknown", [False, True])
def test_diagnostic_gate_prevents_paid_execution_before_pilot_audit(tmp_path, monkeypatch, acknowledge_unknown):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"pilot": str(tmp_path / "pilot")}))
    monkeypatch.setattr(diagnostic, "build_audit", lambda _: {"accounting_complete": False})
    monkeypatch.setattr(diagnostic, "StudyProvider", lambda *a: pytest.fail("must not construct provider"))
    with pytest.raises(ValueError, match="complete and reconcile"):
        diagnostic.run(selection, tmp_path / "diagnostic", tmp_path / "ledger.json", acknowledge_unknown)
    assert not (tmp_path / "diagnostic").exists()
