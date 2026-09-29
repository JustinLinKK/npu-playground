import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from npu_agent.experiment import discover_kernels
from npu_agent.study_v4 import StudyConfig
from npu_agent.testcases import generate_cases
from npu_agent.validation import sha256, write_json

spec = importlib.util.spec_from_file_location("hidden_evaluation", Path(__file__).resolve().parents[1] / "research/hidden_evaluation.py")
hidden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hidden)


def fixture(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(Path("examples/classic/01_cuda_vector_add"), corpus / "add")
    selected = discover_kernels(corpus)
    name = "cuda_vector_add"
    source, manifest, contract = selected[name]
    inputs = {"x": np.full((4096,), .3125, dtype=np.float32), "y": np.full((4096,), -.1875, dtype=np.float32)}
    archive = tmp_path / "private.npz"
    np.savez(archive, **inputs)
    suite = tmp_path / "suite.json"
    document = {"protocol": "hidden-inputs-v1", "reviewed_by": "synthetic test", "review_notes": "test fixture only",
                "kernels": {name: {"manifest_sha256": sha256(manifest), "cases": [
                    {"id": "final-1", "path": archive.name, "sha256": sha256(archive)}]}}}
    write_json(suite, document)
    root = tmp_path / "campaign"
    config = StudyConfig(stage="confirmatory", repetitions=5, holdout_manifest="synthetic-test-only")
    case = {"id": "one", "kernel": name, "family": "add", "target": "intel_npu_4000", "arm": "direct", "repetition": 1}
    campaign = {"status": "ready", "config": config.model_dump(), "cases": [case],
                "identity": {"corpus": {"corpus": str(corpus), "hashes": {"add/manifest.json": sha256(manifest),
                      "add/kernel.cu": sha256(source)}}, "source_hashes": {}}}
    write_json(root / "campaign.json", campaign)
    return root, suite, tmp_path / "commitment.json", selected


def test_load_hidden_rejects_feedback_reuse_and_altered_archives(tmp_path):
    root, suite, commitment, selected = fixture(tmp_path)
    cases, _ = hidden.load_hidden(suite, selected)
    assert len(cases["cuda_vector_add"]) == 1
    path = tmp_path / "private.npz"
    feedback = generate_cases(selected["cuda_vector_add"][2])[0]
    np.savez(path, **{k: v for k, v in feedback.inputs.items() if isinstance(v, np.ndarray)})
    with pytest.raises(ValueError, match="hash changed"):
        hidden.load_hidden(suite, selected)
    document = json.loads(suite.read_text())
    document["kernels"]["cuda_vector_add"]["cases"][0]["sha256"] = sha256(path)
    write_json(suite, document)
    with pytest.raises(ValueError, match="overlap"):
        hidden.load_hidden(suite, selected)


def test_seal_is_prospective_and_immutable(tmp_path):
    root, suite, commitment, _ = fixture(tmp_path)
    hidden.seal(root, suite, commitment)
    with pytest.raises(ValueError, match="overwrite"):
        hidden.seal(root, suite, commitment)
    write_json(root / "cases/one/report.json", {"status": "running"})
    with pytest.raises(ValueError, match="before"):
        hidden.seal(root, suite, tmp_path / "another.json")


def test_no_final_evaluation_while_repair_can_run(tmp_path):
    root, suite, commitment, _ = fixture(tmp_path)
    hidden.seal(root, suite, commitment)
    with pytest.raises(ValueError, match="until the campaign completes"):
        hidden.evaluate(root, suite, commitment, tmp_path / "final")
    assert not (tmp_path / "final").exists()


def test_completed_final_evaluation_uses_hidden_inputs_without_mutating_campaign(tmp_path, monkeypatch):
    from test_experiment import CorpusCompiler
    from npu_agent.models import CodeBundle, Backend
    import npu_agent.compilers as compilers

    root, suite, commitment, selected = fixture(tmp_path)
    hidden.seal(root, suite, commitment)
    campaign = json.loads((root / "campaign.json").read_text())
    campaign["status"] = "completed"
    write_json(root / "campaign.json", campaign)
    write_json(root / "preflight.json", {"identity": {"compilers": {target: {} for target in campaign["config"]["targets"]}}})
    bundle = CodeBundle(backend=Backend.INTEL_OPENVINO, files=[{"relative_path": "model.py", "content": "# synthetic fixture"}])
    state = {"status": "completed", "phase": None, "active_seconds": 1, "cycles": [
        {"number": 1, "status": "passed", "bundle": bundle.model_dump(mode="json"), "calls": {}}]}
    write_json(root / "cases/one/report.json", state)
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original_generator = compilers.generate_cases

    class FinalCompiler(CorpusCompiler):
        def __init__(self, settings):
            super().__init__()

        def identity(self, target):
            return {}

        def compile(self, candidate, output, manifest, target):
            cases = compilers.generate_cases(manifest)
            assert len(cases) == 1 and cases[0].id == "final-1"
            assert cases[0].inputs["x"][0] == .3125
            from npu_agent.validation import finalize_validation
            result = super().compile(candidate, output, manifest, target)
            finalize_validation(result.validation, "offline-validated")
            return result

    monkeypatch.setattr(hidden, "StudyCompiler", FinalCompiler)
    report = hidden.evaluate(root, suite, commitment, tmp_path / "final")
    assert report["status"] == "completed" and report["cases"][0]["hidden_pass"]
    assert compilers.generate_cases is original_generator
    assert all(p.read_bytes() == raw for p, raw in original.items())
    with pytest.raises(ValueError, match="new final-output"):
        hidden.evaluate(root, suite, commitment, tmp_path / "final")

    class BrokenCompiler(FinalCompiler):
        def compile(self, *args):
            raise OSError("unavailable test compiler")

    monkeypatch.setattr(hidden, "StudyCompiler", BrokenCompiler)
    with pytest.raises(OSError):
        hidden.evaluate(root, suite, commitment, tmp_path / "blocked-final")
    failed = json.loads((tmp_path / "blocked-final/results.json").read_text())
    assert failed["status"] == "blocked" and failed["cases"][0]["duration_seconds"] >= 0
    assert compilers.generate_cases is original_generator
