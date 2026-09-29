from pathlib import Path

from npu_agent.config import AMD_XDNA2_NPU2, INTEL_NPU_4000
from npu_agent.database import Database, normalize_error


def test_memory_is_scoped_and_only_verified_lessons_are_retrieved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("npu_agent.database._KNOWLEDGE_ENABLED", True)
    database = Database(tmp_path / "state.sqlite")
    database.import_knowledge(
        "optimize", "Object FIFO", "Use balanced object fifo depths for matmul", "matmul fifo",
        backend="amd_xdna2", hardware="npu2"
    )
    knowledge = database.retrieve_knowledge("optimize", "matmul fifo", backend="amd_xdna2", hardware="npu2")
    assert [item["title"] for item in knowledge] == ["Object FIFO"]
    assert database.retrieve_knowledge("analysis", "matmul fifo") == []

    database.add_lesson(
        AMD_XDNA2_NPU2, "amd_xdna2_npu2:mlir-aie-v1.4.2", ["matmul"], "error-a", "alignment failure", "align buffers", True
    )
    database.add_lesson(
        AMD_XDNA2_NPU2, "amd_xdna2_npu2:mlir-aie-v1.4.2", ["matmul"], "error-b", "tile failure", "no fix", False
    )
    lessons = database.retrieve_lessons(AMD_XDNA2_NPU2, "matmul alignment tile")
    assert len(lessons) == 1
    assert lessons[0]["fix_summary"] == "align buffers"


def test_error_fingerprint_removes_paths_addresses_and_large_numbers() -> None:
    first = normalize_error("/tmp/a.cc:42 address 0x1234 failed on allocation 123456")
    second = normalize_error("/work/b.cc:99 address 0xabcd failed on allocation 987654")
    assert first == second


def test_knowledge_is_inactive_even_in_a_populated_database(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite"
    database = Database(path)
    assert database.connection.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0] == 0
    database.seed_builtin_knowledge()
    monkeypatch.setattr("npu_agent.database._KNOWLEDGE_ENABLED", True)
    database.add_lesson(AMD_XDNA2_NPU2, "fingerprint", ["matmul"], "error", "diagnosis", "fix", True)
    initial = len(database.list_memory()["knowledge"])
    database.close()
    monkeypatch.setattr("npu_agent.database._KNOWLEDGE_ENABLED", False)
    database = Database(path)
    assert len(database.list_memory()["knowledge"]) == initial
    assert len(database.list_memory()["lessons"]) == 1
    assert database.retrieve_knowledge("coding", "matmul", backend="amd_xdna2", hardware="npu2") == []
    assert database.retrieve_lessons(AMD_XDNA2_NPU2, "matmul", compiler_fingerprint="fingerprint") == []
    database.add_lesson(AMD_XDNA2_NPU2, "fingerprint", ["matmul"], "error", "diagnosis", "fix", True)
    assert len(database.list_memory()["lessons"]) == 1
    database.close()


def test_builtin_knowledge_seed_is_idempotent_and_exactly_target_scoped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("npu_agent.database._KNOWLEDGE_ENABLED", True)
    database = Database(tmp_path / "state.sqlite")
    initial = database.connection.execute("SELECT COUNT(*) FROM knowledge_items").fetchone()[0]
    assert initial > 0
    assert database.seed_builtin_knowledge() == {"inserted": 0, "skipped": initial}
    amd = database.retrieve_knowledge(
        "coding",
        "Program Runtime Worker ObjectFifo",
        backend=AMD_XDNA2_NPU2.backend.value,
        hardware=AMD_XDNA2_NPU2.hardware,
        target_id=AMD_XDNA2_NPU2.id,
        vendor=AMD_XDNA2_NPU2.vendor,
        compiler_fingerprint=f"{AMD_XDNA2_NPU2.id}:{AMD_XDNA2_NPU2.compiler_version}",
    )
    intel = database.retrieve_knowledge(
        "coding",
        "Program Runtime Worker ObjectFifo",
        backend=INTEL_NPU_4000.backend.value,
        hardware=INTEL_NPU_4000.hardware,
        target_id=INTEL_NPU_4000.id,
        vendor=INTEL_NPU_4000.vendor,
        compiler_fingerprint=f"{INTEL_NPU_4000.id}:{INTEL_NPU_4000.compiler_version}",
    )
    assert amd
    assert all(item["target_id"] == AMD_XDNA2_NPU2.id for item in amd)
    assert intel == []


def test_legacy_payload_migration_preserves_unknown_execution(tmp_path):
    import json
    from npu_agent.models import Candidate, CodeBundle, CompileResult, TranslationRequest, KernelManifest
    from npu_agent.config import AMD_XDNA2_NPU2
    from npu_agent.mcts import evaluate_compile
    from npu_agent.validation import meets_policy

    path = tmp_path / 'old.sqlite'
    database = Database(path)
    manifest_path = Path('examples/classic/01_cuda_vector_add/manifest.json')
    manifest = KernelManifest.model_validate_json(manifest_path.read_text())
    request = TranslationRequest(source_path='kernel.cu', manifest_path=str(manifest_path), targets=[AMD_XDNA2_NPU2])
    database.start_run('old', manifest, request)
    candidate = Candidate(id='old-amd', run_id='old', target_id=AMD_XDNA2_NPU2.id, label='legacy', rationale='',
                          bundle=CodeBundle(backend=AMD_XDNA2_NPU2.backend, files=[]))
    database.add_candidate(candidate)
    result = CompileResult(success=True, exit_code=0, compiler_fingerprint='old')
    database.add_compile_attempt(candidate.id, 0, result)
    database.add_evaluation(evaluate_compile(candidate.id, result))
    with database.connection:
        payload = evaluate_compile(candidate.id, result).model_dump(mode='json')
        payload.pop('validation')
        database.connection.execute('UPDATE evaluations SET evaluation_json=?', (json.dumps(payload),))
        database.connection.execute('PRAGMA user_version=6')
    database.connection.close()
    migrated = Database(path)
    row = migrated.connection.execute('SELECT evaluation_json FROM evaluations').fetchone()
    from npu_agent.models import Evaluation
    evaluation = Evaluation.model_validate_json(row[0])
    assert evaluation.validation.legacy
    assert evaluation.validation.stages['target_execution'].correct is None
    assert not meets_policy(evaluation.validation, 'offline-validated')
    assert migrated.get_run('old')['status'] == 'running'
