from pathlib import Path

from npu_agent.config import AMD_XDNA2_NPU2, INTEL_NPU_4000
from npu_agent.database import Database, normalize_error


def test_memory_is_scoped_and_only_verified_lessons_are_retrieved(tmp_path: Path) -> None:
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


def test_builtin_knowledge_seed_is_idempotent_and_exactly_target_scoped(tmp_path: Path) -> None:
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
