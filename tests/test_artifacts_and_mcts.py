from pathlib import Path

import pytest

from npu_agent.artifacts import materialize_bundle
from npu_agent.mcts import MCTSTree, evaluate_compile, select_winner
from npu_agent.models import Backend, Candidate, CodeBundle, CompileResult, GeneratedFile


def bundle(content: str = "pass\n") -> CodeBundle:
    return CodeBundle(
        backend=Backend.INTEL_OPENVINO,
        files=[GeneratedFile(relative_path="model.py", content=content)],
    )


def candidate(candidate_id: str, reward: float | None = None) -> Candidate:
    item = Candidate(
        id=candidate_id,
        run_id="run",
        target_id="intel_npu_4000",
        label=candidate_id,
        rationale="test",
        bundle=bundle(candidate_id),
    )
    if reward is not None:
        result = CompileResult(
            success=True,
            exit_code=0,
            compiler_fingerprint="test",
            host_correct=True,
            static_metrics={"source_bytes": 10, "artifact_bytes": 10},
        )
        item.evaluation = evaluate_compile(candidate_id, result)
        item.evaluation.reward = reward
    return item


def test_materialize_bundle_restricts_backend_files(tmp_path: Path) -> None:
    materialize_bundle(bundle(), tmp_path)
    assert (tmp_path / "model.py").read_text() == "pass\n"
    invalid = CodeBundle(
        backend=Backend.INTEL_OPENVINO,
        files=[GeneratedFile(relative_path="helper.py", content="pass")],
    )
    with pytest.raises(ValueError, match="unexpected"):
        materialize_bundle(invalid, tmp_path / "bad")


def test_mcts_backpropagation_and_winner_tiers() -> None:
    root = candidate("root", 0.4)
    tree = MCTSTree(root, branching_factor=3)
    children = [candidate(f"child-{index}", 0.5 + index * 0.01) for index in range(3)]
    for index, child in enumerate(children):
        child.evaluation.static_score = 0.1 * index
    tree.add_children(tree.root, children)
    assert tree.root.visits == 4
    assert tree.select_for_expansion().candidate in children
    assert select_winner([root, *children]) == children[-1]


def test_measured_evidence_outranks_offline_score() -> None:
    offline = candidate("offline")
    offline.evaluation = evaluate_compile(
        offline.id,
        CompileResult(
            success=True,
            exit_code=0,
            compiler_fingerprint="amd",
            static_metrics={"source_bytes": 1, "artifact_bytes": 1, "vectorization_signals": 8},
        ),
    )
    measured = candidate("measured")
    measured.evaluation = evaluate_compile(
        measured.id,
        CompileResult(
            success=True,
            exit_code=0,
            compiler_fingerprint="intel",
            host_correct=True,
            hardware_correct=True,
            latency_p50_ms=100,
            latency_p95_ms=120,
        ),
    )
    assert select_winner([offline, measured]) == measured


def test_hardware_correctness_failure_cannot_win() -> None:
    failed = candidate("hardware-mismatch")
    failed.evaluation = evaluate_compile(
        failed.id,
        CompileResult(
            success=True,
            exit_code=0,
            compiler_fingerprint="device",
            hardware_correct=False,
            latency_p50_ms=1,
            latency_p95_ms=2,
        ),
    )
    assert failed.evaluation.reward == 0
    assert select_winner([failed]) is None
