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


def test_concurrent_artifact_publication_never_exposes_partial_copy(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock
    import npu_agent.artifacts as artifacts

    source = tmp_path / "source.bin"
    content = b"shared compiler artifact" * 100
    source.write_bytes(content)
    started, release, lock = Event(), Event(), Lock()
    original_copy = artifacts.shutil.copyfile
    calls = 0

    def slow_first_copy(source_path, destination):
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        if first:
            Path(destination).write_bytes(content[:10])
            started.set()
            assert release.wait(10)
        return original_copy(source_path, destination)

    monkeypatch.setattr(artifacts.shutil, "copyfile", slow_first_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(artifacts.store_artifact, source, tmp_path / "store")
        try:
            assert started.wait(10)
            second = pool.submit(artifacts.store_artifact, source, tmp_path / "store")
            stored, digest = second.result(timeout=10)
            assert stored.read_bytes() == content
        finally:
            release.set()
        assert first.result(timeout=10) == (stored, digest)


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


def test_host_failure_does_not_invent_target_execution() -> None:
    result = CompileResult(success=True, exit_code=0, compiler_fingerprint="host", host_correct=False)
    evaluation = evaluate_compile("host-mismatch", result)
    assert evaluation.host_correctness is False
    assert evaluation.target_correctness is None
    assert evaluation.reward == 0
    assert evaluation.validation.stages["target_execution"].status == "blocked"
    assert "host output" in evaluation.notes[0]


def test_requested_policy_excludes_compile_only_and_source_tokens_do_not_reward():
    from npu_agent.models import StageResult, ValidationResult, ValidationPolicy
    from npu_agent.mcts import static_score

    plain = CompileResult(success=True, exit_code=0, compiler_fingerprint='test',
                          static_metrics={'vectorization_signals': 0, 'artifact_bytes': 100})
    noisy = plain.model_copy(deep=True)
    noisy.static_metrics = {'vectorization_signals': 100000, 'artifact_bytes': 1}
    assert static_score(plain) == static_score(noisy) == 0
    candidate = Candidate(id='compile-only', run_id='run', target_id='amd_xdna2_npu2', label='compile', rationale='',
                          bundle=CodeBundle(backend=Backend.AMD_XDNA2, files=[]))
    candidate.evaluation = evaluate_compile(candidate.id, noisy)
    assert select_winner([candidate], ValidationPolicy.OFFLINE_VALIDATED) is None
    validated = candidate.model_copy(deep=True)
    validated.id = 'validated'
    plain.validation = ValidationResult(candidate_id=validated.id, target_id=validated.target_id, stages={
        'target_compile': StageResult(status='passed'),
        **{name: StageResult(status='passed', correct=True) for name in
           ('oracle_validation', 'host_execution', 'dataflow_simulation')}})
    plain.host_correct = True
    validated.evaluation = evaluate_compile(validated.id, plain, ValidationPolicy.OFFLINE_VALIDATED)
    assert select_winner([candidate, validated], ValidationPolicy.OFFLINE_VALIDATED).id == 'validated'
