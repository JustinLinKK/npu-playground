from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .artifacts import bundle_hash
from .models import Candidate, CompileResult, Evaluation, EvidenceTier


def static_score(result: CompileResult) -> float:
    if not result.success:
        return 0.0
    source_bytes = max(result.static_metrics.get("source_bytes", 1.0), 1.0)
    artifact_bytes = max(result.static_metrics.get("artifact_bytes", 1.0), 1.0)
    vector_signal = min(result.static_metrics.get("vectorization_signals", 0.0), 8.0) / 8.0
    size_signal = 1.0 / (1.0 + math.log10(source_bytes + artifact_bytes))
    return max(0.0, min(1.0, 0.7 * vector_signal + 0.3 * size_signal))


def evaluate_compile(candidate_id: str, result: CompileResult) -> Evaluation:
    started = time.perf_counter()
    score = static_score(result)
    artifact_bytes = int(result.static_metrics.get("artifact_bytes", 0.0))
    if not result.success:
        return Evaluation(
            candidate_id=candidate_id,
            evidence_tier=EvidenceTier.INVALID,
            compile_success=False,
            correctness=result.hardware_correct if result.hardware_correct is not None else result.host_correct,
            host_correctness=result.host_correct,
            target_correctness=result.hardware_correct,
            reward=0.0,
            static_score=0.0,
            artifact_bytes=artifact_bytes,
            duration_seconds=time.perf_counter() - started,
            notes=["candidate did not compile"],
        )
    if result.hardware_correct is False:
        return Evaluation(
            candidate_id=candidate_id,
            evidence_tier=EvidenceTier.HARDWARE_MEASURED,
            compile_success=True,
            correctness=False,
            host_correctness=result.host_correct,
            target_correctness=False,
            reward=0.0,
            static_score=score,
            latency_p50_ms=result.latency_p50_ms,
            latency_p95_ms=result.latency_p95_ms,
            cpu_latency_p50_ms=result.cpu_latency_p50_ms,
            cpu_latency_p95_ms=result.cpu_latency_p95_ms,
            artifact_bytes=artifact_bytes,
            duration_seconds=time.perf_counter() - started,
            notes=["target-NPU output did not match the oracle"],
        )
    if result.latency_p50_ms is not None and result.hardware_correct is True:
        latency_component = 1.0 / (1.0 + result.latency_p50_ms)
        reward = 0.75 + 0.25 * latency_component
        tier = EvidenceTier.HARDWARE_MEASURED
    elif result.host_correct is True:
        reward = 0.55 + 0.10 * score
        tier = EvidenceTier.HOST_EQUIVALENCE
    else:
        reward = 0.35 + 0.15 * score
        tier = EvidenceTier.OFFLINE_COMPILE
    return Evaluation(
        candidate_id=candidate_id,
        evidence_tier=tier,
        compile_success=True,
        correctness=result.hardware_correct if result.hardware_correct is not None else result.host_correct,
        host_correctness=result.host_correct,
        target_correctness=result.hardware_correct,
        reward=min(reward, 1.0),
        static_score=score,
        latency_p50_ms=result.latency_p50_ms,
        latency_p95_ms=result.latency_p95_ms,
        cpu_latency_p50_ms=result.cpu_latency_p50_ms,
        cpu_latency_p95_ms=result.cpu_latency_p95_ms,
        artifact_bytes=artifact_bytes,
        duration_seconds=time.perf_counter() - started,
    )


@dataclass
class SearchNode:
    candidate: Candidate
    parent: SearchNode | None = None
    children: list[SearchNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def ucb1(self, exploration: float = math.sqrt(2.0)) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = max(self.parent.visits if self.parent else self.visits, 1)
        return self.mean_value + exploration * math.sqrt(math.log(parent_visits + 1) / self.visits)


class MCTSTree:
    def __init__(self, root: Candidate, branching_factor: int = 3) -> None:
        self.root = SearchNode(root)
        self.branching_factor = branching_factor
        if root.evaluation:
            self.backpropagate(self.root, root.evaluation.reward)

    def select_for_expansion(self) -> SearchNode:
        node = self.root
        while len(node.children) >= self.branching_factor:
            node = max(node.children, key=lambda child: (child.ucb1(), child.candidate.id))
        return node

    def add_children(self, parent: SearchNode, candidates: list[Candidate]) -> list[SearchNode]:
        children = [SearchNode(candidate, parent=parent) for candidate in candidates]
        parent.children.extend(children)
        for child in children:
            if child.candidate.evaluation:
                self.backpropagate(child, child.candidate.evaluation.reward)
        return children

    @staticmethod
    def backpropagate(node: SearchNode, reward: float) -> None:
        current: SearchNode | None = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent

    def candidates(self) -> list[Candidate]:
        result: list[Candidate] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            result.append(node.candidate)
            stack.extend(node.children)
        return result


def candidate_sort_key(candidate: Candidate) -> tuple[float, float, float, float, int, str, str]:
    evaluation = candidate.evaluation
    if evaluation is None:
        return (0, 0, float("-inf"), 0, 0, bundle_hash(candidate.bundle), candidate.id)
    latency = -evaluation.latency_p50_ms if evaluation.latency_p50_ms is not None else float("-inf")
    correctness = 1.0 if evaluation.correctness is True else 0.0
    return (
        float(evaluation.evidence_tier),
        correctness,
        latency,
        evaluation.static_score,
        -evaluation.artifact_bytes,
        bundle_hash(candidate.bundle),
        candidate.id,
    )


def select_winner(candidates: list[Candidate]) -> Candidate | None:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.evaluation
        and candidate.evaluation.compile_success
        and candidate.evaluation.correctness is not False
    ]
    return max(eligible, key=candidate_sort_key) if eligible else None
