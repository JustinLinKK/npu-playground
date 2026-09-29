"""Public API for the NPU translation framework."""

from .config import Settings
from .models import (
    BaselineResult,
    Candidate,
    Evaluation,
    KernelIR,
    KernelManifest,
    TargetProfile,
    TranslationRequest,
    TranslationResult,
)

__all__ = [
    "Candidate",
    "BaselineResult",
    "Evaluation",
    "KernelIR",
    "KernelManifest",
    "Settings",
    "TargetProfile",
    "TranslationRequest",
    "TranslationResult",
    "build_workflow",
    "translate",
    "translate_one_shot",
]


def __getattr__(name):
    # Offline evaluators need only NumPy/Pydantic, not orchestration providers.
    if name == "translate_one_shot":
        from .baseline import translate_one_shot
        return translate_one_shot
    if name in ("build_workflow", "translate"):
        from . import workflow
        return getattr(workflow, name)
    raise AttributeError(name)
