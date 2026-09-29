from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import Backend, TargetProfile, ValidationPolicy


AMD_XDNA2_NPU2 = TargetProfile(
    id="amd_xdna2_npu2",
    vendor="amd",
    backend=Backend.AMD_XDNA2,
    hardware="npu2",
    compiler_image="npu-playground-amd-xdna2:v1.4.2",
    compiler_version="mlir-aie-v1.4.2",
    compiler_properties={"device": "npu2", "architecture": "aie2p", "kernel_optimization": "2"},
)

INTEL_NPU_4000 = TargetProfile(
    id="intel_npu_4000",
    vendor="intel",
    backend=Backend.INTEL_OPENVINO,
    hardware="4000",
    compiler_image="npu-playground-intel-npu:2026.3.1",
    compiler_version="openvino-2026.3.1",
    compiler_properties={"NPU_COMPILER_TYPE": "PLUGIN", "NPU_PLATFORM": "4000"},
)

TARGETS = {profile.id: profile for profile in (AMD_XDNA2_NPU2, INTEL_NPU_4000)}


@dataclass(slots=True)
class Settings:
    database_path: Path = Path(".npu-agent/state.sqlite")
    runs_path: Path = Path(".npu-agent/runs")
    repository_path: Path = field(default_factory=Path.cwd)
    provider: str = "openai"
    model: str | None = None
    provider_timeout_seconds: int = 900
    compiler_timeout_seconds: int = 1200
    validation_policy: ValidationPolicy = ValidationPolicy.COMPILE_ONLY
    docker_executable: str = "docker"
    compiler_cpus: int = 4
    compiler_memory: str = "8g"
    max_compiler_jobs: int = 2
    max_generated_bytes: int = 2_000_000
    experiment_id: str | None = None
    role_providers: dict[str, str] = field(default_factory=dict)
    role_models: dict[str, str] = field(default_factory=dict)

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_path.mkdir(parents=True, exist_ok=True)
