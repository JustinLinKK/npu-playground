from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .models import Backend, TargetProfile, ValidationPolicy


def _runner(target_id: str) -> dict:
    # Opt-in host executor, e.g. NPU_AGENT_INTEL_NPU_3720_RUNNER="python scripts/wsl_npu_bridge.py ...".
    # NPU_AGENT_<ID>_TIMING="warmup,measure" requests physical timing samples. Unset means offline only.
    prefix = f"NPU_AGENT_{target_id.upper()}"
    command = os.environ.get(f"{prefix}_RUNNER", "").strip()
    if not command:
        return {}
    warmup, measure = (int(value) for value in os.environ.get(f"{prefix}_TIMING", "0,0").split(","))
    return {"hardware_runner": shlex.split(command), "executor_warmup_count": warmup,
            "executor_measurement_count": measure}


AMD_XDNA2_NPU2 = TargetProfile(
    id="amd_xdna2_npu2",
    vendor="amd",
    backend=Backend.AMD_XDNA2,
    hardware="npu2",
    compiler_image="npu-playground-amd-xdna2:v1.4.2",
    compiler_version="mlir-aie-v1.4.2",
    compiler_properties={"device": "npu2", "architecture": "aie2p", "kernel_optimization": "2"},
    **_runner("amd_xdna2_npu2"),
)

INTEL_NPU_4000 = TargetProfile(
    id="intel_npu_4000",
    vendor="intel",
    backend=Backend.INTEL_OPENVINO,
    hardware="4000",
    compiler_image="npu-playground-intel-npu:2026.3.1",
    compiler_version="openvino-2026.3.1",
    compiler_properties={"NPU_COMPILER_TYPE": "PLUGIN", "NPU_PLATFORM": "4000"},
    **_runner("intel_npu_4000"),
)

# Arrow Lake / Meteor Lake NPUs (PCI 8086:AD1D, 7D1D) report DEVICE_ARCHITECTURE 3720.
INTEL_NPU_3720 = TargetProfile(
    id="intel_npu_3720",
    vendor="intel",
    backend=Backend.INTEL_OPENVINO,
    hardware="3720",
    compiler_image="npu-playground-intel-npu:2026.3.1",
    compiler_version="openvino-2026.3.1",
    compiler_properties={"NPU_COMPILER_TYPE": "PLUGIN", "NPU_PLATFORM": "3720"},
    **_runner("intel_npu_3720"),
)

TARGETS = {profile.id: profile for profile in (AMD_XDNA2_NPU2, INTEL_NPU_4000, INTEL_NPU_3720)}
# Historical experiments and CLI defaults keep their original pair; select intel_npu_3720 explicitly.
DEFAULT_TARGETS = [AMD_XDNA2_NPU2.id, INTEL_NPU_4000.id]


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
