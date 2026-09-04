from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings
from .ir import execute_oracle, generate_inputs
from .models import Backend, CompileResult, KernelManifest, TargetProfile


class Compiler(Protocol):
    def compile(
        self,
        candidate_dir: Path,
        output_dir: Path,
        manifest: KernelManifest,
        target: TargetProfile,
    ) -> CompileResult: ...


def _bounded(value: str, limit: int = 200_000) -> str:
    return value if len(value) <= limit else value[-limit:]


class DockerCompiler:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not shutil.which("docker"):
            raise RuntimeError("docker is required for NPU compilation")

    def _docker_prefix(self, candidate_dir: Path, output_dir: Path, target: TargetProfile) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            "2",
            "--memory",
            "8g",
            "--pids-limit",
            "256",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp",
            "--tmpfs",
            "/tmp:rw,size=2g,exec",
            "-v",
            f"{candidate_dir.resolve()}:/candidate:ro",
            "-v",
            f"{output_dir.resolve()}:/output:rw",
            target.compiler_image,
        ]

    def compile(
        self,
        candidate_dir: Path,
        output_dir: Path,
        manifest: KernelManifest,
        target: TargetProfile,
    ) -> CompileResult:
        started = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        if target.backend == Backend.INTEL_OPENVINO:
            command = self._intel_command(candidate_dir, output_dir, manifest, target)
        else:
            command = self._amd_command(candidate_dir, output_dir, target)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.settings.compiler_timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stdout = _bounded(completed.stdout)
            stderr = _bounded(completed.stderr)
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = _bounded(exc.stdout or "")
            stderr = f"compiler timed out after {self.settings.compiler_timeout_seconds}s"

        if target.backend == Backend.AMD_XDNA2:
            required = [output_dir / "final.xclbin", output_dir / "insts.bin", output_dir / "design.mlir"]
            expected = required
        else:
            required = [output_dir / "model.xml", output_dir / "compiled.blob"]
            expected = [*required, output_dir / "model.bin"]
        artifacts = {
            path.name: str(path)
            for path in expected
            if path.is_file() and path.stat().st_size > 0
        }
        success = exit_code == 0 and all(path.name in artifacts for path in required)
        if target.backend == Backend.AMD_XDNA2 and success:
            mlir = (output_dir / "design.mlir").read_text(encoding="utf-8", errors="replace")
            success = "aie.device(npu2" in mlir
            if not success:
                stderr += "\nGenerated MLIR does not target npu2"
        report_path = output_dir / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        source_bytes = sum(path.stat().st_size for path in candidate_dir.iterdir() if path.is_file())
        artifact_bytes = sum(Path(path).stat().st_size for path in artifacts.values())
        source_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore") for path in candidate_dir.iterdir() if path.is_file()
        ).lower()
        vector_signals = sum(source_text.count(token) for token in ("vector", "v16", "v32", "tile", "objectfifo"))
        result = CompileResult(
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            artifacts=artifacts,
            compiler_fingerprint=f"{target.id}:{target.compiler_version}",
            duration_seconds=time.perf_counter() - started,
            evaluation_duration_seconds=report.get("evaluation_duration_seconds", 0.0),
            host_correct=report.get("host_correct"),
            cpu_latency_p50_ms=report.get("cpu_latency_p50_ms"),
            cpu_latency_p95_ms=report.get("cpu_latency_p95_ms"),
            static_metrics={
                "source_bytes": float(source_bytes),
                "artifact_bytes": float(artifact_bytes),
                "vectorization_signals": float(vector_signals),
            },
        )
        return self._hardware_evaluate(result, target)

    def _amd_command(self, candidate_dir: Path, output_dir: Path, target: TargetProfile) -> list[str]:
        script = (
            "cd /candidate && "
            "export NPU_CACHE_HOME=/tmp/npu-cache && "
            "python design.py --dev npu2 --emit-mlir > /output/design.mlir && "
            "python design.py --dev npu2 --xclbin-path /output/final.xclbin "
            "--insts-path /output/insts.bin && "
            "test -s /output/final.xclbin && test -s /output/insts.bin && "
            "xclbinutil --info --input /output/final.xclbin"
        )
        return self._docker_prefix(candidate_dir, output_dir, target) + ["bash", "-lc", script]

    def _intel_command(
        self,
        candidate_dir: Path,
        output_dir: Path,
        manifest: KernelManifest,
        target: TargetProfile,
    ) -> list[str]:
        inputs = generate_inputs(manifest)
        tensor_inputs = {name: value for name, value in inputs.items() if isinstance(value, np.ndarray)}
        expected = execute_oracle(manifest, inputs)
        np.savez(output_dir / "inputs.npz", **tensor_inputs)
        np.savez(output_dir / "expected.npz", **expected)
        payload = manifest.model_dump(mode="json")
        payload["target_hardware"] = target.hardware
        (output_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        tools_dir = self.settings.repository_path / "playground" / "tools"
        prefix = self._docker_prefix(candidate_dir, output_dir, target)
        insertion = prefix.index(target.compiler_image)
        prefix[insertion:insertion] = ["-v", f"{tools_dir.resolve()}:/tools:ro"]
        return prefix + ["python", "/tools/intel_runner.py"]

    def _hardware_evaluate(self, result: CompileResult, target: TargetProfile) -> CompileResult:
        if not result.success or not target.hardware_runner:
            return result
        payload = json.dumps({"target": target.model_dump(mode="json"), "artifacts": result.artifacts})
        started = time.perf_counter()
        completed = subprocess.run(
            target.hardware_runner,
            input=payload,
            capture_output=True,
            text=True,
            timeout=self.settings.compiler_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            result.evaluation_duration_seconds += time.perf_counter() - started
            result.stderr += f"\nhardware runner failed: {_bounded(completed.stderr, 4000)}"
            return result
        report = json.loads(completed.stdout)
        result.hardware_correct = bool(report["correct"])
        if not result.hardware_correct:
            result.stderr += "\ntarget-NPU output does not match oracle"
        result.latency_p50_ms = float(report["latency_p50_ms"])
        result.latency_p95_ms = float(report["latency_p95_ms"])
        result.hardware_warmup_count = int(report["warmup_count"])
        result.hardware_iteration_count = int(report["iteration_count"])
        if report.get("throughput_per_second") is not None:
            result.hardware_throughput_per_second = float(report["throughput_per_second"])
        result.evaluation_duration_seconds += time.perf_counter() - started
        return result


def build_images(settings: Settings, target_ids: list[str]) -> None:
    from .config import TARGETS

    dockerfiles = {
        Backend.AMD_XDNA2: "playground/dockerfiles/Dockerfile.xdna2-offline",
        Backend.INTEL_OPENVINO: "playground/dockerfiles/Dockerfile.intel-npu-offline",
    }
    for target_id in target_ids:
        target = TARGETS[target_id]
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                target.compiler_image,
                "-f",
                dockerfiles[target.backend],
                ".",
            ],
            cwd=settings.repository_path,
            check=True,
        )


def smoke_images(settings: Settings, target_ids: list[str]) -> None:
    from .config import TARGETS

    for target_id in target_ids:
        target = TARGETS[target_id]
        if target.backend == Backend.AMD_XDNA2:
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--env",
                "HOME=/tmp",
                "--tmpfs",
                "/tmp:rw,size=2g,exec",
                target.compiler_image,
                "bash",
                "-lc",
                "mkdir -p /tmp/smoke && "
                "python /opt/mlir-aie/programming_guide/section-4/section-4a/vector_scalar_mul.py "
                "--dev npu2 --xclbin-path /tmp/smoke/final.xclbin --insts-path /tmp/smoke/insts.bin && "
                "test -s /tmp/smoke/final.xclbin && test -s /tmp/smoke/insts.bin && "
                "xclbinutil --info --input /tmp/smoke/final.xclbin >/dev/null",
            ]
        else:
            tools_dir = settings.repository_path / "playground" / "tools"
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--env",
                "HOME=/tmp",
                "--tmpfs",
                "/tmp:rw,size=2g,exec",
                "-v",
                f"{tools_dir.resolve()}:/tools:ro",
                target.compiler_image,
                "python",
                "/tools/intel_smoke.py",
            ]
        subprocess.run(command, cwd=settings.repository_path, check=True)


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
