from __future__ import annotations

import importlib.util
import inspect
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov


ROOT = Path("/candidate")
OUTPUT = Path("/output")


def load_candidate():
    spec = importlib.util.spec_from_file_location("candidate_model", ROOT / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load candidate model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model"):
        raise RuntimeError("model.py must define build_model(manifest=None)")
    return module


def main() -> None:
    evaluation_started = time.perf_counter()
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    module = load_candidate()
    signature = inspect.signature(module.build_model)
    model = module.build_model(manifest) if signature.parameters else module.build_model()
    if not isinstance(model, ov.Model):
        raise TypeError("build_model must return openvino.Model")

    inputs_archive = np.load(OUTPUT / "inputs.npz")
    inputs = {name: inputs_archive[name] for name in inputs_archive.files}
    expected_archive = np.load(OUTPUT / "expected.npz")

    core = ov.Core()
    cpu_model = core.compile_model(model, "CPU")
    cpu_inputs = {}
    for port in cpu_model.inputs:
        name = port.get_any_name()
        if name not in inputs:
            raise KeyError(f"no generated input for OpenVINO parameter {name}")
        cpu_inputs[name] = inputs[name]
    outputs = cpu_model(cpu_inputs)
    actual_values = list(outputs.values())
    expected_values = [expected_archive[name] for name in expected_archive.files]
    if len(actual_values) != len(expected_values):
        raise ValueError("OpenVINO output count does not match manifest")
    tolerance = manifest["tolerance"]
    correct = all(
        actual.shape == expected.shape
        and np.allclose(
            actual,
            expected,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            equal_nan=tolerance.get("equal_nan", False),
        )
        for actual, expected in zip(actual_values, expected_values, strict=True)
    )
    for _ in range(5):
        cpu_model(cpu_inputs)
    cpu_samples = []
    for _ in range(20):
        started = time.perf_counter_ns()
        cpu_model(cpu_inputs)
        cpu_samples.append((time.perf_counter_ns() - started) / 1_000_000)
    evaluation_duration = time.perf_counter() - evaluation_started
    np.savez(OUTPUT / "actual.npz", **{f"output_{index}": value for index, value in enumerate(actual_values)})
    ov.save_model(model, OUTPUT / "model.xml", compress_to_fp16=False)
    if not correct:
        (OUTPUT / "report.json").write_text(
            json.dumps(
                {
                    "host_correct": False,
                    "cpu_latency_p50_ms": float(np.percentile(cpu_samples, 50)),
                    "cpu_latency_p95_ms": float(np.percentile(cpu_samples, 95)),
                    "evaluation_duration_seconds": evaluation_duration,
                }
            ),
            encoding="utf-8",
        )
        raise ValueError("OpenVINO CPU output does not match oracle")

    config = {"NPU_COMPILER_TYPE": "PLUGIN", "NPU_PLATFORM": manifest["target_hardware"]}
    compiled = core.compile_model(model, "NPU", config)
    blob = compiled.export_model()
    if hasattr(blob, "getvalue"):
        blob = blob.getvalue()
    if not isinstance(blob, bytes) or not blob:
        raise TypeError("OpenVINO exported an invalid or empty compiled blob")
    (OUTPUT / "compiled.blob").write_bytes(blob)
    (OUTPUT / "report.json").write_text(
        json.dumps(
            {
                "host_correct": True,
                "cpu_latency_p50_ms": float(np.percentile(cpu_samples, 50)),
                "cpu_latency_p95_ms": float(np.percentile(cpu_samples, 95)),
                "cpu_warmup_count": 5,
                "cpu_iteration_count": 20,
                "evaluation_duration_seconds": evaluation_duration,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
