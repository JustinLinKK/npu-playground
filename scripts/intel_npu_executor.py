"""Execute an Intel NPU ExecutionRequest on a local NPU through OpenVINO 2026.3.1.

Under WSL2 this runs in Windows-native Python, launched by scripts/wsl_npu_bridge.py.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'playground' / 'tools')]

from npu_agent.executors import decode_tensor, encode_tensor
from npu_agent.models import ExecutionRequest, ExecutionResult
from npu_agent.validation import json_digest


def npu_core():
    import openvino as ov

    core = ov.Core()
    if 'NPU' not in core.available_devices:
        raise RuntimeError(f'OpenVINO reports no NPU device: {core.available_devices}')
    return ov, core


def execution_devices(compiled) -> list[str]:
    # The NPU plugin reports a plain string ("NPU"); other plugins may report a list.
    value = compiled.get_property('EXECUTION_DEVICES')
    return [value] if isinstance(value, str) else [str(item) for item in value]


def run(request: ExecutionRequest) -> ExecutionResult:
    target = request.target
    if target.vendor != 'intel' or target.compiler_properties.get('NPU_PLATFORM') != target.hardware:
        raise ValueError('local executor requires an Intel NPU target compiled for its own platform')
    if request.manifest_sha256 != json_digest(request.manifest.model_dump(mode='json')):
        raise ValueError('manifest hash mismatch')
    if set(request.artifacts) != {'compiled.blob'}:
        raise ValueError('Intel NPU blob required')
    blob = Path(request.artifacts['compiled.blob']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != request.artifact_sha256['compiled.blob']:
        raise ValueError('artifact hash mismatch: compiled.blob')

    ov, core = npu_core()
    from intel_execute_graph import check_ports

    architecture = str(core.get_property('NPU', 'DEVICE_ARCHITECTURE'))
    if architecture != target.hardware:
        raise ValueError(f'NPU architecture {architecture} does not match target platform {target.hardware}')
    # Name the device explicitly; AUTO/HETERO could silently fall back to CPU.
    compiled = core.import_model(io.BytesIO(blob), 'NPU')
    devices = execution_devices(compiled)
    if devices != ['NPU']:
        raise RuntimeError(f'blob did not load on the NPU alone: {devices}')
    inputs, outputs = check_ports(compiled, request.manifest)
    infer = compiled.create_infer_request()
    feeds, results = {}, {}
    for case_id, encoded in request.cases.items():
        feeds[case_id] = {name: np.array(decode_tensor(encoded[name])) for name in inputs}
        infer.infer(feeds[case_id])
        results[case_id] = {name: encode_tensor(np.array(infer.get_tensor(name).data, copy=True)) for name in outputs}
    device = str(core.get_property('NPU', 'FULL_DEVICE_NAME'))
    samples, provenance = [], None
    if request.measurement_count:
        feed = next(iter(feeds.values()))
        for _ in range(request.warmup_count):
            infer.infer(feed)
        for _ in range(request.measurement_count):
            started = time.perf_counter()
            infer.infer(feed)
            samples.append((time.perf_counter() - started) * 1000)
        provenance = (f'perf_counter around synchronous InferRequest.infer on {device} (NPU {architecture}), '
                      f'first case, after {request.warmup_count} warmup runs')
    return ExecutionResult(run_id=request.run_id, target_id=target.id, hardware=target.hardware,
        executor_kind='physical_npu', executed_artifact_sha256=request.artifact_sha256, outputs=results,
        logs=f'{device} executed {len(results)} cases; EXECUTION_DEVICES={devices}',
        runtime_identity=f'openvino/{ov.get_version()};npu-arch/{architecture};'
                         f'npu-driver/{core.get_property("NPU", "NPU_DRIVER_VERSION")}',
        compatibility_checked=True, timing_samples_ms=samples, warmup_count=request.warmup_count,
        measurement_count=request.measurement_count, timing_scope=request.timing_scope, timing_provenance=provenance)


def probe() -> dict:
    """Report the NPU identity and run a graph whose float16 result is exact."""
    ov, core = npu_core()
    from openvino import opset13 as ops

    report = {'openvino': ov.get_version(), 'devices': list(core.available_devices)}
    for name in ('FULL_DEVICE_NAME', 'DEVICE_ARCHITECTURE', 'NPU_DRIVER_VERSION', 'OPTIMIZATION_CAPABILITIES'):
        report[name] = str(core.get_property('NPU', name))
    x, y = ops.parameter([16], np.float32, name='x'), ops.parameter([16], np.float32, name='y')
    compiled = core.compile_model(ov.Model([ops.add(x, y)], [x, y]), 'NPU')
    a = np.arange(16, dtype=np.float32)
    actual = np.asarray(compiled([a, 2 * a])[compiled.output(0)])
    report['execution_devices'] = execution_devices(compiled)
    report['self_test'] = 'passed' if np.array_equal(actual, 3 * a) else 'failed'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true', help='print the NPU identity and run an exact self-test')
    if parser.parse_args().probe:
        report = probe()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report['self_test'] == 'passed' else 1)
    # Bytes in and out: Windows text streams would apply a locale code page and CRLF translation.
    request = ExecutionRequest.model_validate_json(sys.stdin.buffer.read())
    with redirect_stdout(sys.stderr):
        result = run(request)
    sys.stdout.buffer.write(result.model_dump_json().encode())
