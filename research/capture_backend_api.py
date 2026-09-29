"""Capture task-independent signatures from the exact pilot compiler images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

APIS = {
    'amd_xdna2_npu2': {
        'aie.dialects.aie': ['device', 'Device', 'tile', 'core', 'Core', 'buffer', 'external_func', 'objectfifo',
                             'object_fifo', 'object_fifo_link', 'runtime_sequence'],
        'aie.dialects.aiex': ['runtime_sequence', 'npu_dma_memcpy_nd', 'dma_wait'],
        'aie.extras.dialects.arith': ['constant'],
        'aie.extras.types': ['index', 'i32', 'f32', 'memref'],
    },
    'intel_npu_4000': {
        'openvino': ['Model', 'save_model'],
        'openvino.opset13': ['parameter', 'constant', 'result', 'add', 'subtract', 'multiply', 'divide',
                            'matmul', 'transpose', 'reduce_sum', 'reduce_mean', 'sqrt', 'exp', 'maximum',
                            'reshape', 'broadcast', 'convert'],
    },
}

PROBE = '''import hashlib, importlib, inspect, json, pathlib, re
apis = json.loads(INPUT)
result = {}
for name, symbols in apis.items():
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__)
    entries = {}
    for symbol in symbols:
        value = getattr(module, symbol, None)
        if value is None:
            entries[symbol] = {"available": False}
            continue
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError):
            signature = None
        entries[symbol] = {"available": True, "signature": re.sub(r"0x[0-9a-fA-F]+", "<address>", signature) if signature else None}
        if signature is None or "*args" in signature:
            doc = value.__init__.__doc__ if inspect.isclass(value) else value.__doc__
            entries[symbol]["documented_signatures"] = [line.strip() for line in (doc or "").splitlines()
                                                       if re.match(r"(?:[0-9]+[.] )?[A-Za-z_][A-Za-z_0-9.]*[(]", line.strip())]
    result[name] = {"path": str(path), "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "symbols": entries}
print(json.dumps(result))
'''


def capture(campaign: Path, output: Path) -> None:
    if output.exists():
        raise ValueError('use a new output directory to preserve API guidance provenance')
    preflight = campaign / 'preflight.json'
    raw = preflight.read_bytes()
    state = json.loads(raw)
    if state['status'] != 'completed':
        raise ValueError('API guidance must bind to a completed compiler/provider preflight')
    output.mkdir(parents=True)
    report = {'protocol': 'backend-api-reference-v1', 'preflight_sha256': hashlib.sha256(raw).hexdigest(),
              'capture_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Task-independent API signatures only; no kernel implementations, test inputs, or oracle outputs.',
              'status': 'running', 'targets': {}}
    (output / 'capture_backend_api.py').write_bytes(Path(__file__).read_bytes())
    for target, apis in APIS.items():
        started = time.perf_counter()
        identity = state['identity']['compilers'][target]
        script = PROBE.replace('INPUT', repr(json.dumps(apis)))
        command = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '1', '--memory', '1g',
                   '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                   '--tmpfs', '/tmp:rw,nosuid,size=64m', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                   '-i', identity['image_id'], 'python', '-']
        result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=120)
        row = {'image_id': identity['image_id'], 'duration_seconds': time.perf_counter() - started,
               'returncode': result.returncode, 'stderr': result.stderr}
        report['targets'][target] = row
        if result.returncode:
            report['status'] = 'blocked'
            (output / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')
            raise RuntimeError(f'{target} signature capture failed; see reference.json')
        row['modules'] = json.loads(result.stdout)
        (output / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'completed'
    (output / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    capture(args.campaign, args.output)
