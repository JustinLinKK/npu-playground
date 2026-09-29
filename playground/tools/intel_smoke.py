"""Built-in numerical and offline compile probe; no NPU inference claim."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import openvino as ov

parser = argparse.ArgumentParser()
parser.add_argument('--platform', default='4000')
parser.add_argument('--output', type=Path, default=Path('/output'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
left = ov.opset13.parameter([1, 16], ov.Type.f32, name='left')
right = ov.opset13.parameter([1, 16], ov.Type.f32, name='right')
model = ov.Model([ov.opset13.add(left, right)], [left, right], 'npu_agent_smoke')
core = ov.Core()
started = time.perf_counter()
output = core.compile_model(model, 'CPU')([np.ones((1, 16), np.float32), np.full((1, 16), 2, np.float32)])
assert np.array_equal(next(iter(output.values())), np.full((1, 16), 3, np.float32))
report = {'host_execution': {'status': 'passed', 'correct': True, 'duration_seconds': time.perf_counter() - started},
          'target_execution': {'status': 'blocked', 'correct': None, 'reason_code': 'NO_VALIDATED_TARGET_EXECUTOR'}}
(args.output / 'report.json').write_text(json.dumps(report, indent=2))
started = time.perf_counter()
try:
    compiled = core.compile_model(model, 'NPU', {'NPU_COMPILER_TYPE': 'PLUGIN', 'NPU_PLATFORM': args.platform})
    blob = compiled.export_model()
    blob = blob.getvalue() if hasattr(blob, 'getvalue') else blob
    if not isinstance(blob, bytes) or not blob:
        raise ValueError('Intel NPU compiler returned an invalid or empty blob')
    (args.output / 'compiled.blob').write_bytes(blob)
    report['target_compile'] = {'status': 'passed', 'platform': args.platform}
except Exception as exc:
    report['target_compile'] = {'status': 'failed', 'message': str(exc)}
    raise
finally:
    report['target_compile']['duration_seconds'] = time.perf_counter() - started
    temporary = args.output / 'report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output / 'report.json')
