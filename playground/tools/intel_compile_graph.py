from __future__ import annotations

import json
import time
from pathlib import Path

import openvino as ov

from intel_execute_graph import check_ports, read_model
from npu_agent.models import KernelManifest
from npu_agent.validation import sha256, write_json


def main() -> None:
    started = time.perf_counter()
    manifest = KernelManifest.model_validate_json(Path('/contract/manifest.json').read_text())
    profile = json.loads(Path('/contract/target.json').read_text())
    core, model = read_model()
    check_ports(model, manifest)
    config = dict(profile['compiler_properties'])
    if config.get('NPU_PLATFORM') != profile['hardware'] or config.get('NPU_COMPILER_TYPE') != 'PLUGIN':
        raise ValueError('target compiler properties disagree with selected platform')
    compiled = core.compile_model(model, 'NPU', config)
    blob = compiled.export_model()
    blob = blob.getvalue() if hasattr(blob, 'getvalue') else blob
    if not isinstance(blob, bytes) or not blob:
        raise ValueError('OpenVINO exported invalid or empty blob')
    Path('/output/compiled.blob').write_bytes(blob)
    write_json(Path('/output/compile.json'), {'config': config, 'openvino_version': ov.__version__,
        'model_sha256': sha256(Path('/graph/model.xml')), 'weights_sha256': sha256(Path('/graph/model.bin')),
        'blob_sha256': sha256(Path('/output/compiled.blob')), 'duration_seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
