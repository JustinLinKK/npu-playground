"""Candidate process: public contract in, serialized graph out. No test tensors."""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import openvino as ov


def main() -> None:
    spec = importlib.util.spec_from_file_location('candidate_model', '/candidate/model.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('unable to load model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads(Path('/contract/manifest.json').read_text())
    model = module.build_model(manifest) if inspect.signature(module.build_model).parameters else module.build_model()
    if not isinstance(model, ov.Model):
        raise TypeError('build_model must return openvino.Model')
    ov.save_model(model, '/output/model.xml', compress_to_fp16=False)


if __name__ == '__main__':
    main()
