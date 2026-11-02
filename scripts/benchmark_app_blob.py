"""Run OpenVINO's Python benchmark_app on a compiled .blob (Windows-native Python, same arguments as benchmark_app).

benchmark_app 2026.3.1 cannot load a .blob as shipped, for two reasons this file works around without touching its
measurement loop:
- it leaves -ip, -op and -iop at None, and its compiled-model check treats anything but '' as a requested precision,
  so it rejects every blob; their defaults become '' (what the check and the preprocessing code mean by "not set");
- it passes the blob's path to Core.import_model, which in this release accepts only bytes or io.BytesIO; a path is
  read into io.BytesIO first.
"""
import io
import os
import sys

import openvino as ov
from openvino.tools.benchmark import main as benchmark
from openvino.tools.benchmark.parameters import parse_args

import_model = ov.Core.import_model


def import_from_path(core, model, *args, **kwargs):
    if isinstance(model, (str, os.PathLike)):
        with open(model, 'rb') as stream:
            model = io.BytesIO(stream.read())
    return import_model(core, model, *args, **kwargs)


def parser():
    arguments = parse_args()
    arguments.set_defaults(input_precision='', output_precision='', input_output_precision='')
    return arguments


ov.Core.import_model = import_from_path
benchmark.parse_args = parser

if __name__ == '__main__':
    sys.exit(benchmark.main())
