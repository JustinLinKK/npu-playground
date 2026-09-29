import json
import sys
from pathlib import Path

from npu_agent.sim.amd_mlir_frontend import UnsupportedMLIR, parse_mlir
from npu_agent.validation import sha256, write_json

try:
    program = parse_mlir(Path('/design/design.mlir').read_text(), sha256(Path('/candidate/kernel.cc')),
                         sha256(Path('/design/design.mlir')))
    write_json(Path('/output/program.json'), program.model_dump(mode='json'))
    write_json(Path('/output/abi.json'), program.kernels)
except UnsupportedMLIR as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(3)
