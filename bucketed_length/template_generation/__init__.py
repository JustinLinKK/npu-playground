"""A fixed AMD transport scaffold; generation fills only a compute function body."""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from npu_agent.models import Backend, CodeBundle, GeneratedFile
from npu_agent.validation import check_contract, sha256


ROOT = Path(__file__).resolve().parents[2]
SCAFFOLD = ROOT / 'tests/fixtures/backends/amd_single_worker.py.template'


class MicroKernel(BaseModel):
    model_config = ConfigDict(extra='forbid')

    body: str = Field(min_length=1, max_length=24000)


def schedule(manifest, target):
    check_contract(manifest)
    if target.backend != Backend.AMD_XDNA2 or target.hardware != 'npu2':
        raise ValueError('the selected skeleton supports AMD npu2 only')
    tensors = manifest.tensors
    inputs = [t for t in tensors if t.direction == 'input']
    outputs = [t for t in tensors if t.direction == 'output']
    if (any(t.dtype != 'float32' for t in tensors) or len(outputs) != 1
            or tensors != inputs + outputs):
        raise ValueError('skeleton requires float32 inputs followed by one output')
    op = manifest.oracle.operation
    shapes = [t.shape for t in tensors]
    if op == 'vector_add' and len(inputs) == 2 and len(shapes[0]) == 1 and len(set(shapes)) == 1:
        kind = 'elementwise_add'
    elif (op == 'reduce_sum' and len(inputs) == 1 and len(shapes[0]) == 1 and shapes[1] == (1,)
          and manifest.oracle.parameters.get('keepdims') is True
          and manifest.oracle.parameters.get('axis', 0) in (0, -1)):
        kind = 'reduction_sum'
    elif (op == 'matmul' and len(inputs) == 2 and all(len(s) == 2 for s in shapes)
          and shapes[1] == (shapes[0][1], 1) and shapes[2] == (shapes[0][0], 1)):
        kind = 'gemv'
    else:
        raise ValueError('no legal elementwise, reduction or GEMV skeleton for this contract')
    import math

    counts = [math.prod(shape) for shape in shapes]
    if sum(counts) * 4 > 16384:
        raise ValueError('single-tile skeleton exceeds its conservative 16 KiB buffer budget')
    return {'template': kind, 'shapes': [list(s) for s in shapes], 'buffer_elements': counts,
            'fifo_depth': 1, 'workers': 1, 'launches': 1, 'lane_hint': 16,
            'tail_required': counts[0] % 16 != 0, 'scaffold_sha256': sha256(SCAFFOLD),
            'signature': 'void compute(' + ', '.join(f'float *x{i}' for i in range(len(inputs))) + ', float *out)'}


def instantiate(manifest, target, microkernel):
    plan = schedule(manifest, target)
    # This is a structural guard, not a C++ sandbox; compilation retains the container boundary.
    import re

    code = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', '', microkernel.body, flags=re.S)
    depth = 0
    for char in code:
        depth += (char == '{') - (char == '}')
        if depth < 0:
            raise ValueError('microkernel closes its enclosing function')
    if depth or '#' in code:
        raise ValueError('microkernel must be a balanced body without preprocessor directives')
    kernel = '#include <aie_api/aie.hpp>\n#include <cmath>\nextern "C" ' + plan['signature'] + ' {\n' + microkernel.body + '\n}\n'
    return CodeBundle(backend=target.backend, files=[GeneratedFile(relative_path='design.py', content=SCAFFOLD.read_text()),
        GeneratedFile(relative_path='kernel.cc', content=kernel)]), plan
