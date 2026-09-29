import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dev', default='npu2')
parser.add_argument('--emit-mlir', action='store_true')
parser.add_argument('--xclbin-path')
parser.add_argument('--insts-path')
args = parser.parse_args()
manifest = json.loads(Path('/contract/manifest.json').read_text())
tensors = manifest['tensors']
dtypes = {'float16': 'f16', 'float32': 'f32', 'float64': 'f64', 'int32': 'i32', 'int16': 'i16'}
types = ['memref<' + 'x'.join(map(str, t['shape'])) + 'x' + dtypes[t['dtype']] + '>' for t in tensors]
print('module { aie.device(npu2_1col) {')
print('%shim = aie.tile(0, 0)')
print('%core = aie.tile(0, 2)')
for i, tensor in enumerate(tensors):
    endpoints = '%shim, {%core}' if tensor['direction'] == 'input' else '%core, {%shim}'
    print(f'aie.objectfifo @f{i}({endpoints}, 1 : i32) : !aie.objectfifo<{types[i]}>')
print('func.func private @compute(' + ', '.join(types) + ') attributes {link_with = "kernel.o"}')
print('%worker = aie.core(%core) {')
for i, tensor in enumerate(tensors):
    port = 'Consume' if tensor['direction'] == 'input' else 'Produce'
    print(f'%a{i} = aie.objectfifo.acquire @f{i}({port}, 1) : !aie.objectfifosubview<{types[i]}>')
    print(f'%v{i} = aie.objectfifo.subview.access %a{i}[0] : !aie.objectfifosubview<{types[i]}> -> {types[i]}')
print('func.call @compute(' + ', '.join(f'%v{i}' for i in range(len(tensors))) + ') : (' + ', '.join(types) + ') -> ()')
for i, tensor in enumerate(tensors):
    port = 'Consume' if tensor['direction'] == 'input' else 'Produce'
    print(f'aie.objectfifo.release @f{i}({port}, 1)')
print('aie.end }')
print('aie.runtime_sequence(' + ', '.join(f'%arg{i}: {t}' for i, t in enumerate(types)) + ') {')
for i, tensor in enumerate(tensors):
    length = 1
    for size in tensor['shape']:
        length *= size
    print(f'%t{i} = aiex.dma_configure_task_for @f{i} {{')
    print(f'aie.dma_bd(%arg{i} : {types[i]} offset = 0 len = {length} sizes = [1, 1, 1, {length}] strides = [0, 0, 0, 1])')
    print('aie.end }' + (' {issue_token = true}' if tensor['direction'] == 'output' else ''))
    print(f'aiex.dma_start_task(%t{i})')
for i, tensor in enumerate(tensors):
    if tensor['direction'] == 'output':
        print(f'aiex.dma_await_task(%t{i})')
for i in range(len(tensors)):
    print(f'aiex.dma_free_task(%t{i})')
print('} } }')
