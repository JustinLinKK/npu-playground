from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class UnsupportedMLIR(ValueError):
    pass


class DataflowProgram(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal['1.0'] = '1.0'
    device: str
    source_sha256: str
    mlir_sha256: str
    tiles: dict[str, list[int]]
    fifos: dict[str, dict[str, Any]]
    kernels: dict[str, list[dict[str, Any]]]
    workers: dict[str, list[dict[str, Any]]]
    transfers: dict[str, dict[str, Any]]
    runtime: list[dict[str, Any]]
    arguments: list[dict[str, Any]]
    links: list[dict[str, Any]] = Field(default_factory=list)


def parse_mlir(text: str, source_sha256: str, mlir_sha256: str, hardware: str = 'npu2') -> DataflowProgram:
    # Load registered dialects; no text/substrings are used as target verification.
    from aie.ir import Context, Module, MemRefType, FunctionType, Type, Value
    from aie.dialects import aie, aiex, arith, func, scf
    from aie.dialects.aie import AIEDevice, ObjectFifoType

    def key(value):
        # IRON downcasts index values to Scalar, whose == emits an arith op.
        # Native Value equality compares identity without constructing IR.
        class Identity:
            def __init__(self, value):
                self.value = value
            def __hash__(self):
                return hash(self.value)
            def __eq__(self, other):
                return Value.__eq__(self.value, other.value)
        return Identity(value)

    def attr(op, name, default=None):
        return op.attributes[name] if name in op.attributes else default

    def val(op, name, default=None):
        value = attr(op, name)
        return value.value if value is not None else default

    def children(op):
        return [o.operation for r in op.regions for b in r.blocks for o in b.operations]

    def abi(ty):
        if isinstance(ty, MemRefType):
            ty = MemRefType(ty)
            if ty.memory_space is not None:
                raise UnsupportedMLIR('explicit memref memory space')
            shape = list(ty.shape)
            strides, offset = ty.get_strides_and_offset()
            expected = 1
            for size, stride in zip(reversed(shape), reversed(strides)):
                if size <= 0 or stride != expected or offset != 0:
                    raise UnsupportedMLIR('noncontiguous/dynamic memref ABI')
                expected *= size
            return {'kind': 'buffer', 'shape': shape, 'dtype': dtype(ty.element_type)}
        return {'kind': 'scalar', 'dtype': dtype(ty)}

    def dtype(ty):
        names = {'f16': 'float16', 'f32': 'float32', 'f64': 'float64',
                 'i8': 'int8', 'i16': 'int16', 'i32': 'int32', 'i64': 'int64', 'index': 'int64'}
        if str(ty) not in names:
            raise UnsupportedMLIR(f'unsupported ABI type: {ty}')
        return names[str(ty)]

    with Context():
        module = Module.parse(text)
        if not module.operation.verify():
            raise ValueError('MLIR verification failed')
        allowed_attributes = {
            'builtin.module': set(), 'aie.device': {'device', 'sym_name'}, 'aie.tile': {'col', 'row'},
            'aie.objectfifo': {'sym_name', 'elemNumber', 'elemType', 'dimensionsFromStreamPerConsumer',
                              'dimensionsToStream', 'disable_synchronization', 'plio', 'via_DMA'},
            'aie.objectfifo.link': {'fifoIns', 'fifoOuts', 'src_offsets', 'dst_offsets'},
            'func.func': {'function_type', 'link_with', 'sym_name', 'sym_visibility'},
            'aie.core': set(), 'arith.constant': {'value'}, 'scf.for': set(), 'scf.yield': set(),
            'aie.end': set(), 'aie.objectfifo.acquire': {'objFifo_name', 'port', 'size'},
            'aie.objectfifo.release': {'objFifo_name', 'port', 'size'},
            'aie.objectfifo.subview.access': {'index'}, 'func.call': {'callee'},
            'aie.runtime_sequence': {'sym_name'}, 'aiex.dma_configure_task_for': {'alloc', 'issue_token'},
            'aie.dma_bd': {'operandSegmentSizes', 'static_len', 'static_offset', 'static_sizes', 'static_strides'},
            'aiex.dma_start_task': set(), 'aiex.dma_await_task': set(), 'aiex.dma_free_task': set(),
        }
        def check_operations(op):
            if op.name not in allowed_attributes:
                raise UnsupportedMLIR(f'unsupported MLIR operation: {op.name}')
            unknown = set(op.attributes) - allowed_attributes[op.name]
            if unknown:
                raise UnsupportedMLIR(f'unsupported {op.name} attributes: {sorted(unknown)}')
            for child in children(op):
                check_operations(child)
        check_operations(module.operation)
        devices = children(module.operation)
        if len(devices) != 1 or devices[0].name != 'aie.device':
            raise UnsupportedMLIR('exactly one aie.device is required')
        device = devices[0]
        selected = AIEDevice(val(device, 'device'))
        if hardware != 'npu2' or not selected.name.startswith('npu2'):
            raise ValueError(f'target_mismatch: expected {hardware}, got {selected.name}')
        program = DataflowProgram(device=selected.name, source_sha256=source_sha256, mlir_sha256=mlir_sha256,
            tiles={}, fifos={}, kernels={}, workers={}, transfers={}, runtime=[], arguments=[])
        values = {}
        externals = {}
        cores = []
        sequence = None
        for op in children(device):
            if op.name == 'aie.tile':
                name = f'tile_{val(op, "col")}_{val(op, "row")}'
                program.tiles[name] = [val(op, 'col'), val(op, 'row')]
                values[key(op.results[0])] = name
            elif op.name == 'aie.objectfifo':
                name = val(op, 'sym_name')
                for field in ('disable_synchronization', 'plio', 'via_DMA'):
                    if val(op, field, False):
                        raise UnsupportedMLIR(f'objectfifo.{field}')
                for field, empty in [('dimensionsToStream', '#aie<bd_dim_layout_array[]>'),
                                      ('dimensionsFromStreamPerConsumer', '#aie<bd_dim_layout_array_array[' + ', '.join('[]' for _ in list(op.operands)[1:]) + ']>')]:
                    if str(attr(op, field)) != empty:
                        raise UnsupportedMLIR(f'objectfifo.{field}')
                capacity = val(op, 'elemNumber')
                if not isinstance(capacity, int) or capacity <= 0:
                    raise UnsupportedMLIR('objectfifo requires uniform positive depth')
                fifo_type = ObjectFifoType(attr(op, 'elemType').value)
                # The pinned binding exposes no element_type accessor for ObjectFifoType.
                match = re.fullmatch(r'!aie.objectfifo<(memref<[^<>]+>)>', str(fifo_type))
                if not match:
                    raise UnsupportedMLIR('unsupported ObjectFifo element type')
                spec = abi(Type.parse(match[1]))
                endpoints = [values[key(v)] for v in op.operands]
                program.fifos[name] = {**spec, 'capacity': capacity, 'producer': endpoints[0], 'consumers': endpoints[1:]}
            elif op.name == 'aie.objectfifo.link':
                program.links.append({'inputs': [a.value for a in attr(op, 'fifoIns')],
                    'outputs': [a.value for a in attr(op, 'fifoOuts')],
                    'src_offsets': [a.value for a in attr(op, 'src_offsets')],
                    'dst_offsets': [a.value for a in attr(op, 'dst_offsets')]})
            elif op.name == 'func.func':
                if children(op):
                    raise UnsupportedMLIR('inline MLIR function bodies')
                name = val(op, 'sym_name')
                typ = FunctionType(attr(op, 'function_type').value)
                if typ.results:
                    raise UnsupportedMLIR('non-void external kernel')
                program.kernels[name] = [abi(t) for t in typ.inputs]
                externals[name] = val(op, 'link_with')
            elif op.name == 'aie.core':
                cores.append(op)
            elif op.name == 'aie.runtime_sequence':
                if sequence is not None:
                    raise UnsupportedMLIR('multiple runtime sequences')
                sequence = op
            elif op.name != 'aie.end':
                raise UnsupportedMLIR(f'unsupported device operation: {op.name}')
        def worker_ops(ops, env, result):
            for op in ops:
                if len(result) > 100000:
                    raise UnsupportedMLIR('finite worker instruction limit')
                name = op.name
                if name == 'arith.constant':
                    env[key(op.results[0])] = {'constant': val(op, 'value')}
                elif name == 'scf.for':
                    bounds = [env[key(v)]['constant'] for v in op.operands]
                    if len(bounds) != 3 or bounds[2] <= 0 or len(range(*bounds)) > 10000:
                        raise UnsupportedMLIR('non-finite or excessive worker loop')
                    block = op.regions[0].blocks[0]
                    for index in range(*bounds):
                        inner = dict(env)
                        inner[key(block.arguments[0])] = {'constant': index}
                        worker_ops(children(op), inner, result)
                elif name in ('aie.objectfifo.acquire', 'aie.objectfifo.release'):
                    port = 'produce' if val(op, 'port') == 0 else 'consume'
                    instruction = {'op': name.rsplit('.', 1)[1], 'fifo': val(op, 'objFifo_name'),
                                   'port': port, 'count': val(op, 'size')}
                    if name.endswith('acquire'):
                        env[key(op.results[0])] = {'window': len(result)}
                        instruction['result'] = len(result)
                    result.append(instruction)
                elif name == 'aie.objectfifo.subview.access':
                    instruction = {'op': 'access', 'window': env[key(op.operands[0])]['window'],
                                   'index': val(op, 'index'), 'result': len(result)}
                    env[key(op.results[0])] = {'view': len(result)}
                    result.append(instruction)
                elif name == 'func.call':
                    symbol = val(op, 'callee')
                    if symbol not in program.kernels:
                        raise UnsupportedMLIR(f'unknown external kernel: {symbol}')
                    result.append({'op': 'call', 'kernel': symbol, 'args': [env[key(v)] for v in op.operands]})
                elif name not in ('scf.yield', 'aie.end'):
                    raise UnsupportedMLIR(f'unsupported worker operation: {name}')
        for core in cores:
            tile = values[key(core.operands[0])]
            if tile in program.workers:
                raise UnsupportedMLIR('multiple workers on one tile')
            program.workers[tile] = []
            worker_ops(children(core), {}, program.workers[tile])
        if sequence is None:
            raise UnsupportedMLIR('missing runtime sequence')
        arguments = list(sequence.regions[0].blocks[0].arguments)
        argument_keys = [key(v) for v in arguments]
        program.arguments = [abi(v.type) for v in arguments]
        tasks = {}
        for op in children(sequence):
            name = op.name
            if name == 'aiex.dma_configure_task_for':
                descriptors = [o for o in children(op) if o.name != 'aie.end']
                if len(descriptors) != 1 or descriptors[0].name != 'aie.dma_bd':
                    raise UnsupportedMLIR('DMA requires one static descriptor')
                bd = descriptors[0]
                if len(bd.operands) != 1 or key(bd.operands[0]) not in argument_keys:
                    raise UnsupportedMLIR('dynamic DMA descriptor')
                task = f'task_{len(tasks)}'
                tasks[key(op.results[0])] = task
                program.transfers[task] = {'fifo': val(op, 'alloc'), 'argument': argument_keys.index(key(bd.operands[0])),
                    'offset': val(bd, 'static_offset'), 'length': val(bd, 'static_len'),
                    'sizes': list(attr(bd, 'static_sizes')), 'strides': list(attr(bd, 'static_strides')),
                    'issue_token': val(op, 'issue_token', False)}
            elif name in ('aiex.dma_start_task', 'aiex.dma_await_task', 'aiex.dma_free_task'):
                for operand in op.operands:
                    program.runtime.append({'op': name.split('_')[-2], 'task': tasks[key(operand)]})
            elif name != 'aie.end':
                raise UnsupportedMLIR(f'unsupported runtime operation: {name}')
        return program
