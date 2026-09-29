import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

from npu_agent.sim.amd_dataflow import Fifo, SimulationError, simulate
from npu_agent.sim.amd_host import HostKernel, build_host
from npu_agent.sim.amd_mlir_frontend import DataflowProgram, UnsupportedMLIR


pytestmark = pytest.mark.amd_sim


def test_acquire_window_and_broadcast_reclamation():
    fifo = Fifo({'shape': [2], 'dtype': 'float32', 'capacity': 2, 'producer': 'p', 'consumers': ['a', 'b']})
    first = fifo.acquire('p', 'produce', 1)[0]
    window = fifo.acquire('p', 'produce', 2)
    assert window[0] is first
    assert len(fifo.slots) == 2
    fifo.release('p', 'produce', 2)
    assert fifo.acquire('a', 'consume', 2)[0] is first
    fifo.release('a', 'consume', 2)
    assert fifo.acquire('p', 'produce', 1) is None
    fifo.acquire('b', 'consume', 2)
    fifo.release('b', 'consume', 1)
    assert fifo.acquire('p', 'produce', 1)[0].generation == 2
    assert not fifo.owns('b', first)
    with pytest.raises(SimulationError, match='excessive release'):
        fifo.release('b', 'consume', 2)


def program():
    def fifo(producer, consumer):
        return {'shape': [2], 'dtype': 'float32', 'capacity': 1, 'producer': producer, 'consumers': [consumer]}
    def worker(input_fifo, output_fifo):
        return [
            {'op': 'acquire', 'fifo': input_fifo, 'port': 'consume', 'count': 1, 'result': 0},
            {'op': 'access', 'window': 0, 'index': 0, 'result': 1},
            {'op': 'acquire', 'fifo': output_fifo, 'port': 'produce', 'count': 1, 'result': 2},
            {'op': 'access', 'window': 2, 'index': 0, 'result': 3},
            {'op': 'call', 'kernel': 'scale', 'args': [{'view': 1}, {'view': 3}]},
            {'op': 'release', 'fifo': input_fifo, 'port': 'consume', 'count': 1},
            {'op': 'release', 'fifo': output_fifo, 'port': 'produce', 'count': 1},
        ]
    tensor = {'kind': 'buffer', 'shape': [2], 'dtype': 'float32'}
    return DataflowProgram(device='npu2_1col', source_sha256='a' * 64, mlir_sha256='b' * 64,
        tiles={'shim': [0, 0], 'a': [0, 2], 'b': [0, 3]},
        fifos={'in': fifo('shim', 'a'), 'middle': fifo('a', 'b'), 'out': fifo('b', 'shim')},
        workers={'a': worker('in', 'middle'), 'b': worker('middle', 'out')}, kernels={'scale': [tensor, tensor]},
        arguments=[tensor, tensor], transfers={
            'input': {'fifo': 'in', 'argument': 0, 'offset': 0, 'length': 2, 'sizes': [2], 'strides': [1], 'issue_token': False},
            'output': {'fifo': 'out', 'argument': 1, 'offset': 0, 'length': 2, 'sizes': [2], 'strides': [1], 'issue_token': True}},
        runtime=[{'op': 'start', 'task': 'input'}, {'op': 'start', 'task': 'output'},
                 {'op': 'await', 'task': 'output'}, {'op': 'free', 'task': 'input'}, {'op': 'free', 'task': 'output'}])


CONTRACT = [{'name': 'x', 'direction': 'input', 'shape': [2], 'dtype': 'float32'},
            {'name': 'output', 'direction': 'output', 'shape': [2], 'dtype': 'float32'}]


def scale(symbol, args, readonly):
    args[1][:] = args[0] * 2


@pytest.mark.parametrize('seed', [0, 1, 7, 31])
def test_two_worker_transfers_and_changed_invocations(seed):
    for values in [[1., -2.], [3., 4.]]:
        x = np.array(values, dtype=np.float32)
        outputs, trace = simulate(program(), {'x': x}, CONTRACT, scale, seed=seed)
        np.testing.assert_array_equal(outputs['output'], x * 4)
        assert trace['logical_steps'] > 0


def test_transfer_and_release_mutations():
    p = program()
    p.transfers['input']['offset'] = 1
    with pytest.raises(SimulationError, match='bounds'):
        simulate(p, {'x': np.ones(2, np.float32)}, CONTRACT, scale)
    p = program()
    p.workers['a'].pop()
    with pytest.raises(SimulationError, match='simulation_deadlock'):
        simulate(p, {'x': np.ones(2, np.float32)}, CONTRACT, scale)


def test_aliased_call_arguments_are_explicitly_unsupported():
    p = program()
    p.workers['a'][4]['args'] = [{'view': 1}, {'view': 1}]
    with pytest.raises(UnsupportedMLIR, match='aliased kernel arguments'):
        simulate(p, {'x': np.ones(2, np.float32)}, CONTRACT, scale)


def test_use_after_release_and_incomplete_output():
    p = program()
    p.workers['a'][4:6] = list(reversed(p.workers['a'][4:6]))
    with pytest.raises(SimulationError, match='use after release'):
        simulate(p, {'x': np.ones(2, np.float32)}, CONTRACT, scale)


def test_exact_cpp_body_and_arithmetic_mutation(tmp_path):
    if not shutil.which('g++'):
        pytest.skip('host C++ compiler unavailable')
    source = tmp_path / 'kernel.cc'
    source.write_text('extern "C" void scale(float *x, float *y) { for(int i=0;i<2;++i) y[i] = x[i] * 2; }')
    p = program()
    for operator, expected in [('*', 4), ('+', 5)]:
        source.write_text(source.read_text().replace('* 2', f'{operator} 2'))
        binary = build_host(source, p.kernels, tmp_path, Path('playground/tools/host_compat'))
        kernel = HostKernel(binary, p.kernels, tmp_path / 'host-sanitizer.log')
        try:
            result, _ = simulate(p, {'x': np.ones(2, np.float32)}, CONTRACT, kernel)
            np.testing.assert_array_equal(result['output'], [expected, expected])
        finally:
            kernel.close()


def test_unsupported_intrinsic_cannot_be_a_noop(tmp_path):
    source = tmp_path / 'kernel.cc'
    source.write_text('void f() { aie::unsupported_intrinsic(); }')
    with pytest.raises(UnsupportedMLIR, match='unsupported_intrinsic'):
        build_host(source, {}, tmp_path, Path('playground/tools/host_compat'))


def test_portable_float16_conversion_matches_numpy(tmp_path):
    if not shutil.which('g++'):
        pytest.skip('host C++ compiler unavailable')
    source = tmp_path / 'kernel.cc'
    source.write_text('#include <npu_numeric.h>\nextern "C" void decode(npu::float16 *x, float *y) { for(int i=0;i<65536;++i) y[i]=float(x[i]); }\n')
    abi = {'decode': [{'kind': 'buffer', 'shape': [65536], 'dtype': 'float16'},
                      {'kind': 'buffer', 'shape': [65536], 'dtype': 'float32'}]}
    binary = build_host(source, abi, tmp_path, Path('playground/tools/host_compat'))
    kernel = HostKernel(binary, abi, tmp_path / 'host-sanitizer.log')
    try:
        x = np.arange(65536, dtype=np.uint16).view(np.float16)
        actual = np.zeros(65536, np.float32)
        kernel('decode', [x, actual], [True, False])
        np.testing.assert_array_equal(actual, x.astype(np.float32))
    finally:
        kernel.close()
    source.write_text('#include <npu_numeric.h>\nextern "C" void encode(float *x, npu::float16 *y) { for(int i=0;i<10000;++i) y[i]=x[i]; }\n')
    abi = {'encode': [{'kind': 'buffer', 'shape': [10000], 'dtype': 'float32'},
                      {'kind': 'buffer', 'shape': [10000], 'dtype': 'float16'}]}
    binary = build_host(source, abi, tmp_path, Path('playground/tools/host_compat'))
    kernel = HostKernel(binary, abi, tmp_path / 'host-sanitizer.log')
    try:
        x = np.random.default_rng(7).uniform(-70000, 70000, 10000).astype(np.float32)
        x[:8] = [0, -0., 65504, 65520, 2**-24, 2**-25, 1 + 2**-11, 1 + 3*2**-11]
        actual = np.zeros(10000, np.float16)
        kernel('encode', [x, actual], [True, False])
        with np.errstate(over='ignore'):
            np.testing.assert_array_equal(actual.view(np.uint16), x.astype(np.float16).view(np.uint16))
    finally:
        kernel.close()


def test_join_offsets_carry_real_data_and_are_bounds_checked():
    p = program()
    p.fifos['extra'] = {'shape': [2], 'dtype': 'float32', 'capacity': 1, 'producer': 'shim', 'consumers': ['join']}
    p.fifos['in']['consumers'] = ['join']
    p.fifos['joined'] = {'shape': [4], 'dtype': 'float32', 'capacity': 1, 'producer': 'join', 'consumers': ['a']}
    p.links = [{'inputs': ['in', 'extra'], 'outputs': ['joined'], 'src_offsets': [0, 2], 'dst_offsets': []}]
    p.workers['a'][0]['fifo'] = 'joined'
    p.workers['a'][5]['fifo'] = 'joined'
    p.transfers['extra'] = {**p.transfers['input'], 'fifo': 'extra'}
    p.runtime.insert(1, {'op': 'start', 'task': 'extra'})
    p.runtime.append({'op': 'free', 'task': 'extra'})
    def compute(symbol, args, readonly):
        args[1][:] = args[0][:2] + args[0][2:] if len(args[0]) == 4 else args[0]
    outputs, _ = simulate(p, {'x': np.array([2, 3], np.float32)}, CONTRACT, compute)
    np.testing.assert_array_equal(outputs['output'], [4, 6])
    p.links[0]['src_offsets'] = [0, 3]
    with pytest.raises(SimulationError, match='link bounds'):
        simulate(p, {'x': np.array([2, 3], np.float32)}, CONTRACT, compute)
