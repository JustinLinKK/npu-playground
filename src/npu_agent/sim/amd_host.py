from __future__ import annotations

import re
import subprocess
import shutil
from pathlib import Path

import numpy as np

from .amd_mlir_frontend import UnsupportedMLIR
from .amd_dataflow import SimulationError


CPP_TYPES = {'float16': 'npu::float16', 'float32': 'float', 'float64': 'double',
             'int8': 'int8_t', 'int16': 'int16_t', 'int32': 'int32_t', 'int64': 'int64_t'}


def build_host(source: Path, kernels: dict, destination: Path, includes: Path) -> Path:
    text = source.read_text()
    directives = re.findall(r'^\s*#\s*(\w+)([^\n]*)', text, re.MULTILINE)
    for directive, rest in directives:
        if directive != 'include' or rest.strip() not in ('<stdint.h>', '<cstdint>', '<stddef.h>', '<cstddef>',
                '<math.h>', '<cmath>', '<npu_numeric.h>', '<aie_api/aie.hpp>', '"aie_api/aie.hpp"'):
            raise UnsupportedMLIR(f'unsupported source preprocessor directive: #{directive}{rest}')
    if re.search(r'\b(asm|__asm__|__builtin_cpu_supports|sizeof|alignof|typeid)\b', text):
        raise UnsupportedMLIR('architecture-dependent source construct')
    api_symbols = set(re.findall(r'aie::([A-Za-z_][A-Za-z0-9_]*)', text))
    unknown = api_symbols - {'vector', 'load_v', 'store_v', 'add', 'reduce_add'}
    if unknown:
        raise UnsupportedMLIR(f'unsupported intrinsic: {sorted(unknown)}')
    declarations, cases = [], []
    for index, (name, args) in enumerate(kernels.items()):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError('invalid kernel symbol')
        types = [CPP_TYPES[a['dtype']] + ('*' if a['kind'] == 'buffer' else '') for a in args]
        for arg_index, abi in enumerate(args):
            actual = f'std::tuple_element_t<{arg_index}, typename Signature<decltype(&{name})>::Arguments>'
            expected = CPP_TYPES[abi['dtype']]
            declarations.append(f'static_assert(std::is_pointer_v<{actual}> == {"true" if abi["kind"] == "buffer" else "false"}, "ABI pointer mismatch");')
            declarations.append(f'static_assert(std::is_same_v<std::remove_cv_t<std::remove_pointer_t<{actual}>>, {expected}>, "ABI dtype or signedness mismatch");')
        sizes = [int(np.prod(a['shape'])) * np.dtype(a['dtype']).itemsize if a['kind'] == 'buffer'
                 else np.dtype(a['dtype']).itemsize for a in args]
        call_args = [f'reinterpret_cast<{typ}>(buffers[{i}].data)' if a['kind'] == 'buffer'
                     else f'*reinterpret_cast<{typ}*>(buffers[{i}].data)' for i, (typ, a) in enumerate(zip(types, args))]
        cases.append(f'case {index}: {{ std::vector<Buffer> buffers; buffers.reserve({len(args)}); ' +
                     ''.join(f'buffers.emplace_back({size});' for size in sizes) +
                     f'{name}({", ".join(call_args)}); for (auto &b : buffers) b.send(); break; }}')
    shutil.copyfile(source, destination / 'kernel_source.cc')
    wrapper = destination / 'host_wrapper.cc'
    wrapper.write_text('''#include <npu_numeric.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sanitizer/asan_interface.h>
#include <tuple>
#include <type_traits>
#include "kernel_source.cc"
template <typename> struct Signature;
template <typename... Args> struct Signature<void (*)(Args...)> { using Arguments = std::tuple<Args...>; };
static_assert(sizeof(npu::float16) == 2);
struct Buffer {
    unsigned char *base, *data;
    size_t size;
    explicit Buffer(size_t n): size(n) {
        if (posix_memalign(reinterpret_cast<void **>(&base), 64, n + 128)) std::abort();
        data = base + 64;
        std::memset(base, 0xA5, n + 128);
        if (std::fread(data, 1, n, stdin) != n) std::exit(4);
        __asan_poison_memory_region(base, 64);
        __asan_poison_memory_region(data + n, 64);
    }
    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&other): base(other.base), data(other.data), size(other.size) { other.base = nullptr; }
    void send() {
        if (std::fwrite(data, 1, size, stdout) != size) std::exit(5);
    }
    ~Buffer() { if (base) { __asan_unpoison_memory_region(base, size + 128); std::free(base); } }
};
''' + '\n'.join(declarations) + '''
int main() {
    uint32_t symbol;
    while (std::fread(&symbol, sizeof(symbol), 1, stdin) == 1) {
        switch (symbol) {
''' + '\n'.join(cases) + '''
        default: return 6;
        }
        std::fflush(stdout);
    }
    return 0;
}
''')
    binary = destination / 'host-kernel'
    command = ['g++', '-std=c++20', '-O1', '-g', '-fno-omit-frame-pointer', '-ffp-contract=off',
               '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-no-pie',
               '-I', str(includes), str(wrapper), '-o', str(binary)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    (destination / 'host-build.log').write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise ValueError(f'host source_compile: {completed.stderr[-4000:]}')
    return binary


class HostKernel:
    def __init__(self, binary: Path, kernels: dict, log: Path):
        self.kernels = kernels
        self.log = log.open('wb')
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)

    def __call__(self, symbol, args, readonly):
        import select
        import struct
        if len(args) != len(self.kernels[symbol]):
            raise SimulationError('host ABI argument count mismatch')
        arrays = []
        for value, abi in zip(args, self.kernels[symbol], strict=True):
            array = np.asarray(value, dtype=abi['dtype']) if abi['kind'] == 'scalar' else np.asarray(value)
            shape = tuple(abi.get('shape', ()))
            if array.shape != shape or array.dtype != np.dtype(abi['dtype']):
                raise SimulationError('host ABI shape/dtype mismatch')
            arrays.append(array)
        raw = [a.tobytes() for a in arrays]
        try:
            self.process.stdin.write(struct.pack('<I', list(self.kernels).index(symbol)) + b''.join(raw))
            self.process.stdin.flush()
            for index, array in enumerate(arrays):
                if not select.select([self.process.stdout], [], [], 20)[0]:
                    raise SimulationError('host kernel timeout')
                received = b''
                while len(received) < array.nbytes:
                    if not select.select([self.process.stdout], [], [], 20)[0]:
                        raise SimulationError('host kernel timeout')
                    chunk = self.process.stdout.read(array.nbytes - len(received))
                    if not chunk:
                        break
                    received += chunk
                if len(received) != array.nbytes:
                    raise SimulationError('host kernel failed; see host-sanitizer.log')
                if readonly[index] and received != raw[index]:
                    raise SimulationError('kernel mutated immutable input')
                if not readonly[index]:
                    array[...] = np.frombuffer(received, dtype=array.dtype).reshape(array.shape)
        except BrokenPipeError as exc:
            raise SimulationError('host kernel failed; see host-sanitizer.log') from exc

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()
        if self.process.returncode:
            raise SimulationError("host kernel failed; see host-sanitizer.log")
