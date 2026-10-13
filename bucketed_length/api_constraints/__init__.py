"""A deliberately bounded API lint/repair experiment, not a grammar decoder."""
import ast
import re

from npu_agent.models import Backend


IMPORTS = {
    'argparse': {'ArgumentParser'}, 'json': {'loads', 'dumps'}, 'pathlib': {'Path'},
    'numpy': {'array', 'asarray', 'ndarray', 'dtype', 'float16', 'float32', 'int32'},
    'openvino': {'Model', 'opset13', 'Type'},
    'openvino.opset13': {'parameter', 'constant', 'add', 'subtract', 'multiply', 'divide', 'matmul',
        'reduce_sum', 'reduce_mean', 'reduce_max', 'exp', 'sqrt', 'softmax', 'reshape', 'transpose',
        'convert', 'maximum', 'minimum', 'select', 'result', 'gather', 'unsqueeze', 'squeeze'},
    'aie.dialects.aie': {'device', 'tile', 'core', 'object_fifo', 'object_fifo_link', 'AIEDevice',
                         'ObjectFifoPort', 'external_func'},
    'aie.dialects.aiex': {'runtime_sequence', 'npu_dma_memcpy_nd', 'dma_wait'},
    'aie.extras.context': {'mlir_mod_ctx'}, 'aie.helpers.dialects.ext.scf': {'_for'},
    'aie.iron': {'ObjectFifo', 'Worker', 'Runtime', 'Program', 'Kernel'},
    'aie.iron.device': {'NPU2'},
}
CPP_HEADERS = {'aie_api/aie.hpp', 'stdint.h', 'cstdint', 'stddef.h', 'cstddef', 'math.h', 'cmath', 'npu_numeric.h'}
CPP_SYMBOLS = {'vector', 'load_v', 'store_v', 'add', 'reduce_add'}


def check(bundle, fingerprint, expected_fingerprint):
    if not fingerprint or fingerprint == 'unresolved' or fingerprint != expected_fingerprint:
        raise ValueError('API inventory and compiler fingerprints must match')
    errors = []
    for file in bundle.files:
        if file.relative_path.endswith('.cc'):
            code = re.sub(r'//[^\n]*|/\*.*?\*/', '', file.content, flags=re.S)
            for directive, value in re.findall(r'^\s*#\s*(\w+)([^\n]*)', code, re.M):
                match = re.fullmatch(r'\s*[<"]([^>"]+)[>"]\s*', value)
                if directive != 'include' or match is None or match[1] not in CPP_HEADERS:
                    errors.append(f'unsupported preprocessor directive: #{directive}{value}')
            symbols = set(re.findall(r'\baie::([A-Za-z_]\w*)', code))
            errors += [f'unsupported AIE symbol: aie::{name}' for name in sorted(symbols - CPP_SYMBOLS)]
        elif file.relative_path.endswith('.py'):
            try:
                tree = ast.parse(file.content)
            except SyntaxError as exc:
                errors.append(f'Python syntax error on line {exc.lineno}: {exc.msg}')
                continue
            aliases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for item in node.names:
                        wrong_backend = ((bundle.backend == Backend.AMD_XDNA2 and item.name.startswith('openvino'))
                                         or (bundle.backend == Backend.INTEL_OPENVINO and item.name.startswith('aie')))
                        if item.name not in IMPORTS or wrong_backend:
                            errors.append(f'unsupported import: {item.name}')
                        aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
                elif isinstance(node, ast.ImportFrom):
                    for item in node.names:
                        wrong_backend = ((bundle.backend == Backend.AMD_XDNA2 and (node.module or '').startswith('openvino'))
                                         or (bundle.backend == Backend.INTEL_OPENVINO and (node.module or '').startswith('aie')))
                        if node.level or item.name not in IMPORTS.get(node.module, set()) or wrong_backend:
                            errors.append(f'unsupported import: {node.module}.{item.name}')
                        aliases[item.asname or item.name] = f'{node.module}.{item.name}'
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in {'getattr', 'eval', 'exec', '__import__', 'compile', 'globals', 'locals'}:
                        errors.append(f'dynamic API resolution is outside this subset: {node.func.id}')
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                path = []
                root = node
                while isinstance(root, ast.Attribute):
                    path.insert(0, root.attr)
                    root = root.value
                if isinstance(root, ast.Name) and root.id in aliases:
                    full = aliases[root.id] + '.' + '.'.join(path)
                    parent, _, symbol = full.rpartition('.')
                    if parent in IMPORTS and symbol not in IMPORTS[parent]:
                        errors.append(f'unsupported API symbol: {full}')
                    if bundle.backend == Backend.AMD_XDNA2 and full.startswith('openvino.'):
                        errors.append('OpenVINO API used in an AMD bundle')
                    if bundle.backend == Backend.INTEL_OPENVINO and full.startswith('aie.'):
                        errors.append('AIE API used in an Intel bundle')
    return sorted(set(errors))


def prompt_surface(target):
    modules = {name: sorted(symbols) for name, symbols in IMPORTS.items()
               if not (target.vendor == 'amd' and name.startswith('openvino'))
               and not (target.vendor == 'intel' and name.startswith('aie'))}
    return {'python_imports': modules, 'cpp_headers': sorted(CPP_HEADERS), 'aie_symbols': sorted(CPP_SYMBOLS),
            'scope': 'repository-supported subset; compiler validation is still required'}
