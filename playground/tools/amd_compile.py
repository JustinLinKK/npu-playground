"""Trusted v1.4.2 Peano/aiecc driver. Candidate Python is never imported here."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from aie.ir import Context, Module, FunctionType
from aie.dialects import aie, aiex, arith, func, scf
from aie.dialects.aie import AIEDevice
from aie.utils import config

from npu_agent.validation import sha256, write_json


def main():
    work = Path('/work/candidate')
    work.mkdir()
    shutil.copyfile('/candidate/kernel.cc', work / 'kernel.cc')
    shutil.copyfile('/design/design.mlir', work / 'design.mlir')
    symbols = {}
    with Context():
        module = Module.parse((work / 'design.mlir').read_text())
        if not module.operation.verify():
            raise ValueError('MLIR verification failed')
        devices = list(module.body.operations)
        if len(devices) != 1 or devices[0].operation.name != 'aie.device':
            raise ValueError('exactly one target device required')
        device = devices[0].operation
        architecture = AIEDevice(device.attributes['device'].value).name
        if not architecture.startswith('npu2'):
            raise ValueError(f'target_mismatch: {architecture}')
        for block in device.regions[0].blocks:
            for view in block.operations:
                op = view.operation
                if op.name == 'func.func':
                    if any(len(r.blocks) for r in op.regions):
                        raise ValueError('external C++ functions required')
                    name = op.attributes['sym_name'].value
                    link = op.attributes['link_with'].value
                    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.o', link):
                        raise ValueError('unsafe external symbol or link path')
                    symbols[name] = {'object': link, 'signature': str(op.attributes['function_type'].value)}
        if not symbols:
            raise ValueError('design has no declared kernel.cc entrypoint')
    profile = json.loads(Path('/contract/target.json').read_text())
    optimization = profile['compiler_properties'].get('kernel_optimization', '2')
    if optimization not in ('0', '1', '2', '3'):
        raise ValueError('invalid kernel optimization level')
    commands = []
    def run(command):
        commands.append(command)
        write_json(Path('/output/commands.json'), commands)
        print(json.dumps(command), flush=True)
        subprocess.run(command, cwd=work, check=True)
    for link in sorted({s['object'] for s in symbols.values()}):
        # Matches aie.utils.compile.utils.compile_cxx_core_function in the pinned release.
        run([config.peano_cxx_path(), str(work / 'kernel.cc'), '-c', '-o', str(work / link),
             f'-I{config.cxx_header_path()}', '-I/tools/host_compat', '-std=c++20', '-Wno-parentheses', '-Wno-attributes',
             '-Wno-macro-redefined', '-Wno-empty-body', f'-O{optimization}', '-DNDEBUG', '-MD', '-MF', str(work / f'{link}.d'),
             '-D__AIE_API_AIE_ADF_HPP__', '--target=aie2p-none-unknown-elf'])
        table = subprocess.check_output(['readelf', '-Ws', str(work / link)], text=True)
        for name, spec in symbols.items():
            if spec['object'] == link and not any(line.split()[-1:] == [name] and ' UND ' not in line for line in table.splitlines()):
                raise ValueError(f'kernel source does not define {name}')
    run([config.aiecc_path(), str(work / 'design.mlir'), f'--peano={config.peano_install_dir()}',
         '--get-npu-insts', '--npu-insts-name=/output/insts.bin', '--get-xclbin', '--xclbin-name=/output/final.xclbin',
         '--tmpdir=/output/intermediates', '--output-dir=/output/intermediates', '--get-input-with-addresses',
         '--dump-intermediates', '--verbose'])
    for link in sorted({s['object'] for s in symbols.values()}):
        shutil.copyfile(work / link, Path('/output') / link)
    elfs = sorted(Path('/output/intermediates').rglob('*.elf'))
    if not elfs:
        raise ValueError('compiler emitted no core ELF')
    linked_symbols = '\n'.join(subprocess.check_output(['readelf', '-Ws', str(p)], text=True) for p in elfs)
    for name in symbols:
        if not any(line.split()[-1:] == [name] and ' UND ' not in line for line in linked_symbols.splitlines()):
            raise ValueError(f'expected core entry symbol not linked: {name}')
    Path('/output/linked-symbols.txt').write_text(linked_symbols)
    inspection = subprocess.check_output(['xclbinutil', '--info', '--input', '/output/final.xclbin'], text=True)
    Path('/output/inspection.txt').write_text(inspection)
    write_json(Path('/output/abi.json'), {'symbols': symbols, 'device': architecture,
        'source_sha256': sha256(work / 'kernel.cc'), 'mlir_sha256': sha256(work / 'design.mlir')})
    for name in ('insts.bin', 'final.xclbin'):
        if not (Path('/output') / name).stat().st_size:
            raise ValueError(f'empty artifact: {name}')


if __name__ == '__main__':
    main()
