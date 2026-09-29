import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

metadata = {'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, 'tools': {}}
for name in ['python', 'aiecc', 'xclbinutil', 'g++']:
    path = shutil.which(name)
    if path:
        file = Path(path).resolve()
        version = subprocess.run([str(file), '--version'], capture_output=True, text=True, timeout=30)
        metadata['tools'][name] = {'path': str(file), 'sha256': hashlib.file_digest(file.open('rb'), 'sha256').hexdigest(),
                                   'version': (version.stdout + version.stderr)[:4000]}
peano = Path('/opt/ironenv/lib/python3.12/site-packages/llvm-aie/bin/clang++')
if peano.exists():
    metadata['tools']['peano'] = {'sha256': hashlib.file_digest(peano.open('rb'), 'sha256').hexdigest(),
                                 'version': subprocess.check_output([str(peano), '--version'], text=True)}
openvino_root = Path('/opt/intel/openvino_2026/runtime/lib/intel64')
if openvino_root.exists():
    import openvino as ov
    metadata['openvino_version'] = ov.__version__
    for file in sorted(openvino_root.glob('*.so*')):
        if not file.is_symlink():
            metadata['tools'][file.name] = {'path': str(file),
                'sha256': hashlib.file_digest(file.open('rb'), 'sha256').hexdigest()}
Path('/opt/toolchain-metadata.json').write_text(json.dumps(metadata, indent=2))
