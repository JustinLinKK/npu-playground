import importlib
import json
from pathlib import Path
import shutil

profile = json.loads(Path('/contract/target.json').read_text())
modules = ['numpy', 'pydantic'] + (['aie.ir', 'ml_dtypes'] if profile['vendor'] == 'amd' else ['openvino'])
report = {'modules': {}, 'tools': {}}
for name in modules:
    importlib.import_module(name)
    report['modules'][name] = True
for name in (['aiecc', 'g++', 'xclbinutil'] if profile['vendor'] == 'amd' else ['python']):
    report['tools'][name] = shutil.which(name)
    if not report['tools'][name]:
        raise RuntimeError(f'missing tool: {name}')
report['toolchain'] = json.loads(Path('/opt/toolchain-metadata.json').read_text())
Path('/output/capabilities.json').write_text(json.dumps(report, indent=2))
