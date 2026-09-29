"""Install checksum-verified top-level vendor wheels inside the image."""
import hashlib
import json
import subprocess
import sys
import urllib.request
import urllib.parse
from pathlib import Path

lock = json.loads(Path('/opt/toolchains.lock.json').read_text())
for key in sys.argv[1:]:
    item = lock['amd'][key]
    path = Path('/tmp') / urllib.parse.unquote(item['url'].split('/')[-1])
    urllib.request.urlretrieve(item['url'], path)
    if hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != item['sha256']:
        raise ValueError(f'checksum mismatch: {key}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', str(path)], check=True)
    path.unlink()
