from pathlib import Path
import shutil
import subprocess

work = Path('/work/candidate')
shutil.copytree('/candidate', work)
with Path('/output/design.mlir').open('wb') as output:
    subprocess.run(['python', 'design.py', '--dev', 'npu2', '--emit-mlir'], cwd=work, stdout=output, check=True)
