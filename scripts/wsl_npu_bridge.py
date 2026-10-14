"""Run scripts/intel_npu_executor.py in Windows-native OpenVINO from WSL2.

WSL2 kernels do not expose the Intel NPU (no /dev/accel), so the device is reachable only from
Windows. This bridge is the TargetProfile.hardware_runner command: it translates artifact paths
for Windows, serializes NPU access across parallel workers, and relays the executor's JSON.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = ROOT / 'scripts' / 'intel_npu_executor.py'
LOCK = ROOT / '.npu-agent' / 'npu.lock'


def windows_path(path) -> str:
    return subprocess.run(['wslpath', '-w', str(Path(path).resolve())], capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows-python', required=True, help='for example /mnt/c/npu-runner/venv/Scripts/python.exe')
    parser.add_argument('--probe', action='store_true', help='print the NPU identity and run an exact self-test')
    args = parser.parse_args(argv)
    # -B: never write Windows bytecode caches into the shared repository.
    command = [args.windows_python, '-B', windows_path(EXECUTOR)]
    if args.probe:
        return subprocess.run(command + ['--probe']).returncode
    request = json.loads(sys.stdin.buffer.read())
    request['artifacts'] = {name: windows_path(path) for name, path in request['artifacts'].items()}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as lock:
        # Execution runs outside the compiler job limiter; keep one NPU client at a time.
        fcntl.flock(lock, fcntl.LOCK_EX)
        process = subprocess.run(command, input=json.dumps(request).encode(), capture_output=True)
    sys.stderr.buffer.write(process.stderr)
    sys.stdout.buffer.write(process.stdout)
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
