"""Compatibility entry point for the isolated Intel stages.

DockerCompiler supplies the separate read-only graph/contract/input mounts.
The legacy all-in-one candidate/oracle process is intentionally retired.
"""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['build', 'execute', 'compile'])
    args = parser.parse_args()
    scripts = {'build': 'intel_build_graph.py', 'execute': 'intel_execute_graph.py', 'compile': 'intel_compile_graph.py'}
    os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name(scripts[args.stage]))])


if __name__ == '__main__':
    main()
