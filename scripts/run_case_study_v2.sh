#!/usr/bin/env bash
set -euo pipefail

repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repository"
exec uv run --no-sync python scripts/case_study_v2.py "$@"
