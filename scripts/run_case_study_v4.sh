#!/usr/bin/env bash
set -euo pipefail
repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository"
exec "$repository/.venv/bin/python" "$repository/scripts/case_study_v4.py" "$@"
