#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${RTOPD_PYTHON:-$ROOT/ke/.venv/bin/python}" "$ROOT/ke/scripts/launch.py" "$@"
