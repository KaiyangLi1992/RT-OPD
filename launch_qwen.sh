#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${RTOPD_PYTHON:-$ROOT/qwen/.venv/bin/python}" "$ROOT/qwen/scripts/launch.py" "$@"
