#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPD_PYTHON=${OPD_PYTHON:-python3.10}
OPD_ENV=${OPD_ENV:-"$ROOT/.venv"}
"$OPD_PYTHON" -m venv "$OPD_ENV"
"$OPD_ENV/bin/python" -m pip install 'pip==25.2' 'setuptools==65.5.0' 'wheel==0.45.1'
"$OPD_ENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124' 'torchaudio==2.5.1+cu124'
"$OPD_ENV/bin/python" -m pip install -r "$ROOT/environment/requirements.lock"
"$OPD_ENV/bin/python" -m pip check
"$OPD_ENV/bin/python" "$ROOT/scripts/preflight.py" --cpu-only
