#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVAL_PYTHON=${EVAL_PYTHON:-python3.12}
EVAL_ENV=${EVAL_ENV:-"$ROOT/.venv-eval"}
"$EVAL_PYTHON" -m venv "$EVAL_ENV"
"$EVAL_ENV/bin/python" -m pip install 'pip==25.2' 'setuptools==80.10.2' 'wheel==0.45.1'
"$EVAL_ENV/bin/python" -m pip install \
 'https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
 --extra-index-url https://download.pytorch.org/whl/cu129 \
 'torch==2.11.0+cu129' 'transformers==5.14.1' 'numpy==2.3.5' \
 'librosa==0.10.0.post2' 'soundfile==0.13.1' 'safetensors==0.8.0' \
 'soxr==1.1.0' 'tokenizers==0.22.2' 'qwen-omni-utils==0.0.4'
"$EVAL_ENV/bin/python" -m pip check
"$EVAL_ENV/bin/python" -c "import sys; sys.path.insert(0, '$ROOT/ke'); from scripts.grid16_eval import verify_inference_runtime; verify_inference_runtime(); import pkg_resources; print('Frozen evaluation runtime verified')"
