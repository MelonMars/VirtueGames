#!/usr/bin/env bash
# Run inside an existing Runpod PyTorch GPU Pod after uploading this project.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"
case "${1:-}" in
  "") python -m pip install -r requirements-transformers.txt ;;
  --activations) python -m pip install -r requirements-activations.txt ;;
  *) printf '%s\n' 'Usage: bash runpod_setup.sh [--activations]' >&2; exit 2 ;;
esac
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Use a Runpod PyTorch GPU template with CUDA-enabled torch.')
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM (GiB):', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
PY
printf '%s\n' 'Setup complete. Set HF_HOME=/workspace/.cache/huggingface in your experiment shell.'
