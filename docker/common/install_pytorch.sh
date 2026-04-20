#!/usr/bin/env bash
set -Eeo pipefail
trap 'echo "[install_pytorch.sh] Error on line $LINENO" >&2' ERR

# Install nightly PyTorch stack for CUDA 13.2.
# Override TORCH_INDEX_URL to target a different channel (e.g. stable cu128).
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"

echo "Installing PyTorch nightly from: ${TORCH_INDEX_URL}"
pip install --no-cache-dir --pre \
    torch \
    torchvision \
    torchaudio \
    torchcodec \
    --index-url "${TORCH_INDEX_URL}" -U
