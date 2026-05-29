#!/usr/bin/env bash
set -Eeo pipefail
trap 'echo "[install_pytorch.sh] Error on line $LINENO" >&2' ERR

# Install nightly PyTorch stack for CUDA 13.2.
# Override TORCH_INDEX_URL to target a different channel (e.g. stable cu128).
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu132}"

echo "Installing PyTorch from: ${TORCH_INDEX_URL}"
pip install --no-cache-dir \
    torch==2.12.0 \
    torchvision==0.27.0 \
    torchcodec==0.13.0 \
    --index-url "${TORCH_INDEX_URL}" -U

echo "Installing PyTorch Audio from: https://github.com/pytorch/audio.git@v2.11.0"
USE_CUDA=1 pip install -v --no-build-isolation "git+https://github.com/pytorch/audio.git@v2.11.0"