#!/usr/bin/env bash
set -Eeo pipefail
trap 'echo "[install.sh] Error on line $LINENO" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

build_deps=0
pytorch=0
k2=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --build_deps) build_deps=1; shift ;;
        --pytorch)    pytorch=1;    shift ;;
        --k2)         k2=1;         shift ;;
        --all)        build_deps=1; pytorch=1; k2=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ $build_deps -eq 1 ]]; then
    echo "Installing build dependencies..."
    apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir apache-tvm-ffi pybind11 jinja2 numpy
fi

if [[ $pytorch -eq 1 ]]; then
    echo "Installing PyTorch..."
    bash "${SCRIPT_DIR}/install_pytorch.sh"
fi

if [[ $k2 -eq 1 ]]; then
    echo "Installing k2..."
    bash "${SCRIPT_DIR}/install_k2.sh"
fi
