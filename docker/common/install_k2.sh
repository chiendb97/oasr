#!/usr/bin/env bash
set -Eeo pipefail
trap 'echo "[install_k2.sh] Error on line $LINENO" >&2' ERR

K2_REPO="${K2_REPO:-https://github.com/chiendb97/k2.git}"
K2_BRANCH="${K2_BRANCH:-feat/cuda13}"
K2_SOURCE_DIR="${K2_SOURCE_DIR:-/opt/k2-src}"
K2_INSTALL_PREFIX="${K2_INSTALL_PREFIX:-/opt/k2-install}"

echo "Cloning k2: ${K2_REPO}@${K2_BRANCH}"
git clone --branch "${K2_BRANCH}" --depth 1 "${K2_REPO}" "${K2_SOURCE_DIR}"

# C++17 required for kaldifeat compatibility with recent PyTorch.
# Tests and benchmarks disabled to keep the build small.
cmake -S "${K2_SOURCE_DIR}" -B "${K2_SOURCE_DIR}/build" \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${K2_INSTALL_PREFIX}" \
    -DCMAKE_CXX_STANDARD=17 \
    -DK2_ENABLE_TESTS=OFF \
    -DK2_ENABLE_BENCHMARK=OFF \
    -DPYTHON_EXECUTABLE="$(which python3)"

cmake --build "${K2_SOURCE_DIR}/build" --target install

# Drop build artifacts; keep source tree for internal headers used by the
# streaming decoder (k2/csrc/, k2/torch/csrc/).
rm -rf "${K2_SOURCE_DIR}/build"
