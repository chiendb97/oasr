# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage Dockerfile for OASR
# Build with: docker build -f docker/Dockerfile -t oasr .
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3

# ---------------------------------------------------------------------------
# Stage 1 – torch base (shared by builder and runtime to ensure ABI parity)
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS torch-base

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu132

RUN --mount=type=bind,source=docker/common,target=/opt/docker/common \
    TORCH_INDEX_URL=${TORCH_INDEX_URL} \
    bash /opt/docker/common/install.sh --pytorch

# ---------------------------------------------------------------------------
# Stage 2 – build base (torch-base + build tools + build-time Python deps)
# ---------------------------------------------------------------------------
FROM torch-base AS build-base

RUN --mount=type=bind,source=docker/common,target=/opt/docker/common \
    bash /opt/docker/common/install.sh --build_deps

# ---------------------------------------------------------------------------
# Stage 3 – build k2
# ---------------------------------------------------------------------------
FROM build-base AS k2-builder

ARG K2_REPO=https://github.com/chiendb97/k2.git
ARG K2_BRANCH=feat/cuda13

RUN --mount=type=bind,source=docker/common,target=/opt/docker/common \
    K2_REPO=${K2_REPO} \
    K2_BRANCH=${K2_BRANCH} \
    bash /opt/docker/common/install.sh --k2

# ---------------------------------------------------------------------------
# Stage 4 – runtime image
# ---------------------------------------------------------------------------
FROM torch-base AS runtime

# Copy k2 install tree into /usr/local so CMake's default search path finds it:
#   /usr/local/include/k2/torch_api.h
#   /usr/local/lib/libk2_*.so
#   /usr/local/share/cmake/k2/k2Config.cmake
COPY --from=k2-builder /opt/k2-install/ /usr/local/

# Copy k2 source tree for internal headers required by the streaming decoder:
#   k2/csrc/intersect_dense_pruned.h, k2/torch/csrc/{fsa_class,decode,...}.h
COPY --from=k2-builder /opt/k2-src /opt/k2-src

ENV K2_SOURCE_DIR=/opt/k2-src

# Refresh the linker cache so libk2_*.so are found at runtime
RUN ldconfig

# Copy OASR source (see .dockerignore for exclusions)
WORKDIR /workspace/oasr
COPY . .

# Build and install OASR with k2 support (offline + streaming decoders).
#
# setup.py resolves k2_DIR automatically:
#   _find_k2_cmake_dir() → /usr/local/share/cmake/k2 (copied above)
#
# K2_SOURCE_DIR is forwarded to CMake via the env-var hook in setup.py,
# enabling the streaming decoder's internal-header includes.
RUN OASR_USE_K2=1 \
    K2_SOURCE_DIR=/opt/k2-src \
    pip install --no-build-isolation -e .

CMD ["/bin/bash"]
