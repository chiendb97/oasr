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

COPY --from=k2-builder /opt/k2-install/ /usr/local/

COPY --from=k2-builder /opt/k2-src /opt/k2-src

ENV K2_SOURCE_DIR=/opt/k2-src

# Refresh the linker cache so libk2_*.so are found at runtime
RUN ldconfig

ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:$PATH

RUN --mount=type=bind,source=docker/common,target=/opt/docker/common \
    bash /opt/docker/common/install.sh --rust

RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "setuptools-rust>=1.10"

# Copy OASR source (see .dockerignore for exclusions)
WORKDIR /workspace/oasr
COPY . .

RUN OASR_USE_K2=1 \
    K2_SOURCE_DIR=/opt/k2-src \
    pip install --no-build-isolation -e .

CMD ["/bin/bash"]
