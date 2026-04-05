# ---------------------------------------------------------------------------
# Stage 1 – build k2
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:26.03-py3 AS k2-builder

ARG K2_REPO=https://github.com/chiendb97/k2.git
ARG K2_BRANCH=feat/cuda13

# Build tools (only needed in this stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Clone k2 source
RUN git clone --branch "${K2_BRANCH}" --depth 1 "${K2_REPO}" /opt/k2-src

# Configure k2
# Install to /opt/k2-install (a dedicated prefix kept isolated from /usr/local).
# C++17 is required for kaldifeat compatibility with recent PyTorch.
# Tests and benchmarks are disabled to keep the layer small.
RUN cmake -S /opt/k2-src -B /opt/k2-src/build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/k2-install \
        -DCMAKE_CXX_STANDARD=17 \
        -DK2_ENABLE_TESTS=OFF \
        -DK2_ENABLE_BENCHMARK=OFF \
        -DPYTHON_EXECUTABLE="$(which python3)"

# Build and install
RUN cmake --build /opt/k2-src/build --target install

# Drop the build directory; keep the source for internal headers.
RUN rm -rf /opt/k2-src/build

# ---------------------------------------------------------------------------
# Stage 2 – runtime image
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:26.03-py3 AS runtime

ARG CUDA_ARCHS=80

# Copy k2 install tree into /usr/local so CMake's default search path finds it:
#   /usr/local/include/k2/torch_api.h
#   /usr/local/lib/libk2_*.so
#   /usr/local/share/cmake/k2/k2Config.cmake
COPY --from=k2-builder /opt/k2-install/ /usr/local/

# Copy k2 source tree for internal headers required by the streaming decoder:
#   k2/csrc/intersect_dense_pruned.h, k2/torch/csrc/{fsa_class,decode,...}.h
COPY --from=k2-builder /opt/k2-src /opt/k2-src

# Refresh the linker cache so libk2_*.so are found at runtime
RUN ldconfig

# OASR Python runtime / build deps
RUN pip install --no-cache-dir \
        apache-tvm-ffi \
        pybind11 \
        jinja2 \
        numpy

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
    CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    pip install --no-build-isolation -e .

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
# Expose K2_SOURCE_DIR so downstream processes can reference the source tree
ENV K2_SOURCE_DIR=/opt/k2-src

CMD ["/bin/bash"]
