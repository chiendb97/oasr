# OASR - Open Automatic Speech Recognition

A high-performance open-source inference framework for ASR models built with C++, CUDA, and CUTLASS.

## Supported Models

- **Conformer** - Convolution-augmented Transformer
- **Paraformer** - Parallel Transformer for ASR
- **Branchformer** - Branch-based Transformer

## Features

- **High-performance CUDA kernels** for ASR inference
- **CUTLASS-based GEMM** – GEMM, Batched GEMM (BMM), and Grouped GEMM (variable-sized problems)
- **Normalization** – LayerNorm, RMSNorm, BatchNorm, GroupNorm (fused add+norm)
- **Convolution** – Depthwise/pointwise conv1d, GLU, Swish, Conformer conv block
- **Attention** – Multi-head and relative-position attention (interfaces in place)
- **Python bindings** via pybind11
- **Benchmarks and profiling** – Triton-style timing and NVTX/Nsight Compute support

## Architecture

```
oasr/
├── csrc/                    # C++/CUDA source
│   ├── common/              # Types, CUDA utils, tensor helpers
│   ├── kernels/             # CUDA kernel implementations
│   │   ├── attention/       # Attention params and kernels
│   │   ├── convolution/    # Conv1D, GLU, Swish, Conformer conv
│   │   ├── gemm/            # GEMM, BMM, Grouped GEMM (CUTLASS)
│   │   ├── normalization/  # LayerNorm, RMSNorm, BatchNorm, GroupNorm
│   │   └── reduction/       # Reduction helpers
│   ├── ops/                 # CUTLASS ops and configs
│   └── pybind/              # Python bindings
├── python/oasr/             # Python package
│   ├── _C.so                # C++ extension (built)
│   ├── layers/              # Python layer wrappers
│   └── utils/               # Utilities
├── benchmarks/              # Performance benchmarks and profiling
│   ├── bench_conv_kernels.py
│   ├── bench_gemm_kernels.py
│   └── bench_norm_kernels.py
├── tests/                   # Unit tests
│   └── cpp/                 # C++/CUDA tests (GTest)
├── CMakeLists.txt
├── setup.py
└── pyproject.toml
```

## Requirements

- **CUDA** >= 11.8
- **CMake** >= 3.18
- **Python** >= 3.8
- **pybind11** >= 2.10
- **CUTLASS** – Fetched automatically by CMake if not present under `third_party/cutlass`
- **cuDNN** – Optional; some features may be disabled if not found

## Installation

From the project root:

```bash
# Editable install (builds extension via CMake)
pip install -e .

# Target a specific GPU architecture (e.g. SM 80)
CUDA_ARCHITECTURES=80 pip install -e .

# Multiple architectures (semicolon-separated)
CUDA_ARCHITECTURES="80;86;90" pip install -e .
```

The build uses CMake under the hood; the Python extension is built and installed into `python/oasr/`.

## Testing

### C++ unit tests

Build and run C++ tests (GTest):

```bash
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON -DBUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=80
make -j
ctest --output-on-failure
```

Tests include normalization kernels, convolution params, and GEMM/BMM correctness and params.

## Benchmarks and profiling

Benchmark scripts live in `benchmarks/` and use `triton.testing.do_bench` when Triton is available.

### Run all benchmarks

```bash
# From repo root (so python/oasr is importable)
python benchmarks/bench_norm_kernels.py
python benchmarks/bench_conv_kernels.py
python benchmarks/bench_gemm_kernels.py
```

### Profiling with Nsight Compute (NVTX)

Use `--profile` for single-iteration runs suitable for `ncu`:

```bash
# Profile GEMM kernel (OASR implementation)
ncu --set full -o gemm_profile python benchmarks/bench_gemm_kernels.py \
  --profile --kernel gemm --target oasr --warmup 0 --iters 1

# Profile normalization
ncu --set full -o norm_profile python benchmarks/bench_norm_kernels.py \
  --profile --kernel layer_norm --target oasr --warmup 0 --iters 1
```

Each script supports `--kernel`, `--target` (oasr/pytorch/both), `--warmup`, and `--iters`; see `--help`.

## Quick start (Python)

```python
import oasr

# Device and types
oasr.init_cuda()
oasr.set_device(0)
# oasr.DataType.FP16, oasr.DataType.BF16, etc.

# Low-level kernels (examples)
# Normalization: oasr.kernels.normalization.layer_norm(...)
# Convolution:  oasr.kernels.convolution.depthwise_conv1d(...)
# GEMM:         oasr.kernels.gemm.invoke_gemm(params)
# BMM:          oasr.kernels.gemm.invoke_bmm(params)
# Grouped GEMM: oasr.kernels.gemm.invoke_group_gemm(params)
```

Model-level APIs (e.g. `oasr.models.Conformer`) are not yet implemented; the package currently exposes kernels and layer building blocks.

## License

Apache 2.0
