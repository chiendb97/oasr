# OASR - Open Automatic Speech Recognition

A high-performance open-source inference framework for ASR models built with C++, CUDA, and CUTLASS. Kernels are JIT-compiled at runtime via TVM-FFI and cached for subsequent use.

## Supported Models

- **Conformer** - Convolution-augmented Transformer
- **Paraformer** - Parallel Transformer for ASR
- **Branchformer** - Branch-based Transformer

## Features

- **JIT-compiled CUDA kernels** via TVM-FFI with automatic caching (`~/.cache/oasr/jit/`)
- **CUTLASS-based GEMM** -- GEMM, Batched GEMM (BMM), and Grouped GEMM (variable-sized problems)
- **Normalization** -- LayerNorm, RMSNorm, BatchNorm1d, GroupNorm (fused add+norm, fused norm+activation)
- **Convolution** -- Depthwise/pointwise Conv1D (including causal), Conv2D via CUTLASS Implicit GEMM, cuDNN Conv2D
- **Activation** -- GLU, Swish
- **Attention** -- Multi-head and relative-position attention with optional rotary embeddings
- **CTC decoder** -- CTC greedy search, prefix beam search, and WFST beam search (CPU-side C++ with Python wrappers)
- **Streaming inference** -- Chunk-by-chunk Conformer inference with paged GPU memory for attention and CNN caches
- **Paged memory cache** -- `BlockPool`-backed paged KV cache and `StreamContext` for multi-request streaming
- **Kernel auto-tuning** -- Built-in profiling and caching framework for GEMM and Conv2D kernels
- **Flash Attention** -- Optional support (enable at build time)
- **Architecture support** -- Volta through Blackwell (SM70--SM120)
- **Python bindings** -- Functional API + nn.Module wrappers
- **Benchmarks and profiling** -- Triton-style timing and NVTX/Nsight Compute support

## Architecture

```
Python functional API (oasr/<family>.py)
    +-- JIT generator (oasr/jit/<family>.py) -> JitSpec
            +-- TVM-FFI JIT binding (csrc/<family>_jit_binding.cu)
                    +-- TVM-FFI launcher (csrc/<family>.cu)
                            +-- Pure CUDA kernels (include/oasr/<family>.cuh)
                                    +-- CUTLASS ops + include/oasr/common/ utilities
```

```
oasr/
├── include/oasr/              # Pure CUDA kernel headers (no framework deps)
│   ├── common/                # Types, vec dtypes, arch dispatch/traits, math
│   ├── conv/                  # conv1d.cuh, conv2d.cuh
│   ├── gemm/                  # gemm.cuh, bmm.cuh, group_gemm.cuh
│   ├── activation.cuh         # GLU, Swish
│   └── norm.cuh               # LayerNorm, RMSNorm, BatchNorm1d, GroupNorm
├── csrc/                      # TVM-FFI launchers + JIT bindings
│   ├── *.cu                   # Per-kernel launchers and _jit_binding files
│   ├── tvm_ffi_utils.h        # DLPack dtype dispatch and validation macros
│   ├── decoder/               # CTC greedy/prefix-beam/WFST decoders + context graph (CPU, C++)
│   └── pybind/                # pybind11 module (decoder + legacy enums)
├── oasr/                      # Python package
│   ├── activation.py          # Functional API: GLU, Swish
│   ├── norm.py                # Functional API: LayerNorm, RMSNorm, etc.
│   ├── conv.py                # Functional API: Conv1D, Conv2D
│   ├── gemm.py                # Functional API: GEMM, BMM, Grouped GEMM
│   ├── ctc_decode.py          # High-level CTC decode helpers
│   ├── aot.py                 # Ahead-of-time compilation registration
│   ├── jit/                   # JIT spec generators (core.py + per-family)
│   ├── tune/                  # Auto-tuning framework (profiler, cache, backends)
│   ├── layers/                # nn.Module wrappers (norm, conv, linear, attention, rotary)
│   ├── decoder/               # Python wrappers: CtcGreedySearch, CtcPrefixBeamSearch, CtcWfstBeamSearch
│   ├── cache/                 # Streaming cache manager (BlockPool, AttentionCacheManager, StreamContext)
│   ├── models/conformer/      # Conformer model, config, weight conversion
│   └── utils/                 # Dtype mappings, timer
├── benchmarks/                # Performance benchmarks
├── tests/                     # Pytest test suite
├── CMakeLists.txt
├── setup.py
└── pyproject.toml
```

## Requirements

- **CUDA** >= 11.8
- **CMake** >= 3.18
- **Python** >= 3.8
- **apache-tvm-ffi** -- TVM FFI runtime for JIT compilation
- **CUTLASS** -- NVIDIA CUDA Templates for Linear Algebra Subroutines and Solvers; required for CUTLASS-based kernel generation and execution
- **cuDNN** -- Optional; Conv2D cuDNN backend disabled if not found

## Installation

```bash
# Editable install (builds pybind11 extension via CMake; kernels are JIT-compiled on first use)
pip install -e .


# Multiple architectures (semicolon-separated)
CUDA_ARCHITECTURES="80;86;90" pip install -e .

# With Flash Attention support
OASR_USE_FLASH_ATTENTION=1 pip install -e .
```

Optional dependency groups:

```bash
pip install -e ".[dev]"      # pytest, black, mypy, ruff
pip install -e ".[audio]"    # soundfile, librosa
pip install -e ".[serving]"  # FastAPI, Uvicorn
```

## Testing

### Python tests

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_conv.py

# Run a single test class
pytest tests/test_conv.py::TestDepthwiseConv1D -v

# Skip slow tests
pytest tests/ -m "not slow"
```

### C++ unit tests

```bash
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON -DBUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=80
make -j
ctest --output-on-failure
```

## Benchmarks and Profiling

Benchmark scripts in `benchmarks/` use `triton.testing.do_bench` when Triton is available.

### Available benchmarks

```bash
# Normalization
python benchmarks/bench_layer_norm.py
python benchmarks/bench_rms_norm.py
python benchmarks/bench_batch_norm.py
python benchmarks/bench_group_norm.py

# Convolution
python benchmarks/bench_depthwise_conv1d.py
python benchmarks/bench_pointwise_conv1d.py
python benchmarks/bench_conv2d.py
python benchmarks/bench_conv_block.py

# GEMM
python benchmarks/bench_gemm.py
python benchmarks/bench_bmm.py
python benchmarks/bench_group_gemm.py

# Activation
python benchmarks/bench_glu.py
python benchmarks/bench_swish.py
```

### Profiling with Nsight Compute

Use `--profile` for single-iteration runs suitable for `ncu`:

```bash
ncu --set full -o gemm_profile python benchmarks/bench_gemm.py \
  --profile --kernel gemm --target oasr --warmup 0 --iters 1

ncu --set full -o norm_profile python benchmarks/bench_layer_norm.py \
  --profile --kernel layer_norm --target oasr --warmup 0 --iters 1
```

Each script supports `--kernel`, `--target` (oasr/pytorch/both), `--warmup`, and `--iters`; see `--help`.

## Quick Start (Python)

```python
import oasr

# Functional API (kernels JIT-compile on first call)
# Normalization
output = oasr.layer_norm(input, weight, bias)
output = oasr.rms_norm(input, weight)

# Convolution
output = oasr.depthwise_conv1d(input, weight, bias)
output = oasr.pointwise_conv1d(input, weight, bias)

# GEMM
output = oasr.gemm(A, B)
output = oasr.bmm(A, B)
output = oasr.group_gemm(A_list, B_list)

# Activation
output = oasr.glu(input)
output = oasr.swish(input)
```

```python
# nn.Module wrappers
from oasr.layers import LayerNorm, Linear, Conv1D

norm = LayerNorm(hidden_size)
linear = Linear(in_features, out_features)
```

## Streaming Inference

The `oasr.cache` subpackage provides a paged-memory streaming cache for chunk-by-chunk Conformer inference:

```python
from oasr.cache import (
    CacheConfig, BlockPool,
    AttentionCacheManager, CnnCacheManager, CtcStateCacheManager,
    StreamContext,
)
from oasr import GpuDecoderConfig

config = CacheConfig(
    num_layers=12, n_kv_head=4, head_dim=64, hidden_dim=256,
    kernel_size=15, chunk_size=16, num_left_chunks=4,
    block_size_frames=16, max_num_blocks=2048,
)
pool    = BlockPool(config)
att_mgr = AttentionCacheManager(pool, config)
cnn_mgr = CnnCacheManager(config)
ctc_mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=10))

# Per-request streaming
sid = 42
att_mgr.allocate_stream(sid)
cnn_mgr.allocate_stream(sid)
ctc_mgr.allocate_stream(sid, batch=1, vocab_size=5000)
ctx = StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)

for chunk_audio in audio_chunks:
    att_cache  = ctx.get_att_cache()
    cnn_cache  = ctx.get_cnn_cache()
    logits, new_att, new_cnn = model.forward_chunk(
        chunk_audio, offset, required_cache_size, att_cache, cnn_cache)
    ctx.commit_chunk(new_att[:, :, -chunk_size:, :], new_cnn)
    ctx.get_decoder().decode_chunk(logits)

result = ctx.get_decoder().finalize_stream()
ctx.free()
```

## Kernel Auto-Tuning

OASR includes a built-in auto-tuning framework for GEMM and Conv2D kernels. See `oasr/tune/` for details.

## License

Apache 2.0
