# OASR - Open Automatic Speech Recognition

A high-performance open-source inference framework for ASR models built with C++, CUDA, and CUTLASS. Kernels are JIT-compiled at runtime via TVM-FFI and cached for subsequent use.

## Supported Models

| Model | Description | Status |
|-------|-------------|--------|
| **Conformer** | Convolution-augmented Transformer | ✅ |
| **Paraformer** | Parallel Transformer for ASR | 🔲 Planned |
| **Branchformer** | Branch-based Transformer | 🔲 Planned |
| **Zipformer** | Zipformer encoder | 🔲 Planned |
| **Transducer** | Transducer | 🔲 Planned |

## Supported Decoders

| Decoder | Description | Status |
|---------|-------------|--------|
| **CTC Greedy Search** | CPU greedy argmax decode | ✅ |
| **CTC Prefix Beam Search (CPU)** | CPU prefix beam search | ✅ |
| **CTC Prefix Beam Search (GPU)** | GPU beam search (offline + multi-request streaming) | ✅ |
| **CTC WFST Beam Search** | Weighted FST beam search (requires k2; offline + streaming) | ✅ |

## Features

- **JIT-compiled CUDA kernels** via TVM-FFI with automatic caching (`~/.cache/oasr/jit/`)
- **CUTLASS-based GEMM** -- GEMM, Batched GEMM (BMM), and Grouped GEMM (variable-sized problems)
- **Normalization** -- LayerNorm, RMSNorm, BatchNorm1d, GroupNorm (fused add+norm, fused norm+activation)
- **Convolution** -- Depthwise/pointwise Conv1D (including causal), Conv2D via CUTLASS Implicit GEMM, cuDNN Conv2D
- **Activation** -- GLU, Swish
- **Attention** -- Multi-head and relative-position attention with optional rotary embeddings
- **CTC decoder** -- CTC greedy, prefix beam, and WFST beam search (CPU-side C++ with Python wrappers); GPU prefix beam search with offline + multi-request streaming APIs
- **Feature extraction** -- Batched FBANK / MFCC via `torchaudio` (default) or `kaldifeat` (optional GPU path), plus `BatchedStreamingFeatureExtractor` for `B` parallel chunked streams
- **Inference engine** -- vLLM-style `ASREngine` (streaming + offline) and `OfflineEngine` for Conformer-CTC, with dynamic length-bucketed batching and paged KV cache
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
│   ├── features/              # Batched FBANK/MFCC + BatchedStreamingFeatureExtractor (torchaudio / kaldifeat)
│   ├── models/conformer/      # Conformer model, config, weight conversion
│   ├── engine/                # Inference engine (ASREngine, OfflineEngine, EngineConfig, Scheduler, Pipeline)
│   └── utils/                 # Dtype mappings, validation decorators, NVTX, timer
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

OASR provides a unified benchmark framework (`benchmarks/oasr_benchmark.py`) with a per-family routine registry under `benchmarks/routines/`. Backend names depend on the kernel family: `cutlass` / `torch` for GEMM and Conv2D; `cuda` / `torch` for Norm, Conv1D, Activation, and composite kernels. Legacy standalone `bench_*.py` scripts remain as thin wrappers.

### Unified CLI

```bash
# GEMM family (cutlass vs. torch reference)
python benchmarks/oasr_benchmark.py --routine gemm --subroutine bmm \
    --backends cutlass torch --batch-count 256 --M 200 --N 200 --K 64 \
    --dtype float16 --refcheck -vv

# Norm / Conv1D / Activation family (cuda vs. torch reference)
python benchmarks/oasr_benchmark.py --routine norm --subroutine layer_norm \
    --backends cuda torch --batch 64 --seq 250 --hidden 512 --refcheck -vv

# List all available routines/subroutines
python benchmarks/oasr_benchmark.py --list

# Batch testing from a testlist file
python benchmarks/oasr_benchmark.py --testlist benchmarks/testlists/conformer_base.txt \
    --output_path results.csv --refcheck
```

### Profiling with Nsight Compute

`--profile` emits NVTX markers and runs in single-iteration mode suitable for `ncu`:

```bash
ncu --set full -o gemm_profile python benchmarks/oasr_benchmark.py \
    --routine gemm --subroutine gemm --backends cutlass --profile --dry_run_iters 0
```

See `benchmarks/README.md` for the full CLI reference.

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

## Engine

The `oasr.engine` subpackage provides a vLLM-inspired inference engine for Conformer-CTC models. It handles feature extraction, streaming chunk-by-chunk encoding with paged KV cache, CTC decoding, and detokenization in one unified interface.

### Offline transcription

```python
from oasr.engine import OfflineEngine, EngineConfig

engine = OfflineEngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))

# Single file
text = engine.transcribe("audio.wav")

# Batch
texts = engine.transcribe(["a.wav", "b.wav", "c.wav"])

# Simulate streaming (chunk-by-chunk, no real-time constraint)
text = engine.transcribe_streaming("audio.wav", chunk_size=16)
```

### Streaming transcription (multi-request)

```python
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))

# Convenience: add requests and run to completion
texts = engine.transcribe(["a.wav", "b.wav", "c.wav"])

# Explicit step loop for online serving
rid = engine.add_request("audio.wav")
results = engine.run()
text = {r.request_id: r.text for r in results}[rid]
```

### EngineConfig

Key parameters for `EngineConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ckpt_dir` | `""` | WeNet-format checkpoint directory |
| `device` | `"cuda"` | CUDA device string |
| `dtype` | `torch.float16` | Model and cache precision |
| `chunk_size` | `16` | Encoder output frames per streaming chunk |
| `num_left_chunks` | `-1` | Past chunks to keep in attention cache (`-1` = unlimited) |
| `max_batch_size` | `32` | Max concurrent streaming requests per step |
| `decoder_type` | `"ctc_prefix_beam"` | One of `ctc_greedy`, `ctc_prefix_beam`, `ctc_gpu`, `ctc_wfst` |
| `audio_scale` | `32768.0` | Scale factor applied before feature extraction (restores int16 range) |

The checkpoint directory is expected to contain `final.pt`, `train.yaml`, `global_cmvn`, and optionally a SentencePiece `.model` file and `units.txt` vocabulary. These are auto-detected when `ckpt_dir` is set.

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
