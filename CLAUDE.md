# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OASR (Open Automatic Speech Recognition) is a high-performance CUDA inference framework for ASR models (Conformer, Paraformer, Branchformer). It exposes custom CUDA/CUTLASS kernels to Python via TVM-FFI JIT compilation (FlashInfer-style).

## Build

```bash
# Editable install (recommended for development)
pip install -e .

# Target specific GPU architecture
CUDA_ARCHITECTURES=80 pip install -e .

# With Flash Attention support
OASR_USE_FLASH_ATTENTION=1 pip install -e .
```

The build compiles the `_C.so` pybind11 extension (for decoder + enums) via CMake. CUDA kernels are JIT-compiled on first use via TVM-FFI and cached in `~/.cache/oasr/jit/`.

### C++ tests only

```bash
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON -DBUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=80
make -j
ctest --output-on-failure
```

## Testing

```bash
# Run all Python unit tests
pytest tests/

# Run a single test file
pytest tests/test_conv.py

# Run a single test function
pytest tests/test_conv.py::TestDepthwiseConv1D -v

# Skip slow tests
pytest tests/ -m "not slow"
```

Tests live under `tests/`. Functional API tests follow a flat `tests/test_<kernel>.py` layout (FlashInfer convention). The conftest at `tests/conftest.py` provides fixtures: `device` (CUDA, skips if unavailable), `dtype`/`dtype_all` (FP32/FP16/BF16), `batch_seq_hidden` (common shape tuples). Default pytest options (`-v --tb=short`) are set in `pyproject.toml`.

## Linting & Formatting

```bash
# Format Python code
black oasr/ tests/ benchmarks/
isort oasr/ tests/ benchmarks/

# Lint
ruff check oasr/ tests/ benchmarks/

# Type check
mypy oasr/

# Format C++/CUDA (requires clang-format)
clang-format -i csrc/**/*.cu csrc/**/*.h csrc/**/*.cpp
```

Style: Python uses 100-char line length (black + isort/black profile). C++ uses Google style with 100-char limit and C++17.

## Benchmarks & Profiling

The unified benchmark framework (`benchmarks/oasr_benchmark.py`) replaces standalone scripts. It uses a routine registry (`benchmarks/routines/`) with per-family modules.

Backend names differ by kernel family:
- `cutlass` / `torch` — GEMM, Conv2D (CUTLASS-based)
- `cuda` / `torch` — Norm, Conv1D, Activation, Composite (handwritten CUDA)

```bash
# Single kernel benchmark (GEMM family uses cutlass/torch)
python benchmarks/oasr_benchmark.py --routine gemm --subroutine bmm \
    --backends cutlass torch --batch-count 256 --M 200 --N 200 --K 64 --dtype float16 --refcheck -vv

# Single kernel benchmark (Norm/Conv1D/Activation family uses cuda/torch)
python benchmarks/oasr_benchmark.py --routine norm --subroutine layer_norm \
    --backends cuda torch --batch 64 --seq 250 --hidden 512 --refcheck -vv

# List all available routines/subroutines
python benchmarks/oasr_benchmark.py --list

# Batch testing from testlist files
python benchmarks/oasr_benchmark.py --testlist benchmarks/testlists/conformer_base.txt \
    --output_path results.csv --refcheck

# Profiling with Nsight Compute (NVTX markers via --profile)
ncu --set full -o gemm_profile python benchmarks/oasr_benchmark.py \
    --routine gemm --subroutine gemm --backends cutlass --profile --dry_run_iters 0
```

Legacy `bench_*.py` scripts still work as thin wrappers. See `benchmarks/README.md` for full CLI reference.

## Architecture

### Layered design

```
Python functional API (oasr/<family>.py)  — @oasr_api decorated
    └── JIT generator (oasr/jit/<family>.py) → JitSpec / JinjaJitSpec
            └── TVM-FFI JIT binding (csrc/<family>_jit_binding.cu)
                    └── TVM-FFI launcher (csrc/<family>.cu)
                            └── Pure CUDA kernels (include/oasr/<family>.cuh)  — facade
                                    └── Config  (cutlass_*_configs.h)
                                    └── Template (*_cutlass_template.h)
                                    └── Dispatch (*_cutlass.h / *_dispatch.inc)
```

### C++ / CUDA layer (FlashInfer-style config/template/dispatch split)

Each CUTLASS kernel family uses a three-header pattern:

| Header | Purpose | Example (GEMM) |
|--------|---------|-----------------|
| `cutlass_*_configs.h` | Config structs (`GemmConfig`), per-SM MMA traits (`SmMMATraits`), default configs (`DefaultGemmConfig`) | `gemm/cutlass_gemm_configs.h` |
| `*_cutlass_template.h` | CUTLASS kernel template parameterized by Config + MMATraits | `gemm/gemm_cutlass_template.h` |
| `*_cutlass.h` | Public dispatch interface (JIT mode via `OASR_TARGET_SM`, AOT mode via `OASR_DISPATCH_SM`) | `gemm/gemm_cutlass.h` |

Non-CUTLASS kernels (Conv1D, Norm, Activation) use `*_dispatch.inc` files with VecSize/block_size dispatch macros instead.

- **`include/oasr/`** — Pure CUDA kernel headers (no framework dependencies):
  - `common/` — Shared types (`types.h`), vector dtypes (`vec_dtypes.h`), SM dispatch (`arch_dispatch.h`), epilogue functors, math utilities.
  - `activation.cuh` + `activation_dispatch.inc` — GLU, Swish activation kernels with VecSize dispatch.
  - `norm.cuh` + `norm_dispatch.inc` — LayerNorm, RMSNorm, BatchNorm1d, GroupNorm, fused norm+activation with VecSize/block_size dispatch.
  - `conv/` — `conv1d.cuh` + `conv1d_dispatch.inc` (depthwise, pointwise, causal), `conv2d.cuh` facade → `cutlass_conv2d_configs.h` / `conv2d_cutlass_template.h` / `conv2d_cutlass.h`.
  - `gemm/` — `gemm.cuh` facade → `cutlass_gemm_configs.h` / `gemm_cutlass_template.h` / `gemm_cutlass.h`. Also `bmm.cuh`, `group_gemm.cuh`.
- **`csrc/`** — TVM-FFI launcher layer (`<family>.cu`) and JIT binding exports (`<family>_jit_binding.cu`). Also contains `tvm_ffi_utils.h` with DLPack dtype dispatch and validation macros.
- **`csrc/templates/`** — Jinja2 templates for config-specific CUTLASS instantiations (`gemm_cutlass_template.cu.jinja`, `bmm_cutlass_template.cu.jinja`, `group_gemm_cutlass_template.cu.jinja`).
- **`csrc/decoder/`** — CPU-side C++ decoder implementations: CTC greedy search, prefix beam search, WFST beam search (via k2), streaming WFST decoder, and `ContextGraph` for phrase boosting.
- **`csrc/pybind/`** — pybind11 module for decoder bindings and legacy enums (`pybind_main.cpp`, `pybind_decoder.h`).

### Dispatch modes

| Kernel Family | Dispatch Mode | Config Source | Source Generation |
|---------------|---------------|---------------|-------------------|
| GEMM, BMM, GroupGEMM | **jinja** | `cutlass_gemm_configs.h` | Jinja renders `.cu` with baked-in config |
| Conv2D | **jinja** | `cutlass_conv2d_configs.h` | Jinja renders `.cu` with baked-in config |
| Conv1D | **dispatch** | `conv1d_dispatch.inc` | Direct compilation, VecSize macro |
| Norm | **dispatch** | `norm_dispatch.inc` | Direct compilation, block/vec macro |
| Activation | **dispatch** | `activation_dispatch.inc` | Direct compilation, VecSize macro |

**JIT mode** (`OASR_TARGET_SM` defined): single SM instantiation, optional `JitGemmConfig`/`JitConv2dConfig` via `-D` flags.
**AOT mode** (no `OASR_TARGET_SM`): `OASR_DISPATCH_SM` macro switches on runtime SM version.

CUTLASS is fetched from GitHub (v4.4.2) at CMake time if not present under `third_party/cutlass`. CUDA SM targets default to 70, 75, 80, 86, 89, 90, 100, 120 (CMakeLists.txt); `setup.py` defaults to 70–90 only. Override with `CUDA_ARCHITECTURES` env var.

### Python layer (`oasr/`)

- **`__init__.py`** — Exposes all functional API functions (e.g., `oasr.gemm`, `oasr.layer_norm`) and nn.Module wrappers. Lazy-loads `_C` extension for decoder access.
- **`activation.py`**, **`norm.py`**, **`conv.py`**, **`gemm.py`** — Functional API: `@oasr_api` decorated, JIT-compile kernels on first call via `@functools.cache`, allocate output tensors, call into compiled modules.
- **`ctc_decode.py`** — GPU CTC prefix beam search, exposing two orthogonal APIs:
  - `ctc_beam_search_decode(log_prob, seq_lengths, ...)` — offline batched decode (allocates workspace + output, calls C++ in one shot).
  - `GpuStreamingDecoder` — streaming decoder with two usage modes:
    - *Single-request*: `init_stream(batch, vocab_size)` → `decode_chunk(log_prob)` → `finalize_stream()`.
    - *Multi-request (interleaved)*: `create_state(batch, vocab_size)` returns a `StreamState`; pass it to `decode_chunk(log_prob, state=s)` and `finalize_stream(state=s)`. `StreamHandle` wraps a `(decoder, state)` pair so callers need not carry both objects.
  - `GpuDecoderConfig` dataclass configures `beam_size`, `blank_id`, `blank_threshold`, `max_seq_len`, paged-memory options.
  - `GpuDecoderResult` holds `tokens` (nested list), `lengths`, and `scores` tensors.
- **`decode.py`** — Thin helpers wrapping `oasr.decoder` CPU-side decoders.
- **`api_logging.py`** — `@oasr_api` decorator for debug logging and exception context on public API functions.
- **`jit/`** — JIT generators:
  - `core.py` — `JitSpec` (static sources) and `JinjaJitSpec` (Jinja-rendered sources), `gen_jit_spec()`, `gen_jinja_jit_spec()`.
  - `templates.py` — Jinja2 rendering utilities (`get_template_env()`, `render_template()`).
  - `env.py` — Path constants including `OASR_TEMPLATE_DIR`, `OASR_GEN_SRC_DIR`.
  - Per-family modules: `gemm.py`, `conv.py`, `norm.py`, `activation.py`, `ctc_decoder.py`.
- **`decoder/`** — Python wrappers for the C++ decoders: `CtcGreedySearch`, `CtcPrefixBeamSearch`, `CtcWfstBeamSearch` (requires k2), `ContextGraph` (phrase boosting trie). Also exposes `k2_available` flag. Each wrapper lazily imports the compiled `_C` extension and delegates to a `_*Core` C++ object.
- **`cache/`** — Paged-memory streaming cache manager for chunk-by-chunk Conformer inference:
  - `CacheConfig` — master config (layers, heads, dims, chunk size, block size, pool capacity).
  - `BlockPool` — fixed-size paged KV pool; blocks are allocated/freed per stream.
  - `AttentionCacheManager` — per-stream paged KV cache; supports both dense (`commit`) and paged (`commit_chunk_paged`) write paths.
  - `CnnCacheManager` — per-stream depthwise-conv left-context cache.
  - `CtcStateCacheManager` — per-stream `GpuStreamingDecoder` / `StreamHandle` lifecycle.
  - `StreamContext` — unified handle tying all three managers; call `prepare_chunk()` → `get_att_caches()` / `get_cnn_cache()` → `commit_chunk[_paged]()` → `get_decoder().decode_chunk()` per chunk, then `free()`.
- **`testing/`** — `bench_gpu_time(fn, args, ...)` utility: CUDA-event timing with optional CUPTI fallback via `triton.testing.do_bench`; returns `(median_s, std_s)`.
- **`aot.py`** — Ahead-of-time compilation registration for all kernel families. Includes `gen_all_gemm_variants()` for systematic AOT variant enumeration.
- **`tune/`** — Autotuning framework: backend registry, profiler, cache, kernel configs (`TileConfig`). See `docs/autotuning.md` for design and API (`oasr.autotune()` context manager, `enable_autotune()`/`disable_autotune()` global toggles, persistent JSON cache).
- **`layers/`** — Thin `nn.Module`-style wrappers around functional API: `conv.py`, `linear.py`, `norm.py`, `attention/`, `rotary_embedding/`.
- **`models/conformer/`** — Conformer model (`model.py`), config dataclass (`config.py`), weight conversion utility (`convert.py`).
- **`utils/`** — `validation.py` (`@supported_compute_capability`, `@backend_requirement` decorators), `mappings.py` (dtype/enum helpers), `timer.py`.

### Binding pattern (FlashInfer-style)

Each kernel group follows the same pattern:
1. Kernel header in `include/oasr/<family>.cuh`.
2. TVM-FFI launcher in `csrc/<family>.cu`.
3. TVM-FFI JIT binding in `csrc/<family>_jit_binding.cu`.
4. JIT generator in `oasr/jit/<family>.py`.
5. Python functional API in `oasr/<family>.py`.
6. nn.Module wrapper in `oasr/layers/<family>.py`.
7. AOT registration in `oasr/aot.py`.

pybind11 bindings (`csrc/pybind/`) remain only for the CTC decoder (CPU-side C++).

## Developer skills (slash commands)

Two skill files provide step-by-step workflows for common tasks:

- **`/add-cuda-kernel`** — Full walkthrough for adding a new kernel family (CUDA header → csrc launcher → JIT binding → JIT generator → Python API → tests → AOT registration). Use this as the authoritative reference when adding kernels.
- **`/benchmark-kernel`** — Benchmarking guide for the unified CLI (`oasr_benchmark.py`), including testlist files, CSV output, and Nsight Compute profiling.

### Key patterns for new kernels

- **C++ convention**: output tensor is the **first parameter** in all TVM-FFI launcher functions.
- **Python compilation context**: `CompilationContext` (in `oasr/compilation_context.py`) detects GPU SMs at import time; pass `supported_major_versions=[...]` to `get_nvcc_flags_list()` for arch-restricted kernels.
- **Validation decorators** (`oasr.utils`): `@supported_compute_capability([80, 86, ...])` marks check functions with their supported SMs; `@backend_requirement(backend_checks={...}, common_check=fn)` wires validation into the Python API function and adds `.is_backend_supported()` / `.is_compute_capability_supported()` helpers.

## Key constraints

- Requires CUDA >= 11.8, CMake >= 3.18, Python >= 3.8.
- cuDNN is optional; some features are disabled if not found.
- C++ standard: C++17. CUDA flags include `--expt-relaxed-constexpr`, `--expt-extended-lambda`, `-O3`, `--use_fast_math`.
- The compiled `_C.so` lives inside the Python package at `oasr/`; do not import `oasr` before building.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CUDA_ARCHITECTURES` | Override SM targets for build (e.g., `80` or `80;86`) |
| `OASR_USE_FLASH_ATTENTION` | Set to `1` to enable Flash Attention support |
| `OASR_CUDA_ARCH_LIST` | Manual override for JIT CUDA architecture detection |
