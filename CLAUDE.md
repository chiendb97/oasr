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

# Install with serving extras (HTTP/WebSocket client libs for benchmarks)
pip install -e .[serving]

# Build the Rust serving frontend (binary used by the `oasr-server` console script)
cd rust && cargo build --release
```

The build compiles the `_C.so` pybind11 extension (for decoder + enums) via CMake. CUDA kernels are JIT-compiled on first use via TVM-FFI and cached in `~/.cache/oasr/jit/`. The Rust workspace under `rust/` is built separately with `cargo` — the `oasr-server` Python console script execs the resulting binary (resolved via `$OASR_RS_BIN`, `$PATH`, or `rust/target/release/oasr-server`).

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

# Run multi-thread engine stress tests (opt-in marker)
pytest tests/test_engine_concurrent.py -m concurrent -v
```

Tests live under `tests/`. Functional API tests follow a flat `tests/test_<kernel>.py` layout (FlashInfer convention). The conftest at `tests/conftest.py` provides fixtures: `device` (CUDA, skips if unavailable), `dtype`/`dtype_all` (FP32/FP16/BF16), `batch_seq_hidden` (common shape tuples). Default pytest options (`-v --tb=short`) are set in `pyproject.toml`. Registered markers: `slow` (long-running, skip with `-m 'not slow'`) and `concurrent` (multi-thread engine stress, opt-in).

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

# Engine-level benchmark with CUDA Graph capture of the streaming encoder
# (toggles EngineConfig.use_cuda_graphs; default is "on")
python benchmarks/bench_engine.py --cuda-graphs on        # captured (default)
python benchmarks/bench_engine.py --cuda-graphs off       # eager (apples-to-apples profiling)

# Profiling with Nsight Compute (NVTX markers via --profile)
ncu --set full -o gemm_profile python benchmarks/oasr_benchmark.py \
    --routine gemm --subroutine gemm --backends cutlass --profile --dry_run_iters 0
```

Legacy `bench_*.py` scripts still work as thin wrappers. See `benchmarks/README.md` for full CLI reference.

### Engine vs. service benchmarks

Two top-level perf harnesses pair up to measure the GPU ceiling (`bench_engine.py`)
and the end-to-end serving cost (`bench_service.py`). Both pick up defaults
from `.env` — copy `.env.example` to `.env`, edit, then
`set -a; source .env; set +a` to export.

| Env var | Becomes the default for |
|---|---|
| `CKPT_DIR`, `AUDIO_DIR` | `--ckpt-dir`, `--audio-dir` (expanded in shell, both scripts) |
| `OASR_RS_BIN` | Path to `oasr-server` (`bench_service.py` reads it directly to spawn the server) |
| `NUM_UTTERANCES` | `--num-utterances` (both scripts) |
| `MAX_BATCH_SIZE` | `--max-batch-size` (both scripts) |
| `CONCURRENCY` | `--concurrency` (`bench_service.py`) |
| `CHUNK_MS` | `--chunk-ms` (`bench_service.py`) |

CLI flag still wins when both are given. Templates — substitute the bracketed
placeholders, or drop the flag to pick up the matching `.env` default:

```bash
# Engine — pure GPU + Python, no IPC/HTTP
python benchmarks/bench_engine.py \
    --ckpt-dir [CKPT_DIR] \
    --audio-dir [AUDIO_DIR] \
    --subroutines [offline|streaming|offline_wfst|streaming_wfst] \
    --max-batch-size [MAX_BATCH_SIZE] \
    --num-utterances [NUM_UTTERANCES] \
    --chunk-size [CHUNK_SIZE] \
    --cuda-graphs [on|off]

# Service — Rust + HTTP + PyO3 dispatcher (auto-spawns oasr-server)
python benchmarks/bench_service.py \
    --ckpt-dir [CKPT_DIR] \
    --audio-dir [AUDIO_DIR] \
    --subroutines [offline|streaming|grpc_offline|grpc_streaming|whisper] \
    --num-utterances [NUM_UTTERANCES] \
    --concurrency [CONCURRENCY] \
    --max-batch-size [MAX_BATCH_SIZE] \
    --chunk-ms [CHUNK_MS] \
    --wire-encoding [f32_le|i16_le] \
    --realtime [0|1]
```

`--wire-encoding` (default `i16_le`) chooses the PCM format the bench client
sends; `oasr-asr::decode_raw_pcm` widens i16 back to f32 server-side.
`--service-mode` is auto-derived from `--subroutines`. Full recipe in
`docs/benchmarks.md`.

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
- **`attention.py`** — `fmha(q, k, v, *, softmax_scale, attn_bias, cache_seqlens, block_table, out)` fused multi-head attention. Three cache modes share one signature: **offline** (`block_table is None`, `cache_seqlens is None`), **dense streaming** (`block_table is None`, `cache_seqlens is not None` — caller concatenated old + new K/V), **paged streaming** (`block_table is not None`, `cache_seqlens` required — K/V are pool views). Dispatches via `oasr.jit.attention.select_backend()` to either `_sdpa_reference` (PyTorch SDPA fallback, fp32-friendly) or the CuteDSL kernel (fp16/bf16 only). Also exposes `oasr.fmha.persistent_inputs(...)` — a context manager that caches CuteDSL DLPack descriptors for the hot loop when the engine reuses the same Q/K/V/out/bias/block_table/cache_seqlens tensors every call. A `validate=False` fast path skips checks for proven inputs.
- **`softmax.py`**, **`topk.py`**, **`fft.py`**, **`feature.py`** — Additional `@oasr_api` functional entry points: `softmax`, `topk`, `rfft`/`rfft_power`, and feature-extraction primitives (`dct_lifter`, `fbank_preprocess`, `mel_log`). Same JIT-on-first-call pattern as the other top-level modules.
- **`kernels/`** — Low-level kernel implementations that **do not** use the TVM-FFI / Ninja JIT pipeline. `kernels/cute/attention/` holds the CuteDSL FMHA: `base.py` (abstract `FmhaBase` + `pick_arch_cls(major, minor)` dispatcher), `fmha_sm80.py` (`FmhaSm80` — covers sm_80 / sm_86 / sm_89), and `fmha_sm120.py` (`FmhaSm120`, a thin subclass over `FmhaSm80` for consumer Blackwell). `kernels/cute/` also contains FlashAttention-style helper modules used by these backends: `block_info.py`, `seqlen_info.py`, `mask.py`, `softmax.py`, `tile_scheduler.py`, `pack_gqa.py`, `paged_kv.py`, `named_barrier.py`, `copy_utils.py`, `layout_utils.py`, `ampere_helpers.py`, `utils.py`. Compiled via `cutlass.cute.compile()` returning a Python callable; cached per-config in `oasr/jit/attention.py::_compiled_fmha`.
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
  - `attention.py` — **Different model**: not a Ninja JIT spec but a `functools.cache`-keyed wrapper around `cutlass.cute.compile()`. Exposes `select_backend()`, `get_compiled_fmha(...)`, `warmup_fmha(...)`, and `set_backend_mode()` (mostly for tests; clears the compile cache). `select_backend()` probes the device capability eagerly at module load and resolves to `"cute"` on sm_80 / sm_86 / sm_89 / sm_120 (when CuteDSL imports cleanly), otherwise `"sdpa"`.
- **`decoder/`** — Python wrappers for the C++ decoders: `CtcGreedySearch`, `CtcPrefixBeamSearch`, `CtcWfstBeamSearch` (requires k2), `ContextGraph` (phrase boosting trie). Also exposes `k2_available` flag. Each wrapper lazily imports the compiled `_C` extension and delegates to a `_*Core` C++ object.
- **`engine/`** — Inference engine for offline + streaming Conformer-CTC on a single GPU:
  - `EngineConfig` — unified config aggregating model, cache, feature, decoding, detokenization settings. `use_cuda_graphs: bool = True` toggles CUDA Graph capture of the steady-state streaming encoder forward.
  - `ASREngine` — unified streaming + offline engine.  Step loop: schedule → batched GPU fbank ingest → encoder forward (length-bucketed offline micro-batches via `OfflinePipeline` overlap with one chunk per active streaming request) → CTC postprocess.  Handles offline + streaming requests in one pool; starvation bounded by `max_wait_time`.  Convenience helpers: `transcribe(...)` (streaming default) and `transcribe_offline(...)` (batched offline).
  - `graph_cache.py` — `EncoderGraphCache` lazily captures one `torch.cuda.CUDAGraph` per `(B, T_input, cache_t1_bucket)` shape, replays via pre-allocated input/output buffers. Persistent paging slots and `cnn_cache` are captured by **address**, so they must be allocated before the first capture and never reallocated. All captures share one CUDA Graph memory pool. Engine paths are slot-based (`forward_chunk_paged`) so the captured code path is stable across calls.
  - `Request` / `RequestOutput` / `RequestState` (`WAITING → RUNNING → FINISHED`).
  - Internal modules: `scheduler.py`, `model_runner.py`, `input_processor.py`, `output_processor.py`, plus the `pipeline/` package — `base.py` (`Pipeline` ABC), `offline.py` (`OfflinePipeline`: length-bucketed micro-batches, persistent cross-step collate producer with `submit`/`drain_ready`/`in_flight`), `streaming.py` (`StreamingPipeline`: chunk-by-chunk paged-KV). Service mode pinning: `EngineConfig.service_mode ∈ {"streaming","offline"}` selects exactly one pipeline per engine lifecycle; mismatched requests are rejected at admission.
- **`features/`** — Batched audio feature extraction (FBANK / MFCC):
  - `FeatureConfig` — shared config for sample rate, mel bins, frame length/shift, dither, etc.
  - `fbank_batch` / `mfcc_batch` / `extract_features_batch` — offline batch extraction over padded `(B, T)` or list of waveforms.
  - `BatchedStreamingFeatureExtractor` — `B` parallel chunked streams (`process_chunk` / `flush`).
  - Backends: `torchaudio.compliance.kaldi` (default) and optional `kaldifeat` GPU path. Batched FBANK/MFCC in `batched.py`.
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

### Companion docs (under `docs/`)

| File | Covers |
|------|--------|
| `docs/autotuning.md` | `oasr.tune` design, `oasr.autotune()` API, JSON cache format |
| `docs/benchmarks.md` | Engine vs. service bench recipes, `.env` workflow, RTF / latency interpretation |
| `docs/cache_manager.md` | `BlockPool` / `AttentionCacheManager` / `CnnCacheManager` / `StreamContext` semantics |
| `docs/ctc_decoder_gpu.md` | `GpuStreamingDecoder` single- vs. multi-request flows, paged-memory options |
| `docs/engine.md` | `ASREngine` step loop, batching, CUDA Graph capture |
| `docs/engine_concurrency.md` | Engine thread-safety (RLock), worker thread modes, multi-process scaling |
| `docs/scheduler.md` | Streaming + offline request scheduling, starvation bounds, micro-batching |
| `docs/serving.md` | Rust `oasr-server` frontend: HTTP/gRPC/WebSocket API, in-process PyO3 engine, wire format |
- **`layers/`** — Thin `nn.Module`-style wrappers around functional API: `conv.py`, `linear.py`, `norm.py`, `attention/`, `rotary_embedding/`.
- **`models/conformer/`** — Conformer model (`model.py`), config dataclass (`config.py`), weight conversion utility (`convert.py`).
- **`utils/`** — `validation.py` (`@supported_compute_capability`, `@backend_requirement` decorators), `mappings.py` (dtype/enum helpers), `timer.py`.
- **`serving/` (removed)** — The Python ZMQ worker (`engine_worker.py` / `ipc.py` / `server.py`) is gone. `oasr-server` now embeds the engine in-process via PyO3 (see the Rust serving section below). The `oasr` Python package is a runtime dependency of the Rust binary — it must be importable at the active Python interpreter the binary linked against.

### Rust serving frontend (`rust/`)

Cargo workspace that builds `oasr-server`, the binary the Python `oasr-server` console script execs. The binary hosts **one in-process Python `ASREngine` per process** via PyO3 (linked against `libpython` through `auto-initialize`) and serves it over HTTP + gRPC. Multi-GPU = launch N `oasr-server` processes behind a process manager, each with `CUDA_VISIBLE_DEVICES` set.

| Crate | Role |
|---|---|
| `oasr-wire` | Shared event/command types (`Cmd`, `Event`, `ErrorCode`, `ModelInfo`). Pure Rust — no codec / no IPC. |
| `oasr-engine-client` | PyO3-backed driver: `PyEngine` wrapper, `EngineDispatcher` thread that owns the GIL and drives `engine.step()`, `EngineClient`/`EnginePool` async facades. |
| `oasr-asr` | Audio decode (WAV via `hound`, raw PCM) to f32 mono `bytes::Bytes` |
| `oasr-server-http` | axum routes (Google STT v1-shaped REST): `POST /v1/speech:recognize`, `/healthz`, `/readyz`, `/metrics`, `/v1/models` |
| `oasr-server-grpc` | tonic `oasr.speech.v1.Speech` service (`Recognize` unary + `StreamingRecognize` bidi) plus the standard `grpc.health.v1.Health` service. Proto in `rust/proto/oasr_speech_v1.proto` |
| `oasr-server` | Binary: CLI, Python interpreter init, engine + HTTP + gRPC wiring. **One process per GPU**; spawn multiple `oasr-server` processes for multi-GPU. |

Routing policy: single in-process engine per `oasr-server` process — no sticky map needed at the pool level (the pool exists for symmetry with a future multi-engine-per-process layout). `/readyz` returns 200 once the dispatcher has taken its first tick. Build deps: a Python development install (PyO3 links against `libpython` via `auto-initialize`), `protobuf-compiler`, a C/C++ toolchain.

**Dispatcher (`oasr-engine-client::dispatcher`)** is the GIL-owning thread that drains commands from the tokio mpsc channel, replays them into Python (`add_request`/`feed_chunk`/`cancel`), runs `engine.step()`, and pushes the resulting events back via per-request channels. Key knobs (CLI flags on `oasr-server`):

| Flag | Default | Purpose |
|---|---|---|
| `--engine-label` | `engine` | tracing label |
| `--service-mode` | `streaming` | `streaming` or `offline` — pins the engine for its lifetime |
| `--max-concurrent-requests` | `256` | engine-side admission cap; over-cap admits emit `Event::Overloaded` |
| `--admit-window-ms` | `3` | wait up to N ms after first envelope for siblings before stepping (HTTP burst coalescing); `0` disables |
| `--admit-threshold` | `64` | stop coalescing early when this many envelopes drained |
| `--preferred-batch-sizes` | none | comma list, pre-warms CUDA-Graph capture per B |
| `--schedule-policy` | engine default (`bucket`) | `bucket` / `fcfs` / `sjf` |
| `--max-offline-pad-ratio` | engine default (`4.0`) | padded-waste cap for `bucket` policy |

Admission coalescing batches contiguous `CreateOffline`/`CreateStreaming` envelopes into one `add_requests_batch` Python call — turns 10–20-deep service batches into 32–64 under `asyncio.gather`-style bursts. `FeedChunk`/`Cancel`/`Ping` flush the admit batch first to preserve `CreateStreaming → FeedChunk` ordering. The Python-side `oasr/serving/` directory still exists but is dead code from the binary's perspective; `bench_service.py` rejects `--num-workers > 1` with a helpful error pointing at the new "one process per GPU" topology.

### Engine concurrency

`ASREngine` is **thread-safe** as of v0.1: every public entry (`add_request`, `add_streaming_request`, `feed_chunk`, `abort_request`, `step`, `run`, `num_running`, `num_waiting`, `transcribe`) acquires a process-wide re-entrant `threading.RLock`. Protects the scheduler queues (`_streaming_waiting`, `_offline_waiting`, `_running`, `_index` in `oasr/engine/scheduler.py`) and per-request audio mutation (`request.audio_chunks`, `request.audio_final`). The lock is coarse — `step()` holds it for the full 10–100 ms GPU-bound step — but the GIL serializes Python anyway and CUDA releases the GIL during forward. Under serving, the PyO3 dispatcher (`oasr-engine-client::dispatcher`) is the only Python caller and runs single-threaded; HTTP/gRPC handlers stay on tokio and never touch the GIL. Horizontal scale is **one process per GPU** — launch multiple `oasr-server` processes behind a process manager.

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
| `OASR_CUDA_ARCH_LIST` | Manual override for JIT CUDA architecture detection |
| `OASR_ATTN_BACKEND` | `sdpa` (force SDPA), `cute` (require CuteDSL FMHA, raise on unsupported arch or import failure), `auto` (default — use cute on sm_80 / sm_86 / sm_89 / sm_120 when CuteDSL imports, else warn + fall back to SDPA) |
| `OASR_RS_BIN` | Absolute path to the Rust `oasr-server` binary; overrides `$PATH` / `rust/target/release/` lookup used by the `oasr-server` console script |
| `OASR_USE_K2` | Set to `1` to build the WFST decoder (requires `pip install k2` and a k2 source tree at `K2_SOURCE_DIR` — default `/opt/k2-src`) |
