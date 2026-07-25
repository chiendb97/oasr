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

# Optional extras: [hub] (huggingface_hub + safetensors — Hub download & native
# checkpoint I/O), [tokenizers] (sentencepiece + tokenizers — TokenizerSpec
# kinds beyond symbol_table)
pip install -e .[hub,tokenizers]

# Optional: build the Rust serving frontend as a standalone binary
cd rust && cargo build --release
```

The build compiles three artifacts into the package: the `_C.so` pybind11 extension (decoder + enums) via CMake, and — via setuptools-rust (`[[tool.setuptools-rust.ext-modules]]` in `pyproject.toml`) — the `oasr._core` PyO3 extension module (the Rust serving core) plus the `oasr-server` console script that loads it. CUDA kernels are JIT-compiled on first use via TVM-FFI and cached in `~/.cache/oasr/jit/`. A Rust toolchain + `protobuf-compiler` must be on `PATH` at build time; with `--no-build-isolation`, `pip install "setuptools-rust>=1.10"` first. `setuptools-rust` runs cargo from the repo root, so the root `.cargo/config.toml` mirrors `rust/.cargo/config.toml`'s target-dir redirect (the repo's NFS mount breaks rustc autocfg probes). The same workspace still builds a standalone `rust/target/release/oasr-server` binary via plain `cargo build`.

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

### Rust workspace (`rust/`)

```bash
cd rust
cargo build --release      # builds default-members (incl. the oasr-server binary)
cargo test                 # tests default-members
cargo test -p oasr-asr     # tests a single crate
cargo fmt                  # rustfmt (run before committing Rust changes)
cargo clippy
```

**Do not run `cargo build/test --workspace`.** `oasr-core` (the `oasr._core`
extension module) enables `pyo3/extension-module` while `oasr-server` (binary)
enables `pyo3/auto-initialize` — those features are mutually exclusive and Cargo
unifies them per build, so building both crates in one invocation fails to
compile. `oasr-core` is therefore excluded from `default-members`; plain `cargo`
commands build the binary, and setuptools-rust builds `oasr-core` on its own
(`pip install` / `python setup.py build_rust --inplace`). Run cargo from `rust/`
(not the repo root) so `rust/.cargo/config.toml`'s target-dir redirect applies
— the repo's NFS mount otherwise breaks rustc autocfg probes.

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
    --subroutines [offline|grpc_offline|grpc_streaming|whisper] \
    --num-utterances [NUM_UTTERANCES] \
    --concurrency [CONCURRENCY] \
    --max-batch-size [MAX_BATCH_SIZE] \
    --chunk-ms [CHUNK_MS] \
    --wire-encoding [f32_le|i16_le] \
    --realtime [0|1] \
    --decoder-type [ctc_cuda|ctc_wfst] \
    --fst-path [/path/to/lang_bpe/HLG.pt]   # ctc_wfst only
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
- **`csrc/decoder/`** — decoder implementations, grouped by decode family:
  - `ctc/` — **GPU** CTC prefix-beam-search TVM-FFI launcher + JIT binding (`ctc_decoder.cu`, `ctc_decoder_jit_binding.cu`), JIT-compiled at runtime via `oasr/jit/ctc_decoder.py` (this is the one JIT launcher/binding pair that does **not** live at the `csrc/` root — see the binding pattern below). `ctc/cpu/` holds the **CPU-side** C++ decoders compiled into `_C.so` via CMake: CTC greedy search, prefix beam search, WFST beam search (via k2), streaming WFST decoder, `ContextGraph` for phrase boosting, and shared `common/utils`.
  - `wfst/` — in-tree **GPU WFST** beam-search decoder (TVM-FFI JIT). Its exact-semantics CPU reference oracle is test-only and lives in a separate JIT module under `csrc/tests/wfst/` (kept out of the production decoder library).
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
- **`activation.py`**, **`norm.py`**, **`conv.py`**, **`gemm.py`** — Functional API: `@oasr_api` decorated, JIT-compile kernels on first call via `@functools.cache`, allocate output tensors, call into compiled modules. `gemm.py` also exposes fused epilogues `gemm_activation` (RELU/GELU/SWISH) and `gemm_log_softmax` (the CTC head fast path), and does **shape-aware backend selection** for `gemm`/`gemm_activation`/`bmm`/`gemm_log_softmax`: `jit.gemm.select_default_config(...)` picks a per-shape CUTLASS variant (incl. serial split-K, parallel split-K "pk", Stream-K), the torch/cuBLAS backend, or (CTC head only) the legacy single-call fused launcher, via measured heuristic rules — falling through to the fixed default on any failure.
- **`gemm_torch.py`** — Torch/cuBLAS GEMM runners (`torch_gemm`, `torch_gemm_activation`, `torch_bmm`, `torch_gemm_log_softmax`) mirroring the CUTLASS launcher contract exactly (output-first, in-place/CUDA-graph-safe, `D = A @ Bᵀ`). Doubles as a `Tactic("torch")` autotuner candidate and the production dispatch target selected by `gemm.py`. Deliberately free of any `oasr.tune` import.
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
- **`engine/`** — Inference engine for offline + streaming ASR on a single GPU. Built around **one registry per extension axis** (see `docs/architecture.md`): decode family (`decode/`), streaming runtime (`streaming_backend/`), batching policy (`batching/`), plus the model + checkpoint registries in `oasr/models/`. New model architectures, decode families, streaming runtimes, batching policies, and checkpoint loaders plug in by subclass + register — no engine-core edits:
  - `EngineConfig` — unified config aggregating model, cache, feature, decoding, detokenization settings. `use_cuda_graphs: bool = True` toggles CUDA Graph capture of the steady-state streaming encoder forward.
  - `ASREngine` — unified streaming + offline engine.  Step loop: schedule → batched GPU fbank ingest → encoder forward (length-bucketed offline micro-batches via `OfflineExecutor` overlap with one chunk per active streaming request) → CTC postprocess.  Handles offline + streaming requests in one pool; starvation bounded by `max_wait_time`.  Convenience helpers: `transcribe(...)` (streaming default) and `transcribe_offline(...)` (batched offline).
  - `graph_cache.py` — `EncoderGraphCache` lazily captures one `torch.cuda.CUDAGraph` per `(B, T_input, cache_t1_bucket)` shape, replays via pre-allocated input/output buffers. Persistent paging slots and `cnn_cache` are captured by **address**, so they must be allocated before the first capture and never reallocated. All captures share one CUDA Graph memory pool. Engine paths are slot-based (`forward_chunk_paged`) so the captured code path is stable across calls.
  - `Request` / `RequestOutput` / `RequestState` (`WAITING → RUNNING → FINISHED`). `RequestOutput.timestamps` (optional) carries per-token `(start_s, end_s)` spans for the best hypothesis when the decode family produces alignments (Paraformer CIF).
  - Internal modules: `scheduler.py` (`Scheduler` — delegates offline batch **selection** to a `BatchingPolicy` and **partition** to a `PartitionPolicy`), `model_runner.py` (offline forward + delegates streaming to a `StreamingEncoderBackend`), `input_processor.py`, `output_processor.py`, plus the `executor/` package — `base.py` (`Executor` ABC), `offline.py` (`OfflineExecutor`: runs each scheduler-partitioned micro-batch back-to-back; flips the forward to the gapless varlen `forward_offline_packed` when `enable_sequence_packing` is set), `streaming.py` (`StreamingExecutor`: chunk-by-chunk). Service mode pinning: `EngineConfig.service_mode ∈ {"streaming","offline"}` selects exactly one executor per engine lifecycle; mismatched requests are rejected at admission.
  - `decode/` — pluggable `DecodeStrategy` (registry keyed on the resolved decode method — `EngineConfig.decode_method` if set (validated against `model.capabilities`), else `model.default_decode_type`; CTC further splits on `config.decoder_type`): `CtcGpuDecodeStrategy` / `CtcWfstDecodeStrategy` (`ctc_gpu.py` / `ctc_wfst.py`, own the CTC beam state), `TransducerDecodeStrategy` (`transducer.py`, `consumes="hidden"`), `CtcAedRescoringStrategy` (`rescoring.py`, `consumes="both"` — offline-only U2++ CTC n-best + one teacher-forced bitransformer pass + WeNet score fusion; knobs `rescoring_ctc_weight`/`rescoring_reverse_weight`), the incremental `AedDecodeStrategy` (`aed.py`, `incremental=True` greedy over the K2 protocol — Whisper), `ParaformerDecodeStrategy` (`paraformer.py`, `consumes="hidden"`, one-shot NAR: `model.predict` CIF → `model.nar_decode` parallel pass → argmax; CIF fire positions become per-token `RequestOutput.timestamps`), the incremental `LlmDecodeStrategy` (`llm.py`, `incremental=True` — speech-LLM: encodes the checkpoint's ChatML template via the tokenizer axis, splices the projected audio embeddings into a **left-padded** embedded prompt (variable per-row audio length), and emits **token-streaming partials** — one `finished=False` output per advanced request per tick; user prompt overridable via `EngineConfig.llm_prompt` or per-request `DecodingOptions.prompt`), and a shared `Detokenizer` (`detokenize.py`). `OutputProcessor` is a thin facade over the active strategy. Strategies declaring `incremental=True` implement `begin_offline`/`advance(StepBudget)`/`has_pending`: the offline executor prefills a micro-batch, parks the requests `RUNNING` in a pending pool, and runs ≤ `EngineConfig.decode_steps_per_tick` batched decoder steps per tick (admission gated by `max_decode_slots`) — bounded work per tick for the serving dispatcher. **Per-request `DecodingOptions`** (`oasr.engine.DecodingOptions`, on `Request.decoding`; dict-coercible at the PyO3 boundary): `n_best` (executor detokenizes top-N into `RequestOutput.nbest_texts` on finals — beam families), `max_new_tokens`, `temperature`/`top_k`/`top_p` sampling (AR families via `generation/sampling.py::select_next_tokens` — greedy stays a batched argmax fast path), `prompt` (LLM only, memoized suffix re-encode). AR finals carry `RequestOutput.finish_reason` (`"stop"`/`"length"`).
  - `generation/` — `StepBudget` (per-tick batched decoder-step allowance) + `Hypothesis` structs for the incremental AR strategies.
  - `streaming_backend/` — pluggable `StreamingEncoderBackend` (registry keyed on `encoder.streaming_kind`): `PagedStreamingBackend` (Conformer paged-KV + slot-CNN + CUDA graphs) and `StatefulStreamingBackend` (Zipformer-style per-layer recurrent state). The stateful backend **batches** ready streams: when the encoder exposes `stack_streaming_states`/`unstack_streaming_states` (Zipformer does), same-chunk-length streams run as one `B = N` forward — 3.5–24× over the sequential `B = 1` loop at pool sizes 4–32, argmax-identical (fp16 diffs are one-ulp batched-kernel reduction noise); encoders without the surface keep the sequential path. The engine reads streaming geometry (`decoding_window`/`stride`) from the backend, not hardcoded constants.
  - `batching/` — pluggable `BatchingPolicy` (`fcfs`/`bucket`/`sjf`) + `PartitionPolicy` (`count`/`frames`/`packing`) the scheduler delegates to.
- **`features/`** — Batched audio feature extraction (FBANK / MFCC):
  - `FeatureConfig` — shared config for sample rate, mel bins, frame length/shift, dither, etc.
  - `FeatureSpec` (`spec.py`) — **checkpoint-derived** feature description emitted by checkpoint converters (kind / sample rate / dim / frame geometry / LFR / window / normalize / audio_scale). The engine materializes `feature_config` from it unless the caller set one explicitly (mismatch logs a warning). `kind ∈ {"kaldi_fbank", "kaldi_mfcc", "whisper_logmel"}`; `raw` lands with its model package. `whisper.py::batched_whisper_logmel` implements the Whisper recipe (30 s pad/trim, n_fft 400 / hop 160, slaney mels, global max-norm; `audio_scale=1.0`), dispatched by `feature_type="whisper_logmel"`; it returns **real** per-row frame counts (`ceil(len/hop)`, HF attention-mask semantics) — Whisper ignores them, the Qwen2-Audio tower masks by them. The engine also adopts `FeatureSpec.audio_scale` unless the caller set a non-default `audio_scale` explicitly.
  - `lfr.py::apply_lfr_batch` — batched low-frame-rate stacking (FunASR/Paraformer: 80-mel LFR 7/6 → 560-dim at a 60 ms hop) as a clamped gather, bit-exact vs the FunASR reference loop. Driven by `FeatureConfig.lfr_m/lfr_n` (spec-emitted); `output_dim` folds the stacking in; applied in the offline `_fbank_batch` path only (streaming `prepare_streaming` rejects LFR configs). The fused `batched_fbank` kernel also gained the `hamming` window (FunASR frontends), so Paraformer collate stays on the fast path.
  - `fbank_batch` / `mfcc_batch` / `extract_features_batch` — offline batch extraction over padded `(B, T)` or list of waveforms.
  - `BatchedStreamingFeatureExtractor` — `B` parallel chunked streams (`process_chunk` / `flush`).
  - Backends: `torchaudio.compliance.kaldi` (default) and optional `kaldifeat` GPU path. Batched FBANK/MFCC in `batched.py`.
- **`tokenizers/`** — Tokenizer axis (sixth registry): `Tokenizer` ABC (`decode`/`encode`/`vocab_size`/`special_ids`) + `TokenizerSpec` (kind + asset files + options) emitted by checkpoint converters and built via `build_tokenizer(spec)`. Kinds: `symbol_table` (units.txt/tokens.txt — bit-compatible with the legacy `Detokenizer`, default `special_ids={0,1,2}`), `sentencepiece` (icefall bpe.model; ids == piece ids), `huggingface` (tokenizer.json; when the spec carries a `tokenizer_config` file, `added_tokens_decoder` entries missing from tokenizer.json are merged in declared-id order — Qwen2-Audio defines its audio/timestamp specials only there, and prompt encoding breaks without them), `whisper` (HF fast tokenizer + control-token stripping above `eot_id`), `funasr_char` (Paraformer tokens.json; decode ports FunASR's `sentence_postprocess` — CJK join, `@@` subword merge, `b b c`→`BBC` abbreviations — case-validated against funasr 1.3.14). `engine/decode/detokenize.py::Detokenizer` is now a thin adapter over this axis.
- **`checkpoints/`** — Checkpoint bundles + native format:
  - `bundle.py` — `ConvertedCheckpoint` (config, aux, weights, `TokenizerSpec`, `FeatureSpec`, `DecodingDefaults`, `source_format`); `convert_checkpoint()` adapts legacy 4-method converters (fills metadata via the historical path sniffing).
  - `native.py` — round-trippable native format: `oasr_config.json` (format_version 1) + `model.safetensors` (post-`load_weights` state dict — loads via strict `load_state_dict`, no name mapping; model-declared `_computed_buffer_suffixes` like Conformer's `pos_enc.pe` are skipped) + `tokenizer/` assets. Native state dicts get the model's own `_metadata` stamped at load so version-gated `_load_from_state_dict` remaps don't re-fire.
  - `convert.py` — `oasr-convert <src> <dst>` CLI (also `python -m oasr.checkpoints.convert`) materializing any supported dir as a native bundle.
- **`cache/`** — Paged-memory streaming cache manager for chunk-by-chunk Conformer inference:
  - `CacheConfig` — master config (layers, heads, dims, chunk size, block size, pool capacity).
  - `BlockPool` — fixed-size paged KV pool; blocks are allocated/freed per stream.
  - `AttentionCacheManager` — per-stream paged KV cache; supports both dense (`commit`) and paged (`commit_chunk_paged`) write paths.
  - `CnnCacheManager` — per-stream depthwise-conv left-context cache.
  - `CtcStateCacheManager` — per-stream `GpuStreamingDecoder` / `StreamHandle` lifecycle.
  - `DecoderKVCacheManager` (`decoder_kv.py`) — decoder-side KV for AR generation (keystone K6): a **separate** decoder-shaped `BlockPool`, append-per-step growth, per-request block tables, **no eviction**; `create(prefill_len)`/`append_step`/`block_tables`/`cache_seqlens`/`free`. Tested standalone; the `aed` strategy currently uses dense per-request KV (paged integration is the planned optimization).
  - `StreamContext` — unified handle tying all three managers; call `prepare_chunk()` → `get_att_caches()` / `get_cnn_cache()` → `commit_chunk[_paged]()` → `get_decoder().decode_chunk()` per chunk, then `free()`.
- **`testing/`** — `bench_gpu_time(fn, args, ...)` utility: CUDA-event timing with optional CUPTI fallback via `triton.testing.do_bench`; returns `(median_s, std_s)`.
- **`aot.py`** — Ahead-of-time compilation registration for all kernel families. Includes `gen_all_gemm_variants()` for systematic AOT variant enumeration.
- **`tune/`** — Autotuning framework: backend registry, profiler, cache, kernel configs (`TileConfig`). See `docs/autotuning.md` for design and API (`oasr.autotune()` context manager, `enable_autotune()`/`disable_autotune()` global toggles, persistent JSON cache).

### Companion docs (under `docs/`)

| File | Covers |
|------|--------|
| `docs/architecture.md` | One-registry-per-extension-axis design (decode family / streaming backend / batching policy / model / checkpoint) |
| `docs/autotuning.md` | `oasr.tune` design, `oasr.autotune()` API, JSON cache format |
| `docs/benchmarks.md` | Engine vs. service bench recipes, `.env` workflow, RTF / latency interpretation |
| `docs/cache_manager.md` | `BlockPool` / `AttentionCacheManager` / `CnnCacheManager` / `StreamContext` semantics |
| `docs/checkpoints.md` | Checkpoint resolution precedence, converter contract, native format, `LoadReport` discipline |
| `docs/ctc_decoder_gpu.md` | `GpuStreamingDecoder` single- vs. multi-request flows, paged-memory options |
| `docs/engine.md` | `ASREngine` step loop, batching, CUDA Graph capture |
| `docs/scheduler.md` | Streaming + offline request scheduling, starvation bounds, micro-batching |
| `docs/serving.md` | Rust `oasr-server` frontend: HTTP/gRPC/WebSocket API, in-process PyO3 engine, wire format, per-request decoding options |
| `docs/tokenizers.md` | Tokenizer axis: kinds, `TokenizerSpec`, the tokenizer_config added-token merge |
| `docs/wfst_decoder_gpu.md` | In-tree GPU WFST decoder: k2-parity semantics, kernel pipeline, graph image, lazy-commit memory model, TVM-FFI API, streaming |
- **`layers/`** — Thin `nn.Module`-style wrappers around functional API: `conv.py`, `linear.py`, `norm.py`, `feature.py`, `softmax.py`, `topk.py`, `attention/`, `rotary_embedding/`, plus `ctc.py` (`CtcProjection` — fused `log_softmax(x @ Wᵀ + b)` via `oasr.gemm_log_softmax`, used by `CTCHead`).
- **`models/`** — Architecture-agnostic model layer (vLLM/SGLang-style). `base.py` (`BaseAsrModel`/`BaseEncoder`/`BaseHead`/`CacheSpec`/`DecodeType`/`LoadReport`): the engine touches a model only through `encode_offline`/`encode_chunk_paged` (raw hidden) + fused `forward_offline`/`forward_chunk_paged` (encoder+head → log-probs, the CTC fast path), `cache_spec`, `decode_type`, and `encoder.streaming_kind`/`subsampling_rate`/`right_context`. `registry.py` (`register_model`, `build_model_from_checkpoint`, `load_checkpoint_bundle`, `CheckpointConverter`) + `loaders.py` (`from_pretrained` — local dir or HuggingFace Hub id, exposed as `oasr.from_pretrained`; `load_pretrained` additionally returns the tokenizer/feature/decoding specs + `LoadReport`, and is what `ASREngine` uses). Checkpoint resolution precedence: **native format** (`oasr_config.json`) → explicit `architecture=` → converter `detect()` (exactly one match; multiple → error; zero → deprecated `"conformer"` fallback with `DeprecationWarning`). `load_weights` returns a `LoadReport{mapped,dropped,missing}` — the registry warns on dropped weights per the converter's `expected_unused_prefixes` (silent, e.g. icefall `simple_*_proj`) / `capability_drop_hints` (named warning, e.g. the U2++ `decoder.*` rescoring branch). `decoders/` holds the autoregressive `BaseDecoder`/`PredictionNetwork`/`Joiner` contracts (transducer/AED/LLM extension points). `heads/ctc.py` (`CTCHead`).
  - **`models/conformer/`** — Conformer (`model.py`, `config.py`), `WenetConverter` (`convert.py`); `streaming_kind="paged"`, supports sequence packing. U2/U2++ dirs whose `train.yaml` declares a `(bi)transformer` decoder get the AED branch loaded as `self.decoder` (`decoder.left_decoder.*` keys 1:1; plain-`transformer` `decoder.decoders.*` keys remapped into `left_decoder`) → `capabilities={"ctc","ctc_aed_rescoring"}`, `default_decode_type="ctc"` (rescoring is opt-in via `EngineConfig.decode_method`). The decoder config (incl. `sos/eos` = raw vocab − 1 and the trained `reverse_weight`) lives in `ConformerModelConfig.decoder`.
  - **`models/zipformer/`** — Zipformer CTC ported from icefall (`model.py`, `encoder.py`, `subsampling.py`, `scaling.py`, `config.py`), `IcefallConverter` (infers config from checkpoint shapes); `streaming_kind="stateful"` (its own per-layer recurrent cache). `ZipformerEncoder.stack_streaming_states`/`unstack_streaming_states` declare the per-kind state batch dims (icefall's `streaming_decode.py` convention: embed + conv caches dim 0, key/nonlin/value caches dim 1) so the stateful backend can batch streams.
  - **`models/decoders/`** — AR decoder contracts (`base.py`) plus `transformer_decoder.py`: WeNet/ESPnet-compatible `TransformerDecoder`/`BiTransformerDecoder` (state-dict keys mirror WeNet's `embed.0.weight`/`decoders.N.self_attn.linear_q.*` layout — U2++ `decoder.*` weights load 1:1; verified ~6e-6 vs the upstream WeNet implementation), `add_sos_eos`/`reverse_pad_list` helpers, and an incremental `forward_one_step`.
  - **`models/whisper/`** — HF-format Whisper (`model.py` encoder/decoder with HF key layout + a batched incremental `prefill`/`step`/`select` decoder surface, `config.py` incl. generation-control ids, `convert.py` `HFWhisperConverter` — auto-detects `config.json: model_type=whisper`, reads `model.safetensors` + `generation_config.json`, emits `TokenizerSpec(kind="whisper")` + `FeatureSpec(kind="whisper_logmel", audio_scale=1.0)`). Offline-only (`streaming_kind="none"`, streaming rejected at engine init — a general gate: any `streaming_kind="none"` model is refused in streaming service mode); decoded by the incremental `aed` strategy (greedy; suppress/begin-suppress lists + SOT prompt from the config). Native format round-trips.
  - **`models/paraformer/`** — FunASR Paraformer, non-autoregressive (`encoder.py` SANM encoder — FSMN-memory self-attention, 560→512 first layer, sinusoidal PE, **LayerNorm eps 1e-12** (ESPnet convention — parity breaks at PyTorch's 1e-5); `predictor.py` `CifPredictor` — CifPredictorV2 with the vectorized `cif_v1` prefix-sum integrate-and-fire, always fp32; `decoder.py` SANM NAR decoder — FFN-first layer order, FSMN-only self-attn, one parallel pass over the CIF acoustic embeddings; `convert.py` `FunASRParaformerConverter` — auto-detects `config.yaml: model: Paraformer`, parses `am.mvn` CMVN into synthetic `encoder.cmvn_shift/scale` state-dict buffers so native round-trip is automatic, emits `TokenizerSpec(kind="funasr_char")` + `FeatureSpec(kaldi_fbank, hamming, LFR 7/6, audio_scale=32768)`). Offline-only; registered `"paraformer"`; parity vs FunASR 1.3.14 on `paraformer-zh`: encoder ≤2e-5, CIF fires bit-exact, token ids + transcript exact (icefall's detect now yields to dirs holding a FunASR `config.yaml` — `model.pt` alone used to over-claim).
  - **`models/speech_llm/`** — Qwen2-Audio-style LLM-ASR (`audio_tower.py` Whisper-geometry encoder ×32 d=1280 with **key-padding-only** attention mask + `AvgPool1d(2)` + post-pool LayerNorm — valid lengths follow HF's two-stage formula `(mel−1)//2+1` then `(feat−2)//2+1`; `llm.py` `Qwen2Lm` — faithful Qwen2 causal LM (fp32-variance RMSNorm, GPT-NeoX rotary, GQA, biased QKV, SiLU MLP) exposing `prefill(inputs_embeds, valid[, capacity])`/`step`/`select` for **left-padded variable-length** prompts (per-row positions = `cumsum(mask)−1`, HF masked-generate convention); with `capacity` (the `llm` strategy passes prompt + generation cap) the per-layer K/V buffers are preallocated and each step writes its slot in place — no per-step `torch.cat` cache re-copy (17% faster 7B step at B=8, token-identical; overflow degrades to cat-growth). Paged decoder-KV via the CuteDSL FMHA stays blocked on the masked-tile NaN fix (left-padded prompts are the failing key-padded shape), so attention is SDPA; `model.py` module names mirror HF 1:1 (`audio_tower`/`multi_modal_projector`/`language_model`) — `load_weights` normalizes 4.x (`language_model.model.*`) and 5.x-resave (`language_model.model.model.*`) key layouts; `convert.py` `HFQwen2AudioConverter` — auto-detects `config.json: model_type=qwen2_audio`, fills omitted `text_config` fields from `Qwen2Config` defaults (the published 7B relies on them), reads sharded safetensors, emits `TokenizerSpec(kind="huggingface")` incl. tokenizer_config.json + `FeatureSpec(whisper_logmel, 128 mels)`). Offline-only; registered `"speech_llm"`; parity: fp32 tiny fixture token-exact vs `transformers` greedy (B=1 + batched left-padded), real 7B bf16 same prefill argmax/top-5, LJ transcripts content-identical. Checkpoint bundles load host-side (`map_location="cpu"`) with dtype cast **before** the device move — an 8.4B checkpoint double-booked on the GPU otherwise.
  - **`models/transducer/`** — RNNT model (`model.py` `TransducerModel`, `decoder.py` stateless predictor, `joiner.py`, `config.py`, `convert.py` `IcefallTransducerConverter`); `decode_type="transducer"`, registered as `"transducer"`. `encoder_type ∈ {"conformer", "zipformer"}` selects the acoustic front-end (streaming follows the encoder: paged vs stateful). **Not auto-detected** — icefall dirs sniff as `zipformer` (CTC) and hybrid checkpoints carry both branches; load with `from_pretrained(dir, architecture="transducer")` (config shape-inferred from `decoder.*`/`joiner.*`; `simple_*_proj` declared-dropped, `ctc_output.*` a named capability hint). The matching `TransducerDecodeStrategy` (`engine/decode/transducer.py`, `consumes="hidden"`) implements offline **and** streaming greedy: one vectorized frame-sync loop, per-request sessions (label window + predictor projection + hypothesis) threaded across chunks; `EngineConfig.transducer_max_sym_per_frame` caps per-frame emissions. Streaming runs through the consumes-aware backends: strategies declaring `consumes="hidden"` get the encoder-only chunk forward (`encode_chunk_paged` / `encoder.streaming_forward`) **eagerly** (no CUDA-graph capture; the captured graph bakes in the fused-head output buffer), resolved at engine init via `get_decode_strategy_class(...)` before `ModelRunner` construction.
- **`utils/`** — `validation.py` (`@supported_compute_capability`, `@backend_requirement` decorators), `mappings.py` (dtype/enum helpers), `timer.py`.
- **`serving/` (removed)** — The Python ZMQ worker (`engine_worker.py` / `ipc.py` / `server.py`) is gone. The serving front-end now embeds the engine in-process via PyO3 (see the Rust serving section below). The `oasr` Python package is a runtime dependency of the front-end — it must be importable at the active Python interpreter. The thin `oasr/_server_cli.py` (the `oasr-server` console-script entry point) just forwards `sys.argv` into `oasr._core.serve`.

### Rust serving frontend (`rust/`)

Cargo workspace that builds the OASR serving core. It ships **two ways** from one shared code path: as `oasr._core`, a PyO3 extension module built by setuptools-rust during `pip install` and run via the `oasr-server` console script (`oasr._server_cli:main`); and as a standalone `oasr-server` binary via `cargo build`. Either hosts **one in-process Python `ASREngine` per process** via PyO3 and serves it over HTTP + gRPC. Multi-GPU = launch N processes behind a process manager, each with `CUDA_VISIBLE_DEVICES` set.

The PyO3 linkage mode is the key split: the binary embeds + links libpython (`pyo3/auto-initialize`), while the extension module is loaded by the host interpreter (`pyo3/extension-module`). These features are mutually exclusive and Cargo unifies features per build, so the shared serving logic lives in `oasr-serve` (mode-agnostic) and the two front-end crates select the mode via `oasr-engine-client`'s `auto-initialize` / `extension-module` features. `oasr-core` is excluded from `default-members` so a plain `cargo build` produces the binary; setuptools-rust builds `oasr-core` on its own.

| Crate | Role |
|---|---|
| `oasr-wire` | Shared event/command types (`Cmd`, `Event`, `ErrorCode`, `ModelInfo`). Pure Rust — no codec / no IPC. |
| `oasr-engine-client` | PyO3-backed driver: `PyEngine` wrapper, `EngineDispatcher` thread that owns the GIL and drives `engine.step()`, `EngineClient`/`EnginePool` async facades. Exposes `auto-initialize` / `extension-module` features forwarding to pyo3. |
| `oasr-asr` | Audio decode (WAV via `hound`, raw PCM) to f32 mono `bytes::Bytes` |
| `oasr-server-http` | axum routes (Google STT v1-shaped REST): `POST /v1/speech:recognize`, `/healthz`, `/readyz`, `/metrics`, `/v1/models` |
| `oasr-server-grpc` | tonic `oasr.speech.v1.Speech` service (`Recognize` unary + `StreamingRecognize` bidi) plus the standard `grpc.health.v1.Health` service. Proto in `rust/proto/oasr_speech_v1.proto` |
| `oasr-serve` | Mode-agnostic serving core: `Cli` + `run(cli)` (builds the engine, tokio runtime, HTTP + gRPC listeners). Shared by the binary and the extension module. |
| `oasr-server` | Standalone binary: thin `main.rs` → `oasr_serve::run`; pulls `oasr-engine-client` with `auto-initialize`. **One process per GPU.** |
| `oasr-core` | cdylib `oasr._core` PyO3 module: `#[pymodule]` exposing `serve(argv)` → `oasr_serve::run` under `allow_threads`; pulls `oasr-engine-client` with `extension-module`. Built by setuptools-rust. |

Routing policy: single in-process engine per process — no sticky map needed at the pool level (the pool exists for symmetry with a future multi-engine-per-process layout). `/readyz` returns 200 once the dispatcher has taken its first tick. Build deps: a Python development install (PyO3 links libpython), `protobuf-compiler`, a C/C++ toolchain.

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
| `--decoder-type` | engine default (`ctc_cuda`) | `ctc_cuda` / `ctc_wfst` (in-tree GPU WFST) |
| `--fst-path` | none | WFST graph for `ctc_wfst`: prebuilt `.img` or k2 `HLG.pt` (words.txt beside it = word table) |
| `--decode-method` | model default | capability selection (e.g. `ctc_aed_rescoring`, `llm`); validated against `model.capabilities` at startup |
| `--llm-prompt` | checkpoint default | speech-LLM user prompt (per-request `prompt` options override) |
| `--max-new-tokens` | engine default (448) | AR generation cap per request |
| `--decode-steps-per-tick` | engine default (32) | bounded batched AR decoder steps per engine tick |
| `--max-decode-slots` | `max_batch_size` | in-flight AR request cap before admission pauses |

Per-request decoding options travel the whole stack: proto `RecognitionConfig` extensions (`max_alternatives` honored + fields 101–105: `max_new_tokens`/`temperature`/`top_k`/`top_p`/`prompt`) and matching HTTP query params → `oasr_wire::DecodingParams` on `Cmd::Create*` → a `decoding` dict kwarg into `add_request[s_batch]` → `oasr.engine.DecodingOptions`. `Event::Final` carries `nbest_texts` (engine-detokenized alternatives) + `end_time_s` (last CIF timestamp) → proto `result_end_time`; `confidence` is the softmax-normalized posterior among the returned n-best scores (`oasr_wire::score_posteriors`; 0.0 when a family emits a single hypothesis).

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
| `OASR_RS_BIN` | Absolute path to an `oasr-server` executable; overrides the `oasr-server`-on-`$PATH` / `rust/target/release/` lookup used by `bench_service.py` |
| `OASR_USE_K2` | Set to `1` to build the WFST decoder (requires `pip install k2` and a k2 source tree at `K2_SOURCE_DIR` — default `/opt/k2-src`) |
| `OASR_CTC_FUSED` | Set to `0` to force the legacy multi-kernel CTC beam-search step (compiles a separate `ctc_decoder_legacy` JIT module). Default: fused single-kernel step for `beam <= 32` (~2.3–3× faster; see `docs/ctc_decoder_gpu.md` §3.4.6). Rollback / A/B-parity switch — set before process start |
| `OASR_GEMM_HEURISTIC` | Set to `0` to disable shape-aware GEMM backend selection (every shape → the fixed `GEMM_DEFAULT` tile). Rollback / A/B-parity switch |
| `OASR_GEMM_STREAMK` | Set to `0` to skip compiling the Stream-K GEMM variants (leaner build; they stay out of the autotune candidate space) |
| `OASR_GEMM_SPLITK_PARALLEL` | Set to `0` to skip compiling the parallel split-K (`GemmSplitKParallel`) GEMM variants |
| `OASR_GEMM_WS_CACHE` | Set to `0` to disable the persistent split-K/Stream-K workspace cache and fall back to per-call `cudaMallocAsync` + memset workspaces |
