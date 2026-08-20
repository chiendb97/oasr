# Kernel Layer — CUDA, JIT, and the Functional API

OASR exposes custom CUDA / CUTLASS kernels to Python through TVM-FFI JIT
compilation, in the style of FlashInfer. Nothing is linked ahead of time in the
default build: a kernel is compiled by `nvcc` on its first *call* and cached, so
`import oasr` works on a machine with no compiled extension at all.

This document covers the C++/CUDA layer, the JIT pipeline, and the Python
functional API. For the `nn.Module` layer that models are built from, see
[architecture.md § The layer waist](architecture.md#the-layer-waist). For a
step-by-step walkthrough of adding a kernel, use the `/add-cuda-kernel` skill.

## Layered design

```
Python functional API (oasr/<family>.py)  — @oasr_api decorated
    └── JIT generator (oasr/jit/<family>.py) → JitSpec / JinjaJitSpec
            └── TVM-FFI JIT binding (csrc/<family>_jit_binding.cu)
                    └── TVM-FFI launcher (csrc/<family>.cu)
                            └── Pure CUDA kernels (include/oasr/<family>.cuh)  — facade
                                    └── Config    (cutlass_*_configs.h)
                                    └── Template  (*_cutlass_template.h)
                                    └── Dispatch  (*_cutlass.h / *_dispatch.inc)
```

## The C++/CUDA layer

### Directory map

| Path | Contents |
|---|---|
| `include/oasr/common/` | Shared types (`types.h`), scalar/vector dtype conversion (`vec_dtypes.h`), warp/block reduction (`reduction.h`), SM dispatch (`arch_dispatch.h`), epilogue functors, and math utilities |
| `include/oasr/activation.cuh` | Vectorized exact GELU, sigmoid, tanh, ReLU, GLU, Swish, and Swoosh activations; unary sigmoid/tanh/ReLU also consume regular padded row strides such as channel chunks without a copy |
| `include/oasr/norm.cuh` + `norm_dispatch.inc` | LayerNorm, RMSNorm, fused add+LayerNorm/RMSNorm (with optional residual passthrough), BatchNorm1d, GroupNorm, fused norm+activation |
| `include/oasr/conv/` | `conv1d.cuh` + `conv1d_dispatch.inc` (depthwise with asymmetric padding and optional FSMN mask/residual fusion, pointwise, causal); dense BTC Conv1D is the height-one specialization of the `conv2d.cuh` CUTLASS facade |
| `include/oasr/pooling.cuh` | BTC AvgPool1D; vectorized 2×2 production specialization plus generic padding/ceil/count semantics |
| `include/oasr/recurrent/` | LSTM and tanh/ReLU RNN inference: fused GEMV/cohort kernels (`recurrent.cuh`), CUTLASS 2.x recurrent GEMM and state epilogues (`recurrent_cutlass.cuh`) with Stream-K/Split-K candidates, and the CUTLASS 3.x TMA warp-specialized path for SM90/SM100 (`recurrent_cutlass_sm90.cuh`) |
| `include/oasr/gemm/` | `gemm.cuh` facade, `bmm.cuh`, `group_gemm.cuh` |
| `include/oasr/{softmax,topk,fft,features}.cuh`, `sort/` | The remaining families |
| `include/oasr/ctc_decoder.cuh`, `include/oasr/wfst/` | GPU decoder kernels |
| `csrc/<family>.cu` | TVM-FFI launcher |
| `csrc/<family>_jit_binding.cu` | JIT binding exports |
| `csrc/tvm_ffi_utils.h` | DLPack dtype dispatch, validation macros (`CHECK_GEMM_ALIGNMENT`, `CHECK_CONTIGUOUS_INPUT`, `FLATTENED_ROWS`) |
| `csrc/templates/` | Jinja2 templates for config-specific CUTLASS instantiations (`gemm_cutlass_template.cu.jinja`, `bmm_cutlass_template.cu.jinja`, `group_gemm_cutlass_template.cu.jinja`) |
| `csrc/decoder/ctc/` | GPU CTC launcher + binding (`ctc_decoder.cu`, `ctc_decoder_jit_binding.cu`); `ctc/cpu/` holds the CPU-side C++ decoders compiled into `_C.so` — greedy search, prefix beam search, WFST beam search (via k2), the streaming WFST decoder, `ContextGraph` for phrase boosting, and shared `common/utils` |
| `csrc/decoder/wfst/` | In-tree GPU WFST decoder (TVM-FFI JIT). Its exact-semantics CPU reference oracle is **test-only** and lives separately under `csrc/tests/wfst/`, out of the production decoder library. |
| `csrc/pybind/` | pybind11 module for the CPU decoder bindings, the alignment bindings and legacy enums (`pybind_main.cpp`, `pybind_decoder.h`, `pybind_alignment.h`) |
| `csrc/alignment/` | The post-decode alignment pass in C++ — emission frames → per-token spans → words, plus `extract_beam_tokens` / `extract_beam_row`, which turn a padded `[batch, beam, max_len]` host tensor into nested lists.  Not a kernel: plain host data shuffling, compiled into `_C.so` |
| `csrc/tokenizers/` | The rendering half of the `symbol_table` tokenizer kind (`token_pieces`), which the word grouping calls once per finished hypothesis |

The GPU CTC launcher/binding pair is the **one** that does not live at the
`csrc/` root (its JIT generator is `oasr/jit/ctc_decoder.py`); everything else
follows the flat convention.

### The three-header CUTLASS pattern

Each CUTLASS kernel family splits config, template, and dispatch:

| Header | Purpose | Example (GEMM) |
|---|---|---|
| `cutlass_*_configs.h` | Config structs (`GemmConfig`), per-SM MMA traits (`SmMMATraits`), default configs (`DefaultGemmConfig`) | `gemm/cutlass_gemm_configs.h` |
| `*_cutlass_template.h` | CUTLASS kernel template parameterized by Config + MMATraits | `gemm/gemm_cutlass_template.h` |
| `*_cutlass.h` | Public dispatch interface (JIT mode via `OASR_TARGET_SM`, AOT mode via `OASR_DISPATCH_SM`) | `gemm/gemm_cutlass.h` |

Non-CUTLASS kernels (Conv1D, Norm, Activation) use `*_dispatch.inc` files with
VecSize / block_size dispatch macros instead.

### Dispatch modes

| Kernel family | Mode | Config source | Source generation |
|---|---|---|---|
| GEMM, BMM, GroupGEMM | **jinja** | `cutlass_gemm_configs.h` | Jinja renders `.cu` with baked-in config |
| Dense Conv1D / Conv2D | **jinja** | `cutlass_conv2d_configs.h` | Jinja renders `.cu` with baked-in config; each tactic exports strict BTC/KSC Conv1D and NHWC/KRSC Conv2D entry points |
| Depthwise / causal Conv1D | **dispatch** | `conv1d_dispatch.inc` | Direct compilation, VecSize macro |
| Grouped / depthwise Conv2D | **direct** | `grouped_conv2d.cuh` | NHWC 3×3/7×7 specializations; bias and optional activation share the convolution launch |
| Norm | **dispatch** | `norm_dispatch.inc` | Direct compilation, block/vec macro |
| Activation | **dispatch** | `activation_dispatch.inc` | Direct compilation, VecSize macro |
| Pooling | **direct** | `pooling.cuh` | 128-bit channel vectors in BTC layout; specialized 2×2 and generic launches |
| Recurrent | **direct + CUTLASS** | `recurrent/recurrent.cuh` + `recurrent/recurrent_cutlass{,_sm90}.cuh` | fused low-latency GEMV at small batch, shared-weight batch warps for cohorts, sequence-wide input projection, state epilogues, autotuned Stream-K/Split-K for wide states, and TMA warp-specialized collectives on SM90/SM100 |

- **JIT mode** (`OASR_TARGET_SM` defined): a single SM instantiation, with an
  optional `JitGemmConfig` / `JitConv2dConfig` passed via `-D` flags.
- **AOT mode** (no `OASR_TARGET_SM`): the `OASR_DISPATCH_SM` macro switches on
  the runtime SM version.

SM targets default to 70, 75, 80, 86, 89, 90, 100, 120 in `CMakeLists.txt`;
`setup.py` defaults to 70–90 only. Override either with `CUDA_ARCHITECTURES`.

### Recurrent execution paths

The CUDA FP16/BF16 recurrent waist retains two complementary implementations:

- `recurrent/recurrent.cuh` owns the latency path. A CTA computes a complete state row at
  batch 1, while the cohort specialization stages one output row's weights in
  shared memory for reuse by several batch warps. Affine accumulation, bias,
  gate activation, and state writes share a launch.
- `recurrent/recurrent_cutlass.cuh` owns the throughput path. The input affine is one
  sequence-wide OASR GEMM. LSTM weights are cached in `[hidden, gate, K]` order,
  so the four gates for a cell are adjacent; the recurrent CUTLASS epilogue can
  apply i/f/g/o and write h/c directly. Vanilla RNN uses the common tanh/ReLU
  CUTLASS epilogue and writes the hidden state directly. Matrix-C and output
  leading dimensions are independent, so BTC input projections feed the
  recurrence as strided batch rows without a transpose or intermediate copy.

The recurrent autotuner compares 16/32/64-row M tiles, Stream-K, and parallel
Split-K. LSTM also exposes serial Split-K: its custom epilogue delays the
nonlinear state transition until the final K partition and reuses GEMM's
self-restoring semaphore path, avoiding a workspace clear per timestep.
Parallel Split-K and
Stream-K materialize one interleaved LSTM gate tile before the state finalizer;
this costs one extra launch but is mathematically safe and can expose more SM
parallelism for a thin M and large K. Vanilla RNN excludes serial Split-K
because applying tanh/ReLU to an intermediate K partition is incorrect.

The direct kernel remains the default for decode-sized and moderate hidden
states. This is intentional: a tensor-core GEMM can underfill the device at
small M, and packing/projection plus dependent GEMM launches may cost more than
the work saved. `benchmarks/routines/recurrent.py` exposes
`native`, `cutlass16`, `cutlass32`, `cutlass64`, `streamk`, `splitk`, and
`serial_splitk` (LSTM only) arms so architecture-specific crossover changes
are measured instead of inferred. The focused matrix is
`benchmarks/testlists/recurrent_tactics.txt`.

**Architecture mapping.** Every target gets a CUTLASS 2.x composition, and
SM90/SM100 additionally get a 3.x one.

`recurrent_cutlass.cuh` holds the 2.x side. For FP16/BF16 that API specialises
exactly two arch tags, so every target maps onto one of them:

| JIT target | CUTLASS tag | MMA | Stages |
|---|---|---|---|
| 75 (Turing) | `Sm75` | `m16n8k8` | 2 |
| 80, 86, 89, 90, 100, 103, 120 | `Sm80` | `m16n8k16` | 3 |

Turing needs its own row twice over: a narrower MMA, and a `kernel::DefaultGemm`
that is specialised for a two-stage pipeline and no other stage count. Nothing
between the two rows is usable — `Sm86` has no 2.x tensor-op specialisation at
all, and `Sm89`/`Sm90` have one whose `DefaultGemmConfiguration` covers FP8
only — so Ada, Hopper and Blackwell all run the SM80 composition, which is
forward compatible and keeps one epilogue output-thread mapping across them.
Two `static_assert`s pin the arch/stage and arch/instruction-shape pairings, so
a remap that would issue an instruction the target lacks fails at compile time
instead of at decode.

`recurrent_cutlass_sm90.cuh` adds the CUTLASS 3.x collective path — TMA plus
wgmma on Hopper, tcgen05 on Blackwell datacenter — as two extra tactics,
`tma_64` (id 6) and `tma_128` (id 7). It is compiled only for targets 90 and
100, the two whose 3.x `OpClassTensorOp` builders accept FP16/BF16; SM120's is
restricted to F8/F6/F4, and no `CutlassArch` entry exists for 103. Everywhere
else the two ids are *refused* rather than rerouted, and the autotuner does not
offer them. Hopper's cooperative schedule `static_assert`s on an M tile below
128 rows, so the 64-row tile takes the pingpong schedule; SM100 selects from
`kSMs` and ignores the flag, which is why one pair of configs covers both.

On the 3.x path the LSTM is *decomposed*: the collective GEMM materialises one
gate-interleaved tile and the existing finalizer applies the state transition.
The fused custom epilogue cannot come along — it reconstructs logical
coordinates from a `PredicatedTileIterator` thread map, and 3.x replaced that
with cute layouts and an epilogue visitor tree, where a four-gates-to-one-cell
column reduction is not an elementwise node. The vanilla RNN has one gate, so
its nonlinearity stays fused in the collective epilogue.

The layer's *automatic* tensor-core selection still requires compute capability
8.0 (`oasr/layers/recurrent.py`), because the crossover was measured on Ampere
and later; on Turing the path is reachable through the functional API and the
autotuner.

### Conventions

- **The output tensor is the first parameter** of every TVM-FFI launcher.
- Launchers take N-D tensors and flatten with `FLATTENED_ROWS` rather than
  making Python call `reshape(-1, K)`.
- Output allocation stays in **Python** (`new_empty` varargs) — allocating in
  the C++ launcher was measured and is slower.
- Every GEMM-family launcher enforces the CUTLASS alignment-8 rule uniformly via
  `CHECK_GEMM_ALIGNMENT`, with a message naming the fix.

## The JIT pipeline

| Module | Role |
|---|---|
| `oasr/jit/core.py` | `JitSpec` (static sources) and `JinjaJitSpec` (Jinja-rendered), `gen_jit_spec()`, `gen_jinja_jit_spec()`, `build_and_load()` |
| `oasr/jit/templates.py` | Jinja2 rendering (`get_template_env()`, `render_template()`) |
| `oasr/jit/env.py` | Path constants (`OASR_TEMPLATE_DIR`, `OASR_GEN_SRC_DIR`), nvcc flags, `cutlass_version_stamp` |
| `oasr/jit/<family>.py` | Per-family generators: `gemm`, `conv`, `norm`, `activation`, `pooling`, `recurrent`, `softmax`, `topk`, `fft`, `features`, `ctc_decoder`, `wfst_decoder` |
| `oasr/jit/attention.py` | **Different model** — see below |
| `oasr/compilation_context.py` | `CompilationContext` detects GPU SMs at import time; pass `supported_major_versions=[...]` to `get_nvcc_flags_list()` for arch-restricted kernels |

Compiled modules are cached in `~/.cache/oasr/jit/`, keyed on a hash that covers
the sources, the `include/` tree, the nvcc flags, **and** the CUTLASS version
stamp.

`oasr/jit/attention.py` is not a Ninja JIT spec. It is a `functools.cache`-keyed
wrapper around `cutlass.cute.compile()`, exposing `select_backend()`,
`get_compiled_fmha(...)`, `warmup_fmha(...)`, `fmha_config_supported(...)` and
`set_backend_mode()`. `select_backend()` probes the device capability eagerly at
module load and resolves to `"cute"` on sm_80 / 86 / 89 / 120 when CuteDSL
imports cleanly, otherwise `"sdpa"`.

### CUTLASS

CUTLASS is the **`3rdparty/cutlass` git submodule, pinned to v4.6.1**.
`git submodule update --init` is what provides it — CMake fetches only pybind11.
Nothing links it: every CUTLASS kernel is JIT-compiled, so `oasr/jit/env.py`
hands the include directories to `nvcc` at runtime.

Its `version.h` is folded into the JIT cache key. Without that, `build_and_load`
short-circuits on an existing `.so` and a submodule bump keeps silently loading
binaries built against the old headers. **Editing a vendored CUTLASS header
without bumping the version still needs `rm -rf ~/.cache/oasr/jit`.**

The CuTeDSL half of CUTLASS is the separate `nvidia-cutlass-dsl` wheel
(`pip install -e .[attention]`, floor in `oasr/jit/attention.py::MIN_CUTEDSL_VERSION`),
kept at the same 4.6.1 release. It is **optional**: `OASR_ATTN_BACKEND=auto`
degrades `oasr.fmha` to SDPA when it is absent.

Evaluation of the 4.4.2 → 4.6.1 move: `.artifacts/cutlass_upgrade.md`.

## The Python functional API

Every entry point is `@oasr_api`-decorated (`oasr/api_logging.py` — debug logging
plus exception context), JIT-compiles on first call via `@functools.cache`,
allocates its output tensor, and calls into the compiled module.

| Module | Exposes |
|---|---|
| `oasr/functionals/gemm.py` | `gemm`, `bmm`, `group_gemm`, and the fused epilogues `gemm_activation` (RELU/tanh-GELU/exact-erf GELU/SWISH) and `gemm_log_softmax` (the CTC head fast path) |
| `oasr/functionals/gemm_torch.py` | Torch/cuBLAS runners — `torch_gemm`, `torch_gemm_activation`, `torch_bmm`, `torch_gemm_log_softmax` — mirroring the CUTLASS launcher contract exactly (output-first, in-place / CUDA-graph-safe, `D = A @ Bᵀ`). Doubles as a `Tactic("torch")` autotuner candidate and as the production dispatch target. Deliberately free of any `oasr.tune` import. |
| `oasr/functionals/norm.py` | `layer_norm`, `rms_norm`, `batch_norm1d`, `group_norm`, fused norm+activation |
| `oasr/functionals/conv.py` | dense / depthwise / pointwise / causal Conv1D; dense, grouped and depthwise Conv2D. Dense NHWC 1×1 Conv2D dispatches as GEMM; Conv1D depthwise padding may be an integer or `(left, right)` pair |
| `oasr/functionals/activation.py` | standalone exact-erf `gelu`, `sigmoid`, `tanh`, `relu`, `glu`, `swish`, `swoosh_l`, `swoosh_r` |
| `oasr/functionals/pooling.py` | BTC/TC `avg_pool1d`, including symmetric padding, ceil mode, and `count_include_pad` |
| `oasr/functionals/softmax.py`, `oasr/functionals/topk.py`, `oasr/functionals/fft.py` | `softmax`, `topk`, `rfft` / `rfft_power` |
| `oasr/functionals/feature.py` | `stft_frame`, `dct_lifter`, `fbank_preprocess`, `mel_log`, `whisper_logmel`, `lfr_gather` — see [features.md](features.md) |
| `oasr/functionals/attention.py` | `fmha(...)` and `fmha.persistent_inputs(...)` |
| `oasr/functionals/ctc_decode.py` | `ctc_beam_search_decode`, `GpuStreamingDecoder` — see [ctc_decoder_gpu.md](ctc_decoder_gpu.md) |
| `oasr/decode.py` | Thin helpers over the CPU-side `oasr.decoder` decoders |

`oasr/decoder/` holds the Python wrappers for the CPU-side C++ decoders —
`CtcGreedySearch`, `CtcPrefixBeamSearch`, `CtcWfstBeamSearch` (requires k2), and
`ContextGraph` (a phrase-boosting trie) — plus the `k2_available` flag. Each
lazily imports the compiled `_C` extension and delegates to a `_*Core` C++
object.

### Shape-aware backend selection

`gemm`, `gemm_activation`, `bmm` and `gemm_log_softmax` route per shape.
`jit.gemm.select_default_config(op, M, N, K)` picks:

- a CUTLASS variant — default tile, serial split-K, parallel split-K (`pk`), or
  Stream-K;
- the torch/cuBLAS backend (`oasr/functionals/gemm_torch.py`);
- or, for the CTC head only, the legacy single-call fused launcher.

The rules come from measured sweeps (`scripts/tune_asr_gemm.py`) and are keyed on
the exact `(op, N, K)`, so **the table is per model width**. A shape with no
rule falls through to the fixed `GEMM_DEFAULT` tile; the fall-through is counted
and reportable via `jit.gemm.rule_miss_report()` — which is both the coverage
check and the shape list to feed the tuner.

Two rules are structural rather than tuned:

- **`GEMM_MIN_ROWS`** — a row floor below which CUTLASS's M-tiling leaves most of
  every tile empty and cuBLAS's GEMV-shaped kernel wins.
- The dispatch decision is a **pure function of the call** and is deliberately
  *not* relaxed under CUDA-graph capture, even though dispatch cost is free
  there: a capture-dependent branch makes the graph pick a different kernel than
  eager, and the resulting one-ulp fp16 difference has produced different tokens.

`OASR_GEMM_HEURISTIC=0` disables the whole thing. Measurements and re-tuning
recipe: `.artifacts/gemm_tuning.md`.

### Fused attention

```python
oasr.fmha(q, k, v, *, softmax_scale, attn_bias, cache_seqlens, cache_seqstarts,
          block_table, out)
```

Three cache modes share one signature:

| Mode | `block_table` | `cache_seqlens` |
|---|---|---|
| Offline | `None` | `None` |
| Dense streaming (caller concatenated old + new K/V) | `None` | set |
| Paged streaming (K/V are pool views) | set | required |

It dispatches to either `_sdpa_reference` (PyTorch SDPA, fp32-friendly) or the
CuteDSL kernel (fp16/bf16 only). `oasr.fmha.persistent_inputs(...)` caches the
CuteDSL DLPack descriptors when the engine reuses the same tensors every call;
`validate=False` skips checks for proven inputs.

Routing policy and measurements: `.artifacts/fmha_tuning.md`.

### CuteDSL kernels (`oasr/kernels/`)

`oasr/kernels/` holds low-level implementations that do **not** use the TVM-FFI /
Ninja pipeline.

- `kernels/cute/attention/base.py` — abstract `FmhaBase` + `pick_arch_cls(major, minor)`
- `kernels/cute/attention/fmha_sm80.py` — `FmhaSm80`, covering sm_80 / 86 / 89
- `kernels/cute/attention/fmha_sm120.py` — `FmhaSm120`, a thin subclass for consumer Blackwell
- `kernels/cute/` — FlashAttention-style helpers: `block_info.py`, `seqlen_info.py`,
  `mask.py`, `softmax.py`, `tile_scheduler.py`, `pack_gqa.py`, `paged_kv.py`,
  `named_barrier.py`, `copy_utils.py`, `layout_utils.py`, `ampere_helpers.py`, `utils.py`

Each is compiled via `cutlass.cute.compile()` into a Python callable and cached
per config in `oasr/jit/attention.py::_compiled_fmha`.

## Utilities

`oasr/utils/`:

| Module | Contents |
|---|---|
| `validation.py` | `@supported_compute_capability([80, 86, ...])` marks a check function with the SMs it supports; `@backend_requirement(backend_checks={...}, common_check=fn)` wires validation into the public API function and adds `.is_backend_supported()` / `.is_compute_capability_supported()` helpers |
| `mappings.py` | dtype and enum helpers |
| `timer.py` | timing helpers |

`oasr/testing/bench_gpu_time(fn, args, ...)` is the measurement primitive: CUDA
event timing with an optional CUPTI fallback via `triton.testing.do_bench`,
returning `(median_s, std_s)`.

## Ahead-of-time compilation

`oasr/aot.py` registers every kernel family for AOT builds, including
`gen_all_gemm_variants()` for systematic variant enumeration. AOT is optional —
the default path is JIT-on-first-call.

## Autotuning

`oasr/tune/` is a separate mechanism from the shape-aware heuristic: a backend
registry, profiler, persistent JSON cache and `TileConfig` search, driven by the
`oasr.autotune()` context manager or the `enable_autotune()` / `disable_autotune()`
toggles. See [autotuning.md](autotuning.md).

## Adding a kernel family

Seven steps, in order:

1. Kernel header in `include/oasr/<family>.cuh`
2. TVM-FFI launcher in `csrc/<family>.cu`
3. TVM-FFI JIT binding in `csrc/<family>_jit_binding.cu`
4. JIT generator in `oasr/jit/<family>.py`
5. Python functional API in `oasr/<family>.py`
6. `nn.Module` wrapper in `oasr/layers/<family>.py`
7. AOT registration in `oasr/aot.py`

The `/add-cuda-kernel` skill (`.claude/skills/add-cuda-kernel/SKILL.md`) walks
through each with worked code. `/benchmark-kernel` covers measuring the result.

pybind11 bindings (`csrc/pybind/`) remain for the work that is **not** a kernel:
the CPU-side CTC/WFST decoders and the post-decode alignment pass. New *kernels*
do not use them — TVM-FFI JIT is the route for anything that runs on the device.

The distinction is about where the work runs, not how fast it is. `csrc/alignment/`
holds no CUDA at all; it is there because the pass runs on the engine's step-loop
thread, which holds the GIL for every request the engine finishes, and in Python
it cost more than the CTC decode it decorated. The beam read-back is the same
story on the *untimed* path: `out_lengths[b, k]` is a 0-d tensor plus an
`item()`, and a slice is another tensor, so materialising a 16-beam block row by
row cost more than the decode's own device→host copy.

Neither has a Python twin in the package — `oasr/engine/decode/alignment.py` is
marshalling only, because a fallback here is a slow path a deployment lands on
silently. Both files are in `OASR_SOURCES`, so a successful build always has
them and no call site checks. So these are the one part of the tree where
`test-cpu.yml`, which compiles nothing, cannot cover the implementation: the
rule is checked against a Python oracle kept inside `tests/test_alignment_cpp.py`
(exact agreement over randomised input and the whole Unicode plane), and that
file skips without the extension.
