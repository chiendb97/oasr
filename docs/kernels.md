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
| `include/oasr/pooling.cuh` | BTC AvgPool1D and MaxPool1D; vectorized 2×2 specialization, generic padding/ceil/count semantics, and a flat narrow-channel path for one-channel traces |
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
| Dense Conv1D / Conv2D | **jinja** | `cutlass_conv2d_configs.h` | Jinja renders `.cu` with baked-in config; each tactic exports strict BTC/KSC Conv1D and NHWC/KRSC Conv2D entry points. SM75–89 and SM120 use the CUTLASS 2.x implicit GEMM; SM90/SM100 use the 3.x `conv::CollectiveBuilder` — see [Conv2D on SM90/SM100](#conv2d-on-sm90--sm100) for why that config is *not* a copy of the GEMM one |
| Depthwise / causal Conv1D | **dispatch** | `conv1d_dispatch.inc` | Direct compilation, VecSize macro |
| Grouped / depthwise Conv2D | **direct** | `grouped_conv2d.cuh` | NHWC 3×3/7×7 specializations; bias and optional activation share the convolution launch |
| Norm | **dispatch** | `norm_dispatch.inc` | Direct compilation, block/vec macro |
| Activation | **dispatch** | `activation_dispatch.inc` | Direct compilation, VecSize macro |
| Pooling | **direct** | `pooling.cuh` | 128-bit channel vectors in BTC layout; specialized 2×2, generic, and narrow-channel launches |
| Recurrent | **direct + CUTLASS** | `recurrent/recurrent.cuh` + `recurrent/recurrent_cutlass{,_sm90}.cuh` | fused low-latency GEMV at small batch, shared-weight batch warps for cohorts, sequence-wide input projection, state epilogues, autotuned Stream-K/Split-K for wide states, and TMA warp-specialized collectives on SM90/SM100 |

- **JIT mode** (`OASR_TARGET_SM` defined): a single SM instantiation, with an
  optional `JitGemmConfig` / `JitConv2dConfig` passed via `-D` flags.
- **AOT mode** (no `OASR_TARGET_SM`): the `OASR_DISPATCH_SM` macro switches on
  the runtime SM version.

SM targets default to 75, 80, 86, 89, 90, 100, 120 in `CMakeLists.txt`;
`setup.py` queries the host GPUs and falls back to 80;86;89;90. Override either
with `CUDA_ARCHITECTURES`.

### Conv2D on SM90 / SM100

The CUTLASS 3.x implicit-GEMM conv builder is close enough to the GEMM one to
invite a copy, and far enough away that the copy does not compile. Four
differences, each load-bearing:

| | GEMM | Conv |
|---|---|---|
| K mode of `TileShape` | flat `Int<BK>` | **nested** `Shape<Int<BK>>` — implicit GEMM's K axis is the filter's (C, S, R) modes |
| Mainloop schedule | pingpong / cooperative | `conv::KernelScheduleAuto`; SM90 conv has **one** schedule (CUTLASS's own auto-selector has the cooperative branch commented out, and `conv/dispatch_policy.hpp` `static_assert`s on the persistent tags) |
| 2-SM SM100 atom | M tile doubled | M tile passed as-is; the atom comes from the **cluster**, and `BM * 2` is `"Invalid TileShape_M."` |
| `Arguments` | leading `GemmUniversalMode` | the mode-less constructor — SM90's `ConvUniversal` inherits `GemmUniversal` and has the mode, SM100's does not |

Two tile rules bound the space, both measured by compiling the grid rather than
documented upstream, and both enforced as filters in `oasr/jit/conv.py` because
a single unbuildable variant fails the whole JIT module:

- **SM90** — `(BM + BN) * BK * 2 ≤ 106,496 B`. `StageCountAutoCarveout` has to fit
  two stages plus the epilogue into Hopper's 227 KiB; above that it resolves to
  zero stages and the build stops.
- **SM100** — a 256-row tile requires `cluster_m == 2`.

The tile ladders themselves are **unmeasured**: they span the M range at two N
widths. They are also sized against build cost — each 3.x conv translation unit
peaks near 3.7 GB in `cicc`, and `run_ninja` only bounds parallelism when
`MAX_JOBS` is set.

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
the work saved. `benchmarks/kernels/recurrent.py` exposes
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
| `oasr/jit/fmha.py` | Fused attention, C++ CUTLASS/CuTe lane — one module per `(sm, dtype, head_dim)` **cell**, holding all 17 feature variants |
| `oasr/jit/attention.py` | The backend arbiter, **and** the CuTeDSL lane — different model, see below |
| `oasr/compilation_context.py` | `CompilationContext` detects GPU SMs at import time; pass `supported_major_versions=[...]` to `get_nvcc_flags_list()` for arch-restricted kernels |

Compiled modules are cached in `~/.cache/oasr/jit/`, keyed on a hash that covers
the sources, the `include/` tree, the nvcc flags, **and** the CUTLASS version
stamp.

Fused attention has **two** kernel lanes and one arbiter over them.

`oasr/jit/attention.py` is the arbiter: `select_backend()`, `set_backend_mode()`,
`fmha_config_supported(...)`, `fmha_backend_for(...)`, `warmup_fmha(...)`. It is
also the CuTeDSL lane — a `functools.cache`-keyed wrapper around
`cutlass.cute.compile()` rather than a Ninja JIT spec, which is why it sits
apart from the table above. `select_backend()` probes the device capability
eagerly at module load and resolves on sm_80 / 86 / 89 / 120, otherwise
`"sdpa"`.

`oasr/jit/fmha.py` is the C++ lane and *is* an ordinary Ninja spec. Jinja
renders one translation unit per feature variant and ninja builds them into one
`.so` per **cell** = `(target_sm, dtype, padded_head_dim)` — the axes that
change the shared-memory layouts and so cannot share a binary. Inside a cell:

    3 (none / causal / local) × 2 (bias) × 2 (dense / paged)   = 12
    + 4 split-KV (unmasked only) + 1 static binding             = 17

Everything else is runtime, not a compile axis: head counts, page size, window
bounds, `cache_seqlens` / `cache_seqstarts`, packed-vs-dense input, and whether
the bias is vectorisable. nvcc parallelises across translation units but not
within one, so a cell costs about one variant's wall time on a many-core box and
then covers every shape that cell will ever be asked for.

`select_backend()` answers a *mode* question and cannot see the shape.
`fmha_backend_for(...)` answers the rest — whether the preferred lane can serve
*this* call, and what to degrade to when it cannot. The case that needs it is
the sliding window, which the C++ lane has and the CuTeDSL one has no argument
for.

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
| `oasr/functionals/pooling.py` | BTC/TC `avg_pool1d` and `max_pool1d`, including symmetric padding and ceil mode (`count_include_pad` for avg; `dilation` and `return_indices` are refused, not ignored) |
| `oasr/functionals/softmax.py`, `oasr/functionals/topk.py`, `oasr/functionals/fft.py` | `softmax`, `log_softmax`, `masked_softmax` (one pass over an attention score tensor: an additive bias and two boolean masks, each **broadcast against the scores through its own strides**, so a shifted `as_strided` relative-position window or a `[..., ::ds]` mask slice is consumed where it is), `topk`, `rfft` / `rfft_power` |
| `oasr/functionals/feature.py` | `stft_frame`, `dct_lifter`, `fbank_preprocess`, `mel_log`, `whisper_logmel`, `lfr_gather` — see [features.md](features.md) |
| `oasr/functionals/attention.py` | `fmha(...)` and `fmha.persistent_inputs(...)` |
| `oasr/functionals/mlp.py` | `gated_mlp` — a whole SwiGLU/GeGLU gate+up+activation+multiply in one dual-B GEMM, plus `gated_mlp_available` (the routing question, so a layer can ask before building anything) and `gated_mlp_backend` (which of the two lanes answered it). Refuses rather than falling back |
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

**And per architecture.** `jit.gemm._GEMM_HEURISTIC_RULES` maps an SM family to
that family's table, and Conv1D's `jit.conv._CONV1D_HEURISTIC_RULES` does the
same; the tuner already emits its literal named for the card it measured
(`_GEMM_HEURISTIC_RULES_SM<sm>`), so tuning a second architecture is a paste plus
a registry line, not an edit to the selector. Only **SM120** is measured today.

An architecture with no table is not an error — `GEMM_DEFAULT` computes the right
answer — but it is the largest gap the heuristic can have, because it is *every*
shape rather than one width, and it is the one gap that used to be invisible: the
arch fall-through returned before recording anything, so `rule_miss_report()`
printed "every shape this process issued had a tuned rule" on a box that had
never opened the table. It is now counted per architecture (not per shape, which
would name every GEMM the process issued) and reported by both
`rule_miss_report()` and `oasr.layers.format_gap_report()`:

```
GEMM heuristic inactive on sm80: no tuned rule table (tuned: sm120), so all 8
shape lookup(s) used GEMM_DEFAULT. Tune this card with scripts/tune_asr_gemm.py.
```

Adding a table is a *measurement*. Rules that were reasoned about rather than
timed have shipped a 4.6x regression and an empty transcript here; the entries a
new table may name are held to its own architecture's emitted variant set by
`tests/kernels/test_gemm_heuristic.py::TestPerArchRuleRegistry`.

Three rules are structural rather than tuned:

- **`GEMM_MIN_ROWS`** — a row floor below which CUTLASS's M-tiling leaves most of
  every tile empty and cuBLAS's GEMV-shaped kernel wins.
- **The candidate space is constrained before it is tuned**
  (`jit.gemm._epilogue_covers_warp`). CUTLASS's tensor-op epilogue folds a
  warp's 32 lanes into a grid derived from the *tile*, and asserts nothing about
  that grid covering a warp; a tile where it does not compiles, launches at full
  speed and returns **wrong numbers**. At the 8-element access width half
  precision uses, that is every `block_n < 32` tile. Such a tile is refused by
  the per-SM builders — for GEMM, BMM, grouped GEMM and Conv2D alike, since all
  four render from the same list — and the refusal is readable through
  `jit.gemm.rejected_tiles()`. This is not a tuning preference: a rule naming
  such a tile is a correctness bug that neither dispatch nor timing can see, and
  one shipped (`.artifacts/gemm_thin_n_tile_epilogue.md`).
- The dispatch decision is a **pure function of the call** and is deliberately
  *not* relaxed under CUDA-graph capture, even though dispatch cost is free
  there: a capture-dependent branch makes the graph pick a different kernel than
  eager, and the resulting one-ulp fp16 difference has produced different tokens.

`OASR_GEMM_HEURISTIC=0` disables the whole thing. Measurements and re-tuning
recipe: `.artifacts/gemm_tuning.md`.

### The BMM general lane

`bmm` has a second lane that the shape rules do not reach. The rendered tile
variants are alignment-8 iterators over contiguous 3-D operands; a decomposed
attention block does not have that shape. Zipformer's is the case that forced
the lane: five products per layer, 4-D permuted views of a
`(time, batch, head, dim)` activation, one operand broadcast over the request
batch, head dims of 32 / 4 / 12 and a relative-position extent that is always
odd. FMHA cannot substitute — the probabilities are materialized once and
*shared* by `SelfAttention` and `NonlinAttention`, and a fused kernel never
materializes them.

`oasr.bmm` therefore accepts one or two broadcasting batch dimensions, arbitrary
N and K, and — the part that saves a copy at every call site — a `B` operand
contiguous along **either** trailing axis, so `[..., N, K]` and its transposed
view are both legal and neither needs `.contiguous()`. A contiguous 3-D
alignment-8 call still takes the tuned lane; everything else lands in
`include/oasr/gemm/bmm.cuh`, which turns three run-time facts into one
instantiation:

| Chosen from | Values | Why it is compile-time |
|---|---|---|
| B's memory layout | CUTLASS `ColumnMajor` / `RowMajor` | the iterator's contiguous axis |
| operand alignment along K | 8 / 4 / 2 elements | `cp.async` cannot issue a 2-byte copy, so alignment 1 is not a tensor-op case at all |
| alignment along N (epilogue, and B when RowMajor) | 8 / 4 / 2 / 1 | an odd N has to be able to store one element at a time |
| threadblock tile | 128×128 / 64×64 / 32×32 / 64×16 (thin-N) | see below |

The grid is 124 instantiations (62 per dtype), rendered one translation unit per
(layout, dtype) from `csrc/templates/bmm_general_template.cu.jinja` — the same
template-per-configuration pattern as the tile variants, and for a measured
reason: nvcc parallelizes across translation units but not within one, so the
module's cold build is set by its largest TU. Inlining the grid into `bmm.cu`
costs 112 s; this split costs 40 s.

Anything outside that grid runs on CUTLASS SIMT `GemmBatched`, which constrains
nothing. **Nothing falls back to PyTorch**: a shape either has a kernel or the
call raises.

Two properties of the lane are worth knowing before changing it:

- **The tile ladder is the difference between the lane and a regression.** These
  problems are 2–70 MFLOP, so they are latency-bound and the only thing that
  matters is how much of the device the grid covers. A single 128×128 tile put
  8–64 CTAs on an RTX 5090's 170 SMs and burned 246 registers/thread for 8.3%
  achieved occupancy — 1.74× slower than cuBLAS. Selecting the largest tile that
  still fills one wave brought that to 1.07×. Note the ceiling: CUTLASS's
  tensor-op epilogue divides the tile's rows by the warp count in M before it
  computes an iteration count, so a 32-row tile is a **single-warp** shape and
  nothing smaller than 64 rows can use 128 threads. The same ceiling has a
  *column* half, and that one is not an assert: the epilogue derives its lane
  width from the tile's columns and its lane rows from the tile's rows
  independently, and never checks that the two cover a warp — so the 16-column
  tile is addressable at 4 elements per access and **silently wrong** at 8.
  `epilogueAlignment` in `bmm.cuh` caps the store width for that tile alone;
  the `cp.async` load keeps its 8 elements.
- **`GemmBatched` advances every operand by one constant stride**, so two batch
  axes are one launch only when all three tensors are affine in the flattened
  index. Both flattening orders are tried, because a contiguous output satisfies
  one and a head-major view of a `(time, batch, head, dim)` activation satisfies
  the other. A broadcast *inner* axis satisfies neither and costs
  `min(batch0, batch1)` launches.

Measurements, including the profile that produced the ladder:
`.artifacts/kg5_strided_bmm.md`.

### Fused attention

```python
oasr.fmha(q, k, v, *, softmax_scale, attn_bias, cache_seqlens, cache_seqstarts,
          block_table, causal, window_left, window_right, out, backend)
```

Three cache modes share one signature:

| Mode | `block_table` | `cache_seqlens` |
|---|---|---|
| Offline | `None` | `None` |
| Dense streaming (caller concatenated old + new K/V) | `None` | set |
| Paged streaming (K/V are pool views) | set | required |

Three backends share that signature — `cxx` (C++ CUTLASS/CuTe), `cute`
(CuTeDSL) and `sdpa` (PyTorch, fp32-friendly). `OASR_ATTN_BACKEND` selects
process-wide; the per-call `backend=` overrides it *without* touching the
global, which matters because `set_backend_mode()` clears three compile caches
and an in-process A/B driven through the environment variable recompiles the
CuTeDSL kernel on every flip. `validate=False` skips checks for proven inputs.

Naming a backend **requires** it: a request it cannot serve raises rather than
quietly computing something else. `backend=None` (the default) is the only mode
that may degrade, and it degrades through `fmha_backend_for(...)`, which asks
whether the preferred lane can serve *this* shape rather than only which lane
the process prefers.

Two capabilities exist on the `cxx` lane only:

| Capability | Why not on `cute` |
|---|---|
| `window_left` / `window_right` — a per-row sliding window, top-left aligned | its `local` axis is the per-*stream* `[seqstart_k, seqlen_k)` pair, a different thing with no argument for this |
| causal and windowed **varlen**, and varlen at all without a second kernel | packed input there is a separate ~600-line `kernel_varlen`; here it is the same instantiation with a zero batch stride |

The `cxx` lane also splits the K range when the grid cannot fill the machine
(flash-decoding). `num_splits` is a pure function of `(shape, SM count)` —
never of CUDA-graph capture state, because splitting changes the order the fp32
partials are summed (rule 11). The workspace is allocated in Python, from
torch's caching allocator, deliberately **not** from `oasr::getCachedWorkspace`,
which branches on `cudaStreamIsCapturing`.

Routing policy and measurements: `.artifacts/fmha_tuning.md` (CuTeDSL) and
`.artifacts/fmha_cpp_validation.md` (the C++ lane's falsifications and the A/B).

### The C++ attention kernel (`include/oasr/attention/`)

Fifteen headers, structured the way CUTLASS 3.x structures a collective:
`CollectiveMainloopSm80` + `CollectiveEpilogue` + `SingleTileScheduler`,
composed by `FmhaKernelSm80` — a shell that names no architecture and
touches no layout. `fmha_launch_template.h` is the **one** file that names
one, so a Hopper or Blackwell lane is a second `FmhaArch<SM>`, a second
collective and one more `conditional_t` arm; the arguments, the FFI signature
and the whole Python side do not move.

Two rules in that header are load-bearing rather than stylistic:

- **`ArchTag` selects instructions; a separate `kIsSm86Or89` selects tuning.**
  `cutlass::arch::Sm86` exists, but the collective is the same one sm_80 uses.
- **Never branch on `ArchTag::kMinComputeCapability >= 90`.** `Sm120`'s *is*
  120, so FlashAttention's `Use_TMA_O` test is true on a part with no TMA at
  all. `FmhaArch<SM>::kHasTma` / `::kIsWarpSpecialized` say what they mean.

`fmha_softmax.h`'s header comment carries the numerics contract — twelve
numbered items, each one a place where FlashAttention's reference is wrong for
OASR's masking semantics, with the failure each one prevents.

### The C++ gated-MLP kernel (`include/oasr/mlp/`)

The same collective decomposition, one family over:
`CollectiveGatedMlpMainloopSm80` + `CollectiveGatedMlpEpilogue`, composed by
`GatedMlpKernel`, with `gated_mlp_launch_template.h` the one file that names
an architecture. `gated_mlp_tiles.h` is deliberately CuTe-free — it holds the
CTA tiles and the arithmetic that picks one, as plain `constexpr` integers, so
`csrc/gated_mlp_jit_binding.cu` can export them as a cheap capability oracle
and `tests/kernels/test_gated_mlp_cpp.py` can hold the Python mirror to them
over four architectures from one box.

Where it differs from the attention family, and why:

| | attention | gated MLP |
|---|---|---|
| tile | **resolved**: `fmhaResolveTile` derives it from the architecture, because the question is "what fits" | **tabled**: the question is a wave count, so it depends on `rows`, `N` *and* the SM count, and all six tiles are compiled as siblings in one module |
| cell | `(sm, dtype, head_dim)` — the axes that change the smem layouts | `(sm, dtype, activation)` — the activation is the one axis that changes the arithmetic, and a checkpoint means one of them |
| variants in a cell | 3 masks x 2 bias x 2 paged = 12 | 6 tiles x 2 bias = 12 |

Two things it does that the CuTeDSL lane cannot:

| Capability | Why not on `cute` |
|---|---|
| a **K residue** — `K` need not be a whole number of K tiles | there the mainloop predicates only the row axis, so a partial K tile reads the next row of `x`; `gated_mlp_shape_supported` has to refuse it |
| **arbitrary row strides** on every operand | there the compiled signature marks the tensors compact, so a row-slice of a wider buffer has to be copied first |

One implementation note is worth carrying, because nothing in the C++ says it.
The K residue is predicated into the ZFILL `cp.async`'s own `src_size` operand
(`copy_zfill_2d`), not into control flow. Written the obvious way —
`if (pred) copy(...) else clear(...)` — ptxas has to order a synchronous `STS`
against an asynchronous `LDGSTS` to the same shared address, and it does that
by bracketing every copy in `BSSY`/`BSYNC` and padding it with three dead
`@!PT LDS RZ, [RZ]`. Measured on the 64x64x32 tile: the K loop's load section
went from ~60 instructions to ~12, the LSU pipe from 2.34M instructions to
1.2M, and the kernel from 0.89x of the CuTeDSL lane to ahead of it everywhere.

### The C++ fused recurrent step (`include/oasr/recurrent/recurrent_step_*.h`)

The same collective decomposition again, and the second lane for the kernel
`oasr/kernels/cute/recurrent/step.py` already implements in CuTeDSL:
`CollectiveRecurrentStepMainloopSm80` + `CollectiveRecurrentStepEpilogue`,
composed by `RecurrentStepKernel`, with `recurrent_step_launch_template.h` the
one file that names an architecture. `recurrent_step_tiles.h` is CuTe-free, so
`csrc/recurrent_step_jit_binding.cu` exports it as a capability oracle and
`tests/kernels/test_recurrent_cpp.py` holds the Python mirror to it over six
architectures from one box.

One fused timestep, with the gate dimension *interleaved* so column `n` is
`(hidden n / G, gate n % G)`:

```
gates[m, n] = sum_k previous_h[m, k] * weight_hh[n, k] + input_gates[m, n]
c[m, i]     = sigmoid(g1) * previous_c[m, i] + sigmoid(g0) * tanh(g2)
h[m, i]     = sigmoid(g3) * tanh(c[m, i])
```

The interleaving is what keeps the epilogue inside one CTA tile — a hidden
unit's gates are adjacent columns, so no cross-tile reduction is needed, and
`recurrentStepTileValid` refuses any `block_n` that would straddle one.

| | gated MLP | recurrent step |
|---|---|---|
| tile | **wave count**: is the last wave still saturating DRAM? | **tabled ladder** on `(hidden, batch)`: a dependent launch has no next kernel to overlap its tail with, so what the measurements found was a boundary between two regimes, not a wave count |
| cell | `(sm, dtype, activation)` | `(sm, dtype, kind)` — `lstm` / `rnn_tanh` / `rnn_relu` |
| variants in a cell | 6 tiles x 2 bias = 12 | 8 tiles |

The ladder is *inherited* from the CuTeDSL lane rather than re-derived, so both
lanes run the same tile on every shape and an A/B between them compares
kernels. `test_the_ladder_reproduces_the_cutedsl_lane` pins that.

Three things it does that the CuTeDSL lane cannot:

| Capability | Why not on `cute` |
|---|---|
| a **K residue** — the hidden width need not be a whole number of K tiles | there the mainloop loops `ceil_div(K, k_block)` and predicates only the row axis, so a partial K tile reads past the end of *both* operands |
| **arbitrary row strides** on every operand | there the compiled signature marks the tensors compact, so a row-slice has to be copied first |
| a staging buffer larger than the ring | there the epilogue *aliases* the ring; here it is a `union`, so `can_implement` needs no such clause |

Two epilogue details are worth carrying, because nothing in the C++ says them:

- **The global loads are issued before the staging barrier.** The transition
  needs `input_gates` and `previous_c` from global memory and the affine from
  shared; issuing the two gmem loads first puts their latency behind the
  accumulator store and the `__syncthreads()` rather than in front of the
  arithmetic.
- **The accumulator staging is padded by 8 floats per row, not 1.** Both avoid
  the MMA-C store's bank conflict; only 8 also keeps every row 16-byte aligned,
  so a cell's four gates come back in one `LDS.128` instead of four `LDS.32`.
  `stride % 32 == 8` puts a 16-lane `STS.64` phase on 32 distinct banks and an
  8-lane `LDS.128` phase likewise. `cuobjdump -sass` confirms both.

One declared limitation: the widest **one-gate** variant (tile 7, 32 epilogue
slots on a 512-thread CTA) sits at REG:128 with a 32-byte stack frame, where
every LSTM variant is at REG:54-120 with no spill. Nothing reaches it — a
vanilla RNN is never routed under `auto`, and tile 7 needs `hidden > 1536` and
`batch > 128` — so it is declared rather than fixed. The header comment records
the two causes that were tested and refuted, so a later attempt starts past
them.

Routing is `oasr/jit/recurrent_cute.py` — the arbiter, and the CuTeDSL lane's
compile cache. `OASR_RECURRENT_CUTE` decides whether to fuse;
`OASR_RECURRENT_BACKEND` decides which lane. Measurements:
`.artifacts/recurrent_cpp_validation.md`.

### CuteDSL kernels (`oasr/kernels/`)

`oasr/kernels/` holds low-level implementations that do **not** use the TVM-FFI /
Ninja pipeline.

- `kernels/cute/attention/base.py` — abstract `FmhaBase` + `pick_arch_cls(major, minor)`
- `kernels/cute/attention/fmha_sm80.py` — `FmhaSm80`, the mainloop (sm_80: A100, A30)
- `kernels/cute/attention/fmha_sm{86,89,120}.py` — thin subclasses for Ampere
  consumer, Ada and consumer Blackwell.
- `kernels/cute/recurrent/step.py` — `RecurrentStepCute`, one fused LSTM/RNN
  timestep as a tensor-core GEMM with the state transition in the epilogue
- `kernels/cute/mlp/gated.py` — `GatedMlpCute`, a **dual-B** GEMM: one A tile in
  shared memory feeding *two* `mma.sync` chains against two B tiles, with
  `activation(gate) * up` applied to the FP32 accumulators. The shape of
  CUTLASS's `examples/45_dual_gemm`, and the reason a gated MLP needs no
  intermediate tensor
- `kernels/cute/` — FlashAttention-style helpers: `block_info.py`, `seqlen_info.py`,
  `mask.py`, `softmax.py`, `tile_scheduler.py`, `pack_gqa.py`, `paged_kv.py`,
  `named_barrier.py`, `copy_utils.py`, `layout_utils.py`, `ampere_helpers.py`, `utils.py`

Each is compiled via `cutlass.cute.compile()` into a Python callable and cached
per config — `oasr/jit/attention.py::_compiled_fmha`,
`oasr/jit/recurrent_cute.py::_compiled_step`,
`oasr/jit/mlp.py::_compiled_gated_mlp` — always with
`options="--enable-tvm-ffi"`, which is what lets the callable take torch tensors
directly and be captured into a CUDA graph.

Each of the three also owns a **routing** module beside its compile cache, because
a fused kernel that is faster on some shapes and slower on others has to say
which: `OASR_ATTN_BACKEND`, `OASR_RECURRENT_CUTE`, `OASR_GATED_MLP_CUTE`, each
with the measured band in its module docstring. For the gated MLP the band is
**one m-tile** (`M <= 64`): with one m-tile every weight element is read from
DRAM once, which is the bandwidth argument the fusion rests on; with two it is
an ordinary GEMM reading its operands twice and cuBLAS wins. The *tile* inside
the band is chosen by `N` rather than by `M` — see `select_gated_mlp_tile`, and
the ten lines of it that exist because a rows-keyed table lost 10% at one model
width.

Attention and the gated MLP each have **two** fused lanes, `cute` and `cxx`,
and they spell the choice differently on purpose. Attention folds both
questions into `OASR_ATTN_BACKEND`, because its fallback (`sdpa`) is itself a
backend and sits in the same list. The gated MLP cannot: its fallback is *two
GEMMs*, which is not a lane of this kernel at all, so "fuse or not"
(`OASR_GATED_MLP_CUTE`) and "which lane" (`OASR_GATED_MLP_BACKEND`, `auto` /
`cute` / `cxx`) are separate switches. Collapsing them would make a rollback
and an A/B the same knob, and they are different decisions: the first is an
operator's call about a shape, the second is a claim about two implementations
of the same thing.

`auto` resolves to `cxx` for the MLP — 1.00x-1.11x of the CuTeDSL lane over
four interleaved, graph-replayed sweeps with no row regressing, plus a strictly
wider shape contract — and to `cute` for attention, where the C++ lane's paged
mode still reads 0.85-0.94x.

### Which routing decisions travel, and which are extrapolations

The routing splits cleanly in two, and only one half follows the card it runs on:

| | reads the machine | example |
|---|---|---|
| **derived** | yes | `gated_mlp_ctas_per_sm` takes `multi_processor_count`, the real opt-in shared memory and `max_threads_per_multi_processor`, and `select_gated_mlp_tile` does wave arithmetic against them; `selectBmmTile` takes `getDeviceMultiProcessorCount()` |
| **measured** | no | `_LSTM_BANDS`'s `(hidden, batch)` cut-offs, the `_TILES` ranking, the gated-MLP candidate list — timed once and written down |

A measured cut-off is neither wrong nor a guess; it is the best available
estimate. But the crossover it encodes is a function of SM count and memory
bandwidth, and the supported set — sm_80, sm_86, sm_89, sm_120 — spans an A30
(56 SMs, 933 GB/s), an A100 (108, 1555), an L40S (142, 864) and an RTX 5090
(170, 1792). All four got the same numbers, with nothing saying whose they were.

`oasr/jit/measured.py` is where each table now names the GPU it was timed on, the
note holding the protocol, and what moves the crossover. Off that card the table
still applies — which side wins at each extreme is a property of the algorithm,
only the boundary between them is a property of the machine — but the fact is
logged once and reported by `oasr.layers.format_gap_report()`:

```
  routing tables measured on another GPU (applied as an extrapolation):
    jit.recurrent_cute._LSTM_BANDS / _TILES
        measured on NVIDIA GeForce RTX 5090 (sm_120, 170 SMs), 1792 GB/s; running on NVIDIA A30 (sm_80, 56 SMs)
```

Identity is `(compute capability, SM count)`, both queried exactly. Bandwidth is
deliberately *not* computed from `memory_clock_rate × bus_width`: that formula is
right for HBM and GDDR6 and wrong for GDDR7, so on an RTX 50-series card it would
report the measured machine as an extrapolation of itself.

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
rule is checked against a Python oracle kept inside `tests/decoders/test_alignment.py`
(exact agreement over randomised input and the whole Unicode plane), and that
file skips without the extension.
