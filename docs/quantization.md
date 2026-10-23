# Quantization

OASR quantizes the dense GEMMs of the layer waist — `Linear`, `LinearActivation`
and `PointwiseConv1d` — and, when asked, its dense convolutions — `Conv1d` and
`Conv2d` ([Convolutions and `module_types`](#convolutions-and-module_types)) — so
every architecture inherits it.  It is the ninth
registry axis: a format is a preset plus a method plus its kernel lanes, an
algorithm is a registration, and an architecture declares what its structure
allows.  The engine core has no per-scheme branch.

Two W8A8 formats are served, by one method:

* **int8** -- per-output-channel int8 weights and int8 activations, per-token
  *dynamic* scales (`int8_w8a8`) or a calibrated static per-tensor scale
  (`int8_w8a8_static`), run by a CUTLASS 2.x int8 `mma.sync` GEMM with an exact
  int32 accumulator;
* **fp8 e4m3** (`fp8_w8a8`, `fp8_w8a8_static`; sm89+) -- the same layout in
  `float8_e4m3fn`, run by CUTLASS 2.x's Ada e4m3 `mma.sync` (sm_120 executes it
  too) with an fp32 accumulator ([FP8](#fp8)).

Either GEMM's epilogue dequantizes, adds the bias and applies the layer's
activation, and can quantize its own output for the next GEMM.  The format layer
also describes NVFP4 (`oasr.quantization.spec.PRESETS`); no method serves it yet,
and asking for it is refused by name.

## Using it

**Online, no calibration** — round-to-nearest weights on a full-precision checkpoint:

```python
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir=..., quantization="int8_w8a8"))  # or "fp8_w8a8"
# or {"scheme": "int8_w8a8", "ignore": "encoder.encoders.0.*"}
```

```bash
oasr-server --ckpt-dir <dir> --quantization int8_w8a8 [--quantization-option ignore=GLOB,...]
    [--quantization-option module_types=linear,pointwise_conv]
```

`module_types` selects layers by kind; the default is the dense GEMMs.
Convolutions need a calibrated static scale, so online quantization refuses them
and `oasr quantize --module-types` writes them
([Convolutions and `module_types`](#convolutions-and-module_types)).

On the U2++ Conformer (LJSpeech-200 gate, fp16): online `int8_w8a8` 4.14 % /
3.70 % (offline / streaming), online `fp8_w8a8` 4.00 % / 3.73 %, fp16 4.08 % /
3.70 %.

**Offline, SmoothQuant** — calibrate, smooth, quantize, write a native checkpoint:

```bash
python -m benchmarks.asrbench prepare librispeech --subset dev-clean --download
oasr quantize $CKPT_DIR /path/to/u2pp_w8a8_sq \
    --recipe smoothquant_w8a8 \
    --calib-manifest ~/.cache/oasr/asrbench/librispeech/dev-clean/manifest.jsonl \
    [--calib-utterances 128] [--alpha 0.5] [--ignore GLOB]
```

Recipes: `int8_w8a8_rtn`, `smoothquant_w8a8`, `smoothquant_w8a8_fused`, and the
fp8 trio `fp8_w8a8_rtn`, `smoothquant_fp8_w8a8`, `smoothquant_fp8_w8a8_fused`.
`--recipe smoothquant_w8a8_fused` adds a calibrated static input scale to every
GEMM-chain consumer (the FFN's `w_2`), so `w_1` quantizes for it and the pair
exchanges int8 ([Fewer quantize launches](#fewer-quantize-launches)): faster, and
on the U2++ gate 4.11 % offline / 3.64 % streaming against 4.03 / 3.70 for
`smoothquant_w8a8` and 4.08 / 3.70 for fp16.

A checkpoint written by `oasr quantize` is served quantized with **no flag**: its
`oasr_config.json` carries the spec, and the spec is authoritative — an online
request for a different scheme raises, the same scheme is a no-op.

`--calib-manifest` takes any `{id, audio, ...}` jsonl (paths relative to
`--audio-root`, default the manifest's directory), so production recipes
calibrate on their own domain.  Calibrate on audio at the model's sample rate;
the loader refuses another rate rather than resample.

`ASREngine.quantization_info()` and the engine's startup log report what is
served: schemes, layer count, and the kernel lane each layer bound.

## How it fits together

```
OFFLINE (oasr quantize)                       RUNTIME
fp32 master model ◄─ quantization_layout()    native checkpoint v2 ─► QuantizationSpec
calibration audio ─► checkpoint frontend      apply_quantization: a QuantMethod per selected layer
observers on GEMM inputs                       strict load (dtype-checked) ─► .to(dtype) ─► .to(device)
algorithm: smoothquant | rtn                   bind: each layer picks its kernel lane once
QuantMethod.quantize_weight  ◄── shared ──►    link: fused norms, GEMM chains (from the layout)
save_native(format_version=2)                 forward: [norm | producer | quantize_*] ─► scaled_mm[_quant]
```

| Module | Owns |
|---|---|
| `oasr/quantization/spec.py` | Number formats (`QuantArgs`, `QuantScheme`, presets) and `QuantizationSpec` — which layers, which scheme, provenance |
| `oasr/quantization/layout.py` | `SmoothGroup`, `GemmChain`, `QuantLayout` (incl. `batch_invariant` pins): the data a model declares |
| `oasr/quantization/methods/` | `QuantMethod`: storage slots, weight quantization, lane binding, forward, torch reference |
| `oasr/quantization/kernels/lanes.py` | `ScaledMMLane`: which kernel serves a scheme on this GPU (`is_supported(sm)`, `can_implement(scheme, N, K)`) |
| `oasr/quantization/calibration/` | Observers; `asr_forward_loop` through the checkpoint's own frontend |
| `oasr/quantization/algorithms/` | `rtn`, `smoothquant`, `static_inputs`, `static_conv_inputs`, registered with an `options_cls` |
| `oasr/quantization/apply.py`, `export.py`, `cli.py` | Glue, report, `oasr quantize` |
| `oasr/layers/_quantizable.py` | `QuantizableLayer` (the `quant_method` slot, `quant_kind`, `quant_unsupported()`, the dtype-preserving `_apply`) and its two halves, `QuantizableGemm` and `QuantizableConv`; `QuantizedActivation` |
| `oasr/functionals/quant.py`, `scaled_mm.py`, `scaled_conv.py`, `norm.py` | `quantize_per_token`, `quantize_static`, `scaled_mm`, `scaled_mm_quant`, `scaled_conv2d`, `scaled_conv1d`, `layer_norm_quant`, `add_layer_norm_residual_quant`, and their torch twins |

### The layer seam

A quantized layer keeps its type, path and state-dict key names; its
`quant_method` (a plain object, not a module) takes over `forward` **before** the
fp row floor (`GEMM_MIN_ROWS` would hand `F.linear` int8 storage).  CPU, fp32 and
`OASR_LAYERS_BACKEND=torch` run `apply_reference`, the torch emulation of the same
numerics.  Layers whose weight a fused kernel reads raw — the transducer
joiner's `decoder_proj` / `output_linear` (the fused decode) — set
`quantizable = False`, and a spec that selects one is refused.  The gated MLP's
gate / up / down are quantizable: when gate or up carries a method, the block
leaves its fused dual-B kernel for two scaled GEMMs (gate's SiLU in its
epilogue), the elementwise product and `down`; unquantized it is byte-identical.

State dict of a quantized layer (names follow compressed-tensors):

| Key | dtype | Shape |
|---|---|---|
| `weight` | int8 / float8_e4m3fn | the layer's own (`[N, K]`; `[N, K, 1]` for a pointwise conv; KRSC / KSC for a `Conv2d` / `Conv1d`) |
| `weight_scale` | fp32 | `[N]` |
| `smooth_inv` | fp32 | `[K]`, only on runtime-smoothed layers |
| `input_scale` | fp32 | `[1]`, only with a static activation scale (`*_w8a8_static`) |
| `bias` | served dtype | unchanged |

`Module.to(dtype)` casts every floating tensor — fp32 scales included — so
`QuantizableLayer._apply` moves quantized slots by device only.

### Load order

`oasr.models.registry.materialize_model` is the one load sequence (engine,
`oasr-convert`, `oasr quantize`, the round-trip tests):

* **native quantized checkpoint** — quantized slots are created *before* the
  strict load, and their dtypes are checked first (`param.copy_` would cast an
  fp32 tensor into an int8 slot silently);
* **online** — the fp weights load first (a converted checkpoint's legacy
  remaps run inside `_load_from_state_dict`), then each selected layer quantizes
  its own fp32 weight on the host;
* then the dtype cast, the device move, and lane binding.

The native format writes `format_version: 2` only for a quantized checkpoint; a
full-precision export stays version 1.  Readers accept both.

### Kernel lanes

| Lane | SMs | Notes |
|---|---|---|
| `cutlass_sm80_int8` | 80, 86, 89, 120 (native); 90, 100 (the Sm80 `mma.sync` kernel, as fp16 GEMM runs there) | int8.  Default.  Needs `K % 16 == 0`, `N % 8 == 0`.  Per-token or per-tensor activation scale (a runtime switch in the epilogue).  No serial/parallel split-K (the Sm80 EVT rejects it only by an `assert` NDEBUG removes); Stream-K instead, below |
| `cutlass_sm89_fp8` | 89, 120 (native); 90, 100 (the same `mma.sync` kernel) | fp8.  Default.  Same template, epilogue and alignment; no split of any kind ([FP8](#fp8)) |
| `torch_int_mm` | ≥ 75 | int8: `torch._int_mm` + a torch epilogue; A/B and baseline only |
| `torch_scaled_mm` | ≥ 89 | fp8: `torch._scaled_mm` at unit tensorwise scales into fp32, then a torch epilogue; A/B and baseline only, and not batch-invariant (cuBLASLt picks its algorithm per M) |

Below sm89 an fp8 scheme is refused at bind with a pointer to `int8_w8a8`: the
GPU has no fp8 tensor cores, which is not a missing kernel.

Turing (sm75) is the declared kernel gap `scaled-mm-int8-sm75`: a quantized layer
there runs the torch reference.  `OASR_SCALED_MM_BACKEND=auto|cutlass|torch`
(read at import) is the A/B switch; naming a lane requires it.

**Stream-K.**  Two Stream-K variants are built.  Their partials are the int32
accumulator, so the reduction is an exact integer sum and a Stream-K launch
returns the same bits as the data-parallel one -- batch invariance survives it,
unlike the fp16 GEMM's fp32 partials.  Measured against the best data-parallel
tile on an RTX 5090 it loses on every Conformer shape (0.35-0.69x: K <= 2048 is
16 int8 K-tiles, too few to amortise the barrier memset and partial-tile
fixup), is 0.5-0.9x at K = 4864-8192, at parity at K = 16384 and 1.09-1.16x
only on a 7B-size down-projection (N=896, K=18944, M >= 256).  So the shape
heuristic never picks it; a measured tuning entry may.

Tile selection (`oasr.jit.scaled_mm.select_config`, and `select_quant_config` for
the 8-bit-output launch) reads the tuning DB first -- the `scaled_mm` family for
int8, `scaled_mm_fp8` (its own `scaled_mm_fp8.json`, measured on its own lane)
for fp8 -- a
measured entry per `(op, N, K)`, M rounded up to a region, user tier over the
shipped `oasr/tune/db/sm<family>/scaled_mm.json` — and falls back to the shape
heuristic.  For int8 that is a ladder over seven data-parallel tiles
(`_INT8_LADDER`): the largest tile no taller than twice M that launches enough
CTAs for the GPU, with a K-dependent rung.  Short-K tiles keep two CTAs resident
per SM: at K = 256 (every Conformer width) a CTA runs four K tiles, so its
prologue and its dequant epilogue are most of its life, and only a second CTA
hides them.  A deep K, or a launch with fewer CTAs than SMs, takes a deeper
pipeline instead.  Fitted to a 25-tile sweep over five architectures' shapes, it
is within 1 % of the best of the 25 (geomean; worst 1.23x).  The first int8 set
was sized for one CTA per SM: 1.13x slower (geomean, up to 1.39x) on the Conformer
widths above M = 256.  For
fp8 the heuristic is the largest data-parallel tile that still launches half a
wave of CTAs.  A region may name the heuristic itself (`"heuristic"`: measured, nothing
beat it), and only a measured entry picks a Stream-K tile.  Every variant computes
the same bits, so tuning changes speed only.  The tuning census records the int8
GEMMs (`oasr tune census --quantization int8_w8a8`, or `fp8_w8a8`), `oasr tune
build` writes `scaled_mm.json` / `scaled_mm_fp8.json` beside `gemm.json`,
`tuning.strict` / `tuning.mode="prewarm"`
cover them like fp GEMMs, and `ASREngine.tuning_report()` ends with how many
quantized layers have a tuning entry and how many run on the heuristic.  See
[`autotuning.md`](autotuning.md#the-int8-scaled-gemm-scaled_mm).

Two norm kernels emit the quantized activation themselves, so a norm-fed 8-bit
GEMM skips its quantize launch (bit-exact with the unfused pair; see
[kernels.md](kernels.md#layernorm-with-a-per-token-int8-output)).
`oasr.functionals.norm.layer_norm_quant` is LayerNorm whose output is per-token
int8 or e4m3 (`dtype=`) plus a per-row fp32 scale: `quantize_per_token(layer_norm(x))` in one launch,
with no smoother, because SmoothQuant folds its scale into the norm's affine.
`oasr.functionals.norm.add_layer_norm_residual_quant` does the same for the fused
add + LayerNorm, and also returns the unnormalized residual sum exactly as
`add_layer_norm_residual` stores it.

## Convolutions and `module_types`

`QuantizationConfig.module_types` (and `oasr quantize --module-types`) chooses
the *kinds* of layer to quantize.  Every quantizable layer declares one
(`QuantizableLayer.quant_kind`):

| Kind | Layers | Activation scale |
|---|---|---|
| `linear` | `Linear`, `LinearActivation` | per token (dynamic) or static |
| `pointwise_conv` | `PointwiseConv1d` (a 1x1 conv: a GEMM over channels) | per token (dynamic) or static |
| `conv1d` | `Conv1d`, `Conv1dActivation` | **static** (calibrated) |
| `conv2d` | `Conv2d`, `Conv2dActivation` | **static** (calibrated) |

`conv` names every convolution kind and `all` every kind.  The default is
`linear,pointwise_conv`, the dense GEMMs: the same set as before convolutions
could be selected.  A kind list without `linear` quantizes no linear layer.

A convolution's input scale must be **static**.  An output pixel sums over a
receptive field spanning several input pixels, and per-token scales would not
factor out of that sum.  So:

* `QuantizationSpec.for_model` gives a selected convolution the static variant
  of the scheme's format (`int8_w8a8_static`);
* `oasr quantize` calibrates its scale (`static_conv_inputs`, the same clip
  search as `static_inputs`, `observer=mse` by default; options via
  `--option static_conv_inputs.observer=...`) with **any** recipe, so
  `--calib-manifest` is required;
* an online request (`EngineConfig(quantization={"scheme": ..., "module_types":
  "conv2d"})`) is refused by name, pointing at `oasr quantize`.

fp8 has no convolution kernel: a convolution kind with an fp8 scheme is refused.

```bash
oasr quantize $CKPT_DIR /path/to/u2pp_w8a8_conv --recipe smoothquant_w8a8_fused \
    --module-types linear,pointwise_conv,conv2d --calib-manifest .../manifest.jsonl
```

Only a dense, int8-aligned convolution has a kernel: `groups == 1`,
`in_channels % 16 == 0` and `out_channels % 8 == 0`.  Anything else is skipped,
and `QuantizableLayer.quant_unsupported()` gives the reason, recorded in the
spec's `producer["skipped"]`.  The usual skips are the one-channel stems of the
Conformer, Nemotron and Zipformer subsamplings and the depthwise convolutions.
An explicit target that names such a layer is refused.

The kernel (`oasr.scaled_conv2d`; `scaled_conv1d` is the height-one problem) is
a CUTLASS 2.x int8 implicit-GEMM forward convolution, `include/oasr/scaled_conv/`.
It reads an NHWC int8 input and a KRSC int8 filter, accumulates exactly in int32,
and writes fp16/bf16.  CUTLASS 2.x convolutions have no EVT epilogue, so it runs
through `DefaultConv2dFpropWithBroadcast`:

* the per-channel factor `input_scale * weight_scale[k]`, folded once at bind, is
  the broadcast vector;
* the bias is a zero-stride source tensor;
* `ScaledConvOutputOp` computes `act(fma(scale, acc, bias))` with one rounding.

The exact accumulator makes every tile compute the same bits, so the kernel is
batch-invariant.  Its tile is the int8 GEMM's occupancy-sized ladder, measured
on the U2++ and Whisper front-ends (`oasr/jit/scaled_conv.py`).  The forward is
`quantize_static` then the convolution.  The quantize is a separate memory pass,
and for a front-end conv whose input is large, it is most of what is left:

| U2++ subsampling conv (256→256, 3x3/2), RTX 5090 | fp16 | int8 conv | int8 conv + quantize |
|---|---|---|---|
| batch 1 | 57 us | 11 us | 15 us (3.7x) |
| batch 8 | 192 us | 52 us | 86 us (2.2x) |
| batch 64 | 1001 us | 352 us | 752 us (1.3x) |

Whisper's conv1d front-ends gain 2.2-2.7x with the quantize included.

Accuracy on the LJSpeech-200 gate is unchanged:

| Model | Without the conv | With it | fp16 |
|---|---|---|---|
| U2++ `smoothquant_w8a8_fused` + `conv2d` | 4.11 / 3.64 | 4.09 / 3.70 | 4.08 / 3.70 |
| whisper-tiny `smoothquant_w8a8` + `conv1d` | 3.76 | 3.67 | 3.67 |

U2++ figures are offline / streaming.  A convolution neither consumes a
producer's 8-bit output nor emits one.

## Fewer quantize launches

A quantized GEMM needs 8-bit input.  Unfused, every one pays a quantize launch
and reads its input in fp16; three mechanisms let the producer of that input
emit it instead, so on the Conformer only three of a block's eight quantized
GEMMs (`linear_out`, `pointwise_conv1`, `pointwise_conv2`) still quantize
their own input.

| Producer | Emits | Declared by | Armed when |
|---|---|---|---|
| a pre-norm (`LayerNorm.quantize_output`, `quantize_dtype`) | per-token int8 / e4m3 + scales: `layer_norm_quant` / `add_layer_norm_residual_quant` | `SmoothGroup(..., fuse_output_quant=True)` | every consumer takes a per-token pre-quantized input (a bound lane, no runtime smoother) in one format |
| a GEMM (`W8A8Method.emit`) | 8-bit at its consumer's static scale, in the producer's own format: `scaled_mm_quant` | `GemmChain(producer, consumer)` | the consumer has a static input scale, both have a lane, and both use the same format |

A producer only ever emits its consumers' format; a consumer handed another
format raises rather than reinterpret the bytes.

The quantized activation travels as a `QuantizedActivation` (`data`, `scale`,
`dtype`): what passes between producer and consumer may read its `shape` and
nothing else -- the Conformer's attention reads `x.shape` before `linear_qkv`,
the FFN calls `w_2(w_1(x))`.  `bind_quantization` re-derives every link after
binding lanes (a rebind on the host disarms them all), and the report says
which inputs were quantized by whom.  `OASR_QUANT_FUSION=0` (read at import)
leaves them all disarmed: the A/B switch for attributing a change to the fusion.

**Every fused kernel is bit-identical to the composition it replaces** -- the
norm rounds to the input dtype before taking the amax, the producer's epilogue
rounds `act(y)` to the layer's dtype before multiplying by the consumer's
multiplier -- so a model with its links armed returns exactly what it returns
with them disarmed (`tests/quantization/test_quant_fusion.py`, `torch.equal` on
the encoder output).

**Why a GEMM chain needs a static scale.**  A per-token scale needs the whole
row, which no single output tile of the producer sees.  The consumer instead
takes a calibrated per-tensor scale (`int8_w8a8_static`, `input_scale`): the
`static_inputs` algorithm measures the input the quantize kernel will see
(smoothed by the consumer's runtime `smooth_inv`, so it runs after
`smoothquant`) and chooses a clip.  The clip is searched, not taken at the
maximum.  On the U2++ `w_2` inputs (mean SQNR over 24 layers) per-token scales
give 29.8 dB; a static scale at the calibration maximum 17.3 dB -- the median
frame spans 22 % of that range -- which cost 0.44 WER in streaming; the 99.999th
percentile 24.9 dB.  `observer=mse` (the default) picks the clip minimising the
expected quantization error of a histogram of the calibration values (rounding
below the clip, saturation above it); it measured 4.11 / 3.64 % against 4.20 /
3.73 % for the percentile and 4.08 / 4.14 % for the maximum (offline / streaming).

## FP8

`fp8_w8a8` is int8's layout in `float8_e4m3fn` (`Q = 448`): per-channel weights,
per-token dynamic or calibrated static activations, the same storage names, the
same fused producers and the same recipes (`fp8_w8a8_rtn`, `smoothquant_fp8_w8a8`,
`smoothquant_fp8_w8a8_fused`).  What differs:

* **The kernel.**  The scaled-GEMM template is generic over its operands; an fp8
  config (`CutlassScaledMmFp8Config`) names `float_e4m3_t`, an fp32 accumulator
  and CUTLASS's Sm89 `mma.sync` (`16x8x32` e4m3), which sm_120 runs as well.  Its
  own JIT module (`scaled_mm_fp8`) and production set (eight data-parallel tiles),
  so an int8 user never compiles it.
* **Batch invariance by construction, without an exact accumulator.**  fp32 sums
  depend on their order.  CUTLASS's default Sm89 fp8 operator stages its
  accumulation (each warp sums a K tile's MMAs into a zeroed fragment, then adds
  it), which makes the order a function of the warp's K extent; *fast*
  accumulation sums every MMA straight into the running fragment in K order.
  With fast accumulation, no K sliced across warps (`block_k == warp_k`) and no
  split, every tile sums every output in the same order, so all eight compute
  the same bits and a row is bitwise the same alone or in any batch.  Measured:
  tiles bit-identical at K = 2048 / 4096 / 18944, and on the U2++ gate
  `max_batch_size` 8 / 16 / 32 give identical transcripts.  The accuracy cost of
  fast accumulation is invisible beside the e4m3 operands (mean error identical
  to the staged order's; 0.3-1 % of fp16 outputs one ulp off the correctly
  rounded value, against 0.1 % staged).  The config space enforces the rule
  (`static_assert(BK == WK)`, `_buildable`), and
  `tests/kernels/test_scaled_mm.py::TestFp8` fails with the staged operator.
* **Speed, on a GeForce part.**  An RTX 5090 runs fp8 with fp32 accumulation at
  half its int8 rate: at large M the fp8 GEMM measured 1.7-2.0x the fp16 GEMM
  and 0.55-0.62x the int8 one; at small M (weight-bandwidth-bound) the two are
  equal and 1.3-2.5x fp16.  So on this card int8 is the faster W8A8 and fp8's
  case is accuracy without calibration (online `fp8_w8a8` beat online int8 on
  the U2++ gate) and portability to datacenter parts, where fp8 runs at the
  int8 rate or faster.
* **Calibration.**  `static_inputs` and the `mse` clip search model e4m3's
  error: its step follows the value (`2^(floor(log2 v) - 3)`), so the bulk's
  error barely depends on the clip and outliers are kept more often than int8's
  uniform step allows.

## FP8 KV cache

`quantization.kv_cache="fp8_e4m3"` (`EngineConfig(quantization={"kv_cache":
"fp8_e4m3"})`, `oasr-server --quantization-option kv_cache=fp8_e4m3`) stores an AR
decoder's attention KV cache -- Whisper's self- and cross-attention, the
Qwen2-Audio LM -- as e4m3, independently of the weight `scheme`:

* **Scale.**  One fp32 power-of-two scale per (token, kv-head), computed as the
  token is written (`kv_quant`'s `quantize_kv_store`, one launch per layer for K
  and V): no calibration, row-local and so batch-invariant, ~0.52x the bytes of
  fp16.  Power-of-two scales make dequantization exact, so the fp8 read is
  bit-identical to the fp16 kernel fed the dequantized cache.
* **Read.**  Steps read the cache through the C++ attention lane's fp8 variants
  (`fmha_kv8_*` cells; dense and paged, unsplit and split-KV), which stage fp8 in
  the cp.async ring and dequantize each tile in shared memory.  The arbiter
  routes fp8 KV there as a declared rule; a causal, windowed or biased read, a
  head dim the gather cannot tile, `OASR_ATTN_BACKEND=cute|sdpa` or a device
  outside the lane (sm80/86/89/120) is refused by name.  Prefill attends its own
  prompt in the served dtype; it is quantized into the cache as it is written.
* **Capacity.**  The VRAM-derived decoder-KV budget and paged pool count fp8 plus
  scales (Qwen2-7B 524,288 -> 270,336 bytes per token, Whisper-large-v3 163,840 ->
  87,040), so about 1.9x more rows fit; raise `max_decode_slots` to use them.
* **A decode family or model without an AR KV cache refuses it at construction**
  (`BaseDecoder.kv_cache_dtypes` declares the formats a decoder stores).

On the LJSpeech-200 gate: whisper-tiny 3.70 % (fp16 KV 3.67), Qwen2-Audio-7B
(bf16) 5.44 % (bf16 KV 5.67).  Measured decode attention: Whisper-large-v3
cross-attention at batch 16 1.6x faster than fp16 (less to read), Qwen2-7B steps
at batch 16 1.1x, at batch 4 0.78x (little KV, so the in-shared-memory
conversion is the critical path).  End to end it pays where the KV budget limits
admission.  See [cache_manager.md](cache_manager.md) for the storage layout.

## Numerics

Pinned, and reproduced bit for bit by the torch reference
(`oasr.functionals.quant.reference_quantize_per_token`).  For fp8, `127` reads
`448` and the rounding is e4m3 round-to-nearest-even with saturation
(`cvt.rn.satfinite`, torch's `clamp(±448).to(float8_e4m3fn)`):

```
quantize (per token, row r):
  x'       = f32(x) * smooth_inv                 # applied before the amax
  amax     = max(max_j |x'_j|, 1e-8)
  inv      = __fdiv_rn(127, amax)                # IEEE division under --use_fast_math
  q_j      = clamp(__float2int_rn(x'_j * inv), -127, 127)
  scale[r] = __fdiv_rn(amax, 127)

weights (per output channel, host, fp32):  same rule, eps floor for all-zero rows

static quantize (per-tensor scale s, consumer multiplier mul, computed once at bind):
  mul[k]   = smooth_inv[k] * (1 / s)            # one IEEE division, one multiply
  q        = clamp(rne(f32(x) * mul[k]), -127, 127)

epilogue:  out = round(act(fma(a_scale[m], b_scale[n] * f32(acc), bias[n])))
           a_scale[m] per token, or one per-tensor scale
quantizing epilogue (scaled_mm_quant), for a static consumer:
           q = clamp(rint(f32(round_T(act(...))) * mul[n]), -127, 127)
           (fp8: e4m3_rne_satfinite(f32(round_T(act(...))) * mul[n]))
```

The int32 accumulator is exact (fp8's has one summation order for every tile)
and the activation scale is row-local (or a constant), so a quantized layer
returns bitwise-equal rows for bitwise-equal inputs however the batch is made up.  That alone does not make the *model*
batch-invariant: a full-precision GEMM's config (cuBLAS, split-K, Stream-K) is
chosen from M by the rule table, so its rounding depends on the batch, and int8
rounding amplifies those ulps into flipped words.  A model therefore names its
fp GEMMs ahead of int8 layers in `QuantLayout.batch_invariant`, and
`apply_quantization` pins them (`Linear.batch_invariant`: data-parallel CUTLASS
only, see [kernels.md](kernels.md)) -- the Conformer pins the subsampling
projection and `linear_pos`, which made `max_batch_size` 8 / 16 / 32 give
identical transcripts.  What remains is attention at batch 1: alone, its grid
underfills the GPU and it takes the split-KV path, which sums the fp32 partials
in another order (a word or two on the 200-utterance gate; the rows record it).

A per-tensor *dynamic* activation scale is refused outright: padding and
co-batched utterances would change every row's quantization.

## SmoothQuant

For GEMMs sharing an input `X`, a per-channel `s` rewrites `X Wᵀ` as
`(X / s)(W s)ᵀ` — exact in real arithmetic — with
`s_j = max|X_j|^α / max|W_j|^(1-α)` (α = 0.5 by default), clamped so the folded
parameters stay inside fp16.  The model declares where `X / s` is absorbed:

| Fold | Producer change | Valid when |
|---|---|---|
| `norm_affine` | LayerNorm / RMSNorm `γ /= s`, `β /= s` | only commuting ops (zero mask, zero pad, a same-domain cache `cat`) sit between norm and GEMM |
| `v_to_o` | V rows of the fused QKV projection `/= s` | attention is linear in V per channel; tied per KV head under GQA |
| `producer_rows` | the previous GEMM's rows and bias `/= s` | the activation between is positively homogeneous (ReLU) |
| `runtime` | none: `smooth_inv = 1/s` on the consumer, applied inside `quantize_per_token` | always |

Every scale is computed from the original weights before any fold is applied,
so a layer that is a consumer in one group (its input columns) and a producer in
another (its output rows) folds consistently.

The Conformer's layout (`ConformerModel.quantization_layout`): per block,
`norm_ff_macaron → feed_forward_macaron.w_1`, `norm_mha → self_attn.linear_qkv`,
`norm_conv → conv_module.pointwise_conv1` and `norm_ff → feed_forward.w_1` fold
into the norm; V folds into `linear_out`; both `w_2` and `pointwise_conv2` carry a
runtime smoother.  The subsampling front-end, `linear_pos` (its input is the
sinusoidal table) and the CTC head stay full precision.  The CNN cache holds
post-`norm_conv` values in the smoothed domain, which is consistent because only
this encoder reads it.

Every architecture with a real checkpoint declares its layout (pure data;
`tests/models/test_quant_layouts.py` checks every path, and the SmoothQuant
invariance oracle -- smoothed, unquantized output unchanged -- runs on every tiny
config and forward path):

| Architecture | Folded into the norm | `v_to_o` | `producer_rows` / runtime | Kept fp / pinned | Why it differs |
|---|---|---|---|---|---|
| U2++ rescoring decoder (`decoder.*`, both directions) | `norm1 → linear_q/k/v`, `norm2 → src_attn.linear_q`, `norm3 → w_1` (fusable) | self and cross `linear_v → linear_out` | `w_1 → w_2` (ReLU) + chain; every `src_attn.linear_k/v` runtime | `output_layer` | `encoder.after_norm` also feeds the fp CTC head, so the cross-attention K/V take a runtime smoother rather than change the CTC path |
| Whisper | `self_attn_layer_norm → q/k/v`, `encoder_attn_layer_norm → q`, `final_layer_norm → fc1` (fusable); `encoder.layer_norm → every decoder encoder_attn.k/v` | `v_proj → out_proj` (all three attentions) | `fc2` runtime (exact-erf GELU before it); chain `fc1 → fc2` | none (the conv front-end and tied head are not quantizable GEMMs) | — |
| Nemotron (encoder) | `norm_feed_forward{1,2} → linear1`, `norm_self_att → q/k/v`, `norm_conv → pointwise_conv1` (fusable: the mask, pad and cache come after the GLU) | `v_proj → o_proj` | `linear2`, `pointwise_conv2` runtime; chain `linear1 → linear2` | subsampling, `relative_k_proj`, predictor, joint, `prompt_projector`; pins `subsampling.linear`, `relative_k_proj` | — |
| Paraformer | encoder `norm1 → linear_q_k_v`, `norm2 → w_1`; decoder `norm1 → w_1`, the FFN's inner norm `→ w_2`, `norm3 → src_attn.linear_q` | decoder `src_attn.linear_k_v` V rows `→ linear_out` | encoder `w_1 → w_2` (ReLU) + chain; encoder `linear_out`, decoder `linear_k_v` runtime | `predictor.*`, `output_layer`; pin `predictor.cif_output` | no encoder `v_to_o` (the FSMN reads V); `after_norm` cannot fold (the CIF integrates its raw output) |
| Qwen2-Audio | tower as Whisper's encoder; LM `input_layernorm → q/k/v`, `post_attention_layernorm → gate + up` (RMSNorm γ) | LM `v_proj → o_proj`, tied per KV head (GQA) | LM `up → down` (`act(gate) * up` is linear in `up`) | `lm_head`, `multi_modal_projector`; pin `multi_modal_projector.linear` | RMSNorm has no 8-bit-emitting kernel yet, so the declared fusions do not arm |

Two rules the real checkpoints taught:

* **A channel calibration never saw active keeps `s = 1`.**  At `s`'s floor a
  ReLU channel that never fired on English audio later owned its token's scale on
  Chinese audio (`|x / s|` 1e4-2.7e5 against a median of ~2) and turned every
  Paraformer encoder FFN to noise.
* **The fp16 range check covers only what is stored in the served dtype**: norm
  affines, producer biases, and a producer's weight only when it stays full
  precision.  A layer that is both a consumer (its columns times `s`) and a
  producer (its rows over `s`) is stored 8-bit with per-row scales.

`BaseAsrModel.calibration_forward` drives calibration: the encoder by default,
and each AED / speech-LLM decoder the way it decodes (U2++ teacher-forced over the
greedy CTC path in both directions, Whisper and Qwen2 greedily from their prompt,
Paraformer through CIF and its NAR pass), so decoder groups are observed.
`oasr quantize --calib-dtype bfloat16` keeps the fp32 master on the host and
calibrates a reduced-precision twin on the device, for a model whose fp32 master
does not fit (a 7B LM is 33.6 GB fp32).

Calibration runs one unpadded utterance at a time through the checkpoint's own
frontend (`audio_scale` and the `FeatureSpec`, exactly as `InputProcessor`), in
fp32, TF32 off, on a sorted and evenly strided selection of the manifest.  The
manifest hash, selection rule and α are recorded in the spec's `producer`, and
`quantization_report.json` beside the checkpoint lists every layer's scale
ranges.

## Extending

| To add… | Touch |
|---|---|
| a format (NVFP4; FP8 is the worked example) | a preset in `spec.py`, a `QuantMethod` (`register_quant_method`) or a format of an existing one, its lanes, its quantize kernel |
| a backend (a native sm90 int8 kernel) | a `ScaledMMLane` + `register_lane`; methods pick it through `choose_lane` |
| an algorithm (GPTQ, AWQ) | `algorithms/<name>.py` + `register_quant_algorithm`; reuse the observers and `quantize_weight` |
| an architecture | `quantization_layout()` on the model — data only; online RTN works without it |
| a GEMM-bearing layer | mix in `QuantizableGemm`, implement `gemm_weight()`, branch on `quant_method` first |
| a convolution-bearing layer | mix in `QuantizableConv` (KRSC / KSC weight, `in_features` = input channels), branch on `quant_method` first; `W8A8ConvMethod` serves it |
| a layer kind | a `quant_kind` in `QUANT_KINDS` (`oasr/layers/_quantizable.py`), a method whose `serves(layer)` accepts it, and — if its input scale must be static — an entry in `STATIC_INPUT_KINDS` |

## Tests and benchmarks

| Where | What |
|---|---|
| `tests/kernels/test_quant.py` | quantize (int8 and e4m3) is bit-exact against the reference; row-local; graph replay |
| `tests/kernels/test_scaled_conv.py` | the int8 convolution: every tile over padded / strided / dilated geometries and every activation; exact accumulator; a sample bitwise independent of the batch; conv1d; contracts |
| `tests/quantization/test_quant_conv.py` | `module_types` (kinds, aliases, defaults, the online refusal), conv selection and skip reasons, the conv method's storage, reference, calibration, native round trip, kernel vs reference in the layer |
| `tests/kernels/test_scaled_mm.py` | every compiled variant of both formats; exact accumulator; fp8: every tile the same bits, rows bitwise independent of the batch, the fp8-output epilogue == the unfused pair; contracts |
| `tests/quantization/test_quant_fp8.py` | the fp8 format end to end: storage, native round trip, kernel vs reference, fused norms and chains in e4m3 == unfused, formats never mixed across a link, the fp8 clip model |
| `tests/kernels/test_fmha_kv8.py`, `tests/engine/test_fp8_kv_cache.py` | the fp8 KV cache: quantize-on-write bit-exact, the fp8 attention read == fp16 on the dequantized cache, dense == paged == graph replay, refusals, capacity |
| `tests/models/test_quantized_layers.py` | the seam: state dict, casts, CPU reference, kernel parity, per-architecture layout and raw readers |
| `tests/quantization/` | spec, SmoothQuant invariance on every encoder path (per-op, fused, packed, paged streaming), export round trip, calibration frontend == engine frontend; `test_quant_fusion.py`: the static scheme, links armed where declared, fused model == unfused model bitwise, `static_inputs` and the clip searches, the pins |
| `tests/kernels/test_norm_quant.py`, `tests/models/test_layer_norm_quant_output.py` | the fused norm kernels == LayerNorm then `quantize_per_token`, bitwise; `LayerNorm.quantize_output` |
| `tests/kernels/test_gemm_batch_invariant.py`, `tests/models/test_linear_batch_invariant.py` | pinned GEMM rows bitwise independent of M; the pin bypasses the row floor |
| `tests/kernels/tune/test_scaled_mm_tuning.py`, `tests/engine/test_tuning_scaled_mm_engine.py` | the `scaled_mm` tuning tier: resolution, capture, build, `strict` |
| `tests/engine/test_quantized_engine.py` | capture == eager, per-layer batch invariance, a quantized checkpoint served offline and streaming |
| `ci/wer-reference.json` | `conformer_int8_rtn`, `conformer_w8a8_sq`, `conformer_w8a8_sq_streaming`, `conformer_w8a8_sq_fused`, `conformer_w8a8_sq_fused_streaming`, `conformer_fp8_rtn`, `conformer_fp8_sq_fused`, `conformer_fp8_sq_fused_streaming`, `conformer_w8a8_sq_fused_conv`, `conformer_w8a8_sq_fused_conv_streaming`, `whisper_w8a8_sq_conv`, `whisper_kv_fp8`, `llm_kv_fp8`; a row whose format the GPU cannot run (fp8 weights below sm89, fp8 KV off the C++ attention lane) is skipped with the reason |

Calibration data is the `LIBRISPEECH_DEV_DIR` asset (asrbench-prepared
LibriSpeech dev-clean): in-domain for LibriSpeech-trained checkpoints and a
different corpus from the LJSpeech WER gate.  The kernel benchmark families are
`quant` (`--fp8` for e4m3), `scaled_mm` (subroutines `scaled_mm_fp8`,
`quant_scaled_mm_fp8`) and `scaled_conv` (`scaled_conv2d`, `quant_scaled_conv2d`,
`scaled_conv1d`; `python benchmarks/run.py --list`); the accuracy and
engine families take `--quantization`.
