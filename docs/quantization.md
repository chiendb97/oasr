# Quantization

OASR quantizes the dense GEMMs of the layer waist — `Linear`, `LinearActivation`
and `PointwiseConv1d` — and, when asked, its dense convolutions — `Conv1d` and
`Conv2d` ([Convolutions and `module_types`](#convolutions-and-module_types)) — so
every architecture inherits it.  It is the ninth registry axis: a format is a
preset plus a method plus its kernels, an algorithm is a registration, and an
architecture declares what its structure allows.  The engine core has no
per-scheme branch.

Two W8A8 formats are served:

* **int8** (`int8_w8a8`, `int8_w8a8_static`) -- per-output-channel int8 weights
  and int8 activations, with a per-token *dynamic* scale or a calibrated static
  per-tensor scale.  The GEMM accumulates exactly in int32;
* **fp8 e4m3** (`fp8_w8a8`, `fp8_w8a8_static`; sm89 and newer) -- the same layout
  in `float8_e4m3fn`, accumulated in fp32 ([FP8](#fp8)).

The GEMM dequantizes, adds the bias and applies the layer's activation in one
kernel.  Separately, an AR decoder's KV cache can be stored in fp8
([FP8 KV cache](#fp8-kv-cache)).  The format layer also describes NVFP4; no
method serves it yet, and asking for it is refused by name.

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

**Offline, SmoothQuant** — calibrate, smooth, quantize, write a native checkpoint:

```bash
python -m benchmarks.asrbench prepare librispeech --subset dev-clean --download
oasr quantize $CKPT_DIR /path/to/u2pp_w8a8_sq \
    --recipe smoothquant_w8a8 \
    --calib-manifest ~/.cache/oasr/asrbench/librispeech/dev-clean/manifest.jsonl \
    [--calib-utterances 128] [--alpha 0.5] [--ignore GLOB]
```

| Recipe | What it writes |
|---|---|
| `int8_w8a8_rtn`, `fp8_w8a8_rtn` | Round-to-nearest weights, per-token activations: online quantization, saved.  No calibration |
| `smoothquant_w8a8`, `smoothquant_fp8_w8a8` | [SmoothQuant](#smoothquant), then per-token activations |
| `smoothquant_w8a8_fused`, `smoothquant_fp8_w8a8_fused` | As above, plus a calibrated static input scale on each GEMM-chain consumer (the FFN's second GEMM), so the GEMM before it emits 8-bit output ([Fewer quantize launches](#fewer-quantize-launches)).  The fastest recipe |

A checkpoint written by `oasr quantize` is served quantized with **no flag**: its
`oasr_config.json` carries the spec, and the spec is authoritative.  Naming one of
the schemes it was written with is a no-op (a `smoothquant_w8a8_fused` checkpoint
answers to both `int8_w8a8` and `int8_w8a8_static`), naming another raises, and so
do `ignore` / `module_types` — they choose the layers *online* quantization
quantizes, and a quantized checkpoint's layers are already chosen.  For the same
reason neither is accepted without a `scheme`.

`--calib-manifest` takes any `{id, audio, ...}` jsonl (paths relative to
`--audio-root`, default the manifest's directory).  Calibrate on the domain you
serve, at the model's sample rate; the loader refuses another rate rather than
resample.

`ASREngine.quantization_info()` and the engine's startup log report what is
served: schemes, layer count, the KV-cache format, and the kernel each layer
runs.  The accuracy and engine benchmark families take `--quantization`.

### Choosing a format

* **int8** is the faster W8A8 on GeForce parts, which run fp8 with fp32
  accumulation at half their int8 rate.  With calibration (`smoothquant_*`) it
  matches full precision on every gated architecture.
* **fp8** needs no calibration to stay accurate (online `fp8_w8a8` is the better
  online scheme on the U2++ gate), and runs at the int8 rate or faster on
  datacenter parts.  It needs sm89 or newer.
* The end-to-end speedup is bounded by the share of GPU time spent in the
  quantized GEMMs.  Attention, norms, depthwise convolutions and the decode
  search stay at full precision, so a small encoder gains less than a large
  model.

### Accuracy

WER (%) on the 200-utterance LJSpeech gate; `ci/wer-reference.json` holds the
gated values and the commands that produced them.

| Model | Full precision | Online int8 | Online fp8 | `smoothquant_w8a8` | `smoothquant_w8a8_fused` |
|---|---|---|---|---|---|
| U2++ Conformer, fp16, offline | 4.08 | 4.14 | 4.00 | 4.03 | 3.97 (fp8: 4.06; + `conv2d`: 4.00) |
| U2++ Conformer, fp16, streaming | 3.70 | | | 3.73 | 3.64 (fp8: 3.67; + `conv2d`: 3.76) |
| Whisper-tiny, fp16 | 3.67 | 3.64 | 3.67 | 3.61 (+ `conv1d`: 3.70) | |
| Nemotron, fp16, offline / streaming | 2.59 / 2.41 | | 2.67 (offline) | 2.44 / 2.38 | |
| Qwen2-Audio-7B, bf16 | 5.58 | 5.88 | | 5.52 | |

An fp8 KV cache measures 3.70 on Whisper-tiny and 5.44 on Qwen2-Audio-7B.

## How it fits together

```
OFFLINE (oasr quantize)                       RUNTIME
fp32 master model ◄─ quantization_layout()    native checkpoint v2 ─► QuantizationSpec
calibration audio ─► checkpoint frontend      apply_quantization: a QuantMethod per selected layer
observers on GEMM inputs                       strict load (dtype-checked) ─► .to(dtype) ─► .to(device)
algorithm: smoothquant | rtn                   bind: each layer picks its kernel once
QuantMethod.quantize_weight  ◄── shared ──►    link: fused norms, GEMM chains (from the layout)
save_native(format_version=2)                 forward: [norm | producer | quantize] ─► scaled GEMM
```

| Module | Owns |
|---|---|
| `oasr/quantization/spec.py` | Number formats (`QuantArgs`, `QuantScheme`, presets) and `QuantizationSpec` — which layers, which scheme, provenance |
| `oasr/quantization/layout.py` | `SmoothGroup`, `GemmChain`, `QuantLayout`: the data a model declares |
| `oasr/quantization/methods/` | `QuantMethod`: storage, weight quantization, kernel binding, forward, torch reference |
| `oasr/quantization/kernels/lanes.py` | `ScaledMMLane`: which kernel serves a scheme on this GPU |
| `oasr/quantization/calibration/`, `algorithms/` | Observers and the calibration loop; `rtn`, `smoothquant`, `static_inputs`, `static_conv_inputs` |
| `oasr/quantization/apply.py`, `export.py`, `cli.py` | Glue, report, `oasr quantize` |
| `oasr/layers/_quantizable.py` | `QuantizableLayer` and its two halves, `QuantizableGemm` and `QuantizableConv`; `QuantizedActivation` |

### The layer seam

A quantized layer keeps its type, module path and state-dict key names; its
`quant_method` (a plain object, not a module) takes over `forward`.  CPU, fp32
and `OASR_LAYERS_BACKEND=torch` run the torch reference of the same numerics.  A
layer whose weight a fused kernel reads directly — the transducer joiner's
`decoder_proj` / `output_linear` — is not quantizable, and a spec that selects
one is refused.  A quantized gated MLP runs its gate and up projections as two
scaled GEMMs instead of its fused kernel.

State dict of a quantized layer (names follow compressed-tensors):

| Key | dtype | Shape |
|---|---|---|
| `weight` | int8 / float8_e4m3fn | the layer's own (`[N, K]`; `[N, K, 1]` for a pointwise conv; KRSC / KSC for a `Conv2d` / `Conv1d`) |
| `weight_scale` | fp32 | `[N]` |
| `smooth_inv` | fp32 | `[K]`, only on runtime-smoothed layers |
| `input_scale` | fp32 | `[1]`, only with a static activation scale (`*_w8a8_static`) |
| `bias` | served dtype | unchanged |

`model.to(dtype)` leaves these dtypes alone: a quantized layer moves its
quantized tensors by device only.

### Load order

`oasr.models.registry.materialize_model` is the one load sequence (engine,
`oasr-convert`, `oasr quantize`):

* **native quantized checkpoint** — quantized slots are created *before* the
  strict load, and their dtypes are checked first;
* **online** — the fp weights load first, then each selected layer quantizes
  its own fp32 weight on the host;
* then the dtype cast, the device move, and kernel binding.

The native format writes `format_version: 2` only for a quantized checkpoint; a
full-precision export stays version 1.  Readers accept both.

## The U2++ Conformer, quantized

The WeNet U2++ Conformer (`$CKPT_DIR`) is the worked example: 12 encoder blocks
(d = 256, 4 heads, FFN 2048, a causal convolution module) and a bidirectional
attention-rescoring decoder of 3 + 3 layers.  For any model, the engine's
startup report gives each quantized layer's scheme and kernel, and the norm or
GEMM that quantizes its input.

### What is quantized

`8-bit` is the scheme's format (int8, or e4m3 under an fp8 scheme) and `fp16` the
served dtype (bf16 when served in bf16).  Each quantized GEMM's weight columns
are multiplied by its SmoothQuant scale `s`; the last column says where the
matching `1/s` goes: into a norm's affine, into a producer's output rows, or into
a *runtime smoother* applied as the input is quantized ([SmoothQuant](#smoothquant)).

```
fbank [B, T, 80]
│
▼ encoder.embed (Conv2dSubsampling4)                          weights  s is folded into
    convs.0                      Conv2d 1→256, 3x3/2, ReLU    fp16     skipped: one input channel
    convs.1                      Conv2d 256→256, 3x3/2, ReLU  int8 ¹   static input scale, no s
    out.0                        Linear 4864→256              fp16 ²
    out.1, pos_enc               LayerNorm, rel. positions    fp16
│
▼ encoder.encoders.0 … 11, each:
  ┌ norm_ff_macaron              LayerNorm                    fp16
  │ feed_forward_macaron.w_1     Linear 256→2048, swish       8-bit    norm_ff_macaron
  │ feed_forward_macaron.w_2     Linear 2048→256              8-bit    runtime smoother
  └ x += ½ · ffn
  ┌ norm_mha                     LayerNorm                    fp16
  │ self_attn.linear_qkv         Linear 256→768               8-bit    norm_mha
  │ self_attn.linear_pos         Linear 256→256               fp16 ²
  │ rel-pos attention            fmha                         fp16
  │ self_attn.linear_out         Linear 256→256               8-bit    linear_qkv's V rows
  └ x += attention
  ┌ norm_conv                    LayerNorm                    fp16
  │ mask, causal pad, CNN cache                               fp16
  │ conv_module.pointwise_conv1  1x1 conv 256→512             8-bit    norm_conv
  │ GLU, depthwise conv (k=15),                               fp16
  │   LayerNorm + swish
  │ conv_module.pointwise_conv2  1x1 conv 256→256             8-bit    runtime smoother
  └ x += conv
  ┌ norm_ff                      LayerNorm                    fp16
  │ feed_forward.w_1             Linear 256→2048, swish       8-bit    norm_ff
  │ feed_forward.w_2             Linear 2048→256              8-bit    runtime smoother
  └ x = norm_final(x + ½ · ffn)                               fp16
│
▼   encoder.after_norm           LayerNorm                    fp16
├─► ctc                          Linear 256→5002              fp16     → CTC prefix beam search
└─► decoder (attention rescoring, below)
```

```
decoder.{left,right}_decoder, 3 layers each                   weights  s is folded into
    embed                        Embedding, positions         fp16
  per layer:
  ┌ norm1                        LayerNorm                    fp16
  │ self_attn.linear_q, _k, _v   Linear 256→256               8-bit    norm1
  │ self-attention                                            fp16
  │ self_attn.linear_out         Linear 256→256               8-bit    linear_v's rows
  └ x += self-attention
  ┌ norm2                        LayerNorm                    fp16
  │ src_attn.linear_q            Linear 256→256               8-bit    norm2
  │ src_attn.linear_k, _v        Linear 256→256 (encoder)     8-bit    runtime smoother
  │ cross-attention                                           fp16
  │ src_attn.linear_out          Linear 256→256               8-bit    linear_v's rows
  └ x += cross-attention
  ┌ norm3                        LayerNorm                    fp16
  │ feed_forward.w_1             Linear 256→2048, ReLU        8-bit    norm3
  │ feed_forward.w_2             Linear 2048→256              8-bit    w_1's rows
  └ x += ffn
    after_norm, output_layer     LayerNorm, Linear 256→5002   fp16
```

¹ Only with `--module-types linear,pointwise_conv,conv2d`, and int8 only.
`convs.0` has one input channel, which no 8-bit kernel serves.
² Pinned to batch-invariant kernels ([Numerics](#numerics)).

That is 96 quantized GEMMs in the encoder (8 per block) and 60 in the decoder
(10 per layer), plus `convs.1` when asked.  Everything else runs in the served
dtype: the residual stream, the norms' arithmetic, attention (scores, softmax,
the KV cache), the GLU, the depthwise convolution and the CNN cache, the CTC head
and the search.

### One quantized GEMM

```
x  fp16 [M, K]
│  quantize, per row m:  a_scale[m] = max_k |x[m, k] · smooth_inv[k]| / 127
▼
q  int8 [M, K],  a_scale fp32 [M]          W  int8 [N, K],  w_scale fp32 [N]   (checkpoint)
└────────────────────┬──────────────────────┘
                     ▼  tensor cores
acc = q · Wᵀ         int32 [M, N], exact
                     ▼  epilogue, same kernel
y = act(a_scale[m] · w_scale[n] · acc + bias[n])        fp16 [M, N], rounded once
```

There is no dequantize kernel.  Multiplying the exact int32 accumulator by
`a_scale[m] · w_scale[n]` in the GEMM's epilogue is the dequantization, and the
bias and the layer's activation (swish for `w_1`) are applied in the same pass.
Two variants:

* **A static input scale** (`*_static`; the fused recipe's `w_2`): `a_scale` is
  one calibrated number, `input_scale`, instead of one per row.
* **A quantizing epilogue** (`scaled_mm_quant`, for a GEMM-chain producer): the
  epilogue rounds `y` to fp16, multiplies it by the consumer's
  `smooth_inv / input_scale` and writes 8-bit, so the consumer's input never
  exists in fp16.

Under fp8 the same holds with e4m3 (`Q = 448` in place of 127) and an fp32
accumulator.

### The quantized forward of one encoder block

One block under `smoothquant_w8a8_fused`, one line per kernel launch, in order.
`══` is an 8-bit tensor with its fp32 scales, `──` the served dtype.  The left
rail is the residual stream `r`: `├ +f ►` means that kernel adds `f` to `r`,
keeps the sum in fp16 as the new `r`, and normalizes it.

```
        kernel                          layer                        output
x
│
├────►  layer_norm_quant                norm_ff_macaron              ══ int8 + scales
│       scaled_mm_quant                 feed_forward_macaron.w_1     ══ int8 at w_2's scale
│       scaled_mm                       feed_forward_macaron.w_2     ── fp16 f
├ +½f ► add_layer_norm_residual_quant   norm_mha                     ══ int8 + scales
│       scaled_mm                       self_attn.linear_qkv         ── fp16 q, k, v
│       gemm, matmul, fmha              linear_pos, attention        ── fp16 a
│       quantize_per_token              self_attn.linear_out         ══ int8 + scales
│       scaled_mm                       self_attn.linear_out         ── fp16 o
├ +o ─► add_layer_norm_residual         norm_conv                    ── fp16
│       mask, pad or cache concat       conv_module                  ── fp16 (CNN cache)
│       quantize_per_token              conv_module.pointwise_conv1  ══ int8 + scales
│       scaled_mm                       conv_module.pointwise_conv1  ── fp16
│       glu, depthwise, norm + swish    conv_module                  ── fp16
│       quantize_per_token + smoother   conv_module.pointwise_conv2  ══ int8 + scales
│       scaled_mm                       conv_module.pointwise_conv2  ── fp16 c
├ +c ─► add_layer_norm_residual_quant   norm_ff                      ══ int8 + scales
│       scaled_mm_quant                 feed_forward.w_1             ══ int8 at w_2's scale
│       scaled_mm                       feed_forward.w_2             ── fp16 f
└ +½f ► add_layer_norm_residual         norm_final                   ── fp16 x'
```

What is fused, and why there:

* **Norm → GEMM.** `norm_ff_macaron`, `norm_mha` and `norm_ff` each feed one GEMM
  and nothing else, so each emits that GEMM's 8-bit input itself: the residual
  add, the LayerNorm and the per-token quantize are one launch
  (`add_layer_norm_residual_quant`, or `layer_norm_quant` where there is no add to
  fold).  The GEMM's `1/s` is already in the norm's γ and β.  `norm_conv` cannot
  do this: the mask, the causal pad and the CNN-cache concatenation sit between it
  and `pointwise_conv1`, and the cache holds the served dtype.
* **GEMM → GEMM.** In each FFN, `w_1`'s epilogue applies the bias and swish and
  quantizes the result at `w_2`'s calibrated static scale, with `w_2`'s runtime
  smoother folded into the same multiply.  The 2048-wide hidden activation is
  written once, as int8.
* **GEMM → served dtype.** Every other `scaled_mm` returns fp16, so attention, the
  GLU, the depthwise convolution, the residual adds and both caches never see an
  8-bit tensor.
* **Still quantizing their own input:** `linear_out` (the attention output),
  `pointwise_conv1` (after the cache concatenation) and `pointwise_conv2` (its
  smoother applied in the same pass), with one `quantize_per_token` launch each.

The rescoring decoder runs the same pieces.  `norm1`, `norm2` and `norm3` emit
8-bit for their GEMMs.  Under the fused recipe, `w_1` quantizes for `w_2`.
`src_attn.linear_k` / `linear_v` quantize the encoder output with their runtime
smoother.  A streaming chunk runs the same block, and the attention KV cache and
the CNN cache stay in the served dtype.

### What each recipe fuses

| Recipe | Norm → GEMM (3 per block) | `w_1` → `w_2` (2 per block) | Standalone quantize launches per block |
|---|---|---|---|
| online `int8_w8a8` / `fp8_w8a8`, `*_rtn`, `smoothquant_w8a8`, `smoothquant_fp8_w8a8` | fused | not fused: `w_2` quantizes its input per token | 5 |
| `smoothquant_w8a8_fused`, `smoothquant_fp8_w8a8_fused` | fused | fused | 3 |
| any, with `OASR_QUANT_FUSION=0` | not fused | not fused | 8 |

## GPU support

| GPU | int8 GEMM | fp8 GEMM | int8 convolution |
|---|---|---|---|
| sm80, sm86 | `mma.sync` kernel | refused: no fp8 tensor cores (use `int8_w8a8`) | `mma.sync` kernel |
| sm89, sm120 | `mma.sync` kernel | `mma.sync` kernel | `mma.sync` kernel |
| sm90 (H100, H200) | native Hopper kernel | native Hopper kernel | `mma.sync` kernel (non-native) |
| sm100 (B200, GB200) | native Blackwell kernel | native Blackwell kernel | `mma.sync` kernel (non-native) |
| sm103 (B300) | `mma.sync` kernel (non-native) | native Blackwell kernel | `mma.sync` kernel (non-native) |
| sm75 | the declared kernel gap `scaled-mm-int8-sm75`: the torch reference, counted in the gap report | refused | the declared gap `scaled-conv-int8-sm75` |

The **native kernels** are CUTLASS 3.x: TMA loads, a warp-specialized persistent
kernel, and the architecture's own tensor-core instruction, with one output
tile's epilogue (dequantize, bias, activation, store) overlapping the next
tile's main loop:

* **Hopper** -- wgmma, with *ping-pong* warpgroups (two consumer warpgroups own
  alternate tiles; while one multiplies, the other runs its epilogue) for small
  and mid M, and *cooperative* warpgroups (both on one larger tile) for large M.
  fp8 accumulates in the tensor core for 128 K elements at a time, then in fp32
  registers.
* **Blackwell data-center** -- tcgen05, accumulating in tensor memory, which is
  double-buffered so the next tile's MMA starts while the epilogue drains the
  last; 1-SM tiles, and 2-SM (CTA-pair) tiles for large M.  int8 needs the
  `sm_100a` build target, which exists for compute capability 10.0 only, so B300
  serves int8 with the `mma.sync` kernel.

The native kernels compute what the `mma.sync` kernels compute -- the same
epilogue, the same launch contract -- and keep each format's guarantee: int8
accumulates exactly, fp8 sums every output in the same order whatever the tile,
so a row's result does not depend on the batch.  The quantizing epilogue of a
GEMM chain stores 8-bit output through TMA, which needs `N % 16 == 0`; a
producer with another N runs the GEMM and the quantize as two launches, with the
same bits.  `ASREngine.quantization_info()` names the kernel each layer runs.

A device runs its architecture family's kernels (sm_121 the sm_120 ones, sm_87
the sm_86 ones).  A GEMM needs `in_features % 16 == 0` and
`out_features % 8 == 0`; one that does not is left full precision when layers
are selected by kind, with the reason recorded in the spec's
`producer["skipped"]`, and naming it explicitly is refused.  No shipped
architecture has one.

`OASR_SCALED_MM_BACKEND` selects the GEMM backend:

| Value | Runs |
|---|---|
| `auto` (default), `cutlass` | the CUTLASS kernels: the native kernel where one runs, else `mma.sync` |
| `mma_sync` | the CUTLASS 2.x `mma.sync` kernels only -- on Hopper and Blackwell, the A/B arm for what the native kernels buy |
| `torch` | `torch._int_mm` / `torch._scaled_mm`, an A/B baseline never chosen automatically |

Naming one requires it: what it cannot serve is refused.

Every tile of a quantized GEMM computes the same bits, so tuning changes speed
only.  Tile selection and the tuning database are in
[autotuning.md](autotuning.md#the-int8-scaled-gemm-scaled_mm), the kernels in
[kernels.md](kernels.md).

## Convolutions and `module_types`

`QuantizationConfig.module_types` (and `oasr quantize --module-types`) chooses
the *kinds* of layer to quantize:

| Kind | Layers | Activation scale |
|---|---|---|
| `linear` | `Linear`, `LinearActivation` | per token (dynamic) or static |
| `pointwise_conv` | `PointwiseConv1d` (a 1x1 conv: a GEMM over channels) | per token (dynamic) or static |
| `conv1d` | `Conv1d`, `Conv1dActivation` | **static** (calibrated) |
| `conv2d` | `Conv2d`, `Conv2dActivation` | **static** (calibrated) |

`conv` names every convolution kind and `all` every kind.  The default is
`linear,pointwise_conv`, the dense GEMMs.  A kind list without `linear`
quantizes no linear layer.

A convolution's input scale must be **static**: an output pixel sums over a
receptive field spanning several input pixels, and per-token scales would not
factor out of that sum.  So:

* a selected convolution gets the static variant of the scheme's format
  (`int8_w8a8_static`);
* `oasr quantize` calibrates its scale with **any** recipe, so
  `--calib-manifest` is required;
* an online request (`EngineConfig(quantization={"scheme": ..., "module_types":
  "conv2d"})`) is refused by name, pointing at `oasr quantize`.

Convolutions are int8 only: a convolution kind with an fp8 scheme is refused.

```bash
oasr quantize $CKPT_DIR /path/to/u2pp_w8a8_conv --recipe smoothquant_w8a8_fused \
    --module-types linear,pointwise_conv,conv2d --calib-manifest .../manifest.jsonl
```

Only a dense, aligned convolution is quantized: `groups == 1`,
`in_channels % 16 == 0` and `out_channels % 8 == 0`.  Anything else is skipped,
with the reason recorded in the spec's `producer["skipped"]`; the usual skips
are the one-channel stems of the Conformer, Nemotron and Zipformer subsamplings
and the depthwise convolutions.  An explicit target that names such a layer is
refused.

A quantized convolution quantizes its input in a separate pass, which for a
front-end convolution over a large input is much of its remaining cost; the
gain is largest at small batch.  It neither consumes another layer's 8-bit
output nor emits one.

## Fewer quantize launches

A quantized GEMM needs 8-bit input.  Unfused, every one pays a quantize launch;
instead, the layer that produces that input can emit it already quantized:

| Producer | Emits | Declared by | Armed when |
|---|---|---|---|
| a pre-norm (LayerNorm, or the fused add + LayerNorm) | per-token 8-bit + scales | `SmoothGroup(..., fuse_output_quant=True)` | every consumer takes a per-token input without a runtime smoother, in one format |
| a GEMM | 8-bit at its consumer's static scale | `GemmChain(producer, consumer)` | the consumer has a static input scale, and both layers use the same format |

On the Conformer this leaves three of a block's eight quantized GEMMs
(`linear_out`, `pointwise_conv1`, `pointwise_conv2`) quantizing their own input
([the block's launches](#the-quantized-forward-of-one-encoder-block)).

The quantized activation travels as a `QuantizedActivation`; code between the
producer and its consumers may read its `shape` and nothing else.
`bind_quantization` derives every link after binding kernels, and the report
says which inputs were quantized by whom.  **A fused producer is bit-identical
to the unfused pair**, so a model with its links armed returns exactly what it
returns with them disarmed.  `OASR_QUANT_FUSION=0` (read at import) disarms
them all, for A/B attribution.

**Why a GEMM chain needs a static scale.**  A per-token scale needs the whole
row, which no single output tile of the producer sees.  The consumer instead
takes a calibrated per-tensor scale (`input_scale`), measured by the
`static_inputs` algorithm on the input the consumer will see.  The clip is
searched rather than taken at the calibration maximum: `observer=mse` (the
default) picks the clip minimising the expected quantization error, and
`--option static_inputs.observer=...` selects another observer.

## FP8

`fp8_w8a8` is int8's layout in `float8_e4m3fn` (`Q = 448`): per-channel weights,
per-token dynamic or calibrated static activations, the same storage names, the
same fused producers and the same recipes (`fp8_w8a8_rtn`, `smoothquant_fp8_w8a8`,
`smoothquant_fp8_w8a8_fused`).  What differs:

* **Batch invariance without an exact accumulator.**  fp32 sums depend on their
  order, so the fp8 kernels are built to sum every output in K order whatever
  the tile: every tile computes the same bits, and a row is bitwise the same
  alone or in any batch.
* **Its own kernels.**  An int8-only deployment never compiles them.
* **Speed** depends on the part ([Choosing a format](#choosing-a-format)).
* **No convolutions** ([Convolutions](#convolutions-and-module_types)).

## FP8 KV cache

`quantization.kv_cache="fp8_e4m3"` (`EngineConfig(quantization={"kv_cache":
"fp8_e4m3"})`, `oasr-server --quantization-option kv_cache=fp8_e4m3`) stores an AR
decoder's attention KV cache -- Whisper's self- and cross-attention, the
Qwen2-Audio LM -- as e4m3, independently of the weight `scheme`:

* **Scale.**  One power-of-two scale per (token, kv-head), computed as the token
  is written: no calibration, batch-invariant, about half the bytes of fp16.
  Dequantization is exact, so attention over the fp8 cache equals the
  served-dtype kernel over the dequantized cache, bit for bit.
* **Read.**  Decode steps read the cache through the C++ attention kernel on
  sm80 / 86 / 89 / 120.  `OASR_ATTN_BACKEND=cute|sdpa`,
  `OASR_LAYERS_BACKEND=torch` and other devices are refused by name.  Prefill
  attends its own prompt in the served dtype.
* **Capacity.**  The VRAM-derived decoder-KV budget counts fp8 plus scales, so
  about 1.9x more rows fit; raise `max_decode_slots` to use them.
* **A decode family or model without an AR KV cache refuses it at construction.**

It pays where the KV budget limits admission or a step reads a long cache
(cross-attention over 30 s of audio); at small batch a step can be slower than
fp16.  See [cache_manager.md](cache_manager.md#106-fp8-kv-for-the-ar-decoders)
for the storage layout.

## Numerics

The kernels and the torch reference (`oasr.functionals.quant`) agree bit for
bit.  `Q` is 127 for int8 and 448 for e4m3; int8 rounds half to even, e4m3 rounds
to nearest even and saturates:

```
quantize, per token (row r):   x'       = x · smooth_inv          (runtime-smoothed layers only)
                               amax     = max(max_j |x'_j|, 1e-8)
                               q_j      = clamp(round(x'_j · (Q / amax)), -Q, Q)
                               scale[r] = amax / Q
static quantize:               q        = clamp(round(x · (smooth_inv · (1 / input_scale))), -Q, Q)
weights, per output channel:   the per-token rule, on the host in fp32
GEMM:                          out = act(a_scale[m] · (weight_scale[n] · acc) + bias[n]),
                                     rounded once to the output dtype
```

A quantized layer returns bitwise-equal rows for bitwise-equal inputs however
the batch is made up.  That alone does not make the *model* batch-invariant: a
full-precision GEMM ahead of a quantized layer may round differently at another
batch size, and 8-bit rounding amplifies the difference into changed words.  A
model therefore names those GEMMs in `QuantLayout.batch_invariant`, which pins
them to batch-invariant kernels when it is quantized; the Conformer gives
identical transcripts at `max_batch_size` 8, 16 and 32.  At batch 1 attention
can still differ by a word or two.

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
| `runtime` | none: `smooth_inv = 1/s` on the consumer, applied by the quantize step; the weight takes `W · s` when it is quantized | always |

Smoothing leaves the full-precision model computing the same function (up to
rounding); only the quantized model sees the difference.  A channel calibration never saw active
keeps `s = 1`, so out-of-domain audio cannot make it dominate its token's
scale.

Each architecture with a real checkpoint declares its layout in
`quantization_layout()`; Zipformer and the icefall transducer do not, so online
quantization serves them and SmoothQuant refuses them by name.  What stays full
precision:

| Architecture | Kept full precision |
|---|---|
| U2++ Conformer (encoder and rescoring decoder) | the subsampling front-end, `linear_pos`, the CTC head, the decoder's `output_layer` |
| Whisper | the conv front-end (unless `module_types` adds `conv1d`), the tied output head |
| Nemotron | the subsampling, `relative_k_proj`, the predictor, the joint, `prompt_projector` |
| Paraformer | `predictor.*`, `output_layer` |
| Qwen2-Audio | `lm_head`, `multi_modal_projector`.  RMSNorm cannot emit 8-bit output yet, so the LM's fused producers do not arm |

### Calibration

Calibration runs one unpadded utterance at a time through the checkpoint's own
frontend, in fp32 with TF32 off, over a sorted and evenly strided selection of
the manifest.  It drives the encoder, and each AED / speech-LLM decoder the way
it decodes, so decoder layers are calibrated too.  `oasr quantize --calib-dtype
bfloat16` keeps the fp32 master on the host and calibrates a bf16 copy on the
device, for a model whose fp32 master does not fit (a 7B LM is 33.6 GB in fp32).
The manifest hash, selection rule and α are recorded in the spec's `producer`,
and `quantization_report.json` beside the checkpoint lists every layer's scale
ranges.

## Extending

| To add… | Touch |
|---|---|
| a format (NVFP4; FP8 is the worked example) | a preset in `spec.py`, a `QuantMethod` (`register_quant_method`) or a format of an existing one, its lanes, its quantize kernel |
| a backend (another kernel generation; the native sm90 / sm100 lanes are the worked example) | a `ScaledMMLane` + `register_lane`, its lane id, config space and module in `oasr/jit/scaled_mm.py`; methods pick it through `choose_lane` |
| an algorithm (GPTQ, AWQ) | `algorithms/<name>.py` + `register_quant_algorithm`; reuse the observers and `quantize_weight` |
| an architecture | `quantization_layout()` on the model — data only; online RTN works without it |
| a GEMM-bearing layer | mix in `QuantizableGemm`, implement `gemm_weight()`, branch on `quant_method` first |
| a convolution-bearing layer | mix in `QuantizableConv` (KRSC / KSC weight, `in_features` = input channels), branch on `quant_method` first |
| a layer kind | a `quant_kind` in `QUANT_KINDS` (`oasr/layers/_quantizable.py`), a method whose `serves(layer)` accepts it, and — if its input scale must be static — an entry in `STATIC_INPUT_KINDS` |
