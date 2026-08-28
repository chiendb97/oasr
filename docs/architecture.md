# Engine Architecture — Extension Points

OASR's inference engine is built around **one registry per extension axis** so new
model architectures, decode families, streaming runtimes, batching policies,
checkpoint formats, tokenizers, and feature frontends plug in by *subclassing a base
+ registering* —
never by editing the engine core. This mirrors the `model_executor` split in
vLLM / SGLang, adapted to ASR (an acoustic **encoder** feeding a **decode** stage
that is either non-autoregressive CTC/Paraformer or an autoregressive
transducer/AED/LLM loop).

## The eight seams

| Axis | Base class | Registry / builder | Selected by | Detail |
|------|-----------|--------------------|-------------|--------|
| Encoder architecture | `oasr.models.BaseEncoder` / `BaseAsrModel` | `oasr.models.registry` (`register_model`, `build_model_from_checkpoint`) | native format → `architecture=` override → `CheckpointConverter.detect` | [models.md](models.md) |
| Checkpoint format | `oasr.models.registry.CheckpointConverter` | same registry; `oasr.from_pretrained` resolves local dir / HF Hub id | `converter.detect()` | [checkpoints.md](checkpoints.md) |
| Decode family | `oasr.engine.decode.DecodeStrategy` | `oasr.engine.decode` (`register_decode_strategy`, `build_decode_strategy`) | `EngineConfig.decode_method` validated against `model.capabilities`, else `model.default_decode_type` (+ `config.decoder_type` splits CTC into `ctc_cuda` / `ctc_wfst`) | [decoding.md](decoding.md) |
| Streaming runtime | `oasr.engine.streaming_backend.StreamingEncoderBackend` | `oasr.engine.streaming_backend` (`register_streaming_backend`, `build_streaming_backend`) | `model.encoder.streaming_kind` | [cache_manager.md](cache_manager.md) |
| Batching | `oasr.engine.batching.BatchingPolicy` / `PartitionPolicy` | `oasr.engine.batching` (`register_*_policy`, `build_*_policy`) | `config.schedule_policy` / partition flags | [scheduler.md](scheduler.md) |
| Tokenizer | `oasr.tokenizers.Tokenizer` | `oasr.tokenizers` (`register_tokenizer`, `build_tokenizer`) | converter-emitted `TokenizerSpec.kind` | [tokenizers.md](tokenizers.md) |
| Feature frontend | `oasr.features.ExtractorSpec` | `oasr.features` (`register_extractor`, `build_extractor`) | `FeatureConfig.feature_type`, materialized from the converter-emitted `FeatureSpec` | [features.md](features.md) |

## The layer waist

Orthogonal to the seven registries — which are about *what plugs in* — is the
one about *what every architecture is built from*. Model implementations use
`oasr.layers`, never `nn.Linear` / `nn.Conv1d` / `nn.AvgPool1d` / `nn.LSTM` /
`nn.RNN` / `nn.LayerNorm` / `nn.Embedding` directly.
That is what makes a kernel improvement, CUDA-graph capture or a future
quantized path apply to **every** architecture instead of one. The kernels
underneath are documented in [kernels.md](kernels.md).

| What | Use |
|---|---|
| Projections | `Linear`, and the TP-shaped `ColumnParallelLinear` / `RowParallelLinear` where the shard axis is unambiguous (QKV and gate/up are column, output and down are row) |
| Activation | `Gelu`, `Relu`, `Sigmoid`, and `Tanh` for standalone activation; `LinearActivation` for fused `relu` / `swish` / exact-erf `gelu` / `gelu_tanh` GEMM epilogues |
| Normalization | `LayerNorm`, `RMSNorm`, `BiasNorm`, `AddLayerNorm`, and `AddRMSNorm`; LayerNorm/RMSNorm expose fused residual-add methods (including residual passthrough for pre-norm chains), with the eps conventions named (`TORCH_EPS`, `ESPNET_EPS`, `QWEN2_RMS_EPS`) |
| Embeddings | `Embedding` (alias `VocabParallelEmbedding`) |
| Attention compute | `Attention` — takes projected, head-split q/k/v; the projections stay on the model under their checkpoint's names |
| Position-wise FFN | `FeedForward`, `GatedMLP` — where the upstream layout already nests them under a name. `GatedMLP` offers its whole gate/up/activation/multiply to `oasr.gated_mlp` as one kernel, inside a measured band |
| Rotary | `NeoxRotaryEmbedding` + `apply_rotary_pos_emb` for HF-style per-row positions; `RotaryEmbedding` for the complex `freqs_cis` form |
| Convolution | BTC-native `Conv1d`, `DepthwiseConv1d`, and `PointwiseConv1d`; `DepthwiseConv1d` accepts `(left, right)` padding and an optional fused masked residual for FSMN blocks; NHWC-native `Conv2d` / `Conv2dActivation` |
| Pooling | BTC-native `AvgPool1d`; the CUDA kernel covers symmetric padding, ceil mode, and include/exclude-pad divisors without transposing the residual stream to BCT |
| Recurrent | `LSTM` and tanh/ReLU `RNN`; PyTorch-compatible checkpoint parameters with fused CUDA inference and formula-level CPU/fp32 oracles |

### OASR is the backend; torch is one you select

OASR targets GPU inference, so `oasr.layers._backend` does **not** treat
PyTorch as a safety net. A layer that quietly reroutes an unsupported shape to
`F.linear` makes a missing kernel invisible, and an invisible gap never gets
closed. Three cases, kept apart on purpose:

| Case | What happens |
|---|---|
| **Out of scope** — CPU tensor, fp32 | Torch serves it, reported once. The framework does not target these; they exist so the upstream parity oracles can run. Not debt — there is nothing to close. |
| **In scope, no kernel** | A **kernel gap**. It must be declared in `KERNEL_GAPS` naming what is missing and *which layer has to fix it*, or the call **raises**. Declared gaps are counted and printable (`format_gap_report()`). |
| **Kernel is slower** | A performance choice, counted separately so it is never mistaken for a gap — and itself a standing argument for kernel work. |

`OASR_LAYERS_BACKEND` selects the backend: `oasr` (default) or `torch` (the
optional backend, used by the CPU oracles and as the "is this the kernels'
fault" A/B). There is no `auto`.

One kernel-side gap is declared:

| Gap | Missing | Reached by |
|---|---|---|
| `fmha-head-dim` | head dims so wide that even a 1-deep cp.async ring overflows smem (>256 on a 99 KB arch) | nothing in-tree |

`conv2d-groups` is closed: grouped and depthwise calls use the direct NHWC
kernel, dense 1×1 calls use the layout-equivalent GEMM path, and Zipformer's
subsampler stays NHWC through its ConvNeXt block. The whole conv family retains
its torch path so convolutional front-ends can run under fp32 CPU parity oracles.

`avg_pool1d` is also closed: Speech-LLM pools its BTC tower output directly,
using a vectorized `kernel=2, stride=2` path rather than the former
BTC→BCT→BTC expression. Generic pooling arguments share the same direct CUDA
family, and the layer retains a torch path for CPU/fp32 parity.

`fmha-mask-form`, `norm-strided-rows`, and `conv2d-groups` were here and are closed. So were
`fmha-head-dim`'s 128-wide case (a smem-budget bug, not a missing config) and
every unaligned output projection (closed at the *model* layer via
`align_out_features` + `pad_output_projection`, the pattern the WeNet CTC head
has used since day one). See `.artifacts/fmha_tuning.md` and
`.artifacts/layer_waist_migration.md`.

### Routing rules — capability is necessary, not sufficient

A kernel that *can* serve a shape is not automatically the one that *should*.
Three measured policies live here rather than in any model:

* **Attention fuses only when there is a mask to fuse.** The fused kernel needs
  canonical row-major q/k/v, so `_ensure_canonical` copies all three — a per-call
  cost paid regardless of how little attention work there is. Masked shapes win,
  unmasked ones are a wash, and a masked shape with a short query extent can
  lose. Whisper, whose attention is never masked, runs SDPA end to end without
  any model-side flag. Closing the canonical-stride requirement is worth more
  than further tuning this rule.
* **Causal alone stays on SDPA** (`fmha-causal-short`); **causal combined with a
  window is fused** above `FMHA_CAUSAL_WINDOW_MIN_MACS`. The kernel implements
  causal masking with per-CTA block skipping, verified against SDPA — it is a
  policy, not a gap. SDPA has a flash path for causal alone and the fused path
  has a fixed floor at any `T`; but SDPA *refuses* `is_causal` alongside
  `attn_mask`, so the windowed case must materialize a `(B, 1, T_q, T_k)` tensor
  and forfeits flash with it. The crossover is a **work** floor, not a shape one.
* **GEMM has a row floor** (`GEMM_MIN_ROWS`). CUTLASS tiles the M axis at 128
  rows, so a GEMM with fewer rows leaves most of every tile empty and cuBLAS's
  GEMV-shaped kernel wins. This is a *shape* rule, not a work rule — a 51 M-MAC
  decode-step GEMM is the worst shape measured. **It is a pure function of the
  call and is deliberately not relaxed under CUDA-graph capture**, even though
  dispatch cost is free there: a capture-dependent branch makes the graph pick a
  different kernel than eager, and that one-ulp fp16 difference has come out of
  the transducer decoder as different tokens.

The measurements behind each: `.artifacts/fmha_tuning.md`,
`.artifacts/gemm_tuning.md`.

### Two behaviours to know

* **A fully-masked query row comes back zero** from the fused kernel (its
  documented empty-row clamp) where SDPA's math backend gives NaN. Zero is the
  safe answer — a NaN pad row is not inert, because in the next layer a masked
  key still contributes `0 * NaN` and poisons the *real* rows. The torch path
  reaches the same end by keeping the diagonal open, so both backends agree on
  every row a caller can legitimately read.
* **`v` must be finite wherever the kernel can read**, which is up to the K
  *tile* boundary above `cache_seqlens`, not up to the length itself. Finite
  stale data past the length is provably inert; `NaN` / `Inf` are not, and enter
  through `P @ V` where no mask can intercept them. Capacity caches are therefore
  zeroed, guarded by a zeroed-tail assertion rather than a parity check — the
  corruption is non-deterministic, which parity tests structurally cannot catch.

`tests/test_layer_waist.py` is the ratchet: it builds every registered
architecture tiny on CPU, walks `named_modules()`, and fails on a bare torch
layer. Its tiny-config table is keyed off `list_models()`, so a new
architecture with no entry fails rather than going uncovered. It also pins that
each layer's kernel and torch paths agree on CUDA — without that, the CPU parity
oracles are evidence about nothing the server runs.

One non-migration is deliberate: `fc1` / `fc2` stay flat modules in HF Whisper
and the Qwen2-Audio tower because composing a `FeedForward` there would insert a
level into every checkpoint key. Their `fc1` is a `LinearActivation("gelu")`,
which keeps the same `weight` / `bias` keys and selects the exact-erf epilogue;
the post-conv sites use standalone `Gelu`. `gelu_tanh` remains a distinct name
for checkpoints trained with the approximation.

## Data flow

```
Request → [VAD segmenter] → InputProcessor (fbank) → Scheduler (BatchingPolicy + PartitionPolicy)
   → ModelRunner
        ├─ offline:   model.forward_offline / forward_offline_packed → enc_out
        └─ streaming: StreamingEncoderBackend.forward_step           → enc_out
   → OutputProcessor (facade) → DecodeStrategy (+ Detokenizer) → RequestOutput
```

- The **model** splits into `encoder` (acoustic) + `head` (CTC) **or** `decoder`
  (autoregressive). `encode_offline` / `encode_chunk_paged` return raw hidden
  states; the fused `forward_offline` / `forward_chunk_paged` (encoder+head →
  log-probs) are the CTC fast path that CUDA-graph capture preserves.
- A `DecodeStrategy` declares `consumes` = `"log_probs"` (CTC fused-head fast
  path), `"hidden"` (autoregressive families drive `model.decoder` themselves),
  or `"both"` (hidden + log-probs — CTC+AED rescoring); and `incremental = True`
  for label-synchronous AR families driven via the bounded
  `begin_offline` / `advance(StepBudget)` / `has_pending` protocol.
- The encoder declares `streaming_kind` (`"paged"` / `"stateful"` / `"none"`) plus
  `subsampling_rate` / `right_context` / (stateful) `streaming_chunk_frames`; the
  engine reads streaming geometry from there, not from hardcoded constants.
- The encoder also declares **what it carries across chunks**, so a new streaming
  cache is data rather than a new manager: `streaming_state_specs` (extra
  fixed-extent per-stream tensors — see `docs/cache_manager.md` §10),
  `fixed_attention_window` (a *trained* attention span, which makes the engine
  pre-fill the K/V window so one shared position table is correct), and
  `streaming_geometry(chunk_size)` (a front-end the generic window formula does not
  describe, and the place to **refuse** a chunk size it cannot serve). All three
  default to "nothing declared", so an encoder that needs none of them is unaffected
  by their existence.
- The **feature frontend** declares its streaming frame grid the same way:
  `ExtractorSpec.framing` → `StreamingFraming(span, hop, history, prefill)` plus an
  optional `streaming_fn`. `supports_streaming` is derived from it, so streamability
  cannot disagree with the arithmetic; every number the streaming feature loop used
  to hardcode as a Kaldi `snip_edges` assumption now comes from here.

## Extension cookbook

**Add an encoder architecture** (e.g. Paraformer, Branchformer):
1. `class FooEncoder(BaseEncoder)` — implement `forward`, the introspection
   properties, and (for streaming) either `forward_chunk_paged`
   (`streaming_kind="paged"`) or `get_streaming_init_states`/`streaming_forward`
   (`streaming_kind="stateful"`).
   `streaming_kind` must describe what *this config's weights* can actually do,
   not what the class implements — return `"none"` when the loaded checkpoint has
   no chunk-wise path (Zipformer does this for `causal=False`). It is the value
   the engine gates on: `"none"` makes streaming service mode fail at
   construction with a clear message and makes `cache_spec` `None` so no paged
   pool is allocated. An encoder that over-claims builds an engine that raises on
   its first request instead.
2. `class FooModel(BaseAsrModel)` with `from_config` + `load_weights`.
3. A `CheckpointConverter` + `register_model("foo", ...)` in the package
   `__init__`. No engine edits.
4. Build it from `oasr.layers` (see **The layer waist** above) and add a tiny
   config to `tests/test_layer_waist.py`. Both are enforced: the conformance
   test fails on a bare `nn.Linear`, and it fails again if your architecture
   has no tiny config to check.

**Add a decode family** (e.g. a new AR paradigm):
1. `class FooDecoder(BaseDecoder)` (`oasr.models.decoders`) — for
   frame-synchronous families `init_state` + `step`; for label-synchronous AR
   families the batched incremental surface `prefill` / `step` / `select`
   (see `oasr/models/whisper/model.py` and `oasr/models/speech_llm/llm.py`).
   Transducers compose a `PredictionNetwork` + `Joiner`.
2. Declare the family's required model surface in
   `oasr/models/interfaces.py::CAPABILITIES` — dotted attribute paths plus a one-line
   `why`.  `build_decode_strategy` validates every model against it once, so a
   checkpoint advertising a capability it cannot serve fails at engine construction
   naming the missing members, and `tests/test_model_contract.py` checks the table
   against every registered architecture (built tiny on CPU).  This is the answer to
   "what must a model implement to support family X".
3. `@register_decode_strategy("foo")` on a `DecodeStrategy`.  Frame-synchronous:
   implement `decode_offline` (one shot).  Label-synchronous AR: set
   `incremental = True` and implement `begin_offline` / `advance(StepBudget)` /
   `has_pending`.  Call `budget.take()` before every batched decoder step and
   stop when it returns `False`: the budget carries **both** a step cap
   (`EngineConfig.decode_steps_per_tick`) and a wall-clock deadline
   (`EngineConfig.max_tick_ms`), whichever binds first.  The deadline is the one
   that matters for a serving deployment — the dispatcher holds the GIL for a
   whole tick, and step cost is model-dependent — an order of magnitude between
   a tiny AED decoder and a 7B LM (`.artifacts/engine_perf.md` §3) — so a step
   count alone lets one model's tick run far longer than another's.  Working
   references: `transducer.py` (frame-sync greedy, offline + streaming sessions),
   `rescoring.py` (`consumes="both"`), `aed.py` / `llm.py` (incremental).
4. Declare the family's knobs as an **options dataclass** and point
   `options_cls` at it (`oasr/engine/decode/options.py`).  Read them through
   `self.options`; do **not** add fields to `EngineConfig`.  Each field is
   `option(default, legacy=..., doc=...)`, where `legacy` names a deprecated flat
   `EngineConfig` attribute carrying the same default — that is how the existing
   families keep their public API and their `oasr-server` flags working.
   Resolution order is defaults → legacy field → `EngineConfig.decode_options`,
   and an unknown key in `decode_options` **raises** at engine construction
   naming the valid ones.  Operators reach every knob through the generic
   `oasr-server --decode-option k=v`, typed from the declared default, so a new
   family needs no new flag.  This is what stopped `EngineConfig` growing a field
   per family (it had accumulated nine) and what stopped a Whisper engine
   constructing a CTC beam config and a WFST config it would never read.

   A knob that governs the *tick loop* rather than one family — `max_tick_ms`,
   `decode_steps_per_tick`, `max_decode_slots`, `decode_kv_budget_gib` — still
   belongs on `EngineConfig`: the executor owns those, not the strategy.

   Per-**request** options are a different axis: they live on
   `oasr.engine.DecodingOptions`, are mirrored by `oasr_wire::DecodingParams`,
   and the two key sets are asserted equal at engine startup
   (`DecodingOptions.assert_matches_wire_keys`).  Adding one on a single side
   used to give an option that was accepted and silently ignored.

**Add a streaming runtime:** `@register_streaming_backend("my_kind")` on a
`StreamingEncoderBackend` (`allocate` / `forward_step` / `free` + window
geometry); have the encoder report `streaming_kind = "my_kind"`.

The `consumes` axis is orthogonal to this one and must stay that way.  A backend
gets the decode strategy's declared `consumes` and picks the matching chunk
forward — fused `forward_chunk_paged` (encoder + head → log-probs) or
encoder-only `encode_chunk_paged` (→ hidden).  Everything downstream of that
choice, CUDA-graph capture included, must treat the result as an opaque
`(B, chunk, C)` tensor: `GraphedEncoderForward` takes the *callable*
(`chunk_forward`) and never inspects its output, so `consumes` never decides
which optimisations a family gets.  It used to — capture was gated on `consumes == "log_probs"`, which left
streaming transducer several times slower than it needed to be for no reason
beyond a hardcoded output-buffer name.

One contract the backend owns: a captured graph reuses **one output buffer per
shape key**, so a tensor it hands out is live only until the next replay at that
key.  `PagedStreamingBackend` copies exactly when a step can replay a key twice
(a `B=1` cohort plus a single, or two singles — singles always replay at `B=1`).

**Beam search** is available to both AR shapes and is worth reading before adding
a third: `decode/transducer_beam.py` (frame-synchronous) keeps hypotheses on the
**device** because it runs one beam step per encoder frame and a host-side
reorder would be Θ(T²); `decode/incremental_beam.py` (label-synchronous) keeps
them as host lists because an AR step is a full decoder forward and `k` list
copies are free next to it.  The label-synchronous one needs **no new model
method**: `select(state, idx)` is an `index_select`, so repeated indices both
expand a prefilled batch into a `B*k` grid and reorder it onto each slot's
parent.  Both are gated by the same property — beam width 1 must reproduce
greedy token-for-token — which is the only exactness oracle available without a
reference implementation.

**Add a speech detector** (e.g. Silero, MarbleNet, FSMN-VAD):
1. Subclass `oasr.vad.SpeechDetector` and implement whichever entry point your
   declared `consumes` names — `detect` for a waveform, `detect_from_asr` for a
   tensor the ASR produced. Build it from `oasr.layers` like any other model.
2. `register_vad(VadSpec(...))`. Three declarations carry consequences:
   * `consumes` decides which entry point the engine calls, and makes *"an
     ASR-derived detector cannot pre-segment"* a fact of the type — the registry
     **refuses** a spec that claims `presegment` with an ASR-derived `consumes`,
     at registration rather than at first request;
   * `modes` is the role set (`presegment` / `stream` / `posthoc`), checked
     against the engine's service mode and `vad.mode` at construction — a
     streaming `vad.mode="segment"` needs `presegment` *and* `stream`, because
     there the detector has to decide what the encoder sees, incrementally;
   * `min_silence_floor_ms` is the shortest silence the signal can distinguish
     from its own sparsity. The ASR-derived signals are peaky, and without this
     the streaming preset's 100 ms would be applied to a spike train.
3. A detector with a *trained* window declares its own `framing`; one whose grid
   is the encoder's leaves it `None` and is told `seconds_per_frame`.
4. A detector that carries state across chunks sets `stateful` and implements
   `detect_streaming` plus `stack_states` / `unstack_states`. The last pair is
   what keeps the streaming stage's detector call **batched across the pool**:
   the stage holds N opaque per-stream states and only the detector knows how to
   lay them out as a batch — the same protocol the transducer predictor uses.

The segmenter and the endpointer are **not** part of the axis — they are shared
policy, so every detector produces the same segment semantics, the same knobs and
the same events. That split is what makes a neural VAD and a CTC blank posterior
interchangeable downstream, and it is why the policy half is tested against
synthetic probability traces with no GPU and no checkpoint. See
[vad.md](vad.md).

**Add a batching / partition policy:** `@register_batching_policy("my")` /
`@register_partition_policy("my")`; set `config.schedule_policy` or the partition
flags.

**Add a checkpoint format:** subclass `BaseCheckpointConverter`
(`oasr/models/converter.py`), declare `architecture` / `source_format` /
`default_checkpoint_name` / `default_decode_type` / `detect_specificity`, and
implement three methods — `detect`, `build_config`, `load_state_dict`.  Then
`register_model`; `from_pretrained` auto-detects it for both local dirs and HF
Hub ids.

The base provides the bundle assembly (`convert()` was the same twelve lines in
all six converters), the HuggingFace weight-loading chain
(sharded safetensors → single → `.bin`, all with `weights_only=True`), and the
shared `whisper_logmel` feature spec.  Two optional hooks matter:
`build_decoding_defaults(config, ckpt_dir)` for blank/sos/eos ids, and
`build_config_for_convert(ckpt_dir, state_dict)` for formats that *infer* the
architecture from tensor shapes — overriding it reuses the weights `convert`
already loaded instead of deserialising a multi-GB checkpoint twice.

Inheriting is optional: the registry still accepts anything satisfying the
protocol, so out-of-tree converters keep working.  Out-of-tree *architectures*
need no edit here at all — declare a `oasr.models` entry point and the registry
imports it on first access.

**Add a feature frontend** (e.g. raw waveform for wav2vec, an 8 kHz telephony
recipe):
1. Write the batch function — `(padded_waveforms (B, T), lengths (B,), FeatureConfig)
   → (features (B, T', F) fp32, feat_lengths (B,))`.  LFR stacking is **not** part of
   it: the engine applies `apply_lfr_batch` over any extractor's output.
2. `register_extractor(ExtractorSpec(kind="my_kind", fn=..., ...))`.  Two declared
   properties carry real consequences, so set them deliberately:
   * `supports_streaming=False` if the frontend normalizes over a fixed window and
     therefore cannot consume a growing buffer — `prepare_streaming` then rejects the
     request with an actionable message instead of producing wrong features;
   * `window_seconds_attr="..."` names the `FeatureConfig` field holding that window.
     `FeatureConfig.fixed_window_seconds`/`_frames` read it, which is what tells the
     batching policies that **every row costs the same** regardless of its length
     (otherwise `max_offline_pad_ratio` splits batches to avoid padding waste that
     does not exist) and lets the engine reject over-long audio at admission rather
     than silently transcribing a prefix.
3. Emit `FeatureSpec(kind=...)` from the checkpoint converter so the engine
   materializes the matching `FeatureConfig` automatically.

`FeatureConfig.feature_type` is validated against the registry, so registering the
kind is what makes it legal — no edit to the config, the `InputProcessor`, or the
CUDA-graph feature cache.

## Paradigm coverage

Five decode paradigms across seven encoder architectures, all through the same
seams. Per-architecture detail is in [models.md](models.md).

| Paradigm | Model package | Strategy | Mode |
|---|---|---|---|
| CTC (GPU prefix-beam / WFST) | `conformer`, `zipformer` | `ctc_cuda` / `ctc_wfst` | offline + streaming |
| Transducer (RNNT) | `transducer` (icefall converter, explicit `architecture=`) | `transducer` (`consumes="hidden"`) | offline + streaming |
| CTC+AED rescoring (U2++) | `conformer` (decoder branch kept) | `ctc_aed_rescoring` (`consumes="both"`, opt-in via `decode_method`) | offline |
| AED (Whisper) | `whisper` (HF converter) | `aed` (incremental) | offline |
| Paraformer (NAR) | `paraformer` (FunASR converter) | `paraformer` (one-shot, CIF timestamps) | offline |
| LLM-ASR (Qwen2-Audio) | `speech_llm` (HF converter) | `llm` (incremental, token-streaming partials) | offline |
| Transducer, recurrent predictor (Nemotron ASR) | `nemotron` (HF converter) | `transducer` (`consumes="hidden"`, greedy only) | offline + streaming |

`list_models()` prints this table's model column as the registry sees it at
runtime, including any out-of-tree architecture that arrived through the
`oasr.models` entry point group.

**The first row is one capability with two decoders behind it, not two model
families.** A CTC checkpoint decodes through the GPU prefix-beam decoder by
default; `EngineConfig.decoder_type="ctc_wfst"` plus an `fst_path` (a prebuilt
`.img` or a k2 `HLG.pt`) routes the *same* checkpoint through the in-tree GPU
WFST decoder, offline and streaming alike. That split is on `decoder_type`, below
`decode_method` — see [wfst_decoder_gpu.md](wfst_decoder_gpu.md).

Two seams were reshaped rather than extended when the last row landed, and both
are worth reading before adding an eighth architecture:

* **The transducer predictor state is opaque to its strategy**
  (`TransducerPredictor.init_state` / `predict` / `advance` / `stack_states` /
  `unstack_states`), so one greedy loop serves both a stateless label-window
  predictor and a recurrent one. Beam search is the declared exception. See
  [models.md](models.md#the-transducer-predictor-state-is-opaque).
* **A streaming cache is a declaration, not a manager.**
  `streaming_state_specs`, `fixed_attention_window` + `streaming_geometry`, and
  `ExtractorSpec.framing` between them cover the four kinds of state a chunk
  carries, which is why adding them changed no other architecture. See
  [cache_manager.md §10](cache_manager.md) and [features.md](features.md).

Per-request `DecodingOptions` (`oasr.engine.DecodingOptions` — n-best, generation
cap, sampling knobs, LLM prompt override) ride on `Request` and through the
serving front-end; engine-level knobs stay on `EngineConfig`.

Both streaming backends are wired: Conformer/Nemotron (paged) and Zipformer
(stateful). The stateful backend **batches** ready streams when the encoder
exposes `stack_streaming_states` / `unstack_streaming_states`, running all
same-chunk-length streams as one `B = N` forward. Encoders with
`streaming_kind="none"` are rejected in streaming service mode.

Deferred follow-ups, each with the measurement that justified deferring it:
`.artifacts/architecture_review.md` §H11 (the autoregressive decode path) and
`.artifacts/known_issues.md` §6.
