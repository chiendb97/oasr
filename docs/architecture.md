# Engine Architecture — Extension Points

OASR's inference engine is built around **one registry per extension axis** so new
model architectures, decode families, streaming runtimes, batching policies,
checkpoint formats, tokenizers, and feature frontends plug in by *subclassing a base
+ registering* —
never by editing the engine core. This mirrors the `model_executor` split in
vLLM / SGLang, adapted to ASR (an acoustic **encoder** feeding a **decode** stage
that is either non-autoregressive CTC/Paraformer or an autoregressive
transducer/AED/LLM loop).

## The seven seams

| Axis | Base class | Registry / builder | Selected by |
|------|-----------|--------------------|-------------|
| Encoder architecture | `oasr.models.BaseEncoder` / `BaseAsrModel` | `oasr.models.registry` (`register_model`, `build_model_from_checkpoint`) | native format → `architecture=` override → `CheckpointConverter.detect` (see `docs/checkpoints.md`) |
| Checkpoint format | `oasr.models.registry.CheckpointConverter` | same registry; `oasr.from_pretrained` resolves local dir / HF Hub id | `converter.detect()` |
| Decode family | `oasr.engine.decode.DecodeStrategy` | `oasr.engine.decode` (`register_decode_strategy`, `build_decode_strategy`) | `EngineConfig.decode_method` validated against `model.capabilities`, else `model.default_decode_type` (+ `config.decoder_type` splits CTC into `ctc_cuda` / `ctc_wfst`) |
| Streaming runtime | `oasr.engine.streaming_backend.StreamingEncoderBackend` | `oasr.engine.streaming_backend` (`register_streaming_backend`, `build_streaming_backend`) | `model.encoder.streaming_kind` |
| Batching | `oasr.engine.batching.BatchingPolicy` / `PartitionPolicy` | `oasr.engine.batching` (`register_*_policy`, `build_*_policy`) | `config.schedule_policy` / partition flags |
| Tokenizer | `oasr.tokenizers.Tokenizer` | `oasr.tokenizers` (`register_tokenizer`, `build_tokenizer`) | converter-emitted `TokenizerSpec.kind` (see `docs/tokenizers.md`) |
| Feature frontend | `oasr.features.ExtractorSpec` | `oasr.features` (`register_extractor`, `build_extractor`) | `FeatureConfig.feature_type`, materialized from the converter-emitted `FeatureSpec` |

## The layer waist

Orthogonal to the seven registries — which are about *what plugs in* — is the
one about *what every architecture is built from*. Model implementations use
`oasr.layers`, never `nn.Linear` / `nn.LayerNorm` / `nn.Embedding` directly.
That is what makes a kernel improvement, CUDA-graph capture or a future
quantized path apply to all six architectures instead of one.

| What | Use |
|---|---|
| Projections | `Linear`, and the TP-shaped `ColumnParallelLinear` / `RowParallelLinear` where the shard axis is unambiguous (QKV and gate/up are column, output and down are row) |
| Fused GEMM epilogue | `LinearActivation` — `relu` / `swish` / `gelu_tanh` only |
| Normalization | `LayerNorm`, `RMSNorm`, `BiasNorm`, `AddLayerNorm`, … with the eps conventions named (`TORCH_EPS`, `ESPNET_EPS`, `QWEN2_RMS_EPS`) |
| Embeddings | `Embedding` (alias `VocabParallelEmbedding`) |
| Attention compute | `Attention` — takes projected, head-split q/k/v; the projections stay on the model under their checkpoint's names |
| Position-wise FFN | `FeedForward`, `GatedMLP` — where the upstream layout already nests them under a name |
| Rotary | `NeoxRotaryEmbedding` + `apply_rotary_pos_emb` for HF-style per-row positions; `RotaryEmbedding` for the complex `freqs_cis` form |

Each layer owns **both** a kernel path and a torch path and chooses in
`oasr.layers._backend`. The kernels have preconditions the reference does not
— GEMM needs CUDA, fp16/bf16 and both dimensions 8-aligned; norms need CUDA and
a contiguous input; the CuteDSL FMHA cannot compile every head dim — so the
choice is per call, not per model. That is what lets the same model file serve
fp16 on a 5090 and run the fp32 CPU parity oracle.

Capability is necessary but not sufficient, and the two measured policies that
make up the rest live here rather than in any model:

* **Attention fuses only when there is a mask to fuse.** With `kv_lens` or an
  additive bias the fused kernel is 1.7–1.9× faster than SDPA; with no mask at
  all it is 1.25–1.87× *slower*, because the win was the fusion. Whisper, whose
  attention is never masked, therefore runs SDPA end to end without any
  model-side flag.
* **GEMM has a work floor** (`GEMM_MIN_MACS`). Reaching CUTLASS costs a fixed
  ~20 µs of Python; a batched encoder forward amortizes that, an eager
  autoregressive decode step does not. The rule is a pure function of the call
  — deliberately *not* relaxed under CUDA-graph capture, even though the cost
  is free there: a capture-dependent branch makes the graph pick a different
  kernel than eager, and that one-ulp fp16 difference reached the transducer
  decoder as different tokens the first time it was tried.

Net effect on the WER benchmark: Conformer and Zipformer within run-to-run
noise, whisper-tiny p50 +5%, and every architecture now on the kernels at all.
The residual is fixed per-call dispatch on the smallest model in the repo — the
worst case for any waist — and its two fixes (cheaper GEMM dispatch, CUDA-graph
capture of the AR step) are tracked separately.

`OASR_LAYERS_BACKEND` overrides it process-wide: `torch` never calls a kernel
(the debugging fallback — a numerical difference that survives it is not the
kernels' fault), `oasr` raises instead of degrading for GEMM and norm (how you
*prove* a model is on the kernel path rather than assuming it).

`tests/test_layer_waist.py` is the ratchet: it builds every registered
architecture tiny on CPU, walks `named_modules()`, and fails on a bare torch
layer. Its tiny-config table is keyed off `list_models()`, so a new
architecture with no entry fails rather than going uncovered.

Two gaps are deliberate and named rather than papered over. Dense `nn.Conv1d`
stems (Whisper-geometry encoders, the CIF alpha head) have no `oasr.layers`
counterpart yet, so convolutions are not banned. And `fc1`/`fc2` stay flat
`Linear`s in HF Whisper and the Qwen2-Audio tower: composing a `FeedForward`
there would insert a level into every checkpoint key to save one `F.gelu`.

## Data flow

```
Request → InputProcessor (fbank) → Scheduler (BatchingPolicy + PartitionPolicy)
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
   whole tick, and step cost is model-dependent (measured: ~1.5 ms/step for
   whisper-tiny at `B=8` vs ~18 ms/step for Qwen2-Audio-7B at `B=4`), so a step
   count alone lets one model's tick run 10× longer than another's.  Working
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
`(B, chunk, C)` tensor: `GraphedEncoderForward` takes the *callable* and never
inspects its output, so `consumes` never decides which optimisations a family
gets.  It used to — capture was gated on `consumes == "log_probs"`, which made
streaming transducer ~3.5× slower than it needed to be for no reason beyond a
hardcoded output-buffer name.

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

## Paradigm status (all five wired)

| Paradigm | Model package | Strategy | Mode |
|---|---|---|---|
| CTC (GPU prefix-beam / WFST) | `conformer`, `zipformer` | `ctc_cuda` / `ctc_wfst` | offline + streaming |
| Transducer (RNNT greedy) | `transducer` (icefall converter, explicit `architecture=`) | `transducer` (`consumes="hidden"`) | offline + streaming |
| CTC+AED rescoring (U2++) | `conformer` (decoder branch kept) | `ctc_aed_rescoring` (`consumes="both"`, opt-in via `decode_method`) | offline |
| AED (Whisper) | `whisper` (HF converter) | `aed` (incremental greedy) | offline |
| Paraformer (NAR) | `paraformer` (FunASR converter) | `paraformer` (one-shot, CIF timestamps) | offline |
| LLM-ASR (Qwen2-Audio) | `speech_llm` (HF converter) | `llm` (incremental greedy, token-streaming partials) | offline |

Per-request `DecodingOptions` (`oasr.engine.DecodingOptions` — n-best, generation
cap, sampling knobs, LLM prompt override) ride on `Request` and through the
serving front-end; engine-level knobs stay on `EngineConfig`.

The Conformer (paged) and Zipformer (stateful) streaming backends are both wired.
The stateful backend **batches** ready streams: when the encoder exposes
`stack_streaming_states` / `unstack_streaming_states` (Zipformer does), all
same-chunk-length streams run as one `B = N` forward (3.5–24× over the previous
sequential `B = 1` loop at pool sizes 4–32).  Encoders with
`streaming_kind="none"` are rejected in streaming service mode.

Deferred follow-ups (measured rationale in `.artifacts/multi_paradigm.md`):
AED/transducer beam search, paged decoder-KV via the CuteDSL FMHA (blocked on the
masked-tile fix for heavily key-padded rows — exactly the left-padded LLM prompt
shape), and decoder-step CUDA graphs (the prerequisite — capacity-preallocated
static KV buffers in `Qwen2Lm` — has landed).
