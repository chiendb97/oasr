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
2. `class FooModel(BaseAsrModel)` with `from_config` + `load_weights`.
3. A `CheckpointConverter` + `register_model("foo", ...)` in the package
   `__init__`. No engine edits.

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

**Add a streaming runtime:** `@register_streaming_backend("my_kind")` on a
`StreamingEncoderBackend` (`allocate` / `forward_step` / `free` + window
geometry); have the encoder report `streaming_kind = "my_kind"`.

**Add a batching / partition policy:** `@register_batching_policy("my")` /
`@register_partition_policy("my")`; set `config.schedule_policy` or the partition
flags.

**Add a checkpoint format:** implement `CheckpointConverter` (`detect` /
`build_config` / `build_aux` / `load_state_dict`) + `register_model`;
`from_pretrained` auto-detects it for both local dirs and HF Hub ids.

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

Deferred follow-ups (measured rationale in `docs/design/multi_paradigm.md`):
AED/transducer beam search, paged decoder-KV via the CuteDSL FMHA (blocked on the
masked-tile fix for heavily key-padded rows — exactly the left-padded LLM prompt
shape), and decoder-step CUDA graphs (the prerequisite — capacity-preallocated
static KV buffers in `Qwen2Lm` — has landed).
