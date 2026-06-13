# Engine Architecture — Extension Points

OASR's inference engine is built around **one registry per extension axis** so new
model architectures, decode families, streaming runtimes, batching policies, and
checkpoint formats plug in by *subclassing a base + registering* — never by
editing the engine core. This mirrors the `model_executor` split in vLLM / SGLang,
adapted to ASR (an acoustic **encoder** feeding a **decode** stage that is either
non-autoregressive CTC or an autoregressive transducer/AED/LLM loop).

## The five seams

| Axis | Base class | Registry / builder | Selected by |
|------|-----------|--------------------|-------------|
| Encoder architecture | `oasr.models.BaseEncoder` / `BaseAsrModel` | `oasr.models.registry` (`register_model`, `build_model_from_checkpoint`) | checkpoint architecture (`CheckpointConverter.detect`) |
| Checkpoint format | `oasr.models.registry.CheckpointConverter` | same registry; `oasr.from_pretrained` resolves local dir / HF Hub id | `converter.detect()` |
| Decode family | `oasr.engine.decode.DecodeStrategy` | `oasr.engine.decode` (`register_decode_strategy`, `build_decode_strategy`) | `model.decode_type` (+ `config.decoder_type` for CTC) |
| Streaming runtime | `oasr.engine.streaming_backend.StreamingEncoderBackend` | `oasr.engine.streaming_backend` (`register_streaming_backend`, `build_streaming_backend`) | `model.encoder.streaming_kind` |
| Batching | `oasr.engine.batching.BatchingPolicy` / `PartitionPolicy` | `oasr.engine.batching` (`register_*_policy`, `build_*_policy`) | `config.schedule_policy` / partition flags |

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
  path) or `"hidden"` (autoregressive families drive `model.decoder` themselves).
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

**Add a decode family** (e.g. RNNT / AED / LLM):
1. `class FooDecoder(BaseDecoder)` (`oasr.models.decoders`) — `init_state` +
   `step`. Transducers compose a `PredictionNetwork` + `Joiner`.
2. `@register_decode_strategy("transducer")` on a `DecodeStrategy` with
   `consumes="hidden"`, driving `model.decoder.step(...)` token-by-token over a
   decoder-side KV/state cache (reuse `oasr.cache.BlockPool` /
   `AttentionCacheManager` for AED/LLM self/cross-attention KV).
   The `transducer` / `aed` / `llm` names already resolve to documented skeletons
   in `oasr/engine/decode/skeleton.py` — replace the body.

**Add a streaming runtime:** `@register_streaming_backend("my_kind")` on a
`StreamingEncoderBackend` (`allocate` / `forward_step` / `free` + window
geometry); have the encoder report `streaming_kind = "my_kind"`.

**Add a batching / partition policy:** `@register_batching_policy("my")` /
`@register_partition_policy("my")`; set `config.schedule_policy` or the partition
flags.

**Add a checkpoint format:** implement `CheckpointConverter` (`detect` /
`build_config` / `build_aux` / `load_state_dict`) + `register_model`;
`from_pretrained` auto-detects it for both local dirs and HF Hub ids.

## Status of the autoregressive path

CTC (GPU prefix-beam + WFST) is fully wired across offline + streaming. The
transducer / AED / LLM seam is defined end to end — `BaseDecoder`,
`DecodeStrategy(consumes="hidden")` skeletons, and the `encode_*` hidden-state
entry points — but the autoregressive decode *loop* (decoder-side KV cache,
beam/greedy search, per-token batching) is an extension point, not yet
implemented. The Conformer (paged) and Zipformer (stateful) streaming backends
are both wired; Zipformer's stateful streaming is validated at the backend level
(state threading) — full engine-level streaming additionally needs a streaming
Zipformer checkpoint.
