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

The remaining declared gaps, all of them kernel-side:

| Gap | Missing | Costs |
|---|---|---|
| `fmha-head-dim` | head dims so wide that even a 1-deep cp.async ring overflows smem (>256 on a 99 KB arch) | nothing in-tree reaches it |
| `fmha-mask-form` | FMHA has no causal mode and no left-padding mode | Whisper's prefill and Qwen2's left-padded prompts stay on SDPA |

`norm-strided-rows` was also here and is closed. Every norm kernel walks rows
as `base + row * hidden`, and the launchers checked only `stride(-1) == 1` —
both too weak and, once "fixed" to `is_contiguous()`, too strong. Too weak
because a padded row stride (`x[..., :H]` of a wider buffer, `x[:, -1]` of a
`(B, T, D)`) passes it and the kernel then reads the wrong memory *silently*.
Too strong because a **permuted dense** view — Zipformer's `(T, B, C)`
transpose — has rows that still tile memory exactly, so visiting them in memory
order gives the identical result (normalization is per-row independent and
`torch.empty_like` preserves the strides). The launchers now check the real
precondition, `IsRowDense`: trailing dim contiguous *and* rows tiling memory
with no gap or overlap. Zipformer's 100 `BiasNorm` calls per encoder forward
moved onto the kernel, and the padded cases now raise instead of lying.

One gap got closed in the kernel: the cp.async ring depth was
hardcoded at 3 stages, so smem scaled straight off `head_dim` and a 64×64 tile
at `head_dim=128` needed 112 KB against sm_120's 99 KB cap — refused outright,
while sm_80's 163 KB took it fine, which is why it read as "no head_dim-128
config" rather than as a budget bug. `FmhaSm80.select_num_stages` now sizes the
ring to the arch (2 stages, 80 KB), so Paraformer's `d_k=128` SANM attention can
reach the kernel at all — **1.16–1.34×** faster than the SDPA it was stranded
on, measured with head-split views (see the policy note below; a first
measurement on freshly allocated contiguous tensors said 2.05–2.56× and was not
representative). The budget also
now costs the *padded* head dim, matching what the layouts allocate — it used
the raw value and under-counted by a third at e.g. `head_dim=72`.

And gaps that got closed at the model layer rather than declared:
every unaligned output projection. Paraformer's 8404-token head, the
transducer joiner's 500 and the CIF alpha head's 1 are now allocated at an
aligned width (`align_out_features`) and the checkpoint is widened on load
(`pad_output_projection`) — the pattern the WeNet CTC head has used since day
one. `test_no_architecture_needs_an_unaligned_gemm` keeps them closed.

Capability is necessary but not sufficient, and the two measured policies that
make up the rest live here rather than in any model:

* **Attention fuses only when there is a mask to fuse.** Measured with
  head-split views, which is what real call sites pass: `kv_lens` shapes range
  1.16–2.10× faster, unmasked is a wash at 1.01×, and a masked shape with a
  short query extent can still *lose* at 0.90×. The fused kernel needs canonical
  row-major q/k/v, so `_ensure_canonical` copies all three (35.3 → 68.7 µs at
  `B8 H4 T500 D128`) — a per-call cost paid regardless of how little attention
  work there is, and now the highest-value FMHA item. Whisper, whose attention is
  never masked, runs SDPA end to end without any model-side flag.
* **GEMM has a row floor** (`GEMM_MIN_ROWS`). CUTLASS tiles the M axis at 128
  rows, so a GEMM with fewer rows leaves most of every tile empty and cuBLAS's
  GEMV-shaped kernel wins. This started life as a floor on *total work* and
  that was wrong: `(4, 3584, 3584)` — a Qwen2-Audio-7B decode step — is 51 M
  MACs, far above any sane work floor, and is the worst shape measured at
  **5.34× slower** than cuBLAS. The problem is the shape, not the size. The
  rule is a pure function of the call, deliberately *not* relaxed under
  CUDA-graph capture even though the dispatch cost is free there: a
  capture-dependent branch makes the graph pick a different kernel than eager,
  and that one-ulp fp16 difference reached the transducer decoder as different
  tokens the first time it was tried.

Dispatch cost used to be the third policy here and is now gone. The GEMM
launchers take the caller's N-D tensors and flatten with `FLATTENED_ROWS`
instead of making Python `reshape(-1, K)` twice, and the output is allocated
with `new_empty` varargs rather than `torch.empty` on a freshly built list.
That removed ~5 µs per call: `Linear(384, 384)` fp16 went from **1.49× slower
than `F.linear` to 1.08×**, and whisper-tiny's encoder and 30-step decode are
both at **1.01×**. Allocating in the C++ launcher was measured too and is
*worse* (`Tensor::FromEnvAlloc` plus the round trip is 2.38 µs against
`new_empty`'s 1.88), so allocation stays in Python.

`tests/test_layer_waist.py` is the ratchet: it builds every registered
architecture tiny on CPU, walks `named_modules()`, and fails on a bare torch
layer. Its tiny-config table is keyed off `list_models()`, so a new
architecture with no entry fails rather than going uncovered.

One structural gap is deliberate: dense `nn.Conv1d` stems (the
Whisper-geometry encoders, Paraformer's CIF conv) have no `oasr.layers`
counterpart yet, so convolutions are not in the banned set. And `fc1`/`fc2` stay flat
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
