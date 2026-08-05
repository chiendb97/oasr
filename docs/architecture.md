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

One declared gap remains, kernel-side:

| Gap | Missing | Costs |
|---|---|---|
| `fmha-head-dim` | head dims so wide that even a 1-deep cp.async ring overflows smem (>256 on a 99 KB arch) | nothing in-tree reaches it |

`fmha-mask-form` was here and is closed. The kernel masked keys by per-row
*length* only — `[0, len)` — so a per-row key **start** had no form to arrive in
and left padding (HF's masked-generate convention, which is what a batched LLM
prompt is) could not be expressed at all. It now takes a second `(B,)` vector,
`mCacheSeqStarts`, compared against the column index in the same mask predicate
that already handled the length; when the caller passes no starts the wrapper
hands over a zero-rank dummy and the predicate const-folds away, so the existing
path compiles to the same kernel it did before. Verified against SDPA across
start-inside-a-tile / start-past-a-whole-tile / both-ends / wide-head / bf16
shapes, and composed with causal.

The interesting part was the *routing*, not the kernel. `is_causal` was already
implemented and deliberately routed to SDPA (`fmha-causal-short`) because SDPA
has a flash path for it and the fused path has a fixed ~68 µs floor. But causal
*combined* with a window is the opposite case: SDPA refuses `is_causal` alongside
`attn_mask`, so the caller must materialize a `(B, 1, T_q, T_k)` tensor and
forfeits flash with it. Measured on Qwen2-Audio-7B prefill geometry, bf16, with
the strides the real call site produces — **1.80–3.29× faster fused on the
attention op**. That is the op, not the model: a 7B prefill layer is dominated by
its d=3584 GEMMs, so the same change is **1.03–1.05×** over the whole 32-layer
prefill and **1.013×** over an engine `transcribe_offline` with a short
generation, transcript-identical. Both of those were measured with the arms
**interleaved**, which matters — a single-order A/B first read 0.876× and that was
the second arm benefiting from a warm allocator, not the fused path losing.

The crossover is at ~1 G MACs (`FMHA_CAUSAL_WINDOW_MIN_MACS`), and the two shapes
either side of it land on the same ratio despite different B, H, P and D — which
is what a fixed floor being amortized predicts and a shape rule would not. One
LJSpeech utterance at B=1 or B=2 falls below it and stays on SDPA, so the win
lands on batched prefill, which is the serving case.
`Qwen2Lm.prefill` now hands the core its window instead of building the mask;
`step` keeps its explicit mask, for a reason that is about the *cache* rather
than the mask (in capacity-preallocated mode its K/V are `k_buf[:, :, :t]`
slices, which the fused kernel would copy whole, once per layer per step).

One consequence worth knowing: a query row whose entire window is padding comes
back **zero** from the kernel (its documented empty-row clamp) where SDPA's math
backend gives NaN. Zero is the safe answer — a NaN pad row is not inert, because
in the next layer a masked key still contributes `0 * NaN` and poisons the *real*
rows. The torch path reaches the same end differently, by keeping the diagonal
open, so both backends agree on every row a caller can legitimately read.

### What a pass over upstream FlashAttention's CuteDSL kernels changed

Read against `flash_attn/cute` (`block_info.py`, `mask.py`, `seqlen_info.py`,
`interface.py`), three differences mattered.

**FA bounds the K loop at both ends; this kernel only bounded the top.**
`BlockInfo.get_n_block_min_max` computes `n_block_min` from the window-left edge
as well as `n_block_max` from the diagonal. Ours ran to block 0 always, so a
left-padded batch loaded, MMA'd and then `-inf`-masked every block below its
start — the exact waste that made unbounded causal *slower* than SDPA. Fixed.
Measured at Qwen2 prefill geometry, cost is now proportional to what is actually
attended: with the bound 258/214/183/164 µs at 0/25/50/75 % padding, against a
flat ~257 µs without it. Zero padding costs the same either way, so nothing on
the existing path regressed.

**FA expresses a per-row key start as a tensor offset, not a mask.**
`SeqlenInfo.offset_k` and `PagedKV.leftpad_k` shift the K/V base so padded rows
are never touched, and FA has no start predicate at all. That is the better
answer where it applies, but it depends on FA's **bottom-right** causal
alignment: shifting K also shifts the diagonal. This kernel is top-left aligned
to match `torch`'s `is_causal`, which every parity test compares against, so a
K-only offset would move the diagonal out from under the mask. Keeping the
predicate and bounding the loop gets the block-level work back without changing
the convention.

**FA never slices a KV cache — and that was costing us more than the mask.**
FA requires only `stride(-1) == 1` and passes the tensor's real `dim_order()`
into `mark_compact_shape_dynamic`; our `_ensure_canonical` demands strictly
C-order strides and copies otherwise. A `k_buf[:, :, :t]` capacity slice has a
stride *gap*, so it is not expressible either way — which is why FA passes the
**whole buffer plus `cache_seqlens`** and lets the length bound the loop. Doing
the same (`Attention(kv_extent=...)`, `Qwen2Lm._append_kv` returning buffer +
length) is bit-identical and **1.23–1.54×** on prefill geometry, **1.45–1.88×**
at a decode step. It also retired the reason `Qwen2Lm.step` was on SDPA: that
reason was the per-layer cache copy, and the copy was avoidable. Paired,
interleaved, on the real 7B: putting Qwen2's attention on the kernel is now
**1.082×** end-to-end at 8 new tokens and **1.109×** at 32, transcript-identical
— against 1.013× for prefill-only fusion before this pass.

That change also surfaced a precondition worth stating: **`v` must be finite
wherever the kernel can read**, which is up to the K *tile* boundary above
`cache_seqlens`, not up to the length itself. Columns past the length get zero
softmax weight and *finite* stale data there is provably inert (verified to
1e4 — that is what makes a recycled paged pool and a padded feature batch safe).
`NaN`/`Inf` are not: the weight is zero, but `0 * NaN` is `NaN` and it enters
through `P @ V` where no mask can intercept it. A `new_empty` capacity cache
therefore corrupted output — and did so *non-deterministically*, which parity
tests structurally cannot catch, since each individual run looks plausible. The
cache is zeroed now, and the guard is a zeroed-tail assertion rather than a
parity check. Predicating the load against the length, as FA does, is what would
retire the precondition entirely.

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

* **Causal stays on SDPA — measured, not missing.** The kernel does causal
  masking with per-CTA block skipping (`n_block_max` bounded by the diagonal),
  verified against SDPA. Plumbing the flag through *without* the skipping
  measured 1.4–4.8× slower, because the mask was applied per element while every
  row block still scanned all of K; adding the bound took a qwen2-prefill shape
  from 282.6 → 199.8 µs and the fused path now overtakes SDPA at T=2048. It is
  still not selected, because the fused path has a ~78 µs floor at any T
  (canonical-stride copies + wrapper) and every causal shape in this repo is
  short: Whisper's SOT prefill is 4 tokens, the WeNet decoder's teacher-forced
  pass ~40. The crossover moves when the stride requirement goes.
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

## Paradigm status (all five paradigms wired, seven architectures)

| Paradigm | Model package | Strategy | Mode |
|---|---|---|---|
| CTC (GPU prefix-beam / WFST) | `conformer`, `zipformer` | `ctc_cuda` / `ctc_wfst` | offline + streaming |
| Transducer (RNNT greedy) | `transducer` (icefall converter, explicit `architecture=`) | `transducer` (`consumes="hidden"`) | offline + streaming |
| CTC+AED rescoring (U2++) | `conformer` (decoder branch kept) | `ctc_aed_rescoring` (`consumes="both"`, opt-in via `decode_method`) | offline |
| AED (Whisper) | `whisper` (HF converter) | `aed` (incremental greedy) | offline |
| Paraformer (NAR) | `paraformer` (FunASR converter) | `paraformer` (one-shot, CIF timestamps) | offline |
| LLM-ASR (Qwen2-Audio) | `speech_llm` (HF converter) | `llm` (incremental greedy, token-streaming partials) | offline |
| Transducer, recurrent predictor (Nemotron ASR) | `nemotron` (HF converter) | `transducer` (`consumes="hidden"`, greedy only) | offline |

The last row is the one that reshaped an interface rather than adding a leaf.
Its predictor is a 2-layer LSTM, and an LSTM state cannot be recomputed from
the last `k` labels the way the icefall predictor's can — which is what the
greedy loop assumed when it shifted a `(B, context_size)` int tensor itself.
The state is now opaque to the strategy behind
`TransducerPredictor.init_state` / `predict` / `advance` / `stack_states` /
`unstack_states` (`oasr/models/decoders/base.py`), so one loop serves both.
Beam search is the exception and says so: it keeps the beam's states in one
`(B, k, ctx)` buffer and gather-reorders them onto their parents, which only
expresses a label window, so `beam_size > 1` is refused at construction for a
recurrent predictor rather than silently reordering something else.

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
