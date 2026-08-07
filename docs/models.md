# Model Layer

OASR's model layer is architecture-agnostic in the vLLM / SGLang sense: the
engine touches a model only through a small base-class surface, and a new
architecture plugs in by subclassing and registering. No engine edits.

This document covers the contracts, the registry, and the seven built-in
architectures. The registry seams themselves are in
[architecture.md](architecture.md); checkpoint conversion is in
[checkpoints.md](checkpoints.md); parity and WER results are in
`.artifacts/model_validation.md`.

## The engine-facing surface

`oasr/models/base.py` defines what the engine may assume.

| Member | Meaning |
|---|---|
| `encode_offline` / `encode_chunk_paged` | Encoder only → raw hidden states |
| `forward_offline` / `forward_chunk_paged` | Fused encoder + head → log-probs (the CTC fast path CUDA-graph capture preserves) |
| `forward_offline_packed` | Gapless varlen offline forward, used when sequence packing is enabled |
| `cache_spec` | `CacheSpec` describing the streaming cache; **`None`** when `streaming_kind == "none"` |
| `decode_type` / `default_decode_type` / `capabilities` | Which decode families this checkpoint can serve (`DecodeType` in `base.py`) |
| `encoder.streaming_kind` | `"paged"` / `"stateful"` / `"none"` |
| `encoder.subsampling_rate`, `encoder.right_context` | Streaming geometry inputs |
| `load_weights(state_dict) → LoadReport` | Weight loading, reporting `mapped` / `dropped` / `missing` |

`BaseEncoder.n_kv_head` and `head_dim` are **paged-streaming only** — they are
non-abstract with a self-explaining raise, so an offline-only encoder implements
neither.

### Three optional encoder declarations

All three default to "nothing declared", so an encoder that needs none of them is
unaffected by their existence. Together they are what makes a new streaming cache
*data* rather than a new manager.

| Declaration | What it says |
|---|---|
| `streaming_state_specs` | Extra fixed-extent per-stream tensors → `CacheSpec.stream_states`. See [cache_manager.md §10](cache_manager.md). |
| `fixed_attention_window` | A **trained** attention span. The engine pre-fills the paged K/V window so `cache_seqlens` is uniform across the cohort — the precondition for one *shared* relative-position table, since a Transformer-XL table's distances are `cache + i - j` and a per-row table would run the positional projection per row. It also makes `cache_t1` constant, so the graph cache captures one graph per batch size instead of one per `(B, cache_t1 bucket)`. |
| `streaming_geometry(chunk_size)` | A front-end the generic `(chunk_size - 1) * sub + right_context + 1` window formula does not describe — a *cached causal* subsampling consumes exactly `chunk_size * sub` frames with no lookahead. Also the place to **refuse** a chunk size the encoder cannot serve; the backend calls it once at construction. |

`streaming_kind` must describe what *this config's weights* can actually do, not
what the class implements. Returning `"none"` makes streaming service mode fail
at construction with a clear message and makes `cache_spec` `None`, so no paged
pool is allocated. An encoder that over-claims builds an engine that raises on
its first request instead.

## Capabilities

`oasr/models/interfaces.py::CAPABILITIES` is a declarative table: for each decode
family, the dotted model-attribute paths it requires plus a one-line `why`.

`build_decode_strategy` calls `require_capability` once, so a checkpoint
advertising a capability it cannot serve **fails at engine construction naming
the missing members**. `tests/test_model_contract.py` validates the table against
every registered architecture, built tiny on CPU.

This is the answer to "what must a model implement to support family X".

## Registry and loading

`oasr/models/registry.py` provides `register_model`, `build_model_from_checkpoint`,
`load_checkpoint_bundle`, and the `CheckpointConverter` protocol.

- Built-ins come from one `_BUILTIN_PACKAGES` list.
- Out-of-tree architectures arrive through **`oasr.models` entry points**, so
  adding one needs no registry edit. A broken plugin warns and is skipped rather
  than taking the built-ins down.
- `oasr/models/loaders.py::from_pretrained` (re-exported as `oasr.from_pretrained`)
  resolves a local directory or a Hugging Face Hub id. `load_pretrained`
  additionally returns the tokenizer / feature / decoding specs plus a
  `LoadReport`, and is what `ASREngine` uses.

### Config deserialization

Model configs inherit one type-driven `from_dict` (`oasr.models.base.coerce_config`):
it restores `Tuple` fields, recurses into nested dataclasses, and unwraps
`Optional`. A config overrides `_from_dict_overrides` only for a *flat* encoder
dict or a *polymorphic* field — e.g. `TransducerModelConfig.encoder`, whose class
comes from the sibling `encoder_type`.

### Weight-load accounting

`load_weights` returns a `LoadReport{mapped, dropped, missing}`, built via
`LoadReport.build` (which folds `unexpected` into `dropped` and uses set
membership). The registry then warns on dropped weights according to the
converter's declarations:

| Converter declaration | Effect |
|---|---|
| `expected_unused_prefixes` | Silent — e.g. icefall `simple_*_proj` |
| `capability_drop_hints` | Named warning — e.g. the U2++ `decoder.*` rescoring branch |

## Autoregressive decoder contracts

`oasr/models/decoders/base.py` holds `BaseDecoder`, `PredictionNetwork`, `Joiner`
and `TransducerPredictor` — the transducer / AED / LLM extension points.

Two batched decoder shapes exist, and both are driven from
`oasr/engine/decode/`:

- **Frame-synchronous** (transducer): `init_state` + `step`.
- **Label-synchronous AR** (Whisper, speech-LLM): the incremental surface
  `prefill` / `step` / `select`. `select(state, idx)` is an `index_select`, so
  repeated indices are legal — which is what lets beam search both *expand* a
  prefilled batch into a `B × k` grid and reorder it onto each slot's parent,
  with no new model method.

### The transducer predictor state is opaque

`TransducerPredictor` declares `init_state` / `predict` / `advance(state, tokens, emit)` /
`stack_states` / `unstack_states`. One greedy loop therefore serves both:

- a **stateless** label-window predictor (icefall: recomputable from the last `k`
  labels), and
- a **recurrent** one (NeMo's 2-layer LSTM: not).

Two consequences:

1. A recurrent predictor's start state is **not zeros**. NeMo runs the LSTM once
   on the blank embedding (the zero row, via `padding_idx`) from a zero hidden
   state, and that response is what the first frame's joint sees.
2. `beam_size > 1` is refused at construction for such a predictor
   (`label_window_state = False`), because modified beam search keeps the beam's
   states in one `(B, k, ctx)` buffer and gather-reorders them onto their
   parents — which only expresses a label window.

`oasr/models/decoders/transformer_decoder.py` holds the WeNet/ESPnet-compatible
`TransformerDecoder` / `BiTransformerDecoder` (state-dict keys mirror WeNet's
`embed.0.weight` / `decoders.N.self_attn.linear_q.*` layout, so U2++ `decoder.*`
weights load 1:1), the `add_sos_eos` / `reverse_pad_list` helpers, and an
incremental `forward_one_step`. It is built on `oasr.layers`: WeNet's
`linear_q` / `linear_k` / `linear_v` / `linear_out` names over the shared
`Attention` core, and `feed_forward` as a `FeedForward(names=("w_1", "w_2"))`
whose ReLU folds into the GEMM epilogue. Cross-attention takes `memory_lens`
rather than a materialized memory mask — the form the fused kernel enforces
directly.

## Heads

`oasr/models/heads/ctc.py::CTCHead` wraps `oasr.layers.ctc.CtcProjection`, the
fused `log_softmax(x @ Wᵀ + b)` via `oasr.gemm_log_softmax`.

**Output projections are allocated at a GEMM-aligned width.** CUTLASS's
alignment-8 iterators mean an unaligned head cannot reach the kernel at all, so
`oasr.models.base.align_out_features` rounds the width up and `pad_output_projection`
widens the checkpoint on load, biasing the padding rows to `PAD_LOGIT` so they can
never win an argmax. `test_no_architecture_needs_an_unaligned_gemm` keeps this
closed.

## The seven built-in architectures

| Registry key | Package | Decode families | Streaming | Source format |
|---|---|---|---|---|
| `conformer` | `models/conformer/` | `ctc`, `ctc_aed_rescoring` | paged | WeNet |
| `zipformer` | `models/zipformer/` | `ctc` | stateful (config-dependent) | icefall |
| `transducer` | `models/transducer/` | `transducer` | follows the encoder | icefall (**explicit only**) |
| `nemotron` | `models/nemotron/` | `transducer` | paged | Hugging Face |
| `whisper` | `models/whisper/` | `aed` | none | Hugging Face |
| `paraformer` | `models/paraformer/` | `paraformer` | none | FunASR |
| `speech_llm` | `models/speech_llm/` | `llm` | none | Hugging Face |

`list_models()` prints this table's key column as the registry sees it at
runtime, including any out-of-tree architecture that arrived through the entry
point group.

Every architecture is built from [`oasr.layers`](architecture.md#the-layer-waist),
not from bare `nn.Linear` / `nn.LayerNorm` / `nn.Embedding`.
`tests/test_layer_waist.py` enforces it, and its tiny-config table is keyed off
`list_models()` so a newly registered architecture with no entry fails rather
than going uncovered.

### Conformer (WeNet U2/U2++)

`model.py`, `config.py`, `convert.py::WenetConverter`. `streaming_kind="paged"`;
supports sequence packing.

A U2/U2++ directory whose `train.yaml` declares a `(bi)transformer` decoder gets
the AED branch loaded as `self.decoder` (`decoder.left_decoder.*` keys map 1:1;
plain-`transformer` `decoder.decoders.*` keys are remapped into `left_decoder`),
giving `capabilities={"ctc", "ctc_aed_rescoring"}` with `default_decode_type="ctc"`.
Rescoring is opt-in via `EngineConfig.decode_method`. The decoder config — including
`sos`/`eos` (raw vocab − 1) and the trained `reverse_weight` — lives on
`ConformerModelConfig.decoder`.

### Zipformer (icefall CTC)

`model.py`, `encoder.py`, `subsampling.py`, `scaling.py`, `config.py`,
`convert.py::IcefallConverter` (which infers the config from checkpoint shapes).

`streaming_kind` is `"stateful"` only when the config is streaming-capable
(`causal=True and chunk_size > 0`), otherwise `"none"` — so a non-causal release
such as `zipformer-large-cr-ctc` is refused in streaming service mode at engine
construction rather than raising on its first request.

`ZipformerEncoder.stack_streaming_states` / `unstack_streaming_states` declare the
per-kind state batch dimensions following icefall's `streaming_decode.py`
convention (embed + conv caches dim 0; key / nonlin / value caches dim 1), which
is what lets the stateful backend batch streams into one `B = N` forward.

### Transducer (icefall RNNT)

`model.py::TransducerModel`, `decoder.py` (stateless predictor), `joiner.py`,
`config.py`, `convert.py::IcefallTransducerConverter`.
`encoder_type ∈ {"conformer", "zipformer"}` selects the acoustic front-end, and
streaming follows the encoder (paged vs stateful).

**Not auto-detected.** icefall directories sniff as `zipformer` (CTC) and hybrid
checkpoints carry both branches, so load with
`from_pretrained(dir, architecture="transducer")`. The config is shape-inferred
from `decoder.*` / `joiner.*`; `simple_*_proj` is declared-dropped and
`ctc_output.*` is a named capability hint.

### Nemotron ASR (FastConformer + RNN-T)

`encoder.py`, `subsampling.py`, `predictor.py`, `convert.py::HFNemotronConverter`
(auto-detects `config.json: model_type=nemotron3_5_asr`). Offline **and**
cache-aware streaming. Reference release:
`nvidia/nemotron-3.5-asr-streaming-0.6b`.

- 24-layer macaron Conformer over **causal depthwise-separable 8× Conv2d
  subsampling**, kept in NHWC so the projection's input columns are permuted once
  at load.
- **Transformer-XL** relative position (`rel_shift`). WeNet's convention needs no
  shift, which is why `oasr.layers.RelPositionMultiHeadedAttention` has none and
  this is a separate module.
- A `chunked_limited` attention mask that **applies offline too** — it is
  trained, not a streaming-only device: a query sees its own chunk of
  `num_lookahead_tokens + 1` frames plus `(sliding_window - 1) // chunk` earlier
  ones.
- `predictor.py` — 2-layer **LSTM** predictor + additive joint
  `head(relu(enc_proj + dec_proj))` + the language-prompt projector (a 128-wide
  one-hot concatenated onto every encoder frame, with **no residual** — its
  output replaces the hidden state).
- `TokenizerSpec(kind="huggingface", special_ids=[blank])` — blank 13087 sits
  *past* the tokenizer's 13087 pieces, so it must be filtered before the backend
  sees it.

Streaming carries four kinds of state, each on a declared axis rather than as a
special case: the three subsampling-stage tails (`streaming_state_specs`), the
per-layer post-GLU conv tail (the engine's existing `"conv"` slot cache —
`conv_kernel_size - 1` is exactly what this encoder needs), the **prefilled**
paged K/V window (`fixed_attention_window`), and the frontend frame grid (on the
extractor, not here).

Two alignment preconditions, both silent if violated, both refused at engine
construction by `streaming_geometry`:

1. `chunk_size` must be a whole number of trained attention chunks
   (`num_lookahead_tokens + 1`) — the mask groups *absolute* frame positions, so
   a partial trained chunk would need keys from the future.
2. The resulting feature window must be a multiple of the subsampling factor, so
   every stage's input length is a multiple of its stride.

### Whisper (Hugging Face)

`model.py` (encoder + decoder in HF key layout), `config.py` (including
generation-control ids), `convert.py::HFWhisperConverter` (auto-detects
`config.json: model_type=whisper`, reads `model.safetensors` +
`generation_config.json`).

Offline-only (`streaming_kind="none"`), decoded by the incremental `aed` strategy
with suppress / begin-suppress lists and the SOT prompt from the config. The
decoder exposes a batched incremental `prefill` / `step` / `select` surface.

`fc1` / `fc2` stay flat `Linear`s because HF's layout puts them on the layer, and
GELU stays exact-erf and unfused. The post-conv `transpose(1,2)` is followed by
`.contiguous()` — without it the whole residual stream carries the conv's strided
last dim.

### Paraformer (FunASR, non-autoregressive)

- `encoder.py` — SANM encoder: FSMN-memory self-attention over the shared
  `oasr.layers.Attention`, 560 → 512 first layer, sinusoidal PE, **LayerNorm
  eps 1e-12** (ESPnet convention; parity breaks at PyTorch's 1e-5).
- `predictor.py` — `CifPredictor` (CifPredictorV2) with the vectorized `cif_v1`
  prefix-sum integrate-and-fire, always fp32.
- `decoder.py` — SANM NAR decoder: FFN-first layer order, FSMN-only self-attention,
  one parallel pass over the CIF acoustic embeddings. Cross-attention takes
  encoder **lengths**, not a mask.
- `convert.py::FunASRParaformerConverter` — auto-detects `config.yaml: model: Paraformer`,
  parses `am.mvn` CMVN into synthetic `encoder.cmvn_shift` / `cmvn_scale`
  state-dict buffers so native round-trip is automatic.

CIF fire positions become per-token `RequestOutput.timestamps`.

### speech_llm (Qwen2-Audio-style LLM-ASR)

- `audio_tower.py` — Whisper-geometry encoder (×32, d=1280) with
  **key-padding-only** attention mask + `AvgPool1d(2)` + post-pool LayerNorm.
  Valid lengths follow HF's two-stage formula: `(mel − 1) // 2 + 1`, then
  `(feat − 2) // 2 + 1`.
- `llm.py::Qwen2Lm` — a faithful Qwen2 causal LM built on `oasr.layers`
  (`RMSNorm`, `NeoxRotaryEmbedding`, `GatedMLP`, `Attention`). RMSNorm accumulates
  in fp32 to the store where HF rounds before the weight multiply — the more
  accurate order, one rounding step apart under bf16. `prefill(inputs_embeds,
  valid[, capacity])` / `step` / `select` handle **left-padded variable-length**
  prompts (per-row positions = `cumsum(mask) − 1`, HF's masked-generate
  convention). With `capacity`, the per-layer K/V buffers are preallocated and
  each step writes its slot in place.
- `model.py` — module names mirror HF 1:1 (`audio_tower` / `multi_modal_projector` /
  `language_model`); `load_weights` normalizes both the 4.x (`language_model.model.*`)
  and 5.x-resave (`language_model.model.model.*`) key layouts.
- `convert.py::HFQwen2AudioConverter` — auto-detects `config.json: model_type=qwen2_audio`,
  fills omitted `text_config` fields from `Qwen2Config` defaults (the published 7B
  relies on them), and reads sharded safetensors.

Checkpoint bundles load host-side (`map_location="cpu"`) with the dtype cast
**before** the device move — an 8.4B checkpoint double-books the GPU otherwise.

## Adding an architecture

See the cookbook in [architecture.md](architecture.md#extension-cookbook). In
brief:

1. `class FooEncoder(BaseEncoder)` — `forward`, the introspection properties, and
   (for streaming) either `forward_chunk_paged` or
   `get_streaming_init_states` / `streaming_forward`.
2. `class FooModel(BaseAsrModel)` with `from_config` + `load_weights`.
3. A `CheckpointConverter` + `register_model("foo", ...)` in the package
   `__init__` — or an `oasr.models` entry point, out of tree.
4. Build it from `oasr.layers` and add a tiny config to
   `tests/test_layer_waist.py`.
