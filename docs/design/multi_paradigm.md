# Multi-Paradigm ASR Support — Design Proposal

| | |
|---|---|
| **Status** | Proposal — awaiting review |
| **Date** | 2026-07-13 |
| **Baseline** | `main` @ `339395d` (code references below are accurate at this commit) |
| **Related** | `docs/architecture.md` (current seams), `docs/engine.md`, `docs/serving.md`, `docs/scheduler.md` |

This document assesses the current OASR inference architecture and proposes the changes
required to support the full ASR paradigm matrix — CTC, attention-encoder-decoder (AED),
transducer (RNNT), non-autoregressive Paraformer, and LLM-based ASR — across encoder
architectures (Conformer, Zipformer, SANM/Paraformer, Whisper, speech-LLM towers), with
unified checkpoint loading and conversion from the WeNet, icefall, HuggingFace, and FunASR
ecosystems.

**Summary.** OASR already has the right skeleton: five registry seams (model, checkpoint
converter, decode strategy, streaming backend, batching policy), CTC fully wired, and a
working offline transducer greedy path. What blocks the full paradigm matrix is not the seam
design but six concrete gaps: (1) the streaming path is hardwired to the fused
encoder+CTC-head forward — `DecodeStrategy.consumes` is honored only offline; (2) there is
no bounded autoregressive *generation phase* — an AR loop must run to completion inside one
engine step, which would stall the GIL-owning serving dispatcher; (3) there is no tokenizer
abstraction — just a `units.txt` join with hardcoded special ids; (4) feature extraction is
Kaldi-fbank-only and engine-defaulted, not checkpoint-derived; (5) checkpoint conversion
emits only `(config, weights, aux)` — no tokenizer/feature/decoding metadata, no native
round-trippable format, silent weight drops; (6) models are single-objective — hybrid
CTC+AED checkpoints silently lose their decoder branch. The proposal keeps the existing
seams and the CTC fast path bit-for-bit intact, and fills the gaps in five phases:
**foundations → transducer GA → AED (rescoring first, then Whisper) → Paraformer → LLM-ASR**.

---

## 1. Goals and non-goals

**Goals**

- Serve all five decode paradigms through the existing engine: CTC (today), transducer,
  AED (both U2++-style CTC+attention rescoring and full AR decoding, e.g. Whisper),
  non-autoregressive Paraformer, and LLM-based ASR (audio encoder + projector + LLM).
- Support new encoder architectures (SANM, Whisper encoder, LLM audio towers) purely by
  adding a model package + converter — no engine-core edits (preserve the
  `docs/architecture.md` extension contract).
- Unified checkpoint loading: `oasr.from_pretrained` works for WeNet, icefall, HuggingFace,
  and FunASR checkpoints, with a native round-trippable on-disk format.
- Tokenization, feature extraction, and decoding metadata travel *with the checkpoint*
  instead of being engine-side defaults or path sniffing.
- Zero regression on the CTC production paths (throughput, latency, and decode output).

**Non-goals**

- Training or fine-tuning support (inference-only framework).
- Changes to the serving topology (still one process / one engine per GPU) or to the wire
  protocol (the existing `Event::Partial/Final{text, tokens, scores}` already fits AR
  output; extensions like timestamps are additive and deferred to the last phase).
- Delegating LLM decode to an external engine (vLLM/TRT-LLM). Kept as a documented escape
  hatch (§8.3), not built in this proposal.
- Streaming AED/Whisper (not genuinely streamable; rejected at admission instead).

---

## 2. Current architecture (baseline)

```
Request → InputProcessor (Kaldi fbank/mfcc, batched GPU)
        → Scheduler (BatchingPolicy select + PartitionPolicy split; streaming pool)
        → ModelRunner
             ├ offline:   model.forward_offline[_packed]  → log-probs   (CTC fused path)
             │            model.encode_offline            → hidden      (iff strategy.consumes == "hidden")
             └ streaming:  StreamingEncoderBackend.forward_step → log-probs ONLY (fused head, CUDA-graphed)
        → OutputProcessor (facade) → DecodeStrategy (+ Detokenizer) → RequestOutput{text, tokens, scores, finished}
```

The five registries and what is actually wired at the baseline commit:

| Seam | Contract | Wired today |
|---|---|---|
| Model architecture | `BaseAsrModel` / `BaseEncoder` / `BaseHead` (`oasr/models/base.py`) | `conformer` (WeNet), `zipformer` (icefall); `oasr/models/transducer/` exists but is **not registered** (no `register_model`, not imported) |
| Checkpoint format | `CheckpointConverter` protocol — `detect` / `build_config` / `build_aux` / `load_state_dict` (`oasr/models/registry.py`) | `WenetConverter`, `IcefallConverter`; HF Hub *download* exists (`loaders.py` → `huggingface_hub.snapshot_download`) but there is no HF *format* converter |
| Decode family | `DecodeStrategy` + `consumes ∈ {"log_probs","hidden"}` (`oasr/engine/decode/base.py`) | `ctc_cuda`, `ctc_wfst` (full offline+streaming); `transducer` (offline greedy, batched); `aed` / `llm` are registered raising skeletons |
| Streaming runtime | `StreamingEncoderBackend` (`allocate`/`forward_step`/`free` + geometry) | `paged` (Conformer, CUDA-graphed), `stateful` (Zipformer, B=1 per stream), `none` |
| Batching | `BatchingPolicy` + `PartitionPolicy` | `fcfs`/`bucket`/`sjf` × `count`/`frames`/`packing` |

This is a good foundation. The proposal is an **evolution of these seams**, not a rewrite.

---

## 3. Limitations assessment

### 3.1 Engine: the streaming path structurally excludes non-CTC decoding

- `DecodeStrategy.consumes` is consulted in **exactly one place** —
  `OfflineExecutor._run_stage` (`oasr/engine/executor/offline.py:179`). Both streaming
  backends unconditionally call the fused-head entry points (`model.forward_chunk_paged` in
  `streaming_backend/paged.py`, `model.streaming_forward` in `streaming_backend/stateful.py`),
  and the CUDA-graph capture bakes in a `(B, chunk, vocab)` `log_probs_out` buffer
  (`engine/graph_cache.py`). `model.encode_chunk_paged` — the hidden-states streaming entry
  on `BaseAsrModel` — has **zero callers** in the tree. Streaming transducer/AED/LLM is
  therefore impossible without engine changes, regardless of what a strategy implements.
- **No generation phase.** Offline requests admit-and-finalize inside a single `step()`;
  the transducer strategy consequently runs its entire AR loop inside `decode_offline`
  (`engine/decode/transducer.py`). Tolerable for frame-synchronous greedy; disqualifying
  for LLM decode: the Rust serving dispatcher runs
  `drain cmds → Python::with_gil { admit; engine.step(); extract }` synchronously per tick
  (`rust/crates/oasr-engine-client/src/dispatcher.rs`) — an unbounded generation inside one
  `step()` holds the GIL and starves every other stream, all partials, pings, and admission.
  There is also no continuous batching across requests in a decode phase, and no preemption
  (`docs/scheduler.md`: block-pool exhaustion fails hard).
- Sequence packing is silently ignored on the hidden path (`encode_offline_packed` has no
  engine caller; `_run_stage` routes `consumes=="hidden"` to plain `encode_offline`).

### 3.2 CTC assumptions leak into shared components

- The detokenizer hardcodes `SPECIAL_IDS = frozenset([0, 1, 2])` (blank/unk/sos-eos) and
  strips them from **every** family's output (`engine/decode/detokenize.py:20`) — a
  transducer or AED vocab using ids 1/2 as real tokens silently loses them. The
  SentencePiece model is loaded but deliberately unused for decoding (its piece ids differ
  from the CTC unit ids).
- `EngineConfig.decoder_type` validates only `{ctc_cuda, ctc_wfst}`;
  `ctc_decoder_config` / `wfst_decoder_config` are unconditionally instantiated even for
  non-CTC models (`engine/config.py.__post_init__`). The transducer strategy reads
  `config.transducer_max_sym_per_frame` via `getattr(..., 10)` — a knob that **does not
  exist** on `EngineConfig` (`engine/decode/transducer.py:52`).
- `finalize_silence_pad` ("trailing silence decodes to blanks") lives in the input
  processor and paged backend (`engine/input_processor.py`, `streaming_backend/paged.py`) —
  a CTC-blank assumption outside any strategy.

### 3.3 Model layer: single objective, static decode selection

- `BaseAsrModel` holds exactly one `head` plus an optional `decoder`, and `decode_type`
  selects exactly one path. Hybrid models — WeNet U2/U2++ is *precisely* CTC +
  attention-decoder rescoring — cannot be represented: both converters **silently drop**
  the attention-decoder / transducer branches (`ConformerModel.load_weights` keeps only
  `encoder.*` + `ctc.ctc_lo.*`, `models/conformer/model.py:1081-1091`;
  `ZipformerModel.load_weights` likewise). No warning names the capability being lost.
- `decode_type` is a static model property; there is no per-request choice (greedy vs beam
  vs rescoring) and no notion of a model exposing *multiple* decode capabilities.
- `BaseHead.forward` is `(B,T,D) → (B,T,V)` frame-synchronous — Paraformer's
  predictor→NAR-decoder pipeline (output length U ≠ T) doesn't fit the head contract; and
  `BaseDecoder.step` is a minimal single-token entry with no prefill, no batched-step, and
  no KV-cache contract.

### 3.4 Tokenization: no abstraction at all

Text production is a `units.txt` id→piece join + `▁→space`
(`engine/decode/detokenize.py`). Discovery is engine-side path sniffing over `ckpt_dir`
(`EngineConfig.__post_init__`: any `*.model` → `sentencepiece_model`; first of
`units.txt`/`words.txt` → `unit_table`) that misses icefall's `tokens.txt`/`bpe.model`
entirely — zipformer checkpoints get **no symbol table** unless the user passes one
explicitly. Nothing exists for AED/LLM needs: real BPE decode, byte fallback, HF
`tokenizer.json`, Whisper's tokenizer, or prompt/special-token *encoding* (`encode()`
does not exist anywhere).

### 3.5 Feature extraction: Kaldi-only, engine-defaulted

`FeatureConfig` supports `fbank|mfcc` via torchaudio/kaldifeat/fused-CUDA backends only.
Missing: Whisper log-mel (n_fft 400 / hop 160, global max-normalization, 30 s pad/trim),
raw-waveform passthrough (wav2vec-style), and LFR frame stacking (Paraformer consumes
80×7 = 560-dim LFR features). Worse, the config is an **engine-side default** —
`EngineConfig.__post_init__` sets `FeatureConfig(dither=0.0)` and never consults the
checkpoint; an 8 kHz or 40-mel checkpoint would silently get 80-mel/16 kHz features. CMVN
is baked into the model as a `GlobalCMVN` layer (a fine mechanism, but converter-specific
and undeclared as part of a feature pipeline).

### 3.6 Checkpoint layer: narrow emission, no native format, fragile detection

- Converters emit only `(config, aux, state_dict)`. Tokenizer, feature spec, and decoding
  defaults (blank/sos/eos ids) do not travel with the conversion — they are re-derived (or
  lost) engine-side.
- **No native OASR on-disk format**: nothing writes/reads a self-describing bundle;
  `ConformerModelConfig.from_dict` / `ZipformerModelConfig.from_dict` are dead code in the
  load path.
- Detection is fragile: first-`detect()`-wins in registry insertion order; icefall's
  detect is "any `.pt` and no `train.yaml`" (over-claims); and `resolve_architecture`
  **falls back to `"conformer"`** when nothing matches (`models/registry.py`) —
  misdetection instead of a clear error.
- Dependency hygiene: `huggingface_hub`, `sentencepiece`, and PyYAML are used but
  undeclared; `setup.py` and `pyproject.toml` extras diverge (`wfst`,
  torchaudio/kaldifeat only in `pyproject.toml`).

### 3.7 Cache and serving: reusable primitives, wrong policies for AR

- `BlockPool`'s free-list allocator and the per-slot block-table + `cache_seqlens`
  mechanism are exactly the vLLM-style primitives an AR decoder KV cache needs — but
  `AttentionCacheManager` is chunk-quantized (one block per encoder chunk) with
  **sliding-window eviction** (`_evict_oldest`) — the opposite of AR decode's monotonic
  growth-to-EOS.
- The serving wire (`Event::Partial/Final{text, tokens, scores}`) fits AR output with
  **no protocol change**; the real constraint is the dispatcher's bounded-work-per-tick
  cadence (§3.1). Token timestamps exist at the CPU-decoder level
  (`SearchInterface.times`) but are dropped by the GPU fast paths and never reach the wire
  (`result_end_time` is always `None` in `oasr-server-grpc`).
- `docs/engine.md` §10 is stale (describes editing `OutputProcessor`/`ModelRunner` rather
  than the registries) and contradicts `docs/architecture.md`.

---

## 4. Recommended changes (keystones)

Six keystone changes, ordered by leverage. §5–§7 elaborate them.

**K1 — Capability-typed model composition (replace "one head, one decode_type").**
A model becomes a composition of optional components — `encoder` +
`heads: Dict[str, BaseHead]` + `predictor` (CIF) + `decoder` (AR) + `joiner` — and exposes
`capabilities: FrozenSet[str]` (e.g. `{"ctc", "aed_rescoring"}` for U2++) with a
`default_decode_type`. Engine-facing compute contracts become `Protocol`s (`SupportsCtc`,
`SupportsTransducer`, `SupportsAed`, `SupportsParaformer`, `SupportsLlmDecode`) that decode
strategies type-check against. The existing `head` / `decode_type` properties remain as
compatibility aliases; the fused `forward_offline` / `forward_chunk_paged` CTC fast path is
untouched.

**K2 — A bounded generation phase in the engine (two-phase execution).**
Keep the encode phase exactly as-is (it is the mature, CUDA-graphed asset). Add an
*incremental decode protocol* to `DecodeStrategy` — `begin_offline()` / `advance(budget)` /
`has_pending()` — so label-synchronous families (AED, LLM) run **N bounded decoder steps
per engine tick** with continuous batching across requests, instead of looping to EOS
inside one step. Frame-synchronous families (CTC, WFST, transducer greedy, Paraformer NAR)
keep the one-shot path. This preserves the serving dispatcher's tick contract with zero
Rust changes.

**K3 — Route `consumes` through streaming.**
`ModelRunner` and both streaming backends produce what the active strategy declares:
`"log_probs"` (fused head, today's graph-captured path), `"hidden"` (`encode_chunk_paged`
/ encoder-only `streaming_forward`), or `"both"` (hidden + head applied — needed for
CTC+AED rescoring). CUDA-graph capture is parameterized on the output spec instead of
hardcoding `log_probs_out`.

**K4 — A sixth registry axis: `Tokenizer`.**
`decode(ids)→str`, `encode(text)→ids`, `special_ids`, `vocab_size`; implementations
`symbol_table` (today's units.txt behavior, bit-compatible), `sentencepiece`,
`huggingface`, `whisper`. Selected by a `TokenizerSpec` **emitted by the checkpoint
converter**, never by path sniffing. `Detokenizer` becomes a thin adapter for backward
compatibility.

**K5 — Checkpoint conversion emits a complete bundle, with a native on-disk format.**
`CheckpointConverter.convert()` returns `ConvertedCheckpoint = (model config, weights, aux,
TokenizerSpec, FeatureSpec, DecodingDefaults)`. A native format (`oasr_config.json` +
`model.safetensors` + tokenizer assets) makes any converted model round-trippable and is
detected *first* and unambiguously; the `"conformer"` fallback becomes a deprecation
warning, then an error. An `oasr convert` CLI materializes conversions offline.

**K6 — Decoder-side KV cache manager.**
A new `DecoderKVCacheManager` reuses `BlockPool` (separate pool instance with decoder
geometry) with append-per-step growth, per-request block tables, **no eviction**, plus
one-shot dense cross-attention KV computed at prefill. This is the storage layer under K2,
and it feeds the existing paged FMHA kernels (which already support `block_table` + GQA —
the CuteDSL FMHA and `pack_gqa` machinery are ready assets for an in-tree LLM decoder).

Supporting changes: per-request `DecodingOptions` (n-best, max-new-tokens,
language/task/prompt; engine-level knobs like the CTC kernel beam width stay engine-level),
`FeatureSpec` as checkpoint-derived truth with explicit override, weight-load accounting
(`LoadReport` with declared-expected-drops — no more silent discards), and `RequestOutput`
gains `timestamps` / `finish_reason`.

---

## 5. Module and interface design

### 5.1 Module tree (additions ★, changes Δ)

```
oasr/
├── models/
│   ├── base.py                Δ  heads: Dict[str, BaseHead]; capabilities; predictor slot; compat aliases
│   ├── interfaces.py          ★  SupportsCtc / SupportsTransducer / SupportsAed / SupportsParaformer / SupportsLlmDecode
│   ├── registry.py            Δ  ModelEntry unchanged; detection precedence + explicit-arch override
│   ├── loaders.py             Δ  from_pretrained → native-format first, then converters
│   ├── decoders/              Δ  base.py (+ prefill/step batched contract) ★ transformer_decoder.py (AED)
│   ├── heads/                    ctc.py (unchanged)
│   ├── conformer/             Δ  keep U2++ decoder weights; emit TokenizerSpec/FeatureSpec
│   ├── zipformer/             Δ  same; fix icefall detect + tokens.txt/bpe.model
│   ├── transducer/            Δ  register_model + icefall pruned-transducer converter
│   ├── paraformer/            ★  SANM encoder, CIF predictor, NAR decoder, FunASR converter
│   ├── whisper/               ★  encoder+decoder, HF converter, 30 s frontend
│   └── speech_llm/            ★  audio encoder + projector + LLM decoder (HF converter)
├── tokenizers/                ★  base.py, symbol_table.py, sentencepiece.py, huggingface.py, whisper.py, registry
├── checkpoints/               ★  bundle.py (ConvertedCheckpoint), native.py (read/write), convert.py (CLI)
├── features/                  Δ  FeatureSpec; extractor registry; whisper_logmel; LFR transform; raw passthrough
├── cache/                     Δ  decoder_kv.py ★ (DecoderKVCacheManager over BlockPool)
└── engine/
    ├── decode/                Δ  base.py (incremental protocol, consumes="both"); ★ rescoring.py, paraformer.py; aed/llm implemented
    ├── generation/            ★  StepBudget, Hypothesis/Beam structs, batched AR step driver
    ├── streaming_backend/     Δ  consumes-aware forward_step; graph capture parameterized on output spec
    ├── model_runner.py        Δ  EncodeOutput{hidden?, log_probs?, lengths}
    ├── config.py              Δ  decode_method, per-family knobs declared; FeatureSpec-derived features
    └── request.py             Δ  DecodingOptions; RequestOutput += timestamps, finish_reason
```

### 5.2 Key interfaces (sketches; signatures follow existing conventions)

**Model composition & capabilities (K1)**

```python
class BaseAsrModel(nn.Module, ABC):
    encoder: BaseEncoder
    heads: Mapping[str, BaseHead]            # {"ctc": CTCHead, ...}; may be empty
    decoder: Optional[BaseDecoder]           # AR families
    predictor: Optional[BasePredictor]       # Paraformer CIF; None otherwise

    @property
    def capabilities(self) -> FrozenSet[str]:
        """Decode families this checkpoint supports, e.g. {"ctc", "aed_rescoring"}."""

    @property
    def default_decode_type(self) -> str: ...

    # compat aliases (deprecation-warned for one release):
    @property
    def head(self) -> BaseHead: return self.heads["ctc"]
    @property
    def decode_type(self) -> str: return self.default_decode_type
```

```python
@runtime_checkable
class SupportsAed(Protocol):
    sos_id: int
    eos_id: int

    def decoder_prefill(self, enc_out, enc_lens, prompt_ids, kv: "DecoderKV") -> None:
        """Compute cross-attn KV once; seed self-attn KV with the prompt."""

    def decoder_step(self, tokens: Tensor, kv: "DecoderKV") -> Tensor:
        """(B_active,) last tokens → (B_active, V) logits; appends to kv."""


@runtime_checkable
class SupportsParaformer(Protocol):
    def predict(self, enc_out, enc_lens) -> Tuple[Tensor, Tensor]:
        """CIF → (acoustic_embeds (B, U, D), token_lens (B,))."""

    def nar_decode(self, enc_out, enc_lens, acoustic_embeds, token_lens) -> Tensor:
        """→ (B, U, V) log-probs, one parallel pass."""
```

`SupportsTransducer` formalizes what `TransducerDecodeStrategy` already uses
(`decoder` / `joiner` / `blank_id`); `SupportsCtc` is the existing head fusion.

**Incremental decode protocol (K2)** — additive on the existing ABC
(`oasr/engine/decode/base.py`):

```python
class DecodeStrategy(ABC):
    decode_type: ClassVar[str]
    consumes: ClassVar[str] = "log_probs"     # "log_probs" | "hidden" | "both"
    incremental: ClassVar[bool] = False       # True ⇒ label-synchronous AR (AED/LLM)

    # existing one-shot + streaming methods unchanged ...

    # -- incremental protocol (only when incremental=True) -------------------
    def begin_offline(self, requests: List[Request], enc: "EncodeOutput") -> None:
        """Prefill: stash encoder output, init beams + decoder KV per request."""

    def advance(self, budget: "StepBudget") -> List[RequestOutput]:
        """Run ≤ budget batched decoder steps across ALL active requests
        (continuous batching); return partials + finals produced this tick."""

    def has_pending(self) -> bool: ...
```

`OfflineExecutor.step()` becomes: **(1)** `strategy.advance(budget)` if anything is
pending → **(2)** admit + collate + encode a new micro-batch → **(3)** frame-sync:
one-shot decode (today's path, unchanged); incremental: `begin_offline`, requests stay
`RUNNING` across steps and count against a decode-slot budget in the scheduler. One engine
tick therefore always does bounded work — the dispatcher, wire protocol, and
`Event::Partial` cadence need no changes, and LLM requests get token-streaming partials
for free.

**Tokenizer axis (K4)**

```python
class Tokenizer(ABC):
    vocab_size: int
    special_ids: FrozenSet[int]                      # replaces the {0,1,2} hardcode

    def decode(self, ids: Sequence[int]) -> str: ...
    def encode(self, text: str) -> List[int]: ...    # AED prompts / LLM / hotwords
```

**Checkpoint bundle (K5)**

```python
@dataclass
class ConvertedCheckpoint:
    architecture: str                       # registry key
    model_config: BaseModelConfig
    aux: Dict[str, Any]                     # e.g. GlobalCMVN
    state_dict: Mapping[str, Tensor]        # or lazy safetensors iterator
    tokenizer: TokenizerSpec                # kind + files + options
    features: FeatureSpec                   # kind, sr, mels, LFR, normalization
    decoding: DecodingDefaults              # default family, blank/sos/eos, special_ids
```

Old 4-method converters keep working via an adapter that fills tokenizer/features/decoding
with today's sniffing behavior.

**Decoder KV cache (K6)** — `DecoderKVCacheManager(pool: BlockPool)`:
`create(request, max_new_tokens) → slot`, `append_step(slots)` (per-token growth,
allocates a block when the current one fills), `block_tables(slots)`, `free(slot)`.
Backed by a **separate** `BlockPool` instance shaped
`(num_decoder_layers, blocks, block_tokens, n_kv_head, head_dim)`; cross-attention KV is
dense per request (fixed length, computed once at prefill).

### 5.3 Paradigm mapping

| Paradigm | Model package (arch axis) | Strategy (decode axis) | `consumes` | Streaming? |
|---|---|---|---|---|
| CTC (Conformer/Zipformer) | existing | `ctc_cuda` / `ctc_wfst` — **unchanged** | `log_probs` | ✓ today |
| Transducer (RNNT) | `transducer/` (registered) + encoder reuse | `transducer` greedy (exists) + streaming + beam | `hidden` | ✓ (frame-sync; K3) |
| CTC+AED rescoring (U2++) | conformer + `TransformerDecoder` | ★ `ctc_aed_rescoring` — CTC n-best + **one teacher-forced decoder forward** (not AR) | `both` | offline; streaming-final later |
| AED (Whisper) | `whisper/` | `aed` — incremental beam/greedy | `hidden` | ✗ (reject at admission) |
| Paraformer | `paraformer/` (SANM + CIF + NAR decoder) | ★ `paraformer` — argmax over `(B, U, V)`; **no AR loop** | `log_probs` (opaque: the model's `forward_offline` returns token-logits + token-lens) | offline first; chunked-CIF later |
| LLM-ASR (Qwen2-Audio-style) | `speech_llm/` (encoder + projector + LLM) | `llm` — incremental generation | `hidden` | offline + token-streaming partials |

Note how cheaply Paraformer fits: because the offline executor already passes an opaque
`(tensor, lengths)` pair to the strategy, a Paraformer model whose `forward_offline` runs
encoder→CIF→NAR-decoder internally needs **zero engine changes** — the work is the model
package, the FunASR converter, LFR features, and the tokenizer. Its CIF weights also yield
token timestamps essentially for free.

---

## 6. Separating the six concerns

| Concern | Owner | Selected by | Travels via |
|---|---|---|---|
| Model architecture (encoder/predictor/decoder modules) | `oasr/models/<arch>/` | model registry key | `oasr_config.json: architecture` |
| Training objective (CTC head, joiner, AED decoder, LM head) | `heads` / `decoder` / `joiner` components + capability protocols | `model.capabilities` | model config + weights |
| Decoding algorithm | `DecodeStrategy` registry (+ per-request `DecodingOptions`) | `config.decode_method` ∈ `model.capabilities`, else `default_decode_type` | `DecodingDefaults` in the bundle |
| Tokenization | `oasr/tokenizers/` registry | `TokenizerSpec.kind` | converter-emitted `TokenizerSpec` |
| Feature extraction | `oasr/features/` extractor registry | `FeatureSpec.kind` | converter-emitted `FeatureSpec` |
| Checkpoint loading | `oasr/checkpoints/` converters | native format first, else `detect()` | `ConvertedCheckpoint` bundle |

Three principles make the separation real rather than nominal:

1. **Objective ≠ decoding.** A CTC-trained model supports greedy, prefix-beam, and WFST
   decoding; a hybrid supports CTC-only *or* rescoring. Hence `capabilities` (what the
   weights can do) is distinct from `decode_method` (what this deployment/request chose) —
   today's conflation into a single `decode_type` is what forces converters to throw
   hybrid branches away.
2. **Metadata flows with the checkpoint; the engine consumes specs.** Every "which
   tokenizer / which features / which blank id" question is answered at conversion time
   and serialized in the bundle. The engine never sniffs `ckpt_dir` paths again (the
   current sniffing both misses icefall files and mislabels — §3.4). Explicit
   `EngineConfig` overrides remain, but a spec-vs-override mismatch logs loudly.
3. **Model = tensors, strategy = control flow.** Capability protocols define batched
   tensor ops (`decoder_step`, `predict`, `nar_decode`); strategies own hypothesis
   bookkeeping, beams, budgets, and sessions. This is what lets one `aed` strategy serve
   Whisper and a future icefall AED model, and one `llm` strategy serve different
   projector+LLM combinations.

The payoff is the extension matrix: a new **architecture** = model package + converter (no
engine edits); a new **decode family** = strategy + protocol (no model edits beyond
implementing the protocol); a new **ecosystem** = converter only.

---

## 7. Checkpoint compatibility and conversion layer

### 7.1 Resolution pipeline

```
from_pretrained(id_or_path, architecture=None, ...)
  1. HF Hub id?  → snapshot_download (existing)
  2. Native format?  oasr_config.json present → load directly (fast path: safetensors, no conversion)
  3. architecture= override → that converter, no sniffing
  4. Converter sniff, specific markers only (below); ambiguity or no match → error listing candidates
     (the silent "conformer" fallback becomes a deprecation warning for one release, then an error)
  5. Optional: `oasr convert <src> <dst>` writes the native bundle for reuse
```

### 7.2 Per-ecosystem converters

| Ecosystem | Detect (specific markers) | Reads | Emits (beyond config+weights) | New work |
|---|---|---|---|---|
| **WeNet** | `train.yaml` (existing) | `train.yaml`, `final.pt`, `global_cmvn`, `units.txt` | TokenizerSpec(symbol_table: units.txt), FeatureSpec(from `dataset_conf` fbank params — **stop defaulting engine-side**), CMVN aux | Keep `decoder.*` (U2++ bi-transformer) → `TransformerDecoder` for the rescoring capability; WeNet-transducer variant later |
| **icefall** | `tokens.txt`/`bpe.model` or `exp/` + `epoch-*.pt`/`pretrained.pt` (tighten the current "any .pt" rule) | ckpt (shape-inferred config — existing), `tokens.txt`, `bpe.model` | TokenizerSpec(sentencepiece or tokens.txt), FeatureSpec(fbank80) | Pruned-transducer converter: map `encoder.*` / `decoder.*` (stateless predictor) / `joiner.*`; declared-drop `simple_am_proj` / `simple_lm_proj` |
| **HuggingFace** | `config.json` with `model_type`/`architectures` | `config.json`, `model.safetensors`, `tokenizer.json`, `preprocessor_config.json` | per-model | ★ `HFWhisperConverter` (encoder/decoder mapping tables, whisper_logmel FeatureSpec, Whisper tokenizer); ★ `HFSpeechLlmConverter` (Qwen2-Audio-style: audio tower + projector + LM); Wav2Vec2-CTC optional later (needs raw-waveform frontend) |
| **FunASR (Paraformer)** | `config.yaml` with a paraformer model class | `config.yaml`, `model.pt`, `am.mvn`, `tokens.json`/`seg_dict` | TokenizerSpec(char/seg_dict), FeatureSpec(fbank80 + **LFR 7/6** + CMVN from `am.mvn`) | ★ full converter + SANM/CIF weight maps |

### 7.3 Weight mapping and validation discipline

- Keep the vLLM-style split that already works: converters translate *config*;
  `Model.load_weights` owns name mapping via explicit tables + `_load_from_state_dict`
  hooks (the Conformer v1→v2 permutation and fused-QKV remaps are good precedents). Add a
  small `WeightMap` helper (ordered exact/prefix/regex rules) for the bigger HF maps.
- **Kill silent drops**: `load_weights` returns a `LoadReport{mapped, dropped, missing}`;
  converters declare `expected_unused_prefixes` (e.g. icefall's `simple_*_proj`); anything
  dropped outside that list logs a warning naming the capability being lost.
- Every converter ships a **logit-parity harness**: run the reference implementation
  (WeNet / icefall / `transformers` / FunASR) and OASR on the same inputs, assert
  encoder-output and final-logit parity within fp tolerance. The repo already does exactly
  this for Conformer (vs WeNet SDPA) and Zipformer (bit-exact vs icefall) — extend the
  pattern per converter, plus tiny (~1–5 MB) golden checkpoint fixtures for CI so
  converter drift against upstream formats is caught mechanically.
- Native format: `format_version` field from day one; safetensors for weights (lazy,
  mmap-able, no pickle); tokenizer assets copied verbatim into `tokenizer/`.
- Dependency packaging: new extras `oasr[hub]` (`huggingface_hub`, `safetensors`) and
  `oasr[tokenizers]` (`sentencepiece`, `tokenizers`); declare PyYAML; make
  `pyproject.toml` the single source and sync/strip `setup.py`'s stale `extras_require`.

---

## 8. Migration risks, backward compatibility, trade-offs

### 8.1 Top risks and mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **CTC hot-path regression** — the fused forward + CUDA-graph streaming path is the production asset | High | Capability protocols and `consumes` routing are strictly additive; the `log_probs` path keeps identical code and graph shapes. Gate every phase on `bench_engine` offline+streaming CTC subroutines within noise (paired-bench protocol for power-cap drift) plus a decode-identical oracle vs a pinned baseline — the approach that made the last engine refactor zero-regression. |
| 2 | **Offline request lifecycle change** (incremental requests stay `RUNNING` across steps) touches scheduler/RLock/abort paths | Medium | Only `incremental=True` strategies opt in; frame-sync offline keeps admit-and-finalize-in-one-step. Extend `test_engine_concurrent.py` stress to mixed frame-sync+incremental loads. |
| 3 | **Dispatcher starvation by AR decode** | High if unmitigated | The `StepBudget` in K2 is the design answer; add a serving-level test asserting tick p99 under mixed CTC+LLM load. No Rust changes needed. |
| 4 | **Detokenizer behavior change** (special-id handling) alters production transcripts | Medium | The `symbol_table` tokenizer defaults to today's exact behavior (`special_ids={0,1,2}`, `▁`-join); changes apply only when a converter emits a different spec. |
| 5 | **Detection precedence change** breaks loosely-structured checkpoint dirs | Low | One-release deprecation warning before the "conformer" fallback becomes an error; `architecture=` escape hatch. |
| 6 | **Converter drift vs upstream** (WeNet/icefall/HF/FunASR formats evolve) | Medium | Pin tested versions in docs; golden tiny-fixture CI per converter; parity harness runnable against a live reference install. |
| 7 | **VRAM pressure** — decoder KV pool + encoder pool + LLM weights | Medium | Separate pools sized independently; reuse the WFST lazy-commit VMM arena pattern for the decoder pool if needed; document per-model budgets. |
| 8 | **Python-loop AR overhead** (the transducer per-emit row loop is already scalar-ish) | Medium | Acceptable initially (GIL serializes anyway; CUDA overlaps); vectorize the emit path in Phase 1; decoder-step CUDA graphs in Phase 5. |
| 9 | **CUDA-graph capture matrix growth** (output-spec × B × cache buckets) | Low-Medium | Hidden-mode streaming starts eager; capture added only for shapes that prove hot. |
| 10 | **Streaming semantics per family** — AED/Paraformer aren't genuinely streamable | Low | Admission-time capability check (extends the existing `service_mode` validation); clear error instead of wrong results. |

### 8.2 Backward-compatibility contract

- `oasr.from_pretrained`, `EngineConfig` field names, `transcribe` / `transcribe_offline`,
  the Rust wire protocol, and the CTC decoder CLI flags all keep working unchanged.
- Deprecated-but-functional for one release: `model.head` / `model.decode_type` (aliases),
  the 4-method converter protocol (adapter), path-sniffed tokenizer discovery (used only
  when the bundle lacks a spec), the `"conformer"` detect fallback (warning).
- `EngineConfig.decoder_type` keeps its CTC meaning; a new `decode_method: Optional[str]`
  selects among `model.capabilities` (`None` = model default). `oasr-server` gains
  `--decode-method` mirroring it; existing flags untouched.

### 8.3 Explicit trade-off decisions (with recommendations)

1. **Engine-owned generation executor vs strategy-owned loop** → *Hybrid (recommended)*:
   the incremental protocol keeps the loop inside the strategy (simple, per-family
   freedom) while the engine owns the budget and tick cadence. A fully engine-owned
   `GenerationRunner` (vLLM-style) is a heavier rewrite with payoff only if OASR later
   needs cross-family fused decode batches — not worth it now.
2. **Capability protocols + composition vs paradigm subclasses**
   (`CtcModel`/`TransducerModel`/…) → *Protocols (recommended)*: hybrids (CTC+AED,
   transducer+CTC) make a subclass lattice explode; composition matches what the code
   already does (`getattr(model, "decoder", None)`).
3. **In-tree LLM decode vs delegating to vLLM/TRT-LLM** → *In-tree, bounded scope
   (recommended)*: OASR already owns paged FMHA (block tables + GQA), rotary embedding
   layers, tuned GEMMs, and a `BlockPool` — greedy/beam for ≤~7B decoders is a modest
   addition and keeps the single-process, single-GPU serving story. Keep an
   external-engine adapter as a documented escape hatch for big decoders; don't build it
   until a concrete model demands it.
4. **Native format vs reconvert-on-every-load** → *Native format (recommended)*: costs a
   versioning discipline, buys deterministic loads, faster startup (safetensors mmap),
   serving hosts that don't need WeNet/icefall/transformers installed, and an unambiguous
   `detect`.
5. **`consumes="both"`** costs an extra head forward + output buffer when rescoring —
   negligible offline; for streaming rescoring the graph must emit two buffers, which is
   why streaming rescoring is deferred to a "final-only rescoring" variant.
6. **Separate decoder-KV pool vs sharing the encoder pool** → *Separate (recommended)*:
   block geometry differs (tokens vs encoder frames, layer counts, head dims); sharing
   saves little and couples eviction policies that must differ.

---

## 9. Phased implementation plan

Sizing: S ≈ days, M ≈ 1–2 weeks, L ≈ several weeks of focused work. Every phase ends with
two standing gates: **(G1)** CTC decode-identical vs pinned baseline + `bench_engine`
offline/streaming within noise; **(G2)** full `pytest tests/` green.

**Phase 0 — Foundations (M).** Tokenizer axis + `TokenizerSpec`; `FeatureSpec` emitted by
both existing converters and consumed by `EngineConfig` (override + mismatch warning);
`ConvertedCheckpoint` bundle protocol + adapter; native format read/write + `oasr convert`
CLI; detection precedence + `LoadReport` accounting; extras/deps cleanup. *No engine
behavior change.*
*Tests:* seam-level unit tests (no GPU): tokenizer registry, bundle round-trip
(`convert → native → load` decode-identical to direct load), detection-precedence matrix,
LoadReport warnings on hybrid checkpoints. Extends the existing `test_engine_seams.py` /
`test_model_registry.py` pattern.

**Phase 1 — Transducer GA (M-L).** Icefall pruned-transducer converter + `register_model`
+ package import; wire `consumes` through `ModelRunner` and both streaming backends (eager
hidden mode first); streaming greedy (predictor context + frame pointer threaded via
`create_session` / `free_session`); vectorize the per-emit row loop; declare
`transducer_max_sym_per_frame` on `EngineConfig`; transducer beam search (optional, can
trail).
*Tests:* WER parity vs icefall greedy on a real checkpoint; batched-vs-reference-loop
parity (extend `test_transducer.py`); streaming-vs-offline token parity on identical
audio; `bench_engine --subroutines transducer_offline|transducer_streaming`.

**Phase 2 — Label-synchronous foundation + AED (L).**
*2a:* `TransformerDecoder` module (self-attn KV + cross-attn prefill) +
`DecoderKVCacheManager`.
*2b:* **U2++ attention rescoring** — the WeNet converter keeps `decoder.*`;
`ctc_aed_rescoring` strategy (`consumes="both"`, CTC n-best + one teacher-forced decoder
forward + score fusion). Deliberately first: it is *not* autoregressive, needs no budget
machinery, and is an immediate accuracy win on checkpoints users already run.
*2c:* incremental protocol + `StepBudget` in the offline executor + scheduler decode
slots; **Whisper**: HF converter, `whisper_logmel` frontend, Whisper tokenizer, `aed`
strategy (greedy + beam).
*Tests:* rescoring WER parity vs the WeNet runtime; Whisper greedy token-exact
(fp-tolerant) vs `transformers` on a clip suite; budget-scheduling unit tests (bounded
steps/tick, fairness across requests); dispatcher tick-latency test under mixed CTC+AED
load.

**Phase 3 — Paraformer (M-L, parallelizable with Phase 2** since it needs no AR
infrastructure**).** FunASR converter (`config.yaml` / `model.pt` / `am.mvn` / tokens);
SANM encoder + CIF predictor + NAR decoder; LFR feature transform; `paraformer` strategy;
CIF-derived token timestamps into `RequestOutput.timestamps`.
*Tests:* CER parity vs FunASR reference on AISHELL-style data; timestamp sanity checks;
offline RTF benchmark.

**Phase 4 — LLM-based ASR (L).** `speech_llm` package (encoder + projector + LLM decoder
using the paged FMHA / rotary / GEMM assets); HF converter (Qwen2-Audio-style); HF
tokenizer + prompt template handling; `llm` strategy over the incremental protocol (greedy
first, sampling knobs via `DecodingOptions`); token-streaming partials; VRAM budgeting.
*Tests:* greedy transcript parity vs HF `generate`; sustained mixed-load
latency/starvation test; memory-budget regression test.

**Phase 5 — Productization (M).** Per-request `DecodingOptions` through serving (map the
already-present proto fields: `max_alternatives`, `confidence`, `result_end_time` ←
timestamps); decoder-step CUDA graphs; stateful-backend batching; `--decode-method` on
`oasr-server`; docs refresh (rewrite the stale `docs/engine.md` §10 against
`docs/architecture.md`; new `docs/checkpoints.md`, `docs/tokenizers.md`).

**Priority rationale.** Transducer first because it is 80% built, low-risk, and unlocks
the streaming-AR wiring (K3) that everything later reuses. Rescoring second because it is
a cheap, high-value accuracy win that exercises the new converter/bundle machinery on
existing checkpoints. Whisper third as the flagship for the HF converter + AR generation
core. Paraformer runs in parallel (independent NAR track). LLM last — it composes
everything (HF conversion, tokenizers, generation budget, decoder KV) and is the least
valuable to rush.

**Testing strategy (cross-phase).** Three tiers, all with existing precedents in `tests/`:
(1) seam/contract tests, CPU-only, no checkpoints (`test_engine_seams.py`,
`test_model_registry.py`, `test_streaming_backend.py`); (2) golden parity oracles vs
reference implementations (`test_conformer.py` vs WeNet SDPA, `test_zipformer.py`
bit-exact vs icefall, `test_transducer.py` batched-vs-reference-loop,
`test_ctc_decoder_fused_parity.py` A/B variants); (3) end-to-end engine + bench gates
(`test_pipeline.py`, `bench_engine.py` subroutines per family, zero-regression rule on
CTC).

---

## 10. Open questions (decisions wanted before Phase 2)

1. **In-tree LLM decode vs external-engine adapter** (§8.3-3). Recommendation: in-tree,
   bounded scope. Affects Phase 4 only; does not block Phases 0–1.
2. **Streaming rescoring in scope?** (§8.3-5). Recommendation: defer; ship "final-only
   rescoring" for streaming CTC sessions and full rescoring offline.
3. **Should `from_pretrained` auto-materialize the native bundle** (write-through cache on
   first conversion) or only via the explicit `oasr convert` CLI? Recommendation: explicit
   CLI only, to keep loads side-effect-free; revisit if conversion time becomes annoying.
