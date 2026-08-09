# Decode Families

The decode stage is a registry axis: a `DecodeStrategy` turns encoder output into
text, and the engine selects one per deployment. Adding a paradigm is a subclass
plus a `@register_decode_strategy` — no engine edit, and no new CLI flag.

Code lives in `oasr/engine/decode/`. The registry seam is described in
[architecture.md](architecture.md); this document is the axis itself.

## Selection

```
EngineConfig.decode_method   (validated against model.capabilities)
      └─ else model.default_decode_type
             └─ CTC further splits on EngineConfig.decoder_type
                    ("ctc_cuda" | "ctc_wfst")
```

`build_decode_strategy` calls `require_capability` once, so a checkpoint
advertising a family it cannot serve fails **at engine construction**, naming the
missing model members — see [models.md § Capabilities](models.md#capabilities).

## What a strategy declares

| Declaration | Meaning |
|---|---|
| `consumes` | `"log_probs"` — the fused encoder+head CTC fast path; `"hidden"` — the strategy drives `model.decoder` itself; `"both"` — hidden **and** log-probs |
| `incremental` | `True` for label-synchronous AR families driven by the bounded tick protocol |
| `options_cls` | The family's own options dataclass (see [Per-family options](#per-family-options)) |

`consumes` is orthogonal to every optimisation: a streaming backend reads it to
pick the matching chunk forward (`forward_chunk_paged` → log-probs, or
`encode_chunk_paged` → hidden) and everything downstream, CUDA-graph capture
included, treats the result as an opaque `(B, chunk, C)` tensor.

## The built-in families

| Name | File | `consumes` | Mode | Notes |
|---|---|---|---|---|
| `ctc_cuda` | `ctc_gpu.py` — `CtcGpuDecodeStrategy` | `log_probs` | offline + streaming | Owns the GPU prefix-beam state. See [ctc_decoder_gpu.md](ctc_decoder_gpu.md). |
| `ctc_wfst` | `ctc_wfst.py` — `CtcWfstDecodeStrategy` | `log_probs` | offline + streaming | Same checkpoints, in-tree GPU WFST decoder. See [wfst_decoder_gpu.md](wfst_decoder_gpu.md). |
| `transducer` | `transducer.py` — `TransducerDecodeStrategy` | `hidden` | offline + streaming | One vectorized frame-sync loop; per-request sessions (label window + predictor projection + hypothesis) threaded across chunks. |
| `ctc_aed_rescoring` | `rescoring.py` — `CtcAedRescoringStrategy` | `both` | offline | U2++: CTC n-best + one teacher-forced bitransformer pass + WeNet score fusion. Knobs: `rescoring_ctc_weight`, `rescoring_reverse_weight`. |
| `aed` | `aed.py` — `AedDecodeStrategy` | `hidden` | offline | Incremental AR over the batched decoder protocol (Whisper). |
| `paraformer` | `paraformer.py` — `ParaformerDecodeStrategy` | `hidden` | offline | One-shot NAR: `model.predict` (CIF) → `model.nar_decode` → argmax. CIF fire positions become `RequestOutput.timestamps`. |
| `llm` | `llm.py` — `LlmDecodeStrategy` | `hidden` | offline | Speech-LLM. Encodes the checkpoint's ChatML template through the tokenizer axis, splices projected audio embeddings into a **left-padded** embedded prompt, and emits token-streaming partials. |

The two incremental families share `decode/incremental.py::IncrementalArStrategy`,
which owns the group bookkeeping, the budget loop and row retirement — so `aed.py`
and `llm.py` carry only their prefill, logit filtering, EOS predicate and
partial-emission policy.

`OutputProcessor` is a thin facade over whichever strategy is active.

## Per-family options

**Do not add a field to `EngineConfig` for one family.** Each strategy declares an
options dataclass and points `options_cls` at it (`decode/options.py`), then reads
`self.options`.

```python
@dataclass
class MyOptions:
    beam_size: int = option(4, legacy="my_beam_size", doc="Beam width.")
```

`legacy=` names a deprecated flat `EngineConfig` attribute carrying the same
default — that is how the existing families keep their public API and their
`oasr-server` flags working.

Resolution order: **defaults → the legacy field → `EngineConfig.decode_options`**.
An unknown key in `decode_options` **raises at engine construction**, naming the
valid ones.

Operators reach every knob through the generic `oasr-server --decode-option k=v`,
typed from the declared default, so a new family needs no new flag. This is what
stopped `EngineConfig` growing a field per family, and what stopped a Whisper
engine constructing a CTC beam config and a WFST config it would never read.

Knobs that govern the *tick loop* rather than one family — `max_tick_ms`,
`decode_steps_per_tick`, `max_decode_slots`, `decode_kv_budget_gib`,
`decode_admit_window_ms` — stay on `EngineConfig`: the executor owns those, not
the strategy.

## The incremental AR protocol

A strategy with `incremental = True` implements three methods:

| Method | Called when |
|---|---|
| `begin_offline(...)` | the offline executor has prefilled a micro-batch |
| `advance(budget: StepBudget)` | once per engine tick, while rows remain |
| `has_pending()` | the executor asks whether to keep ticking |

Prefilled requests are parked `RUNNING` in a pending pool and advanced within a
**dual** per-tick budget (`StepBudget.for_tick`):

- at most `EngineConfig.decode_steps_per_tick` batched decoder steps, **and**
- at most `EngineConfig.max_tick_ms` of wall clock,

whichever binds first. Call `budget.take()` before every batched decoder step and
stop when it returns `False`.

**A step count alone does not bound tick time.** Step cost is model-dependent by
an order of magnitude between a tiny AED decoder and a 7B LM, and the dispatcher
holds the GIL for a whole tick, so the deadline is what actually bounds cancel
latency, admission latency and inter-partial cadence
(`.artifacts/engine_perf.md` §3).

A tick that spends its decode budget does not also prefill — bounded by
`_MAX_SKIPPED_ADMITS` so admission cannot starve. Admission is additionally gated
by `max_decode_slots` and, optionally, by `decode_admit_window_ms`: a coalescing
window that holds a thin waiting queue so near-simultaneous arrivals prefill as
**one** decode group. That matters because an AR decoder step is weight-read
bound — total forwards is the sum over groups of each group's step count, and
groups cannot be merged after the fact (both decoder surfaces keep a **shared
scalar** generation offset; per-row KV offsets are the prerequisite, shared with
paged decoder KV).

## Beam search

Both AR shapes have beam search, and the two implementations differ for a reason
worth understanding before adding a third.

| | File | Hypotheses live | Why |
|---|---|---|---|
| Frame-synchronous (transducer) | `decode/transducer_beam.py` | on the **device**, in a `(B, k, cap)` buffer | one beam step per encoder frame, so a host-side list-of-lists reorder would be Θ(T²) |
| Label-synchronous (AED, LLM) | `decode/incremental_beam.py` — `ArBeamGroup` | as host lists | an AR step is a full decoder forward, so `k` list copies are free next to it |

The label-synchronous one needs **no new model method**: `select(state, idx)` is
an `index_select`, so repeated indices both *expand* a prefilled batch into a
`B × k` grid and reorder it onto each slot's parent. A slot emitting EOS is banked
as a completed candidate rather than retiring its request; a request retires when
its whole beam is done.

Options: `beam_size`, `length_penalty`.

Both are gated by the same property — **beam width 1 must reproduce greedy
token-for-token** — which is the only exactness oracle available without a
reference implementation.

The transducer strategy is the exception that says so: `beam_size > 1` is refused
at construction for a *recurrent* predictor, because modified beam search
gather-reorders states in one `(B, k, ctx)` buffer, which only expresses a label
window. See [models.md](models.md#the-transducer-predictor-state-is-opaque).

Transducer greedy is `beam_size=1` by construction and caps per-frame emissions
with `EngineConfig.transducer_max_sym_per_frame`.

## Per-request options

`oasr.engine.DecodingOptions` rides on `Request.decoding` and is dict-coercible
at the PyO3 boundary.

| Field | Applies to |
|---|---|
| `n_best` | beam families — the executor detokenizes the top N into `RequestOutput.nbest_texts` on finals |
| `max_new_tokens` | AR families |
| `temperature`, `top_k`, `top_p` | AR families, via `generation/sampling.py::select_next_tokens`; greedy stays a batched argmax fast path |
| `prompt` | LLM only (memoized suffix re-encode); also settable deployment-wide as `EngineConfig.llm_prompt` |
| `task` (`"transcribe"` / `"translate"`) | families with a task token — AED (Whisper). **Rejected** elsewhere |
| `language` (ISO-639 primary subtag) | families with a language token — AED (Whisper). **Rejected** elsewhere |
| `word_timestamps` | families that can align, in the mode the request runs in. **Rejected** elsewhere — see [Word timings](#word-timings) |

AR finals carry `RequestOutput.finish_reason` — `"stop"` or `"length"`.

### Two classes of option

Most options describe *how thoroughly* to decode. A family that ignores one
returns the same transcript, so ignoring is a performance surprise at worst.

`task`, `language` and `word_timestamps` are different: they describe *what is
decoded*, or a field the caller will look for. A family without the control that
silently ignored them would return a transcript of something else — a different
task, a different language — with nothing in the response to say the request was
not honoured; or a response whose missing `words` array is indistinguishable
from an utterance that had no words in it.

So a strategy declares what it can act on:

```python
class AedDecodeStrategy(IncrementalArStrategy):
    selective_options: ClassVar[Tuple[str, ...]] = ("task", "language")
```

and `DecodeStrategy.validate_options` rejects anything set outside that list. It
runs at admission — from *every* entry point, the batched serving one and the
single-request `add_request` the Python API uses — so the rejection is scoped to
its own request rather than to the admit batch it was coalesced into. A family
that *can* act on an option overrides the method to add the checkpoint-level
check — whether *this* Whisper snapshot knows `<|yue|>` is a checkpoint question,
and answering it at admission is what keeps an unknown language out of a prefill
shared with unrelated requests.

Adding a selective option to a family means listing it and, if the answer depends
on the checkpoint, overriding `validate_options`. Nothing else changes.

## Word timings

`word_timestamps` fills `RequestOutput.words` with
`WordTiming(word, start, end, confidence)`, plus per-token
`RequestOutput.timestamps` and an utterance-level `confidence`. Code lives in
`decode/alignment.py` (the shared half), `decode/ctc_align.py` and
`decode/attention_align.py` (the two aligners).

### Where the spans come from

Alignment is genuinely per family; only the *second* half is shared.

**The rule is: ask the decoder, do not re-derive.** A frame-synchronous decoder
already knows *when* it emitted each token at the moment it emits it, and
recovering that afterwards costs far more than recording it — the CTC path was
briefly a forced-alignment pass and measured **ten times the cost of the decode
it decorated**, for information the beam had already had.

| Family | Where the frame comes from | What a span means |
|---|---|---|
| `ctc_cuda` | the beam **records the emitting frame beside the token** as it decodes (`ctc_decoder.cuh`: `ctime` flat, `time_storage` paged); reading it out is a copy | the frames between two emissions; spans tile |
| `ctc_aed_rescoring` | the same, indexed at the beam row that **won the fusion** | as above |
| `transducer` | the encoder frame each label was emitted at, recorded in the greedy loop | as above — the two frame-synchronous families share `emission_fields` so they cannot describe a span differently |
| `paraformer` | CIF fire boundaries — free, the predictor computes them to produce the acoustic embeddings at all | the integration window that fired that token |
| `aed` | cross-attention DTW over the checkpoint's published `alignment_heads` (`attention_align.py`) — a label-synchronous decoder has no frame index, so this is the one family where deriving is the only option | where the decoder looked while producing the token |
| `ctc_wfst`, `llm` | **none**, declared | — |

`ctc_align.py`'s `forced_align` is retained as a **test oracle**, not a decode
path: it is an independent implementation of the same question, checkable
bit-for-bit against `torchaudio.functional.forced_align`, and the kernel's
frames are validated against it. The beam commits at the *leading edge* of a
label's posterior peak — on the peak frame for about three quarters of tokens
and one frame before it for the rest, never later — which is why the confidence
lookup takes the max over `[t, t+1]` rather than reading `t` alone.

### Frames, not seconds, cross the boundary

A strategy reports `TokenAlignment` in *encoder frames*; `FrameClock` converts
using `frame_shift_ms × lfr_n × encoder.subsampling_rate` — three numbers every
architecture already declares (Conformer 40 ms, Whisper 20 ms, Paraformer 60 ms,
Nemotron 80 ms). When the geometry cannot be resolved the clock is `None` and the
request is **refused**, because a wrong seconds-per-frame produces spans that are
uniformly wrong by a constant factor and look entirely plausible.

### Words are cut out of the transcript

`word_timings()` asks the tokenizer for `token_pieces(ids)` — whose contract is
that the pieces concatenate to exactly what `decode` returns — and splits the
resulting string. So every emitted `word` is a literal substring of
`RequestOutput.text`, in order. Reassembling words from tokenizer pieces instead
would break for every kind in the tree: sentencepiece `▁`, byte-BPE `Ġ`,
FunASR's `@@` merges and CJK spacing all depend on neighbouring tokens. Runs of
space-less scripts (CJK, kana, Hangul) split per character, the convention both
FunASR and Whisper use.

`token_pieces` is a tokenizer-axis method with a correct default (drive
`decode_incremental` one id at a time) and a one-pass override where rendering is
piece-local. It exists because this grouping is **on the decode path**: the
default is quadratic in tokens for any tokenizer that has not also overridden
`decode_incremental`, and the character-by-character scan that used to sit around
it profiled larger than the CTC decode it was decorating.

### Where the pass runs

In C++ — `csrc/alignment/word_timings.cc`, bound as `oasr._C.alignment`. Emission
frames go in and words, per-token timestamps and the utterance confidence come
out in **one call per row**, so a frame-synchronous family builds no per-token
Python object at all. The step loop holds the GIL for every request the engine
finishes, and this pass used to spend it one interpreter operation per token and
per rendered character.

There is **no Python implementation and no switch to select one**. A fallback is
a slow path a deployment can land on without noticing — here, one costing more
per request than the decode itself, on the GIL-holding thread — so
`oasr/engine/decode/alignment.py` is marshalling only. The same applies to the
beam read-back and to `SymbolTableTokenizer.token_pieces`, the one tokenizer
method the grouping calls.

Nor is the extension's absence guarded against: `csrc/alignment/word_timings.cc`
is in `OASR_SOURCES`, so a `pip install -e .` that succeeded has it, and there is
no state where `oasr._C` imports but `oasr._C.alignment` does not. The one
configuration without it is `test-cpu.yml`, which compiles nothing — the module
resolves to `None` there so `import oasr` keeps working, and the tests that
exercise the pass skip.

The rule is still checked against a Python statement of itself, but that oracle
lives in `tests/test_alignment_cpp.py`, where nothing at runtime can reach it:
randomised input through both, including the whole Unicode plane for the
whitespace and space-less classifications, required to agree **exactly**.
Changing the C++ means changing the oracle in the same commit.

Two things had to be pinned rather than approximated to make that equality hold:
whitespace is Python's own `str.isspace()` (29 code points, not ASCII and not
`std::isspace`), and the confidence mean is an explicit accumulation on both
sides rather than `sum()`, which is Neumaier-compensated on CPython 3.12 and not
on 3.10 — the oracle must not agree only on the interpreter it happens to run
under.

Deferring the pass off the step loop to overlap it with the next micro-batch's
encoder forward was tried and reverted: an engine step is usually one
micro-batch, and its outputs must be complete when `step()` returns, so there is
nothing to overlap with.

Word confidence is the mean of its tokens' posteriors, and
`RequestOutput.confidence` the mean over all of them. The mean rather than the
product: a joint sequence probability decays geometrically with length, which
ranks a long correct transcript below a short uncertain one.

### Declared per family **and per mode**

```python
@property
def word_timing_modes(self) -> Tuple[str, ...]:
    return () if self._beam > 1 else ("offline", "streaming")
```

A property rather than a class attribute, for the same reason
`BaseEncoder.streaming_kind` is one: the honest answer can depend on the
configuration. A transducer times both modes under greedy and **neither** under
beam search (the beam's device-side hypothesis buffer carries labels, not
frames). An AED engine times offline only, and only with a decoder exposing
`cross_attention` — under beam search its group retains no encoder row to
re-forward against, so it declares nothing.

CTC times **both** modes, which is the direct consequence of recording rather
than deriving: a stream's log-probs are gone by the time its transcript is
final, so a forced-alignment design could not have served streaming at all.

`validate_options` refuses at admission with a message naming the modes that do
work, so the request fails rather than returning a response with the field
silently absent.

### Cost, and who pays it

Nothing runs unless a request asked. `OutputProcessor.decode_offline` drops the
request list entirely when no row in the micro-batch set the option, so an
ordinary batch takes exactly the path it took before.

The frames themselves are recorded unconditionally — one `int` write per beam
extension, and one more load/store in the copy-on-write loop that already
computes those addresses. Measured on a Conformer at batch 16, that is **within
noise** (1.710 ms → 1.704 ms over interleaved arms) and the transcripts are
identical. What is opt-in is the device→host copy and the word pass: asking for
timings costs ~4 ms on that batch, most of it the token→word grouping rather
than the timing.

The AED path is the exception: it costs one extra teacher-forced decoder forward
per finished row — deliberately *not* a hook on generation, which would put the
cost on every step of every request.

### How `task` / `language` reach the prompt

Whisper's SOT sequence is `[<|startoftranscript|>, <|lang|>, <|task|>,
<|notimestamps|>]`, built from `forced_decoder_ids` — i.e. **fixed when the
checkpoint was converted**. `WhisperModelConfig.sot_sequence(task=, language=)`
substitutes the corresponding slot, identifying it by the forced token *being* one
of the checkpoint's known task/language ids (recorded by the converter from the
tokenizer). Positions and ids move between Whisper releases — large-v3 added a
language and shifted every id after it — so anything pinned to a number would
select the neighbouring language, silently.

Substitution is **length-preserving**, which is what lets a batch mixing tasks
prefill as one rectangular tensor; `AedDecodeStrategy._prefill` keeps the single
`expand` when no request overrode anything.

`DecodingOptions` is mirrored by `oasr_wire::DecodingParams` on the Rust side, and
the two key sets are asserted equal at engine startup
(`DecodingOptions.assert_matches_wire_keys`) — adding one on a single side used
to give an option that was accepted and silently ignored.

## Detokenization

`decode/detokenize.py::Detokenizer` is a thin adapter over the
[tokenizer axis](tokenizers.md). It also exposes `detokenize_incremental(new_ids,
state)` for the append-only families: `symbol_table` renders piece-locally, and
`huggingface` uses an anchored window (byte-BPE fragments cannot be decoded
alone), so a partial no longer re-decodes its whole prefix.

## Adding a family

1. Implement the model-side decoder surface in `oasr/models/decoders/` — see
   [models.md](models.md#autoregressive-decoder-contracts).
2. Declare the required model surface in `oasr/models/interfaces.py::CAPABILITIES`.
3. `@register_decode_strategy("foo")` on a `DecodeStrategy`. Frame-synchronous:
   implement `decode_offline`. Label-synchronous AR: set `incremental = True` and
   implement `begin_offline` / `advance` / `has_pending`.
4. Declare the family's knobs in an options dataclass and point `options_cls` at
   it.

Working references: `transducer.py` (frame-sync greedy, offline + streaming
sessions), `rescoring.py` (`consumes="both"`), `aed.py` / `llm.py` (incremental).
