# Voice Activity Detection

Speech activity is the eighth extension axis. It answers one question — *where is
the speech* — and three components divide the work, of which only the first
varies per model:

```
SpeechDetector  ->  SpeechSegmenter  ->  Endpointer
(registry axis)     (shared policy)      (shared policy, streaming only)
p(speech) per       hysteresis over      Kaldi rule disjunction over
frame, on a         p[t] -> segments     trailing silence + turn state
declared grid       and events
```

Writing the last two once is what makes a neural VAD and a CTC blank posterior
produce identical segment semantics, identical knobs and identical events. It is
the same split [`features.md`](features.md) already uses, where `ExtractorSpec`
declares the grid and `StreamingFraming` owns the arithmetic for every frontend.

Adding a detector is a subclass plus one `register_vad` call. There is no engine
edit and no `EngineConfig` field; out-of-tree detectors arrive through the
`oasr.vad` entry-point group.

## Quick start

```bash
# Streaming: end a turn on silence, and emit the events.
oasr-server --ckpt-dir $CKPT_DIR --service-mode streaming --vad-mode endpoint

# Offline: cut long audio at speech boundaries, with a separate detector,
# so the silence between segments never reaches the encoder.
oasr-server --ckpt-dir $CKPT_DIR --service-mode offline \
    --vad-mode segment --vad-backend energy

# Tune it.  Named flags select; `--vad-option k=v` tunes, so a new knob never
# adds a flag — the same split as --decode-method / --decode-option.
oasr-server ... --vad-mode endpoint --vad-option min_silence_ms=1500
```

```python
from oasr.engine import ASREngine, EngineConfig, DecodingOptions

engine = ASREngine(EngineConfig(ckpt_dir=..., vad={"mode": "observe"}))
rid = engine.add_request(wav, streaming=False,
                         decoding=DecodingOptions(vad_events=True))
out = engine.run()[0]
for s in out.segments or ():
    print(f"{s.start:.2f} -> {s.end:.2f}  p={s.speech_prob:.2f}")
```

## Two detector families

| Family | Kinds | Consumes | Can pre-segment? |
|---|---|---|---|
| **Standalone** | `energy` | the waveform | yes |
| **ASR-derived** | `ctc_blank`, `transducer_blank`, `cif_alpha`, `aed_no_speech` | the ASR's own per-frame output | **no** |

`vad.backend = None` (the default) means *auto*: the engine resolves it to the
detector the running decode family declares, so **"no separate VAD model
configured" is a first-class configuration rather than a degraded one**. On this
codebase it is also nearly free — `ctc_blank` reads a column of the log-probs the
fused head already produced.

It is not the consolation prize, either. Deepgram documents the failure of its
own acoustic endpointer plainly: background noise keeps the VAD hot, so silence
never registers and the endpoint never fires; their fix was to add a
*decoder-derived* signal that "works effectively despite background noise". A
blank posterior is that signal — the acoustic model has already decided the fan
in the room is not a token.

What an ASR-derived detector cannot do is run *before* the encoder, so it cannot
drive offline segmentation. `register_vad` refuses a spec that claims otherwise,
at registration; asking for `vad.mode="segment"` with one raises at engine
construction naming the gap. Degrading to one whole-file segment would be
indistinguishable, to a client, from audio that really was one long utterance.

### The peakiness floor

CTC and transducer outputs are **peaky**: a head emits a non-blank at a handful
of frames per second and blank everywhere else. Measured on read speech at a
40 ms frame rate, only ~15 % of frames clear `p = 0.5`, and runs below threshold
reach **840 ms** with no pause in the audio. Handing that trace a 100 ms
minimum-silence would shred one utterance into dozens of segments.

Two mechanisms handle it, and they are deliberately different sizes:

* a short **dilation** inside the detector (±100 ms for `ctc_blank`, ±200 ms for
  `transducer_blank`) closes the one- and two-frame gaps *between the tokens of a
  word*, and makes the endpointer's windowed activity test see a run rather than
  a spike train;
* `VadSpec.min_silence_floor_ms` declares the shortest silence the signal can
  distinguish from its own sparsity, and the engine raises the preset to meet it
  — with a log line saying so. `ctc_blank` and `transducer_blank` declare 1 s,
  which is also the trailing-blank rule WeNet ships.

Bridging 840 ms by dilation alone would smear every boundary by 420 ms, which is
why the floor exists instead.

## Modes

| `vad.mode` | Offline | Streaming | Encoder sees silence? |
|---|---|---|---|
| `off` (default) | — | — | n/a; nothing runs |
| `observe` | labels the audio | events + segments | yes |
| `endpoint` | **refused** | ends the turn on silence | yes |
| `segment` | cuts at speech boundaries | **not yet** | offline: no |

`endpoint` is refused offline because an offline request already has exactly one
utterance boundary: its end.

`segment` is refused in streaming, and the reason is worth stating: both
streaming backends assume every chunk is contiguous in encoder-frame time — the
paged backend advances `req.offset` only on a forward, and `cache_t1` derives
from it. Skipping a chunk tells the encoder "this immediately follows the last
one", splicing non-adjacent audio at contiguous relative positions, with the
convolution cache carrying left context across a gap that no longer exists. The
reset primitive that would make it sound is a separate change.

## Configuration

One field on `EngineConfig`, one sub-config — the way `feature_config` is, not a
dozen flat knobs.

```python
EngineConfig(vad={"mode": "endpoint", "min_silence_ms": 1500})
```

### Two presets, not one default

The same detector is configured an order of magnitude differently depending on
what it feeds. Silero ships `min_silence 100 ms / pad 30 ms` for turn-taking;
faster-whisper re-tunes *the same model* to `2000 / 400` for long-form
pre-segmentation, because aggressive segmentation clips word onsets and costs
WER. A single default set would be wrong for one of the two, so the unset knobs
come from a preset chosen by service mode.

| Preset | when | `threshold` | `min_speech_ms` | `min_silence_ms` | `speech_pad_ms` | `max_speech_s` |
|---|---|---|---|---|---|---|
| `turn` | streaming | 0.5 | 250 | 100 | 30 | 20 |
| `segment` | offline | 0.5 | 0 | 2000 | 400 | 30 |

`speech_pad_ms` only matters where audio is actually dropped: in `observe` and
`endpoint` the encoder still sees every sample, so padding only shapes the
*reported* boundaries.

### Vocabulary

The knobs use the Silero/OpenAI names, because OASR's OpenAI-compatible surface
already speaks them. pyannote, NeMo and Riva spell the same six concepts
differently:

| Concept | pyannote / NeMo / Riva | here |
|---|---|---|
| enter-speech threshold | `onset` | `threshold` |
| exit threshold | `offset` | `neg_threshold` |
| drop short speech | `min_duration_on` | `min_speech_ms` |
| fill short gaps | `min_duration_off` | `min_silence_ms` |
| pad the segment | `pad_onset` / `pad_offset` | `speech_pad_ms` |

### Endpoint rules

Kaldi's `OnlineEndpointConfig` shape — an OR of clauses, each an AND of
conditions — collapsed to three the way WeNet's `CtcEndpointConfig` does:

| rule | `must_contain_nonsilence` | `min_trailing_silence_s` | `min_utterance_length_s` | catches |
|---|---|---|---|---|
| 1 | false | 5.0 | 0 | nothing was ever decoded |
| 2 | true | 1.0 | 0 | the ordinary end of a turn |
| 3 | false | 0 | 20.0 | hard cap |

`max_relative_cost` is **absent by design**, not approximated. It is the gap
between the best cost over all active tokens and the best cost over tokens that
can reach a WFST final state, and no end-to-end decoder has one; every downstream
port (WeNet, sherpa, sherpa-onnx, Vosk) drops it for the same reason. OASR has an
in-tree GPU WFST decoder and could one day restore it.

**Why 1.0 s and not the ~500 ms hosted APIs converge on.** The providers that get
away with 500 ms pair the timer with a *turn-confidence* signal (Deepgram Flux
`eot_threshold`, AssemblyAI `end_of_turn_confidence_threshold`, OpenAI
`semantic_vad`); their silence threshold is a floor on a confident decision
rather than the decision itself. A pure-silence rule has to be longer or it cuts
people off mid-sentence, and the open-source consensus for one is 1.0–2.4 s.

### Trailing silence is counted robustly

A plain run-length counter — what WeNet and sherpa-onnx use — is reset to zero by
a single spurious non-blank frame in a pause, so one bad frame costs a whole
endpoint. Riva instead asks what fraction of a trailing window is active. The
endpointer here uses Riva's test to *qualify* Kaldi's counter: a frame resets it
only if the window ending at it is genuinely active. A lone frame in silence
cannot qualify; a real last-word frame always does. Kaldi's semantics, Riva's
robustness, rather than a trade between them.

## What clients see

| Surface | Field / event |
|---|---|
| gRPC `StreamingRecognitionConfig` | `single_utterance` (now honoured), `enable_voice_activity_events`, `voice_activity_timeout` |
| gRPC `SpeechEventType` | `END_OF_SINGLE_UTTERANCE` (now emitted), `SPEECH_ACTIVITY_BEGIN`, `SPEECH_ACTIVITY_END`, `SPEECH_ACTIVITY_TIMEOUT`, with `speech_event_offset` |
| `/v1/realtime` | `turn_detection: {"type": "server_vad", "silence_duration_ms": ...}` in `session.update`; `input_audio_buffer.speech_started` / `.speech_stopped` / `.committed` |
| `/v1/audio/transcriptions` | `chunking_strategy`; real per-segment rows and `no_speech_prob` in `verbose_json` |
| `RequestOutput` | `segments`, `speech_events`, `no_speech_prob`, `endpoint_reason` |

Event offsets are **audio** milliseconds, never wall clock — the same reason
Google computes its own from bytes received: a jittery uplink must not move a
reported boundary.

Per-request options are *selective*: they change what is decoded or name a
response field, so a family or an engine that cannot honour one **rejects the
request** rather than returning a response with the array quietly missing.

| Option | Meaning |
|---|---|
| `single_utterance` | stop at the first turn boundary (streaming only) |
| `vad_events` | emit transitions and fill `segments` |
| `endpoint_silence_ms` | override the trailing silence that ends a turn |

## Metrics

`oasr_engine_endpoints_total{reason}` counts turns the endpointer closed, by the
rule that fired. A turn that ended because the audio ran out is deliberately not
counted there — the two are different events.
`oasr_engine_vad_segments_total` and `oasr_engine_audio_seconds_skipped_total`
are the pair that says what segmentation bought: the second is encoder work that
did not happen. The per-tick cost shows up as
`oasr_engine_stage_host_seconds{stage="streaming.vad"}`.

## Adding a detector

1. Subclass `SpeechDetector` and implement whichever entry point your `consumes`
   names — `detect` for a waveform, `detect_from_asr` for an ASR tensor. Build
   the model from `oasr.layers` (rule 2), never bare `nn` modules.
2. `register_vad(VadSpec(...))`. Set `consumes`, `modes` and — if the signal is
   sparse — `min_silence_floor_ms`, deliberately: the registry enforces the
   first two against each other, and the third is what stops a preset being
   applied to a trace that cannot support it.
3. A detector with weights declares its own `framing`; one whose grid is the
   encoder's leaves it `None` and is told `seconds_per_frame` by the engine.

Nothing else changes: no `EngineConfig` field, no engine edit, no serving change.
