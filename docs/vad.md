# Voice Activity Detection

Speech activity is the eighth extension axis. It answers one question — *where
is the speech* — and three components divide the work, of which only the first
varies per model:

```
SpeechDetector  ->  SpeechSegmenter  ->  Endpointer
(registry axis)     (shared policy)      (shared policy, streaming only)
p(speech) per       hysteresis over      rule disjunction over
frame               p[t] -> segments     trailing silence + turn state
                    and events
```

Because the last two are written once, a neural VAD and a CTC blank posterior
produce identical segment semantics, identical knobs and identical events.
Adding a detector is a subclass plus one `register_vad` call — no engine edit
and no `EngineConfig` field; out-of-tree detectors arrive through the
`oasr.vad` entry-point group.

## Quick start

```bash
# Streaming: end a turn on silence, and emit the events.
oasr-server --ckpt-dir $CKPT_DIR --service-mode streaming --vad-mode endpoint

# Offline: cut long audio at speech boundaries with a separate detector, so the
# silence between segments never reaches the encoder.
oasr-server --ckpt-dir $CKPT_DIR --service-mode offline \
    --vad-mode segment --vad-backend energy

# The same, chunk by chunk: a confirmed silence closes the turn and resets the
# encoder.
oasr-server --ckpt-dir $CKPT_DIR --service-mode streaming \
    --vad-mode segment --vad-backend energy

# With the neural detector, which is what to use on anything but clean audio.
# `energy` needs no download; `silero` needs its 2.2 MB of weights.
oasr-server --ckpt-dir $CKPT_DIR --service-mode offline \
    --vad-mode segment --vad-backend silero --vad-model-dir $SILERO_VAD_DIR

# Named flags select; `--vad-option k=v` tunes, so a new knob never adds a
# flag — the same split as --decode-method / --decode-option.
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

## Modes

| `vad.mode` | Offline | Streaming | Encoder sees silence? |
|---|---|---|---|
| `off` (default) | — | — | n/a; nothing runs |
| `observe` | labels the audio | events + segments | yes |
| `endpoint` | **refused** | ends the **request** on silence | yes |
| `segment` | cuts at speech boundaries | closes the **turn** on silence | no |

`off` leaves every transcript byte-identical. `endpoint` is refused offline
because an offline request already has exactly one utterance boundary: its end.

`endpoint` and `segment` both act on silence but are not variants of each
other. `endpoint` stops recognising and hands the client its result — Google's
`single_utterance` — so the stream ends. `segment` closes the turn, resets the
encoder, and keeps going on the same connection; the client sees one transcript
that keeps growing, with the turn structure reported as speech events and
segments.

### What `segment` does to a stream

* A confirmed silence **closes the turn**: the transcript so far is finalized
  and folded into the stream's running transcript, and the encoder and decoder
  contexts are reset, so the KV cache stops growing and the next turn starts on
  a clean context.
* Silence is **skipped** — never encoded at all — only when the detector has
  judged past the far edge of a window and nothing inside it, widened by
  `speech_pad_ms` on both sides, is speech. That needs the detector to be ahead
  of the encoder, which is the case while a stream is backlogged — exactly when
  saving encoder work is worth something. A stream arriving at real time still
  closes turns, it just encodes the silence first. Anything undecided is
  encoded: encoding silence costs time, dropping speech costs words.
* Reported times **keep accumulating across turns**. Word timings, token
  timestamps, segment boundaries and events are always relative to the start of
  the stream, never to the start of the current turn.
* A boundary with **no gap around it is not a boundary**. `max_speech_s` closes a
  run that never pauses, and that cut lands wherever the audio happened to be —
  so segments that touch are merged back offline, and in streaming such a cut
  does not close the turn. Cutting there would drop no audio and would split a
  word across two contexts.

**Requires a waveform detector.** Only a detector that runs before the encoder
can decide what the encoder sees, so `segment` needs a backend declaring
`presegment` — and in streaming `stream` as well, because there the verdict has
to be reached incrementally. `silero` and `energy` declare both; an ASR-derived
backend is refused at engine construction naming the gap, as is a backend with
weights when `--vad-model-dir` is unset.

Two combinations are refused rather than degraded in this mode: `n_best > 1`
(alternatives of separate turns do not compose into alternatives of the stream)
and `overlap_partial_readback`.

## Detectors

| Family | Kinds | Consumes | Can pre-segment? | Weights? |
|---|---|---|---|---|
| **Standalone** | `silero`, `energy` | the waveform | yes | `silero` only |
| **ASR-derived** | `ctc_blank`, `transducer_blank`, `cif_alpha`, `aed_no_speech` | the ASR's own per-frame output | **no** | no |

`vad.backend = None` (the default) means *auto*: the engine resolves it to the
detector the running decode family declares, so **"no separate VAD model
configured" is a first-class configuration rather than a degraded one** — and
nearly free, since `ctc_blank` reads a column of the log-probs the fused head
already produced. A decoder-derived signal is also the one that survives
background noise, where an acoustic detector stays hot and silence never
registers.

The two families are complementary rather than alternative, which is why both
exist: a cheap pre-ASR detector decides what to encode (`vad.mode="segment"`),
and an ASR-derived one decides when a turn ended (`vad.mode="endpoint"`).

| Kind | Consumes | Roles | Notes |
|---|---|---|---|
| `silero` | waveform | presegment, stream, posthoc | Silero VAD v5 (MIT, 309 633 params); needs `--vad-model-dir` |
| `energy` | waveform | presegment, stream, posthoc | peak-relative log energy; dependency-free baseline |
| `ctc_blank` | CTC log-probs | stream, posthoc | 1 − P(blank) per encoder frame |
| `transducer_blank` | greedy emission frames | stream, posthoc | greedy decoding only |
| `cif_alpha` | Paraformer CIF α | posthoc | token rate, boxcar-smoothed |
| `aed_no_speech` | Whisper prefill logits | posthoc | `<\|nospeech\|>`; one frame per decoding window |

The roles say where a detector may be used: `presegment` runs before the
encoder (required by `vad.mode="segment"`), `stream` reaches a verdict
incrementally chunk by chunk (required by `segment` and `endpoint` in
streaming), and `posthoc` only labels audio the ASR has already processed.

### `energy` and `silero`

`energy` needs no weights and no download. In streaming it carries a **running
peak** across chunks rather than re-normalising each one against its own
loudest frame, and two consequences are visible in the output: a stream that
opens on room tone is judged against a reference not yet established, so it
reads as speech and is encoded (the safe direction); and speech more than
~35 dB below an earlier loud burst reads as silence. Noise inside that dynamic
range makes a whole recording read as one continuous utterance and nothing is
cut — the documented failure of every energy VAD, and what `silero` fixes. Use
`energy` on clean audio, `silero` on anything else.

`silero` is **Silero VAD v5** (MIT, 309 633 parameters), rebuilt on
`oasr.layers` and loaded from the upstream TorchScript archive:

```bash
mkdir -p silero-vad && curl -L -o silero-vad/silero_vad.jit \
  https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit
oasr-server ... --vad-mode segment --vad-backend silero --vad-model-dir silero-vad
```

There is no conversion step: the archive is 2.2 MB, `torch.jit.load` is already
in the dependency set, and the weights are remapped at construction. A plain
`torch.save` of either the upstream tensors or the converted ones works too.

**Placement.** Waveform detectors run on the **host** by default — the audio
arrives there, and moving them to the GPU adds a device synchronisation inside
the step loop. ASR-derived detectors run beside the tensors they read, on the
engine's device. `vad.device` overrides both; CUDA is worth measuring for a
large streaming pool on a server whose GPU has headroom.

### ASR-derived detectors and the silence floor

CTC and transducer outputs are **peaky**: the head emits a non-blank at a
handful of frames per second and blank everywhere else, so runs below threshold
reach several hundred milliseconds with no pause in the audio. A 100 ms
minimum-silence applied to that trace would shred one utterance into dozens of
segments.

`VadSpec.min_silence_floor_ms` declares the shortest silence a detector's
signal can distinguish from its own sparsity, and the engine raises the preset
to meet it, with a log line saying so: 1 s for `ctc_blank` and
`transducer_blank`, 500 ms for `cif_alpha`. A `min_silence_ms` below the floor
is raised the same way.

## Configuration

One field on `EngineConfig`, one sub-config — the way `feature_config` is, not
a dozen flat knobs:

```python
EngineConfig(vad={"mode": "endpoint", "min_silence_ms": 1500})
```

Every segmentation knob defaults to `None` meaning "take the preset", so a
config that names only `mode` is fully specified.

### Two presets, not one default

The same detector is configured an order of magnitude differently depending on
what it feeds. Turn-taking wants short silences and little padding; long-form
pre-segmentation wants long silences and generous padding, because aggressive
segmentation clips word onsets and costs WER. The preset is chosen from the
service mode unless `preset` names one — and `mode="segment"` takes the
segmentation preset in *either* service mode, because it is the mode that drops
audio.

| Preset | when | `threshold` | `min_speech_ms` | `min_silence_ms` | `speech_pad_ms` | `max_speech_s` |
|---|---|---|---|---|---|---|
| `turn` | streaming `observe` / `endpoint` | 0.5 | 250 | 100 | 30 | 20 |
| `segment` | offline, or any `mode="segment"` | 0.5 | 0 | 2000 | 400 | 30 |

### Knobs

**Selection**

| Field | Default | Meaning |
|---|---|---|
| `mode` | `"off"` | `off` / `observe` / `endpoint` / `segment` |
| `backend` | `None` | registry kind; `None` = the decode family's own detector |
| `model_dir` | `None` | checkpoint directory for a detector with weights |
| `device` | `None` | host for waveform kinds, the engine's device for ASR-derived |
| `sample_rate` | the model's | rate the waveform detectors see; the engine does not resample |
| `frame_ms` / `hop_ms` | 25 / 10 | analysis geometry for detectors whose framing is configurable (`energy`); one with a trained window declares its own and ignores these |

**Segmentation** — unset values come from the preset.

| Field | Meaning |
|---|---|
| `preset` | `turn` or `segment`; `None` picks from the service mode |
| `threshold` | probability at or above which a frame enters speech |
| `neg_threshold` | probability below which a frame leaves speech; `None` derives `threshold - 0.15`, and the gap is what stops a trace hovering at the threshold from chattering |
| `min_speech_ms` | drop speech runs shorter than this; in streaming it also delays the `speech_started` event by that much, though the timestamp still marks the true onset |
| `min_silence_ms` | silence shorter than this does not end a run |
| `speech_pad_ms` | padding on each side of an emitted segment; inert in `observe` and `endpoint`, where the encoder sees every sample anyway, and load-bearing in `segment`, where audio is dropped |
| `max_speech_s` | cut a run this long even without a silence. The cut lands at the current frame, not at the quietest one, so it is a bound on segment length rather than a place to split a transcript — see `segment` above |

**Endpointing** (streaming only)

| Field | Default | Meaning |
|---|---|---|
| `endpoint_rules` | the three rules below | the rule disjunction |
| `activity_window_ms` | 300 | width of the trailing window the activity test looks at |
| `activity_threshold` | 0.2 | fraction of that window that must be non-silent to reset the trailing-silence counter |
| `speech_start_timeout_s` | `None` | close the stream if speech never begins; cancelled for the rest of the stream once the first speech-start fires. 0.5–60 s |
| `speech_end_timeout_s` | `None` | close the stream this long after speech ends; reset by a new speech-start. 0.5–60 s |

### Vocabulary

The knobs use the Silero/OpenAI names, because OASR's OpenAI-compatible surface
already speaks them. pyannote, NeMo and Riva spell the same concepts
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
conditions — collapsed to three. A rule fires when all of its conditions hold;
the endpointer fires when any rule does.

| rule | `must_contain_nonsilence` | `min_trailing_silence_s` | `min_utterance_length_s` | catches |
|---|---|---|---|---|
| 1 | false | 5.0 | 0 | nothing was ever decoded |
| 2 | true | 1.0 | 0 | the ordinary end of a turn |
| 3 | false | 0 | 20.0 | hard cap |

Replace the set with `vad.endpoint_rules`, or override just the ordinary case
per request with `endpoint_silence_ms`.

Rule 2's 1.0 s is deliberately longer than the ~500 ms hosted APIs converge on:
those pair the timer with a *turn-confidence* signal, so their silence
threshold is a floor on a confident decision rather than the decision itself. A
pure-silence rule has to be longer or it cuts people off mid-sentence, and the
open-source consensus for one is 1.0–2.4 s.

Trailing silence is counted robustly: a frame resets the counter only if the
window ending at it is genuinely active (`activity_window_ms`,
`activity_threshold`), so one spurious non-blank frame in a pause does not cost
a whole endpoint.

## What clients see

| Surface | Field / event |
|---|---|
| gRPC `StreamingRecognitionConfig` | `single_utterance`, `enable_voice_activity_events` |
| gRPC `SpeechEventType` | `END_OF_SINGLE_UTTERANCE`, `SPEECH_ACTIVITY_BEGIN`, `SPEECH_ACTIVITY_END`, `SPEECH_ACTIVITY_TIMEOUT`, with `speech_event_offset` |
| `/v1/realtime` | `turn_detection: {"type": "server_vad", "silence_duration_ms": ...}` in `session.update`; `input_audio_buffer.speech_started` / `.speech_stopped` / `.committed` |
| `/v1/audio/transcriptions` | `chunking_strategy`; per-segment rows and `no_speech_prob` in `verbose_json` |
| `RequestOutput` | `segments`, `speech_events`, `no_speech_prob`, `endpoint_reason` |

Event offsets are **audio** milliseconds, never wall clock, so a jittery uplink
cannot move a reported boundary.

The segmenter's own knobs are engine-level, so the per-request fields that name
them — gRPC `voice_activity_timeout`, `turn_detection.threshold` and
`prefix_padding_ms` on `/v1/realtime`, `chunking_strategy`'s `threshold` /
`prefix_padding_ms` / `silence_duration_ms` — are **rejected** rather than
ignored, pointing at the `--vad-option` that sets them.

Per-request options are *selective*: they change what is decoded or name a
response field, so a family or an engine that cannot honour one **rejects the
request** rather than returning a response with the field quietly missing.

| Option | Meaning |
|---|---|
| `single_utterance` | stop at the first turn boundary (streaming only) |
| `vad_events` | emit transitions and fill `segments` |
| `endpoint_silence_ms` | override the trailing silence that ends a turn |

## Metrics

| Metric | Says |
|---|---|
| `oasr_engine_endpoints_total{reason}` | turns the endpointer closed, by the rule that fired — a turn that ended because the audio ran out is deliberately not counted |
| `oasr_engine_vad_segments_total` | segments produced |
| `oasr_engine_audio_seconds_skipped_total` | encoder work segmentation avoided |
| `oasr_engine_stage_host_seconds{stage="streaming.vad"}` | per-tick cost |

## Adding a detector

1. Subclass `SpeechDetector` and implement whichever entry point your
   `consumes` names — `detect` / `detect_streaming` for a waveform,
   `detect_from_asr` for an ASR tensor. Build the model from `oasr.layers`
   (rule 2), never bare `nn` modules.
2. `register_vad(VadSpec(...))`. Set `consumes`, `modes` and — if the signal is
   sparse — `min_silence_floor_ms` deliberately: the registry enforces the
   first two against each other, and the third is what stops a preset being
   applied to a trace that cannot support it.
3. A detector with weights declares its own `framing` and sets `needs_weights`;
   one whose grid is the encoder's leaves `framing` as `None` and is told
   `seconds_per_frame` by the engine.

Nothing else changes: no `EngineConfig` field, no engine edit, no serving
change.
