# OASR Serving — Rust frontend

The `rust/` Cargo workspace builds the OASR serving core, which exposes the
engine over both HTTP and gRPC.  The API surface is shaped after **Google Cloud
Speech-to-Text v1** so existing tooling (REST conventions, `grpcurl`,
OpenAPI-style clients) feels familiar.

`pip install` compiles the core into the `oasr._core` PyO3 extension module (via
setuptools-rust) and installs the `oasr-server` console script that runs it
(`oasr/_server_cli.py`, which just forwards `sys.argv` into `oasr._core.serve`),
so the front-end ships with the wheel — no separate build step.  The same code
also builds as a standalone binary (`rust/crates/oasr-server`) for `cargo`-only
workflows; both share the `oasr-serve` crate.

The `oasr` Python package is a **runtime dependency** of the front-end: it must
be importable by the active interpreter.  There is no Python serving process —
the former ZMQ worker was replaced by the in-process PyO3 engine.

The engine runs **in-process** via PyO3 — one Python `ASREngine` per
`oasr-server` process.  Multi-GPU scale is achieved by launching N
`oasr-server` processes (each with `CUDA_VISIBLE_DEVICES` set), not by
multiplexing inside one process.

## Workspace layout

| Crate | Role |
|---|---|
| `oasr-wire` | Shared event/command types (`Cmd`, `Event`, `ErrorCode`, `ModelInfo`, `DecodingParams`). Pure Rust — no codec, no IPC. |
| `oasr-engine-client` | PyO3-backed driver: the `PyEngine` wrapper, the `EngineDispatcher` thread that owns the GIL and drives `engine.step()`, and the `EngineClient` / `EnginePool` async facades. Exposes `auto-initialize` / `extension-module` features forwarding to pyo3. |
| `oasr-asr` | Audio decode (WAV via `hound`, raw PCM) to f32 mono `bytes::Bytes`, plus sample-rate conversion (`resample.rs`, windowed-sinc via `rubato`). |
| `oasr-server-http` | axum routes (Google STT v1-shaped REST). |
| `oasr-server-grpc` | tonic `oasr.speech.v1.Speech` service plus the standard `grpc.health.v1.Health` service. Proto in `rust/proto/oasr_speech_v1.proto`. |
| `oasr-serve` | Mode-agnostic serving core: `Cli` + `run(cli)` — builds the engine, the tokio runtime, and both listeners. Shared by the binary and the extension module. |
| `oasr-server` | Standalone binary: thin `main.rs` → `oasr_serve::run`; pulls `oasr-engine-client` with `auto-initialize`. |
| `oasr-core` | cdylib `oasr._core` PyO3 module: `#[pymodule]` exposing `serve(argv)` → `oasr_serve::run` under `allow_threads`; pulls `oasr-engine-client` with `extension-module`. Built by setuptools-rust. |

**The PyO3 linkage mode is the key split.** The binary embeds and links
libpython (`pyo3/auto-initialize`); the extension module is loaded by the host
interpreter (`pyo3/extension-module`). Those features are mutually exclusive and
Cargo unifies features per build, which is why the shared logic lives in
`oasr-serve` and why `oasr-core` is excluded from `default-members`. Never run
`cargo build/test --workspace`.

Three request handles cross the PyO3 boundary: `OfflineHandle` (unary — one
terminal event), `StreamingHandle` (chunked audio in), and `OfflineStreamHandle`
(audio in one shot, **every** event streamed out). All three arm
`CancelOnDrop`, so a client disconnect stops the request.

## Quick start

```bash
# Install the Python package — builds the _C decoder extension, the oasr._core
# serving extension (Rust), and the `oasr-server` console script.  Needs a Rust
# toolchain + protobuf-compiler on PATH at build time.
pip install -e .

# Launch — one engine per process
oasr-server \
    --ckpt-dir /path/to/wenet/ckpt \
    --service-mode offline \
    --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051

# Transcribe something
oasr transcribe meeting.mp3
```

`--ckpt-dir` takes the same sources as `EngineConfig.ckpt_dir`: a directory in
any supported checkpoint format, or a HuggingFace Hub repo id (downloaded on
first use — see `docs/checkpoints.md`).

`oasr-server --help` lists every flag.  `--service-mode` pins the engine to
either `offline` or `streaming` for its entire lifecycle.  A `streaming` engine
rejects the unary `Recognize` with `FAILED_PRECONDITION`; an `offline` engine
serves **both** RPCs — see [Streaming text out of an offline
engine](#streaming-text-out-of-an-offline-engine).

> Building the workspace with `cargo build --release` instead produces the same
> server as the `rust/target/release/oasr-server` binary; substitute that path
> for `oasr-server` in the commands below if you go that route.

## Clients

### The `oasr` command

```bash
oasr transcribe meeting.mp3                        # against a running server
oasr transcribe meeting.mp3 --ckpt-dir ./ckpt      # in-process, no server
oasr translate  entretien.m4a --language fr
oasr transcribe talk.wav --response-format srt -o talk.srt
oasr models
oasr serve --ckpt-dir ./ckpt                       # forwards to oasr-server
oasr convert /path/to/wenet /path/to/native        # forwards to oasr-convert
```

`--ckpt-dir` switches from "call a server" to "load the engine in this process",
which is the path `examples/recognize/local_engine.py` spells out; everything
else uses `--url` (default `http://127.0.0.1:8080`).

### `oasr.client`

```python
from oasr.client import OASRClient

client = OASRClient("http://127.0.0.1:8080")
print(client.transcribe("meeting.mp3").text)
print(client.translate("entretien.m4a", language="fr").text)
```

and, for live audio, an async iterator over `/v1/realtime`:

```python
from oasr.client import AsyncOASRClient

async with AsyncOASRClient("http://127.0.0.1:8080") as client:
    async for event in client.stream(mic_chunks(), sample_rate=16000):
        print(event.text, end="\n" if event.is_final else "\r")
```

`chunks` is any (async) iterable of raw PCM; frames go out as binary, skipping
base64. Both clients need the `serving` extra (`pip install "oasr[serving]"`).

An existing **OpenAI client** works too — see
[`POST /v1/audio/transcriptions`](#post-v1audiotranscriptions).

## API surface

| Surface | Path / Service | Notes |
|---|---|---|
| HTTP | `POST /v1/audio/transcriptions` | **OpenAI-compatible** multipart upload. Any supported container; `json` / `text` / `srt` / `vtt` / `verbose_json`; `stream=true` for SSE. |
| HTTP | `POST /v1/audio/translations` | The same, forcing `task=translate` (Whisper-family checkpoints). |
| HTTP | `GET /v1/realtime` | **WebSocket** streaming transcription. Binary PCM frames or base64; OpenAI realtime-transcription event names. |
| HTTP | `POST /v1/speech:recognize` | Google-STT-shaped unary recognition (offline mode). Raw body, config in the query string — the lowest-overhead HTTP path. |
| HTTP | `GET /v1/models` | Loaded model metadata, in OpenAI's list shape. |
| HTTP | `GET /healthz` | Process liveness. |
| HTTP | `GET /readyz` | 200 once the engine dispatcher has produced its first Pong. |
| HTTP | `GET /metrics` | Prometheus exposition. |
| gRPC | `oasr.speech.v1.Speech/Recognize` | Synchronous unary (offline mode). |
| gRPC | `oasr.speech.v1.Speech/StreamingRecognize` | Bidi streaming. In `streaming` mode audio is fed to the engine chunk by chunk; in `offline` mode audio is buffered until half-close and the **text** streams back (token streaming for AR families). |
| gRPC | `grpc.health.v1.Health/Check` and `Watch` | Standard gRPC health checking. |

**Two request shapes, one engine.** The OpenAI-shaped routes exist because
every ASR client library, LLM app framework and "swap the endpoint" migration
already speaks them — pointing existing code at OASR is a base-URL change. The
Google-shaped route is unchanged and remains the fastest HTTP path (no
multipart framing, no container parse). Both are thin adapters over the same
engine handles; neither is privileged.

## HTTP — the OpenAI-compatible surface

### `POST /v1/audio/transcriptions`

A multipart upload, exactly as OpenAI's audio API takes it:

```bash
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
     -F file=@meeting.mp3 \
     -F model=whisper-1 \
     -F response_format=json | jq
# {"text": "..."}
```

and from an existing OpenAI client, unchanged apart from the base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.audio.transcriptions.create(model="whisper-1", file=open("a.mp3", "rb")).text)
```

| Field | Meaning |
|---|---|
| `file` | **Required.** The audio, in any [supported container](#audio-formats). |
| `model` | Accepted and ignored unless `--served-model-name` is set, in which case an unknown name is `404`. One process serves one model. |
| `language` | Source language. BCP-47 (`en-US`) or ISO-639 (`en`); reduced to the primary subtag. Honored by families with a language control (Whisper); **rejected** by families without one, rather than ignored. |
| `prompt` | Prompt override for the speech-LLM family. |
| `response_format` | `json` (default) · `text` · `srt` · `vtt` · `verbose_json`. |
| `temperature` | `0` (default) is greedy; `> 0` samples (AR families). |
| `timestamp_granularities[]` | `segment` is served; `word` returns `501` — see below. |
| `stream` | `true` streams the transcript as SSE (`json` / `text` formats only). |

**Two deliberate differences from OpenAI, both loud rather than silent:**

* `timestamp_granularities[]=word` returns **`501 UNIMPLEMENTED`**. Word
  alignment is a real gap (`.artifacts/architecture_review.md` H7); answering
  without the `words` array a client asked for reads as "this audio had no
  words".
* `verbose_json` carries **one segment** spanning the utterance — no decode
  family produces segment boundaries today — and omits `avg_logprob`,
  `no_speech_prob` and `compression_ratio` rather than filling them with
  plausible numbers. `end` is the last token's alignment when the family has one
  (Paraformer CIF), the audio's duration otherwise. `request_id` and
  `finish_reason` are OASR extensions.

`srt` / `vtt` produce a single cue over that same span.

### `POST /v1/audio/translations`

Identical, minus `timestamp_granularities`, and it forces `task=translate`.
`language` is the **source** language hint.

This needs a checkpoint whose decode family has a task control — Whisper today.
Anything else answers `400` naming the limitation instead of transcribing and
calling the result a translation. Before OASR had a per-request `task`, the
transcribe-vs-translate choice was frozen in the checkpoint's
`forced_decoder_ids` at conversion time, which is why this route did not exist.

### `stream=true` (server-sent events)

```
data: {"type":"transcript.text.delta","delta":"hello"}
data: {"type":"transcript.text.delta","delta":" world"}
data: {"type":"transcript.text.done","text":"hello world"}
data: [DONE]
```

The engine's partials are cumulative and the protocol's `delta` is an
increment, so each event carries what the partial *added*. A frame-synchronous
family can revise a partial rather than extend it; such an update yields no
delta, and `transcript.text.done` — always the complete text — settles it.

Only `json` and `text` may stream: `srt`, `vtt` and `verbose_json` need the
utterance's total duration, which does not exist until the last token.

### `GET /v1/realtime` (WebSocket)

Streaming transcription for clients that cannot speak gRPC — every browser, and
most scripting clients. Event names follow OpenAI's realtime *transcription*
session.

Client → server:

| Message | Meaning |
|---|---|
| `{"type":"session.update","session":{…}}` | Configure; honoured before the first audio. Keys: `sample_rate`, `encoding`, `language`, `task`, `prompt`, `interim_results`, `model`. |
| `{"type":"input_audio_buffer.append","audio":"<base64>"}` | One audio chunk. |
| *a binary frame* | The same chunk without base64 — a third fewer bytes and no JSON parse. |
| `{"type":"input_audio_buffer.commit"}` | End of utterance. |

Server → client:

| Message | Meaning |
|---|---|
| `{"type":"transcription_session.created","session":{…}}` | Handshake, echoing the resolved configuration. |
| `{"type":"conversation.item.input_audio_transcription.delta","delta":"…","text":"…"}` | Interim. `delta` is the increment; `text` (an OASR extension) is the transcript so far. |
| `{"type":"conversation.item.input_audio_transcription.completed","transcript":"…"}` | Final. |
| `{"type":"error","error":{…}}` | Terminal failure. |

Configuration is optional — a client that opens the socket and starts sending
16 kHz `LINEAR16` gets the defaults. The session takes **headerless PCM only**
(`LINEAR16`, `LINEAR32F`, `MULAW`, `ALAW`): a container header arrives once, at
the front of a stream whose chunks are decoded independently. Post containers to
`/v1/audio/transcriptions`.

Works in both service modes. A `streaming` engine consumes chunks as they
arrive; an `offline` one buffers them and streams the *text* back, exactly as
the gRPC surface does and for the same reason.

**Browsers need CORS**, which is off by default:

```bash
oasr-server --ckpt-dir ... --cors-allow-origin 'https://app.example.com'
oasr-server --ckpt-dir ... --cors-allow-origin '*'          # local demo
```

Whether an inference endpoint should be callable from any page is an operator's
decision, so it is never a default. `examples/web` is a working browser client
against these two endpoints; it used to ship a 731-line FastAPI relay because
neither existed.

## Audio formats

The front-end decodes, downmixes to mono and resamples to the model's rate
before anything crosses PyO3.

| Family | Formats |
|---|---|
| Containers | WAV · FLAC · MP3 · AAC / M4A (ISO-MP4) · ALAC · OGG (Vorbis) · AIFF · CAF · MKV/WebM |
| Headerless PCM | `LINEAR16` (i16-LE) · `LINEAR32F` (f32-LE) · `MULAW` · `ALAW` |
| **Not supported** | **Opus** (in any container), AMR, AMR-WB, Speex |

Opus is a **declared** gap, not an oversight: there is no pure-Rust decoder, and
linking libopus would put a C dependency in every build. A WebM/Ogg body
carrying an Opus track demuxes and then fails with a message naming the codec,
rather than producing something wrong. Transcode first (`ffmpeg -i in.webm
out.wav`) or send PCM.

Decoding is `symphonia`, behind the `codecs` feature of `oasr-asr` — on by
default. Building with `--no-default-features` leaves WAV, raw PCM and G.711
working and turns every other container into a clear rejection.

**Two caps, and they measure different things.** `--max-audio-mib` (default 256)
bounds the *encoded* request body on both surfaces. `--max-audio-seconds`
(default 4 h) bounds the *decoded* waveform — once compressed containers are
accepted the two stop being related, since a few MiB of MP3 is hours of audio
and the allocation happens before anything could notice. Exceeding it is `413`,
never a truncated transcript.

## HTTP — the Google-shaped surface

### `POST /v1/speech:recognize`

The request **body is the raw audio payload** (no base64, no JSON envelope) and
the recognition config travels in the **query string**.  Dropping the base64 +
JSON request framing avoids the ~33% wire inflation and a multi-MB JSON parse —
measured at **~2× the throughput** of a base64-JSON body at moderate
concurrency (188 → 377 req/s at c=128 on the reference box).  The response is a
small JSON `RecognizeResponse` (transcript + tokens — no base64).

```bash
# POST a WAV container (server reads the embedded sample rate)
curl -sS -X POST 'http://127.0.0.1:8080/v1/speech:recognize?encoding=WAV' \
     -H 'Content-Type: application/octet-stream' \
     --data-binary @audio.wav | jq

# POST headerless raw PCM (little-endian f32 mono)
curl -sS -X POST \
  'http://127.0.0.1:8080/v1/speech:recognize?encoding=LINEAR32F&sample_rate=16000' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @audio.f32 | jq
```

Query parameters: `encoding` (required), `sample_rate` (defaults to the model's
own rate; ignored when the body carries a container header), `priority`,
`max_alternatives`, plus the per-request decoding options (autoregressive decode
families only — AED / speech-LLM; CTC ignores them): `max_new_tokens`,
`temperature` (0 = greedy), `top_k`, `top_p`, `prompt` (speech-LLM user-prompt
override), `task` and `language`.  `max_alternatives > 1` makes the engine
detokenize that many n-best hypotheses (beam decode families), so every returned
alternative carries a real transcript.

`encoding` accepted values (the same names the proto enum uses, so the REST and
gRPC surfaces cannot disagree — they share one parser):

| Value | Meaning |
|---|---|
| `AUTO` | Sniff the container from the body's own header *(OASR extension — the right choice when relaying files you did not create)*. |
| `LINEAR16` | Little-endian 16-bit signed PCM, mono. |
| `LINEAR32F` | Little-endian 32-bit float PCM, mono *(OASR extension)*. |
| `MULAW` / `ALAW` | ITU-T G.711, one byte per sample — what telephony sends. |
| `WAV`, `FLAC`, `MP3`, `M4A`, `AIFF`, `CAF`, `OGG` | The container in the body; its embedded rate wins over `sample_rate`. |
| `OGG_OPUS`, `WEBM_OPUS`, `AMR`, `AMR_WB`, `SPEEX_WITH_HEADER_BYTE` | `UNIMPLEMENTED` — see [Audio formats](#audio-formats). |

A body whose magic bytes name a container is decoded as one even when the caller
declared PCM — except for MP3/AAC, whose 11-bit frame sync headerless PCM hits by
chance. Sniffing never overrides a declared `LINEAR16` on that basis.

> For the lowest-overhead binary transport, gRPC `Recognize` is still preferred.

### Sample rates

The **engine accepts exactly one rate** — the model's, from the checkpoint's
`FeatureSpec` (16 kHz for every checkpoint in tree, reported as `sample_rate` in
`GET /v1/models`).  It does not resample: every frame count comes from
`FeatureConfig.sample_rate` and the mel filterbank is built for it, so audio at
another rate would be transcribed at the wrong speed — confidently, and with no
error anywhere.  `ASREngine` therefore *rejects* a mismatched
`Request.sample_rate` outright.

The front-end converts instead, so clients need not care.  Any rate in
`[4000, 384000]` Hz is accepted on both surfaces and resampled to the model's
rate (windowed-sinc via `rubato`, in `oasr-asr::resample`) before the waveform
crosses PyO3; anything outside that range is `INVALID_ARGUMENT`.  The streaming
RPC keeps one resampler per stream, so the filter state carries across chunks
and the tail is flushed on half-close.  Cost is a few hundred µs per
audio-second on one core — negligible next to the GPU step, and skipped entirely
when the rates already match (the common 16 kHz case builds no filter).

Measured on 12 LJSpeech utterances (201 words) against the u2pp-conformer
checkpoint, taking each file's **native 16 kHz transcript as the reference** and
feeding the same audio resampled to another rate:

| Client rate | Divergence from native 16 kHz |
|---|---|
| 44100 Hz | **0.00%** — transcript-identical |
| 8000 Hz | **6.47%** — the missing 4–8 kHz band, not a resampler artifact |

Two things worth knowing:

- **8 kHz telephony is correct but degraded**, which is what that 6.47% is.
  Upsampling cannot invent the band above 4 kHz that the source never carried.
  The honest fix is an 8 kHz model variant, which the feature-frontend registry
  already anticipates.  For scale: the same 8 kHz bytes submitted *without*
  conversion — what happened before this existed — decoded to
  `LAURE HO OFRESS WILL NOT CUT TWOS FROM NORTH THE SECOND HER UNCLE` in place
  of `PRINTING IN THE ONLY SENSE WITH WHICH WE ARE AT PRESENT CONCERNED …`.
- **A WAV body's header wins.**  It is the only rate signal for `encoding=WAV`,
  so a 44.1 kHz media file is converted from 44.1 kHz even if the client also
  passes `sample_rate=16000`.

Success response (`200`):

```json
{
  "results": [
    {
      "alternatives": [
        {
          "transcript": "hello world",
          "confidence": 0.93,
          "tokens": [12, 305, 119]
        }
      ],
      "resultEndTimeS": 3.42
    }
  ],
  "requestId": "8f4c9b..."
}
```

`confidence` is the hypothesis's softmax-normalized posterior among the
returned n-best scores (`0.0` when the decode family emits a single
hypothesis — Google's "unset when unavailable" convention).
`resultEndTimeS` (end time of the last decoded token, seconds) appears only
for decode families with token alignments (Paraformer CIF).

Error responses use the canonical Google error envelope:

```json
{ "error": { "code": 400, "status": "INVALID_ARGUMENT", "message": "..." } }
```

| HTTP status | `status` field | When |
|---|---|---|
| 400 | `INVALID_ARGUMENT` | missing/empty body, missing `encoding`, undecodable audio bytes, out-of-range decoding options, audio longer than a fixed-window frontend allows |
| 400 | `FAILED_PRECONDITION` | server is in `streaming` mode |
| 404 | `NOT_FOUND` | unknown request id (internal bug) |
| 501 | `UNIMPLEMENTED` | unsupported encoding |
| 503 | `RESOURCE_EXHAUSTED` | over-capacity, retry with backoff |
| 503 | `UNAVAILABLE` | dispatcher shutting down or engine lost |
| 500 | `INTERNAL` | otherwise |

**Rejections are per-request.** Decoding options are range-checked at the mapping
layer (`oasr_wire::DecodingParams::validated`, shared by HTTP and gRPC) and the
engine's bulk admission (`ASREngine.add_requests_batch_checked`) validates and
admits per spec. Since the dispatcher coalesces up to `--admit-threshold`
envelopes into one Python call, this matters: one client sending `top_p=1.5` gets
a 400 while every request coalesced with it is served normally. Bounds:
`temperature` ∈ {0} ∪ [0.01, 100], `top_p` ∈ (0, 1], `max_alternatives` ≤ 30,
`prompt` ≤ 4096 bytes.

Two engine-side rejections also surface here as 400s rather than as wrong output:
audio exceeding a fixed-window frontend's capacity (`whisper_logmel`: the 30 s
Whisper window, shared by Qwen2-Audio — longer audio used to be silently
truncated), and a per-request `streaming` flag that disagrees with the engine's
mode.

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "/path/to/ckpt",
      "object": "model",
      "owned_by": "oasr",
      "info": {
        "ckpt_dir": "/path/to/ckpt",
        "device": "cuda",
        "dtype": "torch.float16",
        "chunk_size": 16,
        "max_batch_size": 64,
        "decoder_type": "ctc_cuda",
        "vocab_size": 5000,
        "service_mode": "offline",
        "decode_method": "ctc",
        "capabilities": ["ctc", "ctc_aed_rescoring"],
        "sample_rate": 16000
      }
    }
  ]
}
```

One entry per name this process answers to: the checkpoint, plus any
`--served-model-name` aliases. `object: "list"` and the per-row `object` /
`owned_by` are what an OpenAI client expects; `info` is the OASR extension
carrying what the engine actually loaded.

`service_mode` / `decode_method` / `capabilities` / `sample_rate` are read back
**from the
engine**, not from the CLI flags: `--engine-config` JSON wins on the Python side
and several decode families are offline-only (`aed`, `llm`, `paraformer`,
`ctc_aed_rescoring`), so the engine is the authority on what this process can
serve. The front-ends configure themselves from the engine's `service_mode` and
log a warning when `--service-mode` disagrees. Note `decoder_type` is only the
CTC *kernel* selector (`ctc_cuda` / `ctc_wfst`) — `decode_method` is the family.

## gRPC

Service: `oasr.speech.v1.Speech` in `rust/proto/oasr_speech_v1.proto`.
Messages mirror Google's v1 schema; `tokens` (CTC token IDs), `requestId`,
and the `RecognitionConfig` decoding extensions (`max_new_tokens`,
`temperature`, `top_k`, `top_p`, `prompt`, `task` — per-request
`DecodingOptions` for the AR decode families) are OASR extensions in the
reserved field-number range.  `max_alternatives` is honored (n-best transcripts
on beam decode families), `confidence` carries the softmax-normalized n-best
posterior, and `result_end_time` is set when the decode family produces
token alignments (Paraformer CIF).

`language_code` is **no longer accepted-and-ignored**: it now selects the
language token on families that have one (Whisper), reduced from BCP-47 to its
primary subtag (`en-US` → `en`). A family with no language control rejects a set
value rather than transcribing in the checkpoint's own language while the client
believes otherwise. `model`, `audio_channel_count`, `single_utterance` and
`speech_event_type` remain accepted and ignored.

`StreamingRecognize` requires a **headerless PCM** encoding, for the same reason
the realtime WebSocket does. Previously `encoding=WAV` on a stream silently
mapped to raw f32, so the client's 44-byte RIFF header was decoded as eleven
samples of noise at the front of every stream; it is now `INVALID_ARGUMENT`.

### `Recognize` (unary, offline mode)

```bash
B64=$(base64 -w0 audio.wav)
grpcurl -plaintext -import-path rust/proto -proto oasr_speech_v1.proto \
        -d "$(jq -n --arg b64 "$B64" \
              '{config:{encoding:"WAV",sampleRateHertz:16000,languageCode:"en-US"},
                audio:{content:$b64}}')" \
        127.0.0.1:50051 oasr.speech.v1.Speech/Recognize
```

### `StreamingRecognize` (bidi)

The first inbound message **must** carry `streaming_config.config`;
subsequent messages carry `audio_content` (raw PCM bytes in the declared
encoding).  Set `streaming_config.interim_results=false` to suppress
partials.  Each response message contains one `StreamingRecognitionResult`
with `is_final=true` on the terminal frame.

`sampleRateHertz` is validated at *open*, before any audio is admitted, and a
non-model rate builds one resampler for the life of the stream (see
[Sample rates](#sample-rates)).  Two consequences for a resampling stream: an
outbound chunk is not the same duration as the inbound one that produced it (the
filter holds up to ~64 ms back), and a chunk that splits a sample across the
message boundary is still `INVALID_ARGUMENT` — frame your chunks on whole
samples.

```bash
python scripts/grpc_stream.py --addr 127.0.0.1:50051 \
    --wav tests/fixtures/hello.wav --chunk-ms 640
```

#### Streaming text out of an offline engine

Four decode families are offline-only — `aed`, `llm`, `paraformer` and
`ctc_aed_rescoring` — because they cannot start before the utterance is complete
(and `whisper_logmel` normalizes over a fixed 30 s window).  That is a constraint
on **audio in**, not on **text out**: the autoregressive families emit one partial
per request per engine tick, which is the normal token-streaming UX for an LLM
ASR client.

So `StreamingRecognize` is served in `offline` mode too, with a different
mechanism: the server buffers the inbound `audio_content` frames, submits the
utterance as **one** offline request on client half-close, and then streams the
generated text back as interim results.  The client-visible shape is identical —
`is_final=false` partials followed by one `is_final=true` final.

Two consequences to plan for:

* **`--max-tick-ms` sets the inter-token cadence.**  It bounds how long the
  dispatcher holds the GIL per tick, and one partial is emitted per tick, so it is
  a *user-visible latency knob* here rather than only an internal bound.  Measured
  cadence on a real 7B: `.artifacts/serving_perf.md` §4.
* **One-shot families are unaffected.**  A CTC / Paraformer / rescoring engine
  produces a single final through the same path (`interim_results` simply yields
  nothing extra) — verified against a WeNet Conformer: exactly one response.

Half-close is the submit trigger; a client that never half-closes gets no result,
exactly like a unary client that never finishes its request body.  A client that
disconnects mid-generation cancels the request, so the AR row stops occupying a
decode slot instead of running to its `max_new_tokens` cap.

### gRPC health checking

The binary exposes the standard `grpc.health.v1.Health` service.  Both an
empty service name (overall process health) and `oasr.speech.v1.Speech`
track **engine readiness**, re-evaluated once a second against the same
dispatcher heartbeat `/readyz` reads and on the same 5 s staleness bound, so the
two probes cannot disagree about a process.  That means a wedged engine — three
consecutive `step()` failures, after which the dispatcher stops the heartbeat —
now drains: the check goes `NOT_SERVING` and a k8s deployment following the
probe recipe below actually removes the pod.  A shutdown signal also flips it
`NOT_SERVING` *before* the drain, so a load balancer sees the transition while
in-flight requests are still finishing.

```bash
# k8s-style probe
grpc-health-probe -addr 127.0.0.1:50051

# Specific service
grpcurl -plaintext -d '{"service":"oasr.speech.v1.Speech"}' \
        127.0.0.1:50051 grpc.health.v1.Health/Check
```

gRPC status mapping mirrors HTTP: `RESOURCE_EXHAUSTED`, `NOT_FOUND`,
`INVALID_ARGUMENT`, `UNIMPLEMENTED`, `FAILED_PRECONDITION`, `UNAVAILABLE`,
`INTERNAL`.

## Multi-GPU topology

`oasr-server` hosts exactly one engine.  To scale horizontally:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 oasr-server \
    --ckpt-dir /path/to/ckpt --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 oasr-server \
    --ckpt-dir /path/to/ckpt --http-bind 127.0.0.1:8081 --grpc-bind 127.0.0.1:50052 &
```

Put any L4/L7 load balancer (nginx, envoy, …) in front — sticky routing is
not required since one process serves a request end-to-end.

### Processes per GPU is mode-dependent

The right number of `oasr-server` processes **per GPU** differs by service mode:

| Mode | Recommended | Why |
|---|---|---|
| **offline** | **1 / GPU** | The batched forward already saturates the GPU. Two engines on one GPU thrash on CUDA-graph capture + memory and regress badly. Scale offline only **across** GPUs. |
| **streaming** | **2–3 / GPU** | Chunk-by-chunk decoding is launch/CPU-bound and leaves the GPU under-utilised. A second process is a second GIL, which interleaves kernel launches and fills the idle. |

For streaming, enabling **CUDA MPS** (`nvidia-cuda-mps-control -d`) lets the
processes' kernels run concurrently rather than time-slice, improving the
multiplier further. Each process is independent — front them with the same
load balancer.

### Streaming throughput levers

Three composable knobs, in order of simplicity:

1. **`--max-batch-size`** — batches more streams into each encoder / fbank / CTC
   launch, amortising launch overhead. Simplest, but with **diminishing returns**,
   and each step gets longer, so it **raises per-stream chunk latency**. Use it
   for throughput-biased or batch streaming, not interactive. Needs
   `--max-num-blocks ≥ max_batch_size × blocks_per_seq`.
2. **`--chunk-size`** — twice the audio per step is half the steps. Same latency
   trade-off; keep it small for interactive first-token latency.
3. **Processes per GPU** — the only lever that **also preserves per-stream
   latency**, because each process keeps a modest batch. It adds a second Python
   GIL, breaking the single-GIL launch-issuing ceiling that batch and chunk alone
   cannot.

They stack: throughput-biased → large batch + large chunk, plus extra processes
for the GIL-bound remainder; interactive → keep batch and chunk modest and scale
with processes.

Measured deltas for each lever, on a specific box, are in
`.artifacts/serving_perf.md`.

### Tuning knobs (exposed on `oasr-server`)

Beyond `--max-batch-size` / `--chunk-size` / `--preferred-batch-sizes` /
`--schedule-policy` / `--max-offline-pad-ratio`, the server now forwards the full
`EngineConfig` tuning surface: `--max-batch-frames`, `--length-bucket-ratio`,
`--max-wait-time`, `--streaming-cohort-admit`, `--partial-decode-interval`,
`--overlap-partial-readback`, `--enable-sequence-packing` / `--max-packed-frames`,
and the default-off `--use-ctc-cuda-graphs` / `--use-feature-cuda-graphs`
(**keep them off** — both measured to regress; `.artifacts/engine_perf.md` §1).
`oasr-server --help` lists them all.

**Memory sizing.** `--max-num-blocks 0` hands the paged KV pool to the engine,
which derives it from free VRAM at startup (`--gpu-memory-utilization`, default
0.90, is the share of the card it may occupy in total). That is what makes one
launch command portable across card sizes; unset keeps the fixed 2048-block
default. The same profile sizes the AR decoder-KV ceiling: leave
`--decode-kv-budget-gib` unset and it is derived, pass `0` to switch the byte
budget off. Both derivations are logged with their full arithmetic — see
[engine.md §6.1](engine.md#61-vram-aware-capacity-sizing).

> **Known limitation.** The front-end configures `tracing`, not Python's
> `logging`, so **no engine-side INFO line reaches the server's output** — the
> VRAM derivation, the streaming-cache ceiling and the long-form messages are all
> invisible. Only Python *warnings* surface. To see a derivation, construct the
> engine from Python with `logging.basicConfig(level=logging.INFO)`. Details and
> the outside-in workaround: `.artifacts/serving_perf.md` §5.

Multi-paradigm serving: `--decode-method` selects among the checkpoint's
advertised capabilities (e.g. `ctc_aed_rescoring` on a U2++ hybrid, `llm` on
a Qwen2-Audio checkpoint; unset = model default, validated at startup).  The
incremental AR families additionally take `--max-new-tokens`,
`--decode-steps-per-tick` (step cap per engine tick), `--max-tick-ms`
(wall-clock cap per tick — the actual dispatcher-starvation guard),
`--decode-admit-window-ms` (coalesce near-simultaneous arrivals into one decode
batch), `--max-decode-slots` (in-flight AR request cap), and `--llm-prompt`
(deployment-wide speech-LLM user prompt; per-request `prompt` decoding options
override it).  LLM decode emits token-streaming partials over the same
`Event::Partial` wire streaming CTC uses.

**`--decode-admit-window-ms` is the knob that buys AR throughput.** An AR decoder
step is weight-read bound, so its cost barely depends on how many rows it carries:
total decoder forwards is the *sum over groups* of each group's step count, and
groups cannot be merged after the fact (both decoder surfaces keep a shared scalar
generation offset — per-row offsets are the prerequisite, shared with paged decoder
KV). **So requests that arrive together are much cheaper than the same requests
arriving apart.**

The window holds a thin waiting queue until it reaches `max_batch_size` or expires,
recovering most of that loss. It costs up to one window of first-token latency for
an *isolated* request, so it is **off by default** — turn it on for
throughput-oriented deployments, leave it off when time-to-first-token dominates.

**`--max-tick-ms` is the knob that bounds latency, not `--decode-steps-per-tick`.**
A step count bounds work, not time, and step cost is model-dependent, so one
fixed step budget behaves very differently per model. Since the dispatcher holds
the GIL for a whole tick, tick p99 is the floor on cancel latency, admission
latency, and the interval between streaming partials — the deadline cuts it
sharply for a throughput cost inside run-to-run noise.

The residual p99 is the **prefill** tick (audio tower + projector + one LM forward
over the whole prompt), which the decode deadline deliberately does not bound; a
tick that spends its decode budget will not also prefill, so the two never stack.

Measurements for both knobs: `.artifacts/engine_perf.md` §3.

## The dispatcher

`oasr-engine-client::dispatcher` is the GIL-owning thread. It drains commands
from the tokio mpsc channel, replays them into Python (`add_request` /
`feed_chunk` / `cancel`), runs `engine.step()`, and pushes the resulting events
back through per-request channels. HTTP and gRPC handlers stay on tokio and never
touch the GIL.

**Admission coalescing.** Contiguous `CreateOffline` / `CreateStreaming`
envelopes are batched into one `add_requests_batch` Python call, which turns
shallow service batches into deep ones under `asyncio.gather`-style bursts.
`--admit-window-ms` waits that long after the first envelope for siblings;
`--admit-threshold` stops coalescing early once that many have drained.
`FeedChunk` / `Cancel` / `Ping` flush the admit batch first, to preserve
`CreateStreaming → FeedChunk` ordering — and a `Cancel` or `FeedChunk` in hand,
or landing mid-window, **ends** the wait, since the window used to tax the two
most latency-sensitive commands for its full duration.

**Tick pacing has two waits**, both a bounded `recv` (an arriving command wakes
them in microseconds) rather than a sleep:

| Wait | When |
|---|---|
| `IDLE_RECV_TIMEOUT` (500 ms) | the engine is empty |
| `NO_WORK_BACKOFF` (2 ms) | a tick received nothing, emitted nothing, **and** ran faster than `NO_WORK_TICK_MAX` (1 ms) |

The second exists because "the engine has requests" ≠ "the engine has work": one
open stream waiting on the client's next chunk would otherwise spin the thread
through empty steps for the whole session. **Gating on tick *duration* is what
keeps the backoff off the working paths** — an AR decode group grinding through
its per-tick step budget costs far more than 1 ms even when it emits nothing.

## Benchmarking

`benchmarks/bench_service.py` is a load generator for `oasr-server`.  It
auto-spawns the server unless `--server-url` is given, resolving it via
`$OASR_RS_BIN`, then the `oasr-server` console script on `PATH`, then
`rust/target/release/oasr-server`.

```bash
# Offline (HTTP + gRPC unary) — must launch the server in offline mode
python benchmarks/bench_service.py \
    --ckpt-dir /path/to/ckpt --audio-dir /path/to/wavs \
    --subroutines offline grpc_offline \
    --max-batch-size 64 --num-utterances 200 --concurrency 8

# Streaming (gRPC bidi) — must launch the server in streaming mode
python benchmarks/bench_service.py \
    --ckpt-dir /path/to/ckpt --audio-dir /path/to/wavs \
    --subroutines grpc_streaming \
    --max-batch-size 64 --num-utterances 200 --concurrency 8 \
    --chunk-ms 640
```

Available subroutines:

| Name | Transport |
|---|---|
| `offline` | `POST /v1/speech:recognize` (HTTP, raw PCM body — no base64/JSON) |
| `grpc_offline` | gRPC `Recognize` |
| `grpc_streaming` | gRPC `StreamingRecognize` |

> **Throughput note.** For high-throughput offline serving prefer **gRPC**
> (`grpc_offline`) — it stays well ahead of the HTTP path at scale on the
> reference box. The HTTP `offline` path uses a raw-PCM body (no base64/JSON),
> which is ~2× a base64-JSON body would be. The remaining gap between gRPC and
> the raw engine ceiling is GPU-bound + online-padding bound; scale out with
> **one `oasr-server` process per GPU** to multiply total throughput.

Output reports requests `ok` / `rejected` (server backpressure —
`RESOURCE_EXHAUSTED`) / `fail` (transport errors), wall-clock time, total
audio processed, **RTF**, throughput, and latency percentiles.  For
`grpc_streaming` it also reports first-partial latency and
partials-per-request.

## Metrics (`GET /metrics`)

Prometheus exposition of the values the dispatcher already computes per tick.

| Metric | Type | What it tells you |
|---|---|---|
| `oasr_dispatch_tick_seconds` | histogram | Wall time of one tick, **GIL held** (admit + step + extract). Its p99 is the worst-case latency a cancel, a new admission, or a streaming partial can experience — the number to watch when running an autoregressive decode family, where a batched decoder step is orders of magnitude slower than a CTC chunk. |
| `oasr_engine_step_seconds` | histogram | Time inside `ASREngine.step()`. |
| `oasr_dispatch_{admit,extract,route}_seconds` | histogram | The tick's sub-stages; `route` runs with the GIL released. |
| `oasr_engine_{running,waiting}` | gauge | Engine-reported queue depth. `running` includes parked AR generations. |
| `oasr_engine_outputs_total` | counter | `RequestOutput`s returned by `step()`. |
| `oasr_requests_{admitted,rejected,busy}_total` | counter | Accepted / rejected at admission (invalid options, mode mismatch) / refused at `--max-concurrent-requests`. |
| `oasr_engine_step_failures_total` | counter | `step()` raised. Three consecutive failures stop the readiness heartbeat, so `/readyz` and the gRPC health check go NotServing and a load balancer drains the process. |
| `oasr_requests_cancelled_total` | counter | Requests aborted before completion — almost always a client disconnect. A rising count next to a flat `oasr_engine_outputs_total` means clients are giving up before the engine answers. |
| `oasr_requests_failed_total{stage}` | counter | Requests the engine finished with an error, labelled by the stage that failed (`offline_forward`, `offline_oom`, `prefill_oom`, `streaming_forward`, `streaming_features`). Distinct from `oasr_engine_step_failures_total`: this one means the executor *contained* the failure to the requests responsible and kept ticking, so a rising count here with a flat step-failure count is isolation working, not an outage. |
| `oasr_events_{dropped,deferred}_total` | counter | Per-request channel was full: a partial was dropped (harmless — the next one supersedes it) or a **terminal** event was handed to a background task rather than lost. Sustained non-zero here means clients are reading slower than the engine emits. |

`--trace-dispatch` additionally logs rolling 2 s means of the same sub-stages at
INFO; the histograms above are the ones to alert on.

## Limits, timeouts and shutdown

Every bound below is explicit and logged at startup (`"serving limits"`), because
the defaults that bit hardest were the ones nobody had written down.

| Flag | Default | What it bounds |
|---|---|---|
| `--max-audio-mib` | `256` | Largest audio payload per request. Drives **both** the HTTP body cap (including the multipart routes) and gRPC's `max_decoding_message_size`. One number, because tonic's undeclared 4 MiB default against HTTP's 256 MiB meant the same ~2-minute clip was accepted on REST and rejected on the surface this doc recommends for offline throughput — and axum's own 2 MiB multipart default would have reintroduced exactly that asymmetry on the upload routes. |
| `--max-audio-seconds` | `14400` (4 h) | Longest **decoded** waveform. A separate bound because a compressed container breaks the relationship between body bytes and audio seconds: a few MiB of MP3 is hours of waveform, allocated before anything could notice. Exceeding it is `413`, never a truncated transcript. `0` disables. |
| `--served-model-name` | *(unset)* | Names the OpenAI surface's `model` field must match. Unset accepts any name — what makes a client hardcoded to `whisper-1` work after a base-URL change. Set, an unknown name is `404`. Repeatable. |
| `--cors-allow-origin` | *(unset)* | Origins allowed to call the HTTP API from a browser; `*` allows any. No CORS layer is installed when unset. Repeatable. |
| `--request-timeout-secs` | `300` | Deadline for one **unary** request (HTTP `speech:recognize`, gRPC `Recognize`), covering time queued behind the concurrency limit. `0` disables. |
| `--stream-idle-timeout-secs` | `300` | Aborts a streaming RPC that goes this long with no inbound audio (before half-close) or no decode event (after). `0` disables. Deliberately *not* a blanket gRPC deadline — that would cut off healthy long-lived streams; a live stream can only be bounded by inactivity. |
| `--max-inflight-connections` | `4 x --max-concurrent-requests` | Requests either listener processes at once; the rest queue (and are eventually cut off by the timeout). Bounds how many multi-MiB bodies are resident. `/healthz`, `/readyz` and `/metrics` are **exempt** — a saturated server must still answer its own probes. `0` disables. |
| `--shutdown-grace-secs` | `10` | How long in-flight requests get to finish after SIGTERM/SIGINT before the listeners are dropped. |

The dispatcher's command channel is derived from `--max-concurrent-requests`
(2x, floor 64) rather than fixed, because every queued `CreateOffline` envelope
holds its full audio payload: the two bounds have to be related or the channel
buffers several times the admissible work, each up to `--max-audio-mib`.

**Shutdown is graceful.**  SIGTERM flips the gRPC health check and `/readyz` to
not-serving, stops both listeners accepting, and waits up to the grace period for
in-flight requests to complete — none are dropped.

**Client disconnects cancel.**  All three request handles — streaming, offline
streaming, and the unary one behind HTTP `speech:recognize` / gRPC `Recognize` —
cancel their engine request when the handler future is dropped.  Watch
`oasr_requests_cancelled_total`.

## Operational tips

- Tune `--max-concurrent-requests` to a small multiple of the engine's
  `max_batch_size`.  Excess load returns HTTP 503 /
  gRPC `RESOURCE_EXHAUSTED` — clients should back off and retry.
- Watch `oasr_dispatch_tick_seconds` p99. If it is far above your latency budget
  on an AR decode family, lower `--decode-steps-per-tick`.
- For local development, a single process pinned to one GPU keeps things
  simple; multi-GPU scale comes from running additional `oasr-server`
  processes behind a load balancer.
- If the engine OOMs or panics mid-step, the process exits non-zero.  Use a
  process manager (systemd, supervisord, k8s) to restart it.

## Out of scope (v1)

- AuthN/AuthZ — rely on a network policy or reverse proxy. An `Authorization`
  header is accepted and forwarded by the clients so a proxy can enforce one;
  the server itself does not check it.
- Cross-host engine fleets — single-host only at the binary level; clusters
  go through your LB.
- **Opus**, AMR, AMR-WB and Speex — see [Audio formats](#audio-formats). Every
  other common container decodes.
- **Word-level timestamps** — `timestamp_granularities[]=word` returns `501`.
  Tracked as H7 in `.artifacts/architecture_review.md`.
- TLS termination — assume a reverse proxy handles it.
- `LongRunningRecognize` (Google STT v1 LRO) — not implemented.
- `RecognitionConfig.model`, `audio_channel_count`, and
  `StreamingRecognitionConfig.single_utterance` — accepted, ignored.
  (`language_code` is now honored where the decode family can act on it.)
