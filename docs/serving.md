# OASR Serving — Rust frontend

The `rust/` Cargo workspace builds the OASR serving core, which exposes the
engine over both HTTP and gRPC.  The API surface is shaped after **Google Cloud
Speech-to-Text v1** so existing tooling (REST conventions, `grpcurl`,
OpenAPI-style clients) feels familiar.

`pip install` compiles the core into the `oasr._core` PyO3 extension module (via
setuptools-rust) and installs the `oasr-server` console script that runs it, so
the front-end ships with the wheel — no separate build step.  The same code also
builds as a standalone binary (`rust/crates/oasr-server`) for `cargo`-only
workflows; both share the `oasr-serve` crate.

The engine runs **in-process** via PyO3 — one Python `ASREngine` per
`oasr-server` process.  Multi-GPU scale is achieved by launching N
`oasr-server` processes (each with `CUDA_VISIBLE_DEVICES` set), not by
multiplexing inside one process.

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
```

`oasr-server --help` lists every flag.  `--service-mode` pins the engine to
either `offline` (sync `Recognize`) or `streaming` (bidi `StreamingRecognize`)
for its entire lifecycle; the mismatched RPC returns `FAILED_PRECONDITION`.

> Building the workspace with `cargo build --release` instead produces the same
> server as the `rust/target/release/oasr-server` binary; substitute that path
> for `oasr-server` in the commands below if you go that route.

## API surface

| Surface | Path / Service | Notes |
|---|---|---|
| HTTP | `POST /v1/speech:recognize` | Synchronous unary recognition (offline mode). |
| HTTP | `GET /v1/models` | Loaded model metadata. |
| HTTP | `GET /healthz` | Process liveness. |
| HTTP | `GET /readyz` | 200 once the engine dispatcher has produced its first Pong. |
| HTTP | `GET /metrics` | Prometheus exposition. |
| gRPC | `oasr.speech.v1.Speech/Recognize` | Synchronous unary (offline mode). |
| gRPC | `oasr.speech.v1.Speech/StreamingRecognize` | Bidi streaming (streaming mode). |
| gRPC | `grpc.health.v1.Health/Check` and `Watch` | Standard gRPC health checking. |

REST is sync only — there is **no HTTP streaming endpoint** (no WebSocket, no
SSE).  Streaming clients must use the gRPC `StreamingRecognize` RPC, matching
the Google STT v1 contract.

The previous Whisper-compat (`POST /v1/audio/transcriptions`) and native
binary (`POST /v1/transcriptions`) endpoints have been removed.

## HTTP

### `POST /v1/speech:recognize`

Request body mirrors Google's STT v1 JSON transcoding.  Audio is carried
inline as base64 in `audio.content` (a `uri` field is reserved but returns
`UNIMPLEMENTED`).

```bash
# Base64-encode and POST a WAV file
B64=$(base64 -w0 audio.wav)
curl -sS -X POST http://127.0.0.1:8080/v1/speech:recognize \
     -H 'Content-Type: application/json' \
     -d "$(jq -n --arg b64 "$B64" \
           '{config:{encoding:"WAV",sampleRateHertz:16000,languageCode:"en-US"},
             audio:{content:$b64}}')" | jq
```

`encoding` accepted values:

| Value | Meaning |
|---|---|
| `LINEAR16` | Little-endian 16-bit signed PCM, mono. |
| `LINEAR32F` | Little-endian 32-bit float PCM, mono *(OASR extension)*. |
| `WAV` | RIFF/WAV container; embedded sample rate wins over `sampleRateHertz` *(OASR extension)*. |
| any other Google STT v1 codec | Returns `UNIMPLEMENTED`. |

Success response (`200`):

```json
{
  "results": [
    {
      "alternatives": [
        {
          "transcript": "hello world",
          "confidence": 0.0,
          "tokens": [12, 305, 119]
        }
      ]
    }
  ],
  "requestId": "8f4c9b..."
}
```

Error responses use the canonical Google error envelope:

```json
{ "error": { "code": 400, "status": "INVALID_ARGUMENT", "message": "..." } }
```

| HTTP status | `status` field | When |
|---|---|---|
| 400 | `INVALID_ARGUMENT` | malformed JSON, missing encoding, bad audio bytes, missing `audio.content` |
| 400 | `FAILED_PRECONDITION` | server is in `streaming` mode |
| 404 | `NOT_FOUND` | unknown request id (internal bug) |
| 501 | `UNIMPLEMENTED` | unsupported encoding, `audio.uri` |
| 503 | `RESOURCE_EXHAUSTED` | over-capacity, retry with backoff |
| 503 | `UNAVAILABLE` | dispatcher shutting down or engine lost |
| 500 | `INTERNAL` | otherwise |

### `GET /v1/models`

```json
{
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
        "decoder_type": "ctc_gpu",
        "vocab_size": 5000
      }
    }
  ]
}
```

Exactly one entry — the single model loaded by this process.

## gRPC

Service: `oasr.speech.v1.Speech` in `rust/proto/oasr_speech_v1.proto`.
Messages mirror Google's v1 schema; `tokens` (CTC token IDs) and `requestId`
are OASR extensions in the reserved field-number range.

### `Recognize` (unary, offline mode)

```bash
B64=$(base64 -w0 audio.wav)
grpcurl -plaintext -import-path rust/proto -proto oasr_speech_v1.proto \
        -d "$(jq -n --arg b64 "$B64" \
              '{config:{encoding:"WAV",sampleRateHertz:16000,languageCode:"en-US"},
                audio:{content:$b64}}')" \
        127.0.0.1:50051 oasr.speech.v1.Speech/Recognize
```

### `StreamingRecognize` (bidi, streaming mode)

The first inbound message **must** carry `streaming_config.config`;
subsequent messages carry `audio_content` (raw PCM bytes in the declared
encoding).  Set `streaming_config.interim_results=false` to suppress
partials.  Each response message contains one `StreamingRecognitionResult`
with `is_final=true` on the terminal frame.

```bash
python scripts/grpc_stream.py --addr 127.0.0.1:50051 \
    --wav tests/fixtures/hello.wav --chunk-ms 640
```

### gRPC health checking

The binary exposes the standard `grpc.health.v1.Health` service.  Both an
empty service name (overall process health) and `oasr.speech.v1.Speech`
report `SERVING` once the engine dispatcher has produced its first tick, and
flip to `NOT_SERVING` during shutdown.

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
| `offline` | `POST /v1/speech:recognize` (HTTP, JSON+base64) |
| `grpc_offline` | gRPC `Recognize` |
| `grpc_streaming` | gRPC `StreamingRecognize` |

Output reports requests `ok` / `rejected` (server backpressure —
`RESOURCE_EXHAUSTED`) / `fail` (transport errors), wall-clock time, total
audio processed, **RTF**, throughput, and latency percentiles.  For
`grpc_streaming` it also reports first-partial latency and
partials-per-request.

## Operational tips

- Tune `--max-concurrent-requests` to a small multiple of the engine's
  `max_batch_size`.  Excess load returns HTTP 503 /
  gRPC `RESOURCE_EXHAUSTED` — clients should back off and retry.
- For local development, a single process pinned to one GPU keeps things
  simple; multi-GPU scale comes from running additional `oasr-server`
  processes behind a load balancer.
- If the engine OOMs or panics mid-step, the process exits non-zero.  Use a
  process manager (systemd, supervisord, k8s) to restart it.

## Out of scope (v1)

- AuthN/AuthZ — rely on a network policy or reverse proxy.
- Cross-host engine fleets — single-host only at the binary level; clusters
  go through your LB.
- Audio codecs beyond PCM/WAV — MP3/Opus/FLAC follow up behind a `symphonia`
  feature on `oasr-asr`.
- TLS termination — assume a reverse proxy handles it.
- `LongRunningRecognize` (Google STT v1 LRO) — not implemented.
- `RecognitionConfig.language_code`, `model`, `audio_channel_count`,
  `max_alternatives` semantics beyond clipping the alternative list, and
  `StreamingRecognitionConfig.single_utterance` — accepted, ignored.
