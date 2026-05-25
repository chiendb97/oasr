# OASR Serving — Rust frontend

The `rust/` Cargo workspace builds `oasr-server`, a Rust binary that exposes
the OASR engine over both HTTP and gRPC.  It spawns one or more Python worker
processes (each pinned to a CUDA device) and proxies requests to them over
ZeroMQ + MessagePack.

## Quick start

```bash
# 1. Build the Rust binary
cd rust && cargo build --release && cd ..

# 2. Install Python deps for the serving extras (msgpack + pyzmq)
pip install -e .[serving]

# 3. Launch — Rust spawns Python workers itself
./rust/target/release/oasr-server \
    --ckpt-dir /path/to/wenet/ckpt \
    --num-workers 2 --cuda-devices 0,1 \
    --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051
```

`./rust/target/release/oasr-server --help` lists every flag.

## HTTP API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/transcriptions` | Offline.  Body = raw WAV/PCM (see below). |
| `GET`  | `/v1/stream` | WebSocket upgrade for bidirectional streaming. |
| `POST` | `/v1/audio/transcriptions` | OpenAI Whisper-compatible multipart upload. |
| `GET`  | `/v1/models` | Loaded model metadata (echoed from `Pong.model_info`). |
| `GET`  | `/healthz` | Server liveness. |
| `GET`  | `/readyz` | 200 iff `≥ 1` worker has produced a Pong in the last 5 s. |
| `GET`  | `/metrics` | Prometheus exposition. |

### `POST /v1/transcriptions`

```bash
# raw WAV
curl -sS -X POST --data-binary @audio.wav \
     -H 'Content-Type: audio/wav' \
     http://127.0.0.1:8080/v1/transcriptions | jq

# raw f32 LE mono PCM, sample rate via query
curl -sS -X POST --data-binary @audio.f32 \
     -H 'Content-Type: application/octet-stream' \
     'http://127.0.0.1:8080/v1/transcriptions?sample_rate=16000&encoding=f32_le' | jq

# i16 LE
curl -sS -X POST --data-binary @audio.i16 \
     -H 'Content-Type: application/octet-stream' \
     'http://127.0.0.1:8080/v1/transcriptions?sample_rate=16000&encoding=i16_le' | jq
```

Response (success):

```json
{
  "request_id": "8f4c9b...",
  "text": "hello world",
  "tokens": [[1, 2, 3]],
  "scores": [-0.12]
}
```

On error: standard HTTP statuses (400 for decode, 503 for over-capacity, 500
for internal); body is `{"request_id": "...", "code": "...", "message": "..."}`.

### `POST /v1/audio/transcriptions` (Whisper-compat)

```bash
curl -sS -X POST -F file=@audio.wav \
     http://127.0.0.1:8080/v1/audio/transcriptions
```

Returns `{"text": "..."}` by default.  Pass `response_format=text` for plain
text, `response_format=verbose_json` for an OpenAI-shaped object (segments
empty in v1).  `language`, `prompt`, `temperature`, `timestamp_granularities`,
and `model` are accepted and ignored.

### `GET /v1/stream` (WebSocket)

**Client → server**:

| Frame | Form | Purpose |
|---|---|---|
| Text | `{"type":"start","sample_rate":16000,"format":"pcm_f32le","priority":0}` | First message; opens the session.  `format` may also be `"pcm_i16le"`. |
| Binary | raw PCM bytes in the declared `format` | Audio chunks (any size; ~640 ms recommended). |
| Text | `{"type":"end"}` | Flush + finalize. |
| Text | `{"type":"cancel"}` | Abort. |

**Server → client** (all text JSON):

| Field shape | When |
|---|---|
| `{"type":"accepted","request_id":"..."}` | Once admitted. |
| `{"type":"partial","request_id":"...","text":"...","tokens":[[...]],"scores":[...]}` | Per partial. |
| `{"type":"final","request_id":"...","text":"...","tokens":[[...]],"scores":[...]}` | Once finalized. |
| `{"type":"error","code":"...","message":"..."}` | On failure. |

```bash
python scripts/ws_stream.py --url ws://127.0.0.1:8080/v1/stream \
    --wav tests/fixtures/hello.wav --chunk-ms 640
```

## gRPC API

Proto: `rust/proto/oasr_asr.proto`.  Service: `oasr.asr.v1.Speech`.

### `Recognize` (unary, offline)

```bash
grpcurl -plaintext -d @ -import-path rust/proto -proto oasr_asr.proto \
        127.0.0.1:50051 oasr.asr.v1.Speech/Recognize <<EOF
{"config":{"encoding":"WAV","sample_rate_hertz":16000},
 "audio":"$(base64 < audio.wav)"}
EOF
```

### `StreamingRecognize` (bidi)

```bash
python scripts/grpc_stream.py --addr 127.0.0.1:50051 \
    --wav tests/fixtures/hello.wav --chunk-ms 640
```

The first message **must** carry `streaming_config.config`; subsequent
messages carry `audio_content` (raw PCM bytes in the declared encoding).
Set `interim_results=false` to suppress partials.

## Multi-worker fleet

By default `oasr-server` spawns one Python worker per `--num-workers`
(default 1) on TCP ports starting at `--worker-port-base` (55580).  Each
worker is pinned to `CUDA_VISIBLE_DEVICES=k` where `k` indexes into the
comma-separated `--cuda-devices` list.

The pool routes:
- **offline**: least-loaded worker (`num_running + num_waiting` minimum).
- **streaming**: least-loaded at admission, then **sticky** by `request_id`
  for subsequent chunks / Cancel.  Streaming state cannot migrate between
  workers, so a worker death triggers `Event::Error{code: WORKER_LOST}` for
  every in-flight stream it owned (see `EnginePool::fail_worker`).

`Pong` events from each worker advance the load gauge and `last_pong_at`.
`/readyz` returns 200 iff any worker has ponged within the last 5 s.

## Wire format (IPC between Rust and Python)

| Direction | Transport | Encoding |
|---|---|---|
| Rust DEALER → Python ROUTER | ZMTP (TCP) | `[header_msgpack, optional_audio_payload]` |
| Python ROUTER → Rust DEALER | ZMTP (TCP) | `[header_msgpack]` |

The schema is defined in `rust/crates/oasr-wire/src/lib.rs` and mirrored in
`oasr/serving/ipc.py`.  Headers are MessagePack maps with a `"type"` tag:
`Cmd` variants are `CreateOffline`, `CreateStreaming`, `FeedChunk`, `Cancel`,
`Ping`; `Event` variants are `Accepted`, `Partial`, `Final`, `Error`, `Pong`,
`Overloaded`.

Audio payloads are raw little-endian `f32` mono PCM samples, decoded
zero-copy on the Python side via `np.frombuffer(payload, dtype=np.float32)`.

> **Transport note.** The pure-Rust `zeromq` crate (v0.4) currently
> interoperates reliably with `pyzmq` only over `tcp://`.  `ipc://` (unix
> sockets) is supported by both libraries but hangs the ZMTP handshake under
> some Linux configurations.  The default endpoint base is therefore `tcp://`;
> override via `--zmq-endpoint-base` and `--worker-port-base`.

## Benchmarking the service

`benchmarks/bench_service.py` is a load generator for `oasr-server`.  It
mirrors the CLI shape of `benchmarks/bench_engine.py` so the same
`--ckpt-dir` / `--audio-dir` / `--num-utterances` arguments work, and it
auto-spawns the Rust binary unless `--server-url` is given.

```bash
# Auto-spawns oasr-server, runs offline + streaming, prints stats.
python benchmarks/bench_service.py \
    --ckpt-dir /path/to/wenet/ckpt \
    --audio-dir /path/to/wavs \
    --subroutines offline streaming \
    --max-batch-size 64 --num-utterances 200 \
    --concurrency 8 --worker-threads 2

# Against an already-running server, write a JSON summary.
python benchmarks/bench_service.py \
    --audio-dir /path/to/wavs --server-url http://127.0.0.1:8080 \
    --subroutines streaming --num-utterances 500 -c 8 \
    --output-path /tmp/bench.json
```

Output reports requests `ok` / `rejected` (server backpressure — Error event
or HTTP 503) / `fail` (transport errors), wall-clock time, total audio
processed, **RTF**, throughput, and latency percentiles.  For streaming, it
also reports first-partial latency and partials-per-request.

### Tuning

- **`--concurrency`** is the bench-side in-flight cap.  The engine processes
  ~25–30 streams/sec on one RTX 5090 with the bundled u2pp Conformer.  Pick
  `concurrency` around that throughput × desired latency budget; pushing
  much beyond starts to see backpressure responses.
- **`--worker-threads 2`** runs the Python worker with an I/O thread plus a
  step thread (overlaps ZMQ ingestion with `engine.step()`).  Roughly +30 %
  RTF on the streaming path in our test runs.
- **`--max-batch-size 64`** raises the engine's concurrent streaming pool;
  pair with a higher `--num-workers` if you have multiple GPUs.

### Reference numbers (RTX 5090, u2pp Conformer, LJSpeech)

`--max-batch-size 64 --concurrency 8 --worker-threads 2`, audio at 16 kHz:

| Subroutine | N | Wall | ok | rejected | RTF | p50 / p95 / p99 latency |
|---|---|---|---|---|---|---|
| offline (POST /v1/transcriptions) | 200 | 6.45 s | 200 | 0 | **200×** | 140 / 300 / 2904 ms |
| streaming (WebSocket) | 200 | 7.46 s | 192 | 8 | **164×** | 188 / 572 / 3141 ms |

The p99 outliers are first-request cold-path costs (CUDA Graph capture,
JIT warmup).  Steady-state p50/p95 are representative.

## Operational tips

- Tune `--max-concurrent-requests` to the engine's `max_batch_size *
  offline_pipeline_depth`.  Excess load returns HTTP 503 / gRPC
  `RESOURCE_EXHAUSTED` (a tail of `Event::Overloaded` events) — the client
  should back off and retry.
- For local development, `--num-workers 1 --cuda-devices 0` keeps things
  simple; horizontal scale comes from raising `--num-workers` on a multi-GPU
  host.
- If the engine OOMs or panics mid-step, the worker exits non-zero.  In v1
  the supervisor does **not** auto-restart — operator restart is required.

## Out of scope (v1)

- AuthN/AuthZ — rely on a network policy or reverse proxy.
- Cross-host worker fleets — single-host only.
- Audio codecs beyond PCM/WAV — MP3/Opus/FLAC follow up behind a `symphonia`
  feature on `oasr-asr`.
- TLS termination — assume a reverse proxy handles it.
- Whisper `language` / `prompt` / `temperature` / `timestamp_granularities`
  (accepted, ignored).
