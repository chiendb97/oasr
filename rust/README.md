# `rust/` — OASR serving frontend

Cargo workspace that builds `oasr-server`, the binary the Python `oasr-server`
console script execs.  It exposes:

- **HTTP** (`axum`): native `/v1/transcriptions`, `/v1/stream` (WebSocket),
  OpenAI Whisper-compatible `/v1/audio/transcriptions`, `/healthz`, `/readyz`,
  `/metrics`, `/v1/models`.
- **gRPC** (`tonic`): `oasr.asr.v1.Speech/Recognize` (unary) and
  `oasr.asr.v1.Speech/StreamingRecognize` (bidi).

The frontend drives a fleet of N Python worker processes (one per GPU) over
ZeroMQ + MessagePack.  Streaming requests stick to the worker that admitted
them; offline requests go to the least-loaded worker.

## Build

System dependencies:

- `libzmq3-dev` (for the `zmq` crate)
- `protobuf-compiler` (for `tonic-build` to compile `proto/oasr_asr.proto`)
- A C/C++ toolchain (libzmq's build invokes it)

```bash
cd rust
cargo build --release
# binary: rust/target/release/oasr-server
```

## Run

The binary spawns Python workers itself.  Make sure `oasr` is importable
(e.g. `pip install -e .[serving]`):

```bash
./target/release/oasr-server \
    --ckpt-dir /path/to/wenet/ckpt \
    --num-workers 2 --cuda-devices 0,1 \
    --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051
```

See `--help` for the full CLI surface.

## Layout

| Crate | Role |
|---|---|
| `oasr-wire` | MessagePack wire schema shared with `oasr/serving/ipc.py` |
| `oasr-engine-client` | Async ZMQ DEALER client, per-request demux, multi-worker pool |
| `oasr-asr` | Audio decode (WAV via `hound`, raw PCM) to f32 mono `bytes::Bytes` |
| `oasr-server-http` | axum routes (native + Whisper-compat + WebSocket) |
| `oasr-server-grpc` | tonic Speech service (unary + bidi streaming) |
| `oasr-server` | Binary: CLI, worker supervisor, runtime wiring |
