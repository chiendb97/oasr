# `rust/` — OASR serving frontend

Cargo workspace that builds `oasr-server`, the binary the Python `oasr-server`
console script execs.  It hosts **one in-process Python `ASREngine` per
process** via PyO3 and exposes a Google Speech-to-Text v1-shaped API:

- **HTTP** (`axum`):
  - `POST /v1/speech:recognize` — synchronous recognition (offline mode).
  - `GET /v1/models` — loaded model metadata.
  - `GET /healthz`, `GET /readyz`, `GET /metrics`.
- **gRPC** (`tonic`): `oasr.speech.v1.Speech/Recognize` (unary, offline mode)
  and `oasr.speech.v1.Speech/StreamingRecognize` (bidi, streaming mode), plus
  the standard `grpc.health.v1.Health` service.

Multi-GPU scale = launch N `oasr-server` processes behind a load balancer
(each with `CUDA_VISIBLE_DEVICES` set), not multi-worker within one process.

## Build

System dependencies:

- A Python development install (PyO3 links `libpython` via `auto-initialize`)
- `protobuf-compiler` (for `tonic-build` to compile `proto/oasr_speech_v1.proto`)
- A C/C++ toolchain

```bash
cd rust
cargo build --release
# binary: rust/target/release/oasr-server
```

## Run

The binary embeds the engine in-process via PyO3.  Make sure `oasr` is
importable (e.g. `pip install -e .`):

```bash
./target/release/oasr-server \
    --ckpt-dir /path/to/wenet/ckpt \
    --service-mode offline \
    --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051
```

See `--help` for the full CLI surface.

## Layout

| Crate | Role |
|---|---|
| `oasr-wire` | Shared event/command types (`Cmd`, `Event`, `ErrorCode`, `ModelInfo`) |
| `oasr-engine-client` | PyO3-backed driver: `PyEngine`, `EngineDispatcher`, async client/pool |
| `oasr-asr` | Audio decode (WAV via `hound`, raw PCM) to f32 mono `bytes::Bytes` |
| `oasr-server-http` | axum routes for the Google STT v1 REST surface |
| `oasr-server-grpc` | tonic `oasr.speech.v1.Speech` service (unary + bidi) |
| `oasr-server` | Binary: CLI, Python interpreter init, engine + HTTP + gRPC wiring |
