# OASR Web Demo

A browser demo for OASR: upload a file or record from the microphone, pick **offline** or
**streaming** mode, and see the transcript.

```
Browser ──HTTP / WebSocket──▶ FastAPI bridge (server.py) ──gRPC──▶ oasr-server
```

The bridge exists because a web page cannot speak gRPC and the server sends no CORS headers.
`server.py` serves this page *and* relays to OASR, so everything is same-origin: `POST
/api/recognize` for offline, `WS /api/stream` for streaming.

## Run it

```bash
# 1. An OASR server — one streaming server serves both demo modes
oasr-server --ckpt-dir /path/to/ckpt --service-mode streaming --grpc-bind 127.0.0.1:50051

# 2. The bridge
pip install -r examples/web/requirements.txt
python examples/web/server.py

# 3. Open http://localhost:8000
```

> **Microphone needs a secure context** — HTTPS, or `localhost` / `127.0.0.1`. On a bare LAN
> address over plain HTTP the browser hides `getUserMedia` entirely. File upload works everywhere.
> For a remote bridge, forward the port (`ssh -L 8000:localhost:8000 <user>@<host>`) or serve TLS
> with `--ssl-certfile` / `--ssl-keyfile`.

`python examples/web/server.py --help` lists every flag — bind address, upstream addresses,
`--chunk-ms`, TLS, logging. See `docs/serving.md` for the OASR server itself.

## Notes

- Offline mode pushes the whole clip through `StreamingRecognize` and shows only the final
  transcript. To use the unary `Recognize` RPC instead, run a second `--service-mode offline`
  server and point `--offline-addr` at it.
- Streaming + upload paces the file through at ~realtime, so partials stream in as they would live.
- Every hop is traced under one correlation id, also returned as the `x-oasr-trace` response
  header; `--log-level debug` adds per-chunk detail, `--log-file PATH` mirrors it to a file.
- Audio is normalized in-browser to 16 kHz mono little-endian f32 PCM (`LINEAR32F`), the format the
  gRPC RPCs expect. Stubs are compiled at startup from `rust/proto/oasr_speech_v1.proto`.
- CLI equivalents: `examples/recognize/grpc_client.py`, `examples/recognize/http_client.py`.
