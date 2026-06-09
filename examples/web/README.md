# OASR Web Demo

A tiny browser demo for the OASR speech recognizer. Provide audio by **uploading
a file** or **recording from the microphone**, choose **offline** or
**streaming** mode, and see the transcript.

```
Browser (this page)
  ├─ Web Audio API → 16 kHz mono Float32 PCM
  ├─ Offline : POST /api/recognize  → JSON transcript
  └─ Stream  : WS   /api/stream      → live partial / final
        │
        ▼
FastAPI bridge (server.py, same origin)
        │  gRPC
        ▼
oasr-server  (oasr.speech.v1.Speech)
```

## Why a bridge?

The browser cannot talk to the OASR server directly: the server speaks **gRPC**
(`Recognize` unary + `StreamingRecognize` bidi) and a **no-CORS** HTTP endpoint,
neither reachable from a web page cross-origin. `server.py` is a thin FastAPI app
that serves this page *and* relays to OASR over gRPC, so everything is
same-origin. It exposes:

| Endpoint | Used by | Talks to OASR via |
|----------|---------|-------------------|
| `POST /api/recognize` | offline mode | unary `Recognize`, or `StreamingRecognize` (see below) |
| `WS /api/stream` | streaming mode | bidi `StreamingRecognize` |

## Setup

### 1. Start an OASR server

An `oasr-server` process is pinned to **one** `--service-mode` for its lifetime.
The simplest setup is a single **streaming** server that serves both demo modes:

```bash
oasr-server \
    --ckpt-dir /path/to/ckpt \
    --service-mode streaming \
    --grpc-bind 127.0.0.1:50051
```

In this configuration the demo's *offline* button pushes the whole clip through
`StreamingRecognize` and shows only the final transcript, while *streaming* shows
live partials. (See `docs/serving.md` for server details.)

### 2. Install bridge deps + run the bridge

```bash
pip install -r examples/web/requirements.txt
python examples/web/server.py        # serves http://127.0.0.1:8000
```

### 3. Open the demo

Open **http://localhost:8000** in a browser.

> **Microphone requires a secure context.** Browsers only expose
> `getUserMedia` over HTTPS or on `http://localhost` / `http://127.0.0.1`.
> Opening the page on a bare LAN IP/hostname over plain HTTP (e.g.
> `http://10.0.0.5:8000`) hides the microphone API entirely — you'll see
> *"Microphone needs a secure context"*. **File upload works everywhere.**

If the bridge runs on a remote box and you want microphone capture, pick one:

* **SSH tunnel (no certs)** — forward the port so the browser sees `localhost`:

  ```bash
  ssh -L 8000:localhost:8000 <user>@<host>
  # then open http://localhost:8000 on your machine
  ```

* **HTTPS** — serve TLS so any host is a secure context (self-signed is fine;
  accept the browser warning):

  ```bash
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
      -keyout key.pem -out cert.pem -subj "/CN=localhost"
  python examples/web/server.py --host 0.0.0.0 \
      --ssl-certfile cert.pem --ssl-keyfile key.pem
  # then open https://<host>:8000
  ```

## Using the demo

- **Offline + Upload** — pick an audio file, click *Transcribe*; the transcript
  appears once.
- **Offline + Microphone** — *Record*, speak, *Stop*; the transcript appears.
- **Streaming + Microphone** — *Record*; partials update live, finals commit.
- **Streaming + Upload** — the file is paced through ~realtime so you can watch
  partials stream in.

## Options

```
python examples/web/server.py \
    --host 127.0.0.1 --port 8000 \
    --streaming-addr 127.0.0.1:50051 \
    --offline-addr  127.0.0.1:50051 \   # default: same as --streaming-addr
    --sample-rate 16000 \
    --chunk-ms 320 \
    --ssl-certfile cert.pem --ssl-keyfile key.pem   # optional, enables HTTPS
```

### Using the dedicated offline endpoint (two servers)

To exercise the real unary `Recognize` RPC instead of streaming the whole clip,
run a second **offline** server and point `--offline-addr` at it. When the two
addresses differ, offline requests use unary `Recognize`:

```bash
oasr-server --ckpt-dir /path/to/ckpt --service-mode offline   --grpc-bind 127.0.0.1:50051
oasr-server --ckpt-dir /path/to/ckpt --service-mode streaming --grpc-bind 127.0.0.1:50052

python examples/web/server.py \
    --offline-addr 127.0.0.1:50051 --streaming-addr 127.0.0.1:50052
```

## Notes

- Audio is normalized in-browser to **16 kHz mono little-endian f32 PCM**
  (`LINEAR32F`), the format the gRPC streaming/unary RPCs expect.
- The gRPC stubs are compiled at startup from `rust/proto/oasr_speech_v1.proto`
  via `grpc_tools.protoc` (same pattern as `examples/recognize/grpc_client.py`).
- This demo is intentionally minimal: top-1 alternative only, no diarization.
- For a CLI equivalent, see `examples/recognize/grpc_client.py` and
  `examples/recognize/http_client.py`.
