# OASR Web Demo

A browser demo for OASR: upload a file or record from the microphone, pick **offline** or
**streaming** mode, and see the transcript. No build step and no JavaScript toolchain — three
static files and one standard-library Python script.

```
Browser ──HTTP / WebSocket──▶ server.py ──HTTP / WebSocket──▶ oasr-server
```

`server.py` serves the page **and** relays its API calls, so the browser only ever talks to a
single origin. That is what keeps the setup to one command wherever `oasr-server` happens to
run: nothing to configure for CORS, and `oasr-server` needs no network exposure of its own —
which matters, because it has no authentication.

The page exercises the same HTTP API any client would:

| Mode | Call |
|---|---|
| Offline | `POST /v1/audio/transcriptions` — the file is uploaded **as-is**; the server decodes MP3 / M4A / FLAC / OGG / WAV |
| Streaming | `WS /v1/realtime` — a `session.update` frame, then binary PCM chunks, then `input_audio_buffer.commit` |

Both modes work against either `--service-mode`. A `streaming` server decodes as the audio
arrives; an `offline` one buffers the utterance and streams the *text* back, so a speech-LLM
still fills in token by token.

## Quick start

Everything on one machine:

```bash
oasr-server --ckpt-dir /path/to/ckpt --service-mode streaming
python examples/web/server.py
# open http://localhost:8000
```

`--oasr-server` defaults to `$OASR_SERVER_URL`, then to `http://127.0.0.1:8080`, which matches
`oasr-server`'s default HTTP listener (`--http-bind 0.0.0.0:8080`) — so neither command needs a
flag. `server.py` requires only the standard library of the Python you already use for OASR.

## When `oasr-server` runs elsewhere

The relay is the only thing that has to reach `oasr-server`, so pick whichever of these
matches your network. In all three the browser talks only to the relay, so no CORS is
involved and the page needs no configuration.

**Reachable directly** (same LAN, VPN, container network, Kubernetes service):

```bash
python examples/web/server.py --oasr-server http://asr-host:8080
```

**Reachable only through SSH** (bastion, firewalled host) — forward the port, and the relay's
default already points at it, so it takes no flag:

```bash
ssh -N -o ServerAliveInterval=30 -L 8080:127.0.0.1:8080 user@asr-host
python examples/web/server.py
```

`ServerAliveInterval` is worth setting: a tunnel that dies mid-stream looks exactly like a
crashed server.

**No checkout on the machine you browse from** — run the relay alongside `oasr-server` and
forward the page's port instead. The browser still sees `localhost`, which keeps the
microphone available (see the notes):

```bash
# on the server host
python examples/web/server.py
# from the machine with the browser
ssh -N -L 8000:127.0.0.1:8000 user@asr-host
```

## Options

| Flag | Purpose |
|---|---|
| `--oasr-server URL` | Where to relay the API paths. Defaults to `$OASR_SERVER_URL`, then `http://127.0.0.1:8080` |
| `--host`, `--port` | Where the page is served (default `127.0.0.1:8000`) |
| `--ssl-certfile`, `--ssl-keyfile` | Serve the page over HTTPS — required for microphone access from anywhere but localhost |
| `--log-level debug` | Adds static-asset requests, upstream response headers and relay teardown reasons |
| `--log-file PATH` | Append the log to a file as well as stderr |

`/v1/*`, `/healthz`, `/readyz` and `/metrics` are relayed; everything else is served from
`static/`.

## Sharing the page with other people

`--host 0.0.0.0` makes the page reachable from other machines, with two consequences worth
being deliberate about:

- The microphone stops working unless you also pass `--ssl-certfile` — browsers expose
  `getUserMedia` only in a secure context. TLS here is independent of the upstream: the page
  can be HTTPS while the relay speaks plain HTTP to `oasr-server` on a private network.
- Anyone who can open the page can spend the GPU behind it. This is a demo front door: one
  upstream, no authentication, no rate limiting. For anything beyond a demo, put a real
  reverse proxy (nginx, Caddy) in front and let it own TLS, auth and access logs.

## Talking to `oasr-server` directly instead

The relay is a convenience, not a requirement. `?server=` points the page straight at a
server, which then *is* a cross-origin call and needs CORS opened for the page's origin.
Origins are matched exactly, so pass every spelling you might type in the address bar:

```bash
oasr-server --ckpt-dir /path/to/ckpt --service-mode streaming \
            --cors-allow-origin 'http://localhost:8000' \
            --cors-allow-origin 'http://127.0.0.1:8000'
python -m http.server 8000 --directory examples/web/static
# open http://localhost:8000/?server=http://asr-host:8080
```

## Notes

- **A byte relay, not a protocol bridge.** `server.py` forwards the WebSocket handshake and
  then splices the connection without parsing a frame, so the page speaks the real OASR API
  and a new realtime event type needs no change here. An earlier version of this demo
  translated WebSocket ⇆ gRPC and needed fastapi, uvicorn and grpcio; do not reintroduce a
  translator. The cost of not parsing is that the relay's logs count bytes rather than
  partials — the server's own `rid=` lines carry the decode detail.
- **Microphone needs a secure context** — HTTPS, or `localhost` / `127.0.0.1`. On a bare
  hostname or LAN address over plain HTTP the browser hides `getUserMedia` entirely and only
  file upload works.
- Every response carries an `x-oasr-trace` header matching the `[http-…]` / `[ws-…]` prefix in
  the log, so a devtools entry joins a log line.
- Streaming + upload paces the file through at ~realtime, so partials stream in as they would
  live. Microphone audio is captured as 16 kHz mono `LINEAR32F`, which is what the session
  declares — roughly 0.5 Mbit/s, so a distant `oasr-server` costs latency, not bandwidth.
- Interim events carry both `delta` (the increment) and `text` (the transcript so far); the
  page renders `text`, so a revised partial replaces rather than appends.
- Other clients for the same API: `oasr transcribe file.mp3`,
  `examples/recognize/http_client.py`, `examples/recognize/grpc_client.py`. In-process, with
  no server at all: `examples/recognize/local_engine.py`.
