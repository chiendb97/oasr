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
| `--tls-self-signed` | Serve HTTPS with a cached self-signed certificate, generated on first use — the one-flag way to get the microphone working off localhost |
| `--tls-san NAME` | Extra hostname or IP for that certificate to cover; repeatable |
| `--ssl-certfile`, `--ssl-keyfile` | Serve HTTPS with your own certificate instead |
| `--log-level debug` | Adds static-asset requests, upstream response headers and relay teardown reasons |
| `--log-file PATH` | Append the log to a file as well as stderr |
