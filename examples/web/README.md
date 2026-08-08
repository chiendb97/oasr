# OASR Web Demo

A browser demo for OASR: upload a file or record from the microphone, pick **offline** or
**streaming** mode, and see the transcript.

```
Browser ──HTTP / WebSocket──▶ oasr-server
```

There is no bridge process. The page talks to `oasr-server` directly:

| Mode | Call |
|---|---|
| Offline | `POST /v1/audio/transcriptions` — the file is uploaded **as-is**; the server decodes MP3 / M4A / FLAC / OGG / WAV |
| Streaming | `WS /v1/realtime` — a `session.update` frame, then binary PCM chunks, then `input_audio_buffer.commit` |

## Run it

```bash
# 1. An OASR server, with this page's origin allowed
oasr-server --ckpt-dir /path/to/ckpt --service-mode streaming \
            --http-bind 127.0.0.1:8080 --cors-allow-origin '*'

# 2. Any static file server for the page
python -m http.server 8000 --directory examples/web/static

# 3. Open http://localhost:8000
```

Both modes work against either `--service-mode`. A `streaming` server decodes as the audio
arrives; an `offline` one buffers the utterance and streams the *text* back, so a speech-LLM
still fills in token by token.

Point the page at another server with `?server=`:
`http://localhost:8000/?server=http://gpu-box:8080`.

## Notes

- **`--cors-allow-origin` is required** and off by default. Whether an inference endpoint
  should be callable from any page is an operator's decision, so it is never a default.
  `'*'` is fine for a local demo; name the real origin in production.
- **Microphone needs a secure context** — HTTPS, or `localhost` / `127.0.0.1`. On a bare LAN
  address over plain HTTP the browser hides `getUserMedia` entirely. File upload works
  everywhere. For a remote machine, forward the port
  (`ssh -L 8000:localhost:8000 -L 8080:localhost:8080 <user>@<host>`).
- Streaming + upload paces the file through at ~realtime, so partials stream in as they would
  live. Mic audio is captured as 16 kHz mono `LINEAR32F`, which is what the session declares.
- Interim events carry both `delta` (the increment) and `text` (the transcript so far); the
  page renders `text`, so a revised partial replaces rather than appends.
- CLI equivalents: `oasr transcribe file.mp3`, `examples/recognize/http_client.py`,
  `examples/recognize/grpc_client.py`. In-process (no server):
  `examples/recognize/local_engine.py`.
