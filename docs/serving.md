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
| HTTP | `POST /v1/speech:recognize` | Synchronous unary recognition (offline mode). Raw PCM body, config in the query string (no base64/JSON). |
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

Query parameters: `encoding` (required), `sample_rate` (default 16000; ignored
for `WAV`), `priority`, `max_alternatives`, plus the per-request decoding
options (autoregressive decode families only — AED / speech-LLM; CTC ignores
them): `max_new_tokens`, `temperature` (0 = greedy), `top_k`, `top_p`,
`prompt` (speech-LLM user-prompt override).  `max_alternatives > 1` makes the
engine detokenize that many n-best hypotheses (beam decode families), so
every returned alternative carries a real transcript.

`encoding` accepted values:

| Value | Meaning |
|---|---|
| `LINEAR16` | Little-endian 16-bit signed PCM, mono. |
| `LINEAR32F` | Little-endian 32-bit float PCM, mono *(OASR extension)*. |
| `WAV` | RIFF/WAV container in the body; embedded sample rate wins over `sample_rate` *(OASR extension)*. |
| any other Google STT v1 codec | Returns `UNIMPLEMENTED`. |

> For the lowest-overhead binary transport, gRPC `Recognize` is still preferred.

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
        "capabilities": ["ctc", "ctc_aed_rescoring"]
      }
    }
  ]
}
```

Exactly one entry — the single model loaded by this process.

`service_mode` / `decode_method` / `capabilities` are read back **from the
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
`temperature`, `top_k`, `top_p`, `prompt` — per-request `DecodingOptions`
for the AR decode families) are OASR extensions in the reserved
field-number range.  `max_alternatives` is honored (n-best transcripts on
beam decode families), `confidence` carries the softmax-normalized n-best
posterior, and `result_end_time` is set when the decode family produces
token alignments (Paraformer CIF).

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

### Processes per GPU — mode-dependent (measured, RTX 5090/SM120)

The right number of `oasr-server` processes **per GPU** differs by service mode:

| Mode | Recommended | Why |
|---|---|---|
| **offline** | **1 / GPU** | The batched forward saturates the GPU. Two engines on one GPU thrash on CUDA-graph capture + memory and **regress ~17×** (measured 114 vs 1911 utts/s aggregate). Scale offline only **across** GPUs. |
| **streaming** | **2–3 / GPU** | Chunk-by-chunk decoding is launch/CPU-bound and leaves the GPU under-utilised. A second process (a second GIL) interleaves kernel launches and fills the idle: **+34% aggregate at 2/GPU** (257→346 utts/s). |

For streaming, enabling **CUDA MPS** (`nvidia-cuda-mps-control -d`) lets the
processes' kernels run concurrently rather than time-slice, improving the
multiplier further. Each process is independent — front them with the same
load balancer.

### Streaming throughput knobs: `--max-batch-size`, `--chunk-size`, processes/GPU

Three composable levers, in order of simplicity (all measured on the reference box):

1. **`--max-batch-size` 64→256: +21%** (262→316 utts/s). Batches more streams into each
   encoder/fbank/CTC launch, amortising the launch overhead. Simplest (one knob), but
   **diminishing returns** (64→128 is only +4%) and each step gets ~3–4× longer, so it
   **raises per-stream chunk latency** — use it for throughput-biased / batch streaming, not
   interactive. Needs `--max-num-blocks` ≥ `max_batch_size × blocks_per_seq`.
2. **`--chunk-size` 16→32: +24%** (299→372 utts/s). Twice the audio per step → half the steps.
   Same latency tradeoff (keep `16` for interactive first-token latency).
3. **2–3 processes/GPU: +34%** (see table above). The only lever that **also preserves
   per-stream latency** (each process keeps a modest batch) — it adds a second Python GIL,
   breaking the single-GIL launch-issuing ceiling that batch/chunk alone cannot.

They stack: throughput-biased → big batch + chunk32 (+ extra processes for the GIL-bound
remainder); interactive/low-latency → keep batch+chunk modest and scale with processes.

### Tuning knobs (exposed on `oasr-server`)

Beyond `--max-batch-size` / `--chunk-size` / `--preferred-batch-sizes` /
`--schedule-policy` / `--max-offline-pad-ratio`, the server now forwards the full
`EngineConfig` tuning surface: `--max-batch-frames`, `--length-bucket-ratio`,
`--max-wait-time`, `--streaming-cohort-admit`, `--partial-decode-interval`,
`--overlap-partial-readback`, `--enable-sequence-packing` / `--max-packed-frames`,
and the (default-off, **keep off** — measured to regress) `--use-ctc-cuda-graphs`
/ `--use-feature-cuda-graphs`. `oasr-server --help` lists them all.

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
KV). So requests that arrive together are much cheaper than the same requests
arriving apart. Measured on `Qwen2-Audio-7B-Instruct`, 4 utterances / 124 tokens:

| arrival | window | total | tokens/s |
|---|---|---|---|
| together | — | 922 ms | 134.5 |
| one per tick | 0 (default) | 1588 ms | 78.1 |
| one per tick | 200 ms | 982 ms | 126.3 |

The window holds a thin waiting queue until it reaches `max_batch_size` or expires,
recovering ~92% of the loss. It costs up to one window of first-token latency for an
*isolated* request, so it is **off by default** — turn it on for
throughput-oriented deployments, leave it off when time-to-first-token dominates.

**`--max-tick-ms` is the knob that bounds latency, not `--decode-steps-per-tick`.**
A step count bounds work, not time, and step cost is model-dependent, so one
fixed step budget behaves very differently per model. Measured at
`--decode-steps-per-tick 32`, `B=4`, on `Qwen2-Audio-7B-Instruct`:

| `--max-tick-ms` | tick p50 | tick p99 | tokens/s |
|---|---|---|---|
| `0` (step cap only) | 173 ms | 579 ms | 135.3 |
| `25` (default) | 37 ms | 151 ms | 134.8 |

Since the dispatcher holds the GIL for a whole tick, the p99 column is the floor
on cancel latency, admission latency, and the interval between streaming
partials — cut 3.8× here for a 0.3% throughput cost (within run-to-run noise).
The residual 151 ms p99 is the **prefill** tick (audio tower + projector + one LM
forward over the whole prompt), which the decode deadline deliberately does not
bound; a tick that spends its decode budget will not also prefill, so the two
never stack.

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
| `oasr_events_{dropped,deferred}_total` | counter | Per-request channel was full: a partial was dropped (harmless — the next one supersedes it) or a **terminal** event was handed to a background task rather than lost. Sustained non-zero here means clients are reading slower than the engine emits. |

`--trace-dispatch` additionally logs rolling 2 s means of the same sub-stages at
INFO; the histograms above are the ones to alert on.

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

- AuthN/AuthZ — rely on a network policy or reverse proxy.
- Cross-host engine fleets — single-host only at the binary level; clusters
  go through your LB.
- Audio codecs beyond PCM/WAV — MP3/Opus/FLAC follow up behind a `symphonia`
  feature on `oasr-asr`.
- TLS termination — assume a reverse proxy handles it.
- `LongRunningRecognize` (Google STT v1 LRO) — not implemented.
- `RecognitionConfig.language_code`, `model`, `audio_channel_count`, and
  `StreamingRecognitionConfig.single_utterance` — accepted, ignored.
