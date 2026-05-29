<p align="center">
  <picture>
    <img alt="OASR" src="https://raw.githubusercontent.com/chiendb97/oasr/main/docs/assets/logos/oasr-logo-text.png" width=55%>
  </picture>
</p>

<h3 align="center">
Open Automatic Speech Recognition — a high-performance inference framework for ASR models.
</h3>

---

OASR delivers fast, scalable ASR inference on NVIDIA GPUs. It pairs custom CUDA kernels with a dynamic-batching inference engine and a production-ready serving frontend, so you can transcribe audio offline or stream it in real time over HTTP or gRPC.

---

## Key Features

- **High throughput** — paged KV cache, dynamic length-bucketed batching, and CUDA Graph capture of the streaming encoder.
- **Streaming and offline** — one engine handles both, with mixed pools in a single step loop.
- **Production serving** — Rust `oasr-server` binary exposing an HTTP API and a gRPC bidi streaming RPC.
- **Multiple decoders** — CTC greedy, CTC prefix beam (CPU & GPU), and WFST beam search.
- **Broad GPU support** — Volta through Blackwell (SM70–SM120), with FP16 / BF16 / FP32 paths.
- **Drop-in Python API** — functional and `nn.Module` wrappers for normalization, convolution, GEMM, attention, and feature extraction.

## Supported Models

| Model        | Status      |
|--------------|-------------|
| Conformer    | ✅ Available |
| Paraformer   | 🔲 Planned  |
| Branchformer | 🔲 Planned  |
| Zipformer    | 🔲 Planned  |
| Transducer   | 🔲 Planned  |

---

## Getting Started

### Requirements

- CUDA ≥ 11.8
- Python ≥ 3.8
- CMake ≥ 3.18
- NVIDIA GPU (SM70 or newer)

### Install

```bash
# Editable install — kernels are JIT-compiled on first use
pip install -e .

# Target specific GPU architectures
CUDA_ARCHITECTURES="80;86;90" pip install -e .
```

Optional extras:

```bash
pip install -e ".[audio]"    # soundfile, librosa
pip install -e ".[serving]"  # serving client libs (used by bench_service.py)
pip install -e ".[wfst]"     # k2, kaldilm (WFST decoder)
```

Build the Rust serving binary (linked against the active Python interpreter):

```bash
cd rust && cargo build --release
# Requires: protobuf-compiler, a C/C++ toolchain
```

---

## Quick Start

### Offline transcription

```python
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))

# Single file
text = engine.transcribe_offline("audio.wav")

# Batch — dynamic length-bucketed micro-batches
texts = engine.transcribe_offline(["a.wav", "b.wav", "c.wav"])
```

### Streaming transcription

```python
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))

# Attached-audio streaming — chunk-by-chunk decode, paged KV cache
texts = engine.transcribe(["a.wav", "b.wav", "c.wav"], streaming=True)

# Real-time feed loop
rid = engine.add_streaming_request()
for chunk in mic_chunks():
    engine.feed_chunk(rid, chunk, is_last=False)
    for out in engine.step():
        if not out.finished:
            print("partial:", out.text)
engine.feed_chunk(rid, last_chunk, is_last=True)
final = engine.run()
```

The checkpoint directory should contain `final.pt`, `train.yaml`, `global_cmvn`, and optionally a tokenizer `.model` file plus `units.txt`.

---

## Serving

`oasr-server` runs one in-process `ASREngine` per process. `--service-mode` pins the engine to either `offline` (sync `Recognize`) or `streaming` (bidi `StreamingRecognize`) for its entire lifetime; the mismatched RPC returns `FAILED_PRECONDITION`. Scale horizontally by launching one process per GPU.

```bash
# Offline server on GPU 0
./rust/target/release/oasr-server \
    --ckpt-dir /path/to/checkpoint \
    --service-mode offline \
    --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051

# Multi-GPU fleet — one process per device, fronted by your load balancer
CUDA_VISIBLE_DEVICES=0 ./rust/target/release/oasr-server \
    --ckpt-dir /path/to/checkpoint \
    --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051 &
CUDA_VISIBLE_DEVICES=1 ./rust/target/release/oasr-server \
    --ckpt-dir /path/to/checkpoint \
    --http-bind 127.0.0.1:8081 --grpc-bind 127.0.0.1:50052 &
```

### Endpoints

| Surface | Endpoint                                       | Purpose                                  |
|---------|------------------------------------------------|------------------------------------------|
| HTTP    | `POST /v1/speech:recognize`                    | Synchronous unary recognition (offline). |
| HTTP    | `GET /v1/models`                               | Loaded model metadata.                   |
| HTTP    | `GET /healthz` / `/readyz` / `/metrics`        | Liveness, readiness, Prometheus metrics. |
| gRPC    | `oasr.speech.v1.Speech/Recognize`              | Synchronous unary (offline mode).        |
| gRPC    | `oasr.speech.v1.Speech/StreamingRecognize`     | Bidi streaming (streaming mode).         |
| gRPC    | `grpc.health.v1.Health/Check` and `Watch`      | Standard gRPC health checking.           |

REST is synchronous only — streaming clients must use the gRPC `StreamingRecognize` RPC.

### HTTP example

Audio is carried inline as base64 in `audio.content`.

```bash
B64=$(base64 -w0 audio.wav)
curl -sS -X POST http://127.0.0.1:8080/v1/speech:recognize \
     -H 'Content-Type: application/json' \
     -d "$(jq -n --arg b64 "$B64" \
           '{config:{encoding:"WAV",sampleRateHertz:16000,languageCode:"en-US"},
             audio:{content:$b64}}')"
```

Accepted `encoding` values: `LINEAR16` (16-bit PCM mono), `LINEAR32F` (32-bit float PCM mono), and `WAV`. Other codecs return `UNIMPLEMENTED`.

### gRPC streaming

The first inbound message on `StreamingRecognize` must carry `streaming_config.config`; subsequent messages carry `audio_content` (raw PCM bytes). Each response contains a `StreamingRecognitionResult` with `is_final=true` on the terminal frame.

```bash
grpcurl -plaintext -import-path rust/proto -proto oasr_speech_v1.proto \
        127.0.0.1:50051 oasr.speech.v1.Speech/StreamingRecognize
```

---

## Documentation

| Document                                             | Covers                                                |
|------------------------------------------------------|-------------------------------------------------------|
| [`docs/engine.md`](docs/engine.md)                   | Engine step loop, batching, CUDA Graph capture        |
| [`docs/engine_concurrency.md`](docs/engine_concurrency.md) | Engine thread-safety and multi-process scaling   |
| [`docs/scheduler.md`](docs/scheduler.md)             | Request scheduling, starvation bounds, micro-batching |
| [`docs/cache_manager.md`](docs/cache_manager.md)     | Paged streaming cache (`BlockPool`, `StreamContext`)  |
| [`docs/ctc_decoder_gpu.md`](docs/ctc_decoder_gpu.md) | GPU CTC decoder, single- and multi-request flows      |
| [`docs/serving.md`](docs/serving.md)                 | Serving frontend, wire format, deployment             |
| [`docs/benchmarks.md`](docs/benchmarks.md)           | Engine and service benchmark recipes                  |
| [`docs/autotuning.md`](docs/autotuning.md)           | Kernel auto-tuning API and cache format               |

---

## Testing

```bash
# Run all Python tests
pytest tests/

# Skip slow tests
pytest tests/ -m "not slow"
```

---

## Contributing

Contributions are welcome. Please open an issue to discuss substantial changes before submitting a pull request, and run `black`, `ruff`, and `pytest` against your changes.

## License

Apache 2.0
