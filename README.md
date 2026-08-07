<p align="center">
  <picture>
    <img alt="OASR" src="https://raw.githubusercontent.com/chiendb97/oasr/main/docs/assets/logos/oasr-logo-text.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap ASR serving for everyone
</h3>

---

OASR is a fast and easy-to-use framework for the inference and serving of automatic speech recognition (ASR) models. It is designed to deliver low-latency, high-throughput inference.

---

## Key Features

OASR is fast with:

- Custom CUDA / CUTLASS kernels for GEMM, attention, normalization, convolution, feature extraction, and decoding
- Paged KV cache for streaming attention
- Dynamic batching of offline and streaming requests, with length-bucketing and sequence packing for offline
- CUDA Graph capture of the steady-state streaming encoder
- FP16 / BF16 / FP32 paths across Volta through Blackwell (SM70–SM120)

OASR is flexible and easy to use with:

- A single engine for both offline and streaming inference
- Seven architectures across five decode families — CTC, RNN-T, AED, NAR (CIF), and speech-LLM
- Checkpoints load directly from Hugging Face, WeNet, icefall, and FunASR — no conversion step
- Decoders: CTC greedy and prefix beam, GPU WFST beam search, transducer and AED greedy/beam, CTC+AED rescoring
- A production Rust frontend with HTTP and gRPC APIs

## Supported Models

| Architecture | Registry key   | Decode family                | Offline | Streaming | Source format |
|--------------|----------------|------------------------------|---------|-----------|---------------|
| Conformer (U2/U2++) | `conformer` | CTC (beam / WFST) + CTC/AED rescoring | ✅ | ✅ | WeNet |
| Zipformer    | `zipformer`    | CTC (beam / WFST)            | ✅ | ✅ | icefall |
| Transducer (RNN-T) | `transducer` | Transducer (greedy + beam) | ✅ | ✅ | icefall |
| Nemotron ASR (FastConformer + RNN-T) | `nemotron` | Transducer | ✅ | ✅ | Hugging Face |
| Whisper      | `whisper`      | AED (greedy + beam)          | ✅ | — | Hugging Face |
| Paraformer   | `paraformer`   | NAR / CIF (with timestamps)  | ✅ | — | FunASR |
| Qwen2-Audio (speech-LLM) | `speech_llm` | LLM (token-streaming partials) | ✅ | — | Hugging Face |

See [`docs/architecture.md`](docs/architecture.md) for the decode-family matrix and how to register
an architecture of your own.

---

## Getting Started

### Requirements

- CUDA ≥ 11.8
- Python ≥ 3.10
- CMake ≥ 3.18
- NVIDIA GPU (SM70 or newer)
- Rust toolchain + `protobuf-compiler`

### Install

```bash
# Editable install — kernels are JIT-compiled on first use
pip install -e .

# Target specific GPU architectures
CUDA_ARCHITECTURES="80;86;90" pip install -e .

# Optional extras
pip install -e ".[all]"         # audio + hub + tokenizers + attention + serving
pip install -e ".[audio]"       # torchaudio, soundfile, librosa, kaldifeat
pip install -e ".[hub]"         # Hub download + native-checkpoint I/O
pip install -e ".[tokenizers]"  # sentencepiece + tokenizers
pip install -e ".[attention]"   # CuTeDSL fused attention (SDPA fallback without it)
pip install -e ".[serving]"     # client libs for the benchmark scripts
pip install -e ".[wfst]"        # k2 — offline WFST graph export only, never at decode time

# Optional: standalone server binary at rust/target/release/oasr-server
cd rust && cargo build --release
```

---

## Checkpoints

WeNet, icefall, FunASR, and Hugging Face checkpoints load as-is: the format is auto-detected, and the
tokenizer, feature frontend, and decoding defaults travel with the checkpoint.

| Source format | Architectures                          |
|---------------|----------------------------------------|
| Hugging Face  | `whisper`, `speech_llm`, `nemotron`    |
| WeNet         | `conformer`                            |
| icefall       | `zipformer`, `transducer`              |
| FunASR        | `paraformer`                           |
| OASR native   | any                                    |

```python
from oasr.engine import ASREngine, EngineConfig

# A local directory in any supported format, or a Hugging Face Hub repo id
engine = ASREngine(EngineConfig(ckpt_dir="openai/whisper-tiny", service_mode="offline"))
```

`oasr-convert <src> <dst>` materializes any supported directory as a native bundle
(safetensors + tokenizer assets) that loads with no format conversion.

See [`docs/checkpoints.md`](docs/checkpoints.md) for detection rules, the converter contract, and the
native format.

---

## Quick Start

An engine is pinned to one mode for its lifetime by `EngineConfig.service_mode` — `"streaming"` (the
default) or `"offline"`.

### Offline transcription

```python
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint", service_mode="offline"))

# Single file
text = engine.transcribe_offline("audio.wav")

# Batch — dynamic length-bucketed micro-batches
texts = engine.transcribe_offline(["a.wav", "b.wav", "c.wav"])
```

### Streaming transcription

```python
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="/path/to/checkpoint"))  # service_mode="streaming" by default

# Attached-audio streaming — chunk-by-chunk decode, paged KV cache
texts = engine.transcribe(["a.wav", "b.wav", "c.wav"])

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

See [`docs/engine.md`](docs/engine.md) for the step loop, batching, and the full `EngineConfig`.

---

## Serving

`oasr-server` hosts one in-process `ASREngine` and serves it over HTTP and gRPC. Scale horizontally
by launching one process per GPU.

```bash
oasr-server \
    --ckpt-dir /path/to/checkpoint \
    --service-mode offline \              # or: streaming
    --http-bind 127.0.0.1:8080 \
    --grpc-bind 127.0.0.1:50051
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

Audio is carried inline as base64 in `audio.content` (`WAV`, `LINEAR16`, or `LINEAR32F`).

```bash
B64=$(base64 -w0 audio.wav)
curl -sS -X POST http://127.0.0.1:8080/v1/speech:recognize \
     -H 'Content-Type: application/json' \
     -d "$(jq -n --arg b64 "$B64" \
           '{config:{encoding:"WAV",sampleRateHertz:16000,languageCode:"en-US"},
             audio:{content:$b64}}')"
```

### gRPC streaming

The first inbound message carries `streaming_config.config`, the rest carry `audio_content` (raw PCM).

```bash
grpcurl -plaintext -import-path rust/proto -proto oasr_speech_v1.proto \
        127.0.0.1:50051 oasr.speech.v1.Speech/StreamingRecognize
```

See [`docs/serving.md`](docs/serving.md) for the full CLI, the wire format, per-request decoding
options, and deployment.

---

## Documentation

Start at the [documentation index](docs/README.md).

| Document                                             | Covers                                                |
|------------------------------------------------------|-------------------------------------------------------|
| [`docs/architecture.md`](docs/architecture.md)       | Engine extension points (the per-axis registries)     |
| [`docs/engine.md`](docs/engine.md)                   | Engine step loop, batching, CUDA Graph capture        |
| [`docs/scheduler.md`](docs/scheduler.md)             | Request scheduling, starvation bounds, micro-batching |
| [`docs/models.md`](docs/models.md)                   | Model contracts, capabilities, the seven architectures |
| [`docs/decoding.md`](docs/decoding.md)               | Decode families, beam search, decoding options        |
| [`docs/kernels.md`](docs/kernels.md)                 | CUDA/CUTLASS layer, the JIT pipeline, the functional API |
| [`docs/checkpoints.md`](docs/checkpoints.md)         | Checkpoint resolution, converter contract, native format |
| [`docs/features.md`](docs/features.md)               | Feature frontends, `FeatureSpec`, streaming framing   |
| [`docs/tokenizers.md`](docs/tokenizers.md)           | Tokenizer axis: kinds and `TokenizerSpec`             |
| [`docs/cache_manager.md`](docs/cache_manager.md)     | Paged streaming cache (`BlockPool`, `StreamContext`)  |
| [`docs/ctc_decoder_gpu.md`](docs/ctc_decoder_gpu.md) | GPU CTC decoder, single- and multi-request flows      |
| [`docs/wfst_decoder_gpu.md`](docs/wfst_decoder_gpu.md) | In-tree GPU WFST decoder: k2-parity semantics, kernel pipeline, streaming |
| [`docs/serving.md`](docs/serving.md)                 | Serving frontend, wire format, deployment             |
| [`docs/benchmarks.md`](docs/benchmarks.md)           | Engine, service and accuracy benchmark recipes        |
| [`docs/autotuning.md`](docs/autotuning.md)           | Kernel auto-tuning API and cache format               |

---

## Contributing

Contributions are welcome. Please open an issue to discuss substantial changes before submitting a pull request, and run `black`, `ruff`, and `pytest` against your changes.

## License

Apache 2.0
