# Frequently Asked Questions

Short answers to the questions people ask before adopting OASR. Each answer
links to the page that covers it in depth.

- [What is OASR?](#what-is-oasr)
- [Which speech recognition models can OASR serve?](#which-speech-recognition-models-can-oasr-serve)
- [Can OASR replace the OpenAI transcription API?](#can-oasr-replace-the-openai-transcription-api)
- [Does OASR support real-time streaming recognition?](#does-oasr-support-real-time-streaming-recognition)
- [Which GPUs does OASR support?](#which-gpus-does-oasr-support)
- [Does OASR run on CPU, Apple Silicon or AMD GPUs?](#does-oasr-run-on-cpu-apple-silicon-or-amd-gpus)
- [How do I install OASR?](#how-do-i-install-oasr)
- [Do I need to export my model to ONNX or TensorRT?](#do-i-need-to-export-my-model-to-onnx-or-tensorrt)
- [How do I load a WeNet, icefall, FunASR or Hugging Face checkpoint?](#how-do-i-load-a-wenet-icefall-funasr-or-hugging-face-checkpoint)
- [How is OASR different from WeNet, icefall, FunASR or NeMo?](#how-is-oasr-different-from-wenet-icefall-funasr-or-nemo)
- [When is another tool a better fit?](#when-is-another-tool-a-better-fit)
- [Does OASR return word-level timestamps?](#does-oasr-return-word-level-timestamps)
- [Does OASR include voice activity detection (VAD)?](#does-oasr-include-voice-activity-detection-vad)
- [Which languages does OASR support?](#which-languages-does-oasr-support)
- [Does OASR support quantization?](#does-oasr-support-quantization)
- [How fast is OASR?](#how-fast-is-oasr)
- [How do I scale OASR across several GPUs?](#how-do-i-scale-oasr-across-several-gpus)
- [Can I add my own model architecture?](#can-i-add-my-own-model-architecture)
- [Is OASR secure to expose on a network?](#is-oasr-secure-to-expose-on-a-network)
- [What license is OASR released under?](#what-license-is-oasr-released-under)

## What is OASR?

OASR (Open Automatic Speech Recognition) is an open-source inference and
serving engine for speech recognition on NVIDIA GPUs. It loads trained ASR
checkpoints, such as Whisper, WeNet Conformer, icefall Zipformer, Paraformer
and Qwen2-Audio, and serves them for offline (file) and streaming (real-time)
transcription. One engine, with custom CUDA/CUTLASS kernels, dynamic batching
and CUDA-graph capture, sits behind a Python API, an `oasr` command-line tool
and a Rust HTTP/gRPC server with an OpenAI-compatible API. OASR does not train
models.

## Which speech recognition models can OASR serve?

OASR serves seven architectures across five decoding paradigms:

| Architecture | Decoding paradigm | Streaming | Checkpoint source |
|---|---|---|---|
| Conformer (WeNet U2 / U2++) | CTC, with optional attention rescoring | yes | WeNet |
| Zipformer | CTC | yes | icefall (k2) |
| Zipformer transducer | RNN-T (greedy and beam search) | yes | icefall (k2) |
| Nemotron ASR (FastConformer) | RNN-T | yes | Hugging Face |
| Whisper | attention encoder-decoder (greedy and beam search) | no | Hugging Face |
| Paraformer | non-autoregressive (CIF) | no | FunASR |
| Qwen2-Audio | speech-LLM | no | Hugging Face |

The test suite runs against these released checkpoints: `openai/whisper-tiny`,
`nvidia/nemotron-3.5-asr-streaming-0.6b`, `Qwen/Qwen2-Audio-7B-Instruct`,
`funasr/paraformer-zh`, the icefall LibriSpeech Zipformer CTC and
pruned-transducer releases (offline and streaming), and WeNet's LibriSpeech
U2++ Conformer. Other checkpoints of the same architectures load the same way.
See [models.md](models.md).

## Can OASR replace the OpenAI transcription API?

Yes, for clients of `POST /v1/audio/transcriptions` and
`POST /v1/audio/translations`. Point an existing OpenAI client at
`oasr-server` by changing its base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
text = client.audio.transcriptions.create(model="whisper-1", file=open("a.mp3", "rb")).text
```

Uploads can be MP3, M4A/AAC, FLAC, OGG Vorbis, WAV or AIFF at any sample rate.
Opus audio (in Ogg or WebM), the usual codec of browser recordings, is not
decoded yet.
The `json`, `text`, `srt`, `vtt` and `verbose_json` response formats, word
timestamps and `stream=true` (server-sent events) are supported. The `model`
field is accepted and ignored unless `--served-model-name` is set, because one
process serves one model. Translation needs a model that can translate, such
as Whisper. There is also a realtime WebSocket at `/v1/realtime`. See
[serving.md](serving.md).

## Does OASR support real-time streaming recognition?

Yes. Conformer, Zipformer, the Zipformer transducer and Nemotron stream
chunk by chunk, using a paged or fixed-slot cache, and concurrent streams are
batched dynamically. An engine runs in one mode for its lifetime
(`service_mode="streaming"` or `"offline"`). Clients connect over the
`/v1/realtime` WebSocket, server-sent events, or gRPC bidirectional streaming
(`oasr.speech.v1.Speech/StreamingRecognize`). From Python, use
`ASREngine.add_streaming_request`, `feed_chunk` and `step`. See
[engine.md](engine.md) and [cache_manager.md](cache_manager.md).

## Which GPUs does OASR support?

NVIDIA GPUs of compute capability 7.5 or newer: Turing (SM75) through
Blackwell (SM100 and SM120), with fp16, bf16 and fp32 paths. CUDA kernels are
compiled for the GPU on first use, so no per-GPU build step is needed. For
quantized inference, int8 W8A8 has CUTLASS kernels on SM80 and newer, and fp8
W8A8 on SM89 and newer (Ada, Hopper, Blackwell). See
[kernels.md](kernels.md) and [quantization.md](quantization.md#gpu-support).

## Does OASR run on CPU, Apple Silicon or AMD GPUs?

No, OASR targets NVIDIA CUDA GPUs. Each layer also has a PyTorch reference
path, which the tests use as a correctness oracle, but the engine is not built
or tuned to serve from a CPU. For CPU, mobile or edge deployment, see
[When is another tool a better fit?](#when-is-another-tool-a-better-fit)

## How do I install OASR?

From source, with the `cutlass` submodule:

```bash
git clone https://github.com/chiendb97/oasr.git && cd oasr
git submodule update --init
pip install -e ".[all]"
```

The package is named `oasr-server` and imports as `oasr`. The `oasr` package on
PyPI is an unrelated project: `pip install oasr` does not install this engine,
and it replaces the `oasr` command. The CUDA kernels are compiled on first use, and the
Rust server is built during `pip install`. See the
[README](../README.md#install) for the optional extras and a Docker image.

## Do I need to export my model to ONNX or TensorRT?

No. OASR reads PyTorch and safetensors checkpoints directly and runs them with
its own kernels. These are CUDA C++ and CUTLASS kernels that are JIT-compiled
through TVM-FFI on first call and cached in `~/.cache/oasr/jit/`. There is no
export or engine-build step. See [kernels.md](kernels.md).

## How do I load a WeNet, icefall, FunASR or Hugging Face checkpoint?

Pass the checkpoint directory, or a Hugging Face Hub repo id, as `ckpt_dir`.
The format is detected automatically, and the tokenizer, feature frontend and
decoding defaults are read from the checkpoint:

```python
from oasr.engine import ASREngine, EngineConfig

engine = ASREngine(EngineConfig(ckpt_dir="openai/whisper-tiny", service_mode="offline"))
print(engine.transcribe_offline("audio.wav"))
```

`oasr-convert <src> <dst>` writes any supported checkpoint in OASR's native
format (safetensors plus tokenizer assets). An icefall transducer directory is
detected as Zipformer CTC, so load it with `architecture="transducer"`. See
[checkpoints.md](checkpoints.md).

## How is OASR different from WeNet, icefall, FunASR or NeMo?

Those are toolkits for training speech recognition models. OASR is
inference-only. It loads the checkpoints they release, as they are, into one
GPU engine with one API, so models trained with different toolkits can be
served by the same server, scheduler and kernels.

## When is another tool a better fit?

OASR is built for serving on NVIDIA GPUs. Another tool fits better when:

- **You need CPU, mobile, embedded or browser inference.** Look at runtimes
  built for those targets, such as
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) or
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
- **You want to train or fine-tune a model.** Use a training toolkit such as
  [WeNet](https://github.com/wenet-e2e/wenet),
  [icefall](https://github.com/k2-fsa/icefall),
  [FunASR](https://github.com/modelscope/FunASR) or
  [NeMo](https://github.com/NVIDIA/NeMo), then serve the result with OASR.
- **Your architecture is not one of the seven above.** It needs a model
  implementation first. See
  [Can I add my own model architecture?](#can-i-add-my-own-model-architecture)

## Does OASR return word-level timestamps?

Yes, for CTC, transducer, attention encoder-decoder and CIF (Paraformer)
models. Each family derives timings from the model's own alignment, and a
family that cannot align refuses the request rather than estimating. Use
`DecodingOptions(word_timestamps=True)` in Python, or
`--response-format verbose_json --timestamp-granularity word` with the CLI. See
[decoding.md](decoding.md).

## Does OASR include voice activity detection (VAD)?

Yes. VAD segments long audio at speech boundaries offline, and ends turns on
silence when streaming. It can use a separate VAD model (Silero VAD v5,
rebuilt on OASR's own layers) or, with no VAD model configured, the ASR
model's own per-frame output. See [vad.md](vad.md).

## Which languages does OASR support?

The checkpoint decides. OASR adds no language limit of its own. Whisper and
Qwen2-Audio checkpoints are multilingual, and Whisper accepts a `language` per
request. Paraformer `paraformer-zh` is Mandarin Chinese, and the icefall and
WeNet LibriSpeech releases are English.

## Does OASR support quantization?

Yes. OASR supports W8A8 int8 and fp8 weights and activations, either online
with no calibration (`--quantization int8_w8a8`) or calibrated with
SmoothQuant through `oasr quantize`, which writes a quantized checkpoint. It
also supports an fp8 KV cache for autoregressive decoders. See
[quantization.md](quantization.md).

## How fast is OASR?

It depends on the model, GPU, batch size, audio length and whether requests
stream, so OASR does not quote a single number. Measure it on your own
hardware with the bundled benchmark suite, which reports latency percentiles,
throughput (RTFx) and word error rate together:

```bash
python -m benchmarks.asrbench run --profile smoke --backend oasr_engine
```

See [asrbench.md](asrbench.md) and [benchmarks.md](benchmarks.md).

## How do I scale OASR across several GPUs?

Run one `oasr-server` process per GPU and put a load balancer in front. Within
a process, the engine batches concurrent requests dynamically. It serves its
own Prometheus `/metrics` endpoint, and the repository includes Grafana
dashboards. See [serving.md](serving.md).

## Can I add my own model architecture?

Yes, without changing the engine core. Every extension point is a registry:
model architectures, checkpoint formats, decoding strategies, streaming
runtimes, batching policies, tokenizers, feature frontends, speech detectors
and quantization schemes. Subclass the base, register it under a name, and
select it by configuration. Models are built from `oasr.layers`, so they use
OASR's kernels automatically. See the extension cookbook in
[architecture.md](architecture.md#extension-cookbook).

## Is OASR secure to expose on a network?

Not on its own. `oasr-server` performs no authentication and does not
terminate TLS. Run it on a private network or behind an authenticating,
TLS-terminating reverse proxy, and set the request limits
(`--max-audio-seconds`, `--max-audio-mib`, `--max-concurrent-decodes`). See
[SECURITY.md](../SECURITY.md).

## What license is OASR released under?

The [Apache License 2.0](../LICENSE), which permits commercial use. Model
checkpoints keep their own licenses, so check the license of each checkpoint
you serve.
