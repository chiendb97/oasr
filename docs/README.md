# OASR Documentation

OASR is an open-source inference and serving engine for speech recognition on
NVIDIA GPUs. These pages are its stable technical documentation: architecture,
components, algorithms, interfaces and operation. For installation and a quick
start, see the [project README](../README.md).

- Contributor and coding-agent instructions live in [`AGENTS.md`](../AGENTS.md)
  (`CLAUDE.md` includes it); the contribution workflow is in
  [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- Common questions — what OASR is for, which GPUs and models it supports, how it
  relates to other ASR toolkits — are answered in the [FAQ](faq.md).
- Point-in-time material — benchmark results, active issues, investigations —
  lives in a gitignored `.artifacts/` working directory and is deliberately kept
  out of these pages.

## Getting started

| Document | Covers |
|---|---|
| [`../README.md`](../README.md) | Project overview, install, quick start, supported models |
| [faq.md](faq.md) | Short answers: supported models and GPUs, the OpenAI-compatible API, streaming, word timestamps, quantization, and when another tool fits better |
| [checkpoints.md](checkpoints.md) | Loading a checkpoint: resolution precedence, converter contract, `oasr-convert`, the native format |
| [benchmarks.md](benchmarks.md) | Engine, service and accuracy benchmark recipes; measurement protocol |
| [asrbench.md](asrbench.md) | The ASR inference performance suite: datasets, workloads, latency/throughput/concurrency/streaming protocol, backends, regression gating, reports |
| [ci.md](ci.md) | The five workflows, running the gates locally, the asset gate, the accuracy gate |

## Architecture and design

| Document | Covers |
|---|---|
| [architecture.md](architecture.md) | The nine registry seams, the `oasr.layers` narrow waist, data flow, and the extension cookbook — **start here** |
| [engine.md](engine.md) | `ASREngine`: step loop, executors, CUDA-graph capture, `EngineConfig` |
| [scheduler.md](scheduler.md) | Request scheduling, batching and partition policies, starvation bounds |
| [models.md](models.md) | The model layer: base contracts, capabilities, registry, and the seven built-in architectures |
| [decoding.md](decoding.md) | Decode families: selection, the incremental AR protocol, beam search, per-family and per-request options |
| [kernels.md](kernels.md) | The CUDA/CUTLASS layer, the TVM-FFI JIT pipeline, and the Python functional API |
| [quantization.md](quantization.md) | Quantization: W8A8 int8 and fp8 at the layer waist, SmoothQuant and `oasr quantize`, the native v2 checkpoint, GPU support, the fp8 KV cache, numerics, extending to new formats |

## Components

| Document | Covers |
|---|---|
| [features.md](features.md) | Feature frontends: `FeatureSpec`, the extractor registry, streaming framing, the built-in recipes |
| [vad.md](vad.md) | Voice activity: the detector registry, the shared segmenter and endpointer, offline segmentation and streaming turn detection, and what each API surface exposes |
| [tokenizers.md](tokenizers.md) | Tokenizer axis: kinds, `TokenizerSpec`, the `tokenizer_config.json` added-token merge |
| [cache_manager.md](cache_manager.md) | Streaming caches: `BlockPool`, `AttentionCacheManager`, `SlotStateCache`, `StreamContext` |
| [ctc_decoder_gpu.md](ctc_decoder_gpu.md) | GPU CTC prefix beam search: algorithm, kernel pipeline, paged mode, streaming lifecycle |
| [wfst_decoder_gpu.md](wfst_decoder_gpu.md) | In-tree GPU WFST beam search: k2-parity semantics, graph image, lazy-commit memory model |
| [autotuning.md](autotuning.md) | `oasr.tune`: the `autotune()` API and the JSON cache format |

## Operations

| Document | Covers |
|---|---|
| [serving.md](serving.md) | `oasr-server`: the Rust workspace, HTTP + gRPC APIs, dispatcher, limits, metrics, deployment topology |

## Writing docs

Keep these pages **stable**. A page here should describe architecture, features,
components, algorithms, interfaces, usage, configuration concepts, and design
decisions that stay true across releases.

Anything that would need editing after a re-benchmark, a driver upgrade or a bug
fix belongs in the gitignored `.artifacts/` working directory (its own `README.md`
holds the house rules). Leave a one-line, plain-text pointer behind rather than
the numbers themselves — not a link: a fresh clone does not have the directory.
