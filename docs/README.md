# OASR Documentation

Stable technical documentation for OASR: architecture, components, algorithms,
interfaces and operation.

- Contributor and coding-agent instructions live in [`CLAUDE.md`](../CLAUDE.md).
- Point-in-time material — benchmark results, active issues, investigations —
  lives in [`.artifacts/`](../.artifacts/README.md) and is deliberately kept out
  of these pages.

## Getting started

| Document | Covers |
|---|---|
| [`../README.md`](../README.md) | Project overview, install, quick start, supported models |
| [checkpoints.md](checkpoints.md) | Loading a checkpoint: resolution precedence, converter contract, `oasr-convert`, the native format |
| [benchmarks.md](benchmarks.md) | Engine, service and accuracy benchmark recipes; measurement protocol |
| [ci.md](ci.md) | The four workflows, running the gates locally, the asset gate, the accuracy gate |

## Architecture and design

| Document | Covers |
|---|---|
| [architecture.md](architecture.md) | The seven registry seams, the `oasr.layers` narrow waist, data flow, and the extension cookbook — **start here** |
| [engine.md](engine.md) | `ASREngine`: step loop, executors, CUDA-graph capture, `EngineConfig` |
| [scheduler.md](scheduler.md) | Request scheduling, batching and partition policies, starvation bounds |
| [models.md](models.md) | The model layer: base contracts, capabilities, registry, and the seven built-in architectures |
| [decoding.md](decoding.md) | Decode families: selection, the incremental AR protocol, beam search, per-family and per-request options |
| [kernels.md](kernels.md) | The CUDA/CUTLASS layer, the TVM-FFI JIT pipeline, and the Python functional API |

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
fix belongs in `.artifacts/` — see its
[house rules](../.artifacts/README.md#house-rules). Leave a
one-line pointer behind rather than the numbers themselves.
