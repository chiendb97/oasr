# Feature Frontends

Audio feature extraction runs **on the GPU, batched**, and is a registry axis like
every other extension point: a frontend is a registered `ExtractorSpec`, selected
by `FeatureConfig.feature_type`, which the engine materializes from a
converter-emitted `FeatureSpec`.

Registering a frontend is the whole change — no edit to the config, the
`InputProcessor`, the engine, or the CUDA-graph feature cache.

## Module map (`oasr/features/`)

| File | Contents |
|---|---|
| `config.py` | `FeatureConfig` — sample rate, mel bins, frame length/shift, dither, window, LFR, `feature_type` |
| `spec.py` | `FeatureSpec` — the **checkpoint-derived** description a converter emits |
| `registry.py` | `ExtractorSpec`, `register_extractor`, `build_extractor(config)`, `list_extractors()` |
| `extractors.py` | Registers the built-ins |
| `batched.py` | `fbank_batch` / `mfcc_batch` / `extract_features_batch` — the fused Kaldi kernels |
| `whisper.py` | `batched_whisper_logmel` — the 30 s Whisper recipe |
| `nemotron.py` | `batched_nemotron_logmel` — the NeMo recipe |
| `lfr.py` | `apply_lfr_batch` — low-frame-rate stacking |
| `streaming.py` | `BatchedStreamingFeatureExtractor` — `B` parallel chunked streams |
| `backends.py` | `torchaudio.compliance.kaldi` (default) and the optional `kaldifeat` GPU path |

## `FeatureSpec` — what travels with the checkpoint

A converter emits a `FeatureSpec`, and the engine materializes `feature_config`
from it unless the caller set one explicitly (a mismatch logs a warning).

| Field | Notes |
|---|---|
| `kind` | `"kaldi_fbank"` / `"kaldi_mfcc"` / `"whisper_logmel"` / `"nemotron_logmel"` |
| `sample_rate`, `dim`, frame geometry | The grid |
| `lfr_m` / `lfr_n` | Low-frame-rate stacking (FunASR) |
| `window_seconds` | The fixed window, if the recipe has one |
| `normalize` | CMVN / global norm |
| `preemphasis` | Exists so a NeMo config that *disables* the filter can say so |
| `audio_scale` | **Per framework.** WeNet multiplies by `1 << 15`; icefall/lhotse, HF Whisper, Qwen2-Audio and Nemotron use `1.0`. A wrong value silently drops only the leading token. |

The engine adopts `FeatureSpec.audio_scale` unless the caller set a non-default
`audio_scale` explicitly.

## `ExtractorSpec` — what a frontend declares

```python
register_extractor(ExtractorSpec(
    kind="my_kind",
    fn=...,                     # (padded (B,T), lengths (B,), FeatureConfig)
                                #   -> (features (B,T',F) fp32, feat_lengths (B,))
    framing=...,                # StreamingFraming | None
    streaming_fn=...,           # incremental variant of fn, if the grid needs one
    window_seconds_attr="...",  # names the FeatureConfig field holding the window
))
```

Two declarations carry real consequences.

### `framing` — how to reproduce this grid from a growing buffer

`StreamingFraming(span, hop, history, prefill)`:

| Field | Meaning |
|---|---|
| `span` | What one frame *reads* — `n_fft`, **not** `win_length`, for a centered STFT |
| `hop` | Frame advance |
| `history` | Leading samples that are context only. Non-zero exactly when the per-sample transform reaches backwards — NeMo pre-emphasises the *signal* |
| `prefill` | The zeros an offline pass implicitly starts with |

`supports_streaming` is **derived** from `framing is not None`, so streamability
cannot disagree with the arithmetic, and `prepare_streaming` rejects a frontend
that declares none.

Every number the streaming feature loop used to hardcode as a Kaldi `snip_edges`
assumption — the per-frame sample span, the frame count, the initial tail, the
pre-emphasis reach — now comes from here. The *storage* was never missing
(`request.audio_tail` has always been a per-stream sample look-back); only the
declaration was.

`streaming_fn` is the incremental variant of `fn`, for a grid that is not simply
"restart at buffer position 0". Kaldi's `snip_edges` framing *is* its streaming
semantics, so it declares none.

### `window_seconds_attr` — a fixed window makes every row cost the same

It names the `FeatureConfig` field holding the window, which
`FeatureConfig.fixed_window_seconds` / `fixed_window_frames` read. That is what:

- tells the batching policies every row costs the same regardless of its length
  (otherwise `max_offline_pad_ratio` splits batches to avoid padding waste that
  does not exist), and
- lets the engine reject over-long audio **at admission** rather than silently
  transcribing a prefix.

### LFR is deliberately not per-extractor

It is a post-transform the caller applies over any extractor's output.
`lfr.py::apply_lfr_batch` is a clamped gather driven by `FeatureConfig.lfr_m` /
`lfr_n` (spec-emitted; FunASR Paraformer is 80-mel LFR 7/6 → 560-dim at a 60 ms
hop), and `FeatureConfig.output_dim` folds the stacking in. The engine applies it
in the offline `_fbank_batch` path only; streaming `prepare_streaming` rejects
LFR configs.

## Built-in recipes

### `fbank` / `mfcc` — Kaldi-compatible

The fused Kaldi kernels (`batched.py::batched_fbank` and friends), with a
per-utterance fallback for configurations the fused path cannot serve. Windows
include `hamming` (FunASR frontends), so Paraformer's collate stays on the fast
path. `snip_edges` framing means the offline grid *is* the streaming grid.

### `whisper_logmel`

The Whisper recipe: 30 s pad/trim, `n_fft` 400 / hop 160, slaney mels, global
max-norm, `audio_scale=1.0`. Returns **real** per-row frame counts
(`ceil(len/hop)`, HF attention-mask semantics) — Whisper ignores them, the
Qwen2-Audio tower masks by them. Fixed-window, so not streamable.

### `nemotron_logmel`

The NeMo recipe: pre-emphasis on the waveform, STFT `n_fft` 512 / hop 160 /
win 400 with a **non-periodic** Hann window and **constant** (not reflect)
padding, slaney mels, `log(mel + 2**-24)` — a natural log with no normalization
at all.

It runs on kernels — `oasr.stft_frame` → `oasr.rfft_power` → `oasr.mel_log`:
three launches, no waveform temporaries. The torch path is kept as the CPU path,
the fp32 parity oracle, and the `OASR_FEATURE_BACKEND=torch` A/B.

**Streamable.** It declares `StreamingFraming(512, 160, 1, 257)` and a
`center=False` incremental variant that reproduces the offline centered grid
exactly: `n_fft` span, one sample of pre-emphasis history, `n_fft // 2 + 1` of
prefill.

## The feature primitives

`oasr/feature.py` exposes the kernels the recipes are composed from.

### `stft_frame` — the general framing stage

Waveform → pre-emphasised, windowed, zero-padded frames. `hop`, `center_offset`,
`win_offset` and a pre-emphasis boundary mode are **parameters**, so one kernel
covers:

| Grid | `center_offset` |
|---|---|
| Centered NeMo / Whisper | `n_fft / 2` |
| Kaldi `snip_edges` | `0` |
| A streaming buffer whose head is pre-emphasis history | negative |

### `mel_log` — an additive guard alongside the floor

Kaldi is `log(max(m, tiny))`; NeMo is `log(m + 2**-24)`. One knob would move the
value of every silent bin — which *is* the encoder's input scale — so both are
exposed.

It also takes optional per-row `frame_lengths` masking, which has to happen
**after** the log, because `log(0 + 2**-24)` is a large negative constant, not
zero.

### Others

`dct_lifter` (MFCC), `fbank_preprocess`, `rfft` / `rfft_power` (`oasr/fft.py`).

## Extraction entry points

| API | Use |
|---|---|
| `fbank_batch` / `mfcc_batch` / `extract_features_batch` | Offline batch over padded `(B, T)` or a list of waveforms |
| `BatchedStreamingFeatureExtractor` | `B` parallel chunked streams — `process_chunk` / `flush` |

`InputProcessor` and `GraphedFeatureExtraction` each resolve **one** extractor at
construction, so an unregistered `feature_type` fails at engine build rather than
on the first request. `FeatureConfig.__post_init__` validates `feature_type`
against the registry.

## Adding a frontend

1. Write the batch function:
   `(padded_waveforms (B, T), lengths (B,), FeatureConfig) → (features (B, T', F) fp32, feat_lengths (B,))`.
   LFR stacking is **not** part of it.
2. `register_extractor(ExtractorSpec(kind="my_kind", fn=..., ...))`, setting
   `framing` and `window_seconds_attr` deliberately — see above.
3. Emit `FeatureSpec(kind=...)` from the checkpoint converter so the engine
   materializes the matching `FeatureConfig` automatically.

## Environment

`OASR_FEATURE_BACKEND=torch` forces the reference frontend — the
`nemotron_logmel` kernel chain's CPU path, its fp32 parity oracle, and the
rollback / A/B switch.

Frontend parity results are in `.artifacts/model_validation.md`.
