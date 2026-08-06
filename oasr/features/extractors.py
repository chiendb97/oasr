# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Built-in feature extractors, registered on the feature axis.

Each entry wraps an existing implementation — this module adds the dispatch and
the declared properties (streamability, fixed window), not new maths:

* ``fbank`` / ``mfcc`` — the fused Kaldi kernels (:mod:`oasr.features.batched`)
  when the config is exactly Kaldi-compliant, else a per-utterance fallback so
  unusual configs (non-Povey window, dither, ``use_energy``) still produce correct
  features rather than silently wrong ones;
* ``whisper_logmel`` — the 30 s Whisper recipe (:mod:`oasr.features.whisper`),
  shared by Qwen2-Audio.  Fixed-window and therefore **not** streamable: it
  normalises over the whole padded window, so it cannot consume a growing buffer.
* ``nemotron_logmel`` — the NeMo pre-emphasis + ``log(mel + 2**-24)`` recipe
  (:mod:`oasr.features.nemotron`).  Frame-local, so it *is* streamable — but its
  grid comes from one ``center=True`` pass, so it declares a
  :class:`~oasr.features.StreamingFraming` (``n_fft`` span, one sample of
  pre-emphasis history, ``n_fft // 2 + 1`` of prefill) and a ``center=False``
  incremental variant rather than restarting the grid per chunk.

Each entry declares its frame grid via ``framing``, which is what
``supports_streaming`` is derived from.  The Kaldi frontends declare the
``snip_edges`` grid they have always implemented — the same behaviour, now said
out loud instead of hardcoded in the engine's streaming loop.

LFR stacking is deliberately *not* here — it is a post-transform over any
extractor's output, applied once by the caller.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .backends import _extract as _extract_single
from .batched import (
    batched_fbank,
    batched_mfcc,
    supports_batched_fbank,
    supports_batched_mfcc,
)
from .config import FeatureConfig
from .registry import ExtractorSpec, StreamingFraming, register_extractor

__all__ = [
    "kaldi_extract",
    "kaldi_framing",
    "nemotron_extract",
    "nemotron_extract_streaming",
    "whisper_extract",
]


def _per_utterance(
    waveforms: torch.Tensor, lengths: torch.Tensor, config: FeatureConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback: extract row by row, then pad.  Correct for any config, slower."""
    feats = [_extract_single(waveforms[i, :n], config) for i, n in enumerate(lengths.tolist())]
    feat_lengths = torch.tensor(
        [f.size(0) for f in feats], dtype=torch.int32, device=waveforms.device
    )
    padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
    return padded, feat_lengths


def kaldi_extract(
    waveforms: torch.Tensor, lengths: torch.Tensor, config: FeatureConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kaldi FBANK / MFCC — fused kernel when the config allows, else per-row."""
    if supports_batched_fbank(config) or supports_batched_mfcc(config):
        batched = batched_mfcc if config.feature_type == "mfcc" else batched_fbank
        return batched(waveforms, lengths, config)
    return _per_utterance(waveforms, lengths, config)


def whisper_extract(
    waveforms: torch.Tensor, lengths: torch.Tensor, config: FeatureConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whisper log-mel over the fixed 30 s window."""
    from .whisper import batched_whisper_logmel

    return batched_whisper_logmel(waveforms, lengths, config)


def nemotron_extract(
    waveforms: torch.Tensor, lengths: torch.Tensor, config: FeatureConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NeMo pre-emphasis + natural-log mel spectrogram (Nemotron / Parakeet)."""
    from .nemotron import batched_nemotron_logmel

    return batched_nemotron_logmel(waveforms, lengths, config)


def nemotron_extract_streaming(
    waveforms: torch.Tensor, lengths: torch.Tensor, config: FeatureConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Incremental Nemotron log-mel over per-stream carry-over buffers."""
    from .nemotron import batched_nemotron_logmel_streaming

    return batched_nemotron_logmel_streaming(waveforms, lengths, config)


def kaldi_framing(config: FeatureConfig) -> StreamingFraming:
    """Kaldi ``snip_edges`` grid: frame ``f`` spans ``[f*shift, f*shift + length)``.

    No history and no prefill — Kaldi pre-emphasises *within* a frame (with a
    replicate boundary) rather than over the signal, so nothing reaches across a
    frame start, and frame 0 begins at sample 0.  This is exactly what the
    streaming feature path already did; declaring it changes no behaviour.

    ``snip_edges=False`` shifts every frame by half a window and emits a frame per
    hop regardless of the signal end, so this grid does not describe it.  The
    streaming path's frame bookkeeping has always assumed ``snip_edges``; raising
    here is what turns that assumption into a refusal instead of a slow drift in
    the per-stream sample cursor.
    """
    if not config.snip_edges:
        raise NotImplementedError(
            "streaming Kaldi feature extraction requires snip_edges=True: the "
            "per-stream sample cursor advances one hop per emitted frame, which "
            "snip_edges=False's half-window offset does not satisfy"
        )
    return StreamingFraming(
        span=config.frame_length_samples,
        hop=config.frame_shift_samples,
        history=0,
        prefill=0,
    )


register_extractor(ExtractorSpec(kind="fbank", fn=kaldi_extract, framing=kaldi_framing))
register_extractor(ExtractorSpec(kind="mfcc", fn=kaldi_extract, framing=kaldi_framing))
register_extractor(
    ExtractorSpec(
        kind="whisper_logmel",
        fn=whisper_extract,
        # No framing: it normalises across the whole padded window, so no sliding
        # grid reproduces it and it cannot consume a growing buffer.
        framing=None,
        # The window itself is a config knob (default 30 s); the registration only
        # declares that this frontend *has* one and where to read it.
        window_seconds_attr="whisper_chunk_seconds",
    )
)


def _nemotron_framing(config: FeatureConfig) -> StreamingFraming:
    from .nemotron import nemotron_streaming_framing

    return nemotron_streaming_framing(config)


register_extractor(
    ExtractorSpec(
        kind="nemotron_logmel",
        fn=nemotron_extract,
        # Frame-local, so it streams — but the offline grid comes from one
        # ``center=True`` pass, which the framing + incremental variant reproduce
        # (verified bit-exact; see ``oasr/features/nemotron.py``).
        framing=_nemotron_framing,
        streaming_fn=nemotron_extract_streaming,
        # Cost tracks the real utterance length — no padded window.
        window_seconds_attr=None,
    )
)
