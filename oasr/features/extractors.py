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
from .registry import ExtractorSpec, register_extractor

__all__ = ["kaldi_extract", "whisper_extract"]


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


register_extractor(ExtractorSpec(kind="fbank", fn=kaldi_extract, supports_streaming=True))
register_extractor(ExtractorSpec(kind="mfcc", fn=kaldi_extract, supports_streaming=True))
register_extractor(
    ExtractorSpec(
        kind="whisper_logmel",
        fn=whisper_extract,
        # Normalises across the whole padded window, so it cannot consume a
        # growing buffer chunk by chunk.
        supports_streaming=False,
        # The window itself is a config knob (default 30 s); the registration only
        # declares that this frontend *has* one and where to read it.
        window_seconds_attr="whisper_chunk_seconds",
    )
)
