# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Feature-extractor registry — the sixth-and-a-half extension axis (F1).

Every other axis in the engine resolves through a registry: models, checkpoint
converters, decode strategies, streaming backends, batching policies, tokenizers.
Feature extraction was the exception — :class:`~oasr.features.FeatureSpec` even
documents ``kind`` as keying "the (future) extractor registry" — so dispatch lived
as ``if feature_type == ...`` chains, including one inside the *shared*
:class:`~oasr.engine.input_processor.InputProcessor` complete with a
function-body import of an architecture-specific module.  Adding a frontend (raw
waveform for wav2vec, an 8 kHz telephony spec, a different LFR recipe) therefore
meant editing the engine.

An extractor answers three questions:

* **how** to turn a padded waveform batch into features (``fn``), and — when the
  frame grid is not simply "restart at buffer position 0" — how to do it
  incrementally (``streaming_fn``);
* whether it can run **incrementally** at all, and on what grid
  (``framing``) — the streaming feature path windows a growing buffer, which a
  frontend that normalises over a fixed window cannot do;
* whether its cost is **fixed per row** (``window_seconds_attr``) — the batching
  policies need this, because a frontend that pads *and trims* every utterance to
  30 s makes every row cost the same regardless of its length.

Registered under the ``FeatureConfig.feature_type`` value (``"fbank"`` /
``"mfcc"`` / ``"whisper_logmel"``), which is what the engine holds.  Note the
``FeatureSpec.kind`` vocabulary is adjacent but distinct (``"kaldi_fbank"`` /
``"kaldi_mfcc"`` / ``"whisper_logmel"`` / ``"raw"``); ``FeatureSpec.to_feature_config``
is the one place that maps between them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .config import FeatureConfig

__all__ = [
    "ExtractorSpec",
    "StreamingFraming",
    "register_extractor",
    "build_extractor",
    "list_extractors",
]

logger = logging.getLogger(__name__)

#: ``(padded_waveforms (B, T), lengths (B,), config) -> (features (B, T', F), feat_lengths (B,))``
#: Features come back **fp32**; the caller casts to the model dtype.
ExtractorFn = Callable[
    [torch.Tensor, torch.Tensor, FeatureConfig], Tuple[torch.Tensor, torch.Tensor]
]


@dataclass(frozen=True)
class StreamingFraming:
    """How to reproduce a frontend's frame grid from a **growing sample buffer**.

    The streaming feature path is a sliding window over a per-stream sample
    stream, and four integers describe any such grid.  They used to be implicit:
    every one of them was a hardcoded Kaldi ``snip_edges`` assumption inside
    :class:`~oasr.engine.input_processor.InputProcessor`, which is why a centered
    STFT frontend could not stream at all even though its arithmetic is
    frame-local.

    Attributes
    ----------
    span : int
        Samples one frame reads, measured from its own start.  ``frame_length``
        for Kaldi; ``n_fft`` for an STFT frontend whose window is *narrower* than
        the transform — using ``win_length`` there would emit a frame before its
        last samples had arrived.
    hop : int
        Samples between consecutive frame starts.
    history : int
        Leading buffer samples that are **context only** — they influence frame
        values but do not start a frame.  Non-zero exactly when the frontend's
        per-sample transform reaches backwards: NeMo pre-emphasises the *signal*,
        so each frame needs one sample from before it.
    prefill : int
        Zero samples the buffer starts with, standing in for the implicit left
        padding of an offline pass (``n_fft // 2`` for a centered STFT, plus
        ``history``).  ``0`` for a frontend whose frame 0 starts at sample 0.
    """

    span: int
    hop: int
    history: int = 0
    prefill: int = 0

    def frames_for(self, num_samples: int) -> int:
        """Frames a buffer of ``num_samples`` can emit (``>= 0``)."""
        if num_samples < self.history + self.span:
            return 0
        return (num_samples - self.history - self.span) // self.hop + 1

    @property
    def min_samples(self) -> int:
        """Buffered samples needed before the first frame can be emitted."""
        return self.history + self.span


@dataclass(frozen=True)
class ExtractorSpec:
    """One batched feature extractor plus the properties callers need."""

    kind: str
    fn: ExtractorFn
    #: How to reproduce this frontend's frame grid incrementally — a function of
    #: the config, because span and hop are config-derived (Kaldi reads
    #: ``frame_length_samples``; an STFT frontend derives ``n_fft`` from it).
    #: ``None`` means it cannot be reproduced: a frontend that normalises over a
    #: fixed window needs the whole utterance, so no framing describes it.
    #: ``supports_streaming`` reads this, which makes streamability a
    #: *declaration* rather than a flag that can disagree with the arithmetic.
    framing: Optional[Callable[[FeatureConfig], StreamingFraming]] = None
    #: Incremental variant of ``fn``, taking a per-stream buffer laid out as
    #: ``framing`` describes.  ``None`` means ``fn`` already frames from buffer
    #: position 0 with no history — true for the Kaldi frontends, whose
    #: ``snip_edges`` framing *is* their streaming semantics.
    streaming_fn: Optional[ExtractorFn] = None
    #: Name of the :class:`~oasr.features.FeatureConfig` field holding this
    #: frontend's fixed window **in seconds**, or ``None`` when cost tracks the real
    #: utterance length.  The registry declares *whether* a frontend is
    #: fixed-window; the config stays the source of *how wide*, so the window
    #: remains a per-deployment knob.  ``FeatureConfig.fixed_window_seconds`` reads
    #: this — which is what makes a new fixed-window frontend a registration rather
    #: than an edit to the shared config.
    window_seconds_attr: Optional[str] = None

    @property
    def supports_streaming(self) -> bool:
        """Whether the streaming feature path can drive this incrementally."""
        return self.framing is not None

    def __call__(
        self, waveforms: torch.Tensor, lengths: torch.Tensor, config: FeatureConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fn(waveforms, lengths, config)

    def framing_for(self, config: FeatureConfig) -> StreamingFraming:
        """This frontend's streaming frame grid for ``config``."""
        if self.framing is None:
            raise NotImplementedError(
                f"the {self.kind!r} frontend declares no streaming framing; "
                "it cannot consume a growing buffer"
            )
        return self.framing(config)

    def extract_streaming(
        self, waveforms: torch.Tensor, lengths: torch.Tensor, config: FeatureConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the incremental variant over per-stream buffers."""
        self.framing_for(config)  # raises for a non-streamable frontend
        fn = self.streaming_fn or self.fn
        return fn(waveforms, lengths, config)


_REGISTRY: Dict[str, ExtractorSpec] = {}


def register_extractor(spec: ExtractorSpec) -> ExtractorSpec:
    """Register ``spec`` under its ``kind`` (idempotent; last write wins)."""
    if spec.kind in _REGISTRY:
        logger.debug("Overriding feature extractor for %r", spec.kind)
    _REGISTRY[spec.kind] = spec
    return spec


def _ensure_builtins() -> None:
    """Import the built-in extractors so their registration runs.

    Lazy to avoid an import cycle (each extractor module imports this one) and to
    keep the optional-backend imports out of ``import oasr``.
    """
    if not _REGISTRY:
        from . import extractors  # noqa: F401


def build_extractor(config: FeatureConfig) -> ExtractorSpec:
    """Resolve the extractor for ``config.feature_type``.

    Raises ``NotImplementedError`` naming the registered kinds — the extension
    point for a new frontend.
    """
    _ensure_builtins()
    spec = _REGISTRY.get(config.feature_type)
    if spec is None:
        raise NotImplementedError(
            f"No feature extractor registered for feature_type="
            f"{config.feature_type!r}.  Registered: {sorted(_REGISTRY)}.  "
            "Add one by calling oasr.features.register_extractor(ExtractorSpec(...))."
        )
    return spec


def list_extractors() -> List[str]:
    """Names of all registered extractors."""
    _ensure_builtins()
    return sorted(_REGISTRY)
