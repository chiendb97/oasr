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

* **how** to turn a padded waveform batch into features (``fn``);
* whether it can run **incrementally** (``supports_streaming``) — the streaming
  feature path windows a growing buffer, which a fixed-window frontend cannot do;
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
class ExtractorSpec:
    """One batched offline feature extractor plus the properties callers need."""

    kind: str
    fn: ExtractorFn
    #: Whether the streaming feature path can drive this incrementally.  A
    #: fixed-window frontend cannot: it needs the whole utterance to normalise.
    supports_streaming: bool = True
    #: Name of the :class:`~oasr.features.FeatureConfig` field holding this
    #: frontend's fixed window **in seconds**, or ``None`` when cost tracks the real
    #: utterance length.  The registry declares *whether* a frontend is
    #: fixed-window; the config stays the source of *how wide*, so the window
    #: remains a per-deployment knob.  ``FeatureConfig.fixed_window_seconds`` reads
    #: this — which is what makes a new fixed-window frontend a registration rather
    #: than an edit to the shared config.
    window_seconds_attr: Optional[str] = None

    def __call__(
        self, waveforms: torch.Tensor, lengths: torch.Tensor, config: FeatureConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fn(waveforms, lengths, config)


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
