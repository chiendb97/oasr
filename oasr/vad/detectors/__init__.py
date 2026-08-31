# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Built-in speech detectors.

Imported lazily by :func:`oasr.vad.registry._ensure_builtins`, so importing
``oasr.vad`` for its config objects does not pull in torch.  Each module below
registers its own kinds on import, the way ``oasr/tokenizers`` and
``oasr/engine/decode`` do — importing the names here is what runs them.

Two waveform detectors ship.  ``silero`` is the one to use when quality matters
— it is what every open-source pre-ASR segmenter runs, and OASR's segmentation
knobs are already spelled in its vocabulary — but it needs its weights pointed
at.  ``energy`` needs nothing and is the baseline the axis can always fall back
to; its documented failure (a noise floor within ``dynamic_range_db`` of the peak
reads as continuous speech) is exactly what the neural one fixes.  The four
kinds in :mod:`~oasr.vad.detectors.asr` need no weights at all, because the model
that produced their signal is already running.
"""

from __future__ import annotations

from .asr import (
    AedNoSpeechDetector,
    CifAlphaDetector,
    CtcBlankDetector,
    FrameActivityDetector,
    build_aed_no_speech,
    build_cif_alpha,
    build_ctc_blank,
    build_transducer_blank,
)
from .energy import EnergyDetector, EnergyVadState, build_energy, energy_framing
from .silero import (
    SileroDetector,
    SileroVadNet,
    SileroVadState,
    build_silero,
    convert_silero_state_dict,
    load_silero_weights,
    silero_framing,
)

__all__ = [
    "EnergyDetector",
    "EnergyVadState",
    "energy_framing",
    "build_energy",
    "SileroDetector",
    "SileroVadNet",
    "SileroVadState",
    "silero_framing",
    "build_silero",
    "convert_silero_state_dict",
    "load_silero_weights",
    "CtcBlankDetector",
    "FrameActivityDetector",
    "CifAlphaDetector",
    "AedNoSpeechDetector",
    "build_ctc_blank",
    "build_transducer_blank",
    "build_cif_alpha",
    "build_aed_no_speech",
]
