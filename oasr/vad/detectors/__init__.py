# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Built-in speech detectors, and their registration.

Imported lazily by :func:`oasr.vad.registry._ensure_builtins`, so importing
``oasr.vad`` for its config objects does not pull in torch.

Every ASR-derived kind declares ``("stream", "posthoc")`` and never
``"presegment"``: it reads what the encoder produced, so it cannot precede the
encoder.  The registry enforces that at registration rather than at first
request.

Two waveform detectors ship.  ``silero`` is the one to use when quality matters
— it is what every open-source pre-ASR segmenter runs, and OASR's segmentation
knobs are already spelled in its vocabulary — but it needs its weights pointed
at.  ``energy`` needs nothing and is the baseline the axis can always fall back
to; its documented failure (a noise floor within ``dynamic_range_db`` of the peak
reads as continuous speech) is exactly what the neural one fixes.

The energy baseline declares all three roles.  ``"presegment"`` is what lets it
cut audio before the encoder sees it — offline through the long-form fan-out,
streaming through the per-window gate — and ``"stream"`` says it can do that
incrementally, which for this detector means carrying a running peak and the
sub-frame sample remainder across chunk boundaries rather than restarting its
reference at every chunk.  It is ``stateful`` for that reason: without the carry
each chunk would be normalised against its own loudest frame and a chunk of pure
room tone would read as a chunk of pure speech.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..config import VadConfig
from ..registry import VadSpec, register_vad
from .asr import AedNoSpeechDetector, CifAlphaDetector, CtcBlankDetector, FrameActivityDetector
from .energy import EnergyDetector, build_energy, energy_framing
from .silero import SileroDetector, SileroVadNet, build_silero, silero_framing

__all__ = [
    "EnergyDetector",
    "SileroDetector",
    "SileroVadNet",
    "CtcBlankDetector",
    "FrameActivityDetector",
    "CifAlphaDetector",
    "AedNoSpeechDetector",
]


def _require(name: str, kind: str, kwargs: Dict[str, Any]) -> Any:
    """Pull a mandatory factory argument, or say which caller failed to supply it."""
    if name not in kwargs or kwargs[name] is None:
        raise ValueError(
            f"the {kind!r} detector needs {name!r}, which the engine supplies from the "
            "running model; it is missing, so this detector was built by hand without it"
        )
    return kwargs[name]


def _asr_common(kind: str, kwargs: Dict[str, Any], device: Any, dtype: Any) -> Dict[str, Any]:
    return {
        "seconds_per_frame": float(_require("seconds_per_frame", kind, kwargs)),
        "device": device,
        "dtype": dtype,
    }


def _build_ctc_blank(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> CtcBlankDetector:
    common = _asr_common("ctc_blank", kwargs, device, dtype)
    return CtcBlankDetector(
        config,
        blank_id=int(_require("blank_id", "ctc_blank", kwargs)),
        dilate_s=float(kwargs.get("dilate_s", 0.1)),
        **common,
    )


def _build_transducer_blank(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> FrameActivityDetector:
    return FrameActivityDetector(
        config,
        dilate_s=float(kwargs.get("dilate_s", 0.2)),
        **_asr_common("transducer_blank", kwargs, device, dtype),
    )


def _build_cif_alpha(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> CifAlphaDetector:
    common = _asr_common("cif_alpha", kwargs, device, dtype)
    return CifAlphaDetector(
        config,
        smooth_frames=int(kwargs.get("smooth_frames", 5)),
        gain=float(kwargs.get("gain", 4.0)),
        **common,
    )


def _build_aed_no_speech(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> AedNoSpeechDetector:
    common = _asr_common("aed_no_speech", kwargs, device, dtype)
    token = int(_require("no_speech_token_id", "aed_no_speech", kwargs))
    return AedNoSpeechDetector(config, no_speech_token_id=token, **common)


register_vad(
    VadSpec(
        kind="energy",
        factory=build_energy,
        consumes="waveform",
        framing=energy_framing,
        modes=("presegment", "stream", "posthoc"),
        stateful=True,
        doc="peak-relative log-energy; dependency-free baseline",
    )
)

register_vad(
    VadSpec(
        kind="silero",
        factory=build_silero,
        consumes="waveform",
        framing=silero_framing,
        modes=("presegment", "stream", "posthoc"),
        stateful=True,
        needs_weights=True,
        doc="Silero VAD v5 (MIT, 309K params); needs --vad-model-dir",
    )
)

register_vad(
    VadSpec(
        kind="ctc_blank",
        factory=_build_ctc_blank,
        consumes="asr_log_probs",
        modes=("stream", "posthoc"),
        min_silence_floor_ms=1000,
        doc="1 - P(blank) per encoder frame, from the CTC head's own log-probs",
    )
)

register_vad(
    VadSpec(
        kind="transducer_blank",
        factory=_build_transducer_blank,
        consumes="asr_frames",
        modes=("stream", "posthoc"),
        min_silence_floor_ms=1000,
        doc="emission-frame activity from the transducer greedy loop (greedy only)",
    )
)

register_vad(
    VadSpec(
        kind="cif_alpha",
        factory=_build_cif_alpha,
        consumes="asr_alphas",
        modes=("posthoc",),
        min_silence_floor_ms=500,
        doc="Paraformer CIF token rate, boxcar-smoothed (heuristic gain)",
    )
)

register_vad(
    VadSpec(
        kind="aed_no_speech",
        factory=_build_aed_no_speech,
        consumes="asr_prefill_logits",
        modes=("posthoc",),
        doc="Whisper <|nospeech|> probability; one frame per decoding window",
    )
)
