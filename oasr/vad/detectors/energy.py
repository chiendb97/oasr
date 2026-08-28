# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Log-energy speech detection — the dependency-free baseline.

Kaldi's ``compute-vad`` with one deliberate change.  Kaldi's rule is
``threshold + mean_scale * mean(log_energy)`` with ``threshold = 5.0``, and that
absolute term is only meaningful at int16 waveform scale: the same audio at unit
scale shifts every log-energy by ``2 * ln(32768)`` and the threshold stops
meaning anything.  OASR carries ``audio_scale`` as a *per-framework* convention
precisely because both conventions are in the wild, so an absolute energy
threshold here would be a silent, scale-dependent failure — the same class of bug
as a wrong ``audio_scale``, which shifts every log-mel bin by a constant and
costs the transcript's leading token.

So the threshold is **peak-relative**: speech is whatever sits within
``dynamic_range_db`` of the loudest frame in the utterance.  Scale-invariant by
construction, and it needs no configuration coupling to the frontend.

Kaldi's header is blunt about the limits of this whole family — *"not suitable
for automatic speech recognition because it makes independent decisions for each
frame without imposing any notion of continuity"* — and it is right.  The
continuity is supplied here by :class:`~oasr.vad.segmenter.SpeechSegmenter`,
which every detector shares; that split is the reason a detector this crude is
usable at all.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import VadConfig
from ..detector import SpeechDetector
from ..registry import VadFraming

__all__ = ["EnergyDetector", "energy_framing"]

#: Mean-square power at or below which a row is called digital silence outright.
#: Without this the peak-relative rule inverts on an all-zero row: the peak is
#: the floor, every frame sits at the peak, and pure silence reads as pure
#: speech.  Not hypothetical — ``finalize_silence_pad`` appends exactly such a
#: run to the end of every closed stream.
_SILENCE_POWER_FLOOR = 1e-10

#: Power decibels per nat, for reading ``dynamic_range_db`` in log-energy units.
_DB_PER_NAT = 10.0 / math.log(10.0)


def energy_framing(config: VadConfig) -> VadFraming:
    """The configurable analysis grid this detector runs on."""
    span = max(1, int(round(config.sample_rate * config.frame_ms / 1000.0)))
    hop = max(1, int(round(config.sample_rate * config.hop_ms / 1000.0)))
    return VadFraming(span=span, hop=hop)


class EnergyDetector(SpeechDetector):
    """Peak-relative log-energy, squashed into a probability.

    Parameters
    ----------
    dynamic_range_db : float
        How far below the loudest frame a frame may sit and still be speech.
        35 dB suits clean read speech; noisy audio needs less, and a recording
        whose noise floor is within this range of its peak will read as
        continuous speech — which is the documented failure of every energy VAD
        and the reason the ASR-derived detectors exist.
    slope : float
        Sharpness of the logistic in nats.  At ``1.0`` a frame 10 dB above the
        threshold scores about 0.9.
    """

    kind: ClassVar[str] = "energy"

    def __init__(
        self,
        config: VadConfig,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        dynamic_range_db: float = 35.0,
        slope: float = 1.0,
    ) -> None:
        framing = energy_framing(config)
        super().__init__(
            seconds_per_frame=framing.seconds_per_frame(config.sample_rate),
            device=device,
            dtype=dtype,
        )
        if dynamic_range_db <= 0:
            raise ValueError(f"dynamic_range_db must be > 0, got {dynamic_range_db!r}")
        if slope <= 0:
            raise ValueError(f"slope must be > 0, got {slope!r}")
        self._framing = framing
        self._dyn_nats = float(dynamic_range_db) / _DB_PER_NAT
        self._slope = float(slope)

    @property
    def framing(self) -> VadFraming:
        return self._framing

    def detect(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be (B, T) or (T,), got {tuple(waveform.shape)}")
        wav = waveform.to(torch.float32)
        span, hop = self._framing.span, self._framing.hop
        batch = wav.size(0)
        device = wav.device

        lengths_dev = lengths.to(device=device, dtype=torch.int64)
        # The declared grid, evaluated on device: reading the per-row counts back
        # to the host here would synchronise the stream for a few bytes, and the
        # offline collate path already learned that lesson the expensive way.
        frame_lengths = torch.clamp((lengths_dev - span) // hop + 1, min=0)

        if wav.size(1) < span:
            empty = wav.new_zeros(batch, 0)
            return empty, torch.zeros(batch, dtype=torch.int64, device=device)

        # avg_pool1d over the squared signal rather than ``unfold``: unfold
        # materialises (B, F, span), which for one 30 s utterance is 1.2 M floats
        # per row before the reduction even starts.
        power = F.avg_pool1d(
            wav.pow(2).unsqueeze(1), kernel_size=span, stride=hop, count_include_pad=False
        ).squeeze(1)

        n_frames = power.size(1)
        valid = torch.arange(n_frames, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
        # Padding is excluded from the peak, or the batch's widest row would set
        # every shorter row's threshold.
        peak = power.masked_fill(~valid, 0.0).amax(dim=1, keepdim=True)
        silent_row = peak <= _SILENCE_POWER_FLOOR

        floor = _SILENCE_POWER_FLOOR * 1e-2
        log_e = torch.log(power.clamp_min(floor))
        threshold = torch.log(peak.clamp_min(floor)) - self._dyn_nats
        probs = torch.sigmoid((log_e - threshold) * self._slope)
        probs = torch.where(silent_row.expand_as(probs), torch.zeros_like(probs), probs)
        return self._mask_padding(probs, frame_lengths), frame_lengths


def build_energy(
    config: VadConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> EnergyDetector:
    """Factory for the registry.  Unknown kwargs are the engine's ASR-only extras."""
    return EnergyDetector(
        config,
        device=device,
        dtype=dtype,
        dynamic_range_db=float(kwargs.get("dynamic_range_db", 35.0)),  # type: ignore[arg-type]
        slope=float(kwargs.get("slope", 1.0)),  # type: ignore[arg-type]
    )
