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
from typing import Any, ClassVar, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config import VadConfig
from ..detector import SpeechDetector, VadState
from ..registry import VadFraming

__all__ = ["EnergyDetector", "EnergyVadState", "energy_framing"]

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


class EnergyVadState(VadState):
    """Per-stream running peak power, one entry per row of the current call.

    The peak-relative rule needs a reference loudness, and offline that is the
    utterance's own maximum.  A stream has no utterance to take a maximum over
    until it ends, so the reference is the loudest frame **seen so far** and it
    only ever grows.  Two consequences worth stating, because both are visible in
    the output rather than in a log:

    * the first chunks of a stream are judged against a peak that has not been
      established yet, so a stream that opens on room tone reads as speech and is
      *encoded*.  That is the safe direction — the gate's failure mode must be
      doing work, never dropping audio.
    * a stream with one loud burst raises the bar for everything after it, so
      genuinely quieter speech more than ``dynamic_range_db`` below that burst
      reads as silence.  This is the documented failure of every energy VAD
      (Kaldi says as much in ``compute-vad``'s own header) and the reason a
      neural detector is the eventual answer here.
    """

    __slots__ = ("peak",)

    def __init__(self, peak: torch.Tensor) -> None:
        #: ``(B, 1)`` mean-square power, aligned with the rows of the call that
        #: produced it.  The stage that owns the streams scatters it back per
        #: stream, because batch membership changes from tick to tick.
        self.peak = peak


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

    def new_state(self, batch: int) -> EnergyVadState:
        """A peak of zero, which reads as "nothing loud seen yet"."""
        return EnergyVadState(torch.zeros(batch, 1, dtype=torch.float32, device=self._device))

    def stack_states(self, states: Sequence[Optional[VadState]]) -> EnergyVadState:
        peaks = []
        for st in states:
            if isinstance(st, EnergyVadState):
                peaks.append(st.peak.reshape(1, 1))
            else:
                peaks.append(torch.zeros(1, 1, dtype=torch.float32, device=self._device))
        return EnergyVadState(torch.cat(peaks, dim=0))

    def unstack_states(self, state: Optional[VadState], count: int) -> List[Optional[VadState]]:
        if not isinstance(state, EnergyVadState):
            return [self.new_state(1) for _ in range(count)]
        return [EnergyVadState(state.peak[i : i + 1]) for i in range(count)]

    def _frame_power(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(B, T)`` waveform → ``(power (B, F), frame_lengths (B,), valid (B, F))``."""
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
            return (
                empty,
                torch.zeros(batch, dtype=torch.int64, device=device),
                empty.to(torch.bool),
            )

        # avg_pool1d over the squared signal rather than ``unfold``: unfold
        # materialises (B, F, span), which for one 30 s utterance is 1.2 M floats
        # per row before the reduction even starts.
        power = F.avg_pool1d(
            wav.pow(2).unsqueeze(1), kernel_size=span, stride=hop, count_include_pad=False
        ).squeeze(1)

        n_frames = power.size(1)
        valid = torch.arange(n_frames, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
        return power, frame_lengths, valid

    def _probs_from_power(
        self, power: torch.Tensor, frame_lengths: torch.Tensor, peak: torch.Tensor
    ) -> torch.Tensor:
        """Peak-relative logistic over frame power, given a ``(B, 1)`` reference."""
        silent_row = peak <= _SILENCE_POWER_FLOOR
        floor = _SILENCE_POWER_FLOOR * 1e-2
        log_e = torch.log(power.clamp_min(floor))
        threshold = torch.log(peak.clamp_min(floor)) - self._dyn_nats
        probs = torch.sigmoid((log_e - threshold) * self._slope)
        probs = torch.where(silent_row.expand_as(probs), torch.zeros_like(probs), probs)
        return self._mask_padding(probs, frame_lengths)

    def detect(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        power, frame_lengths, valid = self._frame_power(waveform, lengths)
        if power.size(1) == 0:
            return power, frame_lengths
        # Padding is excluded from the peak, or the batch's widest row would set
        # every shorter row's threshold.
        peak = power.masked_fill(~valid, 0.0).amax(dim=1, keepdim=True)
        return self._probs_from_power(power, frame_lengths, peak), frame_lengths

    def detect_streaming(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[VadState],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[VadState]]:
        """One chunk per row, against a peak carried across chunks.

        The base class routes ``detect_streaming`` to ``detect``, and for this
        detector that would be wrong in a way no unit test on a single chunk can
        see: each chunk would be normalised against *its own* loudest frame, so a
        chunk of pure room tone would read as a chunk of pure speech.  Carrying
        the peak is what makes the two flows agree; the running maximum is the
        only state, and :class:`EnergyVadState` documents what it costs.
        """
        power, frame_lengths, valid = self._frame_power(waveform, lengths)
        prev = state.peak if isinstance(state, EnergyVadState) else None
        if power.size(1) == 0:
            # Nothing to measure — hand the reference straight back rather than
            # dropping it, or the next chunk would restart from its own peak.
            if prev is None:
                prev = power.new_zeros(power.size(0), 1)
            return power, frame_lengths, EnergyVadState(prev)
        chunk_peak = power.masked_fill(~valid, 0.0).amax(dim=1, keepdim=True)
        peak = chunk_peak if prev is None else torch.maximum(prev.to(chunk_peak), chunk_peak)
        return (
            self._probs_from_power(power, frame_lengths, peak),
            frame_lengths,
            EnergyVadState(peak),
        )


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
