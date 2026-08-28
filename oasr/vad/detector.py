# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The detector contract: audio (or an ASR tensor) in, per-frame speech probability out.

A detector answers exactly one question — *how likely is frame ``t`` speech* — on
a declared frame grid, and nothing else.  Turning that answer into segments is
:mod:`oasr.vad.segmenter`'s job and turning it into a turn boundary is
:mod:`oasr.vad.endpointer`'s, so a Silero VAD and a CTC blank posterior produce
identical segment semantics, identical knobs and identical events.

Only the entry point a detector's declared ``consumes`` implies is ever called;
the others raise with an actionable message rather than returning something
plausible.  That is the ``ExtractorSpec.framing is None`` idiom — a declared gap,
not a silent fallback.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, List, Optional, Tuple

import torch

__all__ = ["SpeechDetector", "VadState", "as_rows"]


class VadState:
    """Opaque per-stream detector state.

    Held by the engine on behalf of one stream and handed back on the next
    chunk.  The base class is deliberately empty: a stateless detector needs
    nothing, an energy detector carries a running peak, and a recurrent one
    carries its hidden state.  Keeping it opaque is what let the transducer
    predictor serve both a stateless label window and an LSTM through one loop.
    """

    __slots__ = ()


class SpeechDetector(ABC):
    """Per-frame speech probability, on a declared grid.

    Subclasses implement whichever of :meth:`detect` / :meth:`detect_streaming` /
    :meth:`detect_from_asr` their registered ``consumes`` names.
    """

    #: Registry key.  Set by the subclass and matched against its :class:`VadSpec`.
    kind: ClassVar[str]

    def __init__(
        self,
        *,
        seconds_per_frame: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if seconds_per_frame <= 0.0:
            # A rate of zero or a negative rate would scale every reported span
            # by an arbitrary constant, and the result looks entirely plausible.
            # ``FrameClock.resolve`` refuses for the same reason.
            raise ValueError(f"seconds_per_frame must be > 0, got {seconds_per_frame!r}")
        self._seconds_per_frame = float(seconds_per_frame)
        self._device = device
        self._dtype = dtype

    @property
    def seconds_per_frame(self) -> float:
        """Audio seconds one output frame advances."""
        return self._seconds_per_frame

    # -- per-stream state ---------------------------------------------------

    def new_state(self, batch: int) -> Optional[VadState]:
        """Fresh state for ``batch`` streams, or ``None`` when stateless."""
        return None

    # -- waveform detectors -------------------------------------------------

    def detect(self, waveform: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """``(B, T)`` padded waveform + ``(B,)`` sample counts → probabilities.

        Returns ``(probs (B, T_vad) float32, frame_lengths (B,) int64)``, with
        ``probs`` in ``[0, 1]``.  One batched call for the whole set — never one
        per row.  Per-item Python on a batched path is the anti-pattern that cost
        the streaming feature loop 21 % of its wall clock before 146 per-stream
        copies became one ``torch._foreach_copy_``.
        """
        raise NotImplementedError(
            f"the {type(self).__name__} detector does not consume a waveform; it was "
            "registered against an ASR tensor, so the engine must feed it that instead"
        )

    def detect_streaming(
        self,
        waveform: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[VadState],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[VadState]]:
        """Incremental :meth:`detect` over a chunk, carrying ``state`` forward.

        Returns ``(probs, frame_lengths, state)``.  The default routes to the
        one-shot form, which is correct for a detector whose frames are
        independent (energy) and wrong for one that is not — hence
        ``VadSpec.stateful``, which a recurrent detector sets so the engine
        allocates and threads state instead.
        """
        probs, frame_lengths = self.detect(waveform, lengths)
        return probs, frame_lengths, state

    # -- ASR-derived detectors ----------------------------------------------

    def detect_from_asr(
        self, tensor: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The ASR's own per-frame output → probabilities on the encoder grid.

        What ``tensor`` is depends on the registered ``consumes``: ``(B, T, V)``
        CTC log-probs, ``(B, T)`` activity indicators, ``(B, T)`` CIF weights, or
        ``(B, V)`` prefill logits.  Returns the same pair :meth:`detect` does.
        """
        raise NotImplementedError(
            f"the {type(self).__name__} detector does not consume an ASR tensor; it "
            "was registered against a waveform"
        )

    # -- shared helpers ------------------------------------------------------

    @staticmethod
    def _mask_padding(probs: torch.Tensor, frame_lengths: torch.Tensor) -> torch.Tensor:
        """Zero every frame past a row's own length.

        A padded row's tail is whatever the batch's widest member made it, and a
        detector that leaves it alone reports the *padding* as speech — which
        then becomes a segment, and then a transcript request for silence.
        """
        if probs.ndim != 2:
            raise ValueError(f"probs must be (B, T), got {tuple(probs.shape)}")
        idx = torch.arange(probs.size(1), device=probs.device).unsqueeze(0)
        keep = idx < frame_lengths.to(probs.device).unsqueeze(1)
        return probs * keep.to(probs.dtype)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(kind={getattr(self, 'kind', '?')!r}, "
            f"spf={self._seconds_per_frame:.4f})"
        )


def as_rows(probs: torch.Tensor, frame_lengths: torch.Tensor) -> List[List[float]]:
    """``(B, T)`` device probabilities → one host list per row, in one transfer.

    A single ``.cpu()`` for the whole batch rather than one per row: at streaming
    cadence what a device→host copy costs is the host issuing it, and the batch
    is a few kilobytes either way.
    """
    if probs.numel() == 0:
        return [[] for _ in range(int(probs.size(0)))]
    host = probs.detach().to("cpu", dtype=torch.float32).tolist()
    lens = [int(n) for n in frame_lengths.detach().to("cpu").tolist()]
    return [row[: lens[i]] for i, row in enumerate(host)]
