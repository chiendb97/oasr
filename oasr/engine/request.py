# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Request representation and lifecycle state for the ASR engine."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from oasr.cache import StreamContext


class RequestState(enum.Enum):
    """Lifecycle state of an ASR request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class RequestOutput:
    """Output produced by the engine for a single request.

    Attributes
    ----------
    request_id : str
        Identifier matching the originating :class:`Request`.
    text : str
        Detokenized transcript (best hypothesis).
    tokens : List[List[int]]
        N-best token ID sequences (outer list = N-best, inner = tokens).
    scores : List[float], optional
        Log-probability scores per hypothesis.
    finished : bool
        ``True`` when decoding is complete; ``False`` for partial streaming
        results.
    """

    request_id: str
    text: str
    tokens: List[List[int]]
    scores: Optional[List[float]] = None
    finished: bool = False


class Request:
    """A single ASR inference request.

    Parameters
    ----------
    audio : str or Tensor or ndarray
        Either a file path (``str``), a raw waveform ``torch.Tensor`` of
        shape ``(num_samples,)`` or ``(1, num_samples)``, or a NumPy array.
    request_id : str, optional
        Unique identifier.  Auto-generated (UUID4 hex) if not provided.
    streaming : bool
        If ``True``, process via the chunk-by-chunk streaming path with
        paged attention cache.  If ``False``, use the single-pass offline path.
    sample_rate : int
        Sample rate of the audio in Hz.
    """

    def __init__(
        self,
        audio: Union[str, torch.Tensor, "np.ndarray"],
        request_id: Optional[str] = None,
        streaming: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        self.request_id: str = request_id or uuid.uuid4().hex
        self.audio: Union[str, torch.Tensor, "np.ndarray"] = audio
        self.streaming: bool = streaming
        self.sample_rate: int = sample_rate

        # Lifecycle state
        self.state: RequestState = RequestState.WAITING

        # Populated by InputProcessor
        self.features: Optional[torch.Tensor] = None          # (1, T, F)
        self.feature_lengths: Optional[torch.Tensor] = None   # (1,)
        self.chunks_remaining: Optional[List[torch.Tensor]] = None  # (1, W, F) each

        # Assigned by Scheduler
        self.stream_id: Optional[int] = None

        # Populated by ModelRunner (streaming only)
        self.stream_context: Optional[StreamContext] = None
        self.offset: int = 0  # encoder output frame offset

        # Final output
        self.output: Optional[RequestOutput] = None

    @property
    def is_finished(self) -> bool:
        return self.state == RequestState.FINISHED

    @property
    def has_chunks(self) -> bool:
        """True if there are pre-chunked feature windows still to process."""
        return bool(self.chunks_remaining)

    def __repr__(self) -> str:
        audio_repr = self.audio if isinstance(self.audio, str) else type(self.audio).__name__
        return (
            f"Request(id={self.request_id[:8]}, state={self.state.value}, "
            f"streaming={self.streaming}, audio={audio_repr})"
        )
