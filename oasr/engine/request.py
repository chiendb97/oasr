# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Request representation and lifecycle state for the ASR engine."""

from __future__ import annotations

import enum
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Union

import numpy as np
import torch

from oasr.cache import StreamContext


class RequestState(enum.Enum):
    """Lifecycle state of an ASR request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


# Default priority level (lower value = higher priority).
# Streaming requests default to this; offline can be bumped lower-priority.
DEFAULT_PRIORITY = 0


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
        priority: int = DEFAULT_PRIORITY,
    ) -> None:
        self.request_id: str = request_id or uuid.uuid4().hex
        self.audio: Union[str, torch.Tensor, "np.ndarray"] = audio
        self.streaming: bool = streaming
        self.sample_rate: int = sample_rate
        self.priority: int = priority
        self.arrival_time: float = time.monotonic()

        # Lifecycle state
        self.state: RequestState = RequestState.WAITING

        # Populated by InputProcessor
        self.features: Optional[torch.Tensor] = None          # (1, T, F)
        self.feature_lengths: Optional[torch.Tensor] = None   # (1,)
        # Offline-path raw waveform (CPU, scaled) held between ``add_request`` and
        # the batched ``collate_offline`` call.  Released once features are built.
        self.waveform: Optional[torch.Tensor] = None  # (T_samples,) CPU float32
        # Number of feature frames.  For offline this starts as a cheap
        # sample-count-derived estimate so the scheduler can bucket before
        # features are extracted, and is overwritten with the exact value
        # after the batched extraction runs.
        self.num_frames: int = 0

        # --- Streaming audio-chunk state (all populated by InputProcessor) ---
        # Queue of audio-sample chunks awaiting feature extraction.  Each
        # element is a CPU float32 1-D tensor of raw samples.  Streaming
        # ingests strictly left-to-right from this queue; the scheduler never
        # reaches ahead of the currently-enqueued chunks (no future audio).
        self.audio_chunks: Optional[Deque[torch.Tensor]] = None
        # Residual samples from the last fbank call — the suffix of the
        # combined (tail + new) waveform that didn't fit in a whole frame.
        # They get prepended to the next audio chunk so frame boundaries
        # stay aligned across streaming invocations.
        self.audio_tail: Optional[torch.Tensor] = None
        # Device-side feature ring.  Grown by extraction; encoder chunks are
        # sliced from ``features[feature_cursor : feature_cursor + window]``.
        self.feature_buffer: Optional[torch.Tensor] = None
        # Number of valid feature frames currently in ``feature_buffer``.
        self.feature_frames: int = 0
        # Feature-frame index of the next encoder chunk's start.
        self.feature_cursor: int = 0
        # Flips to True when the final audio chunk has been enqueued (no
        # more audio will arrive).  Triggers fbank flush + last-window forward.
        self.audio_final: bool = False

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
    def has_pending_audio(self) -> bool:
        """True if audio samples still need to be turned into features."""
        return bool(self.audio_chunks) or (
            self.audio_final and self.audio_tail is not None
            and self.audio_tail.numel() > 0
        )

    def has_ready_encoder_chunk(self, window: int) -> bool:
        """True if ``feature_buffer`` holds enough frames for a full window.

        At final-flush time (no more audio coming) we emit whatever remains,
        even if it's shorter than ``window``.
        """
        available = self.feature_frames - self.feature_cursor
        if available >= window:
            return True
        if self.audio_final and not self.audio_chunks and available > 0:
            return True
        return False

    @property
    def waited_for(self) -> float:
        """Seconds spent in the waiting queue (monotonic clock)."""
        return time.monotonic() - self.arrival_time

    def __repr__(self) -> str:
        audio_repr = self.audio if isinstance(self.audio, str) else type(self.audio).__name__
        return (
            f"Request(id={self.request_id[:8]}, state={self.state.value}, "
            f"streaming={self.streaming}, audio={audio_repr})"
        )
