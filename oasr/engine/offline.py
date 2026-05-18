# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline (non-streaming) ASR transcription engine.

Thin wrapper over :class:`~oasr.engine.ASREngine` that defaults ``add_request``
to ``streaming=False`` so inputs flow through the dynamic-batching offline
path.  Kept as a separate class so existing call-sites continue to work.
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import torch

from .engine import ASREngine

logger = logging.getLogger(__name__)


class OfflineEngine(ASREngine):
    """Offline batch transcription engine.

    Inherits the full step loop, scheduler, and model runner from
    :class:`ASREngine`.  The only behavioural difference is that
    :meth:`transcribe` defaults to ``streaming=False`` so requests are
    length-bucketed and forwarded in a single padded pass rather than
    streaming chunk-by-chunk.

    Examples
    --------
    Single file::

        engine = OfflineEngine(EngineConfig(ckpt_dir="/path/to/ckpt"))
        text = engine.transcribe("audio.wav")

    Batch with length-aware bucketing (see ``length_bucket_ratio``)::

        texts = engine.transcribe(["a.wav", "b.wav", "c.wav"])
    """

    def transcribe(  # type: ignore[override]
        self,
        audio: Union[str, List[str], torch.Tensor, "np.ndarray", List[torch.Tensor]],
        sample_rate: int = 16000,
        streaming: bool = False,
    ) -> Union[str, List[str]]:
        """Transcribe one or more audio inputs (offline by default).

        See :meth:`ASREngine.transcribe` for parameter semantics; the only
        difference here is the default ``streaming=False``.
        """
        return super().transcribe(audio, sample_rate=sample_rate, streaming=streaming)
