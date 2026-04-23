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
from .request import Request

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

    Streaming simulation (chunk-by-chunk, no real-time constraint)::

        texts = engine.transcribe_streaming("long_audio.wav", chunk_size=16)
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

    # ------------------------------------------------------------------
    # Legacy streaming-simulation helper
    # ------------------------------------------------------------------

    def transcribe_streaming(
        self,
        audio: Union[str, torch.Tensor, "np.ndarray"],
        chunk_size: int = None,  # type: ignore[assignment]
    ) -> str:
        """Simulate streaming via ``model.forward_chunk_by_chunk`` (no cache).

        Kept for backward compatibility and accuracy-benchmarking use cases
        that need deterministic chunk-by-chunk decoding without paged KV
        cache.  For real concurrent streaming use
        :meth:`ASREngine.add_request(..., streaming=True)` instead.

        Parameters
        ----------
        audio : str, Tensor, or ndarray
            Audio to transcribe.
        chunk_size : int, optional
            Encoder output frames per chunk.  Defaults to
            ``config.chunk_size``.
        """
        chunk_size = chunk_size or self._config.chunk_size

        req = Request(audio)
        features, _ = self._input_processor.process_offline([req])

        with torch.no_grad():
            log_probs = self._model.forward_chunk_by_chunk(
                features,
                decoding_chunk_size=chunk_size,
                num_decoding_left_chunks=self._config.num_left_chunks,
            )  # (1, T_out, V)

        T_out = log_probs.size(1)
        lengths = torch.tensor([T_out], dtype=torch.int32, device=self._device)
        outputs = self._output_processor.decode_offline(log_probs, lengths)
        return outputs[0].text if outputs else ""
