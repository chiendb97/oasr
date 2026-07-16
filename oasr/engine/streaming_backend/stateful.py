# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Stateful streaming backend (Zipformer-style encoders).

For encoders whose streaming cache is **per-layer recurrent state** owned by the
encoder itself (``streaming_kind == "stateful"``), rather than the engine's
paged-KV + slot-CNN model.  The backend threads one state list per request
through ``model.streaming_forward(chunk, lens, states)`` chunk by chunk.

Unlike :class:`~oasr.engine.streaming_backend.paged.PagedStreamingBackend`, there
is no shared block pool, no slot CNN cache, and no CUDA-graph capture: the cache
lives inside the per-request state tensors.  Streams are processed at ``B=1``
(each carries its own state); batching stateful streams by stacking states is a
future optimization.

Window geometry comes from the encoder (``streaming_chunk_frames``): the engine's
shared :class:`~oasr.engine.input_processor.InputProcessor` fills each request's
feature buffer, and this backend windows it the same way the paged backend does.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple

import torch

from ..request import Request
from .base import StreamingEncoderBackend, register_streaming_backend

if TYPE_CHECKING:
    from oasr.cache.types import CacheConfig
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig

logger = logging.getLogger(__name__)


@register_streaming_backend("stateful")
class StatefulStreamingBackend(StreamingEncoderBackend):
    """Per-request recurrent-state streaming runtime (Zipformer-style)."""

    streaming_kind: ClassVar[str] = "stateful"

    def __init__(
        self,
        model: "BaseAsrModel",
        config: "EngineConfig",
        cache_config: "CacheConfig",
        *,
        graph_pool: Optional[Tuple[int, int]] = None,
        consumes: str = "log_probs",
    ) -> None:
        self._model = model
        self._config = config
        self._device = torch.device(config.device)
        self._dtype = config.dtype
        # What the active decode strategy consumes: "log_probs" threads chunks
        # through ``model.streaming_forward`` (encoder + head); "hidden" calls
        # the encoder's own ``streaming_forward`` (raw hidden states) for
        # autoregressive families.  Same (out, out_lens, new_states) contract.
        self._chunk_forward = (
            model.encoder.streaming_forward if consumes == "hidden" else model.streaming_forward
        )
        # Per-request encoder streaming state (the encoder's own recurrent cache).
        self._states: Dict[int, List[torch.Tensor]] = {}

        # Window/stride: the encoder declares how many input frames it consumes
        # per streaming chunk.  Stateful encoders carry context in their state,
        # so windows are non-overlapping (stride == window).
        enc = model.encoder
        window = getattr(enc, "streaming_chunk_frames", None)
        if window is None:
            # Fallback: chunk_size encoder frames × total subsampling.
            window = int(config.chunk_size) * int(getattr(enc, "subsampling_rate", 1))
        self._window = int(window)

    # ------------------------------------------------------------------
    # Window geometry
    # ------------------------------------------------------------------

    @property
    def decoding_window(self) -> int:
        return self._window

    @property
    def stride(self) -> int:
        # Non-overlapping: the encoder state carries cross-chunk context.
        return self._window

    # ------------------------------------------------------------------
    # Per-request lifecycle
    # ------------------------------------------------------------------

    def allocate(self, request: Request) -> None:
        sid = request.stream_id
        assert sid is not None, "stream_id must be assigned before allocate"
        self._states[sid] = self._model.get_streaming_init_states(
            batch_size=1, device=self._device, dtype=self._dtype
        )
        # No paged context; mark allocated so the executor's lifecycle checks
        # treat this stream as admitted.
        request.stream_context = None

    def free(self, request: Request) -> None:
        sid = request.stream_id
        if sid is not None:
            self._states.pop(sid, None)

    # ------------------------------------------------------------------
    # Per-tick forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_step(self, requests: List[Request]) -> Dict[str, torch.Tensor]:
        """Run one streaming chunk per ready request (``B=1`` each).

        Slices ``window`` feature frames at ``feature_cursor``, threads the
        per-request state through ``model.streaming_forward``, advances the
        cursor by ``stride``, and returns ``{request_id: log_probs}``.
        """
        window = self._window
        stride = self._window
        results: Dict[str, torch.Tensor] = {}

        for req in requests:
            if req.feature_buffer is None:
                continue
            if not req.has_ready_encoder_chunk(window):
                continue
            sid = req.stream_id
            assert (
                sid is not None and sid in self._states
            ), "stateful stream must be allocated before forward_step"

            available = req.feature_frames - req.feature_cursor
            end = req.feature_cursor + min(window, available)
            chunk = req.feature_buffer[req.feature_cursor : end].unsqueeze(0)  # (1, T, F)
            chunk = chunk.to(device=self._device, dtype=self._dtype)
            lens = torch.tensor([chunk.size(1)], dtype=torch.int32, device=self._device)

            out, _out_lens, new_states = self._chunk_forward(chunk, lens, self._states[sid])
            self._states[sid] = new_states

            req.feature_cursor += stride
            req.offset += int(out.size(1))
            results[req.request_id] = out

        return results
