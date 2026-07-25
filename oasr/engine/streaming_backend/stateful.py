# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Stateful streaming backend (Zipformer-style encoders).

For encoders whose streaming cache is **per-layer recurrent state** owned by the
encoder itself (``streaming_kind == "stateful"``), rather than the engine's
paged-KV + slot-CNN model.  The backend threads one state list per request
through ``model.streaming_forward(chunk, lens, states)`` chunk by chunk.

Unlike :class:`~oasr.engine.streaming_backend.paged.PagedStreamingBackend`, there
is no shared block pool, no slot CNN cache, and no CUDA-graph capture: the cache
lives inside the per-request state tensors.

**Batching**: when the encoder exposes ``stack_streaming_states`` /
``unstack_streaming_states`` (Zipformer does — icefall's per-kind batch dims),
ready streams with the same chunk length run as **one** ``B = N`` forward:
stack states → batched chunk forward → unstack states.  Full windows all share
one length, so steady-state pools batch completely; a stream's final partial
tail runs in its own (usually singleton) group.  Encoders without the
stack/unstack surface keep the sequential ``B = 1`` path.

Window geometry comes from the encoder (``streaming_chunk_frames``): the engine's
shared :class:`~oasr.engine.input_processor.InputProcessor` fills each request's
feature buffer, and this backend windows it the same way the paged backend does.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
        # Batched-state support: the encoder declares how its state tensors
        # stack along batch (per-kind batch dims).  Absent → sequential B=1.
        self._stack = getattr(model.encoder, "stack_streaming_states", None)
        self._unstack = getattr(model.encoder, "unstack_streaming_states", None)

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
        """Advance every ready stream one chunk; return ``{request_id: out}``.

        Ready streams are grouped by chunk length (full windows all share
        ``self._window``; a stream's final partial tail is shorter) and each
        group runs as **one** batched forward when the encoder supports state
        stacking — otherwise streams run sequentially at ``B = 1``.
        """
        window = self._window
        ready: List[Request] = []
        for req in requests:
            if req.feature_buffer is None or not req.has_ready_encoder_chunk(window):
                continue
            sid = req.stream_id
            assert (
                sid is not None and sid in self._states
            ), "stateful stream must be allocated before forward_step"
            ready.append(req)

        results: Dict[str, torch.Tensor] = {}
        if not ready:
            return results

        if self._stack is None or self._unstack is None:
            for req in ready:
                results[req.request_id] = self._forward_one(req)
            return results

        # Group by this tick's chunk length (torch.stack needs uniform T).
        groups: Dict[int, List[Request]] = defaultdict(list)
        for req in ready:
            available = req.feature_frames - req.feature_cursor
            groups[min(window, available)].append(req)

        for t_chunk, reqs in groups.items():
            if len(reqs) == 1:
                results[reqs[0].request_id] = self._forward_one(reqs[0])
                continue
            chunks = torch.stack(
                [r.feature_buffer[r.feature_cursor : r.feature_cursor + t_chunk] for r in reqs]
            ).to(
                device=self._device, dtype=self._dtype
            )  # (B, T, F)
            lens = torch.full((len(reqs),), t_chunk, dtype=torch.int32, device=self._device)
            states = self._stack([self._states[r.stream_id] for r in reqs])
            out, _out_lens, new_states = self._chunk_forward(chunks, lens, states)
            for i, (req, per_states) in enumerate(zip(reqs, self._unstack(new_states))):
                self._states[req.stream_id] = per_states
                req.feature_cursor += self.stride
                req.offset += int(out.size(1))
                results[req.request_id] = out[i : i + 1]

        return results

    def _forward_one(self, req: Request) -> torch.Tensor:
        """Single-stream ``B = 1`` chunk forward (fallback + singleton groups)."""
        window = self._window
        available = req.feature_frames - req.feature_cursor
        end = req.feature_cursor + min(window, available)
        chunk = req.feature_buffer[req.feature_cursor : end].unsqueeze(0)  # (1, T, F)
        chunk = chunk.to(device=self._device, dtype=self._dtype)
        lens = torch.tensor([chunk.size(1)], dtype=torch.int32, device=self._device)

        sid = req.stream_id
        out, _out_lens, new_states = self._chunk_forward(chunk, lens, self._states[sid])
        self._states[sid] = new_states

        req.feature_cursor += self.stride
        req.offset += int(out.size(1))
        return out
