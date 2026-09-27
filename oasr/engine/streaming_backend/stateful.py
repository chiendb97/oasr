# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Stateful streaming backend: per-request state lists (``streaming_kind == "stateful"``).

For an encoder that exposes the list API — ``get_streaming_init_states`` and
``streaming_forward(chunk, lens, states)`` — and declares nothing further.  The
backend threads one state list per request chunk by chunk.  An encoder whose
list is entirely fixed-extent and declared (``slot_state_specs``, Zipformer) runs
on the slot runtime instead
(:class:`~oasr.engine.streaming_backend.slot.SlotStreamingBackend`); this one
stays its parity oracle, bit-identical to it given the same streams.

Unlike :class:`~oasr.engine.streaming_backend.paged.PagedStreamingBackend`, there
is no shared block pool, no slot cache, and no CUDA-graph capture: the cache
lives inside the per-request state tensors.

**Batching**: when the encoder exposes ``stack_streaming_states`` /
``unstack_streaming_states`` (Zipformer does — icefall's per-kind batch dims),
ready streams with the same chunk length run as **one** ``B = N`` forward:
stack states → batched chunk forward → unstack states.  With a declared
``streaming_pad_value`` every window is full, so a pool batches completely;
without one, a stream's short final tail runs in its own group.  Encoders
without the stack/unstack surface keep the sequential ``B = 1`` path.

Window geometry comes from the encoder: ``streaming_chunk_frames`` is the stride
and ``streaming_window_frames`` (when the front-end needs lookahead) the window.
The engine's shared :class:`~oasr.engine.input_processor.InputProcessor` fills
each request's feature buffer, and this backend windows it the same way the paged
backend does.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple

import torch

from oasr.utils.staging import to_device

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
        # ``None`` in practice: this runtime allocates no paged pool
        # (``allocates_paged_pool = False``), so the engine builds no
        # ``CacheConfig`` for it.  Accepted for signature uniformity with the
        # registry, and ignored — the per-request recurrent state comes from the
        # encoder's own ``get_streaming_init_states``.
        cache_config: "Optional[CacheConfig]" = None,
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

        # Window/stride: the encoder declares how many input frames a stream
        # advances per chunk (``streaming_chunk_frames``, the stride) and, when
        # its front-end needs lookahead past the chunk, how many it reads
        # (``streaming_window_frames``; Zipformer's embed needs 13 more).  Without
        # the latter, windows are non-overlapping (stride == window).
        enc = model.encoder
        stride = getattr(enc, "streaming_chunk_frames", None)
        if stride is None:
            # Fallback: chunk_size encoder frames × total subsampling.
            stride = int(config.chunk_size) * int(getattr(enc, "subsampling_rate", 1))
        self._stride = int(stride)
        self._window = int(getattr(enc, "streaming_window_frames", None) or self._stride)
        #: Feature value a short final window is padded to a full one with, or
        #: ``None`` to forward it short.  An encoder that streams only whole
        #: chunks declares one (see ``ZipformerEncoder.streaming_pad_value``).
        self._pad_value: Optional[float] = getattr(enc, "streaming_pad_value", None)

    # ------------------------------------------------------------------
    # Window geometry
    # ------------------------------------------------------------------

    @property
    def decoding_window(self) -> int:
        return self._window

    @property
    def stride(self) -> int:
        # The encoder state carries cross-chunk context; windows overlap only by
        # the front-end's lookahead (``window - stride``).
        return self._stride

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

    # ``reset`` is the base class's ``free`` + ``allocate`` + ``offset = 0``, and
    # that is already exactly right here: ``allocate`` re-seeds the whole state
    # dict from the model's own initialiser and holds nothing else per stream, so
    # there is no in-place rewind to be cheaper than it.  Overriding it to write
    # the same two lines would only be a second copy to keep in step.

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

        # Group by this tick's chunk length (torch.stack needs uniform T).  With
        # a declared pad value every window is full, so this is one group.
        groups: Dict[int, List[Request]] = defaultdict(list)
        for req in ready:
            groups[self._chunk_len(req)].append(req)

        for t_chunk, reqs in groups.items():
            if len(reqs) == 1:
                results[reqs[0].request_id] = self._forward_one(reqs[0])
                continue
            chunks = torch.stack([self._window_of(r, t_chunk) for r in reqs]).to(
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

    def _chunk_len(self, req: Request) -> int:
        """Frames this stream's next forward sees: a full window, or its short tail."""
        available = req.feature_frames - req.feature_cursor
        if self._pad_value is not None:
            return self._window
        return min(self._window, available)

    def _window_of(self, req: Request, t_chunk: int) -> torch.Tensor:
        """``(t_chunk, F)`` features at the cursor, padded when the tail is short.

        Padding is icefall's streaming convention for an encoder that only
        streams whole chunks: the frames past the audio are a constant (its
        ``LOG_EPS``), counted in the chunk's length like real ones.
        """
        buf = req.feature_buffer
        chunk = buf[req.feature_cursor : req.feature_cursor + t_chunk]
        short = t_chunk - chunk.size(0)
        if short > 0:
            assert self._pad_value is not None
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, short), value=float(self._pad_value))
        return chunk

    def _forward_one(self, req: Request) -> torch.Tensor:
        """Single-stream ``B = 1`` chunk forward (fallback + singleton groups)."""
        t_chunk = self._chunk_len(req)
        chunk = self._window_of(req, t_chunk).unsqueeze(0)  # (1, T, F)
        chunk = chunk.to(device=self._device, dtype=self._dtype)
        lens = to_device([chunk.size(1)], dtype=torch.int32, device=self._device)

        sid = req.stream_id
        out, _out_lens, new_states = self._chunk_forward(chunk, lens, self._states[sid])
        self._states[sid] = new_states

        req.feature_cursor += self.stride
        req.offset += int(out.size(1))
        return out
