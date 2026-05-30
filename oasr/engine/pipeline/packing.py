# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Sequence-packing offline pipeline.

A thin :class:`OfflinePipeline` subclass that groups whole utterances into
packs bounded by a **post-subsampling** token budget (``max_packed_frames``)
and runs each pack through the packing encoder forward
(:meth:`ModelRunner.forward_offline_packed`).  All the producer/drain/collate
machinery — including GPU fbank, cross-step overlap, and original-order
restoration — is reused unchanged; only micro-batch formation and the GPU
forward call differ.

Each micro-batch produced by :meth:`_split_chunks` is exactly one packed row,
so the encoder concatenates its whole ``(B, T, F)`` micro-batch into a single
gapless sequence.  The ``(B, T_out, V)`` output shape is identical to the
padded path, so CTC decode + finalisation are inherited verbatim.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple

import torch

from ..request import Request, RequestOutput, RequestState
from .offline import OfflinePipeline


class PackingPipeline(OfflinePipeline):
    """Offline pipeline that packs utterances into one encoder forward.

    Two flavours sharing the same packed forward + producer/drain machinery:

    * ``use_varlen=True`` (this class, sequence-packing mode) groups by a
      **summed** post-subsampling budget and runs the gapless **varlen**
      attention — zero attention padding.
    * ``use_varlen=False`` (:class:`LengthBucketPipeline`, length-bucketing
      mode) groups by a **padded** frame cap and runs the batched-per-segment
      dense attention.
    """

    streaming: ClassVar[bool] = False

    def __init__(
        self,
        *,
        max_packed_frames: int,
        subsampling_rate: int = 4,
        use_varlen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_packed_frames = max(1, int(max_packed_frames))
        self._subsampling_rate = max(1, int(subsampling_rate))
        self._use_varlen = bool(use_varlen)

    # ------------------------------------------------------------------
    # Micro-batching — one chunk == one packed row
    # ------------------------------------------------------------------

    def _split_chunks(
        self, batch: List[Request]
    ) -> Tuple[List[List[Request]], Optional[List[int]]]:
        """Group utterances into packs bounded by a post-subsampling budget.

        Length-sorts (when enabled) then greedily fills a pack until the summed
        estimated post-subsampling length (``num_frames // subsampling_rate``)
        would exceed ``max_packed_frames``.  A single oversized utterance ships
        as its own pack.  ``orig_indices`` threads through the inherited
        original-order restoration in :meth:`OfflinePipeline.run`.
        """
        if self._sort_by_length:
            enumerated = sorted(enumerate(batch), key=lambda p: p[1].num_frames)
            ordered = [r for _, r in enumerated]
            orig_indices: Optional[List[int]] = [i for i, _ in enumerated]
        else:
            ordered = list(batch)
            orig_indices = None

        budget = self._max_packed_frames
        sr = self._subsampling_rate
        chunks: List[List[Request]] = []
        cur: List[Request] = []
        cur_sum = 0
        for r in ordered:
            tlen = max(1, int(r.num_frames) // sr)
            if cur and cur_sum + tlen > budget:
                chunks.append(cur)
                cur = [r]
                cur_sum = tlen
            else:
                cur.append(r)
                cur_sum += tlen
        if cur:
            chunks.append(cur)
        return chunks, orig_indices

    # ------------------------------------------------------------------
    # GPU stage — packed encoder forward
    # ------------------------------------------------------------------

    def _gpu_stage(
        self,
        chunk: List[Request],
        features: torch.Tensor,
        lengths: torch.Tensor,
        event: Optional[torch.cuda.Event],
    ) -> List[RequestOutput]:
        """Packed forward + CTC decode + finalise on the default stream."""
        if event is not None:
            event.wait(torch.cuda.current_stream(self._device))

        log_probs, output_lengths = self._mr.forward_offline_packed(
            features, lengths, use_varlen=self._use_varlen,
        )
        outputs = self._op.decode_offline(log_probs, output_lengths)

        for req, out in zip(chunk, outputs):
            out.request_id = req.request_id
            out.finished = True
            req.output = out
            req.state = RequestState.FINISHED
        return outputs


class LengthBucketPipeline(PackingPipeline):
    """Length-bucketing offline pipeline (batched-per-segment dense backend).

    Reuses :class:`PackingPipeline`'s packed encoder forward (gapless FFN +
    per-segment isolation, bit-exact to ``B=1``) but with the **dense**
    attention backend (``use_varlen=False``) and the **padded-frame** grouping
    inherited from :class:`OfflinePipeline` (``max_batch_frames`` caps
    ``max_len * count`` per bucket) instead of the summed packing budget.
    """

    def __init__(
        self,
        *,
        max_batch_frames: Optional[int],
        subsampling_rate: int = 4,
        **kwargs,
    ) -> None:
        # ``max_packed_frames`` is unused in dense mode (grouping is padded-cap
        # via ``max_batch_frames``); pass a dummy so the parent accepts it.
        super().__init__(
            max_packed_frames=1,
            subsampling_rate=subsampling_rate,
            use_varlen=False,
            **kwargs,
        )
        self._max_batch_frames = max_batch_frames

    def _split_chunks(
        self, batch: List[Request]
    ) -> Tuple[List[List[Request]], Optional[List[int]]]:
        """Group by padded-frame cap, bypassing the summed packing budget.

        Delegates to :meth:`OfflinePipeline._split_chunks` (grandparent), which
        routes to ``_split_by_frames`` when ``max_batch_frames`` is set and
        falls back to count-based micro-batching otherwise.
        """
        return super(PackingPipeline, self)._split_chunks(batch)
