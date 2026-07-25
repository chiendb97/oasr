# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline micro-batch partition policies: count / frames / packing.

Each partitions a selected offline batch into encoder micro-batches and returns
``(chunks, orig_indices)`` (the index map restores arrival order after the
length sort).  Moved verbatim from the scheduler — behaviour is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, List, Optional

from ..request import Request
from .base import (
    PartitionPolicy,
    PartitionResult,
    register_partition_policy,
    request_cost_frames,
    snap_to_preferred,
)

if TYPE_CHECKING:
    from ..config import EngineConfig


@register_partition_policy("count")
class CountPartition(PartitionPolicy):
    """Partition by count, snapping to ``preferred_batch_size`` when set."""

    name: ClassVar[str] = "count"

    def split(self, batch: List[Request], config: "EngineConfig") -> PartitionResult:
        n = len(batch)
        mb = max(1, int(config.max_batch_size))
        pbs = config.preferred_batch_size

        # Fast path: a single micro-batch when nothing forces a split.
        if n <= mb and not pbs:
            return [list(batch)], None

        enumerated = sorted(enumerate(batch), key=lambda p: p[1].num_frames)
        ordered = [r for _, r in enumerated]
        orig_indices: Optional[List[int]] = [i for i, _ in enumerated]

        chunks: List[List[Request]] = []
        if pbs:
            idx = 0
            while idx < n:
                remaining = n - idx
                size = snap_to_preferred(min(remaining, mb), pbs)
                if size == 0:
                    size = remaining  # tail < min(preferred); one odd chunk.
                chunks.append(ordered[idx : idx + size])
                idx += size
        else:
            nchunks = (n + mb - 1) // mb
            base, rem = divmod(n, nchunks)
            idx = 0
            for i in range(nchunks):
                size = base + (1 if i < rem else 0)
                chunks.append(ordered[idx : idx + size])
                idx += size
        return chunks, orig_indices


@register_partition_policy("frames")
class FramePartition(PartitionPolicy):
    """Split into micro-batches bounded by a padded-frame budget."""

    name: ClassVar[str] = "frames"

    def split(self, batch: List[Request], config: "EngineConfig") -> PartitionResult:
        budget = config.max_batch_frames
        assert budget is not None
        mb = max(1, int(config.max_batch_size))

        enumerated = sorted(enumerate(batch), key=lambda p: p[1].num_frames)
        ordered = [r for _, r in enumerated]
        orig_indices: Optional[List[int]] = [i for i, _ in enumerated]

        chunks: List[List[Request]] = []
        cur: List[Request] = []
        cur_max = 0
        for r in ordered:
            rlen = request_cost_frames(r, config)
            new_max = max(cur_max, rlen)
            if cur and (new_max * (len(cur) + 1) > budget or len(cur) >= mb):
                chunks.append(cur)
                cur = [r]
                cur_max = rlen
            else:
                cur.append(r)
                cur_max = new_max
        if cur:
            chunks.append(cur)
        return chunks, orig_indices


@register_partition_policy("packing")
class PackingPartition(PartitionPolicy):
    """Group utterances into gapless packs bounded by a post-subsampling budget."""

    name: ClassVar[str] = "packing"

    def split(self, batch: List[Request], config: "EngineConfig") -> PartitionResult:
        enumerated = sorted(enumerate(batch), key=lambda p: p[1].num_frames)
        ordered = [r for _, r in enumerated]
        orig_indices: Optional[List[int]] = [i for i, _ in enumerated]

        budget = max(1, int(config.max_packed_frames))
        sr = max(1, int(config.subsampling_rate))
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
