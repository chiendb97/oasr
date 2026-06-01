# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``OfflinePipeline._split_chunks`` with preferred batch sizes.

Exercises the pure-Python splitter in isolation by instantiating the pipeline
with stub IO objects; no GPU / model required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch

from oasr.engine.pipeline import OfflinePipeline
from oasr.engine.request import Request


def _make_pipeline(preferred_sizes: Optional[Sequence[int]], mb: int = 8) -> OfflinePipeline:
    """Build a pipeline with stub IO — only ``_split_chunks`` is exercised."""
    inp = SimpleNamespace(_config=SimpleNamespace(dtype=torch.float32))
    return OfflinePipeline(
        scheduler=SimpleNamespace(),
        input_processor=inp,
        model_runner=SimpleNamespace(),
        output_processor=SimpleNamespace(),
        micro_batch_size=mb,
        device=torch.device("cpu"),
        preferred_sizes=preferred_sizes,
    )


def _make_requests(num_frames_list: List[int]) -> List[Request]:
    out = []
    for n in num_frames_list:
        req = Request("audio.wav", streaming=False)
        req.num_frames = n
        out.append(req)
    return out


def _chunk_sizes(chunks):
    return [len(c) for c in chunks]


class TestSplitChunksPreferred:
    def test_greedy_peel_to_largest_preferred(self):
        # 11 requests, preferred [4, 8], mb=8 → [8, 3] (tail < min preferred)
        pipe = _make_pipeline(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(11)])
        chunks, _ = pipe._split_chunks(reqs)
        assert _chunk_sizes(chunks) == [8, 3]

    def test_exact_multiple_no_tail(self):
        pipe = _make_pipeline(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(16)])
        chunks, _ = pipe._split_chunks(reqs)
        assert _chunk_sizes(chunks) == [8, 8]

    def test_smaller_than_max_preferred_picks_lower(self):
        # 7 requests, preferred [4, 8] — first chunk snaps to 4, tail = 3
        pipe = _make_pipeline(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(7)])
        chunks, _ = pipe._split_chunks(reqs)
        assert _chunk_sizes(chunks) == [4, 3]

    def test_micro_batch_caps_chunk_size(self):
        # mb=4 caps the chunk size even though preferred allows 8.
        pipe = _make_pipeline(preferred_sizes=[4, 8], mb=4)
        reqs = _make_requests([i * 10 for i in range(12)])
        chunks, _ = pipe._split_chunks(reqs)
        assert _chunk_sizes(chunks) == [4, 4, 4]


class TestSplitChunksLegacy:
    def test_balanced_split_when_pbs_none(self):
        # 11 requests, mb=8 — legacy balance picks 2 chunks ≈ [6, 5].
        pipe = _make_pipeline(preferred_sizes=None, mb=8)
        reqs = _make_requests([i * 10 for i in range(11)])
        chunks, _ = pipe._split_chunks(reqs)
        sizes = _chunk_sizes(chunks)
        assert sum(sizes) == 11
        # Balance keeps chunks within 1 of each other.
        assert max(sizes) - min(sizes) <= 1

    def test_single_chunk_when_n_le_mb(self):
        pipe = _make_pipeline(preferred_sizes=None, mb=8)
        reqs = _make_requests([10] * 5)
        chunks, orig = pipe._split_chunks(reqs)
        assert _chunk_sizes(chunks) == [5]
        assert orig is None


class TestSortByLength:
    def test_chunks_are_length_sorted(self):
        # Mixed lengths; chunks should land in ascending num_frames order.
        pipe = _make_pipeline(preferred_sizes=[4], mb=8)
        reqs = _make_requests([100, 10, 50, 200, 80, 30, 60, 20])
        chunks, orig = pipe._split_chunks(reqs)
        # 8 requests, preferred=[4] → [4, 4].
        assert _chunk_sizes(chunks) == [4, 4]
        # Within each chunk, num_frames is non-decreasing; across chunks
        # the first chunk holds the smallest 4.
        first_lens = sorted(r.num_frames for r in chunks[0])
        second_lens = sorted(r.num_frames for r in chunks[1])
        assert first_lens == [10, 20, 30, 50]
        assert second_lens == [60, 80, 100, 200]
        assert orig is not None
