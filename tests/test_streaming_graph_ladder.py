# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The streaming encoder-graph shape ladder (``oasr/engine/graph_cache.py``).

The streaming graph is keyed on ``(B_active, T_input, cache_t1_bucket)``.  Both
key axes used to be under-covered by the pre-warm, and each miss capture*s* a
graph on a live tick at ~30 ms:

* ``cache_t1`` was rounded to a flat 64 frames, which with the default
  ``num_left_chunks=-1`` makes that axis **unbounded** — a long stream reached a
  fresh bucket every 64 encoder frames and captured forever.  Measured on a 120 s
  stream: 48 captures over 191 ticks and a p99 of 33 ms against a 2.7 ms p50.
* the batch axis was pre-warmed at ``{1, max_batch_size}`` only, so every active
  width in between captured on its first appearance.

These tests pin the two properties that make the ladder pre-warmable: it is
**finite**, and it is **exactly** the set of buckets the runtime can ask for.
"""

from __future__ import annotations

import pytest
import torch

from oasr.engine.graph_cache import (
    CACHE_BUCKET_KNEE,
    cache_bucket_ladder,
    pick_cache_bucket,
    round_up_bucket,
)

_N_BLOCK = 64


class TestLadderShape:
    @pytest.mark.parametrize("capacity", [64, 512, 1024, 4096, 4984, 8192])
    def test_rungs_are_kernel_tile_multiples_within_capacity(self, capacity):
        ladder = cache_bucket_ladder(capacity)
        assert ladder, "ladder must not be empty"
        assert ladder == sorted(set(ladder)), "rungs must be sorted and unique"
        for rung in ladder:
            assert rung % _N_BLOCK == 0, f"{rung} is not an N_BLOCK multiple"
            # A rung *above* capacity is the out-of-bounds read the capacity
            # exists to prevent — the block table cannot address it and the
            # relative-position table cannot index it.
            assert rung <= capacity, f"rung {rung} exceeds capacity {capacity}"

    def test_it_is_finite_and_small(self):
        """The whole point: a flat 64-frame axis is unbounded, this one is not."""
        for capacity in (4096, 65536, 1 << 20):
            ladder = cache_bucket_ladder(capacity)
            assert len(ladder) < 40, f"{capacity} produced {len(ladder)} rungs"
        # ...and it grows logarithmically, not linearly, in the capacity.
        small = len(cache_bucket_ladder(4096))
        big = len(cache_bucket_ladder(4096 * 64))
        assert big - small <= 12, (small, big)

    def test_below_the_knee_it_is_still_flat_64(self):
        """Short streams keep the fine granularity; a coarse rung there would be
        a large *relative* over-read."""
        ladder = cache_bucket_ladder(4096)
        fine = [r for r in ladder if r <= CACHE_BUCKET_KNEE]
        assert fine == list(range(0, CACHE_BUCKET_KNEE + 1, _N_BLOCK))

    def test_growth_one_restores_the_legacy_flat_ladder(self):
        ladder = cache_bucket_ladder(1024, growth=1.0)
        assert ladder == list(range(0, 1025, _N_BLOCK))


class TestPickMatchesLadder:
    """The pre-warm captures the ladder; the runtime picks with this function.

    If the two ever disagree, the pre-warm covers shapes the runtime never asks
    for and misses the ones it does — which is the tail, silently back.
    """

    @pytest.mark.parametrize("capacity", [512, 1024, 4096])
    def test_every_reachable_length_maps_onto_a_prewarmed_rung(self, capacity):
        ladder = cache_bucket_ladder(capacity)
        rungs = set(ladder)
        for cache_t1 in range(0, capacity + 1):
            got = pick_cache_bucket(cache_t1, ladder)
            assert got in rungs, f"cache_t1={cache_t1} -> {got}, off the ladder"

    @pytest.mark.parametrize("capacity", [512, 4096])
    def test_a_bucket_is_never_shorter_than_the_cache(self, capacity):
        """Handing the kernel a ``host_seqlen_max`` below the real
        ``cache_seqlens`` would truncate a stream's attention history."""
        ladder = cache_bucket_ladder(capacity)
        for cache_t1 in range(0, capacity + 1):
            assert pick_cache_bucket(cache_t1, ladder) >= cache_t1

    def test_over_read_is_bounded_by_the_growth_ratio(self):
        """The trade for a finite ladder: rungs above the knee over-read by at
        most ``growth``, which measured ~4% of a replay at 1.5."""
        ladder = cache_bucket_ladder(8192, growth=1.5)
        for cache_t1 in range(CACHE_BUCKET_KNEE, 8192, 37):
            rung = pick_cache_bucket(cache_t1, ladder)
            assert rung <= cache_t1 * 1.5 + _N_BLOCK, (cache_t1, rung)

    def test_off_ladder_falls_back_to_flat_rounding(self):
        ladder = cache_bucket_ladder(512)
        assert pick_cache_bucket(9999, ladder) == round_up_bucket(9999)


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kw",
        [
            {"streaming_graph_cache_growth": 0.5},
            {"streaming_graph_max_shapes": 0},
            {"streaming_graph_batch_ladder": [0]},
            {"streaming_graph_batch_ladder": [1, 999]},
        ],
    )
    def test_rejects_incoherent_knobs(self, kw):
        from oasr.engine.config import EngineConfig

        with pytest.raises(ValueError):
            EngineConfig(ckpt_dir="/nonexistent", max_batch_size=32, **kw)

    def test_defaults(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/nonexistent")
        assert cfg.streaming_graph_cache_growth > 1.0
        assert cfg.streaming_graph_batch_ladder is None  # None == every width


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestPrewarmCoversTheRuntime:
    """End-to-end: a stream long enough to walk the whole cache ladder must not
    capture a single graph on a live tick."""

    def test_a_long_stream_captures_nothing(self, ckpt_dir):
        from oasr.engine import ASREngine, EngineConfig
        from oasr.engine.graph_cache import GraphedEncoderForward

        seen: list = []
        original = GraphedEncoderForward._capture

        def spy(self, B, T, bucket, *a, **kw):
            seen.append((B, T, bucket))
            return original(self, B, T, bucket, *a, **kw)

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device="cuda",
            dtype=torch.bfloat16,
            service_mode="streaming",
            max_batch_size=4,
            num_left_chunks=-1,  # the default: an unbounded cache axis
        )
        engine = ASREngine(cfg)
        try:
            GraphedEncoderForward._capture = spy
            seen.clear()
            samples = engine._input_processor.streaming_audio_chunk_samples
            # ~60 s of audio walks well past where the old ladder stopped.
            chunk = torch.zeros(samples, dtype=torch.float32)
            rid = engine.add_streaming_request(sample_rate=16000)
            n = int(60 * 16000 / samples)
            for j in range(n):
                engine.feed_chunk(rid, chunk, is_last=(j == n - 1))
            engine.run()
            torch.cuda.synchronize()
        finally:
            GraphedEncoderForward._capture = original
            engine.shutdown()

        assert not seen, (
            f"{len(seen)} graph(s) captured on a live tick — the pre-warm ladder "
            f"no longer covers what the runtime asks for: {seen[:5]}"
        )
