#!/usr/bin/env python3
"""Tests for the decoder-side KV cache manager (AR generation storage, K6).

Pure bookkeeping — runs on CPU with a CPU-device BlockPool.
"""

import pytest
import torch

from oasr.cache import DecoderKVCacheManager


@pytest.fixture()
def mgr():
    pool = DecoderKVCacheManager.build_pool(
        num_layers=2,
        n_kv_head=2,
        head_dim=8,
        block_tokens=4,
        max_num_blocks=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return DecoderKVCacheManager(pool)


class TestLifecycle:
    def test_create_free_roundtrip(self, mgr):
        total = mgr.pool.num_total_blocks
        mgr.create("r1", max_new_tokens=10)
        assert mgr.num_active() == 1
        assert mgr.seqlen("r1") == 0
        assert mgr.pool.num_free_blocks == total - 1  # first page pre-allocated
        mgr.free("r1")
        assert mgr.num_active() == 0
        assert mgr.pool.num_free_blocks == total

    def test_prefill_allocates_pages(self, mgr):
        mgr.create("r1", max_new_tokens=4, prefill_len=6)  # 6 tokens / 4 per page → 2
        assert mgr.seqlen("r1") == 6
        assert mgr.block_tables(["r1"]).shape[1] == 2

    def test_duplicate_create_rejected(self, mgr):
        mgr.create("r1", max_new_tokens=4)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create("r1", max_new_tokens=4)

    def test_free_is_idempotent(self, mgr):
        mgr.create("r1", max_new_tokens=4)
        mgr.free("r1")
        mgr.free("r1")  # no raise

    def test_free_all(self, mgr):
        total = mgr.pool.num_total_blocks
        for rid in ("a", "b", "c"):
            mgr.create(rid, max_new_tokens=4)
        mgr.free_all()
        assert mgr.num_active() == 0
        assert mgr.pool.num_free_blocks == total


class TestGrowth:
    def test_append_returns_write_positions(self, mgr):
        mgr.create("a", max_new_tokens=16)
        mgr.create("b", max_new_tokens=16, prefill_len=3)
        for step in range(5):
            pos = mgr.append_step(["a", "b"])
            assert pos == [step, 3 + step]
        assert mgr.cache_seqlens(["a", "b"]).tolist() == [5, 8]

    def test_page_boundary_allocates_block(self, mgr):
        mgr.create("a", max_new_tokens=16)
        free0 = mgr.pool.num_free_blocks
        for _ in range(4):  # fills page 0 exactly
            mgr.append_step(["a"])
        assert mgr.pool.num_free_blocks == free0
        mgr.append_step(["a"])  # token 5 crosses into page 1
        assert mgr.pool.num_free_blocks == free0 - 1
        assert mgr.block_tables(["a"]).shape == (1, 2)

    def test_no_eviction_growth_to_exhaustion(self, mgr):
        """AR decode grows monotonically — exhaustion raises, nothing evicts."""
        mgr.create("a", max_new_tokens=1000)
        with pytest.raises(RuntimeError, match="exhausted"):
            for _ in range(mgr.pool.num_total_blocks * 4 + 1):
                mgr.append_step(["a"])

    def test_block_tables_padded_per_batch(self, mgr):
        mgr.create("a", max_new_tokens=16, prefill_len=9)  # 3 pages
        mgr.create("b", max_new_tokens=16, prefill_len=1)  # 1 page
        table = mgr.block_tables(["a", "b"])
        assert table.shape == (2, 3)
        assert table.dtype == torch.int32
        # padding entries are 0 (never dereferenced past cache_seqlens)
        assert table[1, 1:].tolist() == [0, 0]

    def test_kv_view_shape(self, mgr):
        k, v = mgr.kv_view(0)
        assert k.shape == v.shape == (8, 4, 2, 8)  # (blocks, tokens, heads, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
