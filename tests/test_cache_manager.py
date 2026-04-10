# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ASR streaming cache manager (oasr/cache/).

Tests are grouped into:
  - BlockPool: allocation, free, exhaustion, tensor views
  - CnnCacheManager: lifecycle, update, shape validation
  - AttentionCacheManager: single/multi-chunk commit, eviction, free
  - CtcStateCacheManager: requires CUDA
  - StreamContext: full end-to-end lifecycle
  - Multi-stream isolation
"""

from __future__ import annotations

import pytest
import torch

from oasr.cache import (
    AttentionCacheManager,
    BlockPool,
    CacheConfig,
    CnnCacheManager,
    CtcStateCacheManager,
    StreamContext,
)
from oasr.cache.attention_cache import _StreamKVState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")


def make_config(
    *,
    num_layers: int = 4,
    n_kv_head: int = 2,
    head_dim: int = 8,
    hidden_dim: int = 16,
    kernel_size: int = 5,
    chunk_size: int = 4,
    num_left_chunks: int = -1,
    block_size_frames: int = 4,
    max_num_blocks: int = 64,
    device: torch.device = CPU,
    dtype: torch.dtype = torch.float32,
) -> CacheConfig:
    return CacheConfig(
        num_layers=num_layers,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        num_left_chunks=num_left_chunks,
        block_size_frames=block_size_frames,
        max_num_blocks=max_num_blocks,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# BlockPool tests
# ---------------------------------------------------------------------------


class TestBlockPool:
    def test_initial_free_count(self):
        cfg = make_config(max_num_blocks=8)
        pool = BlockPool(cfg)
        assert pool.num_free_blocks == 8
        assert pool.num_total_blocks == 8

    def test_allocate_decrements_free(self):
        cfg = make_config(max_num_blocks=8)
        pool = BlockPool(cfg)
        ids = pool.allocate(3)
        assert len(ids) == 3
        assert len(set(ids)) == 3  # unique IDs
        assert pool.num_free_blocks == 5

    def test_free_returns_to_pool(self):
        cfg = make_config(max_num_blocks=8)
        pool = BlockPool(cfg)
        ids = pool.allocate(4)
        pool.free(ids)
        assert pool.num_free_blocks == 8

    def test_reuse_freed_blocks(self):
        cfg = make_config(max_num_blocks=4)
        pool = BlockPool(cfg)
        ids1 = pool.allocate(4)
        pool.free(ids1)
        ids2 = pool.allocate(4)
        assert pool.num_free_blocks == 0
        assert set(ids2) == set(ids1)  # same IDs recycled

    def test_exhaustion_raises(self):
        cfg = make_config(max_num_blocks=2)
        pool = BlockPool(cfg)
        pool.allocate(2)
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.allocate(1)

    def test_allocate_zero_is_noop(self):
        cfg = make_config(max_num_blocks=4)
        pool = BlockPool(cfg)
        ids = pool.allocate(0)
        assert ids == []
        assert pool.num_free_blocks == 4

    def test_free_empty_list_is_noop(self):
        cfg = make_config(max_num_blocks=4)
        pool = BlockPool(cfg)
        pool.free([])
        assert pool.num_free_blocks == 4

    def test_block_view_write_and_read(self):
        cfg = make_config(max_num_blocks=4, block_size_frames=4, n_kv_head=2, head_dim=8)
        pool = BlockPool(cfg)
        (bid,) = pool.allocate(1)
        k_view, v_view = pool.get_kv_block_view(layer=0, block_id=bid)
        # Each view: (block_size_frames, n_kv_head, head_dim)
        assert k_view.shape == (4, 2, 8)
        assert v_view.shape == (4, 2, 8)
        # Write sentinel values and read back via another view call.
        k_view[:] = 3.0
        v_view[:] = 7.0
        k2, v2 = pool.get_kv_block_view(layer=0, block_id=bid)
        assert k2.allclose(torch.full_like(k2, 3.0))
        assert v2.allclose(torch.full_like(v2, 7.0))

    def test_gather_blocks_shape(self):
        cfg = make_config(max_num_blocks=8, block_size_frames=4, n_kv_head=2, head_dim=8)
        pool = BlockPool(cfg)
        ids = pool.allocate(3)
        k_flat, v_flat = pool.gather_kv_blocks(layer=0, block_ids=ids)
        # Each: (N*block_size_frames, n_kv_head, head_dim)
        assert k_flat.shape == (3 * 4, 2, 8)
        assert v_flat.shape == (3 * 4, 2, 8)

    def test_gather_empty_returns_zero_frames(self):
        cfg = make_config(max_num_blocks=4, n_kv_head=2, head_dim=8)
        pool = BlockPool(cfg)
        k_out, v_out = pool.gather_kv_blocks(layer=0, block_ids=[])
        assert k_out.shape[0] == 0
        assert v_out.shape[0] == 0

    def test_gather_preserves_values(self):
        cfg = make_config(max_num_blocks=4, block_size_frames=2, n_kv_head=1, head_dim=4)
        pool = BlockPool(cfg)
        ids = pool.allocate(2)
        k0, v0 = pool.get_kv_block_view(0, ids[0])
        k1, v1 = pool.get_kv_block_view(0, ids[1])
        k0[:] = 1.0; v0[:] = 10.0
        k1[:] = 2.0; v1[:] = 20.0
        k_out, v_out = pool.gather_kv_blocks(layer=0, block_ids=ids)
        # First 2 frames block 0, next 2 frames block 1.
        assert k_out[:2].allclose(torch.ones_like(k_out[:2]))
        assert k_out[2:].allclose(torch.full_like(k_out[2:], 2.0))
        assert v_out[:2].allclose(torch.full_like(v_out[:2], 10.0))
        assert v_out[2:].allclose(torch.full_like(v_out[2:], 20.0))

    def test_layer_independence(self):
        cfg = make_config(num_layers=2, max_num_blocks=4, block_size_frames=2, n_kv_head=1, head_dim=4)
        pool = BlockPool(cfg)
        (bid,) = pool.allocate(1)
        k0, v0 = pool.get_kv_block_view(0, bid)
        k1, v1 = pool.get_kv_block_view(1, bid)
        k0[:] = 1.0; v0[:] = 1.0
        k1[:] = 2.0; v1[:] = 2.0
        k0r, v0r = pool.get_kv_block_view(0, bid)
        k1r, v1r = pool.get_kv_block_view(1, bid)
        assert k0r.allclose(torch.ones(2, 1, 4))
        assert k1r.allclose(torch.full((2, 1, 4), 2.0))


# ---------------------------------------------------------------------------
# CnnCacheManager tests
# ---------------------------------------------------------------------------


class TestCnnCacheManager:
    def test_allocate_and_get_shape(self):
        cfg = make_config(num_layers=4, kernel_size=5, hidden_dim=16)
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0)
        cache = mgr.get_cache(0)
        assert cache.shape == (4, 1, 4, 16)  # (L, 1, K-1, D)

    def test_allocate_zero_initialized(self):
        cfg = make_config()
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0)
        assert mgr.get_cache(0).allclose(torch.zeros_like(mgr.get_cache(0)))

    def test_update_overwrites(self):
        cfg = make_config(num_layers=2, kernel_size=3, hidden_dim=8)
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0)
        new_val = torch.ones(2, 1, 2, 8)
        mgr.update(0, new_val)
        assert mgr.get_cache(0).allclose(new_val)

    def test_update_again_overwrites(self):
        cfg = make_config(num_layers=2, kernel_size=3, hidden_dim=8)
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0)
        mgr.update(0, torch.ones(2, 1, 2, 8))
        new_val = torch.full((2, 1, 2, 8), 3.0)
        mgr.update(0, new_val)
        assert mgr.get_cache(0).allclose(new_val)

    def test_update_shape_mismatch_raises(self):
        cfg = make_config(num_layers=2, kernel_size=3, hidden_dim=8)
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0)
        with pytest.raises(ValueError, match="shape mismatch"):
            mgr.update(0, torch.ones(2, 1, 3, 8))  # wrong cnn_cache_frames

    def test_double_allocate_raises(self):
        cfg = make_config()
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0)
        with pytest.raises(ValueError, match="already allocated"):
            mgr.allocate_stream(0)

    def test_free_stream(self):
        cfg = make_config()
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0)
        mgr.free_stream(0)
        with pytest.raises(KeyError):
            mgr.get_cache(0)

    def test_free_unallocated_raises(self):
        cfg = make_config()
        mgr = CnnCacheManager(cfg)
        with pytest.raises(KeyError):
            mgr.free_stream(99)

    def test_get_unallocated_raises(self):
        cfg = make_config()
        mgr = CnnCacheManager(cfg)
        with pytest.raises(KeyError):
            mgr.get_cache(99)

    def test_stream_isolation(self):
        cfg = make_config(num_layers=2, kernel_size=3, hidden_dim=8)
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0)
        mgr.allocate_stream(1)
        mgr.update(0, torch.ones(2, 1, 2, 8))
        # stream 1 should still be zeros
        assert mgr.get_cache(1).allclose(torch.zeros_like(mgr.get_cache(1)))


# ---------------------------------------------------------------------------
# AttentionCacheManager tests
# ---------------------------------------------------------------------------


class TestAttentionCacheManager:
    def _make_kv_chunk(self, cfg: CacheConfig, chunk_frames: int) -> torch.Tensor:
        """Random KV chunk with shape (L, H, chunk_frames, kv_last_dim)."""
        return torch.randn(
            cfg.num_layers, cfg.n_kv_head, chunk_frames, cfg.kv_last_dim,
            dtype=cfg.dtype, device=cfg.device,
        )

    def test_initial_cache_is_empty(self):
        cfg = make_config()
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        cache = mgr.get_stacked_cache(0)
        # Before any commit the cache should be (0,0,0,0) — matching forward_chunk default.
        assert cache.shape == (0, 0, 0, 0)

    def test_single_chunk_commit_shape(self):
        cfg = make_config(num_layers=2, n_kv_head=2, head_dim=4, chunk_size=4, block_size_frames=4)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        kv = self._make_kv_chunk(cfg, 4)
        mgr.commit(0, kv)
        out = mgr.get_stacked_cache(0)
        assert out.shape == (2, 2, 4, 8)  # (L, H, 4 frames, d_k*2)

    def test_single_chunk_commit_values(self):
        cfg = make_config(num_layers=1, n_kv_head=1, head_dim=4, chunk_size=2, block_size_frames=2)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        kv = torch.ones(1, 1, 2, 8)
        mgr.commit(0, kv)
        out = mgr.get_stacked_cache(0)
        assert out.allclose(torch.ones_like(out))

    def test_multi_chunk_grows_cache(self):
        cfg = make_config(num_layers=2, n_kv_head=2, head_dim=4, chunk_size=4,
                          block_size_frames=4, num_left_chunks=-1, max_num_blocks=32)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        for _ in range(3):
            kv = self._make_kv_chunk(cfg, 4)
            mgr.commit(0, kv)
        out = mgr.get_stacked_cache(0)
        assert out.shape[2] == 12  # 3 chunks * 4 frames each

    def test_eviction_with_num_left_chunks(self):
        # num_left_chunks=2, chunk_size=4, block_size_frames=4 => max_logical_blocks=2
        cfg = make_config(
            num_layers=1, n_kv_head=1, head_dim=4, chunk_size=4,
            block_size_frames=4, num_left_chunks=2, max_num_blocks=32,
        )
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        initial_free = pool.num_free_blocks

        # Commit 4 chunks: only last 2 should survive.
        for i in range(4):
            kv = torch.full((1, 1, 4, 8), float(i + 1))
            mgr.commit(0, kv)

        out = mgr.get_stacked_cache(0)
        assert out.shape[2] == 8  # 2 chunks * 4 frames

        # The oldest 2 blocks were evicted and returned to pool.
        # Now we should have: initial - 2 blocks used (one per surviving chunk).
        assert pool.num_free_blocks == initial_free - 2

        # Verify that the retained data is from the last 2 chunks.
        # chunk 3 => value 3.0, chunk 4 => value 4.0
        expected = torch.cat([
            torch.full((1, 1, 4, 8), 3.0),
            torch.full((1, 1, 4, 8), 4.0),
        ], dim=2)
        assert out.allclose(expected), f"out={out}\nexpected={expected}"

    def test_free_stream_returns_blocks(self):
        cfg = make_config(
            num_layers=1, n_kv_head=1, head_dim=4, chunk_size=4,
            block_size_frames=4, num_left_chunks=-1, max_num_blocks=16,
        )
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        initial = pool.num_free_blocks
        for _ in range(3):
            mgr.commit(0, self._make_kv_chunk(cfg, 4))
        assert pool.num_free_blocks == initial - 3
        mgr.free_stream(0)
        assert pool.num_free_blocks == initial

    def test_double_allocate_raises(self):
        cfg = make_config()
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        with pytest.raises(ValueError, match="already allocated"):
            mgr.allocate_stream(0)

    def test_commit_wrong_num_layers_raises(self):
        cfg = make_config(num_layers=4)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        wrong_kv = torch.ones(2, 2, 4, cfg.kv_last_dim)  # wrong num_layers
        with pytest.raises(ValueError, match="num_layers"):
            mgr.commit(0, wrong_kv)

    def test_commit_chunk_too_large_raises(self):
        cfg = make_config(block_size_frames=4)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        big_kv = torch.ones(cfg.num_layers, cfg.n_kv_head, 8, cfg.kv_last_dim)
        with pytest.raises(ValueError, match="block_size_frames"):
            mgr.commit(0, big_kv)

    def test_get_cache_view_per_layer(self):
        cfg = make_config(num_layers=3, n_kv_head=2, head_dim=4, chunk_size=4, block_size_frames=4)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0)
        # Commit with distinct per-layer values for verification.
        kv = torch.stack([
            torch.full((cfg.n_kv_head, cfg.chunk_size, cfg.kv_last_dim), float(l))
            for l in range(cfg.num_layers)
        ])
        mgr.commit(0, kv)
        for l in range(cfg.num_layers):
            view = mgr.get_cache_view(0, l)
            assert view.shape == (1, 2, 4, 8)
            assert view.allclose(torch.full_like(view, float(l)))


# ---------------------------------------------------------------------------
# CtcStateCacheManager tests (require CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCtcStateCacheManager:
    def test_allocate_and_get_decoder(self, device):
        from oasr import GpuDecoderConfig, GpuStreamingDecoder

        mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=4))
        mgr.allocate_stream(0, batch=1, vocab_size=100, device=device)
        dec = mgr.get_decoder(0)
        assert isinstance(dec, GpuStreamingDecoder)
        assert dec.step == 0

    def test_double_allocate_raises(self, device):
        mgr = CtcStateCacheManager()
        mgr.allocate_stream(0, batch=1, vocab_size=100, device=device)
        with pytest.raises(ValueError, match="already allocated"):
            mgr.allocate_stream(0, batch=1, vocab_size=100, device=device)

    def test_free_stream(self, device):
        mgr = CtcStateCacheManager()
        mgr.allocate_stream(0, batch=1, vocab_size=100, device=device)
        mgr.free_stream(0)
        with pytest.raises(KeyError):
            mgr.get_decoder(0)

    def test_get_unallocated_raises(self):
        mgr = CtcStateCacheManager()
        with pytest.raises(KeyError):
            mgr.get_decoder(99)


# ---------------------------------------------------------------------------
# StreamContext end-to-end lifecycle tests
# ---------------------------------------------------------------------------


class TestStreamContext:
    """Full lifecycle tests using CPU tensors (no CUDA required)."""

    def _setup(self, num_left_chunks: int = -1) -> tuple:
        cfg = make_config(
            num_layers=2, n_kv_head=2, head_dim=4, hidden_dim=8,
            kernel_size=3, chunk_size=4, block_size_frames=4,
            num_left_chunks=num_left_chunks, max_num_blocks=32,
        )
        pool = BlockPool(cfg)
        att_mgr = AttentionCacheManager(pool, cfg)
        cnn_mgr = CnnCacheManager(cfg)
        return cfg, pool, att_mgr, cnn_mgr

    def _make_stream(self, sid: int, att_mgr, cnn_mgr) -> StreamContext:
        # CtcStateCacheManager skipped for CPU tests; pass a stub.
        from unittest.mock import MagicMock
        ctc_mgr = MagicMock()
        ctc_mgr.free_stream = MagicMock()
        att_mgr.allocate_stream(sid)
        cnn_mgr.allocate_stream(sid)
        return StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)

    def test_initial_att_cache_is_empty_tensor(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        att = ctx.get_att_cache()
        assert att.shape == (0, 0, 0, 0)

    def test_initial_cnn_cache_is_zero(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        cnn = ctx.get_cnn_cache()
        assert cnn.shape == (2, 1, 2, 8)  # (L, 1, K-1, D)
        assert cnn.allclose(torch.zeros_like(cnn))

    def test_commit_and_read_att(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        kv = torch.full((2, 2, 4, 8), 5.0)
        cnn = torch.zeros(2, 1, 2, 8)
        ctx.commit_chunk(kv, cnn)
        att = ctx.get_att_cache()
        assert att.shape == (2, 2, 4, 8)
        assert att.allclose(torch.full_like(att, 5.0))

    def test_commit_and_read_cnn(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        kv = torch.zeros(2, 2, 4, 8)
        new_cnn = torch.full((2, 1, 2, 8), 3.0)
        ctx.commit_chunk(kv, new_cnn)
        assert ctx.get_cnn_cache().allclose(torch.full_like(new_cnn, 3.0))

    def test_free_returns_pool_blocks(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        initial = pool.num_free_blocks
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        for _ in range(3):
            ctx.commit_chunk(torch.ones(2, 2, 4, 8), torch.zeros(2, 1, 2, 8))
        ctx.free()
        assert pool.num_free_blocks == initial

    def test_stream_id_property(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(42, att_mgr, cnn_mgr)
        assert ctx.stream_id == 42


# ---------------------------------------------------------------------------
# Multi-stream isolation tests
# ---------------------------------------------------------------------------


class TestMultiStreamIsolation:
    def test_streams_are_independent(self):
        cfg = make_config(
            num_layers=1, n_kv_head=1, head_dim=4, hidden_dim=8,
            kernel_size=3, chunk_size=4, block_size_frames=4,
            max_num_blocks=32,
        )
        pool = BlockPool(cfg)
        att_mgr = AttentionCacheManager(pool, cfg)
        cnn_mgr = CnnCacheManager(cfg)

        for sid in range(4):
            att_mgr.allocate_stream(sid)
            cnn_mgr.allocate_stream(sid)

        # Each stream commits a distinct constant value.
        for sid in range(4):
            kv = torch.full((1, 1, 4, 8), float(sid + 1))
            cnn = torch.full((1, 1, 2, 8), float(sid + 1))
            att_mgr.commit(sid, kv)
            cnn_mgr.update(sid, cnn)

        for sid in range(4):
            att = att_mgr.get_stacked_cache(sid)
            cnn = cnn_mgr.get_cache(sid)
            expected_val = float(sid + 1)
            assert att.allclose(torch.full_like(att, expected_val)), f"sid={sid}"
            assert cnn.allclose(torch.full_like(cnn, expected_val)), f"sid={sid}"

    def test_partial_free_leaves_others_intact(self):
        cfg = make_config(
            num_layers=1, n_kv_head=1, head_dim=4, hidden_dim=8,
            kernel_size=3, chunk_size=4, block_size_frames=4,
            max_num_blocks=32,
        )
        pool = BlockPool(cfg)
        att_mgr = AttentionCacheManager(pool, cfg)
        cnn_mgr = CnnCacheManager(cfg)

        for sid in [0, 1, 2, 3]:
            att_mgr.allocate_stream(sid)
            cnn_mgr.allocate_stream(sid)
            kv = torch.full((1, 1, 4, 8), float(sid))
            cnn = torch.full((1, 1, 2, 8), float(sid))
            att_mgr.commit(sid, kv)
            cnn_mgr.update(sid, cnn)

        # Free streams 1 and 3.
        att_mgr.free_stream(1)
        cnn_mgr.free_stream(1)
        att_mgr.free_stream(3)
        cnn_mgr.free_stream(3)

        # Streams 0 and 2 should be unaffected.
        for sid in [0, 2]:
            att = att_mgr.get_stacked_cache(sid)
            cnn = cnn_mgr.get_cache(sid)
            expected = float(sid)
            assert att.allclose(torch.full_like(att, expected)), f"sid={sid}"
            assert cnn.allclose(torch.full_like(cnn, expected)), f"sid={sid}"

    def test_pool_accounting_after_partial_free(self):
        cfg = make_config(
            num_layers=1, n_kv_head=1, head_dim=4, hidden_dim=8,
            kernel_size=3, chunk_size=4, block_size_frames=4,
            num_left_chunks=-1, max_num_blocks=32,
        )
        pool = BlockPool(cfg)
        att_mgr = AttentionCacheManager(pool, cfg)
        cnn_mgr = CnnCacheManager(cfg)
        initial = pool.num_free_blocks  # 32

        for sid in range(4):
            att_mgr.allocate_stream(sid)
            cnn_mgr.allocate_stream(sid)
            att_mgr.commit(sid, torch.zeros(1, 1, 4, 8))  # uses 1 block each

        assert pool.num_free_blocks == initial - 4

        att_mgr.free_stream(0)
        att_mgr.free_stream(2)
        assert pool.num_free_blocks == initial - 2
