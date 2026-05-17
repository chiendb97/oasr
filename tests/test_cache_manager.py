# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ASR streaming cache manager (oasr/cache/).

Tests are grouped into:
  - BlockPool: allocation, free, exhaustion, tensor views
  - CnnCacheManager: lifecycle, update, shape validation
  - AttentionCacheManager: paged prepare/commit, eviction, free
  - CtcStateCacheManager: requires CUDA
  - StreamContext: paged-mode lifecycle
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
        mgr.allocate_stream(0, slot_id=0)
        cache = mgr.get_cache(0)
        assert cache.shape == (4, 1, 4, 16)  # (L, 1, K-1, D)

    def test_allocate_zero_initialized(self):
        cfg = make_config()
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0, slot_id=0)
        assert mgr.get_cache(0).allclose(torch.zeros_like(mgr.get_cache(0)))

    def test_update_overwrites(self):
        cfg = make_config(num_layers=2, kernel_size=3, hidden_dim=8)
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0, slot_id=0)
        new_val = torch.ones(2, 1, 2, 8)
        mgr.update(0, new_val)
        assert mgr.get_cache(0).allclose(new_val)

    def test_update_again_overwrites(self):
        cfg = make_config(num_layers=2, kernel_size=3, hidden_dim=8)
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0, slot_id=0)
        mgr.update(0, torch.ones(2, 1, 2, 8))
        new_val = torch.full((2, 1, 2, 8), 3.0)
        mgr.update(0, new_val)
        assert mgr.get_cache(0).allclose(new_val)

    def test_update_shape_mismatch_raises(self):
        cfg = make_config(num_layers=2, kernel_size=3, hidden_dim=8)
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0, slot_id=0)
        with pytest.raises(ValueError, match="shape mismatch"):
            mgr.update(0, torch.ones(2, 1, 3, 8))  # wrong cnn_cache_frames

    def test_double_allocate_raises(self):
        cfg = make_config()
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0, slot_id=0)
        with pytest.raises(ValueError, match="already allocated"):
            mgr.allocate_stream(0, slot_id=1)

    def test_free_stream(self):
        cfg = make_config()
        mgr = CnnCacheManager(cfg)
        mgr.allocate_stream(0, slot_id=0)
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
        mgr.allocate_stream(0, slot_id=0)
        mgr.allocate_stream(1, slot_id=1)
        mgr.update(0, torch.ones(2, 1, 2, 8))
        # stream 1 should still be zeros
        assert mgr.get_cache(1).allclose(torch.zeros_like(mgr.get_cache(1)))


# ---------------------------------------------------------------------------
# AttentionCacheManager tests
# ---------------------------------------------------------------------------


class TestAttentionCacheManager:
    def test_double_allocate_raises(self):
        cfg = make_config()
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        with pytest.raises(ValueError, match="already allocated"):
            mgr.allocate_stream(0, slot_id=1)

    def test_prepare_chunk_allocates_block_and_updates_table(self):
        cfg = make_config(num_layers=2, max_num_blocks=8)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        initial_free = pool.num_free_blocks
        mgr.prepare_chunk(0)
        assert pool.num_free_blocks == initial_free - 1
        bt, cs = mgr.get_paged_state_views(0)
        assert bt.shape == (1, cfg.max_blocks_per_seq)
        assert cs.shape == (1,)
        # Block id stored at logical index 0.
        assert int(bt[0, 0].item()) >= 0
        assert int(cs[0].item()) == 0

    def test_commit_chunk_paged_advances_seqlens(self):
        cfg = make_config(num_layers=1, chunk_size=4, block_size_frames=4, max_num_blocks=16)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        mgr.prepare_chunk(0)
        mgr.commit_chunk_paged(0, chunk_frames=4)
        _, cs = mgr.get_paged_state_views(0)
        assert int(cs[0].item()) == 4
        mgr.prepare_chunk(0)
        mgr.commit_chunk_paged(0, chunk_frames=4)
        assert int(cs[0].item()) == 8

    def test_eviction_with_num_left_chunks(self):
        # max_logical_blocks=2 from num_left_chunks=2 (chunk_size=block_size_frames=4)
        cfg = make_config(
            num_layers=1, chunk_size=4, block_size_frames=4,
            num_left_chunks=2, max_num_blocks=16,
        )
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        initial_free = pool.num_free_blocks

        # Commit 4 chunks: only the last 2 blocks should survive after eviction.
        for _ in range(4):
            mgr.prepare_chunk(0)
            mgr.commit_chunk_paged(0, chunk_frames=4)

        # 2 blocks held, 2 evicted back to pool.
        assert pool.num_free_blocks == initial_free - 2
        _, cs = mgr.get_paged_state_views(0)
        assert int(cs[0].item()) == 8  # 2 blocks * 4 frames

    def test_free_stream_returns_blocks(self):
        cfg = make_config(num_layers=1, max_num_blocks=16)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        initial = pool.num_free_blocks
        for _ in range(3):
            mgr.prepare_chunk(0)
            mgr.commit_chunk_paged(0, chunk_frames=cfg.block_size_frames)
        assert pool.num_free_blocks == initial - 3
        mgr.free_stream(0)
        assert pool.num_free_blocks == initial

    def test_prepare_chunks_batched_allocates_one_block_per_stream(self):
        cfg = make_config(num_layers=1, max_num_blocks=32)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        for sid in range(4):
            mgr.allocate_stream(sid, slot_id=sid)
        initial = pool.num_free_blocks
        mgr.prepare_chunks_batched([0, 1, 2, 3])
        assert pool.num_free_blocks == initial - 4
        for sid in range(4):
            bt, _ = mgr.get_paged_state_views(sid)
            assert int(bt[0, 0].item()) >= 0

    def test_get_paged_caches_returns_one_per_layer(self):
        cfg = make_config(num_layers=3, max_num_blocks=8)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        mgr.prepare_chunk(0)
        caches = mgr.get_paged_caches(0)
        assert len(caches) == cfg.num_layers
        # All caches share the same block_table / cache_seqlens.
        for c in caches[1:]:
            assert c.block_table.data_ptr() == caches[0].block_table.data_ptr()
            assert c.cache_seqlens.data_ptr() == caches[0].cache_seqlens.data_ptr()


# ---------------------------------------------------------------------------
# CtcStateCacheManager tests (require CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCtcStateCacheManager:
    def test_allocate_and_get_decoder(self, device):
        from oasr import GpuDecoderConfig, StreamHandle

        mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=4))
        mgr.allocate_stream(0, batch=1, vocab_size=100, device=device)
        dec = mgr.get_decoder(0)
        assert isinstance(dec, StreamHandle)
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

    def test_pool_reuses_state_after_free(self, device):
        """Freed StreamState is pooled and its buffer reused on next allocate."""
        from oasr import GpuDecoderConfig

        mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=4))
        mgr.allocate_stream(0, batch=1, vocab_size=100, device=device)
        state_0 = mgr._states[0]
        buf_ptr = state_0.buffer.data_ptr()

        mgr.free_stream(0)
        assert len(mgr._pool) == 1

        mgr.allocate_stream(1, batch=1, vocab_size=100, device=device)
        state_1 = mgr._states[1]
        assert state_1 is state_0, "StreamState should be reused from pool"
        assert state_1.buffer.data_ptr() == buf_ptr, "Buffer should be reused"
        assert state_1.step == 0, "State should be reset"

    def test_pool_grows_with_concurrent_streams(self, device):
        """Multiple concurrent streams each get their own state."""
        mgr = CtcStateCacheManager()
        mgr.allocate_stream(0, batch=1, vocab_size=50, device=device)
        mgr.allocate_stream(1, batch=1, vocab_size=50, device=device)
        assert mgr._states[0] is not mgr._states[1]

        mgr.free_stream(0)
        mgr.free_stream(1)
        assert len(mgr._pool) == 2

        mgr.allocate_stream(2, batch=1, vocab_size=50, device=device)
        mgr.allocate_stream(3, batch=1, vocab_size=50, device=device)
        assert mgr._states[2] is not mgr._states[3]

    def test_shared_decoder_engine(self, device):
        """All streams share the same underlying decoder engine."""
        mgr = CtcStateCacheManager()
        mgr.allocate_stream(0, batch=1, vocab_size=50, device=device)
        mgr.allocate_stream(1, batch=1, vocab_size=50, device=device)
        h0 = mgr.get_decoder(0)
        h1 = mgr.get_decoder(1)
        assert h0._decoder is h1._decoder, "Handles should share the same engine"


# ---------------------------------------------------------------------------
# StreamContext end-to-end lifecycle tests
# ---------------------------------------------------------------------------


class TestStreamContext:
    """Paged-mode lifecycle tests using CPU tensors (no CUDA required)."""

    def setup_method(self):
        self._next_slot = 0

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
        slot = self._next_slot
        self._next_slot += 1
        att_mgr.allocate_stream(sid, slot_id=slot)
        cnn_mgr.allocate_stream(sid, slot_id=slot)
        return StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)

    def test_initial_cnn_cache_is_zero(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        cnn = ctx.get_cnn_cache()
        assert cnn.shape == (2, 1, 2, 8)  # (L, 1, K-1, D)
        assert cnn.allclose(torch.zeros_like(cnn))

    def test_prepare_chunk_then_get_paged_caches(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        ctx.prepare_chunk()
        caches = ctx.get_att_caches()
        assert len(caches) == cfg.num_layers

    def test_commit_chunk_paged_updates_cnn_and_seqlens(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        ctx.prepare_chunk()
        new_cnn = torch.full((2, 1, 2, 8), 3.0)
        ctx.commit_chunk_paged(chunk_frames=4, new_cnn_cache=new_cnn)
        assert ctx.get_cnn_cache().allclose(torch.full_like(new_cnn, 3.0))
        _, cs = ctx.get_paged_state_views()
        assert int(cs[0].item()) == 4

    def test_free_returns_pool_blocks(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        initial = pool.num_free_blocks
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        for _ in range(3):
            ctx.prepare_chunk()
            ctx.commit_chunk_paged(chunk_frames=4, new_cnn_cache=torch.zeros(2, 1, 2, 8))
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
    def test_streams_have_independent_state(self):
        cfg = make_config(
            num_layers=1, n_kv_head=1, head_dim=4, hidden_dim=8,
            kernel_size=3, chunk_size=4, block_size_frames=4,
            max_num_blocks=32,
        )
        pool = BlockPool(cfg)
        att_mgr = AttentionCacheManager(pool, cfg)
        cnn_mgr = CnnCacheManager(cfg)

        for sid in range(4):
            att_mgr.allocate_stream(sid, slot_id=sid)
            cnn_mgr.allocate_stream(sid, slot_id=sid)

        # Advance each stream a different number of chunks.
        for sid in range(4):
            for _ in range(sid + 1):
                att_mgr.prepare_chunk(sid)
                att_mgr.commit_chunk_paged(sid, chunk_frames=4)
            cnn_mgr.update(sid, torch.full((1, 1, 2, 8), float(sid + 1)))

        for sid in range(4):
            _, cs = att_mgr.get_paged_state_views(sid)
            assert int(cs[0].item()) == 4 * (sid + 1), f"sid={sid}"
            cnn = cnn_mgr.get_cache(sid)
            assert cnn.allclose(torch.full_like(cnn, float(sid + 1))), f"sid={sid}"

    def test_partial_free_leaves_others_intact(self):
        cfg = make_config(
            num_layers=1, n_kv_head=1, head_dim=4, hidden_dim=8,
            kernel_size=3, chunk_size=4, block_size_frames=4,
            max_num_blocks=32,
        )
        pool = BlockPool(cfg)
        att_mgr = AttentionCacheManager(pool, cfg)
        cnn_mgr = CnnCacheManager(cfg)

        for sid_ in range(4):
            att_mgr.allocate_stream(sid_, slot_id=sid_)
            cnn_mgr.allocate_stream(sid_, slot_id=sid_)
            att_mgr.prepare_chunk(sid_)
            att_mgr.commit_chunk_paged(sid_, chunk_frames=4)
            cnn_mgr.update(sid_, torch.full((1, 1, 2, 8), float(sid_)))

        att_mgr.free_stream(1)
        cnn_mgr.free_stream(1)
        att_mgr.free_stream(3)
        cnn_mgr.free_stream(3)

        for sid in [0, 2]:
            _, cs = att_mgr.get_paged_state_views(sid)
            assert int(cs[0].item()) == 4, f"sid={sid}"
            cnn = cnn_mgr.get_cache(sid)
            assert cnn.allclose(torch.full_like(cnn, float(sid))), f"sid={sid}"

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

        for sid_ in range(4):
            att_mgr.allocate_stream(sid_, slot_id=sid_)
            cnn_mgr.allocate_stream(sid_, slot_id=sid_)
            att_mgr.prepare_chunk(sid_)
            att_mgr.commit_chunk_paged(sid_, chunk_frames=4)

        assert pool.num_free_blocks == initial - 4

        att_mgr.free_stream(0)
        att_mgr.free_stream(2)
        assert pool.num_free_blocks == initial - 2
