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
    max_batch_size: int = 32,
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
        max_batch_size=max_batch_size,
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
        # chunk_size tracks block_size_frames: CacheConfig requires one chunk to
        # fit in one block (these pool-only tests don't otherwise use chunk_size).
        cfg = make_config(
            max_num_blocks=4, block_size_frames=2, chunk_size=2, n_kv_head=1, head_dim=4
        )
        pool = BlockPool(cfg)
        ids = pool.allocate(2)
        k0, v0 = pool.get_kv_block_view(0, ids[0])
        k1, v1 = pool.get_kv_block_view(0, ids[1])
        k0[:] = 1.0
        v0[:] = 10.0
        k1[:] = 2.0
        v1[:] = 20.0
        k_out, v_out = pool.gather_kv_blocks(layer=0, block_ids=ids)
        # First 2 frames block 0, next 2 frames block 1.
        assert k_out[:2].allclose(torch.ones_like(k_out[:2]))
        assert k_out[2:].allclose(torch.full_like(k_out[2:], 2.0))
        assert v_out[:2].allclose(torch.full_like(v_out[:2], 10.0))
        assert v_out[2:].allclose(torch.full_like(v_out[2:], 20.0))

    def test_layer_independence(self):
        cfg = make_config(
            num_layers=2,
            max_num_blocks=4,
            block_size_frames=2,
            chunk_size=2,
            n_kv_head=1,
            head_dim=4,
        )
        pool = BlockPool(cfg)
        (bid,) = pool.allocate(1)
        k0, v0 = pool.get_kv_block_view(0, bid)
        k1, v1 = pool.get_kv_block_view(1, bid)
        k0[:] = 1.0
        v0[:] = 1.0
        k1[:] = 2.0
        v1[:] = 2.0
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
            num_layers=1,
            chunk_size=4,
            block_size_frames=4,
            num_left_chunks=2,
            max_num_blocks=16,
            # One stream, so declare one: with eviction on, CacheConfig requires
            # max_num_blocks >= max_batch_size * blocks_per_stream, and the
            # default 32 would claim capacity this 16-block pool cannot back.
            max_batch_size=1,
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
# Capacity accounting / validation
# ---------------------------------------------------------------------------


class TestCacheConfigValidation:
    """``CacheConfig`` must reject geometries that silently corrupt or overflow."""

    def test_chunk_larger_than_block_rejected(self):
        # One block is allocated per chunk, so a chunk wider than a block would
        # spill into the next logical block (the following chunk's page).
        with pytest.raises(ValueError, match="block_size_frames"):
            make_config(chunk_size=32, block_size_frames=16)

    def test_chunk_equal_to_block_allowed(self):
        cfg = make_config(chunk_size=16, block_size_frames=16)
        assert cfg.chunk_size == cfg.block_size_frames

    def test_block_table_too_narrow_for_history_rejected(self):
        with pytest.raises(ValueError, match="max_blocks_per_seq"):
            CacheConfig(
                chunk_size=4,
                block_size_frames=4,
                num_left_chunks=1000,
                max_blocks_per_seq=4,
                device=CPU,
                dtype=torch.float32,
            )

    def test_blocks_per_stream_bounded_by_pool_fair_share(self):
        cfg = make_config(max_num_blocks=64, block_size_frames=4)
        cfg = CacheConfig(**{**cfg.__dict__, "max_batch_size": 8})
        assert cfg.blocks_per_stream == 8  # 64 // 8
        assert cfg.max_stream_frames == 32

    def test_blocks_per_stream_bounded_by_block_table_width(self):
        cfg = CacheConfig(
            chunk_size=4,
            block_size_frames=4,
            max_num_blocks=4096,
            max_batch_size=2,
            max_blocks_per_seq=16,
            device=CPU,
            dtype=torch.float32,
        )
        assert cfg.blocks_per_stream == 16  # min(16, 4096 // 2)

    def test_eviction_cap_wins_when_enabled(self):
        cfg = make_config(num_left_chunks=2, chunk_size=4, block_size_frames=4)
        assert cfg.blocks_per_stream == cfg.max_logical_blocks == 2


class TestStreamCapacityGate:
    """A stream must be reported at capacity *before* the allocator can fail.

    Regression: with unlimited history (the default ``num_left_chunks=-1``)
    eviction is disabled, so a long-running stream used to raise
    ``RuntimeError: BlockPool exhausted`` from inside the encoder forward — an
    exception the serving dispatcher fans out to every in-flight request.
    """

    def test_capacity_reported_before_pool_exhaustion(self):
        cfg = CacheConfig(
            num_layers=1,
            n_kv_head=1,
            head_dim=8,
            hidden_dim=8,
            kernel_size=3,
            chunk_size=4,
            block_size_frames=4,
            max_num_blocks=6,
            max_batch_size=2,
            num_left_chunks=-1,
            device=CPU,
            dtype=torch.float32,
        )
        assert cfg.blocks_per_stream == 3
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        mgr.allocate_stream(1, slot_id=1)

        chunks = 0
        while not (mgr.at_capacity(0) or mgr.at_capacity(1)):
            # Would raise "BlockPool exhausted" if the gate were absent.
            mgr.prepare_chunks_batched([0, 1])
            mgr.commit_chunks_paged_batched([0, 1], cfg.chunk_size)
            chunks += 1
            assert chunks <= cfg.blocks_per_stream + 1, "capacity never reported"
        assert chunks == cfg.blocks_per_stream
        assert pool.num_free_blocks == 0

    def test_never_at_capacity_when_eviction_enabled(self):
        # One stream (see the eviction test above for why max_batch_size is explicit).
        cfg = make_config(
            num_left_chunks=2,
            max_num_blocks=8,
            chunk_size=4,
            block_size_frames=4,
            max_batch_size=1,
        )
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        for _ in range(20):
            assert not mgr.at_capacity(0)
            mgr.prepare_chunks_batched([0])
            mgr.commit_chunks_paged_batched([0], cfg.chunk_size)
        assert pool.num_free_blocks > 0


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
            num_layers=2,
            n_kv_head=2,
            head_dim=4,
            hidden_dim=8,
            kernel_size=3,
            chunk_size=4,
            block_size_frames=4,
            num_left_chunks=num_left_chunks,
            max_num_blocks=32,
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
        gathered = cnn.gather()
        assert gathered.shape == (2, 1, 2, 8)  # (L, 1, K-1, D)
        assert gathered.allclose(torch.zeros_like(gathered))

    def test_prepare_chunk_then_get_paged_caches(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        ctx.prepare_chunk()
        caches = ctx.get_att_caches()
        assert len(caches) == cfg.num_layers

    def test_commit_chunk_paged_advances_seqlens(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        ctx.prepare_chunk()
        ctx.commit_chunk_paged(chunk_frames=4)
        _, cs = ctx.get_paged_state_views()
        assert int(cs[0].item()) == 4

    def test_cnn_cache_scatter_round_trips(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        cnn = ctx.get_cnn_cache()
        new_cnn = torch.full((2, 1, 2, 8), 3.0)
        cnn.scatter(new_cnn)
        assert ctx.get_cnn_cache().gather().allclose(torch.full_like(new_cnn, 3.0))

    def test_free_returns_pool_blocks(self):
        cfg, pool, att_mgr, cnn_mgr = self._setup()
        initial = pool.num_free_blocks
        ctx = self._make_stream(0, att_mgr, cnn_mgr)
        for _ in range(3):
            ctx.prepare_chunk()
            ctx.commit_chunk_paged(chunk_frames=4)
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
            num_layers=1,
            n_kv_head=1,
            head_dim=4,
            hidden_dim=8,
            kernel_size=3,
            chunk_size=4,
            block_size_frames=4,
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
            num_layers=1,
            n_kv_head=1,
            head_dim=4,
            hidden_dim=8,
            kernel_size=3,
            chunk_size=4,
            block_size_frames=4,
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
            num_layers=1,
            n_kv_head=1,
            head_dim=4,
            hidden_dim=8,
            kernel_size=3,
            chunk_size=4,
            block_size_frames=4,
            num_left_chunks=-1,
            max_num_blocks=32,
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


class TestBlockPoolFreeValidation:
    """M6: a double free hands one physical block to two streams.

    Each then overwrites the other's KV with no error anywhere — two silently
    wrong transcripts.  It costs a set membership to catch here and is
    effectively undiagnosable downstream.
    """

    def _pool(self, n=8):
        from oasr.cache import BlockPool
        from oasr.cache.types import CacheConfig

        return BlockPool(
            CacheConfig(
                num_layers=1,
                n_kv_head=1,
                head_dim=8,
                hidden_dim=8,
                kernel_size=3,
                chunk_size=4,
                num_left_chunks=-1,
                block_size_frames=4,
                max_num_blocks=n,
                max_blocks_per_seq=4,
                max_batch_size=2,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )

    def test_double_free_raises(self):
        pool = self._pool()
        ids = pool.allocate(2)
        pool.free(ids)
        with pytest.raises(ValueError, match="already free"):
            pool.free(ids)

    def test_double_free_within_one_call_raises(self):
        pool = self._pool()
        bid = pool.allocate(1)[0]
        with pytest.raises(ValueError, match="already free"):
            pool.free([bid, bid])

    def test_out_of_range_raises(self):
        pool = self._pool(n=8)
        with pytest.raises(ValueError, match="out of range"):
            pool.free([99])
        with pytest.raises(ValueError, match="out of range"):
            pool.free([-1])

    def test_empty_free_is_a_noop(self):
        pool = self._pool()
        before = pool.num_free_blocks
        pool.free([])
        assert pool.num_free_blocks == before

    def test_normal_free_still_returns_blocks(self):
        pool = self._pool(n=8)
        ids = pool.allocate(3)
        assert pool.num_free_blocks == 5
        pool.free(ids)
        assert pool.num_free_blocks == 8


class TestBatchedEviction:
    """M2: eviction must cost a fixed number of kernels, not 4 per stream.

    The per-stream version ran a GPU ``.clone()`` of the block-table row plus
    three scalar GPU writes for every stream every chunk — measured 10.7% of
    streaming throughput at ``num_left_chunks=8`` and 15.4% at 4, which is why
    nobody enabled it.  Batching took that to 3.4% / 4.9%.

    Not the ring block table the review proposed: a ring is a *kernel* change to
    the paged FMHA, a path with two known defects, and batching removes the cost
    that was actually there.
    """

    def _mgr(self, num_left_chunks, streams=4, blocks_per_seq=8, recycle=False):
        from oasr.cache import AttentionCacheManager, BlockPool
        from oasr.cache.types import CacheConfig

        cfg = CacheConfig(
            num_layers=1,
            n_kv_head=1,
            head_dim=8,
            hidden_dim=8,
            kernel_size=3,
            chunk_size=4,
            num_left_chunks=num_left_chunks,
            recycle_streaming_history=recycle,
            block_size_frames=4,
            max_num_blocks=streams * blocks_per_seq,
            max_blocks_per_seq=blocks_per_seq,
            max_batch_size=streams,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        for sid in range(streams):
            mgr.allocate_stream(sid, slot_id=sid)
        return mgr, pool, cfg

    def _drive(self, mgr, streams, chunks):
        for _ in range(chunks):
            mgr.prepare_chunks_batched(list(range(streams)))
            mgr.commit_chunks_paged_batched(list(range(streams)), 4)

    def test_batched_matches_per_stream_eviction(self):
        """The batched path must evict exactly what the per-stream one did.

        Same blocks, same block table, same ``cache_seqlens`` — the equivalence
        oracle for a pure refactor.
        """
        n, chunks = 4, 12
        a, pool_a, _ = self._mgr(num_left_chunks=8)
        b, pool_b, _ = self._mgr(num_left_chunks=8)
        # ``a`` uses the batched entry point, ``b`` the single-stream one.
        for _ in range(chunks):
            a.prepare_chunks_batched(list(range(n)))
            a.commit_chunks_paged_batched(list(range(n)), 4)
            b.prepare_chunks_batched(list(range(n)))
            for sid in range(n):
                b.commit_chunk_paged(sid, 4)
        assert torch.equal(a.block_table, b.block_table)
        assert torch.equal(a.cache_seqlens, b.cache_seqlens)
        assert pool_a.num_free_blocks == pool_b.num_free_blocks
        for sid in range(n):
            assert a._streams[sid].logical_blocks == b._streams[sid].logical_blocks  # noqa: SLF001

    def test_history_stays_at_the_cap(self):
        n = 4
        mgr, _pool, cfg = self._mgr(num_left_chunks=8)
        cap = cfg.max_logical_blocks
        self._drive(mgr, n, chunks=20)
        for sid in range(n):
            assert len(mgr._streams[sid].logical_blocks) == cap  # noqa: SLF001
        assert (mgr.cache_seqlens == cap * cfg.block_size_frames).all()

    def test_a_pool_sized_to_the_invariant_does_not_exhaust(self):
        """Eviction must free *before* allocating.

        Eviction used to run only at commit, i.e. after the allocation, so a
        stream at its cap had to be handed a block before giving one back — the
        pool silently needed ``max_batch_size`` blocks of headroom beyond the
        documented ``max_num_blocks >= max_batch_size * max_logical_blocks``, and
        a pool sized exactly to it raised ``BlockPool exhausted`` the moment the
        cap was reached.  This is that config.
        """
        n, per = 4, 8
        mgr, pool, cfg = self._mgr(num_left_chunks=8, streams=n, blocks_per_seq=per)
        assert pool.num_blocks == n * cfg.max_logical_blocks, "sized to the invariant"
        self._drive(mgr, n, chunks=30)  # well past the cap
        assert pool.num_free_blocks == 0  # fully utilised, never over-subscribed

    def test_block_table_rows_hold_the_right_physical_blocks(self):
        """The shift must preserve logical order, oldest first."""
        n = 2
        mgr, _pool, cfg = self._mgr(num_left_chunks=8, streams=n)
        self._drive(mgr, n, chunks=15)
        for sid in range(n):
            held = mgr._streams[sid].logical_blocks  # noqa: SLF001
            slot = mgr.slot_of(sid)
            row = mgr.block_table[slot, : len(held)].tolist()
            assert row == held
            # Everything past the logical end must be blank.
            assert mgr.block_table[slot, len(held) :].sum().item() == 0


class TestRecycleStreamingHistory:
    """M1(3): recycling at the ceiling instead of terminating the stream."""

    def _cfg(self, recycle, streams=4, blocks_per_seq=8):
        from oasr.cache.types import CacheConfig

        return CacheConfig(
            num_layers=1,
            n_kv_head=1,
            head_dim=8,
            hidden_dim=8,
            kernel_size=3,
            chunk_size=4,
            num_left_chunks=-1,
            recycle_streaming_history=recycle,
            block_size_frames=4,
            max_num_blocks=streams * blocks_per_seq,
            max_blocks_per_seq=blocks_per_seq,
            max_batch_size=streams,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_off_by_default_keeps_unlimited_history(self):
        cfg = self._cfg(recycle=False)
        assert cfg.max_logical_blocks is None

    def test_on_derives_the_cap_from_the_configured_ceiling(self):
        """The cap is exactly where the stream would otherwise be terminated."""
        cfg = self._cfg(recycle=True)
        assert cfg.max_logical_blocks == cfg.blocks_per_stream

    def test_an_explicit_cap_still_wins(self):
        from oasr.cache.types import CacheConfig

        cfg = CacheConfig(
            num_layers=1,
            n_kv_head=1,
            head_dim=8,
            hidden_dim=8,
            kernel_size=3,
            chunk_size=4,
            num_left_chunks=4,
            recycle_streaming_history=True,
            block_size_frames=4,
            max_num_blocks=64,
            max_blocks_per_seq=16,
            max_batch_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert cfg.max_logical_blocks == 4

    def test_a_stream_is_never_at_capacity_when_recycling(self):
        """The whole point: the stream keeps going instead of being finalised."""
        from oasr.cache import AttentionCacheManager, BlockPool

        cfg = self._cfg(recycle=True)
        mgr = AttentionCacheManager(BlockPool(cfg), cfg)
        mgr.allocate_stream(0, slot_id=0)
        for _ in range(40):  # far past blocks_per_stream
            assert not mgr.at_capacity(0)
            mgr.prepare_chunks_batched([0])
            mgr.commit_chunks_paged_batched([0], 4)
        assert len(mgr._streams[0].logical_blocks) == cfg.blocks_per_stream  # noqa: SLF001


# ---------------------------------------------------------------------------
# SlotStateCache — the generic form of the CNN cache
# ---------------------------------------------------------------------------


class TestSlotStateCache:
    """The fixed-extent axis: declared shapes, one buffer each, slot-addressed.

    ``CnnCacheManager`` is the single-spec instance of this and keeps its own
    tests above; these cover what the generic form adds — several states at once,
    a per-state ``slot_axis``, and the invariants a wrong declaration would break.
    """

    @staticmethod
    def _cache(**kw):
        from oasr.cache import SlotStateCache, StreamStateSpec

        specs = kw.pop(
            "specs",
            [
                StreamStateSpec("conv", (3, 2, 8), slot_axis=1),
                StreamStateSpec("subsample.0", (2, 5, 1), slot_axis=0),
            ],
        )
        return SlotStateCache(
            specs, max_batch_size=kw.pop("max_batch_size", 4), device=CPU, dtype=torch.float32
        )

    def test_slot_axis_places_the_batch_axis_where_declared(self):
        """This is what preserves the Conformer conv cache's ``(L, B, K-1, D)``
        layout — and therefore its buffer address, which the graph captures."""
        cache = self._cache()
        assert tuple(cache.buffer_of("conv").shape) == (3, 4, 2, 8)
        assert tuple(cache.buffer_of("subsample.0").shape) == (4, 2, 5, 1)

    def test_views_gather_and_scatter_on_the_declared_axis(self):
        cache = self._cache()
        cache.allocate_stream(7, slot_id=1)
        cache.allocate_stream(9, slot_id=3)
        slots = torch.tensor([1, 3], dtype=torch.long)
        views = cache.views(slots)
        assert tuple(views["conv"].gather().shape) == (3, 2, 2, 8)
        assert tuple(views["subsample.0"].gather().shape) == (2, 2, 5, 1)

        views["subsample.0"].scatter(
            torch.arange(2 * 2 * 5 * 1, dtype=torch.float32).view(2, 2, 5, 1)
        )
        got = views["subsample.0"].gather()
        assert got[0, 0, 0, 0] == 0.0 and got[1, 0, 0, 0] == 10.0
        # Slots outside the view are untouched.
        assert cache.buffer_of("subsample.0")[0].abs().max() == 0
        assert cache.buffer_of("subsample.0")[2].abs().max() == 0

    def test_allocation_zeroes_the_slot_because_zero_is_the_initial_state(self):
        """A zero left-context *is* the padding an offline pass applies, so a
        stream's first chunk must see zeros rather than the previous tenant's."""
        cache = self._cache()
        cache.allocate_stream(1, slot_id=0)
        cache.buffer_of("conv")[:, 0].fill_(5.0)
        cache.free_stream(1)
        cache.allocate_stream(2, slot_id=0)
        assert cache.buffer_of("conv")[:, 0].abs().max() == 0

    def test_an_empty_declaration_allocates_nothing(self):
        cache = self._cache(specs=[])
        assert cache.names == []
        assert cache.nbytes_per_stream() == 0
        cache.allocate_stream(1, slot_id=0)  # still tracks the slot

    def test_duplicate_names_are_refused(self):
        from oasr.cache import StreamStateSpec

        with pytest.raises(ValueError, match="duplicate"):
            self._cache(specs=[StreamStateSpec("a", (1,)), StreamStateSpec("a", (2,))])

    def test_an_out_of_range_slot_axis_is_refused(self):
        from oasr.cache import StreamStateSpec

        with pytest.raises(ValueError, match="slot_axis"):
            self._cache(specs=[StreamStateSpec("a", (2, 3), slot_axis=3)])

    def test_unknown_state_names_say_what_is_declared(self):
        cache = self._cache()
        with pytest.raises(KeyError, match="subsample.0"):
            cache.buffer_of("nope")

    def test_slot_lifecycle_matches_the_other_managers(self):
        cache = self._cache()
        cache.allocate_stream(1, slot_id=0)
        with pytest.raises(ValueError, match="already allocated"):
            cache.allocate_stream(1, slot_id=1)
        with pytest.raises(ValueError, match="already in use"):
            cache.allocate_stream(2, slot_id=0)
        with pytest.raises(ValueError, match="out of range"):
            cache.allocate_stream(3, slot_id=99)
        cache.free_stream(1)
        with pytest.raises(KeyError):
            cache.slot_of(1)

    def test_the_cnn_manager_is_a_single_spec_instance_of_this(self):
        """One implementation, not two: the historical accessors are a facade."""
        from oasr.cache import SlotStateCache

        mgr = CnnCacheManager(make_config(num_layers=2, kernel_size=3, hidden_dim=8))
        assert isinstance(mgr, SlotStateCache)
        assert mgr.names == ["conv"]
        # ``.buffer`` must be the *same object* the generic accessor returns —
        # the CUDA-graph cache captures it by address.
        assert mgr.buffer is mgr.buffer_of("conv")


# ---------------------------------------------------------------------------
# Prefilled K/V window (a trained fixed attention span)
# ---------------------------------------------------------------------------


class TestPrefilledKvWindow:
    """``prefill_kv_window``: the whole retained window, zeroed, at admission.

    Its purpose is uniformity, not memory: a Transformer-XL relative-position
    table's distances are ``cache_seqlens + i - j``, so one shared table is only
    correct if the whole cohort reports the same cached length.
    """

    @staticmethod
    def _cfg(**kw):
        return make_config(
            num_left_chunks=kw.pop("num_left_chunks", 4),
            chunk_size=4,
            block_size_frames=4,
            max_num_blocks=64,
            max_batch_size=4,
            **kw,
        )

    def test_a_new_stream_reports_the_full_window_immediately(self):
        cfg = self._cfg()
        object.__setattr__(cfg, "prefill_kv_window", True)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        assert int(mgr.cache_seqlens[0]) == cfg.max_logical_blocks * cfg.block_size_frames

    def test_cache_seqlens_is_the_same_constant_at_every_chunk(self):
        """What the shared position table depends on, so it is asserted rather
        than assumed: after ``prepare``, every stream is at the same length."""
        cfg = self._cfg()
        object.__setattr__(cfg, "prefill_kv_window", True)
        mgr = AttentionCacheManager(BlockPool(cfg), cfg)
        for sid in range(3):
            mgr.allocate_stream(sid, slot_id=sid)
        expected = cfg.prefilled_cache_frames
        for _ in range(6):
            mgr.prepare_chunks_batched([0, 1, 2])
            assert [int(mgr.cache_seqlens[s]) for s in range(3)] == [expected] * 3
            mgr.commit_chunks_paged_batched([0, 1, 2], cfg.chunk_size)

    def test_a_young_stream_sees_zeros_not_the_previous_tenant(self):
        """Masked either way, but "attends over zeros" should be *true*: a recycled
        block otherwise opens the window on live K/V from another stream."""
        cfg = self._cfg()
        object.__setattr__(cfg, "prefill_kv_window", True)
        pool = BlockPool(cfg)
        mgr = AttentionCacheManager(pool, cfg)
        mgr.allocate_stream(0, slot_id=0)
        k, v = pool.get_kv_view(0)
        blocks = mgr._streams[0].logical_blocks  # noqa: SLF001
        for b in blocks:
            k[b].fill_(7.0)
        mgr.free_stream(0)
        mgr.allocate_stream(1, slot_id=0)
        for b in mgr._streams[1].logical_blocks:  # noqa: SLF001
            assert k[b].abs().max() == 0

    def test_unlimited_history_cannot_be_prefilled(self):
        """There is no window to prefill, and silently ignoring the flag would
        leave the position table wrong instead of the config invalid."""
        with pytest.raises(ValueError, match="num_left_chunks"):
            CacheConfig(
                num_layers=2,
                n_kv_head=2,
                head_dim=8,
                hidden_dim=16,
                chunk_size=4,
                num_left_chunks=-1,
                block_size_frames=4,
                max_num_blocks=64,
                max_batch_size=2,
                prefill_kv_window=True,
                device=CPU,
                dtype=torch.float32,
            )
