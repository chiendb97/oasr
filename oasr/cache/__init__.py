# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR streaming cache manager with paged GPU memory.

Provides a ``BlockPool``-backed paged attention KV cache, a fixed-size CNN
cache, and a CTC decoder state manager for streaming chunk-by-chunk
conformer inference.  A ``StreamContext`` ties all three together into a
single per-request handle.

Typical server setup::

    from oasr.cache import (
        CacheConfig, BlockPool,
        AttentionCacheManager, CnnCacheManager, CtcStateCacheManager,
        StreamContext,
    )
    from oasr import GpuDecoderConfig

    config = CacheConfig(
        num_layers=12, n_kv_head=4, head_dim=64, hidden_dim=256,
        kernel_size=15, chunk_size=16, num_left_chunks=4,
        block_size_frames=16, max_num_blocks=2048,
    )
    pool     = BlockPool(config)
    att_mgr  = AttentionCacheManager(pool, config)
    cnn_mgr  = CnnCacheManager(config)
    ctc_mgr  = CtcStateCacheManager(GpuDecoderConfig(beam_size=10))

Per-request usage (paged-only)::

    sid = 42
    att_mgr.allocate_stream(sid)
    cnn_mgr.allocate_stream(sid)
    ctc_mgr.allocate_stream(sid, batch=1, vocab_size=5000)
    ctx = StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)

    for chunk_audio in audio_chunks:
        ctx.prepare_chunk()
        att_caches = ctx.get_att_caches()
        cnn_cache  = ctx.get_cnn_cache()
        logits, new_cnn = model.forward_chunk_paged(
            chunk_audio, offset, att_caches, cnn_cache, cache_t1=offset)
        ctx.commit_chunk_paged(logits.size(1), new_cnn)
        ctx.get_decoder().decode_chunk(logits)

    result = ctx.get_decoder().finalize_stream()
    ctx.free()
"""

from oasr.cache.types import CacheConfig
from oasr.cache.block_pool import BlockPool
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.slot_pool import StreamSlotPool
from oasr.cache.attention_cache import AttentionCacheManager
from oasr.cache.cnn_cache import CnnCacheManager
from oasr.cache.ctc_state import CtcStateCacheManager
from oasr.cache.stream import StreamContext

__all__ = [
    "CacheConfig",
    "BlockPool",
    "PagedKVCache",
    "StreamSlotPool",
    "AttentionCacheManager",
    "CnnCacheManager",
    "CtcStateCacheManager",
    "StreamContext",
]
