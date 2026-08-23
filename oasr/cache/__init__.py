# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paged attention, fixed-size stream state, and decoder cache managers.

``StreamContext`` binds the managers into one per-request handle.
"""

from oasr.cache.attention_cache import AttentionCacheManager
from oasr.cache.block_pool import BlockPool
from oasr.cache.cnn_cache import CnnCacheManager
from oasr.cache.ctc_state import CtcStateCacheManager
from oasr.cache.decoder_kv import DecoderKVCacheManager, DecoderKvExhausted
from oasr.cache.decoder_state import DecoderKv, PagedDecoderKv, build_kv
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.recurrent_state import (
    RecurrentContinuousBatcher,
    RecurrentStateCache,
    RecurrentStepPlan,
)
from oasr.cache.slot_cnn import SlotCnnCache
from oasr.cache.slot_pool import StreamSlotPool
from oasr.cache.state import SlotStateCache, SlotTensor, StreamStateSpec
from oasr.cache.stream import StreamContext
from oasr.cache.types import CacheConfig

__all__ = [
    "CacheConfig",
    "BlockPool",
    "PagedKVCache",
    "RecurrentStateCache",
    "RecurrentContinuousBatcher",
    "RecurrentStepPlan",
    "SlotCnnCache",
    "SlotTensor",
    "StreamStateSpec",
    "SlotStateCache",
    "StreamSlotPool",
    "AttentionCacheManager",
    "CnnCacheManager",
    "CtcStateCacheManager",
    "DecoderKVCacheManager",
    "DecoderKvExhausted",
    "DecoderKv",
    "PagedDecoderKv",
    "build_kv",
    "StreamContext",
]
