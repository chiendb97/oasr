# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Unified per-request handle for streaming cache managers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from oasr.cache.attention_cache import AttentionCacheManager
from oasr.cache.cnn_cache import CONV_STATE
from oasr.cache.ctc_state import CtcStateCacheManager
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.slot_cnn import SlotCnnCache
from oasr.cache.state import SlotStateCache, SlotTensor
from oasr.ctc_decode import GpuStreamingDecoder, StreamHandle, StreamState
from oasr.utils.staging import to_device


class StreamContext:
    """Delegate one allocated stream's cache operations to shared managers.

    ``ctc_state`` is optional because engine-managed decode strategies own CTC
    state; standalone callers may provide it to use the decoder accessors.
    """

    def __init__(
        self,
        stream_id: int,
        attention_cache: AttentionCacheManager,
        cnn_cache: SlotStateCache,
        ctc_state: Optional[CtcStateCacheManager] = None,
    ) -> None:
        self._stream_id = stream_id
        self._attention_cache = attention_cache
        self._cnn_cache = cnn_cache
        self._ctc_state = ctc_state

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stream_id(self) -> int:
        """The unique identifier for this stream."""
        return self._stream_id

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def get_cnn_cache(self) -> SlotCnnCache:
        """Return the slot-indexed CNN cache descriptor for this stream.

        Returns
        -------
        SlotCnnCache
            Wraps the persistent CNN buffer plus a ``(1,)`` int64 slot-id
            tensor for this stream, matching the ``cnn_cache`` parameter of
            :meth:`~oasr.models.conformer.ConformerModel.forward_chunk_paged`.
        """
        return self.get_states()[CONV_STATE]

    def get_states(self) -> Dict[str, SlotTensor]:
        """Slot-indexed descriptors for **every** declared stream state.

        One entry per :class:`~oasr.cache.StreamStateSpec` the encoder declared,
        keyed by name — ``"conv"`` for the convolutional left-context, plus
        whatever else (Nemotron's per-subsampling-stage tails).  All of them share
        one ``slot_ids`` tensor, so a chunk forward gathers and scatters each with
        a single index.
        """
        slot = self._cnn_cache.slot_of(self._stream_id)
        device = self._cnn_cache.buffer_of(self._cnn_cache.names[0]).device
        slot_ids = to_device([slot], dtype=torch.long, device=device)
        return self._cnn_cache.views(slot_ids)

    def get_decoder(self) -> Union[GpuStreamingDecoder, StreamHandle]:
        """Return the CTC streaming decoder for this stream.

        Returns
        -------
        GpuStreamingDecoder or StreamHandle
            Ready for ``decode_chunk()`` and ``finalize_stream()`` calls.
        """
        assert self._ctc_state is not None, "StreamContext has no CTC state manager"
        return self._ctc_state.get_decoder(self._stream_id)

    def get_ctc_state(self) -> StreamState:
        """Return the raw per-stream :class:`StreamState`.

        Used by the engine's batched streaming-decode path
        (:meth:`~oasr.ctc_decode.GpuStreamingDecoder.decode_chunk_batch`)
        which feeds an array of states into a single C++ launcher.
        """
        assert self._ctc_state is not None, "StreamContext has no CTC state manager"
        return self._ctc_state.get_states([self._stream_id])[0]

    @property
    def ctc_state_manager(self) -> Optional[CtcStateCacheManager]:
        """Underlying :class:`CtcStateCacheManager` (shared across streams), or
        ``None`` for an encoder-only context."""
        return self._ctc_state

    def get_att_caches(self) -> List[PagedKVCache]:
        """Return one :class:`~oasr.layers.attention.PagedKVCache` per layer.

        Used with :meth:`~oasr.models.conformer.ConformerModel.forward_chunk_paged`.
        :meth:`prepare_chunk` must be called first to allocate the write block.

        Returns
        -------
        list[PagedKVCache]
            One entry per encoder layer.
        """
        return self._attention_cache.get_paged_caches(self._stream_id)

    def get_paged_state_views(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the per-stream ``(block_table, cache_seqlens)`` views.

        Cheap accessor for the batched paged forward — avoids building 12
        :class:`PagedKVCache` dataclass instances per stream just to read
        the shared paging tensors.
        """
        return self._attention_cache.get_paged_state_views(self._stream_id)

    def prepare_chunk(self) -> None:
        """Allocate the next physical block before a paged forward pass.

        Must be called **once per chunk** before :meth:`get_att_caches` /
        :meth:`~oasr.models.conformer.ConformerModel.forward_chunk_paged`.
        """
        self._attention_cache.prepare_chunk(self._stream_id)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def commit_chunk_paged(self, chunk_frames: int) -> None:
        """Advance KV ``cache_seqlens`` after a :meth:`forward_chunk_paged` call.

        K/V are already in the pool (written by the attention layer) and the
        CNN cache is already committed in place (scattered by the encoder
        through the :class:`SlotCnnCache` descriptor). This method just
        advances the per-stream ``cache_seqlens`` and evicts if needed.

        Parameters
        ----------
        chunk_frames : int
            Number of encoder-output frames written (usually ``chunk_size``).
        """
        self._attention_cache.commit_chunk_paged(self._stream_id, chunk_frames)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def free(self) -> None:
        """Release all GPU resources for this stream.

        Frees attention KV blocks back to the pool, deletes the CNN cache
        tensor, and deletes the CTC decoder state buffer.

        After calling ``free()``, this ``StreamContext`` must not be used.
        """
        self._attention_cache.free_stream(self._stream_id)
        self._cnn_cache.free_stream(self._stream_id)
        if self._ctc_state is not None:
            self._ctc_state.free_stream(self._stream_id)
