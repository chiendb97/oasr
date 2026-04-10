# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Unified per-stream handle for the ASR streaming cache system.

``StreamContext`` ties together the three cache managers for one streaming
request. Callers interact with a single object rather than three separate
managers, reducing the risk of mismatched stream IDs and making the
streaming loop concise.

Typical usage::

    ctx = StreamContext(stream_id, att_mgr, cnn_mgr, ctc_mgr)

    for chunk_audio in audio_chunks:
        att_cache = ctx.get_att_cache()        # (L, H, cache_t, d_k*2)
        cnn_cache = ctx.get_cnn_cache()        # (L, 1, K-1, D)
        logits, new_att, new_cnn = model.forward_chunk(
            chunk_audio, offset, required_cache_size, att_cache, cnn_cache)
        # Extract only the new chunk's KV (last chunk_size frames)
        new_kv_chunk = new_att[:, :, -chunk_size:, :]
        ctx.commit_chunk(new_kv_chunk, new_cnn)
        ctx.get_decoder().decode_chunk(logits)

    result = ctx.get_decoder().finalize_stream()
    ctx.free()
"""

from __future__ import annotations

from typing import List, Union

import torch

from oasr.cache.attention_cache import AttentionCacheManager
from oasr.cache.cnn_cache import CnnCacheManager
from oasr.cache.ctc_state import CtcStateCacheManager
from oasr.ctc_decode import GpuStreamingDecoder, StreamHandle
from oasr.layers.attention.attention import PagedKVCache


class StreamContext:
    """Unified handle tying all cache managers for one streaming request.

    A ``StreamContext`` is created after all three managers have already
    allocated state for the stream (via ``allocate_stream``). It delegates
    every operation to the appropriate manager using the stored ``stream_id``.

    Parameters
    ----------
    stream_id : int
        Unique stream identifier (must already be allocated in all three
        managers before constructing this object).
    attention_cache : AttentionCacheManager
        Shared paged attention KV cache manager.
    cnn_cache : CnnCacheManager
        Per-stream CNN cache manager.
    ctc_state : CtcStateCacheManager
        Per-stream CTC decoder state manager.

    Examples
    --------
    >>> att_mgr.allocate_stream(sid)
    >>> cnn_mgr.allocate_stream(sid)
    >>> ctc_mgr.allocate_stream(sid, batch=1, vocab_size=5000)
    >>> ctx = StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)
    >>> ctx.get_att_cache()        # pass to forward_chunk
    >>> ctx.commit_chunk(...)      # update after forward_chunk
    >>> ctx.free()                 # release all resources
    """

    def __init__(
        self,
        stream_id: int,
        attention_cache: AttentionCacheManager,
        cnn_cache: CnnCacheManager,
        ctc_state: CtcStateCacheManager,
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

    def get_att_cache(self) -> torch.Tensor:
        """Return the stacked attention KV cache for ``forward_chunk``.

        Returns
        -------
        torch.Tensor
            Shape ``(num_layers, n_kv_head, cache_t1, d_k * 2)`` matching
            the ``att_cache`` parameter of ``ConformerEncoder.forward_chunk``.
            Returns a ``(0, 0, 0, 0)`` empty tensor on the first chunk.
        """
        return self._attention_cache.get_stacked_cache(self._stream_id)

    def get_cnn_cache(self) -> torch.Tensor:
        """Return the CNN cache tensor for ``forward_chunk``.

        Returns
        -------
        torch.Tensor
            Shape ``(num_layers, 1, cnn_cache_frames, hidden_dim)`` matching
            the ``cnn_cache`` parameter of ``ConformerEncoder.forward_chunk``.
        """
        return self._cnn_cache.get_cache(self._stream_id)

    def get_decoder(self) -> Union[GpuStreamingDecoder, StreamHandle]:
        """Return the CTC streaming decoder for this stream.

        Returns
        -------
        GpuStreamingDecoder or StreamHandle
            Ready for ``decode_chunk()`` and ``finalize_stream()`` calls.
        """
        return self._ctc_state.get_decoder(self._stream_id)

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

    def prepare_chunk(self) -> None:
        """Allocate the next physical block before a paged forward pass.

        Must be called **once per chunk** before :meth:`get_att_caches` /
        :meth:`~oasr.models.conformer.ConformerModel.forward_chunk_paged`.
        """
        self._attention_cache.prepare_chunk(self._stream_id)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def commit_chunk(
        self,
        new_kv_chunk: torch.Tensor,
        new_cnn_cache: torch.Tensor,
    ) -> None:
        """Commit encoder cache outputs from the latest ``forward_chunk`` call.

        Dense-mode (``forward_chunk``) path.  For paged-mode use
        :meth:`commit_chunk_paged` instead.

        Parameters
        ----------
        new_kv_chunk : torch.Tensor
            New packed K+V data.  Pass only the **new** frames:
            ``new_att_cache[:, :, -chunk_size:, :]``.
            Shape: ``(num_layers, n_kv_head, chunk_frames, kv_last_dim)``.
        new_cnn_cache : torch.Tensor
            CNN cache returned by ``forward_chunk``.
            Shape: ``(num_layers, 1, cnn_cache_frames, hidden_dim)``.
        """
        self._attention_cache.commit(self._stream_id, new_kv_chunk)
        self._cnn_cache.update(self._stream_id, new_cnn_cache)

    def commit_chunk_paged(
        self,
        chunk_frames: int,
        new_cnn_cache: torch.Tensor,
    ) -> None:
        """Commit cache state after a :meth:`forward_chunk_paged` call.

        K/V were already written into the pool by the attention layer.  This
        method advances ``cache_seqlens``, evicts if needed, and stores the
        updated CNN cache.

        Parameters
        ----------
        chunk_frames : int
            Number of encoder-output frames written (usually ``chunk_size``).
        new_cnn_cache : torch.Tensor
            CNN cache returned by ``forward_chunk_paged``.
            Shape: ``(num_layers, 1, cnn_cache_frames, hidden_dim)``.
        """
        self._attention_cache.commit_chunk_paged(self._stream_id, chunk_frames)
        self._cnn_cache.update(self._stream_id, new_cnn_cache)

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
        self._ctc_state.free_stream(self._stream_id)
