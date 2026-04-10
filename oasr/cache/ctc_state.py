# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-stream CTC decoder GPU state manager.

Uses a **single shared** :class:`~oasr.ctc_decode.GpuStreamingDecoder`
engine with per-stream :class:`~oasr.ctc_decode.StreamState` objects,
enabling interleaved chunk processing across many concurrent requests
while sharing the JIT module, config, and blank-threshold computation.

Freed states are returned to an internal **pool** so that subsequent
``allocate_stream`` calls can reuse their GPU buffers without triggering
``cudaMalloc``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from oasr.ctc_decode import (
    GpuDecoderConfig,
    GpuStreamingDecoder,
    StreamHandle,
    StreamState,
)


class CtcStateCacheManager:
    """Manages per-stream CTC decoder GPU state with state pooling.

    A **single** :class:`GpuStreamingDecoder` engine is shared across all
    streams.  Each stream gets its own lightweight :class:`StreamState`;
    when a stream is freed the state is pooled (not destroyed) so the
    next ``allocate_stream`` reuses the GPU buffer.

    The :meth:`get_decoder` method returns a :class:`StreamHandle` that
    presents the same ``decode_chunk`` / ``finalize_stream`` / ``step`` /
    ``config`` interface as a standalone decoder, so callers do not need
    to know about the underlying state separation.

    Parameters
    ----------
    decoder_config : GpuDecoderConfig, optional
        CTC decoder configuration (beam size, blank ID, thresholds, etc.).
        Defaults to ``GpuDecoderConfig()`` if not provided.

    Examples
    --------
    >>> mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=5))
    >>> mgr.allocate_stream(0, batch=1, vocab_size=5000)
    >>> mgr.allocate_stream(1, batch=1, vocab_size=5000)
    >>> # Interleaved chunk processing — both streams share one engine:
    >>> mgr.get_decoder(0).decode_chunk(chunk_a)
    >>> mgr.get_decoder(1).decode_chunk(chunk_b)
    >>> mgr.get_decoder(0).decode_chunk(chunk_c)
    >>> result_0 = mgr.get_decoder(0).finalize_stream()
    >>> mgr.free_stream(0)            # state returned to pool
    >>> mgr.allocate_stream(2, batch=1, vocab_size=5000)  # reuses pooled state
    """

    def __init__(self, decoder_config: Optional[GpuDecoderConfig] = None) -> None:
        self._decoder_config = decoder_config or GpuDecoderConfig()
        self._decoder = GpuStreamingDecoder(self._decoder_config)
        self._states: Dict[int, StreamState] = {}
        self._pool: List[StreamState] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate_stream(
        self,
        stream_id: int,
        batch: int,
        vocab_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Create and initialize a per-stream CTC decoder state.

        If a previously freed :class:`StreamState` is available in the
        pool it is reused (via :meth:`GpuStreamingDecoder.reset_state`),
        avoiding a fresh GPU buffer allocation.

        Parameters
        ----------
        stream_id : int
            Unique stream identifier.
        batch : int
            Batch size for this stream (number of concurrent utterances).
        vocab_size : int
            Vocabulary size (must match the log-prob last dimension fed to
            ``decode_chunk``).
        device : torch.device, optional
            CUDA device. Defaults to ``torch.device("cuda")``.

        Raises
        ------
        ValueError
            If ``stream_id`` is already allocated.
        """
        if stream_id in self._states:
            raise ValueError(f"CTC state for stream {stream_id} already allocated.")
        if self._pool:
            state = self._pool.pop()
            self._decoder.reset_state(state, batch, vocab_size, device)
        else:
            state = self._decoder.create_state(batch, vocab_size, device)
        self._states[stream_id] = state

    def free_stream(self, stream_id: int) -> None:
        """Return the CTC state for a stream to the internal pool.

        The state's GPU buffer is **not** freed — it is kept alive inside
        the pool so that subsequent ``allocate_stream`` calls can reuse it.

        Parameters
        ----------
        stream_id : int
            Stream to release.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        """
        if stream_id not in self._states:
            raise KeyError(f"CTC state for stream {stream_id} not found.")
        self._pool.append(self._states.pop(stream_id))

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_decoder(
        self, stream_id: int,
    ) -> StreamHandle:
        """Return a handle that binds the shared decoder to this stream's state.

        The returned :class:`StreamHandle` exposes ``decode_chunk()``,
        ``finalize_stream()``, ``step``, and ``config`` — the same
        interface as :class:`GpuStreamingDecoder` — so it can be used
        as a drop-in replacement.

        Parameters
        ----------
        stream_id : int
            Stream identifier.

        Returns
        -------
        StreamHandle
            A lightweight proxy ready for ``decode_chunk()`` calls.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        """
        if stream_id not in self._states:
            raise KeyError(f"CTC state for stream {stream_id} not found.")
        return StreamHandle(self._decoder, self._states[stream_id])
