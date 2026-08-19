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

from typing import Dict, List, Optional

import torch

from oasr.ctc_decode import (
    GpuDecoderConfig,
    GpuStreamingDecoder,
    StreamHandle,
    StreamState,
)


class CtcStateCacheManager:
    """Pool per-stream state behind one shared streaming decoder.

    :meth:`get_decoder` returns a stream-scoped handle with the standalone
    decoder interface. Freed state is reset and reused.
    """

    def __init__(
        self, decoder_config: Optional[GpuDecoderConfig] = None, *, use_cuda_graphs: bool = True
    ) -> None:
        self._decoder_config = decoder_config or GpuDecoderConfig()
        self._decoder = GpuStreamingDecoder(self._decoder_config, use_cuda_graphs=use_cuda_graphs)
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
        self,
        stream_id: int,
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

    def get_states(self, stream_ids: List[int]) -> List[StreamState]:
        """Return :class:`StreamState` objects for ``stream_ids`` in order.

        Used by the engine to feed
        :meth:`~oasr.ctc_decode.GpuStreamingDecoder.decode_chunk_batch`
        with the active set of streams in one call.

        Raises
        ------
        KeyError
            If any of ``stream_ids`` is not allocated.
        """
        result: List[StreamState] = []
        for sid in stream_ids:
            s = self._states.get(sid)
            if s is None:
                raise KeyError(f"CTC state for stream {sid} not found.")
            result.append(s)
        return result

    @property
    def decoder(self) -> GpuStreamingDecoder:
        """The shared :class:`GpuStreamingDecoder` engine."""
        return self._decoder
