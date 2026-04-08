# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-stream CTC decoder GPU state manager.

Wraps ``GpuStreamingDecoder`` instances, providing the same allocate /
get / free lifecycle as the attention and CNN cache managers. Each stream
gets its own ``GpuStreamingDecoder`` whose internal GPU state buffer is
allocated on ``init_stream()`` and released when the stream is freed.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from oasr.ctc_decode import GpuDecoderConfig, GpuStreamingDecoder


class CtcStateCacheManager:
    """Manages per-stream CTC decoder GPU state.

    One ``GpuStreamingDecoder`` instance is maintained per active stream.
    The decoder state (beam scores, hypothesis sequences, paged memory tables)
    lives in a CUDA ``uint8`` buffer owned by the decoder.

    Parameters
    ----------
    decoder_config : GpuDecoderConfig, optional
        CTC decoder configuration (beam size, blank ID, thresholds, etc.).
        Defaults to ``GpuDecoderConfig()`` if not provided.

    Examples
    --------
    >>> mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=5))
    >>> mgr.allocate_stream(0, batch=1, vocab_size=5000)
    >>> decoder = mgr.get_decoder(0)
    >>> decoder.decode_chunk(log_probs)
    >>> result = decoder.finalize_stream()
    >>> mgr.free_stream(0)
    """

    def __init__(self, decoder_config: Optional[GpuDecoderConfig] = None) -> None:
        self._decoder_config = decoder_config or GpuDecoderConfig()
        self._decoders: Dict[int, GpuStreamingDecoder] = {}

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
        """Create and initialize a streaming CTC decoder for a new stream.

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
        if stream_id in self._decoders:
            raise ValueError(f"CTC state for stream {stream_id} already allocated.")
        decoder = GpuStreamingDecoder(self._decoder_config)
        decoder.init_stream(batch=batch, vocab_size=vocab_size, device=device)
        self._decoders[stream_id] = decoder

    def free_stream(self, stream_id: int) -> None:
        """Release the CTC decoder and its GPU state for a stream.

        The decoder's internal state buffer tensor is deleted, returning GPU
        memory to PyTorch's allocator.

        Parameters
        ----------
        stream_id : int
            Stream to release.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        """
        if stream_id not in self._decoders:
            raise KeyError(f"CTC state for stream {stream_id} not found.")
        del self._decoders[stream_id]

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_decoder(self, stream_id: int) -> GpuStreamingDecoder:
        """Return the streaming CTC decoder for a stream.

        The returned decoder is ready for ``decode_chunk()`` calls.

        Parameters
        ----------
        stream_id : int
            Stream identifier.

        Returns
        -------
        GpuStreamingDecoder
            The live decoder instance.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        """
        if stream_id not in self._decoders:
            raise KeyError(f"CTC state for stream {stream_id} not found.")
        return self._decoders[stream_id]
