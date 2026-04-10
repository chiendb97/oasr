# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
GPU-accelerated CTC prefix beam search decoder.

Provides both offline (full-sequence) and streaming (chunk-by-chunk) decoding
on CUDA, processing batched inputs for high throughput.

Example -- offline batched decode::

    import torch
    import oasr

    log_prob = model(audio)  # [batch, T, vocab_size], float32 CUDA
    seq_lengths = torch.tensor([T] * batch, device="cuda", dtype=torch.int32)
    result = oasr.ctc_beam_search_decode(log_prob, seq_lengths, beam_size=10)
    print(result.tokens[0][0])  # best hypothesis for first utterance

Example -- streaming decode::

    decoder = oasr.GpuStreamingDecoder(oasr.GpuDecoderConfig(beam_size=10))
    decoder.init_stream(batch=1, vocab_size=5000)
    for chunk_logp in chunks:           # [batch, chunk_T, vocab_size]
        decoder.decode_chunk(chunk_logp)
    result = decoder.finalize_stream()
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from oasr.api_logging import oasr_api


@dataclass
class GpuDecoderConfig:
    """Configuration for the GPU CTC prefix beam search decoder.

    Args:
        beam_size: Number of beams to maintain.
        blank_id: CTC blank token id.
        blank_threshold: Skip frames where P(blank) exceeds this threshold
            (offline mode only). Reduces computation on blank-heavy frames.
        max_seq_len: Maximum decoded sequence length per utterance. Beams
            longer than this are truncated.
        use_paged_memory: Use paged-attention-style memory for decoded
            sequences. Reduces GPU memory usage when max_seq_len is large
            by storing sequences in fixed-size pages with prefix sharing.
        page_size: Number of tokens per page (must be a positive integer).
            16 (the default) aligns with a single cache line (64 bytes).
    """

    beam_size: int = 10
    blank_id: int = 0
    blank_threshold: float = 0.98
    max_seq_len: int = 200
    use_paged_memory: bool = False
    page_size: int = 16


@dataclass
class GpuDecoderResult:
    """Batched result from GPU CTC decoder.

    Attributes:
        tokens: Nested list ``[batch][beam][token_ids]`` of decoded sequences.
        lengths: Tensor ``[batch, beam]`` of decoded sequence lengths (int32, CUDA).
        scores: Tensor ``[batch, beam]`` of beam log-probabilities (float32, CUDA).
    """

    tokens: List[List[List[int]]] = field(default_factory=list)
    lengths: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None


@functools.cache
def _get_ctc_decoder_module():
    from oasr.jit.ctc_decoder import gen_ctc_decoder_module

    return gen_ctc_decoder_module().build_and_load()


def _extract_tokens(out_tokens: torch.Tensor, out_lengths: torch.Tensor,
                    batch: int, beam_size: int) -> List[List[List[int]]]:
    """Convert padded token tensor to nested Python lists."""
    out_tokens_cpu = out_tokens.cpu()
    out_lengths_cpu = out_lengths.cpu()
    tokens = []
    for b in range(batch):
        batch_tokens = []
        for k in range(beam_size):
            length = out_lengths_cpu[b, k].item()
            batch_tokens.append(out_tokens_cpu[b, k, :length].tolist())
        tokens.append(batch_tokens)
    return tokens


@oasr_api
def ctc_beam_search_decode(
    log_prob: torch.Tensor,
    seq_lengths: torch.Tensor,
    beam_size: int = 10,
    blank_id: int = 0,
    blank_threshold: float = 0.98,
    max_seq_len: int = 200,
    use_paged_memory: bool = False,
    page_size: int = 16,
) -> GpuDecoderResult:
    """GPU-accelerated CTC prefix beam search decode (offline, full sequence).

    Parameters
    ----------
    log_prob : torch.Tensor
        Log-probability tensor ``[batch, seq_len, vocab_size]`` on CUDA, float32.
    seq_lengths : torch.Tensor
        Actual sequence lengths ``[batch]`` on CUDA, int32.
    beam_size : int
        Number of beams.
    blank_id : int
        CTC blank token id.
    blank_threshold : float
        Skip threshold for blank probability.
    max_seq_len : int
        Maximum decoded output sequence length.
    use_paged_memory : bool
        Use paged-attention-style memory for decoded sequences.
    page_size : int
        Tokens per page when ``use_paged_memory=True`` (default 16).

    Returns
    -------
    GpuDecoderResult
        Decoded tokens, lengths, and scores for each batch and beam.
    """
    mod = _get_ctc_decoder_module()
    batch = log_prob.size(0)

    log_prob = log_prob.contiguous().float()
    seq_lengths = seq_lengths.contiguous().to(torch.int32)

    device = log_prob.device

    # Allocate workspace (needs max of input seq len and output seq len)
    input_seq_len = log_prob.size(1)
    ws_seq_len = max(input_seq_len, max_seq_len)
    vocab_size = log_prob.size(2)

    if use_paged_memory:
        ws_bytes = mod.ctc_decoder_paged_workspace_size(
            batch, beam_size, vocab_size, ws_seq_len, page_size)
    else:
        ws_bytes = mod.ctc_decoder_workspace_size(batch, beam_size, vocab_size, ws_seq_len)
    workspace = torch.empty(int(ws_bytes), dtype=torch.uint8, device=device)

    # Allocate outputs
    out_tokens = torch.empty(batch, beam_size, max_seq_len, dtype=torch.int32, device=device)
    out_lengths = torch.empty(batch, beam_size, dtype=torch.int32, device=device)
    out_scores = torch.empty(batch, beam_size, dtype=torch.float32, device=device)

    if use_paged_memory:
        mod.ctc_beam_search_decode_paged(
            out_tokens, out_lengths, out_scores,
            log_prob, seq_lengths, workspace,
            beam_size, blank_id, blank_threshold, page_size,
        )
    else:
        mod.ctc_beam_search_decode(
            out_tokens, out_lengths, out_scores,
            log_prob, seq_lengths, workspace,
            beam_size, blank_id, blank_threshold,
        )

    tokens = _extract_tokens(out_tokens, out_lengths, batch, beam_size)
    return GpuDecoderResult(tokens=tokens, lengths=out_lengths, scores=out_scores)


class GpuStreamingDecoder:
    """Streaming GPU CTC prefix beam search decoder.

    Maintains persistent GPU state across chunks, enabling low-latency
    streaming decoding.

    Parameters
    ----------
    config : GpuDecoderConfig, optional
        Decoder configuration. Uses defaults if not provided.

    Example
    -------
    >>> decoder = GpuStreamingDecoder(GpuDecoderConfig(beam_size=5))
    >>> decoder.init_stream(batch=1, vocab_size=5000)
    >>> for chunk in chunks:
    ...     decoder.decode_chunk(chunk)
    >>> result = decoder.finalize_stream()
    """

    def __init__(self, config: Optional[GpuDecoderConfig] = None):
        self._config = config or GpuDecoderConfig()
        self._mod = _get_ctc_decoder_module()
        self._state_buffer: Optional[torch.Tensor] = None
        self._step = 0
        self._actual_frame_idx = 0
        self._batch = 0
        self._vocab_size = 0

    def init_stream(self, batch: int, vocab_size: int,
                    device: Optional[torch.device] = None) -> None:
        """Initialize (or re-initialize) streaming decoding state.

        Parameters
        ----------
        batch : int
            Batch size.
        vocab_size : int
            Vocabulary size (must match log-prob last dimension).
        device : torch.device, optional
            CUDA device. Defaults to ``torch.device("cuda")``.
        """
        cfg = self._config
        if device is None:
            device = torch.device("cuda")

        if cfg.use_paged_memory:
            state_bytes = self._mod.ctc_decoder_paged_state_size(
                batch, cfg.beam_size, vocab_size, cfg.max_seq_len, cfg.page_size)
        else:
            state_bytes = self._mod.ctc_decoder_state_size(
                batch, cfg.beam_size, vocab_size, cfg.max_seq_len)
        self._state_buffer = torch.empty(int(state_bytes), dtype=torch.uint8, device=device)
        self._batch = batch
        self._vocab_size = vocab_size
        self._step = 0
        self._actual_frame_idx = 0  # total frames seen including skipped blank frames

        if cfg.use_paged_memory:
            self._mod.ctc_beam_search_init_state_paged(
                self._state_buffer, batch, cfg.beam_size,
                vocab_size, cfg.max_seq_len, cfg.blank_id, cfg.page_size)
        else:
            self._mod.ctc_beam_search_init_state(
                self._state_buffer, batch, cfg.beam_size,
                vocab_size, cfg.max_seq_len, cfg.blank_id)

    def decode_chunk(self, log_prob: torch.Tensor) -> None:
        """Process one chunk of log-probability frames.

        Parameters
        ----------
        log_prob : torch.Tensor
            Log-probability tensor ``[batch, chunk_T, vocab_size]`` on CUDA, float32.
        """
        if self._state_buffer is None:
            raise RuntimeError("Call init_stream() before decode_chunk()")

        cfg = self._config
        log_prob = log_prob.contiguous().float()
        chunk_t = log_prob.size(1)

        # Pre-compute log threshold; None means disabled (mirrors offline path).
        blank_log_thresh = (
            math.log(cfg.blank_threshold)
            if 0.0 < cfg.blank_threshold < 1.0
            else None
        )

        for t in range(chunk_t):
            if self._step >= cfg.max_seq_len:
                break  # Reached maximum decoded sequence length
            frame = log_prob[:, t, :].contiguous()  # [batch, vocab_size]

            # Skip blank-dominant frames (mirrors the init_select_kernel filter
            # in the offline ctc_beam_search_decode_batch path).  A frame is
            # skipped when every batch element's blank log-prob ≥ threshold.
            if blank_log_thresh is not None:
                if frame[:, cfg.blank_id].min().item() >= blank_log_thresh:
                    self._actual_frame_idx += 1
                    continue

            # Pass actual_frame_index so select_seqs[step] is updated to reflect
            # any skipped blank frames.  This enables need_add_blank detection in
            # first_step_kernel / prob_matrix_kernel (same as offline mode).
            self._mod.ctc_beam_search_step(
                self._state_buffer, frame,
                cfg.beam_size, cfg.blank_id,
                self._step, cfg.blank_threshold,
                self._actual_frame_idx,
            )
            self._step += 1
            self._actual_frame_idx += 1

    def finalize_stream(self) -> GpuDecoderResult:
        """Finalize streaming decoding and return results.

        Returns
        -------
        GpuDecoderResult
            Decoded tokens, lengths, and scores for each batch and beam.
        """
        if self._state_buffer is None:
            raise RuntimeError("Call init_stream() before finalize_stream()")

        cfg = self._config
        device = self._state_buffer.device

        out_tokens = torch.empty(
            self._batch, cfg.beam_size, cfg.max_seq_len,
            dtype=torch.int32, device=device)
        out_lengths = torch.empty(
            self._batch, cfg.beam_size, dtype=torch.int32, device=device)
        out_scores = torch.empty(
            self._batch, cfg.beam_size, dtype=torch.float32, device=device)

        self._mod.ctc_beam_search_read_state(
            out_tokens, out_lengths, out_scores,
            self._state_buffer, self._step)

        tokens = _extract_tokens(out_tokens, out_lengths, self._batch, cfg.beam_size)
        return GpuDecoderResult(tokens=tokens, lengths=out_lengths, scores=out_scores)

    @property
    def config(self) -> GpuDecoderConfig:
        """The configuration used to build this decoder."""
        return self._config

    @property
    def step(self) -> int:
        """Current timestep index."""
        return self._step
