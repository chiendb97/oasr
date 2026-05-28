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

Example -- streaming decode (single request)::

    decoder = oasr.GpuStreamingDecoder(oasr.GpuDecoderConfig(beam_size=10))
    decoder.init_stream(batch=1, vocab_size=5000)
    for chunk_logp in chunks:           # [batch, chunk_T, vocab_size]
        decoder.decode_chunk(chunk_logp)
    result = decoder.finalize_stream()

Example -- interleaved multi-request decode::

    decoder = oasr.GpuStreamingDecoder(oasr.GpuDecoderConfig(beam_size=10))
    s1 = decoder.create_state(batch=1, vocab_size=5000)
    s2 = decoder.create_state(batch=1, vocab_size=5000)
    decoder.decode_chunk(chunk_a, state=s1)
    decoder.decode_chunk(chunk_b, state=s2)   # interleaved
    decoder.decode_chunk(chunk_c, state=s1)
    result_1 = decoder.finalize_stream(state=s1)
    result_2 = decoder.finalize_stream(state=s2)
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from oasr.utils.nvtx import nvtx_pop, nvtx_push
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


class StreamState:
    """Per-request GPU state for streaming CTC beam search.

    Created via :meth:`GpuStreamingDecoder.create_state`.  Contains the
    GPU buffer holding beam search state (scores, token sequences, page
    tables) and lightweight counters that track decoding progress.

    ``StreamState`` objects are designed to be **pooled and reused** across
    requests.  Call :meth:`GpuStreamingDecoder.reset_state` to reinitialize
    a state for a new request without reallocating the GPU buffer.

    Attributes
    ----------
    buffer : torch.Tensor
        Raw GPU ``uint8`` buffer backing the beam search state.
    step : int
        Number of non-blank frames processed so far.
    actual_frame_idx : int
        Total frames seen (including skipped blank frames).
    batch : int
        Batch size this state was initialized for.
    vocab_size : int
        Vocabulary size this state was initialized for.
    """

    __slots__ = ("buffer", "step", "actual_frame_idx", "batch", "vocab_size",
                 "_buffer_bytes")

    def __init__(self, buffer: torch.Tensor, batch: int, vocab_size: int,
                 buffer_bytes: int) -> None:
        self.buffer: torch.Tensor = buffer
        self.step: int = 0
        self.actual_frame_idx: int = 0
        self.batch: int = batch
        self.vocab_size: int = vocab_size
        self._buffer_bytes: int = buffer_bytes


class StreamHandle:
    """Lightweight proxy that binds a shared decoder engine to a per-request
    :class:`StreamState`.

    Presents the same ``decode_chunk`` / ``finalize_stream`` / ``step`` /
    ``config`` interface as :class:`GpuStreamingDecoder`, so callers that
    receive a handle can use it exactly like a standalone decoder without
    knowing about the underlying state separation.
    """

    __slots__ = ("_decoder", "_state")

    def __init__(self, decoder: GpuStreamingDecoder, state: StreamState) -> None:
        self._decoder = decoder
        self._state = state

    def decode_chunk(self, log_prob: torch.Tensor) -> None:
        """Process one chunk using the bound state."""
        self._decoder.decode_chunk(log_prob, state=self._state)

    def finalize_stream(self) -> GpuDecoderResult:
        """Finalize and return results from the bound state."""
        return self._decoder.finalize_stream(state=self._state)

    @property
    def step(self) -> int:
        """Current timestep index of the bound state."""
        return self._state.step

    @property
    def config(self) -> GpuDecoderConfig:
        """The configuration of the underlying decoder."""
        return self._decoder.config


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

    A single long-lived ``GpuStreamingDecoder`` can serve **multiple
    concurrent requests** by operating on explicit :class:`StreamState`
    objects, or it can be used in the traditional single-request style
    via :meth:`init_stream` / :meth:`decode_chunk` / :meth:`finalize_stream`.

    Parameters
    ----------
    config : GpuDecoderConfig, optional
        Decoder configuration. Uses defaults if not provided.

    Examples
    --------
    **Single-request (backward-compatible):**

    >>> decoder = GpuStreamingDecoder(GpuDecoderConfig(beam_size=5))
    >>> decoder.init_stream(batch=1, vocab_size=5000)
    >>> decoder.decode_chunk(chunk)
    >>> result = decoder.finalize_stream()

    **Interleaved multi-request:**

    >>> decoder = GpuStreamingDecoder(GpuDecoderConfig(beam_size=5))
    >>> s1 = decoder.create_state(batch=1, vocab_size=5000)
    >>> s2 = decoder.create_state(batch=1, vocab_size=5000)
    >>> decoder.decode_chunk(chunk_a, state=s1)
    >>> decoder.decode_chunk(chunk_b, state=s2)
    >>> decoder.decode_chunk(chunk_c, state=s1)
    >>> r1 = decoder.finalize_stream(state=s1)
    >>> r2 = decoder.finalize_stream(state=s2)
    """

    def __init__(self, config: Optional[GpuDecoderConfig] = None,
                 *, use_cuda_graphs: bool = True):
        self._config = config or GpuDecoderConfig()
        self._mod = _get_ctc_decoder_module()

        # Internal default state for the single-request API
        self._state: Optional[StreamState] = None

        # Cache the (batch, vocab_size) → required_bytes mapping so
        # repeated calls with the same shape skip the cross-language
        # size query entirely.
        self._cached_size_key: Optional[tuple] = None
        self._cached_size_val: int = 0

        # Step 4: opt-in per-state CUDA Graph capture of ``streaming_step``.
        # When enabled, each ``StreamState``'s first ``decode_chunk`` call
        # captures four graphs (first/even/odd/blank); subsequent calls
        # replay the appropriate graph per frame, replacing ~6 kernel
        # launches with one D2D memcpy + one graph launch.  Falls back to
        # the eager path inside the C++ chunk launcher if capture fails.
        self._use_cuda_graphs: bool = bool(use_cuda_graphs)

        self._blank_log_thresh: Optional[float] = (
            math.log(self._config.blank_threshold)
            if 0.0 < self._config.blank_threshold < 1.0
            else None
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_bytes_for(self, batch: int, vocab_size: int) -> int:
        """Return the required state buffer size in bytes (cached)."""
        key = (batch, vocab_size)
        if self._cached_size_key == key:
            return self._cached_size_val
        cfg = self._config
        if cfg.use_paged_memory:
            val = int(self._mod.ctc_decoder_paged_state_size(
                batch, cfg.beam_size, vocab_size, cfg.max_seq_len, cfg.page_size))
        else:
            val = int(self._mod.ctc_decoder_state_size(
                batch, cfg.beam_size, vocab_size, cfg.max_seq_len))
        self._cached_size_key = key
        self._cached_size_val = val
        return val

    def _init_gpu_state(self, buffer: torch.Tensor,
                        batch: int, vocab_size: int) -> None:
        """Run the C++ state initializer on *buffer* (memset + header)."""
        cfg = self._config
        if cfg.use_paged_memory:
            self._mod.ctc_beam_search_init_state_paged(
                buffer, batch, cfg.beam_size,
                vocab_size, cfg.max_seq_len, cfg.blank_id, cfg.page_size)
        else:
            self._mod.ctc_beam_search_init_state(
                buffer, batch, cfg.beam_size,
                vocab_size, cfg.max_seq_len, cfg.blank_id)

    def _resolve_state(self, state: Optional[StreamState]) -> StreamState:
        """Return *state* if given, else fall back to the internal default."""
        s = state if state is not None else self._state
        if s is None:
            raise RuntimeError(
                "Call init_stream() or pass a StreamState created by create_state()")
        return s

    # ------------------------------------------------------------------
    # Explicit state management (multi-request / interleaved API)
    # ------------------------------------------------------------------

    def create_state(self, batch: int, vocab_size: int,
                     device: Optional[torch.device] = None) -> StreamState:
        """Create and initialize a new per-request state.

        The returned :class:`StreamState` can be passed to
        :meth:`decode_chunk` and :meth:`finalize_stream` via the
        ``state`` parameter, enabling interleaved chunk processing
        across many concurrent requests with a single decoder engine.

        Parameters
        ----------
        batch : int
            Batch size for this request.
        vocab_size : int
            Vocabulary size (must match log-prob last dimension).
        device : torch.device, optional
            CUDA device.  Defaults to ``torch.device("cuda")``.

        Returns
        -------
        StreamState
            Freshly allocated and initialized state ready for
            :meth:`decode_chunk`.
        """
        if device is None:
            device = torch.device("cuda")
        required = self._state_bytes_for(batch, vocab_size)
        buf = torch.empty(required, dtype=torch.uint8, device=device)
        self._init_gpu_state(buf, batch, vocab_size)
        return StreamState(buf, batch, vocab_size, required)

    def reset_state(self, state: StreamState,
                    batch: Optional[int] = None,
                    vocab_size: Optional[int] = None,
                    device: Optional[torch.device] = None) -> None:
        """Reinitialize an existing state for a new request.

        The GPU buffer inside *state* is reused when the new parameters
        fit; otherwise a fresh buffer is allocated.  This is the
        recommended way to recycle pooled states.

        Parameters
        ----------
        state : StreamState
            State to reinitialize (modified in place).
        batch : int, optional
            New batch size.  Defaults to the state's current value.
        vocab_size : int, optional
            New vocabulary size.  Defaults to the state's current value.
        device : torch.device, optional
            Target device.  Defaults to the state buffer's current device.
        """
        batch = batch if batch is not None else state.batch
        vocab_size = vocab_size if vocab_size is not None else state.vocab_size
        if device is None:
            device = state.buffer.device
        else:
            device = torch.device(device)
            # torch.empty(..., device=torch.device("cuda")) materialises on the
            # current CUDA device, so the buffer ends up with index 0 (or whatever
            # the active device is). Resolve the index here so the equality check
            # below doesn't reallocate when the caller passes an indexless "cuda".
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())

        required = self._state_bytes_for(batch, vocab_size)
        if state._buffer_bytes >= required and state.buffer.device == device:
            pass  # fast path — reuse existing buffer (graphs remain valid)
        else:
            # Reallocating the buffer invalidates any captured CUDA Graphs
            # bound to its old data_ptr. Release them before swapping so a
            # future state that lands at the same address doesn't pick up
            # stale graphs.
            try:
                self._mod.ctc_decoder_release_graphs(state.buffer)
            except Exception:
                pass
            state.buffer = torch.empty(required, dtype=torch.uint8, device=device)
            state._buffer_bytes = required

        state.batch = batch
        state.vocab_size = vocab_size
        state.step = 0
        state.actual_frame_idx = 0
        self._init_gpu_state(state.buffer, batch, vocab_size)

    # ------------------------------------------------------------------
    # Single-request convenience API (backward-compatible)
    # ------------------------------------------------------------------

    def init_stream(self, batch: int, vocab_size: int,
                    device: Optional[torch.device] = None) -> None:
        """Initialize (or re-initialize) the internal default state.

        Equivalent to ``create_state`` on first call, then
        ``reset_state`` on subsequent calls.  Exists for backward
        compatibility with code that treats the decoder as a
        single-request object.

        Parameters
        ----------
        batch : int
            Batch size.
        vocab_size : int
            Vocabulary size (must match log-prob last dimension).
        device : torch.device, optional
            CUDA device. Defaults to ``torch.device("cuda")``.
        """
        if self._state is None:
            self._state = self.create_state(batch, vocab_size, device)
        else:
            self.reset_state(self._state, batch, vocab_size, device)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_chunk(self, log_prob: torch.Tensor,
                     state: Optional[StreamState] = None) -> None:
        """Process one chunk of log-probability frames.

        Parameters
        ----------
        log_prob : torch.Tensor
            Log-probability tensor ``[batch, chunk_T, vocab_size]``
            on CUDA, float32.
        state : StreamState, optional
            Explicit per-request state.  When ``None`` the internal
            default state (set by :meth:`init_stream`) is used.
        """
        nvtx_push("ctc.decode_chunk")
        s = self._resolve_state(state)
        cfg = self._config
        log_prob = log_prob.contiguous().float()
        chunk_t = log_prob.size(1)
        if chunk_t == 0:
            nvtx_pop()
            return

        blank_log_thresh = self._blank_log_thresh

        # Batch blank-threshold check: a single GPU→CPU transfer for the
        # entire chunk replaces one .min().item() sync per frame.  The mask
        # is then handed to the C++ chunk launcher as-is so the per-frame
        # skip happens without a Python round-trip — and, critically, lets
        # the host loop skip ``streaming_step`` launches for blank frames,
        # which is a large net win on streaming workloads even though the
        # mask materialisation itself is a small D2H sync.
        if blank_log_thresh is not None:
            is_speech_mask = (
                log_prob[:, :, cfg.blank_id]
                .min(dim=0).values
                .lt_(blank_log_thresh)
                .to(torch.uint8)
                .cpu()
            )
        else:
            is_speech_mask = torch.empty(0, dtype=torch.uint8)

        new_step = int(self._mod.ctc_beam_search_chunk(
            s.buffer, log_prob, is_speech_mask,
            cfg.beam_size, cfg.blank_id,
            s.step, cfg.blank_threshold,
            s.actual_frame_idx,
            s.batch, s.vocab_size, cfg.max_seq_len,
            1 if cfg.use_paged_memory else 0, cfg.page_size,
            1 if self._use_cuda_graphs else 0,
        ))

        # The chunk launcher decodes ``min(active_frames, max_seq_len -
        # start_step)`` frames; we know ``frame_idx`` advances by the chunk
        # length when the cap isn't hit, and by ``new_step - start_step + (
        # blanks before truncation)`` otherwise.  In practice the cap is
        # never hit during normal streaming so the simple formula below is
        # correct; the worst case (cap hit mid-chunk) leaves frame_idx
        # slightly low which only affects the optional space-id timestamp
        # tracking, not transcription.
        s.step = new_step
        s.actual_frame_idx += chunk_t
        nvtx_pop()

    def decode_chunk_batch(
        self,
        log_probs: torch.Tensor,
        states: List[StreamState],
        is_speech_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Process one chunk of log-probabilities for **many streams at once**.

        Replaces the per-stream Python loop in the engine — instead of N
        ``decode_chunk`` calls (each crossing the Python→C++ boundary), the
        whole ready set is handed to a single C++ launcher that iterates
        streams + frames internally.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log-probability tensor ``[N, chunk_T, vocab_size]`` on CUDA,
            float32.  ``N`` must equal ``len(states)``.
        states : list of StreamState
            One :class:`StreamState` per row of ``log_probs``, in the same
            order.  Each state's ``step`` and ``actual_frame_idx`` are
            updated in place after the call.
        is_speech_mask : torch.Tensor, optional
            CPU uint8 tensor ``[N, chunk_T]``; rows of 0 mark blank frames
            that should be skipped without launching a step.  Pass ``None``
            to decode every frame.
        """
        nvtx_push("ctc.decode_chunk_batch")
        n = len(states)
        if n == 0:
            nvtx_pop()
            return
        if log_probs.size(0) != n:
            raise ValueError(
                f"log_probs.shape[0] ({log_probs.size(0)}) must equal "
                f"len(states) ({n})")

        chunk_t = log_probs.size(1)
        if chunk_t == 0:
            nvtx_pop()
            return

        log_probs = log_probs.contiguous().float()

        cfg = self._config
        # Validate the homogeneous-config assumption: every state was created
        # by this same decoder, so its (batch, vocab_size) is fixed.
        batch = states[0].batch
        vocab_size = states[0].vocab_size
        for s in states[1:]:
            if s.batch != batch or s.vocab_size != vocab_size:
                raise ValueError(
                    "decode_chunk_batch: all states must share (batch, "
                    "vocab_size); got mixed values")

        state_ptrs = torch.tensor(
            [s.buffer.data_ptr() for s in states],
            dtype=torch.int64, device="cpu")
        start_steps = torch.tensor(
            [s.step for s in states], dtype=torch.int32, device="cpu")
        start_frame_idxs = torch.tensor(
            [s.actual_frame_idx for s in states],
            dtype=torch.int32, device="cpu")

        if is_speech_mask is None:
            # Mirror the single-state path: when blank_threshold is set,
            # auto-compute a per-(stream, frame) mask so blank-dominant
            # frames skip without launching a kernel.  ``min(dim=batch)``
            # in the single-state path becomes a no-op for batch=1, so
            # the streaming batched form just gates on the per-row blank
            # log-prob directly.
            blank_log_thresh = self._blank_log_thresh
            if blank_log_thresh is not None:
                mask_tensor = (
                    log_probs[:, :, cfg.blank_id]
                    .lt(blank_log_thresh)
                    .to(torch.uint8)
                    .cpu()
                    .contiguous()
                )  # (N, T)
            else:
                mask_tensor = torch.empty(0, dtype=torch.uint8, device="cpu")
        else:
            mask_tensor = is_speech_mask.to(
                dtype=torch.uint8, device="cpu", copy=False).contiguous()
            if mask_tensor.dim() != 2 or mask_tensor.size(0) != n or \
                    mask_tensor.size(1) != chunk_t:
                raise ValueError(
                    f"is_speech_mask must be (N={n}, chunk_T={chunk_t}); "
                    f"got {tuple(mask_tensor.shape)}")

        self._mod.ctc_beam_search_chunk_batched(
            state_ptrs, log_probs, mask_tensor,
            start_steps, start_frame_idxs,
            cfg.beam_size, cfg.blank_id, cfg.blank_threshold,
            batch, vocab_size, cfg.max_seq_len,
            1 if cfg.use_paged_memory else 0, cfg.page_size,
            1 if self._use_cuda_graphs else 0,
        )

        # Mirror the per-frame counter advance the single-state path does:
        # the C++ launcher decoded ``min(active_frames, max_seq_len - step)``
        # non-blank frames; ``actual_frame_idx`` advances by the full chunk
        # length when the step cap isn't hit (the common case during normal
        # streaming).  Writing it as ``+= chunk_t`` matches the existing
        # ``decode_chunk`` implementation above.
        new_steps = start_steps.tolist()
        for i, s in enumerate(states):
            s.step = int(new_steps[i])
            s.actual_frame_idx += chunk_t
        nvtx_pop()

    def finalize_stream(
        self, state: Optional[StreamState] = None,
    ) -> GpuDecoderResult:
        """Finalize streaming decoding and return results.

        Parameters
        ----------
        state : StreamState, optional
            Explicit per-request state.  When ``None`` the internal
            default state is used.

        Returns
        -------
        GpuDecoderResult
            Decoded tokens, lengths, and scores for each batch and beam.
        """
        s = self._resolve_state(state)
        cfg = self._config
        device = s.buffer.device
        batch = s.batch

        out_tokens = torch.empty(
            batch, cfg.beam_size, cfg.max_seq_len,
            dtype=torch.int32, device=device)
        out_lengths = torch.empty(
            batch, cfg.beam_size, dtype=torch.int32, device=device)
        out_scores = torch.empty(
            batch, cfg.beam_size, dtype=torch.float32, device=device)

        use_paged = 1 if cfg.use_paged_memory else 0
        self._mod.ctc_beam_search_read_state(
            out_tokens, out_lengths, out_scores,
            s.buffer, s.step,
            batch, cfg.beam_size, s.vocab_size, cfg.max_seq_len,
            use_paged, cfg.page_size)

        tokens = _extract_tokens(out_tokens, out_lengths, batch, cfg.beam_size)
        return GpuDecoderResult(tokens=tokens, lengths=out_lengths, scores=out_scores)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> GpuDecoderConfig:
        """The configuration used to build this decoder."""
        return self._config

    @property
    def step(self) -> int:
        """Current timestep of the internal default state."""
        return self._state.step if self._state is not None else 0
