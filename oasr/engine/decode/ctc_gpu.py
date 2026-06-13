# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GPU CTC prefix-beam-search decode strategy.

Wraps :func:`~oasr.ctc_decode.ctc_beam_search_decode` (offline) and a shared
:class:`~oasr.ctc_decode.GpuStreamingDecoder` (streaming, via
:class:`~oasr.cache.ctc_state.CtcStateCacheManager`).  Owns its per-request beam
state so it works regardless of the encoder's streaming kind (paged Conformer or
stateful Zipformer) — the CTC beam state is decode-side, independent of the
encoder cache.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Set

import torch

from oasr.cache.ctc_state import CtcStateCacheManager
from oasr.ctc_decode import GpuDecoderResult, ctc_beam_search_decode
from oasr.utils.nvtx import nvtx_pop, nvtx_push

from ..request import Request, RequestOutput
from .base import DecodeStrategy, register_decode_strategy

if TYPE_CHECKING:
    from ..config import EngineConfig
    from .detokenize import Detokenizer


@register_decode_strategy("ctc_cuda")
class CtcGpuDecodeStrategy(DecodeStrategy):
    """CTC decoding on the GPU (prefix beam search)."""

    decode_type: ClassVar[str] = "ctc"
    consumes: ClassVar[str] = "log_probs"

    def __init__(self, config: "EngineConfig", detok: "Detokenizer") -> None:
        self._config = config
        self._detok = detok
        self._device = torch.device(config.device)
        mcfg = getattr(config, "_model_config", None)
        self._vocab_size = (getattr(mcfg, "vocab_size", None) or 5002) if mcfg else 5002

        # CTC-graph capture is gated by both the global and CTC-specific flags
        # (and CUDA).  Matches the prior ``ModelRunner`` gating exactly.
        self._ctc_graphs_enabled = (
            bool(getattr(config, "use_cuda_graphs", True))
            and bool(getattr(config, "use_ctc_cuda_graphs", True))
            and self._device.type == "cuda"
        )
        # Per-request beam state, built lazily on first streaming admission so
        # the offline path (which decodes via ``ctc_beam_search_decode``) never
        # constructs a streaming decoder.
        self._ctc_mgr: Optional[CtcStateCacheManager] = None
        self._sessions: Set[int] = set()

        # Streaming interim-partial cadence (lifted from OutputProcessor).
        self._stream_decode_step = 0
        self._pending_peek = None  # type: ignore[var-annotated]

    # ------------------------------------------------------------------
    # Offline
    # ------------------------------------------------------------------

    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        cfg = self._config.ctc_decoder_config
        assert cfg is not None
        result: GpuDecoderResult = ctc_beam_search_decode(
            enc_out,
            enc_lengths,
            beam_size=cfg.beam_size,
            blank_id=cfg.blank_id,
            blank_threshold=cfg.blank_threshold,
            max_seq_len=cfg.max_seq_len,
            use_paged_memory=cfg.use_paged_memory,
            page_size=cfg.page_size,
        )
        outputs = []
        scores_t = result.scores.cpu().tolist() if result.scores is not None else None
        for b in range(enc_out.size(0)):
            token_seqs = result.tokens[b]  # list of beam token lists
            best_tokens = token_seqs[0] if token_seqs else []
            beam_scores = scores_t[b] if scores_t is not None else None
            text = self._detok.detokenize(best_tokens)
            outputs.append(
                RequestOutput(
                    request_id="",
                    text=text,
                    tokens=token_seqs,
                    scores=beam_scores,
                    finished=True,
                )
            )
        return outputs

    # ------------------------------------------------------------------
    # Streaming session lifecycle
    # ------------------------------------------------------------------

    def _ensure_ctc_mgr(self) -> CtcStateCacheManager:
        if self._ctc_mgr is None:
            self._ctc_mgr = CtcStateCacheManager(
                self._config.ctc_decoder_config,
                use_cuda_graphs=self._ctc_graphs_enabled,
            )
        return self._ctc_mgr

    def create_session(self, request: Request) -> None:
        sid = request.stream_id
        assert sid is not None, "stream_id must be assigned before create_session"
        self._ensure_ctc_mgr().allocate_stream(
            sid, batch=1, vocab_size=self._vocab_size, device=self._device
        )
        self._sessions.add(sid)

    def free_session(self, request: Request) -> None:
        sid = request.stream_id
        if sid is not None and sid in self._sessions:
            assert self._ctc_mgr is not None
            self._ctc_mgr.free_stream(sid)
            self._sessions.discard(sid)

    # ------------------------------------------------------------------
    # Streaming decode
    # ------------------------------------------------------------------

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        """Batched streaming decode for N ready streams in one launch.

        Groups ready streams by chunk-T (``torch.cat`` can't stack mismatched
        T), runs one batched ``decode_chunk_batch`` per group, then optionally
        emits interim partials on the ``partial_decode_interval`` cadence.
        """
        if not requests:
            return []
        ctc_mgr = self._ensure_ctc_mgr()

        groups: Dict[int, List[Request]] = defaultdict(list)
        group_logp: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for req in requests:
            lp = enc_out_map.get(req.request_id)
            if lp is None:
                continue
            assert req.stream_id is not None, "stream_id must be allocated before decoding"
            t_chunk = lp.size(1)
            groups[t_chunk].append(req)
            group_logp[t_chunk].append(lp)
        if not groups:
            return []

        # Advance every ready stream (one batched C++ launch per distinct
        # chunk-T).  Collect (req, state) in a stable order so the optional
        # interim read-back is a single batched device→host sync below.
        self._stream_decode_step += 1
        nvtx_push("decode_advance")
        decoder = ctc_mgr.decoder
        ordered_reqs: List[Request] = []
        ordered_states = []
        for t_chunk, reqs in groups.items():
            log_probs_batch = torch.cat(group_logp[t_chunk], dim=0)
            states = ctc_mgr.get_states([r.stream_id for r in reqs])  # type: ignore[arg-type]
            decoder.decode_chunk_batch(log_probs_batch, states)
            ordered_reqs.extend(reqs)
            ordered_states.extend(states)
        nvtx_pop()  # decode_advance

        # Interim-partial cadence.  Reading the beam buffer back to the host is
        # a blocking ``cudaStreamSynchronize`` that, profiled, costs about as
        # much as the decode compute itself — and it drains the GPU before the
        # next step's encoder can be dispatched.  We *overlap* it: each emit
        # step issues a **non-blocking** batched read-back (``peek_states_async``)
        # for the current ready set, and emits the partials collected from the
        # **previous** emit step's handle (whose copy completed during the
        # intervening step).  Partials therefore lag exactly one emit step
        # (~one chunk), which the interactive contract allows; the final
        # transcript still comes from the blocking ``finalize``.
        # ``partial_decode_interval <= 0`` skips interim partials entirely
        # (decode state still advances; only the read-back is skipped).
        interval = getattr(self._config, "partial_decode_interval", 1)
        if interval < 1 or (self._stream_decode_step % interval) != 0:
            return []

        nvtx_push("partial_readback")
        if not getattr(self._config, "overlap_partial_readback", False):
            # Default: blocking read-back, emit this step's partial now (lowest
            # first-token latency — the interactive path).
            snaps = decoder.peek_states(ordered_states)
            partials = []
            for req, snap in zip(ordered_reqs, snaps):
                best = snap.tokens[0][0] if snap.tokens and snap.tokens[0] else []
                partials.append(
                    RequestOutput(
                        request_id=req.request_id,
                        text=self._detok.detokenize(best),
                        tokens=snap.tokens[0] if snap.tokens else [],
                        finished=False,
                    )
                )
            nvtx_pop()  # partial_readback
            return partials
        # Opt-in: overlapped (non-blocking) read-back — emit the previous emit
        # step's partial (one-chunk lag), issue this step's async read-back for
        # collection next time.  Backlog/throughput mode.
        partials = self._collect_pending_partials()
        handle = decoder.peek_states_async(ordered_states)
        self._pending_peek = (ordered_reqs, handle, decoder)
        nvtx_pop()  # partial_readback
        return partials

    def _collect_pending_partials(self) -> List[RequestOutput]:
        """Materialise the previous emit step's overlapped read-back, if any.

        Skips requests whose stream was finalised in the meantime
        (``stream_id`` no longer has a session) — their final transcript has
        already been emitted, so a stale interim partial must not follow it.
        """
        if self._pending_peek is None:
            return []
        prev_reqs, handle, decoder = self._pending_peek
        self._pending_peek = None
        if handle is None or not prev_reqs:
            return []
        snaps = decoder.peek_states_collect(handle)
        partials: List[RequestOutput] = []
        for req, snap in zip(prev_reqs, snaps):
            if req.stream_id not in self._sessions:
                continue  # finalised since issue — final already emitted
            best = snap.tokens[0][0] if snap.tokens and snap.tokens[0] else []
            partials.append(
                RequestOutput(
                    request_id=req.request_id,
                    text=self._detok.detokenize(best),
                    tokens=snap.tokens[0] if snap.tokens else [],
                    finished=False,
                )
            )
        return partials

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        """Feed one chunk of log-probs to a single stream's decoder."""
        sid = request.stream_id
        assert sid is not None, "stream_id must be allocated before decoding"
        handle = self._ensure_ctc_mgr().get_decoder(sid)
        handle.decode_chunk(enc_out)
        # ``peek`` is a non-destructive D2D snapshot of the beam buffer.
        snap = handle.peek()
        best = snap.tokens[0][0] if snap.tokens and snap.tokens[0] else []
        return RequestOutput(
            request_id=request.request_id,
            text=self._detok.detokenize(best),
            tokens=snap.tokens[0] if snap.tokens else [],
            finished=False,
        )

    def finalize(self, request: Request) -> RequestOutput:
        sid = request.stream_id
        assert sid is not None
        handle = self._ensure_ctc_mgr().get_decoder(sid)
        result: GpuDecoderResult = handle.finalize_stream()
        token_seqs = result.tokens[0] if result.tokens else []
        best = token_seqs[0] if token_seqs else []
        beam_scores = result.scores.cpu().tolist()[0] if result.scores is not None else None
        text = self._detok.detokenize(best)
        return RequestOutput(
            request_id=request.request_id,
            text=text,
            tokens=token_seqs,
            scores=beam_scores,
            finished=True,
        )
