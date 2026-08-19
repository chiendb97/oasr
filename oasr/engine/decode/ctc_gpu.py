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
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Set, Tuple

import torch

from oasr.cache.ctc_state import CtcStateCacheManager
from oasr.ctc_decode import GpuDecoderConfig, GpuDecoderResult, ctc_beam_search_decode
from oasr.utils.nvtx import nvtx_pop, nvtx_push

from ..request import Request, RequestOutput
from .alignment import wants_word_timings
from .base import DecodeStrategy, register_decode_strategy
from .ctc_align import attach_emission_timings
from .options import option_factory

if TYPE_CHECKING:
    from ..config import EngineConfig
    from .detokenize import Detokenizer


@dataclass(frozen=True)
class CtcGpuOptions:
    """Options for ``decoder_type="ctc_cuda"``.

    ``decoder_config`` is built lazily by the factory, so an engine running a
    non-CTC family never constructs a beam config it will not read — that
    instantiation used to happen in ``EngineConfig.__post_init__`` for *every*
    engine, Whisper and speech-LLM included.
    """

    decoder_config: GpuDecoderConfig = option_factory(
        GpuDecoderConfig,
        legacy="ctc_decoder_config",
        doc="GPU prefix-beam-search config (beam_size, blank_id, ...).",
    )


@register_decode_strategy("ctc_cuda")
class CtcGpuDecodeStrategy(DecodeStrategy):
    """CTC decoding on the GPU (prefix beam search)."""

    decode_type: ClassVar[str] = "ctc"
    consumes: ClassVar[str] = "log_probs"
    options_cls: ClassVar[type] = CtcGpuOptions

    @property
    def word_timing_modes(self) -> Tuple[str, ...]:
        """Both modes.

        The beam records the encoder frame it emitted each token at, as it
        emits it (``ctc_decoder.cuh``'s ``ctime`` / ``time_storage``), so asking
        for word timings is a read rather than a second pass — and works for a
        stream, whose log-probs are long gone by the time the transcript is
        final.
        """
        return ("offline", "streaming")

    def __init__(self, config: "EngineConfig", detok: "Detokenizer", model=None) -> None:
        super().__init__(config, detok, model)
        self._device = torch.device(config.device)
        mcfg = getattr(config, "_model_config", None)
        vocab = getattr(mcfg, "vocab_size", None) if mcfg is not None else None
        if vocab is None:
            # No magic number: the beam state is sized by this, and a wrong
            # value is either an out-of-bounds read or silently truncated
            # vocabulary.  The engine always stamps ``_model_config`` before
            # building a strategy, so reaching here means a caller constructed
            # one by hand without it.
            raise ValueError(
                "CtcGpuDecodeStrategy needs the model's vocab_size; "
                "EngineConfig._model_config is unset (the engine sets it after "
                "loading the checkpoint)."
            )
        self._vocab_size = int(vocab)

        # CTC-graph capture is gated by both the global and CTC-specific flags
        # (and CUDA).  Defaults here mirror ``EngineConfig``: ``use_cuda_graphs``
        # is True, ``use_ctc_cuda_graphs`` is **False** — the getattr fallback
        # used to say True for both, so a config object without the field would
        # have silently enabled a path the engine keeps off by default.
        self._ctc_graphs_enabled = (
            bool(getattr(config, "use_cuda_graphs", True))
            and bool(getattr(config, "use_ctc_cuda_graphs", False))
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
        self,
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
        requests: Optional[List[Request]] = None,
    ) -> List[RequestOutput]:
        cfg = self.options.decoder_config
        want_times = requests is not None
        result: GpuDecoderResult = ctc_beam_search_decode(
            enc_out,
            enc_lengths,
            beam_size=cfg.beam_size,
            blank_id=cfg.blank_id,
            blank_threshold=cfg.blank_threshold,
            max_seq_len=cfg.max_seq_len,
            use_paged_memory=cfg.use_paged_memory,
            page_size=cfg.page_size,
            want_times=want_times,
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
        if want_times:
            attach_emission_timings(self, requests or [], outputs, result.times, enc_out)
        return outputs

    # ------------------------------------------------------------------
    # Streaming session lifecycle
    # ------------------------------------------------------------------

    def _ensure_ctc_mgr(self) -> CtcStateCacheManager:
        if self._ctc_mgr is None:
            self._ctc_mgr = CtcStateCacheManager(
                self.options.decoder_config,
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

        # Interval zero disables partial readback; larger values reduce its sync
        # cost. Overlapped readback emits the previous interval's result, so only
        # partials lag one interval; finalization remains synchronous.
        interval = getattr(self._config, "partial_decode_interval", 1)
        if interval < 1 or (self._stream_decode_step % interval) != 0:
            return []

        nvtx_push("partial_readback")
        if not getattr(self._config, "overlap_partial_readback", False):
            # Blocking path emits immediately and reads only the best hypothesis;
            # partial responses discard the remaining beams.
            bests = decoder.peek_states_best(ordered_states)
            partials = [
                RequestOutput(
                    request_id=req.request_id,
                    text=self._detok.detokenize(best),
                    tokens=[best],
                    finished=False,
                )
                for req, best in zip(ordered_reqs, bests)
            ]
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
                    # One row, same as the blocking path: which read-back an
                    # engine runs is a throughput choice and must not change
                    # what a client sees.
                    tokens=[best],
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
        want_times = wants_word_timings(request)
        handle = self._ensure_ctc_mgr().get_decoder(sid)
        result: GpuDecoderResult = handle.finalize_stream(want_times=want_times)
        token_seqs = result.tokens[0] if result.tokens else []
        best = token_seqs[0] if token_seqs else []
        beam_scores = result.scores.cpu().tolist()[0] if result.scores is not None else None
        text = self._detok.detokenize(best)
        out = RequestOutput(
            request_id=request.request_id,
            text=text,
            tokens=token_seqs,
            scores=beam_scores,
            finished=True,
        )
        if want_times and result.times and result.times[0]:
            # The frames are stream-absolute, recorded as the beam decoded each
            # chunk — no log-probs are retained, which is why this works in
            # streaming at all.  Confidences are not available here for the same
            # reason: the distribution those frames were chosen from is gone.
            frames = result.times[0][0]
            if len(frames) == len(best):
                self.attach_emission_alignment(out, best, frames)
        return out
