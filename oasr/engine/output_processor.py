# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CTC decoding and detokenization for the ASR engine."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from oasr.ctc_decode import GpuDecoderResult, ctc_beam_search_decode
from oasr.decode import Decoder, DecoderResult
from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .config import EngineConfig
from .request import Request, RequestOutput

logger = logging.getLogger(__name__)

# Token IDs to strip during detokenization
_SPECIAL_IDS = frozenset([0, 1, 2])  # <blank>, <unk>, <sos/eos>


class OutputProcessor:
    """Converts raw CTC log-probabilities into detokenized text.

    Supports two GPU decoder types controlled by ``config.decoder_type``:

    * ``"ctc_cuda"`` — GPU CTC prefix beam search via
      :func:`~oasr.ctc_decode.ctc_beam_search_decode` (offline) or
      :class:`~oasr.ctc_decode.GpuStreamingDecoder` (streaming).
    * ``"ctc_wfst"`` — k2 WFST beam search (GPU; requires a k2 build).

    Detokenization uses a SentencePiece model when available, falling back to
    a plain character join using ``units.txt``.

    Parameters
    ----------
    config : EngineConfig
        Engine configuration.
    decode_type : str
        The model's decode family (``model.decode_type``).  Only ``"ctc"`` is
        wired today; ``"transducer"`` / ``"aed"`` are reserved extension points
        and raise until their decode paths are implemented.
    """

    def __init__(self, config: EngineConfig, decode_type: str = "ctc") -> None:
        if decode_type != "ctc":
            raise NotImplementedError(
                f"OutputProcessor only supports CTC decoding; got "
                f"decode_type={decode_type!r}. Transducer/AED decode paths are a "
                "planned extension point (add a branch here keyed on decode_type)."
            )
        self._decode_type = decode_type
        self._config = config
        # Streaming decode-step counter driving ``partial_decode_interval``.
        self._stream_decode_step = 0
        # Overlapped interim read-back: the previous emit step's in-flight
        # ``peek_states_async`` handle plus the requests it covers and the
        # decoder that issued it.  Consumed one step later so the blocking
        # device→host sync leaves the critical path (see ``decode_streaming_batch``).
        self._pending_peek = None  # type: ignore[var-annotated]
        self._sp = self._load_sentencepiece(config.sentencepiece_model)
        self._vocab: Optional[Dict[int, str]] = None
        if config.unit_table is not None:
            self._vocab = self._load_unit_table(config.unit_table)

    # ------------------------------------------------------------------
    # Offline decoding
    # ------------------------------------------------------------------

    def decode_offline(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        """Decode a batched log-probability tensor from an offline forward pass.

        Parameters
        ----------
        log_probs : Tensor
            ``(B, T, V)`` float32 log-probabilities on CUDA.
        lengths : Tensor
            ``(B,)`` valid encoder output lengths (int32 on CUDA).

        Returns
        -------
        List[RequestOutput]
            One output per batch element (best hypothesis, finished=True).
        """
        if self._config.decoder_type == "ctc_cuda":
            return self._decode_offline_ctc(log_probs, lengths)
        return self._decode_offline_wfst(log_probs, lengths)

    def _decode_offline_ctc(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        cfg = self._config.ctc_decoder_config
        assert cfg is not None
        result: GpuDecoderResult = ctc_beam_search_decode(
            log_probs,
            lengths,
            beam_size=cfg.beam_size,
            blank_id=cfg.blank_id,
            blank_threshold=cfg.blank_threshold,
            max_seq_len=cfg.max_seq_len,
            use_paged_memory=cfg.use_paged_memory,
            page_size=cfg.page_size,
        )
        outputs = []
        scores_t = result.scores.cpu().tolist() if result.scores is not None else None
        for b in range(log_probs.size(0)):
            token_seqs = result.tokens[b]  # list of beam token lists
            best_tokens = token_seqs[0] if token_seqs else []
            beam_scores = scores_t[b] if scores_t is not None else None
            text = self.detokenize(best_tokens)
            outputs.append(RequestOutput(
                request_id="",
                text=text,
                tokens=token_seqs,
                scores=beam_scores,
                finished=True,
            ))
        return outputs

    def _decode_offline_wfst(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        cfg = self._config.wfst_decoder_config
        decoder = Decoder(cfg, fst=self._config.fst_path)

        lengths_list = lengths.cpu().tolist()
        outputs = []
        for b in range(log_probs.size(0)):
            t = int(lengths_list[b])
            logp = log_probs[b, :t, :]  # (T, V)
            result: DecoderResult = decoder.decode(logp)
            best = result.tokens[0] if result.tokens else []
            text = self.detokenize(best)
            outputs.append(RequestOutput(
                request_id="",
                text=text,
                tokens=result.tokens,
                scores=result.scores,
                finished=True,
            ))
        return outputs

    # ------------------------------------------------------------------
    # Streaming decoding
    # ------------------------------------------------------------------

    def decode_streaming_batch(
        self,
        requests: List[Request],
        log_probs_map: Dict[str, torch.Tensor],
    ) -> List[RequestOutput]:
        """Batched streaming decode for **N ready streams** in one call.

        For the ``ctc_cuda`` decoder we issue a single C++ launcher
        (:meth:`~oasr.ctc_decode.GpuStreamingDecoder.decode_chunk_batch`)
        over all ready streams at once, replacing the per-stream Python
        loop in :meth:`~oasr.engine.executor.streaming.StreamingExecutor.step`.
        For the ``ctc_wfst`` decoder we fall back to a per-stream Python
        loop here — the k2 decoder is single-threaded per-request anyway.

        Parameters
        ----------
        requests : list of Request
            Streams whose features are ready this step, in the order
            their log-prob slices appear in ``log_probs_map``.
        log_probs_map : dict
            ``{request_id: tensor(1, T_chunk, V)}`` from
            ``ModelRunner.forward_streaming_step``.

        Returns
        -------
        list of RequestOutput
            One partial output per request that had log-probs this step.
        """
        if not requests:
            return []
        if self._config.decoder_type != "ctc_cuda":
            outputs: List[RequestOutput] = []
            for req in requests:
                lp = log_probs_map.get(req.request_id)
                if lp is not None:
                    outputs.append(self.decode_streaming_chunk(req, lp))
            return outputs

        # GPU fast path — gather N streams that have log-probs this step
        # and run one batched chunk launch.  Streams whose chunk T differs
        # (e.g. a final-window stream with a short tail) are processed in
        # a separate batched call per ``T`` group — torch.cat can't stack
        # tensors with mismatched T.  Typical workloads see one big group
        # (the lockstep batched cohort) plus zero or one small groups.
        from collections import defaultdict
        groups: Dict[int, List[Request]] = defaultdict(list)
        group_logp: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for req in requests:
            lp = log_probs_map.get(req.request_id)
            if lp is None:
                continue
            assert req.stream_context is not None, \
                "stream_context must be allocated before decoding"
            t_chunk = lp.size(1)
            groups[t_chunk].append(req)
            group_logp[t_chunk].append(lp)
        if not groups:
            return []

        # Always advance the decode state for every ready stream (one batched
        # C++ launch per distinct chunk-T).  Collect the (req, state) pairs in a
        # stable order so the optional interim read-back can be done in a single
        # batched device→host sync below.
        self._stream_decode_step += 1
        nvtx_push("decode_advance")
        decoder = None
        ordered_reqs: List[Request] = []
        ordered_states = []
        for t_chunk, reqs in groups.items():
            log_probs_batch = torch.cat(group_logp[t_chunk], dim=0)
            states = [r.stream_context.get_ctc_state() for r in reqs]
            if decoder is None:
                decoder = reqs[0].stream_context.ctc_state_manager.decoder
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
        # transcript still comes from the blocking ``finalize_streaming``.
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
                partials.append(RequestOutput(
                    request_id=req.request_id,
                    text=self.detokenize(best),
                    tokens=snap.tokens[0] if snap.tokens else [],
                    finished=False,
                ))
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
        (``stream_context is None``) — their final transcript has already been
        emitted, so a stale interim partial must not follow it.
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
            if req.stream_context is None:
                continue  # finalised since issue — final already emitted
            best = snap.tokens[0][0] if snap.tokens and snap.tokens[0] else []
            partials.append(RequestOutput(
                request_id=req.request_id,
                text=self.detokenize(best),
                tokens=snap.tokens[0] if snap.tokens else [],
                finished=False,
            ))
        return partials

    def decode_streaming_chunk(
        self,
        request: Request,
        log_probs: torch.Tensor,
    ) -> RequestOutput:
        """Feed one chunk of log-probs to the streaming decoder.

        Parameters
        ----------
        request : Request
            The active streaming request (must have a ``stream_context``).
        log_probs : Tensor
            ``(1, T_chunk, V)`` log-probabilities for this chunk.

        Returns
        -------
        RequestOutput
            Partial output with ``finished=False``.
        """
        ctx = request.stream_context
        assert ctx is not None, "stream_context must be allocated before decoding"

        if self._config.decoder_type == "ctc_cuda":
            handle = ctx.get_decoder()
            handle.decode_chunk(log_probs)
            # ``peek`` is a non-destructive D2D snapshot of the beam buffer;
            # surfaces the best-so-far hypothesis without finalising the
            # stream.
            snap = handle.peek()
            best = snap.tokens[0][0] if snap.tokens and snap.tokens[0] else []
            return RequestOutput(
                request_id=request.request_id,
                text=self.detokenize(best),
                tokens=snap.tokens[0] if snap.tokens else [],
                finished=False,
            )
        else:
            # k2 WFST streaming decoder stored on the request
            if not hasattr(request, "_wfst_decoder"):
                request._wfst_decoder = Decoder(
                    self._config.wfst_decoder_config, fst=self._config.fst_path)
                request._wfst_decoder.init_stream()

            # log_probs is (1, T_chunk, V) -- remove batch dim
            chunk_logp = log_probs.squeeze(0)
            result: DecoderResult = request._wfst_decoder.decode_chunk(
                chunk_logp)
            best = result.tokens[0] if result.tokens else []
            return RequestOutput(
                request_id=request.request_id,
                text=self.detokenize(best),
                tokens=result.tokens,
                scores=result.scores,
                finished=False,
            )

    def finalize_streaming(self, request: Request) -> RequestOutput:
        """Finalize streaming decoding and return the complete transcript.

        Parameters
        ----------
        request : Request
            The streaming request to finalize.

        Returns
        -------
        RequestOutput
            Final output with ``finished=True``.
        """
        if self._config.decoder_type == "ctc_cuda":
            ctx = request.stream_context
            assert ctx is not None
            handle = ctx.get_decoder()
            result: GpuDecoderResult = handle.finalize_stream()
            token_seqs = result.tokens[0] if result.tokens else []
            best = token_seqs[0] if token_seqs else []
            beam_scores = result.scores.cpu().tolist(
            )[0] if result.scores is not None else None
            text = self.detokenize(best)
            return RequestOutput(
                request_id=request.request_id,
                text=text,
                tokens=token_seqs,
                scores=beam_scores,
                finished=True,
            )
        else:
            wfst_dec = getattr(request, "_wfst_decoder", None)
            if wfst_dec is None:
                # No chunks were decoded (empty audio)
                return RequestOutput(
                    request_id=request.request_id,
                    text="",
                    tokens=[],
                    finished=True,
                )
            result_wfst: DecoderResult = wfst_dec.finalize_stream()
            best = result_wfst.tokens[0] if result_wfst.tokens else []
            text = self.detokenize(best)
            return RequestOutput(
                request_id=request.request_id,
                text=text,
                tokens=result_wfst.tokens,
                scores=result_wfst.scores,
                finished=True,
            )

    # ------------------------------------------------------------------
    # Detokenization
    # ------------------------------------------------------------------

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a text string.

        Strips special tokens (blank, unk, sos/eos).  Uses ``units.txt`` to
        look up BPE piece strings, then joins them treating ``▁`` (U+2581) as
        a word boundary.  The SentencePiece model is **not** used for decoding
        because its internal piece IDs differ from the CTC output IDs (which
        come from ``units.txt``).

        Parameters
        ----------
        token_ids : List[int]
            CTC output token sequence.
        """
        filtered = [t for t in token_ids if t not in _SPECIAL_IDS]
        if not filtered:
            return ""

        if self._vocab is not None:
            pieces = [self._vocab.get(t, "") for t in filtered]
            text = "".join(pieces)
            return text.replace("\u2581", " ").strip()

        # Last resort: join as-is
        return " ".join(str(t) for t in filtered)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sentencepiece(path: Optional[str]):
        if path is None:
            return None
        try:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.Load(path)
            return sp
        except Exception as exc:
            logger.warning(
                "Could not load SentencePiece model %s: %s", path, exc)
            return None

    @staticmethod
    def _load_unit_table(path: str) -> Dict[int, str]:
        vocab: Dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    token, idx = parts[0], int(parts[1])
                    vocab[idx] = token
        return vocab
