# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CTC decoding and detokenization for the ASR engine."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from oasr.ctc_decode import GpuDecoderResult, ctc_beam_search_decode
from oasr.decode import Decoder, DecoderResult

from .config import EngineConfig
from .request import Request, RequestOutput

logger = logging.getLogger(__name__)

# Token IDs to strip during detokenization
_SPECIAL_IDS = frozenset([0, 1, 2])  # <blank>, <unk>, <sos/eos>


class OutputProcessor:
    """Converts raw CTC log-probabilities into detokenized text.

    Supports four decoder types controlled by ``config.decoder_type``:

    * ``"ctc_greedy"`` — fast CPU greedy collapse.
    * ``"ctc_prefix_beam"`` — CPU prefix beam search.
    * ``"ctc_gpu"`` — GPU CTC prefix beam search via
      :func:`~oasr.ctc_decode.ctc_beam_search_decode` (offline) or
      :class:`~oasr.ctc_decode.GpuStreamingDecoder` (streaming).
    * ``"ctc_wfst"`` — CPU WFST beam search (requires k2 build).

    Detokenization uses a SentencePiece model when available, falling back to
    a plain character join using ``units.txt``.

    Parameters
    ----------
    config : EngineConfig
        Engine configuration.
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
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
        dtype = self._config.decoder_type

        if dtype == "ctc_gpu":
            return self._decode_offline_gpu(log_probs, lengths)
        else:
            return self._decode_offline_cpu(log_probs, lengths)

    def _decode_offline_gpu(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        cfg = self._config.gpu_decoder_config
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

    def _decode_offline_cpu(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[RequestOutput]:
        cfg = self._config.cpu_decoder_config
        search_type = {
            "ctc_greedy": "greedy",
            "ctc_prefix_beam": "prefix_beam",
            "ctc_wfst": "wfst",
        }.get(self._config.decoder_type, "prefix_beam")
        from dataclasses import replace
        # type: ignore[arg-type]
        cfg_override = replace(cfg, search_type=search_type) if cfg else None
        decoder = Decoder(cfg_override, fst=self._config.fst_path)

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

        dtype = self._config.decoder_type

        if dtype == "ctc_gpu":
            handle = ctx.get_decoder()
            handle.decode_chunk(log_probs)
            # Return placeholder partial result (best hypothesis not available mid-stream)
            return RequestOutput(
                request_id=request.request_id,
                text="",
                tokens=[],
                finished=False,
            )
        else:
            # CPU streaming decoder stored on the request
            if not hasattr(request, "_cpu_decoder"):
                search_type = {
                    "ctc_greedy": "greedy",
                    "ctc_prefix_beam": "prefix_beam",
                    "ctc_wfst": "wfst",
                }.get(dtype, "prefix_beam")
                cfg = self._config.cpu_decoder_config
                from dataclasses import replace
                # type: ignore[arg-type]
                cfg_override = replace(cfg, search_type=search_type)
                request._cpu_decoder = Decoder(
                    cfg_override, fst=self._config.fst_path)
                request._cpu_decoder.init_stream()

            # log_probs is (1, T_chunk, V) -- remove batch dim
            chunk_logp = log_probs.squeeze(0)
            result: DecoderResult = request._cpu_decoder.decode_chunk(
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
        dtype = self._config.decoder_type

        if dtype == "ctc_gpu":
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
            cpu_dec = getattr(request, "_cpu_decoder", None)
            if cpu_dec is None:
                # No chunks were decoded (empty audio)
                return RequestOutput(
                    request_id=request.request_id,
                    text="",
                    tokens=[],
                    finished=True,
                )
            result_cpu: DecoderResult = cpu_dec.finalize_stream()
            best = result_cpu.tokens[0] if result_cpu.tokens else []
            text = self.detokenize(best)
            return RequestOutput(
                request_id=request.request_id,
                text=text,
                tokens=result_cpu.tokens,
                scores=result_cpu.scores,
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
