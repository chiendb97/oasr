# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CTC + attention-decoder rescoring (WeNet U2/U2++ ``attention_rescoring``).

Two-stage offline decode for hybrid checkpoints that carry both a CTC head and
an AED (bi)transformer decoder:

1. **CTC n-best** — the GPU prefix beam search
   (:func:`oasr.ctc_decode.ctc_beam_search_decode`) already returns the full
   beam per utterance (``tokens[b][k]`` + ``scores[b, k]``).
2. **One teacher-forced decoder pass** — every hypothesis of every utterance is
   scored in a single batched forward (left-to-right, plus the right-to-left
   branch when the checkpoint has one), *not* an autoregressive loop.
3. **Score fusion** — WeNet semantics, vectorized::

       score = (1 - rw) * Σ log p_l2r(y_j) + rw * Σ log p_r2l(y_j)
               + ctc_weight * ctc_score          (each Σ includes the eos term)

   and the fused argmax per utterance wins.

``consumes = "both"``: the offline executor feeds an
:class:`~oasr.engine.decode.base.EncodeOutput` (hidden states for the decoder's
cross-attention + CTC log-probs for the beam search) from one encoder forward.
Streaming is rejected at engine construction (final-only streaming rescoring is
a planned follow-up); the streaming entry points below raise defensively.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import torch

from oasr.ctc_decode import ctc_beam_search_decode
from oasr.models.decoders.transformer_decoder import add_sos_eos, reverse_pad_list

from ..request import Request, RequestOutput
from .base import DecodeStrategy, EncodeOutput, register_decode_strategy

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer

logger = logging.getLogger(__name__)

#: Padding marker for hypothesis tensors (WeNet ``IGNORE_ID``).
_IGNORE_ID = -1


@register_decode_strategy("ctc_aed_rescoring")
class CtcAedRescoringStrategy(DecodeStrategy):
    """Offline CTC n-best + attention-decoder rescoring for hybrid models."""

    decode_type = "ctc_aed_rescoring"
    consumes = "both"

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        super().__init__(config, detok, model)
        # Surface validation lives in ``build_decode_strategy`` via
        # ``oasr.models.interfaces.CAPABILITIES["ctc_aed_rescoring"]``.
        decoder = model.decoder
        decoder_cfg = model.config.decoder
        self._decoder = decoder
        self._sos = int(decoder_cfg.sos_id)
        self._eos = int(decoder_cfg.eos_id)
        self._vocab = int(decoder_cfg.vocab_size)
        self._ctc_weight = float(config.rescoring_ctc_weight)
        # None → the checkpoint's trained reverse weight; explicitly 0.0 (or a
        # decoder without a right-to-left branch) skips the reverse pass.
        rw = config.rescoring_reverse_weight
        self._reverse_weight = float(decoder_cfg.reverse_weight if rw is None else rw)
        if self._reverse_weight > 0.0 and not getattr(decoder, "has_reverse", False):
            logger.warning(
                "rescoring_reverse_weight=%.2f but the decoder has no "
                "right-to-left branch; running left-to-right only",
                self._reverse_weight,
            )
            self._reverse_weight = 0.0

    # ------------------------------------------------------------------
    # Offline
    # ------------------------------------------------------------------

    def decode_offline(
        self, enc_out: EncodeOutput, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        assert isinstance(enc_out, EncodeOutput), (
            "ctc_aed_rescoring consumes 'both'; the executor must pass an "
            f"EncodeOutput, got {type(enc_out).__name__}"
        )
        hidden, log_probs = enc_out.hidden, enc_out.log_probs
        cfg = self._config.ctc_decoder_config
        assert cfg is not None

        # ---- stage 1: CTC n-best (GPU prefix beam search) -----------------
        result = ctc_beam_search_decode(
            log_probs,
            enc_lengths,
            beam_size=cfg.beam_size,
            blank_id=cfg.blank_id,
            blank_threshold=cfg.blank_threshold,
            max_seq_len=cfg.max_seq_len,
            use_paged_memory=cfg.use_paged_memory,
            page_size=cfg.page_size,
        )
        B = hidden.size(0)
        beam = cfg.beam_size
        device = hidden.device
        ctc_scores = result.scores.to(device=device, dtype=torch.float32)  # (B, beam)

        # ---- stage 2: one batched teacher-forced decoder pass --------------
        hyps: List[List[int]] = [
            (result.tokens[b][k] if k < len(result.tokens[b]) else [])
            for b in range(B)
            for k in range(beam)
        ]
        n = len(hyps)
        max_len = max((len(h) for h in hyps), default=0)
        hyps_pad = torch.full((n, max(max_len, 1)), _IGNORE_ID, dtype=torch.long, device=device)
        for i, h in enumerate(hyps):
            if h:
                hyps_pad[i, : len(h)] = torch.tensor(h, dtype=torch.long, device=device)
        hyps_lens = (hyps_pad != _IGNORE_ID).sum(dim=1)  # (n,)

        # Ids outside the decoder vocab cannot be scored (the CTC head is
        # 8-padded past the raw vocab; its pad logits never win a beam in
        # practice, but stay defensive): clamp for the embedding and force the
        # fused score of any such hypothesis to -inf.
        valid_hyp = (hyps_pad < self._vocab).all(dim=1)
        hyps_pad = torch.where(
            (hyps_pad >= self._vocab) & (hyps_pad != _IGNORE_ID),
            torch.zeros_like(hyps_pad),
            hyps_pad,
        )

        ys_in, ys_out = add_sos_eos(hyps_pad, self._sos, self._eos, _IGNORE_ID)
        ys_in_lens = hyps_lens + 1

        memory = hidden.repeat_interleave(beam, dim=0)
        memory_lens = enc_lengths.to(device).repeat_interleave(beam, dim=0)

        r_ys_in = None
        use_reverse = self._reverse_weight > 0.0
        if use_reverse:
            r_hyps_pad = reverse_pad_list(hyps_pad, hyps_lens, _IGNORE_ID)
            r_ys_in, r_ys_out = add_sos_eos(r_hyps_pad, self._sos, self._eos, _IGNORE_ID)

        with torch.no_grad():
            l_logits, r_logits = self._decoder(memory, memory_lens, ys_in, ys_in_lens, r_ys_in)

        # ---- stage 3: score fusion (vectorized WeNet math) ------------------
        l_scores = self._gather_scores(l_logits, ys_out)  # (n,)
        if use_reverse and r_logits is not None:
            r_scores = self._gather_scores(r_logits, r_ys_out)
            att_scores = (1.0 - self._reverse_weight) * l_scores + self._reverse_weight * r_scores
        else:
            att_scores = l_scores

        fused = att_scores.view(B, beam) + self._ctc_weight * ctc_scores
        fused = fused.masked_fill(~valid_hyp.view(B, beam), float("-inf"))

        fused_cpu = fused.cpu().tolist()
        outputs: List[RequestOutput] = []
        for b in range(B):
            # Re-rank the CTC beam by fused score (stable, so exact ties keep
            # the CTC order); the head of the ranking is the rescored result.
            order = sorted(range(beam), key=lambda k: fused_cpu[b][k], reverse=True)
            token_seqs = [result.tokens[b][k] for k in order]
            outputs.append(
                RequestOutput(
                    request_id="",
                    text=self._detok.detokenize(token_seqs[0]),
                    tokens=token_seqs,
                    scores=[fused_cpu[b][k] for k in order],
                    finished=True,
                )
            )
        return outputs

    @staticmethod
    def _gather_scores(logits: torch.Tensor, ys_out: torch.Tensor) -> torch.Tensor:
        """Sum per-token log-probs along each hypothesis (incl. the eos term).

        ``logits``: ``(n, L, V)`` decoder output; ``ys_out``: ``(n, L)`` target
        layout (``hyp + [eos]``, ``_IGNORE_ID``-padded) — position ``j`` of
        ``ys_out`` is scored by decoder step ``j``, which matches WeNet's
        ``decoder_out[i][j][hyp[j]] … + decoder_out[i][len][eos]``.
        """
        lp = torch.log_softmax(logits.float(), dim=-1)
        mask = ys_out != _IGNORE_ID
        idx = ys_out.masked_fill(~mask, 0).unsqueeze(-1)
        tok_lp = lp.gather(2, idx).squeeze(-1)  # (n, L)
        return (tok_lp * mask).sum(dim=1)

    # ------------------------------------------------------------------
    # Streaming (offline-only strategy; engine rejects at construction)
    # ------------------------------------------------------------------

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        raise NotImplementedError(
            "ctc_aed_rescoring is offline-only (final-only streaming rescoring "
            "is a planned follow-up)"
        )

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        raise NotImplementedError(
            "ctc_aed_rescoring is offline-only (final-only streaming rescoring "
            "is a planned follow-up)"
        )

    def finalize(self, request: Request) -> RequestOutput:
        raise NotImplementedError(
            "ctc_aed_rescoring is offline-only (final-only streaming rescoring "
            "is a planned follow-up)"
        )
