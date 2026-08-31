# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer non-autoregressive decode strategy.

Consumes raw encoder hidden states (``consumes="hidden"``) and drives the
model's CIF predictor + NAR decoder: ``model.predict`` integrates encoder
frames into per-token acoustic embeddings (plus token counts and CIF fire
positions), ``model.nar_decode`` produces ``(B, U, V)`` log-probs in **one
parallel pass**, and the transcript is the per-position argmax — no
autoregressive loop, so the ordinary one-shot offline path fits unchanged
(mirroring the FunASR greedy inference exactly).

The CIF fire positions double as a coarse token alignment: token *k* spans
``(fire[k-1], fire[k])`` encoder frames, converted to seconds via the LFR
frame duration (60 ms for the standard fbank 10 ms × LFR 6 frontend) and
emitted as ``RequestOutput.timestamps``.

Offline-only: Paraformer's ``streaming_kind == "none"`` — the engine rejects
streaming requests at admission, and the streaming session methods raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple

import torch

from ..request import Request, RequestOutput
from .alignment import TokenAlignment, wants_word_timings
from .base import DecodeStrategy, register_decode_strategy, wants_speech_activity

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer


@register_decode_strategy("paraformer")
class ParaformerDecodeStrategy(DecodeStrategy):
    """One-shot NAR decoding over encoder hidden states (offline only)."""

    decode_type: ClassVar[str] = "paraformer"
    consumes: ClassVar[str] = "hidden"
    speech_activity_kind: ClassVar[str] = "cif_alpha"

    @property
    def asr_speech_activity_modes(self) -> Tuple[str, ...]:
        """Offline only — so is the family.

        The signal is the CIF predictor's per-frame weight, which is a token
        *rate* rather than a speech posterior: it is high where tokens are being
        integrated and zero in silence.  That makes it usable but the weakest of
        the four ASR-derived signals, and the detector's gain is a heuristic
        rather than a calibration.
        """
        return ("offline",)

    @property
    def word_timing_modes(self) -> Tuple[str, ...]:
        """Offline only — so is the family.  Alone among the seven, this one
        gets its alignment for free: CIF integration has to decide token
        boundaries to produce the acoustic embeddings at all."""
        return ("offline",)

    def __init__(
        self,
        config: "EngineConfig",
        detok: "Detokenizer",
        model: "BaseAsrModel" = None,
    ) -> None:
        super().__init__(config, detok, model)
        # Surface validation lives in ``build_decode_strategy`` via
        # ``oasr.models.interfaces.CAPABILITIES["paraformer"]``.
        mcfg = model.config
        self._filtered_ids = {int(mcfg.blank_id), int(mcfg.sos_id), int(mcfg.eos_id)}

    # ------------------------------------------------------------------
    # Offline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_offline(
        self,
        enc_out: torch.Tensor,
        enc_lengths: torch.Tensor,
        requests: Optional[List[Request]] = None,
    ) -> List[RequestOutput]:
        B = enc_out.size(0)
        want_activity = requests is not None and any(wants_speech_activity(r) for r in requests)
        # ``predict`` returns three values, or four when asked for the CIF
        # weights; indexing rather than unpacking keeps both arities readable to
        # a checker that only sees the variadic return type.
        predicted = self._model.predict(enc_out, enc_lengths, return_alphas=want_activity)
        acoustic_embeds, token_lens, fires = predicted[0], predicted[1], predicted[2]
        alphas = predicted[3] if want_activity else None
        if acoustic_embeds.size(1) == 0 or int(token_lens.max().item()) < 1:
            empty = [self._empty_output() for _ in range(B)]
            # An utterance CIF found no tokens in is exactly the case a caller
            # asking for speech activity most wants an answer to, so the
            # activity pass still runs; it is the transcript that is empty, not
            # the audio's description of itself.
            if alphas is not None:
                self._attach_alpha_activity(empty, alphas, enc_lengths, requests)
            return empty

        log_probs = self._model.nar_decode(enc_out, enc_lengths, acoustic_embeds, token_lens)
        # Greedy per position; per-row score = sum of best log-probs over the
        # row's valid tokens (the FunASR greedy hypothesis score).
        best_lp, best_ids = log_probs.float().max(dim=-1)  # (B, U)

        ids_cpu = best_ids.cpu()
        lp_cpu = best_lp.cpu()
        lens_cpu = token_lens.cpu()
        fires_cpu = fires.cpu()

        outputs: List[RequestOutput] = []
        for b in range(B):
            n = int(lens_cpu[b].item())
            if n <= 0:
                outputs.append(self._empty_output())
                continue
            row_ids = ids_cpu[b, :n].tolist()
            score = float(lp_cpu[b, :n].sum().item())
            spans = self._token_spans(fires_cpu[b], n)
            probs = lp_cpu[b, :n].exp().tolist()
            kept: List[TokenAlignment] = []
            for k, tok in enumerate(row_ids):
                if tok in self._filtered_ids:
                    continue
                start, end = spans[k]
                kept.append(
                    TokenAlignment(
                        token=int(tok),
                        start_frame=start,
                        end_frame=end,
                        confidence=float(min(max(probs[k], 0.0), 1.0)),
                    )
                )
            out = RequestOutput(
                request_id="",
                text=self._detok.detokenize([a.token for a in kept]),
                tokens=[[a.token for a in kept]],
                scores=[score],
                finished=True,
            )
            # CIF integration produces the boundaries whether or not anyone
            # asked, so per-token timings and the confidence are always filled;
            # the token→word pass is what the request opts into.
            self.attach_alignment(
                out, kept, words=requests is not None and wants_word_timings(requests[b])
            )
            outputs.append(out)
        if alphas is not None:
            self._attach_alpha_activity(outputs, alphas, enc_lengths, requests)
        return outputs

    def _attach_alpha_activity(
        self,
        outputs: List[RequestOutput],
        alphas: torch.Tensor,
        enc_lengths: torch.Tensor,
        requests: Optional[List[Request]],
    ) -> None:
        """Segment on the CIF weights.

        ``alphas`` is ``(B, T + 1)`` — the predictor appends a tail frame so the
        final token can fire — while ``enc_lengths`` counts the encoder's own
        frames.  Passing the longer tensor with the shorter lengths is correct
        and deliberate: the tail is padding as far as the time base is
        concerned, and the detector masks it.
        """
        self.attach_asr_speech_activity(outputs, alphas, enc_lengths, requests)

    def _token_spans(self, fires_row: torch.Tensor, n_tokens: int) -> List[Tuple[float, float]]:
        """CIF fire positions → per-token ``(start_frame, end_frame)`` spans.

        Token *k* ends at the *k*-th frame whose integrated weight fired and
        starts where the previous token ended (0 for the first).  If the fire
        count disagrees with the token count (rounding at the tail), missing
        spans reuse the last boundary — timestamps stay monotonic.

        Frames, not seconds: the shared :class:`FrameClock` converts, and for
        this frontend one encoder frame *is* one LFR frame (``subsampling_rate``
        1 × ``lfr_n`` 6 × 10 ms = 60 ms), which is the same number this method
        used to multiply in itself.
        """
        fire_idx = (fires_row >= 1.0).nonzero(as_tuple=False).reshape(-1).tolist()
        spans: List[Tuple[float, float]] = []
        prev = 0.0
        for k in range(n_tokens):
            end_frame = fire_idx[k] if k < len(fire_idx) else (fire_idx[-1] if fire_idx else 0)
            end = float(end_frame)
            if end < prev:
                end = prev
            spans.append((prev, end))
            prev = end
        return spans

    @staticmethod
    def _empty_output() -> RequestOutput:
        return RequestOutput(
            request_id="", text="", tokens=[[]], scores=[0.0], finished=True, timestamps=[]
        )

    # ------------------------------------------------------------------
    # Streaming (unsupported — Paraformer is offline-only)
    # ------------------------------------------------------------------

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        raise RuntimeError("the paraformer decode strategy is offline-only")

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        raise RuntimeError("the paraformer decode strategy is offline-only")

    def finalize(self, request: Request) -> RequestOutput:
        raise RuntimeError("the paraformer decode strategy is offline-only")
