# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer non-autoregressive decode strategy.

Consumes raw encoder hidden states (``consumes="hidden"``) and drives the
model's CIF predictor + NAR decoder: ``model.predict_weights`` gives the per-frame
token weights and per-row token counts, ``model.integrate`` turns encoder frames
into per-token acoustic embeddings (plus CIF fire positions),
``model.nar_decode`` produces ``(B, U, V)`` log-probs in **one parallel pass**,
and the transcript is the per-position argmax — no autoregressive loop, so the
ordinary one-shot offline path fits unchanged (mirroring the FunASR greedy
inference exactly).

On CUDA with offline graphs on, both halves replay as CUDA graphs
(:mod:`oasr.engine.nar_graph`): the weights keyed by the batch's shape, the rest
by that and the token width, which the counts read back in between decide.

The CIF fire positions double as a coarse token alignment: token *k* spans
``(fire[k-1], fire[k])`` encoder frames, converted to seconds via the LFR
frame duration (60 ms for the standard fbank 10 ms × LFR 6 frontend) and
emitted as ``RequestOutput.timestamps``.

Offline-only: Paraformer's ``streaming_kind == "none"`` — the engine rejects
streaming requests at admission, and the streaming session methods raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Protocol, Tuple, cast

import torch

from ..nar_graph import NarDecodeGraphs
from ..offline_graph import resolve_batch_buckets
from ..request import Request, RequestOutput
from .alignment import wants_word_timings
from .base import DecodeStrategy, register_decode_strategy, wants_speech_activity

if TYPE_CHECKING:
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig
    from .detokenize import Detokenizer


class _Paraformer(Protocol):
    """What this family calls on the model (``CAPABILITIES["paraformer"]``)."""

    def predict_weights(
        self, enc_out: torch.Tensor, enc_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def integrate(
        self, enc_out: torch.Tensor, alphas: torch.Tensor, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def nar_decode(
        self,
        enc_out: torch.Tensor,
        enc_lens: torch.Tensor,
        acoustic_embeds: torch.Tensor,
        token_lens: torch.Tensor,
    ) -> torch.Tensor: ...


def _read_back(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """Host copies of ``tensors``: queued together, then one stream sync.

    A device-to-host ``non_blocking`` copy lands in pinned memory and is safe to
    read once the stream has passed it; CPU tensors are returned as they are.
    """
    if not any(t.is_cuda for t in tensors):
        return list(tensors)
    copies = [t.to("cpu", non_blocking=True) for t in tensors]
    torch.cuda.current_stream(tensors[0].device).synchronize()
    return copies


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
        self._graphs: Optional[NarDecodeGraphs] = self._build_graphs(config)

    def _build_graphs(self, config: "EngineConfig") -> Optional[NarDecodeGraphs]:
        """The capture cache, on the encoder cache's switches and bounds."""
        if not (config.use_cuda_graphs and config.use_offline_cuda_graphs):
            return None
        device = torch.device(config.device)
        if device.type != "cuda":
            return None
        return NarDecodeGraphs(
            device=device,
            weights_fn=self._nar.predict_weights,
            decode_fn=self._decode_tokens,
            batch_buckets=resolve_batch_buckets(config),
            max_frames=config.offline_graph_max_frames,
            max_captures=config.offline_graph_max_captures,
            evict_lru=config.offline_graph_evict,
        )

    @property
    def _nar(self) -> _Paraformer:
        # ``nn.Module.__getattr__`` types every member as ``Tensor | Module``.
        return cast(_Paraformer, self._model)

    def _decode_tokens(
        self,
        enc_out: torch.Tensor,
        enc_lens: torch.Tensor,
        alphas: torch.Tensor,
        token_lens: torch.Tensor,
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CIF integration, the NAR pass and the greedy pick → ``(best_lp (B, U)
        fp32, best_ids (B, U) int64, fires (B, T+1) fp32)``.

        What the decode graph captures, and what an unserved shape runs eagerly.
        The pick runs on the decoder's own dtype and widens only the winners: a
        widening cast is exact and order-preserving, so this is the same pick as
        widening the ``(B, U, V)`` log-probs first, without that copy.
        """
        acoustic_embeds, fires = self._nar.integrate(enc_out, alphas, num_tokens)
        log_probs = self._nar.nar_decode(enc_out, enc_lens, acoustic_embeds, token_lens)
        best_lp, best_ids = log_probs.max(dim=-1)
        return best_lp.float(), best_ids, fires

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
        graphs = self._graphs
        stage = graphs.weights(enc_out, enc_lengths) if graphs is not None else None
        if stage is not None:
            alphas, token_lens = stage.alphas[:B], stage.token_lens[:B]
        else:
            alphas, token_lens = self._nar.predict_weights(enc_out, enc_lengths)
        # The read-back between the halves: each row's token count, which sizes
        # the decoder.  The predictor used to read only the maximum here and the
        # counts again after the decoder; one vector now serves both.
        (lens_cpu,) = _read_back(token_lens)
        u_max = int(lens_cpu.max()) if B else 0
        if want_activity and stage is not None:
            # A replay buffer: the decode replay below may reuse its memory.
            alphas = alphas.clone()
        if u_max <= 0:
            empty = [self._empty_output() for _ in range(B)]
            # An utterance CIF found no tokens in is exactly the case a caller
            # asking for speech activity most wants an answer to, so the
            # activity pass still runs; it is the transcript that is empty, not
            # the audio's description of itself.
            if want_activity:
                self._attach_alpha_activity(empty, alphas, enc_lengths, requests)
            return empty

        # Rounded on both paths when the cache exists, so a shape that falls
        # back to eager decodes at the width its replay would have.
        num_tokens = graphs.token_bucket(u_max) if graphs is not None else u_max
        decoded = (
            graphs.decode(stage, num_tokens) if graphs is not None and stage is not None else None
        )
        if decoded is None:
            decoded = self._decode_tokens(enc_out, enc_lengths, alphas, token_lens, num_tokens)
        # Three copies queued behind the decoder and one wait, not three waits.
        # Synchronous on purpose, twice measured.  Eager, deferring this read-back
        # and the host tail behind the next micro-batch's encoder
        # (``decode_offline_async``) was 4-7 % slower on a 256-request AISHELL
        # backlog (1,834-1,839 vs 1,974-1,980 req/s).  With both halves replayed
        # it gained nothing (2,305-2,467 vs 2,354-2,454 req/s on 512, four
        # interleaved rounds): the count read-back above waits for each batch's
        # encoder, so the previous batch's host tail could only hide behind this
        # batch's decoder replay, a few milliseconds of GPU work.
        lp_cpu, ids_cpu, fires_cpu = _read_back(*decoded)
        outputs = self._outputs(B, lp_cpu, ids_cpu, lens_cpu, fires_cpu, requests)
        if want_activity:
            self._attach_alpha_activity(outputs, alphas, enc_lengths, requests)
        return outputs

    def _outputs(
        self,
        B: int,
        lp_cpu: torch.Tensor,
        ids_cpu: torch.Tensor,
        lens_cpu: torch.Tensor,
        fires_cpu: torch.Tensor,
        requests: Optional[List[Request]],
    ) -> List[RequestOutput]:
        """Per-row outputs from the host copies of one decode.

        One list conversion per tensor for the whole micro-batch, then plain
        list work per row. A per-row ``.item()`` / ``.tolist()`` and a per-token
        loop that built a ``TokenAlignment`` each were half of a post-processing
        tail that was a quarter of the offline wall at batch 64 (audit L7). The
        only per-row tensor ops left are the score's sum and the confidences'
        ``exp``. They run on the same ``[:n]`` slice as before, so both are
        bit-identical: a whole-batch ``exp`` may split vector and scalar lanes
        differently.
        """
        ids_rows = ids_cpu.tolist()
        lens = lens_cpu.tolist()
        fire_rows = self._fire_frames(fires_cpu, B)
        filtered = self._filtered_ids

        outputs: List[RequestOutput] = []
        for b in range(B):
            n = lens[b]
            if n <= 0:
                outputs.append(self._empty_output())
                continue
            row = ids_rows[b][:n]
            lp_row = lp_cpu[b, :n]
            score = float(lp_row.sum())
            # ``exp`` of a log-probability, clamped: a max log-softmax can round
            # just above 0.
            probs = lp_row.exp().clamp_(0.0, 1.0).tolist()
            starts, ends = self._token_spans(fire_rows[b], n)
            if not filtered.isdisjoint(row):
                keep = [k for k, tok in enumerate(row) if tok not in filtered]
                row = [row[k] for k in keep]
                starts = [starts[k] for k in keep]
                ends = [ends[k] for k in keep]
                probs = [probs[k] for k in keep]
            out = RequestOutput(
                request_id="",
                text=self._detok.detokenize(row),
                tokens=[row],
                scores=[score],
                finished=True,
            )
            # CIF integration produces the boundaries whether or not anyone
            # asked, so per-token timings and the confidence are always filled;
            # the token→word pass is what the request opts into.
            self.attach_span_alignment(
                out,
                row,
                starts,
                ends,
                probs,
                words=requests is not None and wants_word_timings(requests[b]),
            )
            outputs.append(out)
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

    @staticmethod
    def _fire_frames(fires: torch.Tensor, batch: int) -> List[List[int]]:
        """Each row's CIF fire frames, ascending, from one ``nonzero`` over the batch.

        ``nonzero`` returns its ``(row, frame)`` pairs in row-major order, so
        each row's frames arrive ascending, which :meth:`_token_spans` relies on.
        """
        rows: List[List[int]] = [[] for _ in range(batch)]
        for b, t in (fires >= 1.0).nonzero(as_tuple=False).tolist():
            rows[b].append(t)
        return rows

    @staticmethod
    def _token_spans(fire_frames: List[int], n_tokens: int) -> Tuple[List[int], List[int]]:
        """CIF fire frames → per-token ``(start_frames, end_frames)``.

        Token *k* ends at the *k*-th frame whose integrated weight fired and
        starts where the previous token ended (0 for the first).  If the fire
        count disagrees with the token count (rounding at the tail), missing
        spans reuse the last boundary — timestamps stay monotonic.  The fire
        frames ascend, so the ends never decrease and no clamp is needed.

        Frames, not seconds: the shared :class:`FrameClock` converts, and for
        this frontend one encoder frame *is* one LFR frame (``subsampling_rate``
        1 × ``lfr_n`` 6 × 10 ms = 60 ms), which is the same number this method
        used to multiply in itself.
        """
        if fire_frames:
            ends = fire_frames[:n_tokens]
            if len(ends) < n_tokens:
                ends = ends + [fire_frames[-1]] * (n_tokens - len(ends))
        else:
            ends = [0] * n_tokens
        return [0] + ends[:-1], ends

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
