# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused decode kernels for a stateless-predictor transducer.

:func:`transducer_greedy_decode` decodes a whole batch in one launch: each CTA
runs the greedy loop for its rows on the device -- joiner, argmax, emit/advance,
and on an emission the predictor step -- so the host waits once per batch
instead of driving ~25 small kernels per decode step.  See
``include/oasr/transducer/greedy_decode.cuh`` for the kernel and its numerical
contract.

:func:`transducer_beam_decode` is the same for modified beam search: one launch
runs every frame of a chunk, a CTA per utterance holding its ``k`` hypotheses
(``include/oasr/transducer/beam_decode.cuh``).  Both read the model through
one :class:`StatelessGreedyWeights`.

Example::

    w = StatelessGreedyWeights.prepare(...)  # the model's joiner + predictor tensors, once
    enc_proj = joiner.encoder_proj(enc_out)  # (B, T, J)
    res = oasr.transducer_greedy_decode(enc_proj, lengths, window, dec_proj, w, max_sym=10)
    counts = res.counts.tolist()
    hyps = [res.tokens[b, : counts[b]].tolist() for b in range(B)]
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from oasr.api_logging import oasr_api

__all__ = [
    "BEAM_DECODE_MAX_BEAM",
    "BEAM_DECODE_MAX_VOCAB",
    "BEAM_TOPK_MAX_BEAM",
    "BEAM_TOPK_MAX_VOCAB",
    "GEMV_TILE",
    "K_STAGE",
    "JOINER_ACTIVATIONS",
    "StatelessGreedyResult",
    "StatelessGreedyWeights",
    "beam_walk_buffer",
    "transducer_beam_decode",
    "transducer_beam_topk",
    "transducer_beam_topk_supports",
    "transducer_greedy_decode",
    "transducer_greedy_capacity",
]

#: Joiner nonlinearities the kernel implements, by the id it is passed.
JOINER_ACTIVATIONS = {"tanh": 0, "relu": 1}


@functools.cache
def _get_transducer_module():
    from oasr.jit.transducer import gen_transducer_module

    return gen_transducer_module().build_and_load()


#: Outputs one pass of the kernel's GEMV produces; both K-major weights are
#: padded to a multiple of it on their output axis.
GEMV_TILE = 512
#: K granularity of the kernel's GEMV: ``J`` and ``D`` must be multiples of it.
K_STAGE = 8


def _k_major(weight: torch.Tensor, rows: int) -> torch.Tensor:
    """``weight[:rows]`` (an ``(N, K)`` linear weight) as a ``(K, N_pad)`` copy.

    The kernel's GEMV gives each lane 16 outputs and each warp a slice of K, so
    a weight is read along its output axis; padding that axis to a whole tile
    keeps every load in bounds without a predicate.
    """
    k = int(weight.size(1))
    n_pad = -(-rows // GEMV_TILE) * GEMV_TILE
    out = torch.zeros(k, n_pad, dtype=weight.dtype, device=weight.device)
    out[:, :rows].copy_(weight[:rows].t())
    return out


@dataclass(frozen=True)
class StatelessGreedyWeights:
    """The tensors a decode step reads, in the layouts the fused kernels expect.

    Shared by the greedy and the beam-search kernels.

    Build it with :meth:`prepare` from the model's own parameters: the two
    projections are re-laid out **K-major** (``(K, N_pad)``, ``N_pad`` a multiple
    of :data:`GEMV_TILE`), once, and everything else is passed as the model holds
    it.  ``conv_w`` is ``None`` for a context-1 predictor,
    ``(D, 1, context, group)`` for a grouped convolution and ``(context, 1, D)``
    for a depthwise one (``group == 1``).
    """

    w_out_t: torch.Tensor  # (J, V_pad)
    b_out: Optional[torch.Tensor]
    vocab: int
    activation: str
    emb: torch.Tensor
    conv_w: Optional[torch.Tensor]
    context: int
    group: int
    w_dp_t: torch.Tensor  # (D, J_pad)
    b_dp: Optional[torch.Tensor]
    blank: int

    @classmethod
    def prepare(
        cls,
        *,
        output_weight: torch.Tensor,
        output_bias: Optional[torch.Tensor],
        vocab: int,
        activation: str,
        embedding: torch.Tensor,
        conv_weight: Optional[torch.Tensor],
        context: int,
        group: int,
        decoder_proj_weight: torch.Tensor,
        decoder_proj_bias: Optional[torch.Tensor],
        blank: int,
    ) -> "StatelessGreedyWeights":
        """Lay out a joiner head ``(>= vocab, J)`` and decoder projection
        ``(J, D)`` for the kernel.  Copies the two weights; call it once per
        model, not per decode."""
        if activation not in JOINER_ACTIVATIONS:
            raise ValueError(f"unsupported joiner activation {activation!r}")
        with torch.no_grad():
            return cls(
                w_out_t=_k_major(output_weight, int(vocab)),
                b_out=output_bias,
                vocab=int(vocab),
                activation=activation,
                emb=embedding.contiguous(),
                conv_w=None if conv_weight is None else conv_weight.contiguous(),
                context=int(context),
                group=int(group),
                w_dp_t=_k_major(decoder_proj_weight, int(decoder_proj_weight.size(0))),
                b_dp=decoder_proj_bias,
                blank=int(blank),
            )

    def supports(self, enc_proj: torch.Tensor) -> bool:
        """Whether the kernel can decode ``enc_proj`` with these weights.

        Half precision only (the GEMVs multiply packed half-precision pairs),
        every operand in that one dtype, and both the joiner and decoder dims
        multiples of :data:`K_STAGE`.
        """
        dtype = enc_proj.dtype
        return (
            enc_proj.is_cuda
            and dtype in (torch.float16, torch.bfloat16)
            and all(t.dtype == dtype for t in (self.w_out_t, self.emb, self.w_dp_t))
            and int(enc_proj.size(-1)) == int(self.w_out_t.size(0))
            and int(self.w_out_t.size(0)) % K_STAGE == 0
            and int(self.w_dp_t.size(0)) % K_STAGE == 0
            and 1 <= self.context <= 8
        )

    def supports_beam(self, enc_proj: torch.Tensor, beam: int) -> bool:
        """Whether :func:`transducer_beam_decode` can search ``enc_proj`` with ``beam``.

        Everything :meth:`supports` asks, plus the beam kernel's own bounds:
        ``beam`` up to :data:`BEAM_DECODE_MAX_BEAM` (the hypotheses' GEMV
        accumulators are registers), a vocabulary in ``[beam,
        BEAM_DECODE_MAX_VOCAB]`` (the selection's warp log-softmax), and a
        working set that fits the device's shared memory.
        """
        beam = int(beam)
        if not (
            self.supports(enc_proj)
            and 1 <= beam <= BEAM_DECODE_MAX_BEAM
            and beam <= self.vocab <= BEAM_DECODE_MAX_VOCAB
        ):
            return False
        index = enc_proj.device.index
        return _beam_fits(
            torch.cuda.current_device() if index is None else int(index),
            beam,
            int(self.w_out_t.size(0)),
            int(self.w_dp_t.size(0)),
            self.vocab,
        )


@functools.cache
def _beam_fits(device: int, beam: int, J: int, D: int, vocab: int) -> bool:
    """The kernel's own answer: its static and dynamic shared memory against the
    device's opt-in limit (``StatelessBeamFits`` in ``beam_decode.cuh``)."""
    with torch.cuda.device(device):
        return bool(_get_transducer_module().stateless_beam_fits(beam, J, D, vocab))


@dataclass
class StatelessGreedyResult:
    """Device-side outputs of one fused decode.

    ``counts[b]`` may exceed ``tokens.size(1)``: the row emitted more than the
    buffer holds and only the first ``tokens.size(1)`` were recorded.  The caller
    owns that case (re-decode with a larger capacity, or another path).
    """

    tokens: torch.Tensor  # (B, cap) int32
    frames: torch.Tensor  # (B, cap) int32
    probs: Optional[torch.Tensor]  # (B, cap) float32, only when tracked
    counts: torch.Tensor  # (B,) int32
    window: torch.Tensor  # (B, context) int64, final label window
    dec_proj: torch.Tensor  # (B, J), decoder projection of that window


def transducer_greedy_capacity(frames: int) -> int:
    """Default emission capacity for a row of ``frames`` encoder frames.

    Two emissions per frame on average, plus slack, covers every real model by a
    wide margin (BPE transducers emit ~0.2 per frame); the exact worst case,
    ``frames * max_sym``, would size long-form batches at gigabytes.  Overflow is
    reported through ``counts``, never silent.
    """
    return 2 * int(frames) + 32


@oasr_api
def transducer_greedy_decode(
    enc_proj: torch.Tensor,
    lengths: torch.Tensor,
    window: torch.Tensor,
    dec_proj: torch.Tensor,
    weights: StatelessGreedyWeights,
    *,
    max_sym: int,
    capacity: Optional[int] = None,
    track: bool = False,
    rows_per_cta: int = 0,
    out: Optional[StatelessGreedyResult] = None,
) -> StatelessGreedyResult:
    """Greedy-decode ``enc_proj`` from the predictor state ``(window, dec_proj)``.

    Parameters
    ----------
    enc_proj : Tensor
        ``(B, T, J)`` encoder output already projected into joiner space
        (``joiner.encoder_proj(enc_out)``), fp16/bf16/fp32, last dim contiguous.
    lengths : Tensor
        ``(B,)`` valid frames per row (int64; cast if not).
    window, dec_proj : Tensor
        The predictor state to start from: the ``(B, context)`` int64 label
        window and its ``(B, J)`` decoder projection.
    weights : StatelessGreedyWeights
        The model's joiner and predictor tensors.
    max_sym : int
        Cap on emissions at one frame before advancing.
    capacity : int, optional
        Emissions recorded per row; defaults to
        :func:`transducer_greedy_capacity` of ``T``.
    track : bool
        Also return each emission's posterior (frames are always returned).
    rows_per_cta : int
        Rows each CTA decodes; ``0`` lets the launcher pick from the batch width.
    out : StatelessGreedyResult, optional
        Preallocated outputs (destination-passing).  Its ``window`` / ``dec_proj``
        may alias the inputs.
    """
    B, T, J = (int(d) for d in enc_proj.shape)
    device = enc_proj.device
    if lengths.dtype != torch.int64:
        lengths = lengths.to(torch.int64)
    if out is None:
        cap = transducer_greedy_capacity(T) if capacity is None else int(capacity)
        out = StatelessGreedyResult(
            tokens=torch.empty(B, cap, dtype=torch.int32, device=device),
            frames=torch.empty(B, cap, dtype=torch.int32, device=device),
            probs=torch.empty(B, cap, dtype=torch.float32, device=device) if track else None,
            counts=torch.empty(B, dtype=torch.int32, device=device),
            window=torch.empty_like(window),
            dec_proj=torch.empty_like(dec_proj),
        )
    w = weights
    _get_transducer_module().stateless_greedy_decode(
        out.tokens,
        out.frames,
        out.probs if track else None,
        out.counts,
        out.window,
        out.dec_proj,
        enc_proj,
        lengths,
        window,
        dec_proj,
        w.w_out_t,
        w.b_out,
        w.emb,
        w.conv_w,
        w.w_dp_t,
        w.b_dp,
        int(w.vocab),
        int(w.group),
        int(max_sym),
        int(w.blank),
        int(JOINER_ACTIVATIONS[w.activation]),
        int(rows_per_cta),
    )
    return out


#: The fused beam step's scope.  The vocabulary bound is the range of torch's
#: warp log-softmax, which the kernel reproduces bit for bit; the beam bound is
#: the kernel's shared-memory candidate grid (``k x k``).
BEAM_TOPK_MAX_VOCAB = 1024
BEAM_TOPK_MAX_BEAM = 32


def transducer_beam_topk_supports(logits: torch.Tensor, beam: int) -> bool:
    """Whether :func:`transducer_beam_topk` serves a frame of this shape."""
    vocab = int(logits.size(-1))
    return (
        logits.is_cuda
        and logits.dtype in (torch.float16, torch.bfloat16)
        and logits.dim() == 2
        and logits.stride(-1) == 1
        and 1 <= int(beam) <= BEAM_TOPK_MAX_BEAM
        and int(beam) <= vocab <= BEAM_TOPK_MAX_VOCAB
    )


@oasr_api
def transducer_beam_topk(
    logits: torch.Tensor,
    scores: torch.Tensor,
    context: torch.Tensor,
    active: torch.Tensor,
    blank: int,
    *,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One modified-beam-search frame after the joiner, in one launch.

    Fuses ``log_softmax(logits.float())``, the add onto each hypothesis's score,
    the top ``k`` over the beam's ``k * V`` candidates, their split into
    ``(parent, label)``, the mask for rows past their utterance and the reorder of
    the predictor label windows -- the tail of
    :func:`oasr.engine.decode.transducer_beam.beam_search_step`.

    Args:
        logits: ``(B * k, V)`` joiner logits, fp16/bf16; rows may be strided.
        scores: ``(B, k)`` float32 hypothesis scores.
        context: ``(B, k, ctx)`` int64 label windows.
        active: ``(B,)`` bool -- rows whose frame lies inside the utterance.
        blank: the blank label.
        out: optional ``(context, scores, parent, label)`` destinations.

    Returns:
        ``(context, scores, parent, label)`` for the new beam: ``(B, k, ctx)``
        int64, ``(B, k)`` float32 and two ``(B, k)`` int64.  An inactive row keeps
        its windows and scores and records ``parent[j] = j``, ``label = blank``.

    The scores are bit-identical to the torch composition (``V <= 1024``, the
    range of torch's warp log-softmax).  Candidates are ranked by score, then by
    lower ``j * V + v``: ``torch.topk`` keeps the same set but may order a tie
    *inside* the selected ``k`` differently.  See
    ``include/oasr/transducer/beam_topk.cuh``.
    """
    B, k = (int(d) for d in scores.shape)
    if out is None:
        device = scores.device
        out = (
            torch.empty(tuple(context.shape), dtype=torch.long, device=device),
            torch.empty(B, k, dtype=torch.float32, device=device),
            torch.empty(B, k, dtype=torch.long, device=device),
            torch.empty(B, k, dtype=torch.long, device=device),
        )
    context_out, scores_out, parent_out, label_out = out
    _get_transducer_module().transducer_beam_topk(
        context_out, scores_out, parent_out, label_out, logits, scores, context, active, int(blank)
    )
    return context_out, scores_out, parent_out, label_out


#: The fused beam search's scope: the hypotheses' accumulators are registers
#: (``beam``), and the selection reproduces torch's warp log-softmax (``V``).
BEAM_DECODE_MAX_BEAM = 8
BEAM_DECODE_MAX_VOCAB = 1024


def beam_walk_buffer(batch: int, beam: int, frames: int, device: torch.device) -> torch.Tensor:
    """Where :func:`transducer_beam_decode` writes its walk, packed so that one
    device-to-host copy reads all of it: int32 roots ``(B * k)``, counts
    ``(B * k)``, then each hypothesis's tokens ``(B * k, frames)``, the first
    ``count`` of them valid.  Hypothesis ``h = b * k + j``."""
    return torch.empty(batch * beam * (2 + frames), dtype=torch.int32, device=device)


@oasr_api
def transducer_beam_decode(
    enc_proj: torch.Tensor,
    lengths: torch.Tensor,
    context: torch.Tensor,
    scores: torch.Tensor,
    weights: StatelessGreedyWeights,
    *,
    walk: Optional[torch.Tensor] = None,
    cluster: int = 0,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Modified beam search over every frame of ``enc_proj``, in one launch.

    The whole of :func:`oasr.engine.decode.transducer_beam.beam_search_history`
    -- per frame: the predictor and decoder projection for the hypotheses that
    took a label, the joiner, and :func:`transducer_beam_topk`'s selection --
    with a CTA per utterance holding its ``k`` hypotheses.

    Args:
        enc_proj: ``(B, T, J)`` encoder output in joiner space
            (``joiner.encoder_proj(enc_out)``), fp16/bf16, last dim contiguous.
        lengths: ``(B,)`` valid frames per utterance (int64; cast if not).
        context: ``(B, k, ctx)`` int64 predictor label windows of the beam.
        scores: ``(B, k)`` float32 hypothesis scores.
        weights: the model's joiner and predictor tensors.
        walk: optional :func:`beam_walk_buffer`: the kernel also walks every
            final slot back to the chunk's first frame -- the slot it began from
            and the labels it emitted -- so a caller reads those instead of the
            history (what :func:`~oasr.engine.decode.transducer_beam.walk_chunk`
            computes from it on the host).
        cluster: CTAs per utterance -- ``2`` splits every GEMV's weight reads
            between a cluster pair (sm_90+), ``1`` keeps one, ``0`` picks: a
            pair while the batch leaves an SM for each.  The result does not
            depend on it.
        out: optional ``(context, scores, parents, labels)`` destinations; the
            first two may alias the inputs.

    Returns:
        ``(context, scores, parents, labels)``: the beam after the chunk, best
        first, and each frame's back-pointers and labels, frame-major
        ``(T, B, k)`` int64.  A frame past an utterance's length records every
        slot as its own parent, emitting blank, and leaves its beam as it was.

    The selection is ``transducer_beam_topk``'s, bit for bit; the GEMVs keep
    every rounding point of the op-by-op path but accumulate in their own
    order, so a logit can differ from it by one ulp (see
    ``include/oasr/transducer/beam_decode.cuh``).
    """
    B, T = int(enc_proj.size(0)), int(enc_proj.size(1))
    k = int(scores.size(1))
    if lengths.dtype != torch.int64:
        lengths = lengths.to(torch.int64)
    if out is None:
        device = enc_proj.device
        out = (
            torch.empty(tuple(context.shape), dtype=torch.long, device=device),
            torch.empty(B, k, dtype=torch.float32, device=device),
            torch.empty(T, B, k, dtype=torch.long, device=device),
            torch.empty(T, B, k, dtype=torch.long, device=device),
        )
    context_out, scores_out, parents, labels = out
    if B == 0 or T == 0:
        # Nothing to search; the beam is unchanged (and the launcher's layout
        # checks are not posed a zero-size history).
        context_out.copy_(context)
        scores_out.copy_(scores)
        if walk is not None and B:
            hyps = B * k
            walk[:hyps].copy_(torch.arange(hyps, device=walk.device) % k)
            walk[hyps : 2 * hyps].zero_()
        return context_out, scores_out, parents, labels
    w = weights
    _get_transducer_module().stateless_beam_decode(
        context_out,
        scores_out,
        parents,
        labels,
        walk,
        enc_proj,
        lengths,
        context,
        scores,
        w.w_out_t,
        w.b_out,
        w.emb,
        w.conv_w,
        w.w_dp_t,
        w.b_dp,
        int(w.vocab),
        int(w.group),
        int(w.blank),
        int(JOINER_ACTIVATIONS[w.activation]),
        int(cluster),
    )
    return context_out, scores_out, parents, labels
