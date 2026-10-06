# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused frame-synchronous greedy decode for a stateless-predictor transducer.

One launch decodes a whole batch: each CTA runs the greedy loop for its rows on
the device -- joiner, argmax, emit/advance, and on an emission the predictor
step -- so the host waits once per batch instead of driving ~25 small kernels
per decode step.  See ``include/oasr/transducer/greedy_decode.cuh`` for the
kernel and its numerical contract.

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
from typing import Optional

import torch

from oasr.api_logging import oasr_api

__all__ = [
    "GEMV_TILE",
    "K_STAGE",
    "JOINER_ACTIVATIONS",
    "StatelessGreedyResult",
    "StatelessGreedyWeights",
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
    """The tensors one greedy step reads, in the layouts the kernel expects.

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
