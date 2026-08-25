# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for softmax operation."""

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api


@functools.cache
def _get_softmax_module():
    from oasr.jit.softmax import gen_softmax_module

    return gen_softmax_module().build_and_load()


@oasr_api
def softmax(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply softmax along the last dimension.

    Args:
        input: Input tensor of any shape [..., num_cols] (CUDA).
        out: Optional pre-allocated output tensor (same shape as input).

    Returns:
        Softmax probabilities with the same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_softmax_module().softmax(out, input)
    return out


@oasr_api
def log_softmax(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply log_softmax along the last dimension.

    In-place is supported (``out is input``); the composed
    ``oasr.gemm_log_softmax`` dispatch path relies on that to normalise the
    GEMM output buffer without an extra allocation.

    Args:
        input: Input tensor of any shape [..., num_cols] (CUDA).
        out: Optional pre-allocated output tensor (same shape as input).

    Returns:
        Log-softmax values with the same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_softmax_module().log_softmax(out, input)
    return out


@oasr_api
def masked_softmax(
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
    mask_value: float = -1000.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused ``softmax((input + bias).to(input.dtype).masked_fill(mask, v)...)``.

    The op sequence an attention with a relative-position bias runs -- add the
    bias, floor the padded keys, softmax -- as one kernel and one pass over the
    score tensor.  ``bias``, ``mask`` and ``mask2`` are **broadcast against**
    ``input`` and read through their own strides, so the caller passes the view
    it already has: a shifted ``as_strided`` window over a ``(H, B, T, 2T-1)``
    relative-position product, a ``[..., ::ds]`` slice of a key-padding mask, an
    ``unsqueeze``.  None of them is copied.

    The biased score is rounded back to ``input``'s dtype before the reduction,
    which makes folding the operands in **numerically free**: this call equals
    ``masked_softmax((input + bias).to(dtype).masked_fill(...))`` bit for bit.
    Keeping fp32 precision through the add would be *more* accurate and would
    move decoded tokens with nothing to attribute the move to.

    Against ``oasr.softmax`` on that same materialized tensor the two agree to
    rounding but not always bit for bit — this kernel walks a 8/4/2/1 vector
    ladder where ``softmax`` tries only the widest width, so a row length that
    is 4- but not 8-divisible groups the online reduction differently.  One
    element in 3000 by one fp16 ulp at ``num_cols = 500``, and Zipformer's
    200-utterance WER gate is unchanged error count for error count.

    ``input`` must be contiguous, and at most 4-D when any operand is given: a
    block finds its leading indices in its own grid position, and there are
    three grid axes.  With no operand any rank works.

    Args:
        input: Scores of any shape ``[..., num_cols]``, contiguous, on CUDA.
            The softmax runs over the last dimension.
        bias: Optional additive term, ``input``'s dtype, broadcastable to its
            shape, arbitrary strides.
        mask: Optional boolean mask, broadcastable to ``input``'s shape; ``True``
            selects ``mask_value``.
        mask2: A second, independently broadcast boolean mask, unioned with
            ``mask``.  Two exist because attention has two — a key-padding mask
            over ``(B, T_kv)`` and an attention mask over ``(T_q, T_kv)`` — with
            different broadcast shapes.
        mask_value: What a masked score becomes.  The default matches icefall's
            Zipformer.  A row that is masked everywhere comes out uniform, as
            ``masked_fill`` + ``softmax`` does; pass ``-inf`` and such a row is
            NaN, also as torch does.
        out: Optional pre-allocated output (same shape and layout as ``input``).

    Returns:
        Softmax probabilities with the same shape as ``input``.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_softmax_module().masked_softmax(out, input, bias, mask, mask2, float(mask_value))
    return out
