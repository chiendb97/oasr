# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""One SDPA oracle for the attention kernels, and the tolerance it is read at.

Four near-copies of this function lived in the suite -- ``test_fmha.py``'s
``_ref_fmha``, two in ``test_layer_waist.py``'s neighbourhood and one inside
``oasr.functionals.attention`` itself -- which is how the sliding window came to
have *no* kernel-level test at all: each copy grew the arguments its own file
needed and nobody owned the union.

The contract this encodes, once:

* **Top-left** causal and top-left windows, matching ``torch``'s ``is_causal``
  and ``include/oasr/attention/fmha_mask.h``.  Not FlashAttention's bottom-right.
* Every mask is *additive* and composed by summing ``-inf`` planes, so a caller
  can hand the same restriction in as a parameter or as a bias and compare the
  two at ``atol=0``.  That equivalence is the sharpest test there is for a mask
  boundary, and it only works if the oracle treats them identically.
* A fully masked row comes back **zero**, which is what the kernel promises and
  what ``F.scaled_dot_product_attention`` does *not* do on its own.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["ref_fmha", "window_bias", "FMHA_TOL", "backend_ratio_ok"]

#: What a fused fp16/bf16 attention is allowed to differ from SDPA by.
#:
#: Deliberately looser than ``tests.helpers.tolerances``'s 1e-2 default: an
#: online-softmax kernel and a materialised one associate the row sum
#: differently, so the gap is a property of the algorithm, not of the
#: implementation.  The suite already used 2e-2 in ~25 hand-written places.
FMHA_TOL = {"rtol": 2e-2, "atol": 2e-2}


def window_bias(
    T_q: int,
    T_k: int,
    *,
    window_left: int = -1,
    window_right: int = -1,
    causal: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Optional[torch.Tensor]:
    """``(1, 1, T_q, T_k)`` additive mask for a top-left window, or ``None``.

    Query row ``r`` may attend to key columns ``[r - window_left,
    r + window_right]``; a negative bound is unbounded on that side.  ``causal``
    adds the plain triangle, which is the same thing as ``window_right == 0``.

    Handing this to the kernel as ``attn_bias`` and passing the same numbers as
    ``window_left=`` / ``window_right=`` must give **bit-identical** output --
    that is the assertion the boundary arithmetic actually lives or dies on.
    """
    if window_left < 0 and window_right < 0 and not causal:
        return None
    rows = torch.arange(T_q, device=device).view(T_q, 1)
    cols = torch.arange(T_k, device=device).view(1, T_k)
    bad = torch.zeros(T_q, T_k, dtype=torch.bool, device=device)
    if causal or window_right >= 0:
        bad |= cols > rows + (0 if causal else window_right)
    if window_left >= 0:
        bad |= cols < rows - window_left
    out = torch.zeros(1, 1, T_q, T_k, dtype=dtype, device=device)
    return out.masked_fill_(bad.view(1, 1, T_q, T_k), float("-inf"))


def ref_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    attn_bias: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    causal: bool = False,
    cache_seqstarts: Optional[torch.Tensor] = None,
    window_left: int = -1,
    window_right: int = -1,
) -> torch.Tensor:
    """SDPA with ``oasr.fmha``'s masking contract.

    ``q`` is ``(B, H, T_q, D)``; ``k``/``v`` are ``(B, H_kv, T_k, D)`` and are
    repeated for GQA/MQA here rather than in the caller.
    """
    B, H, T_q, _ = q.shape
    H_kv = k.size(1)
    T_k = k.size(2)
    if H % H_kv != 0:
        raise ValueError(f"H ({H}) must be divisible by H_kv ({H_kv})")
    if H_kv != H:
        n_repeat = H // H_kv
        k = k.repeat_interleave(n_repeat, dim=1)
        v = v.repeat_interleave(n_repeat, dim=1)

    masks = []
    if attn_bias is not None:
        masks.append(attn_bias.to(q.dtype))
    if cache_seqlens is not None:
        arange = torch.arange(T_k, device=cache_seqlens.device)
        keep = arange.unsqueeze(0) < cache_seqlens.unsqueeze(1)
        if cache_seqstarts is not None:
            keep = keep & (arange.unsqueeze(0) >= cache_seqstarts.unsqueeze(1))
        pad = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
        masks.append(pad.unsqueeze(1).unsqueeze(1))  # (B, 1, 1, T_k)
    geom = window_bias(
        T_q,
        T_k,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        dtype=q.dtype,
        device=q.device,
    )
    if geom is not None:
        masks.append(geom)

    full_mask = None
    if masks:
        full_mask = masks[0]
        for m in masks[1:]:
            full_mask = full_mask + m
        # ``-inf + -inf`` is ``-inf`` but ``-inf + finite`` is what a caller
        # composing a bias with a length mask actually gets, so keep the sum
        # rather than an ``or`` of booleans -- the kernel adds them too.
        full_mask = full_mask.expand(B, -1, T_q, T_k)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask, scale=softmax_scale)
    if full_mask is not None:
        # SDPA leaves an all-`-inf` row as NaN (softmax of a uniform -inf).
        # The kernel defines it as zero, and the difference is load-bearing:
        # a NaN pad row is not inert, because the next layer's masked key still
        # contributes `0 * NaN` and poisons the rows that are real.
        dead = torch.isneginf(full_mask).all(dim=-1, keepdim=True)
        out = torch.where(dead.expand_as(out), torch.zeros_like(out), out)
    return out


def backend_ratio_ok(err_under_test: float, err_reference: float, *, slack: float = 1.5) -> bool:
    """Is a fused backend no worse than another fused backend, against SDPA?

    Two flash kernels with different tile shapes are **not** bit-equal to each
    other and comparing them at a fixed tolerance only measures how similar
    their tiles happen to be.  The claim worth making is the one here: the new
    lane is not further from the shared oracle than the old one was.
    """
    if err_reference <= 0.0:
        return err_under_test <= 0.0
    return err_under_test <= err_reference * slack
