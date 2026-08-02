# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Generic (non-rel-pos) multi-head attention — the waist's attention core.

Six architectures in this repo grew their own copy of the same twelve lines:
reshape to heads, run scaled dot-product attention, merge the heads back.  The
projections around them differ (names, bias flags, fusion) and belong to the
model, because a checkpoint's key layout is the model's business.  The
*compute* does not differ, and that is what lives here — so improving the
attention backend, or fixing it, happens once.

This is deliberately **not** the whole attention block.  Following vLLM's
``Attention``, the module takes q/k/v that are already projected and already
head-split, so a model keeps ``q_proj``/``linear_q``/``linear_q_k_v`` under
whatever name its ecosystem uses and still shares the kernel path.

Backend
-------
``oasr.fmha`` (CuteDSL on supported GPUs, SDPA inside otherwise) when there is
a mask **and** it is one the fused kernel can absorb:

* ``kv_lens``: keys ``[0, kv_lens[b])`` are valid, enforced from a length
  vector rather than a materialized ``(B, H, T_q, T_k)`` bias (Whisper-geometry
  audio towers, SANM, cross-attention over padded memory);
* ``attn_bias``: an additive bias already shaped ``(B, H, T_q, T_k)`` (the
  Conformer rel-pos ``matrix_bd``).

**Unmasked attention goes to SDPA**, and that is a measurement rather than an
oversight: the fused kernel's win is the fusion, so with nothing to fuse
PyTorch's own flash kernel is simply better.  On an RTX 5090, fp16 —

=================================  ==========  =========
shape                              fused       vs SDPA
=================================  ==========  =========
B16 H6 T1500 D64, no mask          329 µs      1.25× slower
B16 H6 Tq1 T1500 D64, no mask      47 µs       1.87× slower
B4 H20 T1500 D64, ``kv_lens``      281 µs      1.9× faster
B16 H4 Tq20 T400 D64, ``kv_lens``  35 µs       1.7× faster
=================================  ==========  =========

Left over for SDPA for structural reasons: a causal mask, an arbitrary boolean
mask, and **left** padding (valid keys are ``[P - len, P)``, which is not a
length).  Materializing any of those into a full bias tensor to reach the
fused kernel would cost more than it saves.

Shapes the CuteDSL kernel cannot compile (head_dim 128 on sm_120, say) also
land on SDPA.  That is asked, not assumed: ``oasr.fmha`` *raises* on such a
config, because a caller naming the kernel explicitly should hear about it, so
the waist queries ``jit.attention.fmha_config_supported`` first.  Nothing here
enumerates head dims — the arch class answers for itself and the answer tracks
the kernel.

``OASR_LAYERS_BACKEND=torch`` forces SDPA here too, so the waist's debugging
switch covers attention as well as GEMM and norm.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .._backend import GEMM_DTYPES, use_fmha_kernel


def split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``(B, T, num_heads * head_dim)`` → ``(B, num_heads, T, head_dim)``."""
    B, T, _ = x.shape
    return x.view(B, T, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``(B, H, T, D)`` → ``(B, T, H * D)`` (contiguous, ready for out-proj)."""
    B, H, T, D = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * D)


def kv_length_mask(kv_lens: torch.Tensor, t_k: int) -> torch.Tensor:
    """``(B,)`` valid key lengths → broadcastable bool mask ``(B, 1, 1, T_k)``."""
    idx = torch.arange(t_k, device=kv_lens.device)
    return (idx.unsqueeze(0) < kv_lens.to(kv_lens.device).unsqueeze(1)).view(-1, 1, 1, t_k)


class Attention(nn.Module):
    """Scaled dot-product attention over pre-projected, head-split q/k/v.

    Parameters
    ----------
    num_heads : int
        Query heads.
    head_dim : int
        Per-head dimension.
    num_kv_heads : int, optional
        K/V heads for MQA/GQA; defaults to ``num_heads``.  Grouping is expanded
        inside the backend (``enable_gqa`` for SDPA, head fan-out in the
        kernel), never by materializing ``repeat_interleave`` on the cache.
    softmax_scale : float, optional
        Defaults to ``head_dim ** -0.5``.  Pass ``1.0`` when the caller already
        scaled ``q`` (FunASR's SANM convention).
    backend : {"auto", "sdpa"}
        ``"sdpa"`` pins this module to ``F.scaled_dot_product_attention``
        regardless of shape — an escape hatch for a call site that must not
        move, kept per-module so pinning one does not pin the process.

    Notes
    -----
    Carries no parameters, so inserting it into an existing module changes no
    state-dict key.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        num_kv_heads: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if backend not in ("auto", "sdpa"):
            raise ValueError(f"backend must be 'auto' or 'sdpa', got {backend!r}")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by num_kv_heads={self.num_kv_heads}"
            )
        self.softmax_scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
        self.backend = backend

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Query-side head split (``num_heads``)."""
        return split_heads(x, self.num_heads, self.head_dim)

    def split_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Key/value-side head split (``num_kv_heads``; same as
        :meth:`split_heads` unless the module is MQA/GQA)."""
        return split_heads(x, self.num_kv_heads, self.head_dim)

    @staticmethod
    def merge_heads(x: torch.Tensor) -> torch.Tensor:
        """``(B, H, T, D)`` → ``(B, T, H * D)``."""
        return merge_heads(x)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_lens: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """``q (B, H, T_q, D)``, ``k``/``v (B, H_kv, T_k, D)`` → ``(B, H, T_q, D)``.

        At most one of ``attn_bias`` / ``attn_mask`` should be given;
        ``kv_lens`` composes with either.

        Parameters
        ----------
        kv_lens : Tensor, optional
            ``(B,)`` — keys ``[0, kv_lens[b])`` are valid.  Right padding only;
            left padding is not a length and must come in as ``attn_mask``.
        attn_bias : Tensor, optional
            Additive float bias shaped exactly ``(B, num_heads, T_q, T_k)``.
            Kernel-eligible.  A bias that only broadcasts (``(B, 1, 1, T_k)``)
            belongs in ``attn_mask``: expanding it to reach the kernel would
            allocate the very tensor the kernel exists to avoid.
        attn_mask : Tensor, optional
            Anything SDPA accepts (bool where True attends, or additive float),
            broadcastable to ``(B, num_heads, T_q, T_k)``.  Forces the SDPA path.
        is_causal : bool
            Causal masking over the ``(T_q, T_k)`` grid.  Forces the SDPA path,
            which has a dedicated flash implementation for it.
        """
        if self._kernel_eligible(q, kv_lens, attn_bias, attn_mask, is_causal):
            from oasr.attention import fmha

            fused: torch.Tensor = fmha(
                q,
                k,
                v,
                softmax_scale=self.softmax_scale,
                attn_bias=attn_bias,
                cache_seqlens=kv_lens,
            )
            return fused

        mask = attn_mask
        if kv_lens is not None:
            length_mask = kv_length_mask(kv_lens, k.size(2))
            mask = length_mask if mask is None else _combine_masks(mask, length_mask)
        if attn_bias is not None:
            mask = attn_bias if mask is None else _combine_masks(mask, attn_bias)

        out: torch.Tensor = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=is_causal,
            scale=self.softmax_scale,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        return out

    def _kernel_eligible(
        self,
        q: torch.Tensor,
        kv_lens: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> bool:
        if self.backend == "sdpa" or not use_fmha_kernel():
            return False
        if attn_mask is not None or is_causal:
            return False
        if kv_lens is None and attn_bias is None:
            # Nothing to fuse — see the table in the module docstring.
            return False
        if not q.is_cuda or q.dtype not in GEMM_DTYPES:
            return False
        from oasr.jit.attention import fmha_config_supported

        return fmha_config_supported(
            head_dim=self.head_dim,
            dtype_str="float16" if q.dtype is torch.float16 else "bfloat16",
            has_bias=attn_bias is not None,
            bias_aligned=attn_bias is not None and attn_bias.size(-1) % 2 == 0,
        )

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"num_kv_heads={self.num_kv_heads}, scale={self.softmax_scale}, "
            f"backend={self.backend}"
        )


def _combine_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Intersect two SDPA masks, whichever form each is in."""
    if a.dtype == torch.bool and b.dtype == torch.bool:
        return a & b
    a_add = _as_additive(a, b)
    b_add = _as_additive(b, a)
    return a_add + b_add


def _as_additive(m: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if m.dtype != torch.bool:
        return m
    dtype = other.dtype if other.dtype != torch.bool else torch.float32
    return torch.zeros_like(m, dtype=dtype).masked_fill_(~m, float("-inf"))


__all__ = ["Attention", "kv_length_mask", "merge_heads", "split_heads"]
