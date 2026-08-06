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

**Unmasked attention goes to SDPA.**  The fused kernel's win is the fusion, so
with nothing to fuse there is nothing to gain.  Measured on an RTX 5090, fp16,
with **head-split views** as inputs — which is what every real call site hands
in, and which matters a great deal: the fused kernel needs canonical row-major
q/k/v, so ``_ensure_canonical`` copies all three (35.3 → 68.7 µs at
``B8 H4 T500 D128``).  An earlier version of this table was measured on freshly
allocated contiguous tensors and overstated both directions.

=====================================  =========  ========  =============
shape                                  fused      SDPA      speedup
=====================================  =========  ========  =============
B16 H6 T1500 D64, no mask              262 µs     265 µs    1.01× (wash)
B4 H20 T1500 D64, ``kv_lens``          252 µs     528 µs    **2.10×**
B16 H4 Tq20 T400 D64, ``kv_lens``      66 µs      59 µs     0.90× (loses)
B8 H4 T500 D128, ``kv_lens``           68 µs      91 µs     1.34×
B8 H4 T100 D128, ``kv_lens``           64 µs      74 µs     1.16×
=====================================  =========  ========  =============

Two things to read off it.  Unmasked is a wash rather than a loss, so routing it
to SDPA costs nothing and keeps the rule simple.  And a *masked* shape with a
short query extent can still lose — the stride copies are paid per call
regardless of how little work the attention itself does — so the fusion rule is
necessary but not sufficient, and closing the canonical-stride requirement is
worth more than any further tuning of it.

``kv_starts`` is the left-padding half of the same window — valid keys are
``[kv_starts[b], kv_lens[b])`` — pushed into the kernel as a second length
vector rather than a materialized mask, so a left-padded batched prompt (HF's
masked-generate convention, which is how the speech-LLM decoder arrives) is
kernel-eligible on the same terms as right padding.

``is_causal`` splits on whether anything else is masked, and the two halves go
opposite ways.  Causal *alone* stays on SDPA: it has a flash path for exactly
that and needs no mask tensor.  Causal *combined* with a window is fused above
:data:`~oasr.layers._backend.FMHA_CAUSAL_WINDOW_MIN_MACS`, because SDPA refuses
``is_causal`` alongside ``attn_mask`` — so the caller has to materialize a
``(B, 1, T_q, T_k)`` tensor and loses flash with it.  Worth 1.8-3.3× on the
attention op at Qwen2-Audio-7B prefill shapes, which dilutes to 1.03-1.05× over
the full 32-layer prefill (the layer is GEMM-dominated).

Left over for SDPA: an arbitrary boolean/broadcast ``attn_mask``.  Not a missing
kernel — any boolean mask is expressible as an additive ``attn_bias`` — but
expanding a broadcast mask to reach the kernel would allocate the very tensor the
fused path exists to avoid.  Padding should arrive as ``kv_lens``/``kv_starts``,
which need no materialization at all.

Shapes the CuteDSL kernel cannot compile at all also land on SDPA.  That is
asked, not assumed: ``oasr.fmha`` *raises* on such a config, because a caller
naming the kernel explicitly should hear about it, so the waist queries
``jit.attention.fmha_config_supported`` first.  Nothing here enumerates head
dims — the arch class answers for itself, and it sizes its cp.async ring to the
arch's shared memory rather than refusing (which is what used to strand
head_dim 128 on the 99 KB parts).

``OASR_LAYERS_BACKEND=torch`` forces SDPA here too, so the waist's debugging
switch covers attention as well as GEMM and norm.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .._backend import (
    FMHA_CAUSAL_WINDOW_MIN_MACS,
    SERVED_DTYPES,
    out_of_scope,
    take_gap,
    take_policy,
    use_fmha_kernel,
)


def split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``(B, T, num_heads * head_dim)`` → ``(B, num_heads, T, head_dim)``."""
    B, T, _ = x.shape
    return x.view(B, T, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``(B, H, T, D)`` → ``(B, T, H * D)`` (contiguous, ready for out-proj)."""
    B, H, T, D = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * D)


def kv_length_mask(
    kv_lens: torch.Tensor,
    t_k: int,
    kv_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``(B,)`` valid key lengths → broadcastable bool mask ``(B, 1, 1, T_k)``.

    With ``kv_starts`` the valid window is ``[start, len)`` (left *and* right
    padding) instead of ``[0, len)``.
    """
    idx = torch.arange(t_k, device=kv_lens.device)
    keep = idx.unsqueeze(0) < kv_lens.to(kv_lens.device).unsqueeze(1)
    if kv_starts is not None:
        keep = keep & (idx.unsqueeze(0) >= kv_starts.to(kv_lens.device).unsqueeze(1))
    return keep.view(-1, 1, 1, t_k)


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
        kv_starts: Optional[torch.Tensor] = None,
        kv_extent: Optional[int] = None,
        block_table: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """``q (B, H, T_q, D)``, ``k``/``v (B, H_kv, T_k, D)`` → ``(B, H, T_q, D)``.

        At most one of ``attn_bias`` / ``attn_mask`` should be given;
        ``kv_lens`` composes with either.

        Parameters
        ----------
        block_table : Tensor, optional
            ``(B, max_blocks_per_seq)`` int32 logical→physical block map.  When
            given, ``k``/``v`` are **pool views** ``(num_blocks, block_size, H_kv,
            D)`` rather than per-batch tensors, and ``kv_lens`` is required.  This
            mode delegates wholesale to :func:`oasr.attention.fmha`, which owns
            both the paged CuteDSL kernel and a paged SDPA reference — so unlike
            every other path here there is no separate fallback to route to, and
            fp32/CPU stays available for the parity oracles.
        kv_lens : Tensor, optional
            ``(B,)`` — keys ``[0, kv_lens[b])`` are valid (right padding).
        kv_extent : int, optional
            Logical length of the cache when ``k``/``v`` are a **capacity
            buffer** wider than the cached region — upstream FlashAttention's
            KV-cache convention, and the reason it exists: a
            ``k_buf[:, :, :t]`` slice has a stride gap, so the fused kernel
            (which needs a compact layout) has to copy the whole cache, once
            per layer per call.  Handing over the buffer plus its length instead
            costs nothing — the K loop is bounded by ``kv_lens`` either way —
            and measured **1.23-1.54×** on prefill geometry and **1.45-1.88×**
            at a decode step, bit-identical.  The SDPA path has the opposite
            preference (it would compute the whole buffer), so it slices here.
        kv_starts : Tensor, optional
            ``(B,)`` — first valid key index, i.e. **left** padding; requires
            ``kv_lens``, which supplies the other end of the window.  Both are
            kernel-eligible: the pair travels as two length vectors, not as a
            materialized mask.  A fully masked query row comes back as zeros
            (the kernel's documented empty-row clamp) where SDPA's math backend
            would give NaN.
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
        if kv_starts is not None and kv_lens is None:
            raise ValueError("kv_starts requires kv_lens — together they bound the key window")
        if kv_extent is not None and kv_lens is None:
            raise ValueError("kv_extent requires kv_lens — the length is what bounds the loop")

        if block_table is not None:
            if kv_lens is None:
                raise ValueError(
                    "block_table requires kv_lens — a paged cache carries no shape "
                    "the key length could be read from"
                )
            if attn_mask is not None:
                raise ValueError(
                    "block_table is incompatible with attn_mask: the paged path has "
                    "no materialized key axis to broadcast a mask against. Pass the "
                    "full (B, H, T_q, T_k) grid as attn_bias instead."
                )
            from oasr.attention import fmha

            paged: torch.Tensor = fmha(
                q,
                k,
                v,
                softmax_scale=self.softmax_scale,
                attn_bias=attn_bias,
                cache_seqlens=kv_lens,
                cache_seqstarts=kv_starts,
                block_table=block_table,
                causal=is_causal,
            )
            return paged

        if self._kernel_eligible(q, k, kv_lens, kv_starts, attn_bias, attn_mask, is_causal):
            from oasr.attention import fmha

            fused: torch.Tensor = fmha(
                q,
                k,
                v,
                softmax_scale=self.softmax_scale,
                attn_bias=attn_bias,
                cache_seqlens=kv_lens,
                cache_seqstarts=kv_starts,
                causal=is_causal,
            )
            return fused

        # SDPA computes every column it is handed, so trim a capacity buffer back
        # to its cached extent here — the mirror of the kernel path above, which
        # wants the buffer whole precisely because slicing it would force a copy.
        if kv_extent is not None and kv_extent < k.size(2):
            k, v = k[:, :, :kv_extent], v[:, :, :kv_extent]

        mask = attn_mask
        if kv_lens is not None:
            length_mask = kv_length_mask(kv_lens, k.size(2), kv_starts)
            mask = length_mask if mask is None else _combine_masks(mask, length_mask)
        if attn_bias is not None:
            mask = attn_bias if mask is None else _combine_masks(mask, attn_bias)

        if is_causal and mask is not None:
            # SDPA raises when given both, so fold the triangle into the mask.
            mask = _fold_causal(mask, q.size(2), k.size(2))
            is_causal = False

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
        k: torch.Tensor,
        kv_lens: Optional[torch.Tensor],
        kv_starts: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> bool:
        if self.backend == "sdpa" or not use_fmha_kernel():
            return False
        if not q.is_cuda:
            return out_of_scope("CPU tensor")
        if q.dtype not in SERVED_DTYPES:
            return out_of_scope(f"dtype {q.dtype}")
        if attn_mask is not None:
            # An arbitrary mask, in whatever form and broadcast shape SDPA
            # takes.  The kernel absorbs a full `(B, H, T_q, T_k)` additive
            # bias, so this is expressible — but expanding a broadcast mask to
            # get there allocates exactly the tensor the fused path exists to
            # avoid.  A caller who already has the full grid should pass it as
            # `attn_bias`; left/right padding should come in as `kv_starts` /
            # `kv_lens`, which need no materialization at all.
            return take_policy("fmha-mask-materialize")
        if is_causal and kv_lens is None and attn_bias is None:
            # Causal and *nothing else*: SDPA has a flash implementation for
            # exactly this and needs no mask tensor, while the fused path pays a
            # ~78 us floor at any T (the canonical-stride copies plus the
            # wrapper).  Measured 0.22x at T=32, 0.83x at T=800 (D=128), 1.19x
            # at T=2048 (D=64), and every causal-only shape in this repo is
            # short — Whisper's SOT prefill is 4 tokens, the WeNet decoder's
            # teacher-forced pass ~40.  Revisit when the stride requirement
            # goes; the crossover moves with it.
            #
            # Causal *combined* with a window is a different question and is
            # handled below: SDPA rejects `is_causal` alongside `attn_mask`, so
            # the combination costs it a materialized (B, 1, T_q, T_k) tensor
            # **and** its flash path.  That is where the fusion pays: measured
            # 1.80-3.29x on the attention op at Qwen2-Audio-7B prefill shapes
            # (causal + left pad, B2-8, P512-1600, D128, bf16, real call-site
            # strides), 1.03-1.05x over the whole prefill.
            return take_policy("fmha-causal-short")
        if is_causal:
            # Causal + a window.  Worth fusing only above the work floor that
            # amortizes the fused path's fixed cost — see
            # FMHA_CAUSAL_WINDOW_MIN_MACS for the sweep.
            macs = q.size(0) * q.size(1) * q.size(2) * k.size(2) * q.size(3)
            if macs < FMHA_CAUSAL_WINDOW_MIN_MACS:
                return take_policy("fmha-causal-window-small")
        if kv_lens is None and attn_bias is None and kv_starts is None:
            # Nothing to fuse, and SDPA measured faster — see the table above.
            # A policy call today; closing it is kernel work.
            return take_policy("fmha-unmasked")

        from oasr.jit.attention import fmha_config_supported

        if not fmha_config_supported(
            head_dim=self.head_dim,
            dtype_str="float16" if q.dtype is torch.float16 else "bfloat16",
            has_bias=attn_bias is not None,
            bias_aligned=attn_bias is not None and attn_bias.size(-1) % 2 == 0,
            causal=is_causal,
        ):
            return take_gap("fmha-head-dim", f"head_dim={self.head_dim}")
        return True

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"num_kv_heads={self.num_kv_heads}, scale={self.softmax_scale}, "
            f"backend={self.backend}"
        )


def _fold_causal(mask: torch.Tensor, t_q: int, t_k: int) -> torch.Tensor:
    """Intersect ``mask`` with a top-left causal triangle, diagonal kept open.

    Needed because SDPA refuses ``is_causal`` alongside an explicit mask, so a
    causal *window* has to arrive as one tensor.  Keeping the diagonal open is a
    NaN guard, not a semantic change: for a query row inside the valid window
    the diagonal is allowed by both the triangle and the window already, so this
    only affects rows outside it — a left pad row, whose every causal key is
    padding.  Those rows would otherwise softmax over all ``-inf`` and come back
    NaN, and a NaN pad row poisons *real* rows in the next layer (a masked key
    contributes weight 0, and ``0 * NaN`` is NaN).  The fused kernel avoids the
    same trap differently, via its empty-row clamp to zero, so the two backends
    agree on every row a caller can legitimately read.
    """
    device = mask.device
    idx_q = torch.arange(t_q, device=device).unsqueeze(1)
    idx_k = torch.arange(t_k, device=device).unsqueeze(0)
    combined = _combine_masks(mask, (idx_k <= idx_q).view(1, 1, t_q, t_k))
    eye = (idx_k == idx_q).view(1, 1, t_q, t_k)
    if combined.dtype == torch.bool:
        return combined | eye
    return combined.masked_fill(eye.expand_as(combined), 0.0)


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
