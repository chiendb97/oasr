# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Public functional API for the OASR fused multi-head attention.

This is the single entry point that dispatches between the SDPA fallback
(always available) and the CuteDSL kernel (SM120 in this revision)
based on ``OASR_ATTN_BACKEND`` and the active GPU's compute capability.

Three cache modes are encoded in one signature:

* **offline**         -- ``block_table is None`` and ``cache_seqlens is None``
                         (attend over the full ``T_k``).
* **dense streaming** -- ``block_table is None`` and ``cache_seqlens is not None``
                         (attend over ``[0, cache_seqlens[b])`` per stream;
                         caller has already concatenated old + new K/V).
* **paged streaming** -- ``block_table is not None``  (k/v are pool views,
                         ``cache_seqlens`` required). **Not implemented in
                         this revision** -- transparently falls back to SDPA.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .api_logging import oasr_api
from .jit.attention import get_compiled_fmha, select_backend


__all__ = ["fmha"]


# ---------------------------------------------------------------------------
# SDPA reference path (fallback)
# ---------------------------------------------------------------------------

def _length_to_pad_bias(
    cache_seqlens: torch.Tensor, T_kv: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Build (B, 1, 1, T_kv) additive mask: 0 in [0, len), -inf outside."""
    arange = torch.arange(T_kv, device=cache_seqlens.device)
    keep = arange.unsqueeze(0) < cache_seqlens.unsqueeze(1)
    return torch.where(keep, 0.0, float("-inf")).to(dtype).unsqueeze(1).unsqueeze(1)


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    attn_bias: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Functional SDPA path producing the same result as the cute kernel.

    Used directly when the backend selects ``sdpa``, and as the parity
    reference for kernel tests.
    """
    B, H, T_q, D = q.shape
    H_kv = k.size(1)
    T_kv = k.size(2)
    if H_kv != H:
        # GQA / MQA: expand K/V heads.
        if H % H_kv != 0:
            raise ValueError(f"H ({H}) must be divisible by H_kv ({H_kv})")
        n_repeat = H // H_kv
        k = k.repeat_interleave(n_repeat, dim=1)
        v = v.repeat_interleave(n_repeat, dim=1)

    if attn_bias is not None and cache_seqlens is not None:
        pad = _length_to_pad_bias(cache_seqlens, T_kv, q.dtype)
        full_mask = attn_bias + pad
    elif attn_bias is not None:
        full_mask = attn_bias
    elif cache_seqlens is not None:
        full_mask = _length_to_pad_bias(cache_seqlens, T_kv, q.dtype)
    else:
        full_mask = None

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=full_mask, scale=softmax_scale,
    )


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------

@oasr_api
def fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float,
    attn_bias: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused multi-head attention.

    Parameters
    ----------
    q : Tensor
        ``(B, H, T_q, D)`` query, fp16 or bf16.
    k, v : Tensor
        Contiguous mode: ``(B, H_kv, T_k, D)``.
        Paged mode    : ``(num_blocks, block_size, H_kv, D)`` (block-pool view).
    softmax_scale : float
        Softmax scale (typically ``1/sqrt(D)``).
    attn_bias : Tensor, optional
        ``(B, H, T_q, T_k_max)`` additive bias (e.g., Transformer-XL
        ``matrix_bd``). Caller is responsible for any pre-scaling — the kernel
        applies the bias *after* multiplying ``Q@K^T`` by ``softmax_scale``.
    cache_seqlens : Tensor, optional
        ``(B,)`` int32. Per-stream valid k-length. When provided, columns
        ``>= cache_seqlens[b]`` are masked out (``-inf``).
    block_table : Tensor, optional
        ``(B, max_blocks_per_seq)`` int32. When provided, ``k``/``v`` are
        interpreted as paged-pool views.
    out : Tensor, optional
        Pre-allocated ``(B, H, T_q, D)`` output. Allocated if ``None``.

    Returns
    -------
    out : Tensor
        ``(B, H, T_q, D)``.
    """
    # ---- Shape / dtype checks ------------------------------------------------
    if q.dim() != 4:
        raise ValueError(f"q must be (B, H, T_q, D), got shape {tuple(q.shape)}")
    B, H, T_q, D = q.shape

    if block_table is not None:
        # Paged mode: not yet routed to the cute kernel (Stage 2). Caller
        # should hold a contiguous gathered view and use the dense path; the
        # nn.Module wrapper handles this transparently.
        raise NotImplementedError(
            "fmha: paged-KV path is not implemented in this revision; "
            "the wrapper falls back to SDPA for paged streaming."
        )

    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"k/v must be (B, H_kv, T_k, D); got {tuple(k.shape)} / {tuple(v.shape)}"
        )
    if k.shape != v.shape:
        raise ValueError(f"k and v must have identical shape: {k.shape} vs {v.shape}")
    if k.size(0) != B or k.size(3) != D:
        raise ValueError(
            f"k batch/D mismatch with q: q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    H_kv = k.size(1)
    T_k = k.size(2)
    if H % H_kv != 0:
        raise ValueError(f"H ({H}) must be divisible by H_kv ({H_kv})")

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must share dtype")
    # fp16/bf16 are required for the cute kernel; fp32 (and other SDPA-supported
    # dtypes) work via the SDPA fallback. The cute-path dtype gate lives inside
    # _call_cute_dsl so the SDPA route stays maximally permissive.

    if attn_bias is not None:
        if attn_bias.shape != (B, H, T_q, T_k):
            raise ValueError(
                f"attn_bias must be (B, H, T_q, T_k) = ({B}, {H}, {T_q}, {T_k}); "
                f"got {tuple(attn_bias.shape)}"
            )
        if attn_bias.dtype != q.dtype:
            attn_bias = attn_bias.to(q.dtype)

    if cache_seqlens is not None:
        if cache_seqlens.dim() != 1 or cache_seqlens.size(0) != B:
            raise ValueError(
                f"cache_seqlens must be (B,) = ({B},); got {tuple(cache_seqlens.shape)}"
            )
        if cache_seqlens.dtype != torch.int32:
            cache_seqlens = cache_seqlens.to(torch.int32)

    if out is None:
        out = torch.empty_like(q)
    else:
        if out.shape != q.shape or out.dtype != q.dtype:
            raise ValueError(
                f"out must match q in shape/dtype; "
                f"out={tuple(out.shape)}/{out.dtype} q={tuple(q.shape)}/{q.dtype}"
            )

    # ---- Backend dispatch ----------------------------------------------------
    # Fall back to SDPA for any dtype the cute kernel can't handle (fp32 etc.),
    # even when the backend is "cute".
    backend = select_backend()
    if backend == "sdpa" or q.dtype not in (torch.float16, torch.bfloat16):
        out.copy_(_sdpa_reference(q, k, v, softmax_scale, attn_bias, cache_seqlens))
        return out

    # backend == "cute"  (SM120 only in this revision)
    return _call_cute_dsl(
        q, k, v, out,
        softmax_scale=softmax_scale,
        attn_bias=attn_bias,
        cache_seqlens=cache_seqlens,
    )


# ---------------------------------------------------------------------------
# CuteDSL kernel invocation
# ---------------------------------------------------------------------------

def _call_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    softmax_scale: float,
    attn_bias: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Wrap torch tensors as CuteDSL descriptors and invoke the compiled kernel."""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    B, H, T_q, D = q.shape
    H_kv = k.size(1)
    dtype_str = "float16" if q.dtype == torch.float16 else "bfloat16"

    fn = get_compiled_fmha(
        head_dim=D, dtype_str=dtype_str,
        num_heads=H, num_kv_heads=H_kv,
        has_bias=(attn_bias is not None), paged=False,
    )

    # cp.async needs the head-dim ptr 16B-aligned; mark_compact_shape_dynamic
    # tells the IR verifier the leading dim has guaranteed divisibility.
    elem_bits = q.element_size() * 8
    align_div = 128 // elem_bits
    def _wrap(t: torch.Tensor):
        return (
            from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=t.dim() - 1)
            .mark_compact_shape_dynamic(
                mode=t.dim() - 1,
                stride_order=t.dim_order(),
                divisibility=align_div,
            )
        )

    mQ = _wrap(q.contiguous())
    mK = _wrap(k.contiguous())
    mV = _wrap(v.contiguous())
    mO = _wrap(out)

    if attn_bias is not None:
        mBias = _wrap(attn_bias.contiguous())
    else:
        # Zero-rank dummy: matches the kernel's compile-time `cute.rank > 0` gate.
        dummy = torch.empty((), dtype=q.dtype, device=q.device)
        mBias = from_dlpack(dummy, assumed_align=16)

    # The kernel's per-stream length mask always reads ``mCacheSeqlens[b]``;
    # in offline mode (no cache_seqlens) we synthesize a (B,) tensor filled
    # with T_k so the masking collapses to the tail-residue mask. Passing a
    # 0-rank dummy and gating on ``cute.rank`` inside the kernel is
    # tempting, but ``cute.rank > 0`` doesn't reliably constexpr-fold inside
    # @cute.jit and the resulting branch produces NaN through the masked
    # softmax.
    if cache_seqlens is not None:
        seqlens = cache_seqlens.contiguous()
    else:
        T_k = k.size(2)
        seqlens = torch.full((q.size(0),), T_k, dtype=torch.int32, device=q.device)
    mCacheSeqlens = (
        from_dlpack(seqlens, assumed_align=4)
        .mark_layout_dynamic(leading_dim=0)
    )

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(mQ, mK, mV, mO, mBias, mCacheSeqlens,
       cutlass.Float32(float(softmax_scale)), stream)
    return out
