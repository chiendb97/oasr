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


def _gather_paged_kv(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple:
    """Gather paged K/V into dense ``(B, H_kv, T_kv_logical, D)`` for SDPA.

    ``k_pool``/``v_pool`` have shape ``(num_blocks, block_size, H_kv, D)``.
    Trailing logical positions past each stream's ``cache_seqlens[b]`` get
    stale data; the caller masks them via ``cache_seqlens`` (passed to
    ``_sdpa_reference``).
    """
    B = block_table.size(0)
    block_size = k_pool.size(1)
    H_kv = k_pool.size(2)
    D = k_pool.size(3)
    num_blocks_per_seq = block_table.size(1)

    block_ids = block_table.long()
    k_full = k_pool[block_ids].reshape(
        B, num_blocks_per_seq * block_size, H_kv, D
    ).permute(0, 2, 1, 3)
    v_full = v_pool[block_ids].reshape(
        B, num_blocks_per_seq * block_size, H_kv, D
    ).permute(0, 2, 1, 3)
    return k_full, v_full


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

    paged = block_table is not None
    if paged:
        # Paged mode: K/V are per-layer pool views with shape
        # (num_blocks, block_size, H_kv, D). cache_seqlens is required
        # so the kernel can mask each stream to its valid kv length.
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"paged k/v must be (num_blocks, block_size, H_kv, D); "
                f"got {tuple(k.shape)} / {tuple(v.shape)}"
            )
        if k.shape != v.shape:
            raise ValueError(
                f"k and v must have identical shape: {k.shape} vs {v.shape}"
            )
        if k.size(3) != D:
            raise ValueError(
                f"paged k head_dim mismatch with q: q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}"
            )
        H_kv = k.size(2)
        block_size = k.size(1)
        if block_size <= 0:
            raise ValueError(f"paged block_size must be > 0; got {block_size}")
        if cache_seqlens is None:
            raise ValueError("paged mode requires cache_seqlens")
        if block_table.dim() != 2 or block_table.size(0) != B:
            raise ValueError(
                f"block_table must be (B, max_blocks_per_seq) = ({B}, ...); "
                f"got {tuple(block_table.shape)}"
            )
        if block_table.dtype != torch.int32:
            block_table = block_table.to(torch.int32)
        T_k = block_table.size(1) * block_size  # logical kv extent
    else:
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
    # even when the backend is "cute". For the SDPA fallback in paged mode we
    # gather the per-stream K/V into dense form first.
    backend = select_backend()
    if backend == "sdpa" or q.dtype not in (torch.float16, torch.bfloat16):
        if paged:
            k_dense, v_dense = _gather_paged_kv(k, v, block_table)
        else:
            k_dense, v_dense = k, v
        out.copy_(
            _sdpa_reference(
                q, k_dense, v_dense, softmax_scale, attn_bias, cache_seqlens,
            )
        )
        return out

    # backend == "cute"
    return _call_cute_dsl(
        q, k, v, out,
        softmax_scale=softmax_scale,
        attn_bias=attn_bias,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
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
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrap torch tensors as CuteDSL descriptors and invoke the compiled kernel."""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    B, H, T_q, D = q.shape
    paged = block_table is not None
    if paged:
        # Paged K/V layout: (num_blocks, block_size, H_kv, D).
        H_kv = k.size(2)
        block_size = k.size(1)
    else:
        H_kv = k.size(1)
        block_size = 0
    dtype_str = "float16" if q.dtype == torch.float16 else "bfloat16"

    fn = get_compiled_fmha(
        head_dim=D, dtype_str=dtype_str,
        num_heads=H, num_kv_heads=H_kv,
        has_bias=(attn_bias is not None),
        paged=paged, block_size=block_size,
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
        # In offline mode the per-column mask doubles as the T_k tail-
        # residue mask -- synthesize a (B,)-of-T_k tensor so the kernel
        # masks columns >= T_k to -inf via the same compare.
        T_k = k.size(2)
        seqlens = torch.full((q.size(0),), T_k, dtype=torch.int32, device=q.device)
    mCacheSeqlens = (
        from_dlpack(seqlens, assumed_align=4)
        .mark_layout_dynamic(leading_dim=0)
    )

    if paged:
        bt = block_table.contiguous()
        mBlockTable = (
            from_dlpack(bt, assumed_align=4)
            .mark_layout_dynamic(leading_dim=bt.dim() - 1)
        )
    else:
        # Dense path: zero-rank dummy block table; the kernel only reads
        # it when self._paged is True.
        dummy_bt = torch.empty((), dtype=torch.int32, device=q.device)
        mBlockTable = from_dlpack(dummy_bt, assumed_align=4)

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(mQ, mK, mV, mO, mBias, mCacheSeqlens, mBlockTable,
       cutlass.Float32(float(softmax_scale)), stream)
    return out
