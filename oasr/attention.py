# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Public functional API for the OASR fused multi-head attention.

This is the single entry point that dispatches between the SDPA fallback
(always available) and the CuteDSL kernel (SM80 / SM86 / SM89 / SM120 in
this revision) based on ``OASR_ATTN_BACKEND`` and the active GPU's compute
capability.

Three cache modes are encoded in one signature:

* **offline**         -- ``block_table is None`` and ``cache_seqlens is None``
                         (attend over the full ``T_k``).
* **dense streaming** -- ``block_table is None`` and ``cache_seqlens is not None``
                         (attend over ``[0, cache_seqlens[b])`` per stream;
                         caller has already concatenated old + new K/V).
* **paged streaming** -- ``block_table is not None``  (k/v are pool views,
                         ``cache_seqlens`` required).

The compiled CuteDSL callable is built with ``--enable-tvm-ffi`` (see
``oasr/jit/attention.py``), so it accepts raw torch tensors at call time and
is safe to capture into a ``torch.cuda.CUDAGraph``. This wrapper therefore
only has to: pick the backend, normalise strides, pad bias for the kernel's
CTA tile, and reuse a couple of small process-cached dummy tensors. A
``validate=False`` fast path skips shape/dtype checks for callers that have
already proven their inputs (the inference engine on the hot path).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .api_logging import oasr_api
from .jit.attention import get_compiled_fmha, select_backend


__all__ = ["fmha", "fmha_varlen"]


# ---------------------------------------------------------------------------
# Hot-path imports (hoisted to module scope)
# ---------------------------------------------------------------------------
# The CuteDSL imports below pull in MLIR / compiler infra and are heavy
# (~100-500 ms cold). We do them once at module-load so each fmha() call
# pays nothing. On hosts without CuteDSL (CPU-only, doc builds), the
# imports fail silently and callers transparently route through the SDPA
# fallback via ``select_backend()``.

try:
    import cuda.bindings.driver as _cuda_driver
    import cutlass as _cutlass
    _CUstream = _cuda_driver.CUstream
    _Float32 = _cutlass.Float32
    _Int32 = _cutlass.Int32
except Exception:
    _cuda_driver = None
    _cutlass = None
    _CUstream = None
    _Float32 = None
    _Int32 = None


# ---------------------------------------------------------------------------
# Module-scope caches for per-call constants
# ---------------------------------------------------------------------------

# Process-cached zero-rank dummies for the "no bias" / "no block_table"
# paths. The kernel never dereferences them, but cute still requires a
# tensor of the right dtype/rank. Tiny and never freed.
_dummy_bias_cache: dict = {}
_dummy_block_table_cache: dict = {}

# Process-cached cache_seqlens buffer per device. Grown lazily as B
# increases; filled in-place per call so we don't allocate a fresh tensor
# each fmha() call in offline mode.
_seqlens_buffer_cache: dict = {}

# Cute fmha CTA tile sizes (must match ``FmhaSm80._m_block_size`` /
# ``_n_block_size`` defaults).  In the paged path the kernel reads the full
# ``(M_BLOCK, N_BLOCK)`` bias tile without per-row T_q predication, so the
# bias must be padded up to these multiples.  Under CUDA Graph capture the
# allocator fragments the address space and an OOB read can land in an
# unmapped segment, raising ``cudaErrorIllegalAddress``; padding keeps every
# tile read in-bounds.
_KERNEL_M_BLOCK = 64
_KERNEL_N_BLOCK = 64


def _get_dummy_bias(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a process-cached zero-rank torch tensor for the no-bias path."""
    key = (dtype, device.index if device.index is not None else 0)
    t = _dummy_bias_cache.get(key)
    if t is None:
        t = torch.empty((), dtype=dtype, device=device)
        _dummy_bias_cache[key] = t
    return t


def _get_dummy_block_table(device: torch.device) -> torch.Tensor:
    """Return the cached zero-rank int32 dummy block_table torch tensor."""
    idx = device.index if device.index is not None else 0
    t = _dummy_block_table_cache.get(idx)
    if t is None:
        t = torch.empty((), dtype=torch.int32, device=device)
        _dummy_block_table_cache[idx] = t
    return t


def _get_offline_seqlens(B: int, T_k: int, device: torch.device) -> torch.Tensor:
    """Return a ``(B,)`` int32 tensor filled with ``T_k``.

    Uses a process-cached buffer that grows as B increases; fills the live
    view in place so we don't allocate per call. The returned tensor is a
    view (not a fresh allocation); the underlying buffer outlives the call.
    """
    idx = device.index if device.index is not None else 0
    buf = _seqlens_buffer_cache.get(idx)
    if buf is None or buf.numel() < B:
        # Round up to a small power-of-two so we don't churn on B+/-1.
        cap = max(8, 1 << (B - 1).bit_length())
        buf = torch.empty((cap,), dtype=torch.int32, device=device)
        _seqlens_buffer_cache[idx] = buf
    view = buf[:B]
    view.fill_(T_k)
    return view


def _canonical_strides(shape) -> tuple:
    """Row-major (C-order) strides for ``shape`` — what ``torch.empty(shape)`` produces."""
    strides = []
    s = 1
    for d in reversed(shape):
        strides.append(s)
        s *= d
    return tuple(reversed(strides))


def _ensure_canonical(t: torch.Tensor) -> torch.Tensor:
    """Return a view of ``t`` with strictly row-major C-order strides.

    ``torch.Tensor.is_contiguous()`` reports True when any size-1 dim is
    present even if that dim's stride is non-canonical (it leaves the
    pre-transpose stride untouched). cute's
    ``mark_compact_shape_dynamic`` rejects that mismatch with
    "stride_order is not consistent with the layout", so we can't just
    rely on ``is_contiguous()``. When the memory layout is already
    row-major (``is_contiguous()`` True) we fix the metadata with
    ``as_strided`` for free; otherwise we materialise via ``.contiguous()``.
    """
    canonical = _canonical_strides(t.shape)
    if t.stride() == canonical:
        return t
    if t.is_contiguous():
        return t.as_strided(t.shape, canonical)
    return t.contiguous()


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
    """Gather paged K/V into dense ``(B, H_kv, T_kv_logical, D)`` for SDPA."""
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


def _sdpa_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    bias_offsets: Optional[torch.Tensor],
    out: torch.Tensor,
) -> torch.Tensor:
    """Varlen (sequence-packed) attention reference via per-segment SDPA.

    Packed layout: ``q`` is ``(total_q, H, D)``, ``k``/``v`` are
    ``(total_k, H_kv, D)``; ``cu_seqlens_*`` are ``(S+1,)`` prefix sums.  Each
    segment attends only to itself.  ``attn_bias`` (when present) is the packed
    block-diagonal rel-pos bias: a flat buffer whose segment ``s`` block is
    ``(H, T_q_s, T_k_s)`` row-major at ``bias_offsets[s]`` (already pre-scaled,
    SDPA semantics).  Writes into ``out`` ``(total_q, H, D)`` and returns it.
    """
    H = q.size(1)
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    bo = bias_offsets.tolist() if bias_offsets is not None else None
    for s in range(len(cu_q) - 1):
        qa, qb = cu_q[s], cu_q[s + 1]
        ka, kb = cu_k[s], cu_k[s + 1]
        if qb == qa:
            continue
        # (T, heads, D) -> (1, heads, T, D)
        qs = q[qa:qb].transpose(0, 1).unsqueeze(0)
        ks = k[ka:kb].transpose(0, 1).unsqueeze(0)
        vs = v[ka:kb].transpose(0, 1).unsqueeze(0)
        if attn_bias is not None and bo is not None:
            bias_s = attn_bias[bo[s]:bo[s + 1]].view(1, H, qb - qa, kb - ka)
        else:
            bias_s = None
        out_s = _sdpa_reference(qs, ks, vs, softmax_scale, bias_s, None)
        out[qa:qb] = out_s.squeeze(0).transpose(0, 1)
    return out


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    attn_bias: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Functional SDPA path producing the same result as the cute kernel."""
    B, H, T_q, D = q.shape
    H_kv = k.size(1)
    T_kv = k.size(2)
    if H_kv != H:
        if H % H_kv != 0:
            raise ValueError(f"H ({H}) must be divisible by H_kv ({H_kv})")
        n_repeat = H // H_kv
        k = k.repeat_interleave(n_repeat, dim=1)
        v = v.repeat_interleave(n_repeat, dim=1)

    # ``attn_bias`` may be padded along the T_q / T_k axes (callers pad up
    # to the cute kernel's (M_BLOCK, N_BLOCK) tile size so the kernel's
    # unpredicated bias-tile reads stay in-bounds under CUDA Graph capture).
    # SDPA wants ``(B, H, T_q, T_kv)`` exactly, so trim back here.
    if attn_bias is not None and (attn_bias.size(-2) != T_q or attn_bias.size(-1) != T_kv):
        attn_bias = attn_bias[..., :T_q, :T_kv]

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
# Validation (single-pass; only run when validate=True)
# ---------------------------------------------------------------------------

def _validate_inputs(
    q, k, v,
    attn_bias, cache_seqlens, block_table, out,
    paged: bool,
    B: int, H: int, T_q: int, D: int,
    H_kv: int, T_k: int,
):
    """Single walk over Q/K/V/bias/seqlens/block_table/out shapes + dtypes."""
    if q.dim() != 4:
        raise ValueError(f"q must be (B, H, T_q, D), got shape {tuple(q.shape)}")

    if paged:
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
        if cache_seqlens is None:
            raise ValueError("paged mode requires cache_seqlens")
        if block_table.dim() != 2 or block_table.size(0) != B:
            raise ValueError(
                f"block_table must be (B, max_blocks_per_seq) = ({B}, ...); "
                f"got {tuple(block_table.shape)}"
            )
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

    if H % H_kv != 0:
        raise ValueError(f"H ({H}) must be divisible by H_kv ({H_kv})")

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must share dtype")

    if attn_bias is not None:
        bs = attn_bias.shape
        # ``attn_bias`` may be padded up to the kernel's (M_BLOCK, N_BLOCK)
        # tile size along the T_q and T_k axes so the cute kernel's
        # unpredicated bias-tile reads stay in-bounds even when the active
        # ``T_q`` / ``T_k`` are smaller. We accept any padded extent.
        if len(bs) != 4 or bs[0] != B or bs[1] != H or bs[2] < T_q or bs[3] < T_k:
            raise ValueError(
                f"attn_bias must be (B, H, ≥T_q, ≥T_k) = ({B}, {H}, ≥{T_q}, ≥{T_k}); "
                f"got {tuple(attn_bias.shape)}"
            )

    if cache_seqlens is not None and (
        cache_seqlens.dim() != 1 or cache_seqlens.size(0) != B
    ):
        raise ValueError(
            f"cache_seqlens must be (B,) = ({B},); got {tuple(cache_seqlens.shape)}"
        )

    if out is not None and (out.shape != q.shape or out.dtype != q.dtype):
        raise ValueError(
            f"out must match q in shape/dtype; "
            f"out={tuple(out.shape)}/{out.dtype} q={tuple(q.shape)}/{q.dtype}"
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
    validate: bool = True,
) -> torch.Tensor:
    """Fused multi-head attention.

    Parameters
    ----------
    q : Tensor
        ``(B, H, T_q, D)`` query, fp16 or bf16 for the cute path.
    k, v : Tensor
        Contiguous mode: ``(B, H_kv, T_k, D)``.
        Paged mode    : ``(num_blocks, block_size, H_kv, D)``.
    softmax_scale : float
        Softmax scale (typically ``1/sqrt(D)``).
    attn_bias : Tensor, optional
        ``(B, H, T_q, T_k_max)`` additive bias (Transformer-XL matrix_bd, etc.).
        Treated as a post-scale logit (SDPA semantics).
    cache_seqlens : Tensor, optional
        ``(B,)`` int32. Per-stream valid k-length.
    block_table : Tensor, optional
        ``(B, max_blocks_per_seq)`` int32. When provided, k/v are paged.
    out : Tensor, optional
        Pre-allocated ``(B, H, T_q, D)`` output.
    validate : bool, default True
        When False, skip shape/dtype validation. Caller takes responsibility
        for input correctness. Used by the engine hot path to shave a few
        microseconds per call.

    Returns
    -------
    out : Tensor
        ``(B, H, T_q, D)``.
    """
    # Fast unpack -- avoid per-attribute size() calls.
    B, H, T_q, D = q.shape
    paged = block_table is not None
    if paged:
        H_kv = k.size(2)
        block_size = k.size(1)
        # Pad the caller's bias up to the cute kernel's CTA tile and trim
        # the block table to match.  The cute kernel reads the bias tile
        # unpredicated; padding keeps every read in-bounds under CUDA Graph
        # capture.  The SDPA fallback trims oversized bias, so this is
        # backend-neutral.
        attn_bias, block_table = _pad_paged_inputs(
            attn_bias, block_table, block_size=block_size,
        )
        T_k = block_table.size(1) * block_size
    else:
        H_kv = k.size(1)
        T_k = k.size(2)
        block_size = 0

    if validate:
        _validate_inputs(
            q, k, v, attn_bias, cache_seqlens, block_table, out,
            paged, B, H, T_q, D, H_kv, T_k,
        )
        # Convenience dtype coercions; only run when validating.
        if attn_bias is not None and attn_bias.dtype != q.dtype:
            attn_bias = attn_bias.to(q.dtype)
        if cache_seqlens is not None and cache_seqlens.dtype != torch.int32:
            cache_seqlens = cache_seqlens.to(torch.int32)
        if paged and block_table.dtype != torch.int32:
            block_table = block_table.to(torch.int32)

    if out is None:
        # ``empty_like(q)`` preserves q's memory format: when q came from a
        # transpose (non-canonical strides) the output buffer would inherit
        # those strides, and ``_call_cute_dsl``'s ``_ensure_canonical`` would
        # then copy into a fresh canonical buffer that the cute kernel writes
        # to — leaving the caller's ``out`` reference uninitialised. Allocate
        # the output directly in canonical contiguous_format so the kernel
        # writes land in the tensor we hand back.
        out = torch.empty(
            q.shape, dtype=q.dtype, device=q.device,
            memory_format=torch.contiguous_format,
        )

    # ---- Backend dispatch ---------------------------------------------------
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

    return _call_cute_dsl(
        q, k, v, out, B=B, H=H, T_q=T_q, D=D, H_kv=H_kv, T_k=T_k,
        paged=paged, block_size=block_size,
        softmax_scale=softmax_scale,
        attn_bias=attn_bias,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
    )


# ---------------------------------------------------------------------------
# Variable-length (sequence-packed) attention
# ---------------------------------------------------------------------------


def _validate_varlen_inputs(
    q, k, v, cu_seqlens_q, cu_seqlens_k, attn_bias, bias_offsets, out,
) -> None:
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            "varlen q/k/v must be packed 3-D (total, H, D); got "
            f"{tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}"
        )
    if k.shape != v.shape:
        raise ValueError(f"k and v must match: {tuple(k.shape)} vs {tuple(v.shape)}")
    if q.size(2) != k.size(2):
        raise ValueError("q and k head_dim must match")
    H, H_kv = q.size(1), k.size(1)
    if H % H_kv != 0:
        raise ValueError(f"H ({H}) must be divisible by H_kv ({H_kv})")
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu.dim() != 1 or cu.numel() < 2:
            raise ValueError(f"{name} must be 1-D with >= 2 entries")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have equal length")
    if (attn_bias is None) != (bias_offsets is None):
        raise ValueError("attn_bias and bias_offsets must be provided together")
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype):
        raise ValueError("out must match q in shape/dtype")


@oasr_api
def fmha_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    attn_bias: Optional[torch.Tensor] = None,
    bias_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    validate: bool = True,
) -> torch.Tensor:
    """Variable-length (sequence-packed) fused multi-head attention.

    Parameters
    ----------
    q : Tensor
        ``(total_q, H, D)`` packed query (segments concatenated, no batch dim).
    k, v : Tensor
        ``(total_k, H_kv, D)`` packed key/value.
    softmax_scale : float
        Softmax scale (typically ``1/sqrt(D)``).
    cu_seqlens_q, cu_seqlens_k : Tensor
        ``(S+1,)`` int32 prefix sums of per-segment lengths.  Segment ``s``
        spans ``[cu[s], cu[s+1])`` and attends only to itself.
    max_seqlen_q, max_seqlen_k : int
        Host-side max segment lengths (size the kernel tile loop / pos-emb).
    attn_bias : Tensor, optional
        Packed block-diagonal additive bias (pre-scaled, SDPA semantics): a
        flat buffer whose segment-``s`` block is ``(H, T_q_s, T_k_s)``
        row-major at ``bias_offsets[s]``.
    bias_offsets : Tensor, optional
        ``(S+1,)`` int64 prefix sum of ``H * T_q_s * T_k_s`` block sizes.
    out : Tensor, optional
        Pre-allocated ``(total_q, H, D)`` output.

    Returns
    -------
    out : Tensor
        ``(total_q, H, D)``.
    """
    if validate:
        _validate_varlen_inputs(
            q, k, v, cu_seqlens_q, cu_seqlens_k, attn_bias, bias_offsets, out,
        )
        if attn_bias is not None and attn_bias.dtype != q.dtype:
            attn_bias = attn_bias.to(q.dtype)
        if cu_seqlens_q.dtype != torch.int32:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        if cu_seqlens_k.dtype != torch.int32:
            cu_seqlens_k = cu_seqlens_k.to(torch.int32)

    if out is None:
        out = torch.empty(
            q.shape, dtype=q.dtype, device=q.device,
            memory_format=torch.contiguous_format,
        )

    backend = select_backend()
    if (
        backend == "cute"
        and q.dtype in (torch.float16, torch.bfloat16)
        and _varlen_cute_available()
    ):
        return _call_cute_dsl_varlen(
            q, k, v, out,
            softmax_scale=softmax_scale,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            attn_bias=attn_bias, bias_offsets=bias_offsets,
        )

    return _sdpa_varlen_reference(
        q, k, v, softmax_scale, cu_seqlens_q, cu_seqlens_k,
        attn_bias, bias_offsets, out,
    )


def _varlen_cute_available() -> bool:
    """Whether the CuteDSL varlen kernel path is wired."""
    return _Int32 is not None


def _call_cute_dsl_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    attn_bias: Optional[torch.Tensor],
    bias_offsets: Optional[torch.Tensor],
) -> torch.Tensor:
    """Invoke the compiled CuteDSL varlen kernel on packed inputs.

    Single launch over the whole pack: grid ``(ceil(max_seqlen_q/M), S, H)``;
    each CTA builds per-segment ``(1, heads, seqlen, D)`` views via
    ``domain_offset`` so the dense mainloop + helpers run unchanged with zero
    attention padding and no host-side scatter.
    """
    from .jit.attention import get_compiled_fmha_varlen

    H, D = q.size(1), q.size(2)
    H_kv = k.size(1)
    dtype_str = "float16" if q.dtype is torch.float16 else "bfloat16"
    has_bias = attn_bias is not None

    fn = get_compiled_fmha_varlen(
        head_dim=D, dtype_str=dtype_str,
        num_heads=H, num_kv_heads=H_kv,
        has_bias=has_bias, bias_aligned=False,
    )

    q_t = _ensure_canonical(q)
    k_t = _ensure_canonical(k)
    v_t = _ensure_canonical(v)
    o_t = _ensure_canonical(out)
    cu_q = _ensure_canonical(cu_seqlens_q)
    cu_k = _ensure_canonical(cu_seqlens_k)

    if has_bias:
        bias_t = _ensure_canonical(attn_bias)
        if bias_offsets.dtype != torch.int32:
            bias_offsets = bias_offsets.to(torch.int32)
        bias_off_t = _ensure_canonical(bias_offsets)
    else:
        bias_t = _get_dummy_bias(q.dtype, q.device)
        bias_off_t = _get_dummy_block_table(q.device)

    stream = _CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q_t, k_t, v_t, o_t, bias_t, bias_off_t, cu_q, cu_k,
        _Int32(int(max_seqlen_q)), _Float32(float(softmax_scale)), stream,
    )
    if o_t.data_ptr() != out.data_ptr():
        out.copy_(o_t)
    return out


# ---------------------------------------------------------------------------
# CuteDSL kernel invocation
# ---------------------------------------------------------------------------

def _pad_paged_inputs(
    attn_bias: Optional[torch.Tensor],
    block_table: torch.Tensor,
    *,
    block_size: int,
) -> tuple:
    """Pad ``attn_bias`` and trim ``block_table`` to the kernel tile.

    The cute fmha kernel reads the full ``(M_BLOCK, N_BLOCK)`` bias tile
    without per-row predication, so the bias must be padded up to those
    multiples or an OOB tile read can land in an unmapped segment under
    CUDA Graph capture (``cudaErrorIllegalAddress``).  The block table is
    trimmed to match so the kernel's ``ceil(T_kv / N_BLOCK)`` walk only
    indexes the rows we actually populated.

    No-op when ``attn_bias is None`` -- the kernel takes its own no-bias
    path and uses ``cache_seqlens`` directly for the K-block bound.
    """
    if attn_bias is None:
        return attn_bias, block_table

    T_q = attn_bias.size(-2)
    T_kv = attn_bias.size(-1)
    T_q_padded = ((T_q + _KERNEL_M_BLOCK - 1) // _KERNEL_M_BLOCK) * _KERNEL_M_BLOCK
    T_kv_padded = ((T_kv + _KERNEL_N_BLOCK - 1) // _KERNEL_N_BLOCK) * _KERNEL_N_BLOCK

    if T_q_padded > T_q or T_kv_padded > T_kv:
        B, H = attn_bias.shape[:2]
        padded = torch.zeros(
            B, H, T_q_padded, T_kv_padded,
            dtype=attn_bias.dtype, device=attn_bias.device,
        )
        padded[:, :, :T_q, :T_kv] = attn_bias
        attn_bias = padded

    max_blocks_needed = T_kv_padded // block_size
    if block_table.size(1) > max_blocks_needed:
        block_table = block_table[:, :max_blocks_needed]

    return attn_bias, block_table


def _call_cute_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    B: int, H: int, T_q: int, D: int, H_kv: int, T_k: int,
    paged: bool, block_size: int,
    softmax_scale: float,
    attn_bias: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
) -> torch.Tensor:
    """Invoke the TVM-FFI compiled CuteDSL kernel with raw torch tensors.

    ``oasr/jit/attention.py`` builds the callable with
    ``cute.compile(..., options="--enable-tvm-ffi")`` and
    ``from_dlpack(..., enable_tvm_ffi=True)``, so the compiled ``fn``
    accepts torch tensors directly -- no per-call DLPack wrappers needed.
    This is the same pattern Flash Attention's ``flash_attn/cute`` uses;
    it is the only kernel-side call pattern that is safe to capture into
    a ``torch.cuda.CUDAGraph`` and replay across mutating chunks (the
    legacy per-call wrapper produced fresh Python capsule objects whose
    ownership was freed between replays, causing in-graph
    ``CUDA_ERROR_ILLEGAL_ADDRESS``).
    """
    dtype_str = "float16" if q.dtype is torch.float16 else "bfloat16"
    device = q.device

    # When bias is present, prefer the vectorized rmem-load path. The
    # ``bias_aligned`` flag tells the kernel that the bias trailing-axis
    # stride (= T_k * 2 B for fp16/bf16) is divisible by 4 B so a b32
    # col-pair load is always safe. Falls back to a slower predicated path
    # for odd T_k (e.g. real-world ASR T_k = 33/249 audio frames).
    bias_aligned = attn_bias is not None and (attn_bias.size(-1) % 2 == 0)

    fn = get_compiled_fmha(
        head_dim=D, dtype_str=dtype_str,
        num_heads=H, num_kv_heads=H_kv,
        has_bias=(attn_bias is not None),
        paged=paged, block_size=block_size,
        bias_aligned=bias_aligned,
    )

    # Q/K/V/O must have strictly canonical row-major strides; see
    # ``_ensure_canonical`` for why ``.is_contiguous()`` alone is unsafe.
    q_t = _ensure_canonical(q)
    k_t = _ensure_canonical(k)
    v_t = _ensure_canonical(v)
    o_t = _ensure_canonical(out)

    if attn_bias is not None:
        bias_t = _ensure_canonical(attn_bias)
    else:
        bias_t = _get_dummy_bias(q.dtype, device)

    # cache_seqlens: synthesize a (B,) buffer of T_k in offline mode using
    # the cached process buffer; otherwise use the caller's tensor.
    if cache_seqlens is not None:
        seqlens_t = _ensure_canonical(cache_seqlens)
    else:
        seqlens_t = _get_offline_seqlens(B, T_k, device)

    if paged:
        block_table_t = _ensure_canonical(block_table)
    else:
        block_table_t = _get_dummy_block_table(device)

    stream = _CUstream(torch.cuda.current_stream().cuda_stream)
    # TVM-FFI compiled callable accepts torch tensors directly.
    fn(q_t, k_t, v_t, o_t, bias_t, seqlens_t, block_table_t,
       _Float32(float(softmax_scale)), stream)
    # When ``out`` had non-canonical strides (e.g. caller passed a transpose
    # view) ``_ensure_canonical`` allocated a fresh canonical buffer that the
    # cute kernel writes to; copy the result back so the caller's reference
    # observes the output. ``fmha``'s ``out is None`` path now allocates
    # canonical-contiguous so this guard only fires when the caller hands in
    # a non-canonical pre-allocated ``out``.
    if o_t.data_ptr() != out.data_ptr():
        out.copy_(o_t)
    return out
