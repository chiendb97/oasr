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

Per-call wrapper overhead is kept low by:

* Hoisting hot-path imports (``cuda.bindings.driver``, ``cutlass``,
  ``from_dlpack``) to module scope so they cost nothing at call time.
* Caching the CuteDSL descriptors for the "I'm not using this feature"
  zero-rank dummy tensors per ``(dtype, device)``.
* Reusing a process-cached ``cache_seqlens`` buffer in offline mode --
  avoids a fresh ``torch.full((B,), T_k, ...)`` allocation per call.
* Skipping ``.contiguous()`` when the tensor is already row-major.
* A ``validate=False`` fast path for callers that have already proven
  their inputs (the inference engine on the hot path, for example).
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
import torch.nn.functional as F

from .api_logging import oasr_api
from .jit.attention import get_compiled_fmha, select_backend


__all__ = ["fmha", "persistent_inputs"]


# Expose persistent_inputs as an attribute of fmha for the
# ``oasr.fmha.persistent_inputs()`` idiom (matches what's documented in the
# `oasr/attention.py` docstring); the assignment lives after the function
# definition below.


# ---------------------------------------------------------------------------
# Hot-path imports (hoisted to module scope)
# ---------------------------------------------------------------------------
# The CuteDSL imports below pull in MLIR / compiler infra and are heavy
# (~100-500 ms cold). We do them once at module-load so each fmha() call
# pays nothing. On hosts without CuteDSL (CPU-only, doc builds), the
# imports fail silently and ``_CUTE_AVAILABLE`` falls to False -- callers
# transparently route through the SDPA fallback.

try:
    import cuda.bindings.driver as _cuda_driver
    import cutlass as _cutlass
    from cutlass.cute.runtime import from_dlpack as _from_dlpack
    _CUTE_AVAILABLE = True
    _CUstream = _cuda_driver.CUstream
    _Float32 = _cutlass.Float32
except Exception:
    _CUTE_AVAILABLE = False
    _cuda_driver = None
    _cutlass = None
    _from_dlpack = None
    _CUstream = None
    _Float32 = None


# ---------------------------------------------------------------------------
# Module-scope caches for per-call constants
# ---------------------------------------------------------------------------

# Wrapped CuteDSL descriptors for the "no bias" / "no block_table"
# zero-rank dummies. Built lazily per ``(dtype, device_index)`` -- a handful
# of combos in practice. We keep both the underlying torch tensor and the
# wrapped descriptor alive in module scope; the tensors are tiny and never
# freed for the life of the process.
_dummy_bias_cache: dict = {}
_dummy_block_table_cache: dict = {}

# Process-cached cache_seqlens buffer per device. Grown lazily as B
# increases; filled in-place per call so we don't allocate a fresh tensor
# each fmha() call in offline mode.
_seqlens_buffer_cache: dict = {}

# ---------------------------------------------------------------------------
# Persistent-inputs descriptor cache
# ---------------------------------------------------------------------------
# DLPack capsule extraction + ``.mark_layout_dynamic`` + ``.mark_compact_
# shape_dynamic`` together cost ~5-10 us per tensor; on a 4-tensor (Q/K/V/O)
# hot path with ~40 us kernel time, that's wrapper > kernel.
#
# When the caller is reusing the same tensors back-to-back (the inference
# engine's chunked streaming loop, or a benchmark sweep), they can opt into
# descriptor caching by entering :func:`persistent_inputs`. Inside that
# context, ``_wrap`` looks up the (already-wrapped) descriptor by ``id(t)``
# instead of redoing the DLPack chain. Outside the context the cache is
# cleared and lookups fall back to the per-call wrap.
#
# We cache ``(tensor_ref, wrapped_descriptor)`` so the underlying tensor
# can't be GC'd while the descriptor lives -- if id() were reused for a
# different tensor we'd return a stale descriptor pointing at freed data.

_descriptor_cache: dict = {}
_persistent_depth: int = 0


@contextlib.contextmanager
def persistent_inputs():
    """Cache wrapped CuteDSL descriptors keyed on ``id(tensor)``.

    Enter this context when calling ``oasr.fmha(...)`` repeatedly with the
    same Q/K/V/out/bias/block_table/cache_seqlens tensors (engine hot path,
    benchmarks). The first call wraps as usual; subsequent calls hit the
    cache and skip the DLPack + ``mark_*`` chain.

    Nesting is supported; the cache is cleared only when the outermost
    context exits. Holding the cache prevents GC of the cached tensors,
    so don't enter the context around a tensor-allocation-heavy section.

    Example::

        with oasr.fmha.persistent_inputs():
            for chunk in stream:
                oasr.fmha(q, k_view, v_view, ..., out=out, validate=False)
    """
    global _persistent_depth
    _persistent_depth += 1
    try:
        yield
    finally:
        _persistent_depth -= 1
        if _persistent_depth == 0:
            _descriptor_cache.clear()


def _get_dummy_bias(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a process-cached zero-rank torch tensor for the no-bias path.

    The TVM-FFI compiled callable accepts torch tensors directly, so we
    keep the dummy as a torch tensor (no per-call ``from_dlpack`` wrap).
    ``has_bias=False`` kernels read ``mBias`` as a zero-rank tensor and
    the kernel never dereferences it.
    """
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


# cp.async with 128-bit copies requires the head-dim ptr 16B-aligned.
# 128 / 16 = 8 for fp16 / bf16 (both 16-bit).
_ALIGN_DIV_16B = 8


def _wrap16(t: torch.Tensor):
    """Wrap a 16-bit Q/K/V/O tensor with cp.async-friendly alignment.

    Q/K/V are loaded via 128-bit ``cp.async`` G2S; O is stored via the
    matching 128-bit universal copy. The leading head-dim axis is guaranteed
    divisible by 8 (``can_implement`` enforces ``head_dim % 8 == 0``), so
    we can tell the IR verifier so.
    """
    return (
        _from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=t.dim() - 1)
        .mark_compact_shape_dynamic(
            mode=t.dim() - 1,
            stride_order=t.dim_order(),
            divisibility=_ALIGN_DIV_16B,
        )
    )


def _wrap_bias(t: torch.Tensor):
    """Wrap a 16-bit attn_bias tensor.

    Unlike Q/K/V/O, ``attn_bias`` is read scalar-style by
    ``_add_bias_tile`` (one ``Float32(tBias_mn[r, c])`` per accumulator
    element -- no cp.async, no ldmatrix, no 128-bit vector load). Its
    trailing T_k axis is therefore allowed to be any positive integer
    (e.g. 249 frames in a real audio batch), so we don't assert a
    divisibility hint on the leading dim.
    """
    return _from_dlpack(t, assumed_align=16).mark_layout_dynamic(
        leading_dim=t.dim() - 1
    )


def _wrap_seqlens(t: torch.Tensor):
    return _from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=0)


def _wrap_block_table(t: torch.Tensor):
    return _from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _cached(t: torch.Tensor, wrap_fn):
    """Look up the wrapped descriptor by ``id(t)`` under
    :func:`persistent_inputs`, otherwise wrap fresh.

    The cache entry stores ``(tensor_ref, wrapped)`` so the tensor can't
    be GC'd while we hold a stale descriptor for it.
    """
    if _persistent_depth == 0:
        return wrap_fn(t)
    tid = id(t)
    entry = _descriptor_cache.get(tid)
    if entry is not None and entry[0] is t:
        return entry[1]
    wrapped = wrap_fn(t)
    _descriptor_cache[tid] = (t, wrapped)
    return wrapped


def _wrap16_cached(t):
    return _cached(t, _wrap16)


def _wrap_bias_cached(t):
    return _cached(t, _wrap_bias)


def _wrap_seqlens_cached(t):
    return _cached(t, _wrap_seqlens)


def _wrap_block_table_cached(t):
    return _cached(t, _wrap_block_table)


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


# Attach as ``oasr.fmha.persistent_inputs(...)`` so callers don't have to
# remember the separate import path. ``oasr_api`` uses ``functools.wraps``,
# so the wrapper object accepts attribute assignment.
fmha.persistent_inputs = persistent_inputs


# ---------------------------------------------------------------------------
# CuteDSL kernel invocation
# ---------------------------------------------------------------------------

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

    With ``options="--enable-tvm-ffi"`` baked into ``cute.compile`` and
    ``enable_tvm_ffi=True`` passed at descriptor build time, the compiled
    callable accepts torch tensors directly — no per-call ``from_dlpack``
    wrappers needed. This is the same path Flash Attention's
    ``flash_attn/cute`` uses; it is the only kernel-side call pattern
    that is safe to capture into a ``torch.cuda.CUDAGraph`` and replay
    across mutating chunks (the legacy per-call wrapper produces fresh
    Python capsule objects whose ownership is freed between replays,
    which causes the in-graph ``CUDA_ERROR_ILLEGAL_ADDRESS``).
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
