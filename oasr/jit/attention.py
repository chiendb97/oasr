# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT dispatch + compile cache for the OASR fused-attention kernels.

This module is the bridge between the public functional API (``oasr.fmha``)
and the per-arch CuteDSL backends under ``oasr.kernels.attention.cute``.

It does **not** use the Ninja-based C++ JIT pipeline (``oasr/jit/core.py``):
CuteDSL kernels are pure Python and are compiled via ``cutlass.cute.compile()``
which returns a callable closure. We cache one compiled callable per unique
configuration tuple and reuse it across invocations.

Backend selection is gated by ``OASR_ATTN_BACKEND``:

* ``sdpa`` -- always-on fallback; never calls into CuteDSL (use this when
  CuteDSL is unavailable or you want to bypass the cute path for debugging).
* ``cute`` -- require the CuteDSL kernel; raise on non-SM120 GPUs or compile
  failure.
* ``auto`` (default) -- use the CuteDSL kernel on SM120; on any other
  arch (or when the CuteDSL import fails) log a warning and fall back to
  PyTorch SDPA.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional, Tuple

logger = logging.getLogger("oasr.jit.attention")

# ---------------------------------------------------------------------------
# Backend mode
# ---------------------------------------------------------------------------

_BACKEND_ENV = "OASR_ATTN_BACKEND"
_VALID_BACKENDS = ("sdpa", "cute", "auto")


def _read_backend_mode() -> str:
    mode = os.environ.get(_BACKEND_ENV, "auto").lower()
    if mode not in _VALID_BACKENDS:
        logger.warning(
            "%s=%r is invalid; valid choices are %s. Falling back to 'auto'.",
            _BACKEND_ENV, mode, _VALID_BACKENDS,
        )
        mode = "auto"
    return mode


_BACKEND_MODE = _read_backend_mode()


def get_backend_mode() -> str:
    """Return the currently selected backend mode (``sdpa`` / ``cute`` / ``auto``)."""
    return _BACKEND_MODE


def set_backend_mode(mode: str) -> None:
    """Override the backend mode for the rest of the process. Mostly useful in tests."""
    global _BACKEND_MODE
    if mode not in _VALID_BACKENDS:
        raise ValueError(f"invalid backend mode {mode!r}; valid: {_VALID_BACKENDS}")
    _BACKEND_MODE = mode
    # Clear the per-config compile cache when switching modes so re-tests see the change.
    _compiled_fmha.cache_clear()
    _capability_probe.cache_clear()


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------

@functools.cache
def _capability_probe() -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    """Detect (major, minor) compute capability and which backend is usable.

    Returns ``((major, minor), backend)`` where ``backend`` is one of
    ``"cute"`` / ``"sdpa"`` / ``None``. The first element is ``None`` if
    no CUDA device is visible.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None, "sdpa"
        cap = torch.cuda.get_device_capability()
    except Exception:
        return None, "sdpa"

    if _BACKEND_MODE == "sdpa":
        return cap, "sdpa"

    sm = cap[0] * 10 + cap[1]
    cute_supported = sm in (80, 86, 89, 120)
    if cute_supported:
        # Try importing the per-arch backend; if CuteDSL isn't installed or
        # the subclass import explodes, log + degrade to SDPA in 'auto',
        # re-raise in 'cute'.
        try:
            from oasr.kernels.cute.attention.base import pick_arch_cls
            pick_arch_cls(*cap)
        except Exception as exc:
            if _BACKEND_MODE == "cute":
                raise
            logger.warning(
                "OASR CuteDSL attention requested but CuteDSL import failed (%s); "
                "falling back to PyTorch SDPA.", exc,
            )
            return cap, "sdpa"
        return cap, "cute"

    if _BACKEND_MODE == "cute":
        raise NotImplementedError(
            f"OASR_ATTN_BACKEND=cute but no CuteDSL kernel exists for "
            f"sm{cap[0]}{cap[1]} (supported: sm_80 / sm_86 / sm_89 / sm_120). "
            f"Set OASR_ATTN_BACKEND=auto to fall back to SDPA."
        )
    return cap, "sdpa"


def select_backend() -> str:
    """Return ``"cute"`` or ``"sdpa"`` based on the active mode + GPU."""
    return _capability_probe()[1]


# ---------------------------------------------------------------------------
# Compile cache
# ---------------------------------------------------------------------------

@functools.cache
def _compiled_fmha(
    arch: Tuple[int, int],
    head_dim: int,
    dtype_str: str,                  # "float16" or "bfloat16"
    num_heads: int,
    num_kv_heads: int,
    has_bias: bool,
    paged: bool,
    block_size: int,
    m_block: int,
    n_block: int,
    num_threads: int,
):
    """Return a compiled CuteDSL callable for the given configuration.

    Cache space is small in practice (a handful of variants per process), so
    ``functools.cache`` is sufficient.
    """
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    if dtype_str == "float16":
        cute_dtype = cutlass.Float16
        torch_dtype = torch.float16
    elif dtype_str == "bfloat16":
        cute_dtype = cutlass.BFloat16
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"unsupported dtype {dtype_str!r} (need float16 or bfloat16)")

    from oasr.kernels.cute.attention.base import pick_arch_cls
    cls = pick_arch_cls(*arch)
    if not cls.can_implement(
        dtype=cute_dtype, head_dim=head_dim,
        m_block_size=m_block, n_block_size=n_block,
        num_threads=num_threads, has_bias=has_bias,
    ):
        raise RuntimeError(
            f"{cls.__name__}.can_implement returned False for "
            f"head_dim={head_dim}, m_block={m_block}, n_block={n_block}, "
            f"num_threads={num_threads}, has_bias={has_bias}"
        )
    inst = cls(
        head_dim=head_dim, dtype=cute_dtype,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        has_bias=has_bias, paged=paged, block_size=block_size,
        m_block_size=m_block, n_block_size=n_block, num_threads=num_threads,
    )

    # Build dummy descriptor tensors for cute.compile — shapes only matter for
    # rank/dtype/dynamic-leading-dim signalling; values are unused.
    B, H, T_q, T_k = 1, num_heads, max(m_block, 8), max(n_block, 16)
    H_kv = num_kv_heads
    device = "cuda"
    q = torch.empty(B, H, T_q, head_dim, dtype=torch_dtype, device=device)
    k = torch.empty(B, H_kv, T_k, head_dim, dtype=torch_dtype, device=device)
    v = torch.empty(B, H_kv, T_k, head_dim, dtype=torch_dtype, device=device)
    o = torch.empty(B, H, T_q, head_dim, dtype=torch_dtype, device=device)

    # cp.async with 128-bit copies requires the head-dim ptr to be
    # 16B-aligned at IR-verify time. ``mark_compact_shape_dynamic`` with
    # divisibility=128/dtype.width tells the compiler that the leading dim
    # has guaranteed alignment so the verifier accepts the copy. Without it,
    # the upstream FA2 example also fails on this same alignment check.
    elem_bits = 16 if dtype_str == "float16" else 16  # bf16 is also 16b
    align_div = 128 // elem_bits
    def _wrap(t: torch.Tensor) -> "cute.Tensor":
        return (
            from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=t.dim() - 1)
            .mark_compact_shape_dynamic(
                mode=t.dim() - 1,
                stride_order=t.dim_order(),
                divisibility=align_div,
            )
        )

    mQ = _wrap(q); mK = _wrap(k); mV = _wrap(v); mO = _wrap(o)
    if has_bias:
        bias = torch.empty(B, H, T_q, T_k, dtype=torch_dtype, device=device)
        mBias = _wrap(bias)
    else:
        # Zero-rank dummy: cute.rank(mBias) > 0 in the kernel == False.
        mBias = from_dlpack(
            torch.empty((), dtype=torch_dtype, device=device), assumed_align=16,
        )

    seqlens = torch.zeros(B, dtype=torch.int32, device=device)
    mCacheSeqlens = (
        from_dlpack(seqlens, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    )

    import cuda.bindings.driver as cuda_driver
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    softmax_scale = cutlass.Float32(1.0 / (head_dim ** 0.5))
    return cute.compile(
        inst, mQ, mK, mV, mO, mBias, mCacheSeqlens, softmax_scale, stream,
    )


def get_compiled_fmha(
    *,
    head_dim: int,
    dtype_str: str,
    num_heads: int,
    num_kv_heads: int,
    has_bias: bool,
    paged: bool,
    block_size: int = 0,
    m_block: int = 64,
    n_block: int = 64,
    num_threads: int = 128,
):
    """Public accessor — returns a compiled CuteDSL callable, compiling on first call."""
    cap = _capability_probe()[0]
    if cap is None:
        raise RuntimeError("no CUDA device available")
    return _compiled_fmha(
        cap, head_dim, dtype_str, num_heads, num_kv_heads,
        has_bias, paged, block_size, m_block, n_block, num_threads,
    )


# ---------------------------------------------------------------------------
# Warmup helper
# ---------------------------------------------------------------------------

def warmup_fmha(
    *,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    max_batch_size: int,         # noqa: ARG001 -- reserved for future use
    chunk_size: int,             # noqa: ARG001
    max_attention_key_size: int, # noqa: ARG001
    device: Any,                 # noqa: ARG001
    dtype: Any,
) -> None:
    """Eagerly populate the compile cache for a given Conformer config.

    Mirrors :func:`oasr.layers.attention.attention.warmup_flex_attention`. Skips
    silently on archs other than SM120 or when the active backend is ``sdpa``.
    """
    if select_backend() != "cute":
        return
    import torch

    dtype_str = "float16" if dtype == torch.float16 else (
        "bfloat16" if dtype == torch.bfloat16 else None
    )
    if dtype_str is None:
        return
    for has_bias in (False, True):
        try:
            get_compiled_fmha(
                head_dim=head_dim, dtype_str=dtype_str,
                num_heads=n_head, num_kv_heads=n_kv_head,
                has_bias=has_bias, paged=False,
            )
        except Exception as exc:
            logger.warning("warmup_fmha (has_bias=%s) failed: %s", has_bias, exc)
