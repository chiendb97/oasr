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
            _BACKEND_ENV,
            mode,
            _VALID_BACKENDS,
        )
        mode = "auto"
    return mode


_BACKEND_MODE = _read_backend_mode()

# Resolved at module load (or first call from a test that flipped the env)
# so the steady-state ``select_backend()`` call is a bare attribute read,
# not a functools.cache dict lookup.
_RESOLVED_BACKEND: Optional[str] = None


def get_backend_mode() -> str:
    """Return the currently selected backend mode (``sdpa`` / ``cute`` / ``auto``)."""
    return _BACKEND_MODE


def set_backend_mode(mode: str) -> None:
    """Override the backend mode for the rest of the process. Mostly useful in tests."""
    global _BACKEND_MODE, _RESOLVED_BACKEND
    if mode not in _VALID_BACKENDS:
        raise ValueError(f"invalid backend mode {mode!r}; valid: {_VALID_BACKENDS}")
    _BACKEND_MODE = mode
    _RESOLVED_BACKEND = None  # force re-probe on next select_backend()
    # Clear the per-config compile cache when switching modes so re-tests see the change.
    _compiled_fmha.cache_clear()
    _capability_probe.cache_clear()
    fmha_config_supported.cache_clear()


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


#: Oldest verified CuTeDSL release. Reject older APIs before compilation so a
#: dependency mismatch is not reported as a kernel failure.
MIN_CUTEDSL_VERSION = (4, 5, 2)


def _version_tuple(text: str) -> Tuple[int, ...]:
    parts: list = []
    for chunk in text.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _check_cutedsl_version() -> None:
    """Raise if the installed CuTeDSL predates :data:`MIN_CUTEDSL_VERSION`.

    Called from the capability probe, so ``auto`` degrades to SDPA with a
    warning naming the version and ``cute`` re-raises — the same handling as a
    missing CuTeDSL install, which is what a too-old one amounts to.
    """
    import cutlass

    raw = getattr(cutlass, "__version__", "")
    found = _version_tuple(raw)
    # An unparseable version is not evidence of anything; let the kernels try.
    if found and found < MIN_CUTEDSL_VERSION:
        want = ".".join(str(p) for p in MIN_CUTEDSL_VERSION)
        raise RuntimeError(
            f"CuTeDSL {raw} is older than the minimum this build targets "
            f"({want}). Upgrade with `pip install -U 'nvidia-cutlass-dsl>={want},<5'`, "
            f"or set OASR_ATTN_BACKEND=sdpa to use the PyTorch fallback."
        )


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

            _check_cutedsl_version()
            pick_arch_cls(*cap)
        except Exception as exc:
            if _BACKEND_MODE == "cute":
                raise
            logger.warning(
                "OASR CuteDSL attention requested but CuteDSL import failed (%s); "
                "falling back to PyTorch SDPA.",
                exc,
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
    """Return ``"cute"`` or ``"sdpa"`` based on the active mode + GPU.

    Cached at module scope after the first call so the hot path is a bare
    global read instead of a ``functools.cache`` dict lookup. Cleared by
    :func:`set_backend_mode`.
    """
    global _RESOLVED_BACKEND
    if _RESOLVED_BACKEND is None:
        _RESOLVED_BACKEND = _capability_probe()[1]
    return _RESOLVED_BACKEND


# Resolve eagerly at module load if CUDA is sitting there ready to go --
# this moves the one-time probe cost out of the first ``fmha()`` call.
try:
    import torch as _torch_probe

    if _torch_probe.cuda.is_available():
        _RESOLVED_BACKEND = _capability_probe()[1]
    del _torch_probe
except Exception:
    # torch not importable on this host; leave _RESOLVED_BACKEND=None so
    # the first call lazily probes.
    pass


# ---------------------------------------------------------------------------
# Compile cache
# ---------------------------------------------------------------------------


@functools.cache
def _compiled_fmha(
    arch: Tuple[int, int],
    head_dim: int,
    dtype_str: str,  # "float16" or "bfloat16"
    num_heads: int,
    num_kv_heads: int,
    has_bias: bool,
    paged: bool,
    block_size: int,
    m_block: int,
    n_block: int,
    num_threads: int,
    bias_aligned: bool = False,
    causal: bool = False,
    has_seqstart: bool = False,
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
    # Resolve the tile before validating/building: a head_dim whose requested
    # 64-wide K tile cannot fit any ring depth gets a narrower one rather than
    # being refused (``select_tile``).  The cache key keeps the *requested*
    # n_block, which is fine — the resolution is a pure function of the key.
    n_block_eff, num_stages_eff = cls.select_tile(
        head_dim=head_dim,
        m_block_size=m_block,
        n_block_size=n_block,
        paged=paged,
        block_size=block_size,
    )
    if num_stages_eff:
        n_block = n_block_eff
    if not cls.can_implement(
        dtype=cute_dtype,
        head_dim=head_dim,
        m_block_size=m_block,
        n_block_size=n_block,
        num_threads=num_threads,
        has_bias=has_bias,
        paged=paged,
        block_size=block_size,
        bias_aligned=bias_aligned,
        causal=causal,
        has_seqstart=has_seqstart,
    ):
        raise RuntimeError(
            f"{cls.__name__}.can_implement returned False for "
            f"head_dim={head_dim}, m_block={m_block}, n_block={n_block}, "
            f"num_threads={num_threads}, has_bias={has_bias}, "
            f"paged={paged}, block_size={block_size}, "
            f"bias_aligned={bias_aligned}"
        )
    inst = cls(
        head_dim=head_dim,
        dtype=cute_dtype,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        has_bias=has_bias,
        paged=paged,
        block_size=block_size,
        m_block_size=m_block,
        n_block_size=n_block,
        num_threads=num_threads,
        bias_aligned=bias_aligned,
        causal=causal,
        has_seqstart=has_seqstart,
        num_stages=num_stages_eff or None,
    )

    # Build dummy descriptor tensors for cute.compile — shapes only matter for
    # rank/dtype/dynamic-leading-dim signalling; values are unused.
    B, H, T_q, T_k_dense = 1, num_heads, max(m_block, 8), max(n_block, 16)
    H_kv = num_kv_heads
    device = "cuda"
    q = torch.empty(B, H, T_q, head_dim, dtype=torch_dtype, device=device)
    o = torch.empty(B, H, T_q, head_dim, dtype=torch_dtype, device=device)

    if paged:
        # Paged K/V is a per-layer pool view: (num_blocks, block_size, H_kv, D).
        # T_k_max is the logical kv extent; for the bias/length-mask shape we
        # use ``max_blocks_per_seq * block_size``.
        max_blocks_per_seq = max(n_block // block_size, 1)
        num_blocks = max(max_blocks_per_seq, 2)
        k = torch.empty(num_blocks, block_size, H_kv, head_dim, dtype=torch_dtype, device=device)
        v = torch.empty(num_blocks, block_size, H_kv, head_dim, dtype=torch_dtype, device=device)
        T_k_logical = max_blocks_per_seq * block_size
    else:
        k = torch.empty(B, H_kv, T_k_dense, head_dim, dtype=torch_dtype, device=device)
        v = torch.empty(B, H_kv, T_k_dense, head_dim, dtype=torch_dtype, device=device)
        T_k_logical = T_k_dense

    # The compiler needs a 16-byte head-dimension alignment guarantee for
    # asynchronous copies. TVM-FFI mode also keeps graph-replayed tensor
    # descriptors alive instead of creating per-call DLPack wrappers.
    elem_bits = 16 if dtype_str == "float16" else 16  # bf16 is also 16b
    align_div = 128 // elem_bits

    def _wrap(t: torch.Tensor) -> "cute.Tensor":
        return (
            from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)
            .mark_layout_dynamic(leading_dim=t.dim() - 1)
            .mark_compact_shape_dynamic(
                mode=t.dim() - 1,
                stride_order=t.dim_order(),
                divisibility=align_div,
            )
        )

    mQ = _wrap(q)
    mK = _wrap(k)
    mV = _wrap(v)
    mO = _wrap(o)
    if has_bias:
        bias = torch.empty(B, H, T_q, T_k_logical, dtype=torch_dtype, device=device)
        mBias = _wrap(bias)
    else:
        # Zero-rank dummy: cute.rank(mBias) > 0 in the kernel == False.
        mBias = from_dlpack(
            torch.empty((), dtype=torch_dtype, device=device),
            assumed_align=16,
            enable_tvm_ffi=True,
        )

    seqlens = torch.zeros(B, dtype=torch.int32, device=device)
    mCacheSeqlens = from_dlpack(seqlens, assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=0
    )

    if has_seqstart:
        seqstarts = torch.zeros(B, dtype=torch.int32, device=device)
        mCacheSeqStarts = from_dlpack(
            seqstarts, assumed_align=4, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=0)
    else:
        # Zero-rank dummy; the mask predicate that reads it is compiled out.
        mCacheSeqStarts = from_dlpack(
            torch.empty((), dtype=torch.int32, device=device),
            assumed_align=4,
            enable_tvm_ffi=True,
        )

    if paged:
        block_table = torch.zeros(
            B,
            max(n_block // block_size, 1),
            dtype=torch.int32,
            device=device,
        )
        mBlockTable = from_dlpack(
            block_table, assumed_align=4, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=block_table.dim() - 1)
    else:
        # Zero-rank dummy when paged=False; kernel never reads it.
        mBlockTable = from_dlpack(
            torch.empty((), dtype=torch.int32, device=device),
            assumed_align=4,
            enable_tvm_ffi=True,
        )

    import cuda.bindings.driver as cuda_driver

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    softmax_scale = cutlass.Float32(1.0 / (head_dim**0.5))
    return cute.compile(
        inst,
        mQ,
        mK,
        mV,
        mO,
        mBias,
        mCacheSeqlens,
        mCacheSeqStarts,
        mBlockTable,
        softmax_scale,
        stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def fmha_config_supported(
    *,
    head_dim: int,
    dtype_str: str,
    has_bias: bool = False,
    paged: bool = False,
    block_size: int = 0,
    m_block: int = 64,
    n_block: int = 64,
    num_threads: int = 128,
    bias_aligned: bool = False,
    causal: bool = False,
) -> bool:
    """Would :func:`get_compiled_fmha` accept this configuration?

    ``get_compiled_fmha`` *raises* on a shape the arch class cannot implement
    — head_dim 128 on sm_120, for instance — which is the right contract for a
    caller that asked for the kernel by name (``OASR_ATTN_BACKEND=cute`` means
    "require it").  A caller that is merely *choosing* a backend needs to ask
    first, so :class:`oasr.layers.Attention` uses this and quietly stays on
    SDPA for shapes the kernel does not cover.  Answered from the arch class's
    own ``can_implement``, so it cannot go stale as the kernel gains shapes.
    """
    cap = _capability_probe()[0]
    if cap is None or select_backend() != "cute":
        return False
    try:
        import cutlass

        from oasr.kernels.cute.attention.base import pick_arch_cls

        # CuteDSL ships no stubs, so its dtype singletons are invisible to mypy.
        f16 = cutlass.Float16  # type: ignore[attr-defined]
        bf16 = cutlass.BFloat16  # type: ignore[attr-defined]
        cute_dtype = f16 if dtype_str == "float16" else bf16
        return bool(
            pick_arch_cls(*cap).can_implement(
                dtype=cute_dtype,
                head_dim=head_dim,
                m_block_size=m_block,
                n_block_size=n_block,
                num_threads=num_threads,
                has_bias=has_bias,
                paged=paged,
                block_size=block_size,
                bias_aligned=bias_aligned,
                causal=causal,
            )
        )
    except Exception:  # CuteDSL missing / probe failed -> SDPA is the answer
        return False


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
    bias_aligned: bool = False,
    causal: bool = False,
    has_seqstart: bool = False,
):
    """Public accessor — returns a compiled CuteDSL callable, compiling on first call."""
    cap = _capability_probe()[0]
    if cap is None:
        raise RuntimeError("no CUDA device available")
    return _compiled_fmha(
        cap,
        head_dim,
        dtype_str,
        num_heads,
        num_kv_heads,
        has_bias,
        paged,
        block_size,
        m_block,
        n_block,
        num_threads,
        bias_aligned,
        causal,
        has_seqstart,
    )


# ---------------------------------------------------------------------------
# Variable-length (sequence-packed) compile cache
# ---------------------------------------------------------------------------


@functools.cache
def _compiled_fmha_varlen(
    arch: Tuple[int, int],
    head_dim: int,
    dtype_str: str,
    num_heads: int,
    num_kv_heads: int,
    has_bias: bool,
    m_block: int,
    n_block: int,
    num_threads: int,
    bias_aligned: bool = False,
):
    """Compile ``FmhaSm80.forward_varlen`` for packed ``(total, H, D)`` inputs.

    Dummy descriptors signal rank/dtype/dynamic-dim only.  The varlen kernel
    consumes packed q/k/v + ``cu_seqlens_q/k`` (+ a packed block-diagonal bias
    and its ``bias_offsets`` when ``has_bias``) and writes a packed output.
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
        raise ValueError(f"unsupported dtype {dtype_str!r}")

    from oasr.kernels.cute.attention.base import pick_arch_cls

    cls = pick_arch_cls(*arch)
    if not cls.can_implement(
        dtype=cute_dtype,
        head_dim=head_dim,
        m_block_size=m_block,
        n_block_size=n_block,
        num_threads=num_threads,
        has_bias=has_bias,
        paged=False,
        varlen=True,
        bias_aligned=bias_aligned,
    ):
        raise RuntimeError("FmhaSm80.can_implement(varlen=True) returned False")
    inst = cls(
        head_dim=head_dim,
        dtype=cute_dtype,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        has_bias=has_bias,
        paged=False,
        varlen=True,
        m_block_size=m_block,
        n_block_size=n_block,
        num_threads=num_threads,
        bias_aligned=bias_aligned,
    )

    device = "cuda"
    H, H_kv = num_heads, num_kv_heads
    S = 2
    total_q = max(m_block, 8) * S
    total_k = max(n_block, 16) * S
    q = torch.empty(total_q, H, head_dim, dtype=torch_dtype, device=device)
    o = torch.empty(total_q, H, head_dim, dtype=torch_dtype, device=device)
    k = torch.empty(total_k, H_kv, head_dim, dtype=torch_dtype, device=device)
    v = torch.empty(total_k, H_kv, head_dim, dtype=torch_dtype, device=device)

    elem_bits = 16
    align_div = 128 // elem_bits

    def _wrap(t: torch.Tensor) -> "cute.Tensor":
        return (
            from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)
            .mark_layout_dynamic(leading_dim=t.dim() - 1)
            .mark_compact_shape_dynamic(
                mode=t.dim() - 1,
                stride_order=t.dim_order(),
                divisibility=align_div,
            )
        )

    mQ, mK, mV, mO = _wrap(q), _wrap(k), _wrap(v), _wrap(o)

    if has_bias:
        bias = torch.empty(H * total_q * total_k, dtype=torch_dtype, device=device)
        mBias = from_dlpack(bias, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=0
        )
        bias_off = torch.zeros(S + 1, dtype=torch.int32, device=device)
        mBiasOff = from_dlpack(bias_off, assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=0
        )
    else:
        mBias = from_dlpack(
            torch.empty((), dtype=torch_dtype, device=device),
            assumed_align=16,
            enable_tvm_ffi=True,
        )
        mBiasOff = from_dlpack(
            torch.empty((), dtype=torch.int32, device=device),
            assumed_align=4,
            enable_tvm_ffi=True,
        )

    cu_q = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(S + 1, dtype=torch.int32, device=device)
    mCuQ = from_dlpack(cu_q, assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=0
    )
    mCuK = from_dlpack(cu_k, assumed_align=4, enable_tvm_ffi=True).mark_layout_dynamic(
        leading_dim=0
    )

    import cuda.bindings.driver as cuda_driver

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    softmax_scale = cutlass.Float32(1.0 / (head_dim**0.5))
    max_seqlen_q = cutlass.Int32(max(m_block, 8))

    return cute.compile(
        inst.forward_varlen,
        mQ,
        mK,
        mV,
        mO,
        mBias,
        mBiasOff,
        mCuQ,
        mCuK,
        max_seqlen_q,
        softmax_scale,
        stream,
        options="--enable-tvm-ffi",
    )


def get_compiled_fmha_varlen(
    *,
    head_dim: int,
    dtype_str: str,
    num_heads: int,
    num_kv_heads: int,
    has_bias: bool,
    m_block: int = 64,
    n_block: int = 64,
    num_threads: int = 128,
    bias_aligned: bool = False,
):
    """Public accessor for the compiled varlen kernel."""
    cap = _capability_probe()[0]
    if cap is None:
        raise RuntimeError("no CUDA device available")
    return _compiled_fmha_varlen(
        cap,
        head_dim,
        dtype_str,
        num_heads,
        num_kv_heads,
        has_bias,
        m_block,
        n_block,
        num_threads,
        bias_aligned,
    )


# ---------------------------------------------------------------------------
# Warmup helper
# ---------------------------------------------------------------------------


def warmup_fmha(
    *,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    max_batch_size: int,  # noqa: ARG001 -- reserved for future use
    chunk_size: int,  # noqa: ARG001
    max_attention_key_size: int,  # noqa: ARG001
    device: Any,  # noqa: ARG001
    dtype: Any,
) -> None:
    """Eagerly populate the compile cache for a given Conformer config.

    Skips silently on archs other than SM120 or when the active backend is
    ``sdpa``.
    """
    if select_backend() != "cute":
        return
    import torch

    dtype_str = (
        "float16" if dtype == torch.float16 else ("bfloat16" if dtype == torch.bfloat16 else None)
    )
    if dtype_str is None:
        return
    for has_bias in (False, True):
        try:
            get_compiled_fmha(
                head_dim=head_dim,
                dtype_str=dtype_str,
                num_heads=n_head,
                num_kv_heads=n_kv_head,
                has_bias=has_bias,
                paged=False,
            )
        except Exception as exc:
            logger.warning("warmup_fmha (has_bias=%s) failed: %s", has_bias, exc)
