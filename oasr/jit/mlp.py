# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT dispatch + compile cache for the OASR fused MLP kernels.

This module is the bridge between the public functional API (``oasr.gated_mlp``)
and the CuTeDSL backends under ``oasr.kernels.cute.mlp``.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional, Tuple, cast

logger = logging.getLogger("oasr.jit.mlp")

# ---------------------------------------------------------------------------
# Backend mode
# ---------------------------------------------------------------------------

_GATED_MLP_ENV = "OASR_GATED_MLP_CUTE"
_VALID_MODES = ("auto", "always", "off")


def _read_gated_mlp_mode() -> str:
    raw = os.environ.get(_GATED_MLP_ENV, "auto").lower()
    if raw in ("1", "always", "on"):
        return "always"
    if raw in ("0", "off", "never"):
        return "off"
    if raw != "auto":
        logger.warning("%s=%r is not recognised; using 'auto'.", _GATED_MLP_ENV, raw)
    return "auto"


_GATED_MLP_MODE = _read_gated_mlp_mode()


def get_gated_mlp_mode() -> str:
    """Return the active gate (``auto`` / ``always`` / ``off``)."""
    return _GATED_MLP_MODE


def set_gated_mlp_mode(mode: str) -> None:
    """Override the gate for the rest of the process.  Used by tests and A/Bs."""
    global _GATED_MLP_MODE
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; valid: {_VALID_MODES}")
    _GATED_MLP_MODE = mode
    _compiled_gated_mlp.cache_clear()
    _capability_probe.cache_clear()
    _machine.cache_clear()
    # ``routed_gated_mlp`` memoises the whole decision, gate included, so a mode
    # change that did not clear it would keep serving the old routing -- which is
    # exactly what an A/B or a rollback switch is for.
    _ROUTE.clear()


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------

#: Architectures whose CuTeDSL warp-level ``mma.sync`` composition is validated.
#: SM90 and SM100 would want wgmma / tcgen05 mainloops of their own and are not
#: covered by this one -- declared, not silently routed onto an Ampere path.
_SUPPORTED_SM = (80, 86, 89, 120)


@functools.cache
def _capability_probe() -> Optional[Tuple[int, int]]:
    """Compute capability if the CuTeDSL gated MLP is usable here, else ``None``.

    Unlike :mod:`oasr.jit.attention` this is not resolved eagerly at import: the
    steady-state hot path is a :data:`_ROUTE` dict lookup that never reaches the
    probe, so the only thing an import-time probe would buy is pulling CuTeDSL
    into every ``import oasr``.
    """
    if _GATED_MLP_MODE == "off":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        cap = torch.cuda.get_device_capability()
    except Exception:
        return None
    if cap[0] * 10 + cap[1] not in _SUPPORTED_SM:
        return None
    try:
        import cutlass  # noqa: F401

        from oasr.kernels.cute.mlp import GatedMlpCute  # noqa: F401
    except Exception as exc:
        logger.warning("CuTeDSL gated MLP unavailable (%s); leaving it out.", exc)
        return None
    return cap


# ---------------------------------------------------------------------------
# Tile selection and the measured band
# ---------------------------------------------------------------------------

#: The 128-bit vector contract.  The epilogue stores ``out`` and the mainloop
#: loads ``x`` / both weights in 128-bit pieces, so the two contiguous extents
#: have to be 8-element multiples.  Same number, same reason, as
#: ``oasr.layers._backend.GEMM_ALIGNMENT``.
ALIGNMENT = 8

#: Candidate tiles per ``m_block``, **in preference order**, as
#: ``(m, n, k, stages, threads, warps_n)``.  Two entries each, and the
#: difference between them is not the tile but the *ring*: the first is a deep
#: 4-stage ring at 64-wide K that fills shared memory and gets one CTA per SM;
#: the second is a 32-wide K tile whose ring is half the size and therefore fits
#: **two**.  :func:`select_gated_mlp_tile` chooses between them by wave
#: arithmetic, and which one wins is decided entirely by ``N`` -- see there.
#:
#: ``n_block`` is 64 throughout because at every LLM width in scope the N axis
#: already supplies more CTAs than the machine has SMs, so a wider N tile only
#: deepens the ring and halves the grid.  ``m_block`` above 64 is absent on
#: purpose; see :data:`_BAND_MAX_ROWS`.
_CANDIDATES: tuple = (
    (16, ((16, 64, 64, 4, 128, 4), (16, 64, 32, 4, 64, 2))),
    (32, ((32, 64, 64, 4, 128, 4), (32, 64, 32, 4, 64, 2))),
    (64, ((64, 64, 64, 4, 256, 4), (64, 64, 32, 4, 128, 4))),
)

#: Inclusive row band the fused kernel owns: **one m-tile**.
#:
#: That is the whole rule, and it is mechanical rather than fitted.  With a
#: single m-tile every weight element is read from DRAM exactly once, which is
#: the bandwidth argument the fusion rests on.  With two, each weight tile is
#: loaded by two CTAs, the kernel becomes an ordinary GEMM reading its operands
#: twice, and it is competing with cuBLAS on cuBLAS's own terms -- which it
#: loses, because the ring carries A *and both* Bs and cannot afford the tiles a
#: library GEMM picks.
_BAND_MAX_ROWS = 64

#: Hardware ceiling on resident blocks per SM; the tiles here never approach it,
#: but leaving it out would let a hypothetical tiny tile claim absurd occupancy.
_MAX_BLOCKS_PER_SM = 24


@functools.cache
def _machine() -> Tuple[int, int, int]:
    """``(SMs, opt-in smem per block, max threads per SM)`` for device 0."""
    import torch

    from oasr.kernels.cute.mlp.gated import smem_capacity

    props = torch.cuda.get_device_properties(0)
    return (
        props.multi_processor_count,
        smem_capacity(),
        getattr(props, "max_threads_per_multi_processor", 1536),
    )


def gated_mlp_ctas_per_sm(tile: Tuple[int, int, int, int, int, int]) -> int:
    """How many of these CTAs are resident at once, by shared memory and warps.

    Registers are deliberately not modelled: every tile here measured 64
    registers per thread, which binds at 8 blocks -- far above the 1-2 that
    shared memory allows.  A tile that changed that would show up as a
    *measured* regression, not as a wrong number here.
    """
    m_block, n_block, k_block, num_stages, num_threads, _ = tile
    sms, smem, max_threads = _machine()
    del sms
    import cutlass

    from oasr.kernels.cute.mlp import GatedMlpCute

    bytes_per_cta = GatedMlpCute.smem_bytes(
        dtype=cutlass.Float16,  # type: ignore[attr-defined]
        m_block=m_block,
        n_block=n_block,
        k_block=k_block,
        num_stages=num_stages,
    )
    by_smem = smem // bytes_per_cta if bytes_per_cta else _MAX_BLOCKS_PER_SM
    by_threads = max_threads // num_threads
    return max(1, min(by_smem, by_threads, _MAX_BLOCKS_PER_SM))


def _waves(tile: Tuple[int, int, int, int, int, int], rows: int, n: int) -> int:
    sms = _machine()[0]
    slots = sms * gated_mlp_ctas_per_sm(tile)
    grid = -(-n // tile[1]) * -(-rows // tile[0])
    return -(-grid // slots)


def select_gated_mlp_tile(rows: int, n: int) -> Optional[Tuple[int, ...]]:
    """Tuned tile for this problem: fewest waves, ties broken by the ranking.

    Why ``N`` is in the decision, and not only ``M``
    ------------------------------------------------
    The kernel is bandwidth bound, so the thing that decides its time is whether
    the *last* wave still has enough CTAs in flight to saturate DRAM.
    """
    if rows <= 0 or n <= 0:
        return None
    candidates = _CANDIDATES[-1][1]
    for m_max, tiles in _CANDIDATES:
        if rows <= m_max:
            candidates = tiles
            break
    best_key = None
    best_tile = None
    for rank, tile in enumerate(candidates):
        key = (_waves(tile, rows, n), rank)
        if best_key is None or key < best_key:
            best_key, best_tile = key, tile
    return cast(Tuple[int, ...], best_tile)


def gated_mlp_shape_supported(*, rows: int, n: int, k: int, k_block: int) -> bool:
    """Does this problem meet the kernel's static contract?

    ``K`` has to be a whole number of K tiles: the mainloop iterates
    ``ceil_div(K, k_block)`` times and predicates only the *row* axis, so a
    partial K tile would read the next row of ``x`` (silently wrong) or past the
    tensor (a fault).  ``N`` and ``K`` also carry the 128-bit vector contract.
    """
    return rows > 0 and n % ALIGNMENT == 0 and k % ALIGNMENT == 0 and k % k_block == 0


# ---------------------------------------------------------------------------
# Compile cache
# ---------------------------------------------------------------------------


@functools.cache
def _compiled_gated_mlp(
    arch: Tuple[int, int],
    dtype_str: str,  # "float16" or "bfloat16"
    activation: str,
    has_bias: bool,
    tile: Tuple[int, int, int, int, int, int],
):
    """Compile one configuration.  Shapes stay dynamic, so M/N/K are not in the key."""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from oasr.kernels.cute.mlp import GatedMlpCute

    # CuTeDSL ships no type stubs, so its dtype singletons are invisible to mypy.
    if dtype_str == "float16":
        cute_dtype = cutlass.Float16  # type: ignore[attr-defined]
        torch_dtype = torch.float16
    elif dtype_str == "bfloat16":
        cute_dtype = cutlass.BFloat16  # type: ignore[attr-defined]
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"unsupported dtype {dtype_str!r} (need float16 or bfloat16)")

    m_block, n_block, k_block, num_stages, num_threads, warps_n = tile
    kwargs = {
        "dtype": cute_dtype,
        "activation": activation,
        "has_bias": has_bias,
        "m_block": m_block,
        "n_block": n_block,
        "k_block": k_block,
        "num_stages": num_stages,
        "num_threads": num_threads,
        "warps_n": warps_n,
    }
    if not GatedMlpCute.can_implement(**kwargs):
        raise RuntimeError(f"GatedMlpCute cannot implement tile {tile}")
    inst = GatedMlpCute(**kwargs)

    def _wrap(t: torch.Tensor):
        # divisibility=8 is the 128-bit cp.async / store guarantee.  Without it
        # the compiler only knows the leading dim is dynamic and refuses the atom.
        return (
            from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)
            .mark_layout_dynamic(leading_dim=t.dim() - 1)
            .mark_compact_shape_dynamic(
                mode=t.dim() - 1, stride_order=t.dim_order(), divisibility=ALIGNMENT
            )
        )

    # Descriptors only: rank, dtype and which dims are dynamic.  Values unused.
    def empty(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, device="cuda", dtype=torch_dtype)

    args = (
        _wrap(empty(m_block, k_block)),
        _wrap(empty(n_block, k_block)),
        _wrap(empty(n_block, k_block)),
        _wrap(empty(ALIGNMENT)),
        _wrap(empty(ALIGNMENT)),
        _wrap(empty(m_block, n_block)),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    # --enable-tvm-ffi is not an optimisation detail: without it every call
    # rebuilds six DLPack descriptors, which measured ~148 us on the recurrent
    # step and is also the only call pattern that is safe to capture into a
    # CUDA graph and replay (see oasr/functionals/attention.py::_call_cute_dsl).
    return cute.compile(inst, *args, stream, options="--enable-tvm-ffi")


def gated_mlp_config_supported(*, rows: int, n: int, k: int) -> bool:
    """Would :func:`get_compiled_gated_mlp` accept this problem?

    Capability only -- arch, CuTeDSL, a tile for these bounds, and the static
    shape contract.  It says nothing about whether fusing is *faster* here; that
    is :func:`should_use_gated_mlp`.  The same split as
    :func:`oasr.jit.attention.fmha_config_supported`, and for the same reason: a
    caller that is choosing between two working paths has to be able to ask the
    capability question without also asking the policy one.
    """
    if _capability_probe() is None:
        return False
    tile = select_gated_mlp_tile(rows, n)
    if tile is None:
        return False
    return gated_mlp_shape_supported(rows=rows, n=n, k=k, k_block=tile[2])


def get_compiled_gated_mlp(*, dtype_str: str, activation: str, has_bias: bool, rows: int, n: int):
    """Public accessor — the compiled callable for this problem's tuned tile.

    Raises rather than declining, which is the right contract for a caller that
    asked for the kernel by name; :func:`routed_gated_mlp` is the one that
    chooses.  Compiles on first use.
    """
    cap = _capability_probe()
    if cap is None:
        raise RuntimeError("the CuTeDSL gated MLP is not available on this device")
    tile = select_gated_mlp_tile(rows, n)
    if tile is None:
        raise RuntimeError(f"no tuned tile for rows={rows} n={n}")
    return _compiled_gated_mlp(
        cap,
        dtype_str,
        activation,
        has_bias,
        cast(Tuple[int, int, int, int, int, int], tile),
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

#: ``(dtype_str, activation, has_bias, rows, n, k)`` -> the compiled kernel, or
#: ``None`` when this shape is outside the band or the kernel is unavailable.
#: Populated by :func:`routed_gated_mlp`; cleared by :func:`set_gated_mlp_mode`.
_ROUTE: dict = {}


def should_use_gated_mlp(*, rows: int, n: int, k: int) -> bool:
    """Is this shape inside the band where the fused kernel measured fastest?"""
    if not gated_mlp_config_supported(rows=rows, n=n, k=k):
        return False
    return _GATED_MLP_MODE == "always" or rows <= _BAND_MAX_ROWS


def routed_gated_mlp(*, dtype_str: str, activation: str, has_bias: bool, rows: int, n: int, k: int):
    """The kernel to run for this shape, or ``None`` to leave it on the GEMM path.

    ``should_use_gated_mlp()`` followed by ``get_compiled_gated_mlp()`` is the
    readable spelling, and both are pure functions of the shape, so the whole
    decision memoises to one dict lookup -- which matters because a 28-layer
    decoder asks 28 times per step.  Deciding *and* compiling under the same key
    also means a shape whose tile fails to build is remembered as declined rather
    than retried per step; a compile failure is a property of the configuration,
    not of the call.
    """
    key = (dtype_str, activation, has_bias, rows, n, k)
    try:
        return _ROUTE[key]
    except KeyError:
        pass
    fn = None
    if should_use_gated_mlp(rows=rows, n=n, k=k):
        try:
            fn = get_compiled_gated_mlp(
                dtype_str=dtype_str, activation=activation, has_bias=has_bias, rows=rows, n=n
            )
        except Exception as exc:  # unsupported arch, missing CuTeDSL, unbuildable tile
            logger.warning("CuTeDSL gated MLP declined for rows=%d n=%d k=%d: %s", rows, n, k, exc)
            fn = None
    _ROUTE[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Warmup helper
# ---------------------------------------------------------------------------


def warmup_gated_mlp(
    *, dtype_str: str, activation: str, has_bias: bool, rows: int, n: int, k: int
) -> None:
    """Populate the route + compile cache ahead of the first call.  Never raises."""
    routed_gated_mlp(
        dtype_str=dtype_str, activation=activation, has_bias=has_bias, rows=rows, n=n, k=k
    )
