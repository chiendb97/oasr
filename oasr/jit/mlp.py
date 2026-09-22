# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT dispatch, routing and compile cache for the OASR fused MLP kernels.

This module is the bridge between the public functional API (``oasr.gated_mlp``)
and the two kernel lanes that implement it:

``cute``
    the CuTeDSL kernel under :mod:`oasr.kernels.cute.mlp`, compiled here
    through ``cutlass.cute.compile()``.
``cxx``
    the C++ CUTLASS/CuTe kernel under ``include/oasr/mlp/``, compiled by
    :mod:`oasr.jit.gated_mlp` through the ordinary ninja pipeline.

This module is the **arbiter**; :mod:`oasr.jit.gated_mlp` is one of the answers
and never imports it back.  Two gates, deliberately separate, because they
answer different questions:

``OASR_GATED_MLP_CUTE``    -- *fuse or not*: ``auto`` / ``1`` / ``0``.
``OASR_GATED_MLP_BACKEND`` -- *which lane*: ``auto`` / ``cute`` / ``cxx``.

Collapsing them would make "roll the fusion back" and "A/B the two lanes" the
same switch, and they are not: the first is an operator decision about a
shape, the second is a claim about two implementations of the same thing.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional, Tuple, cast

from .measured import Machine, MeasuredOn, note_extrapolation

logger = logging.getLogger("oasr.jit.mlp")

# ---------------------------------------------------------------------------
# Backend mode
# ---------------------------------------------------------------------------

_GATED_MLP_ENV = "OASR_GATED_MLP_CUTE"
_VALID_MODES = ("auto", "always", "off")

_BACKEND_ENV = "OASR_GATED_MLP_BACKEND"
#: ``cute`` keeps meaning the CuTeDSL kernel, as it does for attention: it is
#: the spelling in every recorded A/B command, so repointing it would silently
#: change what those measurements measured.  The C++ CUTLASS/CuTe lane is
#: ``cxx``.  Same names and same aliases as ``OASR_ATTN_BACKEND``.
_VALID_BACKENDS = ("auto", "cute", "cxx")
_BACKEND_ALIASES = {"cutedsl": "cute", "cpp": "cxx", "cutlass": "cxx"}


def _read_gated_mlp_mode() -> str:
    raw = os.environ.get(_GATED_MLP_ENV, "auto").lower()
    if raw in ("1", "always", "on"):
        return "always"
    if raw in ("0", "off", "never"):
        return "off"
    if raw != "auto":
        logger.warning("%s=%r is not recognised; using 'auto'.", _GATED_MLP_ENV, raw)
    return "auto"


def _read_gated_mlp_backend() -> str:
    raw = os.environ.get(_BACKEND_ENV, "auto").lower()
    raw = _BACKEND_ALIASES.get(raw, raw)
    if raw not in _VALID_BACKENDS:
        logger.warning(
            "%s=%r is not recognised; valid choices are %s. Using 'auto'.",
            _BACKEND_ENV,
            raw,
            _VALID_BACKENDS,
        )
        return "auto"
    return raw


_GATED_MLP_MODE = _read_gated_mlp_mode()
_GATED_MLP_BACKEND = _read_gated_mlp_backend()


def get_gated_mlp_mode() -> str:
    """Return the active gate (``auto`` / ``always`` / ``off``)."""
    return _GATED_MLP_MODE


def set_gated_mlp_mode(mode: str) -> None:
    """Override the gate for the rest of the process.  Used by tests and A/Bs."""
    global _GATED_MLP_MODE
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; valid: {_VALID_MODES}")
    _GATED_MLP_MODE = mode
    _clear_all_caches()


def get_gated_mlp_backend() -> str:
    """Return the lane preference (``auto`` / ``cute`` / ``cxx``)."""
    return _GATED_MLP_BACKEND


def set_gated_mlp_backend(backend: str) -> None:
    """Override the lane preference for the rest of the process.  Tests and A/Bs.

    Note this clears both compile caches, so flipping in a loop recompiles.
    For an in-process A/B -- a parametrised fixture, a differential test, an
    interleaved benchmark arm -- ask :func:`routed_gated_mlp` for the lane you
    want by name and leave the global alone.
    """
    global _GATED_MLP_BACKEND
    backend = _BACKEND_ALIASES.get(backend, backend)
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"invalid backend {backend!r}; valid: {_VALID_BACKENDS}")
    _GATED_MLP_BACKEND = backend
    _clear_all_caches()


def _clear_all_caches() -> None:
    _compiled_gated_mlp.cache_clear()
    _capability_probe.cache_clear()
    _machine.cache_clear()
    _cxx_probe.cache_clear()
    try:
        from oasr.jit import gated_mlp as _cxx

        _cxx.clear_caches()
    except ImportError:  # the C++ lane is optional; the arbiter must still import
        pass
    # ``routed_gated_mlp`` memoises the whole decision, gate and lane included,
    # so a change that did not clear it would keep serving the old routing --
    # which is exactly what an A/B or a rollback switch is for.
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

#: Where the candidate list came from — *not* how one is chosen from it.
#:
#: The choice is derived: :func:`select_gated_mlp_tile` scores candidates by wave
#: count against this machine's ``multi_processor_count`` and real opt-in shared
#: memory, so it follows the card it runs on.  That is the half of this module
#: that travels, and it was written that way because a rows-keyed table tuned at
#: N=18944 picked a one-CTA-per-SM ring that left 172 CTAs on 170 SMs — one wave
#: plus a tail of two — and read 0.987x end to end.
#:
#: What does not travel is which six tiles are in the list at all, and the rank
#: used to break a wave-count tie.  Both were chosen on one card.  A tile that
#: would win only on a machine with a different smem budget or SM count is not in
#: the list to be scored, and no amount of wave arithmetic can find it.
_MEASURED = MeasuredOn(
    table="jit.mlp._CANDIDATES",
    machine=Machine(name="NVIDIA GeForce RTX 5090", sm=120, sms=170),
    bandwidth="1792 GB/s",
    source=".artifacts/ (2026-08-25)",
    moves_with="shared-memory budget and SM count — they decide which rings fit "
    "and how many CTAs are resident, so a different card may want a tile this "
    "list does not contain. The choice *among* these is already derived",
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
#:
#: Derived from the candidate list rather than written down next to it.  The rule
#: is "one m-tile", so the band *is* the largest m-tile that exists — they were
#: two numbers that happened to agree, and agreeing is not the same as being
#: linked (audit A8).  A 128-row candidate added for its own sake would have
#: widened what the kernel can do and left the band at 64, quietly declining the
#: very shapes it was added for; now widening the list widens the band, which is
#: the only relationship the argument above supports.
_BAND_MAX_ROWS = max(m_max for m_max, _ in _CANDIDATES)

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

    The scoring reads this machine, so it travels; the *list* it scores was
    measured on one card, so it does not.  :data:`_MEASURED` records which, and
    off that card the fact is counted rather than assumed away (audit A8).
    """
    if rows <= 0 or n <= 0:
        return None
    note_extrapolation(_MEASURED)
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


def cute_config_supported(*, rows: int, n: int, k: int) -> bool:
    """Would the **CuTeDSL** lane accept this problem?

    Capability only -- arch, CuTeDSL, a tile for these bounds, and the static
    shape contract.  It says nothing about whether fusing is *faster* here;
    that is :func:`should_use_gated_mlp`.  The same split as
    :func:`oasr.jit.attention.fmha_config_supported`, and for the same reason:
    a caller that is choosing between two working paths has to be able to ask
    the capability question without also asking the policy one.
    """
    if _capability_probe() is None:
        return False
    tile = select_gated_mlp_tile(rows, n)
    if tile is None:
        return False
    return gated_mlp_shape_supported(rows=rows, n=n, k=k, k_block=tile[2])


@functools.cache
def _cxx_probe() -> Optional[int]:
    """The target SM if the C++ lane is usable here, else ``None``.

    Not resolved at import, for the same reason the CuTeDSL probe is not: the
    steady-state hot path is a :data:`_ROUTE` dict lookup that never reaches
    either.
    """
    if _GATED_MLP_MODE == "off" or _GATED_MLP_BACKEND == "cute":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        from oasr.jit.core import _get_target_sm
        from oasr.jit.gated_mlp import SUPPORTED_SM
    except Exception:
        return None
    try:
        sm = _get_target_sm()
    except Exception:  # an architecture OASR compiles for nothing at all
        return None
    return sm if sm in SUPPORTED_SM else None


def cxx_config_supported(*, dtype_str: str, activation: str, rows: int, n: int, k: int) -> bool:
    """Would the **C++** lane accept this problem?

    Its shape contract is strictly wider than the CuTeDSL lane's: ``K`` need
    not be a whole number of K tiles, because the residue is predicated and the
    ZFILL cp.async makes the skipped elements zero.
    """
    sm = _cxx_probe()
    if sm is None:
        return False
    from oasr.jit import gated_mlp as _cxx

    return _cxx.config_supported(
        sm=sm, dtype_str=dtype_str, activation=activation, rows=rows, n=n, k=k
    )


def gated_mlp_config_supported(
    *, rows: int, n: int, k: int, dtype_str: str = "float16", activation: str = "silu"
) -> bool:
    """Can **any lane this process is willing to use** serve this problem?

    Asks the lanes in :func:`_lane_order`, so a pinned
    ``OASR_GATED_MLP_BACKEND`` narrows the answer -- otherwise a shape only the
    unpinned lane can serve would read as available here and then be declined
    by :func:`routed_gated_mlp`, and the two would disagree.

    The default arguments keep the pre-two-lane call signature working: every
    caller that only knew about the CuTeDSL lane was implicitly asking about a
    configuration both lanes support.
    """
    for backend in _lane_order():
        if backend == "cxx":
            if cxx_config_supported(
                dtype_str=dtype_str, activation=activation, rows=rows, n=n, k=k
            ):
                return True
        elif cute_config_supported(rows=rows, n=n, k=k):
            return True
    return False


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


def get_cxx_gated_mlp(*, dtype_str: str, activation: str, has_bias: bool, rows: int, n: int):
    """Public accessor -- the C++ launcher for this problem's tuned tile.

    Raises rather than declining, for the same reason
    :func:`get_compiled_gated_mlp` does.  Builds its cell on first use.
    """
    sm = _cxx_probe()
    if sm is None:
        raise RuntimeError("the C++ gated MLP lane is not available on this device")
    from oasr.jit import gated_mlp as _cxx

    tile = _cxx.routed_tile(rows=rows, n=n, sm=sm)
    if tile < 0:
        raise RuntimeError(f"no tile for rows={rows} n={n} on sm_{sm}")
    return _cxx.get_gated_mlp_fn(
        dtype_str=dtype_str,
        activation=activation,
        has_bias=has_bias,
        tile_index=tile,
        sm=sm,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

#: ``(dtype_str, activation, has_bias, rows, n, k)`` -> ``(backend, callable)``,
#: or ``None`` when this shape is outside the band or no lane is available.
#: Populated by :func:`routed_gated_mlp`; cleared by :func:`set_gated_mlp_mode`
#: and :func:`set_gated_mlp_backend`.
_ROUTE: dict = {}

#: Lane preference under ``auto``, best first.
#:
#: ``cxx`` is first because it measured **1.00x - 1.11x** of the CuTeDSL lane
#: over four interleaved, graph-replayed sweeps of 13 shapes each (fp16 with
#: and without bias, bf16, gelu_tanh) with **no row regressing** -- ahead by
#: most where the weights fit in L2 and the kernel is issue-bound rather than
#: DRAM-bound.  Its shape contract is also strictly wider (no
#: ``K % k_block`` constraint), and it is the lane that does not need CuTeDSL
#: installed at all.  Numbers and protocol:
#: ``.artifacts/gated_mlp_cpp_validation.md``; re-measure before reordering.
_AUTO_ORDER: Tuple[str, ...] = ("cxx", "cute")


def should_use_gated_mlp(
    *, rows: int, n: int, k: int, dtype_str: str = "float16", activation: str = "silu"
) -> bool:
    """Is this shape inside the band where the fused kernel measured fastest?

    The band is one m-tile, and it is the same for both lanes: with a single
    m-tile every weight element is read from DRAM exactly once, which is the
    bandwidth argument the fusion rests on.  With two, each weight tile is
    loaded by two CTAs and the kernel is an ordinary GEMM competing with cuBLAS
    on cuBLAS's own terms.  That argument is about the *tiling*, which the two
    lanes share, so the band does not move with the lane.
    """
    if not gated_mlp_config_supported(
        rows=rows, n=n, k=k, dtype_str=dtype_str, activation=activation
    ):
        return False
    return _GATED_MLP_MODE == "always" or rows <= _BAND_MAX_ROWS


def _lane_order() -> Tuple[str, ...]:
    return _AUTO_ORDER if _GATED_MLP_BACKEND == "auto" else (_GATED_MLP_BACKEND,)


def routed_gated_mlp(*, dtype_str: str, activation: str, has_bias: bool, rows: int, n: int, k: int):
    """``(backend, callable)`` for this shape, or ``None`` to leave it on the GEMM path.

    The two lanes have **different call signatures** -- the C++ one takes its
    output first (``AGENTS.md`` rule 4) and reads the stream from the FFI
    environment, the CuTeDSL one takes the output last and a stream handle --
    so the backend name comes back with the callable rather than being hidden
    behind a wrapper.  A wrapper would be one more Python frame on a path a
    28-layer decoder walks 28 times per step, and the whole point of
    :data:`_ROUTE` is that the steady state is one dict lookup.

    Deciding *and* compiling under the same key also means a shape whose kernel
    fails to build is remembered as declined rather than retried per step; a
    build failure is a property of the configuration, not of the call.
    """
    key = (dtype_str, activation, has_bias, rows, n, k)
    try:
        return _ROUTE[key]
    except KeyError:
        pass
    route = None
    if should_use_gated_mlp(rows=rows, n=n, k=k, dtype_str=dtype_str, activation=activation):
        for backend in _lane_order():
            supported = (
                cxx_config_supported(
                    dtype_str=dtype_str, activation=activation, rows=rows, n=n, k=k
                )
                if backend == "cxx"
                else cute_config_supported(rows=rows, n=n, k=k)
            )
            if not supported:
                continue
            getter = get_cxx_gated_mlp if backend == "cxx" else get_compiled_gated_mlp
            try:
                route = (
                    backend,
                    getter(
                        dtype_str=dtype_str,
                        activation=activation,
                        has_bias=has_bias,
                        rows=rows,
                        n=n,
                    ),
                )
                break
            except Exception as exc:  # unsupported arch, missing toolchain, bad tile
                logger.warning(
                    "the %s gated MLP declined rows=%d n=%d k=%d: %s", backend, rows, n, k, exc
                )
    _ROUTE[key] = route
    return route


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
