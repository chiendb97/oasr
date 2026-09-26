# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Compile cache and routing for the fused recurrent step.

This module is the **arbiter** between the two lanes that implement the fused
step, and the compile cache for one of them:

``cute``
    the CuTeDSL kernel under :mod:`oasr.kernels.cute.recurrent`, compiled here
    through ``cutlass.cute.compile()``.  Like :mod:`oasr.jit.attention`, and
    unlike :mod:`oasr.jit.core`, that is not the Ninja C++ pipeline: CuTeDSL
    kernels are Python and compile to a callable.  One callable is cached per
    configuration and reused.
``cxx``
    the C++ CUTLASS/CuTe kernel under ``include/oasr/recurrent/``, compiled by
    :mod:`oasr.jit.recurrent_step` through the ordinary ninja pipeline.
    :mod:`oasr.jit.recurrent_step` is one of the answers and never imports this
    module back.

Two gates, deliberately separate, because they answer different questions:

``OASR_RECURRENT_CUTE``    -- *fuse or not*: ``auto`` / ``1`` / ``0``.
``OASR_RECURRENT_BACKEND`` -- *which lane*: ``auto`` / ``cute`` / ``cxx``.

Collapsing them would make "roll the fused step back" and "A/B the two lanes"
the same switch, and they are not: the first is an operator decision about a
shape, the second is a claim about two implementations of the same thing.  Same
split, and the same spellings, as ``OASR_GATED_MLP_CUTE`` /
``OASR_GATED_MLP_BACKEND`` and ``OASR_ATTN_BACKEND``.

``OASR_RECURRENT_CUTE`` gates the fusion:

* ``auto`` (**default**) -- take it inside the measured band, leave every other
  shape on whichever path wins there.
* ``1`` / ``always`` -- take it wherever it can implement the shape, which is how
  to A/B the band itself.
* ``0`` / ``off`` -- never; the rollback switch.

The kernel owns most of the range
--------------------------------
Under a 64-step dependent chain in one CUDA graph -- the protocol a recurrence
actually imposes -- it is ahead of both the scalar cohort kernel and cuBLAS plus a
finalizer at every width and batch measured, except the large-batch/large-width
corner.  Gains run 1.11x to 1.70x; see the table on ``_TILES`` below.

Two things it is *not* ahead by, and why:

* At small batch the step is bound by reading the recurrent weight matrix once.
  At B=16, H=640 that is 3.28 MB, and L2 delivers it in about 3.3 us against 3.56
  measured -- 93% of the bandwidth floor.  There is no headroom there to win, only
  headroom to lose, which is what the scalar kernel does.
* At B >= 128 with H >= 1024 a tuned library GEMM keeps a 7-11% mainloop edge.
  Closing it is deep GEMM work (register double-buffering, K-specific swizzle
  phase) with diminishing returns, not a tiling or occupancy fix.

What the layer sees
-------------------
``lstm_gemm_layer`` at T=1 also runs the sequence-wide input projection, which is
identical in both paths and larger than the recurrent step, so a step-level gain
arrives diluted.  Measured at the layer, over the routed band:

    CUDA-graph replay (GPU only)   1.00 - 1.19x
    eager                          0.97 - 1.07x

Graph replay is the number that decides the default: the engine captures the
decoder step (`oasr/engine/decoder_graph.py`), so that is the production path.
Eager is within noise, worst case 3% down at B=16.

Getting here took removing host cost that had nothing to do with the kernel --
per-call DLPack wrappers (~148 us), the stream handle (8.2 us), and three
allocations the fused path never uses (~5 us).  The first revision of this module
defaulted to off because those had not been removed yet and the eager path was
0.82-0.91x.  What remains is not in this module: ``oasr.gemm`` costs 24.2 us of
host per call against ``torch.addmm``'s 10.9 for the same projection.

Most of the eager gap was avoidable and is now gone -- ``torch.cuda.current_stream()``
alone cost 4.1 us per call against a 6 us kernel.  See
:mod:`oasr.jit.cute_runtime`, which the FMHA call sites now share.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional, Tuple, cast

from .measured import Machine, MeasuredOn, note_extrapolation

logger = logging.getLogger("oasr.jit.recurrent_cute")

_ENV = "OASR_RECURRENT_CUTE"

_BACKEND_ENV = "OASR_RECURRENT_BACKEND"
#: ``cute`` keeps meaning the CuTeDSL kernel, as it does for attention and the
#: gated MLP: it is the spelling in every recorded A/B command, so repointing it
#: would silently change what those measurements measured.  The C++
#: CUTLASS/CuTe lane is ``cxx``.
_VALID_BACKENDS = ("auto", "cute", "cxx")
_BACKEND_ALIASES = {"cutedsl": "cute", "cpp": "cxx", "cutlass": "cxx"}

#: Lane preference under ``auto``, best first.
#:
#: ``cxx`` is first because it measured **1.05x - 1.31x** of the CuTeDSL lane,
#: geomean 1.13x, over four interleaved reps of all 32 ``(hidden, batch)``
#: shapes of the table below, under the 64-step graph-replayed dependent chain
#: protocol -- with **no row regressing**.  Both lanes run the same tile on
#: every shape (``kRecurrentStepRoutes`` reproduces ``_TILES``), so that is a
#: comparison of kernels and not of tile choices.  The mechanism is the load
#: section: this lane's cp.async is branch-free and zero-filling where the
#: CuTeDSL one is an ``if`` around the copy, which ncu shows as 19.1 against
#: 22.6 warp-cycles per issued instruction at H=1024 B=128.
#:
#: Its shape contract is also strictly wider -- a hidden width that is not a
#: whole number of K tiles, and arbitrary row strides -- and it is the lane
#: that does not need CuTeDSL installed at all.  Numbers and protocol:
#: ``.artifacts/recurrent_cpp_validation.md``; re-measure before reordering.
_AUTO_ORDER: Tuple[str, ...] = ("cxx", "cute")

#: Architectures whose CuTeDSL warp-level ``mma.sync`` composition is validated.
#: SM90 and SM100 would want wgmma / tcgen05 mainloops of their own and are not
#: covered by this one -- declared, not silently routed onto an Ampere path.
_SUPPORTED_SM = (80, 86, 89, 120)

#: Measured on SM120 (RTX 5090), microseconds per step, LSTM, step only.
#:
#: Protocol: a **64-step dependent chain captured in one CUDA graph**, replayed.
#: That is what a recurrent layer actually does -- step t+1 cannot start until t
#: lands -- and it is the only protocol that compares these three fairly.  A
#: back-to-back loop of *independent* launches lets consecutive kernels overlap
#: and flatters whichever kernel leaves the machine emptiest; timing single calls
#: measures the harness (a Python TVM-FFI call and a C++ launch do not cost the
#: same); and ncu inflates kernels this small with instrumentation.  An earlier
#: revision of this table used per-call loops and concluded the scalar cohort won
#: below B=32.  It does not: that was the harness.
#:
#:   H=256   B      1     8    16    32    64   128   256   512
#:           cohort  2.08  2.78  3.19  4.86  7.73 14.84 27.32 54.02
#:           cuBLAS  2.66  2.75  2.84  3.01  3.04  3.14  3.91  5.47
#:           cute    1.88  1.98  1.99  2.40  2.47  2.63  3.06  4.13
#:   H=640   cohort  3.26  4.06  5.88 12.01 22.55 42.37 81.25 161.7
#:           cuBLAS  4.33  4.52  4.78  4.23  5.25  6.95 11.19 12.17
#:           cute    2.94  3.01  3.11  3.55  4.39  5.53  7.49 12.46
#:   H=1024  cohort  4.04  6.88 10.75 21.10 41.12 79.24 164.8 308.6
#:           cuBLAS  5.71  5.15  6.17  6.82  8.28  8.80 15.56 28.59
#:           cute    4.04  4.26  4.31  5.54  7.24  9.89 16.94 31.72
#:   H=2048  cohort 13.57 19.94 34.52 67.76 131.3 259.1 513.9 1018.6
#:           cuBLAS 13.51 10.83 11.12 12.78 15.82 27.43 52.49 103.9
#:           cute    7.93  7.97  8.89 10.50 17.59 30.30 57.33 111.6
#:
#: ``(hidden_max, batch_max) -> (m, n, k, stages, threads, warps_n)``, scanned in
#: order; the first entry whose bounds both fit wins.  Note how many winners use
#: 512 threads: at large batch the profile said occupancy, not tiling -- 15%
#: achieved, 0.47 waves per SM -- so what helped was *more warps at a fixed
#: tile*, not a bigger tile.  A hand-picked candidate list that stopped at 256
#: threads missed it and left 11-26% on the table at B >= 128.
_TILES: tuple = (
    (256, 128, (32, 32, 64, 4, 128, 2)),
    (256, 256, (32, 64, 64, 3, 256, 4)),
    (256, 1 << 30, (64, 64, 64, 4, 512, 4)),
    (768, 64, (32, 32, 64, 4, 128, 2)),
    (768, 128, (32, 64, 64, 3, 256, 4)),
    (768, 256, (64, 64, 64, 4, 512, 4)),
    (768, 1 << 30, (128, 64, 64, 3, 512, 2)),
    (1536, 16, (32, 32, 64, 4, 128, 2)),
    (1536, 32, (16, 64, 64, 5, 128, 4)),
    (1536, 64, (32, 64, 64, 3, 256, 4)),
    (1536, 128, (64, 64, 64, 3, 512, 4)),
    (1536, 1 << 30, (128, 64, 64, 3, 512, 2)),
    (1 << 30, 8, (16, 64, 64, 4, 128, 4)),
    (1 << 30, 16, (16, 64, 64, 5, 128, 4)),
    (1 << 30, 32, (32, 64, 64, 3, 256, 4)),
    (1 << 30, 64, (64, 64, 64, 3, 512, 4)),
    (1 << 30, 128, (128, 64, 64, 3, 512, 2)),
    (1 << 30, 1 << 30, (128, 128, 64, 3, 512, 4)),
)

#: Inclusive batch band the CuTeDSL step owns, by hidden width, for the LSTM.
#: From the table above: it is ahead everywhere except the large-batch,
#: large-width corner, where a tuned library GEMM keeps a 7-11% mainloop edge.
#:
#:   H<=256   all measured batches      1.11 - 1.43x
#:   H<=768   B <= 256                  1.11 - 1.54x   (B=512 is 0.98x, excluded)
#:   H<=1536  B <= 64                   1.00 - 1.43x   (B>=128 is 0.89-0.92x)
#:   larger   B <= 32                   1.22 - 1.70x   (B>=64  is 0.90-0.93x)
#:
#: The vanilla RNN is deliberately absent.  Its kernel is implemented and
#: validated and its own timings are recorded, but the C++ comparison harness is
#: LSTM-only, so there is no matched single-step reference to route against.
#: Routing it on the LSTM's bands because the shapes rhyme is exactly the guess
#: this table replaced.  Reachable with ``OASR_RECURRENT_CUTE=1``.
_LSTM_BANDS: tuple = (
    (256, (1, 1 << 30)),
    (768, (1, 256)),
    (1536, (1, 64)),
    (1 << 30, (1, 32)),
)

#: Where the two tables above came from.
#:
#: Both are fixed ``(hidden, batch)`` cut-offs, and a cut-off is the one kind of
#: routing decision in this package that does *not* travel: the tile choice in
#: ``jit.mlp`` reads ``multi_processor_count`` and the real opt-in smem and does
#: wave arithmetic against them, so it follows the machine, whereas a number
#: somebody timed follows the machine it was timed on.  The supported set here is
#: sm_80 / 86 / 89 / 120 — an A30 at 56 SMs and 933 GB/s, an A100 at 108 and
#: 1555, an L40S at 142 and 864, a 5090 at 170 and 1792 — and the band is applied
#: identically on all four.
#:
#: It still applies, because what the band encodes is not arbitrary: at the small
#: end the step is at the weight-read bandwidth floor (B=16, H=640 is 3.28 MB,
#: L2 delivers it in ~3.3 us against 3.56 measured) and at the large end a tuned
#: library GEMM keeps a 7-11% mainloop edge.  Which side wins at each extreme is
#: a property of the algorithm.  Where the two meet is a property of the machine,
#: and *that* is what this record is about: off the measured card the boundary is
#: an extrapolation, and :func:`oasr.jit.measured.note_extrapolation` makes it say
#: so once instead of never.
_MEASURED = MeasuredOn(
    table="jit.recurrent_cute._LSTM_BANDS / _TILES",
    machine=Machine(name="NVIDIA GeForce RTX 5090", sm=120, sms=170),
    bandwidth="1792 GB/s",
    source=".artifacts/recurrent_cute_envelope.md (2026-08-23)",
    moves_with="SM count and memory bandwidth — both edges of the band are set by "
    "them, the small one by the weight read and the large one by when a library "
    "GEMM's mainloop starts to win",
)


def _read_mode() -> str:
    raw = os.environ.get(_ENV, "auto").lower()
    if raw in ("1", "always", "on"):
        return "always"
    if raw in ("0", "off", "never"):
        return "off"
    if raw != "auto":
        logger.warning("%s=%r is not recognised; using 'auto'.", _ENV, raw)
    return "auto"


def _read_backend() -> str:
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


_MODE = _read_mode()
_BACKEND = _read_backend()


#: ``(dtype_str, gate_count, activation, hidden, batch)`` ->
#: ``(backend, callable)``, or ``None`` when this shape is outside the band or
#: no lane is available.  Populated by :func:`routed_step`; cleared by
#: :func:`set_mode` and :func:`set_backend`.
_ROUTE: dict = {}


def get_mode() -> str:
    return _MODE


def _clear_all_caches() -> None:
    _compiled_step.cache_clear()
    _probe.cache_clear()
    _cxx_probe.cache_clear()
    try:
        from oasr.jit import recurrent_step as _cxx

        _cxx.clear_caches()
    except Exception:  # the C++ lane is optional; nothing to clear if absent
        pass
    # ``routed_step`` memoises the *whole* decision, gate and lane included, so
    # a change that did not clear it would keep serving the old routing --
    # which is exactly what an A/B or a rollback switch is for.
    _ROUTE.clear()


def set_mode(mode: str) -> None:
    """Override the gate for the rest of the process.  Used by tests and A/Bs."""
    global _MODE
    if mode not in ("auto", "always", "off"):
        raise ValueError(f"invalid mode {mode!r}; valid: auto / always / off")
    _MODE = mode
    _clear_all_caches()


def get_backend() -> str:
    """Return the lane preference (``auto`` / ``cute`` / ``cxx``)."""
    return _BACKEND


def set_backend(backend: str) -> None:
    """Override the lane preference for the rest of the process.  Tests and A/Bs.

    Note this clears both compile caches, so flipping in a loop recompiles.
    For an in-process A/B -- a parametrised fixture, a differential test, an
    interleaved benchmark arm -- ask :func:`get_compiled_step` or
    :func:`get_cxx_step` for the lane you want by name and leave the global
    alone.
    """
    global _BACKEND
    backend = _BACKEND_ALIASES.get(backend, backend)
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"invalid backend {backend!r}; valid: {_VALID_BACKENDS}")
    _BACKEND = backend
    _clear_all_caches()


def _lane_order() -> Tuple[str, ...]:
    return _AUTO_ORDER if _BACKEND == "auto" else (_BACKEND,)


@functools.cache
def _probe() -> Optional[Tuple[int, int]]:
    """Compute capability if the CuTeDSL step is usable here, else ``None``.

    Returns ``None`` when the lane is pinned away as well as when it is
    unusable, so ``OASR_RECURRENT_BACKEND=cxx`` never imports CuTeDSL at all --
    which is the point of a lane that does not need it installed.
    """
    if _MODE == "off" or _BACKEND == "cxx":
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

        from oasr.kernels.cute.recurrent import RecurrentStepCute  # noqa: F401
    except Exception as exc:
        logger.warning("CuTeDSL recurrent step unavailable (%s); leaving it out.", exc)
        return None
    return cap


def select_tile(hidden: int, batch: int) -> Optional[Tuple[int, ...]]:
    """Tuned tile for this shape, or ``None`` if the table does not cover it."""
    for width, batch_cap, tile in _TILES:
        if hidden <= width and batch <= batch_cap:
            return cast(Tuple[int, ...], tile)
    return None


def should_use(gate_count: int, hidden: int, batch: int) -> bool:
    """Is this shape inside the band where the fused step measured fastest?

    Policy, not capability: it asks whether *fusing* wins here, and the answer
    does not depend on which lane runs it.  The band is a property of the
    tiling -- both lanes run the same tile on every shape -- so it does not
    move with the lane.  :func:`routed_step` asks the capability question per
    lane afterwards.
    """
    if not any(_lane_available(b) for b in _lane_order()):
        return False
    tile = select_tile(hidden, batch)
    if tile is None:
        return False
    if _MODE == "always":
        return True
    if gate_count != 4:
        return False
    # Consulting the band on a card it was not measured on is an extrapolation.
    # Say so once and count it, rather than let a number timed on one GPU look
    # like a property of the kernel (audit A8).  This is reached once per shape:
    # ``routed_step`` memoises the whole decision.
    note_extrapolation(_MEASURED)
    for width, (low, high) in _LSTM_BANDS:
        if hidden <= width:
            return bool(low <= batch <= high)
    return False


@functools.cache
def _compiled_step(
    arch: Tuple[int, int],
    dtype_str: str,
    gate_count: int,
    activation: str,
    tile: Tuple[int, int, int, int, int, int],
):
    """Compile one configuration.  Shapes stay dynamic, so B/H are not in the key."""
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from oasr.kernels.cute.recurrent import RecurrentStepCute

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
    if not RecurrentStepCute.can_implement(
        dtype=cute_dtype,
        gate_count=gate_count,
        activation=activation,
        m_block=m_block,
        n_block=n_block,
        k_block=k_block,
        num_stages=num_stages,
        num_threads=num_threads,
        warps_n=warps_n,
    ):
        raise RuntimeError(f"RecurrentStepCute cannot implement tile {tile}")
    inst = RecurrentStepCute(
        dtype=cute_dtype,
        gate_count=gate_count,
        activation=activation,
        m_block=m_block,
        n_block=n_block,
        k_block=k_block,
        num_stages=num_stages,
        num_threads=num_threads,
        warps_n=warps_n,
    )

    def wrap(t: torch.Tensor):
        # divisibility=8 is the 128-bit cp.async guarantee.  Without it the
        # compiler only knows the leading dim is dynamic and refuses the atom.
        return (
            from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)
            .mark_layout_dynamic(leading_dim=t.dim() - 1)
            .mark_compact_shape_dynamic(
                mode=t.dim() - 1, stride_order=t.dim_order(), divisibility=8
            )
        )

    # Descriptors only: rank, dtype and which dims are dynamic.  Values unused.
    hidden = 64

    def empty(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, device="cuda", dtype=torch_dtype)

    args = (
        wrap(empty(m_block, hidden)),
        wrap(empty(gate_count * hidden, hidden)),
        wrap(empty(m_block, gate_count * hidden)),
        wrap(empty(m_block, hidden)),
        wrap(empty(m_block, hidden)),
        wrap(empty(m_block, hidden)),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    # --enable-tvm-ffi is not an optimisation detail: without it every call
    # rebuilds six DLPack descriptors, which measured ~148 us and buried a 6 us
    # kernel completely.
    return cute.compile(inst, *args, stream, options="--enable-tvm-ffi")


# ---------------------------------------------------------------------------
# The C++ CUTLASS/CuTe lane
# ---------------------------------------------------------------------------

#: ``(gate_count, activation)`` -- the pair the functional API and the CuTeDSL
#: lane use -- to the single ``kind`` axis the C++ lane compiles.  Over there
#: the two halves have to agree and ``can_implement`` has a clause enforcing
#: it; here the inconsistent combination cannot be spelled.
_KIND: dict = {(4, "lstm"): "lstm", (1, "tanh"): "rnn_tanh", (1, "relu"): "rnn_relu"}


@functools.cache
def _cxx_probe() -> Optional[int]:
    """The target SM if the C++ lane is usable here, else ``None``.

    Not resolved at import, for the same reason the CuTeDSL probe is not: the
    steady-state hot path is a :data:`_ROUTE` dict lookup that never reaches
    either.
    """
    if _MODE == "off" or _BACKEND == "cute":
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        from oasr.jit.core import _get_target_sm
        from oasr.jit.recurrent_step import SUPPORTED_SM
    except Exception:
        return None
    try:
        sm = _get_target_sm()
    except Exception:  # an architecture OASR compiles for nothing at all
        return None
    return sm if sm in SUPPORTED_SM else None


def _lane_available(backend: str) -> bool:
    return (_cxx_probe() is not None) if backend == "cxx" else (_probe() is not None)


def cute_config_supported(*, gate_count: int, activation: str, hidden: int, batch: int) -> bool:
    """Would the **CuTeDSL** lane accept this problem?

    Capability only -- arch, CuTeDSL and a tile for these bounds.  It says
    nothing about whether fusing is *faster* here; that is :func:`should_use`.
    """
    if _probe() is None:
        return False
    if (gate_count, activation) not in _KIND:
        return False
    # The CuTeDSL kernel loops `ceil_div(K, k_block)` and predicates only the
    # row axis, so a hidden width that is not a whole number of K tiles would
    # read past the end of both operands.  It is not a refusal over there --
    # nothing checks it -- so the check has to live here.
    tile = select_tile(hidden, batch)
    if tile is None or hidden % tile[2]:
        return False
    return True


def cxx_config_supported(
    *, dtype_str: str, gate_count: int, activation: str, hidden: int, batch: int
) -> bool:
    """Would the **C++** lane accept this problem?

    Its shape contract is strictly wider than the CuTeDSL lane's: the hidden
    width need not be a whole number of K tiles, because the residue is
    predicated and the ZFILL cp.async makes the skipped elements zero.

    ``k=hidden`` is not an assumption the *kernel* makes -- it reads a ``K``
    wide state and writes an ``H`` wide one, and the two are independent there.
    It is a fact about this arbiter's only caller: ``lstm_gemm_layer`` has
    already required ``weight_hh`` to be ``(4H, H)`` and ``initial_h`` to be
    ``(B, H)``, so the two coincide by the time the question is asked.  A
    projected LSTM would reach the kernel through a different entry point and
    pass its own ``k``.
    """
    sm = _cxx_probe()
    if sm is None:
        return False
    kind = _KIND.get((gate_count, activation))
    if kind is None:
        return False
    from oasr.jit import recurrent_step as _cxx

    return _cxx.config_supported(
        sm=sm, dtype_str=dtype_str, kind=kind, batch=batch, hidden=hidden, k=hidden
    )


def get_cxx_step(*, dtype_str: str, gate_count: int, activation: str, hidden: int, batch: int):
    """The C++ launcher for this shape's tuned tile, building its cell on first use.

    Raises rather than declining, which is the right contract for a caller that
    asked for the lane by name; :func:`routed_step` is the one that chooses.
    """
    sm = _cxx_probe()
    if sm is None:
        raise RuntimeError("the C++ recurrent step lane is not available on this device")
    kind = _KIND.get((gate_count, activation))
    if kind is None:
        raise RuntimeError(f"no C++ recurrent kind for gates={gate_count} {activation!r}")
    from oasr.jit import recurrent_step as _cxx

    tile = _cxx.routed_tile(kind=kind, hidden=hidden, batch=batch, sm=sm)
    if tile < 0:
        raise RuntimeError(f"no tile for hidden={hidden} batch={batch} on sm_{sm}")
    return _cxx.get_recurrent_step_fn(dtype_str=dtype_str, kind=kind, tile_index=tile, sm=sm)


def get_compiled_step(*, dtype_str: str, gate_count: int, activation: str, hidden: int, batch: int):
    """Compiled CuTeDSL callable for this shape's tuned tile, compiling on first use."""
    cap = _probe()
    if cap is None:
        raise RuntimeError("the CuTeDSL recurrent step is not available on this device")
    tile = select_tile(hidden, batch)
    if tile is None:
        raise RuntimeError(f"no tuned tile for hidden={hidden} batch={batch}")
    return _compiled_step(cap, dtype_str, gate_count, activation, tile)


def routed_step(*, dtype_str: str, gate_count: int, activation: str, hidden: int, batch: int):
    """``(backend, callable)`` for this shape, or ``None`` for the other path.

    The two lanes have **different call signatures** -- the C++ one takes its
    outputs first (``AGENTS.md`` rule 4) and reads the stream from the FFI
    environment, the CuTeDSL one takes them last and a stream handle -- so the
    backend name comes back with the callable rather than being hidden behind a
    wrapper.  A wrapper would be one more Python frame on a path a transducer
    predictor walks twice per emitted label, and the whole point of
    :data:`_ROUTE` is that the steady state is one dict lookup.

    ``should_use()`` followed by a getter is the readable spelling and costs
    1.18 us per call -- two table scans, an arch probe and a
    ``functools.cache`` key build, twice per ``LSTM.forward`` because a
    transducer predictor has two layers.  All of it is a pure function of the
    shape, so the whole decision memoises to one dict lookup.

    Deciding *and* compiling under the same key also means a shape whose kernel
    fails to build is remembered as declined rather than retried per step; a
    build failure is a property of the configuration, not of the call.
    """
    key = (dtype_str, gate_count, activation, hidden, batch)
    try:
        return _ROUTE[key]
    except KeyError:
        pass
    route = None
    if should_use(gate_count, hidden, batch):
        for backend in _lane_order():
            supported = (
                cxx_config_supported(
                    dtype_str=dtype_str,
                    gate_count=gate_count,
                    activation=activation,
                    hidden=hidden,
                    batch=batch,
                )
                if backend == "cxx"
                else cute_config_supported(
                    gate_count=gate_count,
                    activation=activation,
                    hidden=hidden,
                    batch=batch,
                )
            )
            if not supported:
                continue
            getter = get_cxx_step if backend == "cxx" else get_compiled_step
            try:
                route = (
                    backend,
                    getter(
                        dtype_str=dtype_str,
                        gate_count=gate_count,
                        activation=activation,
                        hidden=hidden,
                        batch=batch,
                    ),
                )
                break
            except Exception as exc:  # unsupported arch, missing toolchain, bad tile
                logger.warning(
                    "the %s recurrent step declined hidden=%d batch=%d: %s",
                    backend,
                    hidden,
                    batch,
                    exc,
                )
    _ROUTE[key] = route
    return route


#: The stream handle a compiled CuTeDSL callable needs, cached against the raw
#: pointer -- ``torch.cuda.current_stream()`` alone cost 4.1 us against a 6 us
#: kernel.  Shared with the FMHA call sites; see :mod:`oasr.jit.cute_runtime`.
from .cute_runtime import current_stream  # noqa: E402,F401  (re-export)


def warmup(*, dtype_str: str, gate_count: int, activation: str, hidden: int, batch: int) -> None:
    """Populate the route + compile cache ahead of the first step.  Never raises."""
    routed_step(
        dtype_str=dtype_str,
        gate_count=gate_count,
        activation=activation,
        hidden=hidden,
        batch=batch,
    )
