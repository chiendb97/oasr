# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Compile cache and routing for the CuTeDSL fused recurrent step.

Like :mod:`oasr.jit.attention`, and unlike :mod:`oasr.jit.core`, this is not the
Ninja C++ pipeline: CuTeDSL kernels are Python and compile through
``cutlass.cute.compile()``, which hands back a callable.  One callable is cached
per configuration and reused.

``OASR_RECURRENT_CUTE`` gates it:

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

logger = logging.getLogger("oasr.jit.recurrent_cute")

_ENV = "OASR_RECURRENT_CUTE"

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


def _read_mode() -> str:
    raw = os.environ.get(_ENV, "auto").lower()
    if raw in ("1", "always", "on"):
        return "always"
    if raw in ("0", "off", "never"):
        return "off"
    if raw != "auto":
        logger.warning("%s=%r is not recognised; using 'auto'.", _ENV, raw)
    return "auto"


_MODE = _read_mode()


#: ``(dtype_str, gate_count, activation, hidden, batch)`` -> the compiled step, or
#: ``None`` when this shape is outside the band or the kernel is unavailable.
#: Populated by :func:`routed_step`; cleared by :func:`set_mode`.
_ROUTE: dict = {}


def get_mode() -> str:
    return _MODE


def set_mode(mode: str) -> None:
    """Override the gate for the rest of the process.  Used by tests and A/Bs."""
    global _MODE
    if mode not in ("auto", "always", "off"):
        raise ValueError(f"invalid mode {mode!r}; valid: auto / always / off")
    _MODE = mode
    _compiled_step.cache_clear()
    _probe.cache_clear()
    # ``routed_step`` memoises the *whole* decision, gate included, so a mode
    # change that did not clear it would keep serving the old routing -- which is
    # exactly what an A/B or a rollback switch is for.
    _ROUTE.clear()


@functools.cache
def _probe() -> Optional[Tuple[int, int]]:
    """Compute capability if the CuTeDSL step is usable here, else ``None``."""
    if _MODE == "off":
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
    """Is this shape inside the band where the CuTeDSL step measured fastest?"""
    if _probe() is None:
        return False
    tile = select_tile(hidden, batch)
    if tile is None:
        return False
    if _MODE == "always":
        return True
    if gate_count != 4:
        return False
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


def get_compiled_step(*, dtype_str: str, gate_count: int, activation: str, hidden: int, batch: int):
    """Compiled callable for this shape's tuned tile, compiling on first use."""
    cap = _probe()
    if cap is None:
        raise RuntimeError("the CuTeDSL recurrent step is not available on this device")
    tile = select_tile(hidden, batch)
    if tile is None:
        raise RuntimeError(f"no tuned tile for hidden={hidden} batch={batch}")
    return _compiled_step(cap, dtype_str, gate_count, activation, tile)


def routed_step(*, dtype_str: str, gate_count: int, activation: str, hidden: int, batch: int):
    """The step to run for this shape, or ``None`` to leave it on the other path.

    ``should_use()`` followed by ``get_compiled_step()`` is the readable spelling
    and costs 1.18 us per call -- two table scans, an arch probe and a
    ``functools.cache`` key build, twice per ``LSTM.forward`` because a
    transducer predictor has two layers.  Both are pure functions of the shape,
    so the whole decision memoises to one dict lookup.

    Deciding *and* compiling under the same key also means a shape whose tile
    fails to build is remembered as declined rather than retried per step; a
    compile failure is a property of the configuration, not of the call.
    """
    key = (dtype_str, gate_count, activation, hidden, batch)
    try:
        return _ROUTE[key]
    except KeyError:
        pass
    step = None
    if should_use(gate_count, hidden, batch):
        try:
            step = get_compiled_step(
                dtype_str=dtype_str,
                gate_count=gate_count,
                activation=activation,
                hidden=hidden,
                batch=batch,
            )
        except Exception as exc:  # unsupported arch, missing CuTeDSL, unbuildable tile
            logger.warning(
                "CuTeDSL recurrent step declined for hidden=%d batch=%d: %s", hidden, batch, exc
            )
            step = None
    _ROUTE[key] = step
    return step


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
