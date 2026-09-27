# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The C++ CUTLASS/CuTe lane for the fused recurrent step.

Unlike :mod:`oasr.jit.recurrent_cute` -- which wraps ``cutlass.cute.compile()``
for the CuTeDSL kernel and also arbitrates between the two lanes -- this is the
repo's ordinary C++ JIT pipeline: Jinja renders one translation unit per
variant, ninja builds them into one ``.so`` per *cell*, and the result is
cached on disk in ``~/.cache/oasr/jit``.

:mod:`oasr.jit.recurrent_cute` stays the arbiter that answers "which backend?";
this module is one of the answers, and never imports it back.

Cells and variants
------------------
A **cell** is ``(target_sm, dtype, kind)``, and inside a cell the **variants**
are the eight CTA tiles of ``kRecurrentStepTiles``.  All eight go into that one
module, because nvcc parallelises across translation units but not within one
(the reasoning is written out at ``oasr/jit/gemm.py:1000``) -- and then the
cell covers every shape it will ever be asked for.

The CuTeDSL lane recompiles per *tile*, so a predictor whose cohort grows
across the ladder pays a fresh ``cutlass.cute.compile()`` at each rung; here it
pays one ninja wave and never again.

The **kind** stays a cell because it is the axis that changes the arithmetic,
and a layer means exactly one of them.  Everything else is deliberately
**runtime**: ``M``, ``H``, ``K``, all six row strides, and the K residue.  Each
of those would have multiplied the variant count for nothing.

The tile is compiled but not *chosen* here: ``recurrentStepSelectTile`` is
``constexpr`` in ``include/oasr/recurrent/recurrent_step_tiles.h``, and
:func:`select_tile` below only *mirrors* it for routing.  The two are pinned
equal by ``tests/kernels/test_recurrent_cpp.py``, against the C++ answer
exported as ``recurrent_step_select_tile`` -- a mirror nobody checks is a
mirror that drifts.

What this lane can serve that the CuTeDSL one cannot
----------------------------------------------------
* **A hidden width that is not a whole number of K tiles.**  The CuTeDSL
  kernel loops ``ceil_div(K, k_block)`` and predicates only the row axis, so a
  ragged K reads past the end of both operands.  Here the K residue is
  predicated and the ZFILL cp.async makes the skipped elements zero, which is
  the identity for the dot product.
* **Arbitrary row strides.**  The CuTeDSL call site marks its tensors
  ``mark_compact_shape_dynamic``, so every operand has to be fully contiguous;
  this one takes a row-slice of a wider buffer without a copy.
* **A machine with no CuTeDSL installed.**  ``nvidia-cutlass-dsl`` is an extra;
  the vendored CUTLASS headers this lane compiles against are not.
"""

from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import arch_facts, env
from .core import JitSpec, _get_target_sm, gen_jit_spec
from .cubin_loader import write_if_different
from .measured import Machine, MeasuredOn, note_extrapolation
from .templates import render_template

__all__ = [
    "ALIGNMENT",
    "ACC_PAD",
    "KINDS",
    "SUPPORTED_SM",
    "Tile",
    "TILES",
    "ROUTES",
    "gate_count",
    "tile_valid",
    "smem_bytes",
    "ctas_per_sm",
    "select_tile",
    "config_supported",
    "shape_supported",
    "gen_recurrent_step_module",
    "gen_recurrent_step_modules",
    "get_recurrent_step_fn",
    "recurrent_step_coverage_report",
    "variant_symbol",
]

#: Architectures this lane compiles for.  sm_90 / sm_100 would *run* the
#: Ampere-class collective -- mma.sync and cp.async are both available there --
#: but routing to it would be a performance claim with no measurement behind
#: it, and the right answer on those parts is a wgmma / tcgen05 mainloop of
#: their own.  The omission is counted in
#: :func:`recurrent_step_coverage_report`, not hidden (``AGENTS.md`` rule 3).
#: Mirrors the ``RecurrentStepArch<SM>`` specializations in
#: ``include/oasr/recurrent/cutlass_recurrent_step_configs.h``, and is the same
#: set the CuTeDSL lane declares.
SUPPORTED_SM: Tuple[int, ...] = (80, 86, 89, 120)

#: The 128-bit vector contract, in elements.  Mirrors ``kRecurrentStepAlignment``.
ALIGNMENT = 8

#: Floats of padding per staged accumulator row.  Mirrors ``kRecurrentStepAccPad``.
ACC_PAD = 8

#: The recurrences this lane compiles, mapped to ``oasr::recurrent::RecurrentKind``.
#: The names are the ``(gate_count, activation)`` pairs the functional API and
#: the CuTeDSL lane use, collapsed into the single axis that actually varies --
#: over there the two halves have to agree and ``can_implement`` has a clause
#: enforcing it; here the inconsistent combination cannot be spelled.
_KIND_ID: Dict[str, int] = {"lstm": 0, "rnn_tanh": 1, "rnn_relu": 2}
_KIND_ENUM: Dict[str, str] = {"lstm": "LSTM", "rnn_tanh": "RNN_TANH", "rnn_relu": "RNN_RELU"}
KINDS: Tuple[str, ...] = tuple(_KIND_ID)

_DTYPES = {"float16": "cutlass::half_t", "bfloat16": "cutlass::bfloat16_t"}

#: Hardware tables, mirrored once for every C++ CuTe family in
#: :mod:`oasr.jit.arch_facts`.  Re-exported under the names the mirror tests use.
_SMEM_CAPACITY = arch_facts.SMEM_CAPACITY
_MAX_THREADS_PER_SM = arch_facts.MAX_THREADS_PER_SM
#: Mirrors ``oasr::recurrent::kRecurrentStepUnbounded``.
UNBOUNDED = 1 << 30


def gate_count(kind: str) -> int:
    """Gate columns per hidden unit.  Mirrors ``recurrentGateCount``."""
    return 4 if kind == "lstm" else 1


@dataclass(frozen=True)
class Tile:
    """One entry of ``kRecurrentStepTiles``."""

    block_m: int
    block_n: int
    block_k: int
    stages: int
    threads: int
    warps_n: int


#: Mirrors ``oasr::recurrent::kRecurrentStepTiles``, **in the same order** --
#: ``ROUTES`` indexes into it, so a permutation here is a silent routing
#: change.  Pinned entry-for-entry by ``tests/kernels/test_recurrent_cpp.py``.
TILES: Tuple[Tile, ...] = (
    Tile(16, 64, 64, 4, 128, 4),
    Tile(16, 64, 64, 5, 128, 4),
    Tile(32, 32, 64, 4, 128, 2),
    Tile(32, 64, 64, 3, 256, 4),
    Tile(64, 64, 64, 3, 512, 4),
    Tile(64, 64, 64, 4, 512, 4),
    Tile(128, 64, 64, 3, 512, 2),
    Tile(128, 128, 64, 3, 512, 4),
)

#: ``(hidden_max, batch_max, tile index)``, scanned in order; the first rung
#: whose two bounds both fit wins.  Mirrors
#: ``oasr::recurrent::kRecurrentStepRoutes``, which is itself the CuTeDSL
#: lane's measured ``_TILES`` ladder reproduced -- the two lanes therefore run
#: the *same* tile on every shape, which is what makes an A/B between them
#: measure the kernels.
ROUTES: Tuple[Tuple[int, int, int], ...] = (
    (256, 128, 2),
    (256, 256, 3),
    (256, UNBOUNDED, 5),
    (768, 64, 2),
    (768, 128, 3),
    (768, 256, 5),
    (768, UNBOUNDED, 6),
    (1536, 16, 2),
    (1536, 32, 1),
    (1536, 64, 3),
    (1536, 128, 4),
    (1536, UNBOUNDED, 6),
    (UNBOUNDED, 8, 0),
    (UNBOUNDED, 16, 1),
    (UNBOUNDED, 32, 3),
    (UNBOUNDED, 64, 4),
    (UNBOUNDED, 128, 6),
    (UNBOUNDED, UNBOUNDED, 7),
)

#: Where the ladder came from.
#:
#: Inherited from the CuTeDSL lane rather than re-derived, so that an A/B
#: between the two lanes compares kernels and not tile choices.  It is a fixed
#: ``(hidden, batch)`` cut-off, and a cut-off is the one kind of routing
#: decision in this package that does *not* travel: what the ladder encodes is
#: which of two regimes a shape is in -- weight-read bound at the small end,
#: MMA bound at the large end -- and *which side wins at each extreme* is a
#: property of the algorithm, while *where the two meet* is a property of the
#: machine.  Off the measured card that boundary is an extrapolation, and
#: :func:`oasr.jit.measured.note_extrapolation` makes it say so once instead of
#: never.
_MEASURED = MeasuredOn(
    table="include/oasr/recurrent/recurrent_step_tiles.h::kRecurrentStepRoutes",
    machine=Machine(name="NVIDIA GeForce RTX 5090", sm=120, sms=170),
    bandwidth="1792 GB/s",
    source="oasr/jit/recurrent_cute.py::_TILES (.artifacts/recurrent_cute_envelope.md)",
    moves_with="SM count and memory bandwidth -- both edges of the band are set by "
    "them, the small one by the weight read and the large one by when a library "
    "GEMM's mainloop starts to win",
)

_ARCH_UNDERSERVED: "Counter[int]" = Counter()
_REFUSED: "Counter[str]" = Counter()


# ---------------------------------------------------------------------------
# The mirror of `recurrent_step_tiles.h`
# ---------------------------------------------------------------------------


smem_budget = arch_facts.smem_budget


def smem_bytes(tile: Tile, elem_size: int = 2) -> int:
    """Shared memory the kernel requests, in bytes.  Mirrors ``recurrentStepSmemBytes``.

    A **union** of the ring and the epilogue's staging buffer, not a sum: the
    mainloop has drained and barriered by the time the epilogue writes.  (The
    CuTeDSL lane *aliases* rather than unions, so it has to refuse a tile whose
    staging exceeds its ring -- ``RecurrentStepCute.can_implement``.)
    """
    mainloop = tile.stages * (tile.block_m + tile.block_n) * tile.block_k * elem_size
    epilogue = tile.block_m * (tile.block_n + ACC_PAD) * 4
    return max(mainloop, epilogue)


def tile_valid(tile: Tile, sm: int, elem_size: int = 2, gates: int = 4) -> bool:
    """Would this tile compile, fit, and address only its own tiles?

    Mirrors ``recurrentStepTileValid`` clause for clause.  Each one is a
    *silent* failure if dropped: the MMA clauses become an unreadable CuTe
    layout error, ``block_n % (8 * gates)`` makes a cell's gates straddle two
    tiles, the tiled-copy clauses become an illegal access no predicate can
    intercept (the *partition* is out of range), and the shared-memory clause
    becomes a launch failure with an empty message.
    """
    if elem_size != 2 or gates not in (1, 4):
        return False
    if not arch_facts.sm80_warp_tiling_valid(
        tile.block_m, tile.block_n, tile.threads, tile.warps_n
    ):
        return False
    if tile.block_n % (8 * gates) or tile.block_n % 8:
        return False
    if tile.block_k % 32 or tile.stages < 2:
        return False
    if not (
        arch_facts.sm80_tiled_copy_fits(tile.block_m, tile.block_k, tile.threads, elem_size)
        and arch_facts.sm80_tiled_copy_fits(tile.block_n, tile.block_k, tile.threads, elem_size)
    ):
        return False
    if (tile.block_m * (tile.block_n // gates)) % tile.threads:
        return False
    return arch_facts.fits_sm(smem_bytes(tile, elem_size), tile.threads, sm)


def ctas_per_sm(tile: Tile, sm: int, elem_size: int = 2) -> int:
    """How many of these CTAs are resident at once.  Mirrors ``recurrentStepCtasPerSm``.

    Registers are deliberately not modelled: these tiles measure well inside
    what shared memory already allows.  A tile that changed that would show up
    as a *measured* regression, not as a wrong number here.
    """
    return arch_facts.ctas_per_sm(smem_bytes(tile, elem_size), tile.threads, sm)


@functools.lru_cache(maxsize=None)
def select_tile(sm: int, hidden: int, batch: int, elem_size: int = 2, gates: int = 4) -> int:
    """Index into :data:`TILES` for this shape, or ``-1``.

    Mirrors ``recurrentStepSelectTile``: the first rung of :data:`ROUTES` whose
    two bounds both fit and whose tile fits this architecture wins; failing
    that, the smallest ring that fits at all, so a part with less shared memory
    degrades rather than declining.

    A pure function of ``(shape, architecture, dtype, gate count)`` and
    **nothing else** -- in particular not of CUDA-graph capture state
    (``AGENTS.md`` rule 11): two tiles sum the K loop in different orders, so
    an answer that differed under capture would make a replayed graph produce
    different numbers than eager.
    """
    if hidden <= 0 or batch <= 0:
        return -1
    for hidden_max, batch_max, index in ROUTES:
        if (
            hidden <= hidden_max
            and batch <= batch_max
            and tile_valid(TILES[index], sm, elem_size, gates)
        ):
            return index
    best, best_bytes = -1, 0
    for i, tile in enumerate(TILES):
        if not tile_valid(tile, sm, elem_size, gates):
            continue
        nbytes = smem_bytes(tile, elem_size)
        if best < 0 or nbytes < best_bytes:
            best, best_bytes = i, nbytes
    return best


def shape_supported(*, batch: int, hidden: int, k: int) -> bool:
    """Does this problem meet the kernel's static contract?

    Only the 128-bit vector contract on ``K``, the contiguous axis of both
    mainloop operands.  ``K`` is **not** required to be a whole number of K
    tiles -- the residue is predicated and the ZFILL cp.async makes the skipped
    elements zero -- and ``hidden`` carries no constraint at all, because the
    outputs are written one element per hidden unit.
    """
    return batch > 0 and hidden > 0 and k > 0 and k % ALIGNMENT == 0


def config_supported(
    *, sm: int, dtype_str: str, kind: str, batch: int, hidden: int, k: int
) -> bool:
    """Can this lane serve the shape?  Asked before anything is built.

    Answering only "does a tile fit" would route a caller to a symbol that does
    not exist, so the kind and the shape contract are checked here too --
    either "no" is a "no".
    """
    if dtype_str not in _DTYPES:
        return False
    if kind not in _KIND_ID:
        _REFUSED[f"kind {kind!r}"] += 1
        return False
    if sm not in SUPPORTED_SM:
        if sm in _SMEM_CAPACITY:
            _ARCH_UNDERSERVED[sm] += 1
        return False
    if not shape_supported(batch=batch, hidden=hidden, k=k):
        _REFUSED["K is not a multiple of the 128-bit vector width"] += 1
        return False
    gates = gate_count(kind)
    if select_tile(sm, hidden, batch, 2, gates) < 0:
        _REFUSED[f"no tile fits sm_{sm}'s {smem_budget(sm)} B of shared memory"] += 1
        return False
    return True


# ---------------------------------------------------------------------------
# Rendering and building
# ---------------------------------------------------------------------------


def variant_symbol(sm: int, dtype_str: str, kind: str, tile_index: int) -> str:
    """The exported name of one variant.

    Produced by exactly one function so the renderer and the loader cannot
    disagree about it.
    """
    return f"recurrent_step_sm{sm}_{dtype_str}_{kind}_t{tile_index}"


def cell_name(sm: int, dtype_str: str, kind: str) -> str:
    return f"recurrent_step_sm{sm}_{dtype_str}_{kind}"


@functools.lru_cache(maxsize=None)
def gen_recurrent_step_module(sm: int, dtype_str: str, kind: str) -> JitSpec:
    """Render and describe one cell's translation units -- one per tile."""
    if dtype_str not in _DTYPES:
        raise ValueError(f"recurrent_step: unsupported dtype {dtype_str!r}")
    if kind not in _KIND_ID:
        raise ValueError(f"recurrent_step: unsupported kind {kind!r}")
    gates = gate_count(kind)
    if not any(tile_valid(t, sm, 2, gates) for t in TILES):
        raise RuntimeError(
            f"recurrent_step: no tile fits sm_{sm}'s {smem_budget(sm)} B of shared memory"
        )
    name = cell_name(sm, dtype_str, kind)
    out_dir = env.OASR_GEN_SRC_DIR / "recurrent_step"
    sources: List = []
    for index, tile in enumerate(TILES):
        if not tile_valid(tile, sm, 2, gates):
            # Rendering it would fail the launcher's static_assert at compile
            # time and take the whole cell with it.  `select_tile` cannot
            # return an index that is not here, because it applies the same
            # predicate.
            continue
        func = variant_symbol(sm, dtype_str, kind, index)
        rendered = render_template(
            "recurrent_step_template.cu.jinja",
            func_name=func,
            sm_version=sm,
            dtype=dtype_str,
            cutlass_dtype=_DTYPES[dtype_str],
            kind=kind,
            kind_id=_KIND_ID[kind],
            kind_enum=_KIND_ENUM[kind],
            has_cell=(gates == 4),
            tile_index=index,
            tile=tile,
        )
        path = out_dir / f"{func}.cu"
        write_if_different(path, rendered)
        sources.append(path)
    sources.append(env.OASR_CSRC_DIR / "recurrent_step_jit_binding.cu")
    return gen_jit_spec(name, sources)


def gen_recurrent_step_modules(
    cells: Optional[List[Tuple[str, str]]] = None,
) -> List[JitSpec]:
    """Specs for the AOT set: the ``(dtype, kind)`` cells shipped models reach.

    An LSTM in both served dtypes.  Every shipped recurrent layer in scope -- a
    transducer predictor, Nemotron's prediction network -- is an LSTM; a vanilla
    RNN compiles its cell on first use, which costs one ninja wave.
    """
    if cells is None:
        cells = [(dt, "lstm") for dt in ("float16", "bfloat16")]
    sm = _get_target_sm()
    if sm not in SUPPORTED_SM:
        return []
    return [gen_recurrent_step_module(sm, dt, kind) for dt, kind in cells]


@functools.lru_cache(maxsize=None)
def _cell_module(sm: int, dtype_str: str, kind: str):
    return gen_recurrent_step_module(sm, dtype_str, kind).build_and_load()


#: Cells whose build has already failed once.
_BUILD_FAILED: set = set()


def _load_cell(sm: int, dtype_str: str, kind: str):
    """:func:`_cell_module`, with build *failures* remembered.

    ``functools.lru_cache`` does not memoise exceptions, so a cell that cannot
    build -- no ninja, no nvcc, a toolchain mismatch -- would re-invoke the
    whole build for every new shape key the router asks about.  A build failure
    is a property of the configuration, not of the call.
    """
    key = (sm, dtype_str, kind)
    if key in _BUILD_FAILED:
        raise RuntimeError(f"recurrent_step: cell {cell_name(*key)} failed to build earlier")
    try:
        return _cell_module(*key)
    except Exception:
        _BUILD_FAILED.add(key)
        _REFUSED[f"cell {cell_name(*key)} failed to build"] += 1
        raise


def get_recurrent_step_fn(*, dtype_str: str, kind: str, tile_index: int, sm: Optional[int] = None):
    """The bound launcher for one variant, building its cell on first use."""
    sm = _get_target_sm() if sm is None else sm
    if not (0 <= tile_index < len(TILES)):
        raise ValueError(f"recurrent_step: no tile {tile_index}")
    module = _load_cell(sm, dtype_str, kind)
    return getattr(module, variant_symbol(sm, dtype_str, kind, tile_index))


def routed_tile(*, kind: str, hidden: int, batch: int, sm: Optional[int] = None) -> int:
    """The tile this machine would pick for the shape, or ``-1``.

    Unlike the gated MLP's, this reads no device property: the ladder is keyed
    on the *shape* alone.  :func:`note_extrapolation` is what records that the
    cut-offs were measured on one card.
    """
    sm = _get_target_sm() if sm is None else sm
    note_extrapolation(_MEASURED)
    return select_tile(sm, int(hidden), int(batch), 2, gate_count(kind))


#: Variants known to spill, and why they are shipped anyway.
#:
#: `cuobjdump -res-usage` puts the widest **one-gate** tile (index 7, 32
#: epilogue slots per thread on a 512-thread CTA) at REG:128 with a 32-byte
#: stack frame; every LSTM variant is at REG:54-120 with no spill.  Nothing
#: reaches it -- a vanilla RNN is never routed under ``auto`` and tile 7 needs
#: ``hidden > 1536`` *and* ``batch > 128`` -- so it is declared rather than
#: fixed.  The header comment in ``recurrent_step_epilogue.h`` records the two
#: causes that were tested and refuted, so the next attempt starts past them.
KNOWN_SPILLS: Tuple[Tuple[str, int], ...] = (("rnn_tanh", 7), ("rnn_relu", 7))


def recurrent_step_coverage_report() -> List[str]:
    """What this lane declined, and why.

    Neither a missing kernel (``KERNEL_GAPS``) nor a measured performance
    choice (``take_policy``), so it needs its own channel -- the same shape
    ``oasr.jit.gated_mlp.gated_mlp_coverage_report`` and
    ``oasr.jit.gemm.rule_miss_report`` already use.
    """
    lines: List[str] = []
    if _ARCH_UNDERSERVED:
        lines.append("the fused recurrent step has no C++ kernel for these architectures:")
        for sm, hits in sorted(_ARCH_UNDERSERVED.items()):
            lines.append(
                f"  sm_{sm}: {hits} request(s) -> the CuTeDSL lane, or the CUTLASS GEMM "
                f"plus a finalizer. The Ampere-class collective would run there; it is "
                f"not routed to because nothing has measured it against the alternatives "
                f"on that part."
            )
    if KNOWN_SPILLS:
        lines.append(
            "fused recurrent-step variants that spill (shipped, unrouted): "
            + ", ".join(f"{kind} tile {t}" for kind, t in KNOWN_SPILLS)
        )
    if _REFUSED:
        lines.append("fused recurrent-step configs refused by the C++ lane:")
        for why, hits in sorted(_REFUSED.items()):
            lines.append(f"  {hits}x {why}")
    return lines


def reset_coverage() -> None:
    _ARCH_UNDERSERVED.clear()
    _REFUSED.clear()


def clear_caches() -> None:
    """Drop every memoised answer.  Called by :func:`oasr.jit.recurrent_cute.set_mode`."""
    select_tile.cache_clear()
    gen_recurrent_step_module.cache_clear()
    _cell_module.cache_clear()
    _BUILD_FAILED.clear()
