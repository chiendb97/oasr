# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The C++ CUTLASS/CuTe lane for the fused gated MLP (SwiGLU / GeGLU).

Unlike :mod:`oasr.jit.mlp` -- which wraps ``cutlass.cute.compile()`` for the
CuTeDSL kernel and also arbitrates between the two lanes -- this is the repo's
ordinary C++ JIT pipeline: Jinja renders one translation unit per variant,
ninja builds them into one ``.so`` per *cell*, and the result is cached on
disk in ``~/.cache/oasr/jit``.

:mod:`oasr.jit.mlp` stays the arbiter that answers "which backend?"; this
module is one of the answers, and never imports it back.

Cells and variants
------------------
A **cell** is ``(target_sm, dtype, activation)``.  Inside a cell the
**variants** are

    6 CTA tiles from ``kGatedMlpTiles``  x  2 (bias / no bias)  =  12

and all of them go into that one module, because nvcc parallelises across
translation units but not within one (the reasoning is written out at
``oasr/jit/gemm.py:1000``).  Measured on this box, twelve of these compile in
a single ninja wave on 64 cores -- 28.2 s cold, the same as six -- so a cell
costs about one variant's wall time
-- and then covers every shape, every tile and both bias modes it will ever be
asked for.  The CuTeDSL lane recompiles per *tile*, which is why a decoder that
walks its batch across the tile table pays there and not here.

The **activation** is the one axis that stays a cell, because it is the one
that changes the arithmetic and a checkpoint means exactly one of them.  A
model asks for its own cell once.

Everything else is deliberately **runtime**: ``M``, ``N``, ``K``, all four row
strides, and the K residue.  Each of those would have multiplied the variant
count for nothing.

The tile is compiled but not *chosen* here: ``gatedMlpSelectTile`` is
``constexpr`` in ``include/oasr/mlp/gated_mlp_tiles.h``, and
:func:`select_tile` below only *mirrors* it for routing.  The two are pinned
equal by ``tests/kernels/test_gated_mlp_cpp.py``, against the C++ answer
exported as ``gated_mlp_select_tile`` -- a mirror nobody checks is a mirror
that drifts.
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
    "ACTIVATIONS",
    "SUPPORTED_SM",
    "Tile",
    "TILES",
    "tile_valid",
    "smem_bytes",
    "ctas_per_sm",
    "select_tile",
    "config_supported",
    "gen_gated_mlp_module",
    "gen_gated_mlp_modules",
    "get_gated_mlp_fn",
    "gated_mlp_coverage_report",
    "variant_symbol",
]

#: Architectures this lane compiles for.  sm_90 / sm_100 would *run* the
#: Ampere-class collective -- mma.sync and cp.async are both available there --
#: but routing to it would be a performance claim with no measurement behind
#: it, and the right answer on those parts is a wgmma/TMA mainloop of their
#: own.  The omission is counted in :func:`gated_mlp_coverage_report`, not
#: hidden.  Mirrors the `GatedMlpArch<SM>` specializations in
#: ``include/oasr/mlp/cutlass_gated_mlp_configs.h``.
SUPPORTED_SM: Tuple[int, ...] = (80, 86, 89, 120)

#: The 128-bit vector contract, in elements.  Mirrors ``kGatedMlpAlignment``.
ALIGNMENT = 8

#: Gate activations, by the name ``oasr.layers`` uses, mapped to the
#: ``oasr::ActivationType`` value the kernel is instantiated with.  ``gelu`` is
#: the exact-erf form and ``gelu_tanh`` the tanh approximation; they stay
#: separate names because they are numerically different epilogues and a
#: checkpoint means one of them.  ``swish`` and ``silu`` are the same function
#: and therefore the same cell.
_ACTIVATION_ID: Dict[str, int] = {
    "relu": 0,
    "gelu_tanh": 1,
    "silu": 2,
    "swish": 2,
    "identity": 3,
    "gelu": 4,
}

#: The spelling a cell is named after, so ``silu`` and ``swish`` share one.
_ACTIVATION_CANON: Dict[str, str] = {
    name: ("silu" if ident == 2 else name) for name, ident in _ACTIVATION_ID.items()
}

ACTIVATIONS: Tuple[str, ...] = tuple(sorted(_ACTIVATION_ID))

_DTYPES = {"float16": "cutlass::half_t", "bfloat16": "cutlass::bfloat16_t"}

#: Hardware tables, mirrored once for every C++ CuTe family in
#: :mod:`oasr.jit.arch_facts`.  Re-exported under the names the mirror tests use.
_SMEM_CAPACITY = arch_facts.SMEM_CAPACITY
_MAX_THREADS_PER_SM = arch_facts.MAX_THREADS_PER_SM


@dataclass(frozen=True)
class Tile:
    """One entry of ``kGatedMlpTiles``."""

    block_m: int
    block_n: int
    block_k: int
    stages: int
    threads: int
    warps_n: int


#: Mirrors ``oasr::mlp::kGatedMlpTiles``, **in the same order** -- the order is
#: the preference rank that breaks a wave-count tie, so a permutation here is a
#: silent routing change.  Pinned entry-for-entry by
#: ``tests/kernels/test_gated_mlp_cpp.py``.
TILES: Tuple[Tile, ...] = (
    Tile(16, 64, 64, 4, 128, 4),
    Tile(16, 64, 32, 4, 64, 2),
    Tile(32, 64, 64, 4, 128, 4),
    Tile(32, 64, 32, 4, 64, 2),
    Tile(64, 64, 64, 4, 256, 4),
    Tile(64, 64, 32, 4, 128, 4),
)

#: Where the candidate list came from -- *not* how one is chosen from it.
#:
#: The choice is derived: :func:`select_tile` scores candidates by wave count
#: against the target machine's SM count and its architecture's real opt-in
#: shared memory, so it follows the card it runs on.  What does not travel is
#: which six tiles are in the list at all, and the rank used to break a
#: wave-count tie.  A tile that would win only on a machine with a different
#: smem budget or SM count is not in the list to be scored, and no amount of
#: wave arithmetic can find it.
_MEASURED = MeasuredOn(
    table="include/oasr/mlp/gated_mlp_tiles.h::kGatedMlpTiles",
    machine=Machine(name="NVIDIA GeForce RTX 5090", sm=120, sms=170),
    bandwidth="1792 GB/s",
    source=".artifacts/ (2026-09-21)",
    moves_with="shared-memory budget and SM count -- they decide which rings fit "
    "and how many CTAs are resident, so a different card may want a tile this "
    "list does not contain. The choice *among* these is already derived",
)

_ARCH_UNDERSERVED: "Counter[int]" = Counter()
_REFUSED: "Counter[str]" = Counter()


# ---------------------------------------------------------------------------
# The mirror of `gated_mlp_tiles.h`
# ---------------------------------------------------------------------------


smem_budget = arch_facts.smem_budget


def smem_bytes(tile: Tile, elem_size: int = 2) -> int:
    """Shared memory the kernel requests, in bytes.  Mirrors ``gatedMlpSmemBytes``.

    The ring carries A **and both** Bs, so a stage costs
    ``(m + 2n) * k`` elements -- half again what a plain GEMM's does, which is
    the tuning pressure that makes ``block_k = 32`` a real option here.  The
    epilogue's staging buffer is a union with the ring, not an addition.
    """
    mainloop = tile.stages * (tile.block_m + 2 * tile.block_n) * tile.block_k * elem_size
    epilogue = tile.block_m * tile.block_n * elem_size
    return max(mainloop, epilogue)


def tile_valid(tile: Tile, sm: int, elem_size: int = 2) -> bool:
    """Would this tile compile, fit, and address only its own tiles?

    Mirrors ``gatedMlpTileValid`` clause for clause.  Each one is a *silent*
    failure if dropped: the MMA clauses become an unreadable CuTe layout
    error, the tiled-copy clauses become an illegal access no predicate can
    intercept (the *partition* is out of range), and the shared-memory clause
    becomes a launch failure with an empty message.
    """
    if elem_size != 2:
        return False
    if not arch_facts.sm80_warp_tiling_valid(
        tile.block_m, tile.block_n, tile.threads, tile.warps_n
    ):
        return False
    if tile.block_k % 32 or tile.stages < 2:
        return False
    if not (
        arch_facts.sm80_tiled_copy_fits(tile.block_m, tile.block_k, tile.threads, elem_size)
        and arch_facts.sm80_tiled_copy_fits(tile.block_n, tile.block_k, tile.threads, elem_size)
        and arch_facts.sm80_tiled_copy_fits(tile.block_m, tile.block_n, tile.threads, elem_size)
    ):
        return False
    return arch_facts.fits_sm(smem_bytes(tile, elem_size), tile.threads, sm)


def ctas_per_sm(tile: Tile, sm: int, elem_size: int = 2) -> int:
    """How many of these CTAs are resident at once.  Mirrors ``gatedMlpCtasPerSm``.

    Registers are deliberately not modelled: every tile here measures well
    inside what shared memory already allows (1-2 blocks), so a register bound
    would never bind.  A tile that changed that would show up as a *measured*
    regression, not as a wrong number here.
    """
    return arch_facts.ctas_per_sm(smem_bytes(tile, elem_size), tile.threads, sm)


_ceildiv = arch_facts.ceil_div


def _waves(tile: Tile, sm: int, num_sms: int, rows: int, n: int, elem_size: int) -> int:
    slots = num_sms * ctas_per_sm(tile, sm, elem_size)
    grid = _ceildiv(n, tile.block_n) * _ceildiv(rows, tile.block_m)
    return _ceildiv(grid, slots)


def _group_m(rows: int) -> int:
    """Mirrors ``gatedMlpTileGroupM``: the smallest ``block_m`` that covers ``rows``."""
    largest = max(t.block_m for t in TILES)
    covering = [t.block_m for t in TILES if t.block_m >= rows]
    return min(covering) if covering else largest


@functools.lru_cache(maxsize=None)
def select_tile(sm: int, num_sms: int, rows: int, n: int, elem_size: int = 2) -> int:
    """Index into :data:`TILES` for this problem, or ``-1``.  Mirrors ``gatedMlpSelectTile``.

    Fewest waves wins; ties go to the earlier entry.  ``N`` is in the decision
    and not only ``M`` because the kernel is bandwidth bound, so what decides
    its time is whether the *last* wave still has enough CTAs in flight to
    saturate DRAM.

    A pure function of ``(shape, architecture, SM count)`` and **nothing
    else** -- in particular not of CUDA-graph capture state (``AGENTS.md``
    rule 11): two tiles sum the K loop in different orders, so an answer that
    differed under capture would make a replayed graph produce different
    numbers than eager.
    """
    if rows <= 0 or n <= 0 or num_sms <= 0:
        return -1
    group = _group_m(rows)
    for pool in (
        [i for i, t in enumerate(TILES) if t.block_m == group],
        list(range(len(TILES))),
    ):
        best = -1
        best_waves = 0
        for i in pool:
            if not tile_valid(TILES[i], sm, elem_size):
                continue
            w = _waves(TILES[i], sm, num_sms, rows, n, elem_size)
            if best < 0 or w < best_waves:
                best, best_waves = i, w
        if best >= 0:
            return best
    return -1


def shape_supported(*, rows: int, n: int, k: int) -> bool:
    """Does this problem meet the kernel's static contract?

    Only the 128-bit vector contract on the two contiguous extents.  ``K`` is
    **not** required to be a whole number of K tiles: the residue is predicated
    and the ZFILL cp.async makes the skipped elements zero, which is the
    identity for the dot product.  That is the constraint the CuTeDSL lane has
    to impose and this one does not (``oasr.jit.mlp.gated_mlp_shape_supported``).
    """
    return rows > 0 and n > 0 and k > 0 and n % ALIGNMENT == 0 and k % ALIGNMENT == 0


def config_supported(
    *, sm: int, dtype_str: str, activation: str, rows: int, n: int, k: int
) -> bool:
    """Can this lane serve the shape?  Asked before anything is built.

    Answering only "does a tile fit" would route a caller to a symbol that
    does not exist, so the activation and the shape contract are checked here
    too -- either "no" is a "no".
    """
    if dtype_str not in _DTYPES:
        return False
    if activation not in _ACTIVATION_ID:
        _REFUSED[f"activation {activation!r}"] += 1
        return False
    if sm not in SUPPORTED_SM:
        if sm in _SMEM_CAPACITY:
            _ARCH_UNDERSERVED[sm] += 1
        return False
    if not shape_supported(rows=rows, n=n, k=k):
        _REFUSED["N or K is not a multiple of the 128-bit vector width"] += 1
        return False
    if not any(tile_valid(t, sm) for t in TILES):
        _REFUSED[f"no tile fits sm_{sm}'s {smem_budget(sm)} B of shared memory"] += 1
        return False
    return True


# ---------------------------------------------------------------------------
# Rendering and building
# ---------------------------------------------------------------------------


def variant_symbol(
    sm: int, dtype_str: str, activation: str, has_bias: bool, tile_index: int
) -> str:
    """The exported name of one variant.

    Produced by exactly one function so the renderer and the loader cannot
    disagree about it.
    """
    act = _ACTIVATION_CANON[activation]
    return (
        f"gated_mlp_sm{sm}_{dtype_str}_{act}" f"_{'bias' if has_bias else 'nobias'}_t{tile_index}"
    )


def cell_name(sm: int, dtype_str: str, activation: str) -> str:
    return f"gated_mlp_sm{sm}_{dtype_str}_{_ACTIVATION_CANON[activation]}"


@functools.lru_cache(maxsize=None)
def gen_gated_mlp_module(sm: int, dtype_str: str, activation: str) -> JitSpec:
    """Render and describe one cell's translation units -- one per (tile, bias)."""
    if dtype_str not in _DTYPES:
        raise ValueError(f"gated_mlp: unsupported dtype {dtype_str!r}")
    if activation not in _ACTIVATION_ID:
        raise ValueError(f"gated_mlp: unsupported activation {activation!r}")
    if not any(tile_valid(t, sm) for t in TILES):
        raise RuntimeError(
            f"gated_mlp: no tile fits sm_{sm}'s {smem_budget(sm)} B of shared memory"
        )
    name = cell_name(sm, dtype_str, activation)
    out_dir = env.OASR_GEN_SRC_DIR / "gated_mlp"
    sources: List = []
    for index, tile in enumerate(TILES):
        if not tile_valid(tile, sm):
            # Rendering it would fail the launcher's static_assert at compile
            # time and take the whole cell with it.  `select_tile` cannot
            # return an index that is not here, because it applies the same
            # predicate.
            continue
        for has_bias in (False, True):
            func = variant_symbol(sm, dtype_str, activation, has_bias, index)
            rendered = render_template(
                "gated_mlp_template.cu.jinja",
                func_name=func,
                sm_version=sm,
                dtype=dtype_str,
                cutlass_dtype=_DTYPES[dtype_str],
                activation=activation,
                activation_id=_ACTIVATION_ID[activation],
                has_bias=has_bias,
                tile_index=index,
                tile=tile,
            )
            path = out_dir / f"{func}.cu"
            write_if_different(path, rendered)
            sources.append(path)
    sources.append(env.OASR_CSRC_DIR / "gated_mlp_jit_binding.cu")
    return gen_jit_spec(name, sources)


def gen_gated_mlp_modules(
    cells: Optional[List[Tuple[str, str]]] = None,
) -> List[JitSpec]:
    """Specs for the AOT set: the ``(dtype, activation)`` cells shipped models reach.

    SwiGLU is what every gated feed-forward block in scope uses -- Qwen2-Audio's
    LM and the Nemotron encoder both -- so the default set is that, in both
    served dtypes, and each cell already carries both bias modes and all six
    tiles.  A model with another activation compiles its cell on first use,
    which costs one ninja wave.
    """
    if cells is None:
        cells = [(dt, "silu") for dt in ("float16", "bfloat16")]
    sm = _get_target_sm()
    if sm not in SUPPORTED_SM:
        return []
    return [gen_gated_mlp_module(sm, dt, act) for dt, act in cells]


@functools.lru_cache(maxsize=None)
def _cell_module(sm: int, dtype_str: str, activation: str):
    return gen_gated_mlp_module(sm, dtype_str, activation).build_and_load()


#: Cells whose build has already failed once.
_BUILD_FAILED: set = set()


def _load_cell(sm: int, dtype_str: str, activation: str):
    """:func:`_cell_module`, with build *failures* remembered.

    ``functools.lru_cache`` does not memoise exceptions, so a cell that cannot
    build -- no ninja, no nvcc, a toolchain mismatch -- would re-invoke the
    whole build for every new shape key the router asks about.  A build failure
    is a property of the configuration, not of the call.
    """
    key = (sm, dtype_str, activation)
    if key in _BUILD_FAILED:
        raise RuntimeError(f"gated_mlp: cell {cell_name(*key)} failed to build earlier")
    try:
        return _cell_module(*key)
    except Exception:
        _BUILD_FAILED.add(key)
        _REFUSED[f"cell {cell_name(*key)} failed to build"] += 1
        raise


def get_gated_mlp_fn(
    *,
    dtype_str: str,
    activation: str,
    has_bias: bool,
    tile_index: int,
    sm: Optional[int] = None,
):
    """The bound launcher for one variant, building its cell on first use."""
    sm = _get_target_sm() if sm is None else sm
    if not (0 <= tile_index < len(TILES)):
        raise ValueError(f"gated_mlp: no tile {tile_index}")
    module = _load_cell(sm, dtype_str, activation)
    return getattr(module, variant_symbol(sm, dtype_str, activation, has_bias, tile_index))


def routed_tile(*, rows: int, n: int, sm: Optional[int] = None) -> int:
    """The tile this machine would pick for the shape, or ``-1``.

    Reads the SM count from the device -- the one query the mirror cannot
    avoid, and the reason it is a *parameter* of :func:`select_tile` rather
    than a lookup inside it.
    """
    import torch

    sm = _get_target_sm() if sm is None else sm
    note_extrapolation(_MEASURED)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return select_tile(sm, int(num_sms), int(rows), int(n))


def gated_mlp_coverage_report() -> List[str]:
    """What this lane declined, and why.

    Neither a missing kernel (``KERNEL_GAPS``) nor a measured performance
    choice (``take_policy``), so it needs its own channel -- the same shape
    ``oasr.jit.fmha.fmha_coverage_report`` and ``oasr.jit.gemm.rule_miss_report``
    already use.
    """
    lines: List[str] = []
    if _ARCH_UNDERSERVED:
        lines.append("the fused gated MLP has no C++ kernel for these architectures:")
        for sm, hits in sorted(_ARCH_UNDERSERVED.items()):
            lines.append(
                f"  sm_{sm}: {hits} request(s) -> the CuTeDSL lane or two GEMMs. The "
                f"Ampere-class collective would run there; it is not routed to because "
                f"nothing has measured it against the alternatives on that part."
            )
    if _REFUSED:
        lines.append("fused gated-MLP configs refused by the C++ lane:")
        for why, hits in sorted(_REFUSED.items()):
            lines.append(f"  {hits}x {why}")
    return lines


def reset_coverage() -> None:
    _ARCH_UNDERSERVED.clear()
    _REFUSED.clear()


def clear_caches() -> None:
    """Drop every memoised answer.  Called by :func:`oasr.jit.mlp.set_gated_mlp_mode`."""
    select_tile.cache_clear()
    gen_gated_mlp_module.cache_clear()
    _cell_module.cache_clear()
    _BUILD_FAILED.clear()
