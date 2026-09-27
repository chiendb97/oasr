# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Python mirror of ``include/oasr/common/arch_facts.h`` and ``tile_rules.h``.

The C++ CUTLASS/CuTe kernel families -- attention (:mod:`oasr.jit.fmha`), the
gated MLP (:mod:`oasr.jit.gated_mlp`) and the recurrent step
(:mod:`oasr.jit.recurrent_step`) -- each answer "can you serve this shape, and
at what occupancy?" in Python before anything is built, by mirroring their C++
tile arithmetic.  The hardware facts and the tile rules underneath that
arithmetic are the same for all three, so they are mirrored once, here, the
same way the C++ states them once in ``include/oasr/common/``.

Tables, not device queries: the capability surface is parametrised over six
architectures in the tests, so one box holds the line for all of them, and a
live query would collapse that to whatever card is present -- which is
precisely the coverage that caught sm_86/sm_89 being budgeted with A100's
163 KB (``.artifacts/arch_portability_audit.md`` § A3).  Each family's mirror
test checks its own answers against its C++ capability oracle, so these tables
are checked through every one of them.
"""

from __future__ import annotations

from typing import Dict

__all__ = [
    "SMEM_CAPACITY",
    "MAX_THREADS_PER_SM",
    "DRIVER_SMEM_RESERVE",
    "MAX_BLOCKS_PER_SM",
    "VECTOR_BYTES",
    "ceil_div",
    "smem_budget",
    "smem_swizzle_row_width",
    "sm80_warp_tiling_valid",
    "sm80_tiled_copy_fits",
    "ctas_per_sm",
    "fits_sm",
]

#: Opt-in shared memory per block, in bytes.  Mirrors ``oasr::smemCapacityForSm``.
SMEM_CAPACITY: Dict[int, int] = {
    80: 166912,
    86: 101376,
    89: 101376,
    90: 232448,
    100: 232448,
    120: 101376,
}

#: Resident threads per SM.  Mirrors ``oasr::maxThreadsPerSmForSm``.
MAX_THREADS_PER_SM: Dict[int, int] = {
    80: 2048,
    86: 1536,
    89: 1536,
    90: 2048,
    100: 2048,
    120: 1536,
}

#: Mirrors ``oasr::kDriverSmemReserve``.
DRIVER_SMEM_RESERVE = 1024

#: Mirrors ``oasr::kTileMaxBlocksPerSm``.
MAX_BLOCKS_PER_SM = 24

#: Mirrors ``oasr::kTileVectorBytes``: one ``cp.async.cg`` / vector store.
VECTOR_BYTES = 16


def ceil_div(a: int, b: int) -> int:
    """Mirrors ``oasr::tileCeilDiv``."""
    return -(-a // b)


def smem_budget(sm: int) -> int:
    """Shared memory a launch on ``sm`` is granted.  Mirrors ``oasr::smemBudgetForSm``."""
    cap = SMEM_CAPACITY.get(sm, 0)
    return cap - DRIVER_SMEM_RESERVE if cap else 0


def smem_swizzle_row_width(extent: int, elem_size: int) -> int:
    """Elements per swizzled smem row.  Mirrors ``oasr::smemSwizzleRowWidth``."""
    nbytes = extent * elem_size
    b = 128 if nbytes % 128 == 0 else (64 if nbytes % 64 == 0 else 32)
    return b // elem_size


def sm80_warp_tiling_valid(block_m: int, block_n: int, threads: int, warps_n: int) -> bool:
    """Mirrors ``oasr::sm80WarpTilingValid``."""
    if threads % 32 or threads < 32 or threads > 1024:
        return False
    warps = threads // 32
    if warps_n < 1 or warps % warps_n:
        return False
    warps_m = warps // warps_n
    return block_m % (16 * warps_m) == 0 and block_n % (16 * warps_n) == 0


def _tiled_copy_rows_per_pass(extent: int, threads: int, elem_size: int) -> int:
    """Mirrors ``oasr::sm80TiledCopyRowsPerPass``."""
    vec = VECTOR_BYTES // elem_size
    row = smem_swizzle_row_width(extent, elem_size)
    threads_per_row = row // vec
    if threads_per_row <= 0 or threads % threads_per_row or extent % row:
        return 0
    return threads // threads_per_row


def sm80_tiled_copy_fits(rows: int, extent: int, threads: int, elem_size: int) -> bool:
    """Mirrors ``oasr::sm80TiledCopyFits``: whole passes only, or the partition overruns."""
    rows_per_pass = _tiled_copy_rows_per_pass(extent, threads, elem_size)
    return rows_per_pass > 0 and rows % rows_per_pass == 0


def ctas_per_sm(smem_bytes: int, threads: int, sm: int) -> int:
    """Resident CTAs per SM by shared memory and warps.  Mirrors ``oasr::tileCtasPerSm``."""
    budget = smem_budget(sm)
    by_smem = budget // smem_bytes if smem_bytes > 0 else MAX_BLOCKS_PER_SM
    by_threads = MAX_THREADS_PER_SM.get(sm, 0) // threads if threads > 0 else 0
    return max(1, min(by_smem, by_threads, MAX_BLOCKS_PER_SM))


def fits_sm(smem_bytes: int, threads: int, sm: int) -> bool:
    """Can ``sm`` host one such CTA at all?  Mirrors ``oasr::tileFitsSm``."""
    budget = smem_budget(sm)
    return budget > 0 and MAX_THREADS_PER_SM.get(sm, 0) >= threads and smem_bytes <= budget
