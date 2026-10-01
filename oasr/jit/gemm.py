# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for GEMM kernels (FlashInfer-style).

Tile configurations are defined here in the JIT layer, and ALL variants are
compiled into a single shared library per kernel family.  The autotuner
selects which pre-compiled variant to call — no JIT during tuning.
"""

import contextlib
import itertools
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from . import env
from .core import _TARGET_SMS, JitSpec, _get_target_sm, gen_jit_spec

logger = logging.getLogger("oasr.jit.gemm")

# =============================================================================
# Tile configuration helpers (SM<90)
# =============================================================================


@dataclass(frozen=True)
class TileShape:
    """A CUTLASS tile configuration for GEMM or Conv2D (SM<90)."""

    block_m: int
    block_n: int
    block_k: int
    warp_m: int
    warp_n: int
    warp_k: int


@dataclass(frozen=True)
class TileShapeSm90:
    """Legacy SM90 tile shape; retained for external callers."""

    BM: int
    BN: int
    BK: int


@dataclass(frozen=True)
class ClusterShape:
    """Legacy cluster shape; retained for external callers."""

    CM: int
    CN: int
    CK: int


# =============================================================================
# Config dataclasses
# =============================================================================


@dataclass(frozen=True)
class CutlassGemmConfig:
    """A CUTLASS GEMM configuration for SM<90 (CUTLASS 2.x).

    ``kStages`` and ``split_k`` are both tunable:
      - ``kStages`` is a compile-time template parameter; different values
        produce distinct compiled variants and are encoded in ``compile_name``.
      - ``split_k`` is a runtime argument passed to the launcher; it is NOT
        encoded in ``compile_name`` (same binary serves all split-k factors)
        but IS included in ``name`` and ``to_tactic_config`` so the autotuner
        can explore and cache results per split-k value.
      - ``parallel_split_k`` selects the ``GemmSplitKParallel`` decomposition
        (compile-time): fp32 partials + a reduction kernel that applies the
        epilogue once.  Valid for fused activations (unlike serial split-K,
        which nests the activation per K-partition and is rejected by the
        kernel); the runtime ``split_k`` factor must be > 1.
    """

    block_m: int
    block_n: int
    block_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    kStages: int
    kSmVersion: int
    split_k: int = 1  # runtime split-K factor (1 = disabled)
    stream_k: bool = False  # Stream-K decomposition (compile-time; thin-GEMM fill)
    parallel_split_k: bool = False  # GemmSplitKParallel (compile-time; deep-K thin fill)

    @property
    def name(self) -> str:
        """Unique identifier for this config (includes all params, split_k)."""
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.block_m}x{self.block_n}x{self.block_k}")
        parts.append(f"w{self.warp_m}x{self.warp_n}x{self.warp_k}")
        parts.append(f"s{self.kStages}")
        if self.split_k != 1:
            parts.append(f"spk{self.split_k}")
        if self.stream_k:
            parts.append("sk")
        if self.parallel_split_k:
            parts.append("pk")
        return "_".join(parts)

    @property
    def compile_name(self) -> str:
        """Config name used as the compiled binary key.

        Encodes tile shape, warp shape, kStages, and the Stream-K / parallel
        split-K flags (all compile-time template parameters).  Excludes
        ``split_k`` (runtime argument) so variants differing only in split-K
        share a single ``.so``.
        """
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.block_m}x{self.block_n}x{self.block_k}")
        parts.append(f"w{self.warp_m}x{self.warp_n}x{self.warp_k}")
        parts.append(f"s{self.kStages}")
        if self.stream_k:
            parts.append("sk")
        if self.parallel_split_k:
            parts.append("pk")
        return "_".join(parts)

    @property
    def num_warps(self) -> int:
        return (self.block_m // self.warp_m) * (self.block_n // self.warp_n)

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        """Convert to a ``Tactic.config`` tuple."""
        items = [
            ("block_m", self.block_m),
            ("block_n", self.block_n),
            ("block_k", self.block_k),
            ("warp_m", self.warp_m),
            ("warp_n", self.warp_n),
            ("warp_k", self.warp_k),
            ("kStages", self.kStages),
            ("split_k", self.split_k),
            ("stream_k", int(self.stream_k)),
            ("parallel_split_k", int(self.parallel_split_k)),
        ]
        return tuple(items)


@dataclass(frozen=True)
class CutlassGemmConfigSm90:
    """Quack-aligned CUTLASS GEMM configuration for SM90, SM100, and SM120.

    Field mapping vs. old per-SM config lists:
      ``tile_m`` / ``tile_n`` / ``tile_k``  —  BM / BN / BK
      ``cluster_m`` / ``cluster_n``          —  CM / CN  (CK is always 1)
      ``kSMs``                               —  1 or 2 (SM100 co-operative)
      ``pingpong``                           —  True → Pingpong schedule (SM90/SM120)
                                               False → Cooperative schedule
      ``is_dynamic_persistent``              —  CLC / dynamic tile scheduler (SM100)
      ``swap_ab``                            —  Swap A / B for memory-access optimisation
      ``max_swizzle_size``                   —  Shared-memory swizzle bound
      ``use_tma_gather``                     —  TMA gather for A (SM100 only)
    """

    tile_m: int
    tile_n: int
    tile_k: int  # 128 for SM90/SM120 (WGMMA width)
    cluster_m: int
    cluster_n: int
    pingpong: bool  # True = Pingpong, False = Cooperative (SM90/SM120)
    is_dynamic_persistent: bool  # Dynamic persistent / CLC scheduler (SM100)
    swap_ab: bool  # Swap A and B operands
    max_swizzle_size: int  # Max swizzle size for SMEM layout
    use_tma_gather: bool  # TMA gather for A (SM100 only)
    kSMs: int  # 1 or 2 (SM100 only; always 1 for SM90/SM120)
    kStages: int  # Pipeline stages (typically 3)
    kSmVersion: int  # 90, 100, or 120

    @property
    def name(self) -> str:
        """Unique identifier for this config (includes all distinguishing params)."""
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.tile_m}x{self.tile_n}x{self.tile_k}")
        parts.append(f"c{self.cluster_m}x{self.cluster_n}")
        parts.append(f"k{self.kSMs}")
        parts.append(f"s{self.kStages}")
        parts.append("pp" if self.pingpong else "coop")
        if self.swap_ab:
            parts.append("swapab")
        return "_".join(parts)

    @property
    def compile_name(self) -> str:
        """Config name used as the compiled binary key.

        Includes only parameters that affect C++ compilation (tile shape,
        cluster shape, kSMs, kStages, pingpong schedule).  Pure runtime
        parameters (swap_ab, is_dynamic_persistent, max_swizzle_size,
        use_tma_gather) are excluded so that variants differing only in those
        fields share a single compiled ``.so``.
        """
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.tile_m}x{self.tile_n}x{self.tile_k}")
        parts.append(f"c{self.cluster_m}x{self.cluster_n}")
        parts.append(f"k{self.kSMs}")
        parts.append(f"s{self.kStages}")
        parts.append("pp" if self.pingpong else "coop")
        return "_".join(parts)

    @property
    def num_warps(self) -> int:
        # Approximate: SM90 WGMMA uses 4 warps per 64×64 tile
        return max(1, (self.tile_m // 64) * (self.tile_n // 64)) * 4

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        """Convert to a ``Tactic.config`` tuple."""
        items = [
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
            ("cluster_m", self.cluster_m),
            ("cluster_n", self.cluster_n),
            ("pingpong", int(self.pingpong)),
            ("is_dynamic_persistent", int(self.is_dynamic_persistent)),
            ("swap_ab", int(self.swap_ab)),
            ("kSMs", self.kSMs),
            ("kStages", self.kStages),
        ]
        return tuple(items)


# =============================================================================
# SM<90 tile configurations (CUTLASS 2.x TensorOp)
# =============================================================================

# Retained for backward compatibility (conv.py and external callers).
# Internal config generation uses the per-SM functions below.
#
# ``block_n=16`` entries are declared but never built: CUTLASS's tensor-op
# epilogue cannot address them at half precision, and
# ``_epilogue_covers_warp`` drops them (see that function).  They are left in
# the list rather than deleted so the candidate set stays a record of what was
# considered, and so the filter that rejects them is exercised by the shipped
# configuration instead of only by a synthetic test.
TileShapeConfigs: List[TileShape] = [
    TileShape(block_m=16, block_n=128, block_k=64, warp_m=16, warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=16, block_k=64, warp_m=32, warp_n=16, warp_k=64),
    TileShape(block_m=32, block_n=128, block_k=64, warp_m=32, warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=32, block_k=64, warp_m=32, warp_n=32, warp_k=64),
    TileShape(block_m=64, block_n=128, block_k=64, warp_m=32, warp_n=64, warp_k=64),
    TileShape(block_m=128, block_n=64, block_k=64, warp_m=64, warp_n=32, warp_k=64),
    TileShape(block_m=64, block_n=128, block_k=64, warp_m=64, warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=64, block_k=64, warp_m=64, warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=128, block_k=64, warp_m=64, warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=128, block_k=64, warp_m=64, warp_n=64, warp_k=64),
    TileShape(block_m=128, block_n=128, block_k=64, warp_m=128, warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=256, block_k=64, warp_m=64, warp_n=64, warp_k=64),
    TileShape(block_m=256, block_n=128, block_k=64, warp_m=64, warp_n=64, warp_k=64),
    TileShape(block_m=16, block_n=256, block_k=64, warp_m=16, warp_n=64, warp_k=64),
    TileShape(block_m=256, block_n=16, block_k=64, warp_m=64, warp_n=16, warp_k=64),
]


# Extra thin-N tiles for the GEMM families (not shared with conv2d, which
# imports TileShapeConfigs above).  ASR encoder GEMMs are output-thin
# (N=256/512): block_n=64 quadruples the column-tile count vs the 128/256-wide
# tiles, which — combined with split-K — is what lets CUTLASS fill the GPU on
# small-M shapes where cuBLAS's bespoke thin kernels used to win.
GemmExtraTileConfigs: List[TileShape] = [
    TileShape(block_m=16, block_n=64, block_k=64, warp_m=16, warp_n=32, warp_k=64),
    TileShape(block_m=32, block_n=64, block_k=64, warp_m=16, warp_n=32, warp_k=64),
    TileShape(block_m=64, block_n=64, block_k=64, warp_m=32, warp_n=32, warp_k=64),
]

# =============================================================================
# SM<90 config generation — SMEM-analysed per-SM tile × warp × stage × split_k
# =============================================================================


def _smem_bytes(BM: int, BN: int, BK: int, kStages: int, dtype_bytes: int = 2) -> int:
    """Shared-memory footprint for a CUTLASS 2.x software-pipelined GEMM.

    Each pipeline stage holds one A tile (BM×BK) and one B tile (BN×BK) in the
    operand dtype.  The float32 accumulator lives in registers and is not counted.
    """
    return kStages * (BM + BN) * BK * dtype_bytes


# CUTLASS's tensor-op epilogue writes the accumulator in units of 8 output rows,
# one warp of 32 lanes at a time.  Both numbers are ``kTensorOpRows`` /
# ``kWarpSize`` in
# ``cutlass/epilogue/threadblock/default_thread_map_tensor_op.h``.
_EPILOGUE_TENSOR_OP_ROWS = 8
_WARP_SIZE = 32

#: Tiles a config builder refused, ``{tile: reason}``.  A rejected tile is not a
#: runtime gap — it is a static property of the lists above — but it is the
#: reason a tile that looks tuneable never appears in the autotuner's candidate
#: set, so it is recorded rather than dropped silently.  Read by
#: :func:`rejected_tiles`.
_REJECTED_TILES: Dict[str, str] = {}


def rejected_tiles() -> Dict[str, str]:
    """``{tile: reason}`` for every tile a config builder has refused so far.

    Populated lazily as the per-SM builders run, so call a builder (or
    :func:`get_all_autotune_configs`) first.
    """
    return dict(_REJECTED_TILES)


def _epilogue_output_map(tile: TileShape, element_bits: int = 16) -> Tuple[int, int, int, int]:
    """``(kAccessWidth, kAccessRows, kIterationsColumn, kIterationsRow)`` for *tile*.

    Mirrors ``DefaultThreadMapTensorOp`` → ``OutputTileOptimalThreadMap`` →
    ``detail::RowArrangement`` for the tensor-op epilogue that every CUTLASS 2.x
    variant in this file instantiates.  ``element_bits`` is the epilogue's
    ``ElementC``; the templates take ``kElementsPerAccess = 128 / element_bits``,
    which is 8 for the fp16/bf16 this path serves (fp32 never reaches it).

    ``kAccessRows × kAccessWidth`` is the grid one warp's 32 lanes are folded
    into — lane *l* writes row ``l / kAccessWidth``, column ``(l %
    kAccessWidth) * kElementsPerAccess`` of its own slot — and the two iteration
    counts are how many such accesses each lane makes.
    """
    epa = 128 // element_bits
    partitions_k = max(1, tile.block_k // tile.warp_k)
    warps_m = tile.block_m // tile.warp_m
    warps_n = tile.block_n // tile.warp_n
    warp_count = warps_m * warps_n * partitions_k
    shape_width = tile.block_n // epa
    # Shape::kGroup is the M-warp count; Shape::kCluster is 1 for this epilogue,
    # so the warps left over for rows are whatever the group split does not use.
    warps_for_rows = 1 if warps_m > warp_count else warp_count // warps_m
    if _EPILOGUE_TENSOR_OP_ROWS <= warps_for_rows:
        # RowArrangement's 1-D specialisation: one row, the whole warp along N.
        return _WARP_SIZE, 1, shape_width // _WARP_SIZE, 1
    shape_row = _EPILOGUE_TENSOR_OP_ROWS // warps_for_rows
    target_width = 256 // (epa * element_bits // 8)  # kTargetMemoryAccessWidth
    if _WARP_SIZE // target_width > shape_row:
        width, rows = _WARP_SIZE // shape_row, shape_row
    else:
        width = min(shape_width, _WARP_SIZE, target_width)
        rows = min(_EPILOGUE_TENSOR_OP_ROWS, _WARP_SIZE // width)
    return width, rows, shape_width // width, shape_row // rows


def _epilogue_covers_warp(tile: TileShape, element_bits: int = 16) -> bool:
    """Whether CUTLASS's epilogue can actually address *tile*'s output.

    Two ways it cannot, and ``RowArrangement`` asserts neither:

    * the ``kAccessRows × kAccessWidth`` lane grid is narrower than a warp, so
      the surplus lanes address rows outside their own slot.  At
      ``kElementsPerAccess = 8`` that is ``block_n < 32``: ``kAccessWidth``
      saturates at ``block_n / 8`` while ``kAccessRows`` saturates at 8.
    * an iteration count comes out zero, so a lane stores nothing (reachable
      only on the 1-D arrangement, which needs ``block_n >= 256`` to have a
      column iteration at all).

    A tile that fails still compiles, still launches and still writes every
    output row, and the values it writes are **wrong**: measured 100-170x the
    reference at ``(M, N, K) = (1710, 128, 384)`` for ``b128x16x64_w32x16x64``,
    in fp16 and bf16, in the GEMM, BMM, grouped-GEMM and Conv2D families alike.

    That is why this is a config-space constraint rather than a note: an
    unsatisfiable tile that *runs* cannot be caught by dispatch ("is the config
    compiled?") or by timing ("which config is fastest?"), which is what let
    ``block_n=16`` sit in the tuned rule table.  One ``block_n=16`` rule reached
    production and returned an **empty transcript** for any 1.1-2.2 s utterance
    decoded on its own (Zipformer's ConvNeXt pointwise contraction, N=128 K=384,
    at ``max_batch_size=1``): the wrong values overflowed fp16 to ``inf`` and the
    CTC head produced all-NaN log-probs.
    """
    width, rows, iters_col, iters_row = _epilogue_output_map(tile, element_bits)
    return width * rows == _WARP_SIZE and iters_col >= 1 and iters_row >= 1


def _tile_is_buildable(tile: TileShape, kStages: int, smem_limit: int) -> bool:
    """Whether *tile* can be instantiated at *kStages*: SMEM fits, epilogue works."""
    if _smem_bytes(tile.block_m, tile.block_n, tile.block_k, kStages) > smem_limit:
        return False
    if not _epilogue_covers_warp(tile):
        width, rows, iters_col, iters_row = _epilogue_output_map(tile)
        _REJECTED_TILES[
            f"b{tile.block_m}x{tile.block_n}x{tile.block_k}_"
            f"w{tile.warp_m}x{tile.warp_n}x{tile.warp_k}"
        ] = (
            f"CUTLASS's tensor-op epilogue cannot address block_n="
            f"{tile.block_n} at kElementsPerAccess=8: lane grid "
            f"{rows}x{width} = {rows * width} of {_WARP_SIZE} lanes, "
            f"iterations {iters_row}x{iters_col}; the kernel runs and returns "
            f"wrong results"
        )
        return False
    return True


# Maximum shared memory per threadblock per architecture (bytes).
# CUTLASS 2.x opts in to the maximum via cudaFuncSetAttribute at runtime.
_SM_MAX_SMEM_BYTES: Dict[int, int] = {
    75: 64 * 1024,  # Turing
    80: 164 * 1024,  # Ampere A100
    86: 100 * 1024,  # Ampere RTX 30-series
    89: 100 * 1024,  # Ada Lovelace
    # SM120 (GeForce Blackwell / RTX 50 series) — CUTLASS 3.x SM120 TMA builder
    # is F8F6F4-only, so FP16/BF16 GEMM on SM120 falls back to the CUTLASS 2.x
    # path using the Sm80 tensor-op specialisations (see CutlassArch<120>).
    # Use the Sm80 shared-memory budget for stage calculations.
    120: 100 * 1024,
}


def _build_sm_lt90_configs(
    sm: int,
    tiles: List[TileShape],
    stage_list: List[int],
    split_k_list: List[int],
    smem_limit: int,
) -> Dict[str, CutlassGemmConfig]:
    """Build the full autotune config dict for a SM<90 architecture.

    Iterates over the provided ``tiles`` (``TileShape`` instances from
    ``TileShapeConfigs``), expanding across ``stage_list`` and ``split_k_list``.
    Four constraints are applied:

    1. **SMEM fit** — kStages×(block_m+block_n)×block_k×dtype_bytes ≤ smem_limit.
       Software-pipelined operand buffers must fit in shared memory.
    2. **Epilogue expressible** — :func:`_epilogue_covers_warp`.  A tile that
       fails this compiles and runs and is *wrong*, so it must never enter the
       space; rejections are recorded in :func:`rejected_tiles`.
    3. **split_k applicability** — split_k>1 is only registered when
       block_m≤128 and block_n≤128 (shapes likely to be K-bound).
    4. **deep split_k** — split_k>4 only for block_m≤64 tiles (deep K-splits
       exist to fill the GPU on small-M shapes; large-M tiles never need them).

    Divisibility and warp-count validity are guaranteed by ``TileShapeConfigs``.

    The returned dict is keyed by ``CutlassGemmConfig.name`` (which includes
    split_k) for use in autotuner registration.  Callers that need only the
    compiled-binary set should deduplicate by ``compile_name``.
    """
    seen: Dict[str, CutlassGemmConfig] = {}
    for tile in tiles:
        for kStages in stage_list:
            # 1. SMEM fit + 2. epilogue expressible
            if not _tile_is_buildable(tile, kStages, smem_limit):
                continue
            for split_k in split_k_list:
                # 3. split_k applicability
                if split_k > 1 and (tile.block_m > 128 or tile.block_n > 128):
                    continue
                # 4. deep split_k only for small-M tiles
                if split_k > 4 and tile.block_m > 64:
                    continue
                cfg = CutlassGemmConfig(
                    block_m=tile.block_m,
                    block_n=tile.block_n,
                    block_k=tile.block_k,
                    warp_m=tile.warp_m,
                    warp_n=tile.warp_n,
                    warp_k=tile.warp_k,
                    kStages=kStages,
                    kSmVersion=sm,
                    split_k=split_k,
                )
                key = cfg.name
                if key not in seen:
                    seen[key] = cfg
    return seen


# Tiles used by the GEMM families on the CUTLASS 2.x path: the conv2d-shared
# base set plus the thin-N extras.
_GEMM_TILES: List[TileShape] = TileShapeConfigs + GemmExtraTileConfigs

# Runtime split-K ladder.  Deep factors ({8, 16}) matter on small-M deep-K
# shapes (one M-tile row, few output tiles); the applicability constraints in
# ``_build_sm_lt90_configs`` confine them to block_m ≤ 64 tiles.
_SPLIT_K_LIST = [1, 2, 4, 8, 16]


# =============================================================================
# K-decompositions for the CUTLASS 2.x lane (sm_75 / 80 / 86 / 89 / 120)
#
# These lived below, among the Quack-style SM90+ builders, which is where the
# sm_120-only reachability came from: they read as an SM120 detail because they
# were filed as one.  They belong to the 2.x lane, so they sit in it.
# =============================================================================

# Stream-K variants are part of the autotune candidate space by default, so
# ``oasr.autotune()`` can select them where they win — e.g. deep-K thin GEMMs, or
# other models / GPUs where the data-parallel grid starves the SMs.  Set
# OASR_GEMM_STREAMK=0 for a leaner production build (skips compiling them).
#
# The knob is sharper than it looks: ``_GEMM_HEURISTIC_RULES_SM120`` *does* name
# Stream-K and parallel split-K configs — the sweep that produced the current
# table found them winning on the deep-K thin shapes, which is not what the
# earlier comment here said.  Turning either knob off therefore leaves a rule
# pointing at a variant that was not compiled; ``_plan`` catches the resulting
# ``AttributeError`` and degrades to GEMM_DEFAULT with one warning per shape, so
# it is a slowdown rather than a failure — but it is not the no-op "they remain
# tunable" implied.
_STREAMK_ENABLED = os.environ.get("OASR_GEMM_STREAMK", "1") != "0"

# Curated tile set for Stream-K variants.  Stream-K helps when there are too few
# output tiles to fill the GPU (small M, N=256, large K), so we cover small
# block_m tiles plus a couple of large tiles for the single-output-tile case.
_STREAMK_TILES: List[TileShape] = [
    TileShape(block_m=16, block_n=128, block_k=64, warp_m=16, warp_n=32, warp_k=64),
    TileShape(block_m=32, block_n=128, block_k=64, warp_m=32, warp_n=32, warp_k=64),
    TileShape(block_m=64, block_n=128, block_k=64, warp_m=32, warp_n=64, warp_k=64),
    TileShape(block_m=128, block_n=128, block_k=64, warp_m=64, warp_n=64, warp_k=64),
    TileShape(block_m=128, block_n=256, block_k=64, warp_m=64, warp_n=64, warp_k=64),
]


def _build_streamk_configs(
    sm: int, tiles: List[TileShape], stage_list: List[int], smem_limit: int
) -> Dict[str, CutlassGemmConfig]:
    """Build Stream-K GEMM configs (split_k=1; the swizzle balances K across SMs)."""
    seen: Dict[str, CutlassGemmConfig] = {}
    for tile in tiles:
        for kStages in stage_list:
            if not _tile_is_buildable(tile, kStages, smem_limit):
                continue
            cfg = CutlassGemmConfig(
                block_m=tile.block_m,
                block_n=tile.block_n,
                block_k=tile.block_k,
                warp_m=tile.warp_m,
                warp_n=tile.warp_n,
                warp_k=tile.warp_k,
                kStages=kStages,
                kSmVersion=sm,
                split_k=1,
                stream_k=True,
            )
            seen[cfg.name] = cfg
    return seen


# Parallel split-K (GemmSplitKParallel) variants: partials + reduction kernel,
# epilogue applied once post-reduction — the only split-K decomposition that is
# valid for fused activations.  Confined to the gemm family (like Stream-K).
# Set OASR_GEMM_SPLITK_PARALLEL=0 to skip compiling these variants.
_SPLITK_PARALLEL_ENABLED = os.environ.get("OASR_GEMM_SPLITK_PARALLEL", "1") != "0"

# Curated tiles for parallel split-K: small block_m (the deep splits exist for
# small-M shapes) across the thin-N and 128-wide column tiles.
_SPLITK_PARALLEL_TILES: List[TileShape] = [
    TileShape(block_m=16, block_n=64, block_k=64, warp_m=16, warp_n=32, warp_k=64),
    TileShape(block_m=32, block_n=64, block_k=64, warp_m=16, warp_n=32, warp_k=64),
    TileShape(block_m=64, block_n=64, block_k=64, warp_m=32, warp_n=32, warp_k=64),
    TileShape(block_m=16, block_n=128, block_k=64, warp_m=16, warp_n=32, warp_k=64),
    TileShape(block_m=32, block_n=128, block_k=64, warp_m=32, warp_n=32, warp_k=64),
]


def _build_splitk_parallel_configs(
    sm: int, tiles: List[TileShape], stage_list: List[int], smem_limit: int
) -> Dict[str, CutlassGemmConfig]:
    """Build parallel split-K GEMM configs (runtime split_k ∈ {2,4,8,16})."""
    seen: Dict[str, CutlassGemmConfig] = {}
    for tile in tiles:
        for kStages in stage_list:
            if not _tile_is_buildable(tile, kStages, smem_limit):
                continue
            for split_k in _SPLIT_K_LIST:
                if split_k == 1:
                    continue  # parallel split-K requires > 1 slices
                cfg = CutlassGemmConfig(
                    block_m=tile.block_m,
                    block_n=tile.block_n,
                    block_k=tile.block_k,
                    warp_m=tile.warp_m,
                    warp_n=tile.warp_n,
                    warp_k=tile.warp_k,
                    kStages=kStages,
                    kSmVersion=sm,
                    split_k=split_k,
                    parallel_split_k=True,
                )
                seen[cfg.name] = cfg
    return seen


#: Pipeline depths each CUTLASS 2.x family's templates can be instantiated at.
#:
#: The domain is a template fact; which depths a family *builds* is then its
#: shared-memory budget's call (``_tile_is_buildable``), not a per-SM list.  The
#: per-SM lists this replaced were how sm_86 and sm_89 -- 100 KiB, the same
#: budget as sm_120 -- came to be offered only 3-stage variants while sm_120's
#: rule table picks 4-stage ones in 12 of its 127 buckets.
#:
#: **Turing has no 3-stage tensor-op GEMM at all.**  Compiled here, ``sm_75`` at
#: three or four stages fails identically to the plain path,
#:
#:     default_gemm_universal.h(214): error: incomplete type
#:       "cutlass::gemm::kernel::DefaultGemmUniversal<...>"
#:
#: which is the same ``kernel::DefaultGemm`` 2-stage-only specialisation that
#: makes ``RecurrentArch<75>`` set ``kStages = 2``.  Mirrored by
#: ``oasr.tune.arch.STAGE_DOMAIN``.
_STAGE_DOMAIN: Dict[int, List[int]] = {75: [2], 80: [3, 4], 86: [3, 4], 89: [3, 4], 120: [3, 4]}

#: Pipeline depths the two K-decomposition families are built at, per SM family:
#: derived, not curated.  Stream-K keeps the single depth it has always been built
#: at (the shallowest multistage depth; Turing's two); parallel split-K takes the
#: family's whole domain, shared memory deciding which tiles survive.  A 2.x
#: family with no entry raises ``KeyError`` at config-generation time, which is
#: the intended failure: silently receiving no decompositions is how this became
#: an sm_120-only feature in the first place.
_SM_STREAMK_STAGES: Dict[int, List[int]] = {sm: [d[0]] for sm, d in _STAGE_DOMAIN.items()}
_SM_SPLITK_PARALLEL_STAGES: Dict[int, List[int]] = {sm: list(d) for sm, d in _STAGE_DOMAIN.items()}


def _add_k_decompositions(
    cfgs: Dict[str, CutlassGemmConfig], sm: int, smem_limit: int
) -> Dict[str, CutlassGemmConfig]:
    """Add the Stream-K and parallel split-K variants for *sm* to *cfgs*, and return it.

    Both were reachable on SM120 alone until 2026-09-14 — the two ``if
    _..._ENABLED`` blocks lived inside ``_get_sm120_configs`` — which made
    ``OASR_GEMM_STREAMK`` and ``OASR_GEMM_SPLITK_PARALLEL`` inert on every other
    card despite ``AGENTS.md`` documenting them as global build knobs, and left
    ``oasr.autotune()`` with no Stream-K arm to find on an A100.  It also left
    ``gemm_activation`` with no *valid* split-K anywhere but SM120: serial
    split-K cannot fuse an activation (it would apply per K-partition), so
    parallel split-K is the only decomposition that can, and it was not in the
    space.

    Each architecture's own ``smem_limit`` still decides which tiles survive, so
    Turing keeps only the four Stream-K tiles that fit in 64 KB.
    """
    if _STREAMK_ENABLED:
        cfgs.update(_build_streamk_configs(sm, _STREAMK_TILES, _SM_STREAMK_STAGES[sm], smem_limit))
    if _SPLITK_PARALLEL_ENABLED:
        cfgs.update(
            _build_splitk_parallel_configs(
                sm, _SPLITK_PARALLEL_TILES, _SM_SPLITK_PARALLEL_STAGES[sm], smem_limit
            )
        )
    return cfgs


def _get_sm80_lane_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """The CUTLASS 2.x (``mma.sync``) lane's space for family *sm* -- sm_75/80/86/89/120.

    One generator for every family on the lane.  What differs by family is
    data: the template-legal pipeline depths (:data:`_STAGE_DOMAIN`) and the
    shared-memory budget (:data:`_SM_MAX_SMEM_BYTES`), which together decide
    which ``(tile, depth)`` pairs are buildable.  SM120 runs this lane because
    the CUTLASS 3.x SM120 CollectiveBuilder supports only F8/F6/F4 MMA, so
    FP16/BF16 GEMM there uses the Sm80 forward-compatible ``mma.sync``.

    Turing's single two-stage depth is not a preference, it is the whole
    specialisation (see :data:`_STAGE_DOMAIN`): one unbuildable TU fails the
    whole module, so this list once read ``[2, 3]`` and took `gemm`, `bmm`,
    `group_gemm` and `gemm_log_softmax` down together on Turing.
    """
    smem = _SM_MAX_SMEM_BYTES[sm]
    cfgs = _build_sm_lt90_configs(sm, _GEMM_TILES, _STAGE_DOMAIN[sm], _SPLIT_K_LIST, smem)
    return _add_k_decompositions(cfgs, sm, smem)


# Historical per-family entry points, now one generator.
_get_sm75_configs = _get_sm80_configs = _get_sm86_configs = _get_sm80_lane_configs
_get_sm89_configs = _get_sm120_configs = _get_sm80_lane_configs


# =============================================================================
# Quack-style SM90 / SM100 / SM120 config generation
# =============================================================================


def _get_sm90_configs(sm: int) -> Dict[str, CutlassGemmConfigSm90]:
    """SM90 configs following Quack's ``_get_sm90_configs()`` pattern.

    Produces Cooperative (non-pingpong) and Pingpong variants across a set of
    tile MN shapes and (1×2) / (2×1) cluster shapes.
    """
    tile_k = 128
    kStages = 3

    # Cooperative (non-pingpong) tile shapes.
    #
    # A cooperative mainloop needs at least two pipeline stages, and one stage of
    # (BM + BN) * BK * 2 bytes has to fit Hopper's ~227 KiB of shared memory
    # alongside the epilogue's carveout.  At BK=128 that rules out (256, 192)
    # (2 * 448 * 128 * 2 = 229,376 B) and (256, 208) (237,568 B) outright, and
    # (256, 160) (212,992 B) only clears it for some epilogues: it compiles for
    # GEMM and fails for BMM, whose epilogue leaves less headroom.  Since this
    # list is shared by the gemm, bmm and group_gemm modules, it has to satisfy
    # the tightest consumer -- and one unbuildable variant fails a whole JIT
    # module, not just its own tactic.  All three would fit at BK=64; that is a
    # deliberate re-tune, not a default.  Verified by compiling every remaining
    # combination for sm_90a.
    tile_mn_coop = [
        (256, 128),
        (128, 224),
        (128, 256),
    ]
    # Pingpong tile shapes
    tile_mn_pingpong = [
        (128, 128),
        (128, 160),
        (128, 192),
        (128, 208),
        (192, 128),
    ]
    tile_mn_vals = [(m, n, False) for m, n in tile_mn_coop] + [
        (m, n, True) for m, n in tile_mn_pingpong
    ]
    cluster_vals = [(1, 2), (2, 1)]

    seen: Dict[str, CutlassGemmConfigSm90] = {}
    for (tile_m, tile_n, pingpong), (cluster_m, cluster_n) in itertools.product(
        tile_mn_vals, cluster_vals
    ):
        cfg = CutlassGemmConfigSm90(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            pingpong=pingpong,
            is_dynamic_persistent=False,
            swap_ab=False,
            max_swizzle_size=8,
            use_tma_gather=False,
            kSMs=1,
            kStages=kStages,
            kSmVersion=sm,
        )
        key = cfg.compile_name
        if key not in seen:
            seen[key] = cfg
    return seen


def _sm100_gemm_tile_ok(tile_m: int, tile_n: int, k_sms: int) -> bool:
    """Can CUTLASS build this SM100 tile at all?

    Three constraints, none of them documented outside CUTLASS's own asserts and
    none previously checked here.  They matter more than they look: **one
    unbuildable variant fails the whole JIT module**, so an ungated space means
    no GEMM on the architecture rather than one missing tactic.

    1. MMA tile M -- ``{64, 128}`` for the 1-SM atom, ``{128, 256}`` for the
       2-SM one (``gemm/collective/builders/sm100_common.inl:309,375``).
    2. MMA tile N -- a multiple of 8, at most 256 (same file, ``:313,379``).
    3. The 2-SM TMA epilogue, at 16-bit output with an auto epilogue tile,
       additionally needs ``N <= 128`` or ``N % 64 == 0``
       (``epilogue/collective/builders/sm100_builder.inl:1220``), because at
       ``N % 64 != 0`` the epilogue tile falls back to N and produces
       non-64-aligned smem swizzle strides.  CUTLASS spells out the remedy in
       the assert text: "Use a CtaN that is a multiple of 64 (e.g. 128, 192,
       256) or use a 32-bit output type (f32)."

    Verified by compiling the full emitted space for ``sm_100a``: the predicate
    reproduces the pass/fail split exactly.
    """
    mma_m = tile_m
    if k_sms == 2:
        if mma_m not in (128, 256):
            return False
        # The 16-bit epilogue's N restriction applies to the 2-SM path only.
        if tile_n > 128 and tile_n % 64:
            return False
    elif mma_m not in (64, 128):
        return False
    return tile_n % 8 == 0 and tile_n <= 256


def _get_sm100_configs(sm: int) -> Dict[str, CutlassGemmConfigSm90]:
    """SM100 (Blackwell data-center) configs following Quack's ``_get_sm100_configs()`` pattern.

    Uses kSMs=2 for cluster_m ≥ 2 (2-SM co-operative scheduling via
    ``KernelTmaWarpSpecialized2SmSm100``), kSMs=1 otherwise.
    No pingpong on SM100.
    """
    tile_k = 128
    kStages = 3

    tile_n_vals = [64, 128, 160, 192, 224, 256]
    tile_mn_cluster_vals = (
        [(128, n, (1, 1)) for n in tile_n_vals]
        + [(128, n, (1, 2)) for n in tile_n_vals]
        + [(128, n, (2, 1)) for n in tile_n_vals]
        + [(128, n, (2, 2)) for n in tile_n_vals]
        + [(256, n, (2, 1)) for n in tile_n_vals]
        + [(256, n, (2, 2)) for n in tile_n_vals]
        + [(256, 512, (2, 1))]
    )

    seen: Dict[str, CutlassGemmConfigSm90] = {}
    for tile_m, tile_n, (cluster_m, cluster_n) in tile_mn_cluster_vals:
        # kSMs=2 selects the 2-SM co-operative *schedule* when cluster_m >= 2.
        # It does not scale the tile; see CutlassGemmConfigSm90's comment.
        kSMs = 2 if cluster_m >= 2 else 1
        if not _sm100_gemm_tile_ok(tile_m, tile_n, kSMs):
            continue
        cfg = CutlassGemmConfigSm90(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            pingpong=False,
            is_dynamic_persistent=False,
            swap_ab=False,
            max_swizzle_size=8,
            use_tma_gather=False,
            kSMs=kSMs,
            kStages=kStages,
            kSmVersion=sm,
        )
        key = cfg.compile_name
        if key not in seen:
            seen[key] = cfg
    return seen


def get_all_autotune_configs(
    sm: int,
) -> Dict[str, Union[CutlassGemmConfig, CutlassGemmConfigSm90]]:
    """Return the **full** autotuner config set for *sm* (keyed by ``name``).

    For SM < 90 this includes all split_k and kStages variants; for SM ≥ 90
    it matches the Quack-style set (split_k is not applicable there).

    An unrecognised SM **raises** rather than silently receiving SM120's config
    space.  That ``else`` is how sm_70 and sm_103 came to emit CUTLASS 2.x
    configs and then be rendered through the 3.x template, failing with
    ``AttributeError: 'CutlassGemmConfig' object has no attribute 'tile_m'`` --
    a target that is merely *unlisted* should say so, not inherit another
    architecture's tiles.
    """
    if sm in _STAGE_DOMAIN:
        return _get_sm80_lane_configs(sm)  # type: ignore[return-value]
    elif sm == 90:
        return _get_sm90_configs(sm)  # type: ignore[return-value]
    elif sm == 100:
        return _get_sm100_configs(sm)  # type: ignore[return-value]
    raise ValueError(
        f"no GEMM config space for sm_{sm}; OASR compiles for "
        f"{', '.join(f'sm_{t}' for t in _TARGET_SMS)}"
    )


def get_unique_compile_configs(
    sm: int,
) -> Dict[str, Union[CutlassGemmConfig, CutlassGemmConfigSm90]]:
    """Return the set of uniquely-compiled configs for *sm* (keyed by ``compile_name``).

    This is the **compilation** set — variants differing only in runtime
    parameters (``split_k`` for SM<90; ``swap_ab`` / ``is_dynamic_persistent``
    for SM≥90) are collapsed to a single entry.
    """
    all_cfgs = get_all_autotune_configs(sm)
    seen: Dict[str, Union[CutlassGemmConfig, CutlassGemmConfigSm90]] = {}
    for cfg in all_cfgs.values():
        key = cfg.compile_name
        if key not in seen:
            seen[key] = cfg
    return seen


#: Build knob: which variants the *production* modules compile.
#:
#: ``tuned`` (default): the configs the tuning DB references for this arch
#: (shipped and user tiers), the untuned default, and the coverage basis --
#: typically well under half the space (the SM120 table references 17 of 35).
#: ``all``: the whole tuning space, the historical behaviour.  The tuning
#: modules (``gemm_tune`` ...) always compile the whole space, and are only built
#: by a process that tunes or sweeps.
_COMPILE_SET = os.environ.get("OASR_GEMM_COMPILE_SET", "tuned").strip().lower()

#: The coverage basis of the CUTLASS 2.x lane, as ``(tile, warp, decomposition)``:
#: one config per shape class (thin / small / mid / wide rows, a K-decomposition
#: for deep-K thin shapes), so a signature nobody tuned still has a sensible
#: compiled option for the cost-model fallback -- the alternative is a runtime
#: compile on the step path.  Built at the family's shallowest pipeline depth.
_LANE_BASIS_2X = (
    ((16, 64, 64), (16, 32, 64), ""),
    ((32, 64, 64), (16, 32, 64), ""),
    ((64, 64, 64), (32, 32, 64), ""),
    ((32, 128, 64), (32, 32, 64), ""),
    ((64, 128, 64), (32, 64, 64), ""),
    ((128, 256, 64), (64, 64, 64), ""),
    ((16, 64, 64), (16, 32, 64), "pk"),
    ((64, 128, 64), (32, 64, 64), "sk"),
)


def _basis_compile_names(sm: int) -> List[str]:
    if sm not in _STAGE_DOMAIN:
        return []
    s0 = _STAGE_DOMAIN[sm][0]
    out = []
    for (bm, bn, bk), (wm, wn, wk), dec in _LANE_BASIS_2X:
        name = f"sm{sm}_b{bm}x{bn}x{bk}_w{wm}x{wn}x{wk}_s{s0}" + (f"_{dec}" if dec else "")
        out.append(name)
    return out


def _tuned_compile_names(sm: int) -> List[str]:
    """Compile names of every config the tuning DB's tiers name for *sm*."""
    from oasr.tune import database

    names: List[str] = []
    for _tier, tf in database.tiers("gemm", sm).ordered():
        for cid in tf.referenced_configs():
            try:
                choice = gemm_config_from_params(tf.configs[cid], sm)
            except (KeyError, TypeError, ValueError):
                continue
            if not isinstance(choice, str):
                names.append(choice.compile_name)
    return names


def get_production_configs(
    sm: int,
) -> Dict[str, Union[CutlassGemmConfig, CutlassGemmConfigSm90]]:
    """The configs the production GEMM modules compile for *sm* (keyed by ``compile_name``).

    A subset of :func:`get_unique_compile_configs`: the tuning DB's references,
    the default and the coverage basis (``OASR_GEMM_COMPILE_SET=all`` restores
    the whole space).  The CUTLASS 3.x lanes have no tuned table yet and a small
    space, so they keep all of it.
    """
    space = get_unique_compile_configs(sm)
    if _COMPILE_SET == "all" or sm not in _STAGE_DOMAIN:
        return space
    keep = {default_config_for_sm(sm).compile_name}
    keep.update(_basis_compile_names(sm))
    keep.update(_tuned_compile_names(sm))
    return {name: cfg for name, cfg in space.items() if name in keep}


# =============================================================================
# Helper: render all tile variants for a given template
# =============================================================================


def _render_all_variants(
    template_name: str,
    template_sm90_name: str,
    family: str,
    *,
    with_activation: bool = False,
    configs: Optional[Dict[str, Union["CutlassGemmConfig", "CutlassGemmConfigSm90"]]] = None,
) -> List:
    """Render Jinja templates for all unique tile configs.

    Each unique compile config produces one ``.cu`` file with uniquely-named
    exported functions (e.g., ``gemm_sm90_b128x128x128_c1x2_k1_s3_coop``).

    Args:
        template_name: Jinja template file name for SM<90.
        template_sm90_name: Jinja template file name for SM90+.
        family: Kernel family name (``"gemm"``, ``"bmm"``, ``"group_gemm"``).
        with_activation: Whether to include fused activation variants (GEMM only).

    Returns:
        List of Path objects for the rendered ``.cu`` files.
    """
    from .cubin_loader import write_if_different
    from .templates import render_template

    sm = _get_target_sm()
    unique_configs = configs if configs is not None else get_production_configs(sm)
    source_paths = []

    for config_name, cfg in unique_configs.items():
        # Stream-K and parallel split-K are implemented in the GEMM template
        # only; skip those configs for bmm / group_gemm (their templates have
        # no Stream-K / parallel split-K path).
        if family != "gemm" and (
            getattr(cfg, "stream_k", False) or getattr(cfg, "parallel_split_k", False)
        ):
            continue
        # The grouped template always drives the ptr-array *cooperative*
        # schedule, whose EpilogueTileAuto requires the epilogue tile M to divide
        # CTA_M -- in practice a multiple of 128 ("EPI_TILE_M must divide
        # CTA_M").  The 192-row SM90 tile is the one shape that violates it.  The
        # `pingpong` flag is inert here for the same reason -- those configs
        # would only duplicate a cooperative kernel under another name -- so
        # nothing is lost.  `tile_m` is SM90-only, so the 2.x configs (which
        # spell it `block_m`) are untouched.
        if family == "group_gemm" and getattr(cfg, "tile_m", 0) % 128 != 0:
            continue

        func_name = f"{family}_{config_name}"
        variant_file_name = f"{family}_sm{sm}_{config_name}"

        if sm in [75, 80, 86, 89, 120]:
            rendered = render_template(
                template_name,
                op_name=variant_file_name,
                func_name=func_name,
                tile_m=cfg.block_m,
                tile_n=cfg.block_n,
                tile_k=cfg.block_k,
                warp_m=cfg.warp_m,
                warp_n=cfg.warp_n,
                warp_k=cfg.warp_k,
                stages=cfg.kStages,
                sm_version=sm,
                stream_k=getattr(cfg, "stream_k", False),
                parallel_split_k=getattr(cfg, "parallel_split_k", False),
                with_activation=with_activation,
            )
        else:
            rendered = render_template(
                template_sm90_name,
                op_name=variant_file_name,
                func_name=func_name,
                tile_m=cfg.tile_m,
                tile_n=cfg.tile_n,
                tile_k=cfg.tile_k,
                cluster_m=cfg.cluster_m,
                cluster_n=cfg.cluster_n,
                k_sms=cfg.kSMs,
                stages=cfg.kStages,
                sm_version=sm,
                pingpong=cfg.pingpong,
                with_activation=with_activation,
            )
        gen_path = env.OASR_GEN_SRC_DIR / family / f"{variant_file_name}.cu"
        write_if_different(gen_path, rendered)
        source_paths.append(gen_path)

    return source_paths


#: The general BMM lane's instantiation grid, one translation unit per cell.
#: Splitting it is not cosmetic: nvcc parallelizes across translation units but
#: not within one, and this module's cold build is set by its largest TU.  On a
#: 64-core box the whole ``bmm`` module builds in 112 s with the grid in one TU,
#: 70 s split by B layout, and 42 s split by layout *and* dtype.
#: ``(cell, CUTLASS LayoutB, CUTLASS element)``.  The C++ entry point for a cell
#: is ``generalBmm_<cell>``, **declared in** ``include/oasr/gemm/bmm.cuh`` and
#: called by ``generalBmm`` there — that header owns the name, this table only
#: has to agree with it, and a disagreement is a link error rather than a silent
#: miss.
_BMM_GENERAL_CELLS = (
    ("column_major_f16", "ColumnMajor", "cutlass::half_t"),
    ("column_major_bf16", "ColumnMajor", "cutlass::bfloat16_t"),
    ("row_major_f16", "RowMajor", "cutlass::half_t"),
    ("row_major_bf16", "RowMajor", "cutlass::bfloat16_t"),
)


def _render_bmm_general_variants() -> List:
    """Render the general BMM lane's instantiation TUs.

    Same pattern as the tile variants above — the grid lives in one template
    rather than in near-duplicate files under ``csrc/``.
    """
    from .cubin_loader import write_if_different
    from .templates import render_template

    source_paths = []
    for cell, layout, element in _BMM_GENERAL_CELLS:
        op_name = f"bmm_general_{cell}"
        rendered = render_template(
            "bmm_general_template.cu.jinja",
            op_name=op_name,
            func_name=f"generalBmm_{cell}",
            layout=layout,
            element=element,
        )
        gen_path = env.OASR_GEN_SRC_DIR / "bmm" / f"{op_name}.cu"
        write_if_different(gen_path, rendered)
        source_paths.append(gen_path)
    return source_paths


# =============================================================================
# Module generators — ALL variants compiled into ONE .so per family
# =============================================================================


def _level_configs(level: str):
    """The variants a module of *level* renders.

    The tuning module holds the tuning space **minus** the production set --
    never a variant both modules have.  Two libraries instantiating the same
    CUTLASS kernel share its template static members: GCC emits them as
    ``STB_GNU_UNIQUE`` and the loader merges them process-wide, so
    ``GemmUniversalBase``'s once-per-process ``device_ordinal_`` cache made the
    second module skip ``cudaFuncSetAttribute`` for *its own* Stream-K kernel,
    whose 73.7 KB of dynamic shared memory then failed to launch ("GEMM kernel
    failed") -- measured, the moment both modules had run the same variant.
    Lookups resolve production first, then tuning (``oasr.functionals.gemm``).
    """
    sm = _get_target_sm()
    if level == "production":
        cfgs = get_production_configs(sm)
        _BUILT_PRODUCTION[sm] = frozenset(cfgs)
        _COMPILED_NAMES.pop(sm, None)
        return cfgs
    prod = get_production_configs(sm)
    return {n: c for n, c in get_unique_compile_configs(sm).items() if n not in prod}


def _module_name(family: str, level: str) -> str:
    return family if level == "production" else f"{family}_tune"


def has_tuning_module() -> bool:
    """Whether the tuning space holds any variant the production module lacks."""
    return bool(_level_configs("tune"))


def gen_gemm_module(level: str = "production") -> JitSpec:
    """Generate JIT spec for GEMM: every variant of *level* in one module.

    ``level="production"`` compiles :func:`get_production_configs` -- what the
    tuning DB references, the default and the coverage basis; ``"tune"``
    compiles the whole tuning space (:func:`get_unique_compile_configs`), for
    the tuner, ``oasr.autotune()`` and the per-variant correctness sweeps.

    Each variant exports ``gemm_{config_name}`` and ``gemm_{config_name}_activation``
    as TVM-FFI functions.
    """
    source_paths = _render_all_variants(
        "gemm_cutlass_template.cu.jinja",
        "gemm_cutlass_template_sm90.cu.jinja",
        "gemm",
        with_activation=True,
        configs=_level_configs(level),
    )
    # Plus the workspace-cache diagnostics (``ws_cache_keys`` /
    # ``ws_cache_bytes``).  One extra TU, not part of the rendered template,
    # which is compiled once per tile configuration.
    source_paths = source_paths + [env.OASR_CSRC_DIR / "gemm_ws_cache.cu"]
    return gen_jit_spec(_module_name("gemm", level), source_paths)


def gen_bmm_module(level: str = "production") -> JitSpec:
    """Generate JIT spec for BMM: the tuned tile variants plus the general lane.

    Each tile variant exports ``bmm_{config_name}``; those are the alignment-8
    contiguous-3-D fast lane the shape heuristic selects from.  ``bmm`` is the
    general lane — arbitrary batch strides, either B layout, small/unaligned N
    and K — which is what Zipformer's decomposed attention needs and no tile
    variant can express (``csrc/bmm.cu`` + the two instantiation halves).
    """
    source_paths = _render_all_variants(
        "bmm_cutlass_template.cu.jinja",
        "bmm_cutlass_template_sm90.cu.jinja",
        "bmm",
        configs=_level_configs(level),
    )
    if level == "production":
        # The general lane lives in the production module only: a second copy
        # in the tuning module would duplicate its CUTLASS instantiations (see
        # ``_level_configs`` for why that is a correctness hazard, not waste).
        source_paths = (
            source_paths
            + _render_bmm_general_variants()
            + [
                env.OASR_CSRC_DIR / "bmm.cu",
                env.OASR_CSRC_DIR / "bmm_jit_binding.cu",
            ]
        )
    return gen_jit_spec(_module_name("bmm", level), source_paths)


def gen_group_gemm_module(level: str = "production") -> JitSpec:
    """Generate JIT spec for grouped GEMM with every variant of *level* in one module.

    Each variant exports ``group_gemm_{config_name}`` as a TVM-FFI function.
    """
    source_paths = _render_all_variants(
        "group_gemm_cutlass_template.cu.jinja",
        "group_gemm_cutlass_template_sm90.cu.jinja",
        "group_gemm",
        configs=_level_configs(level),
    )
    return gen_jit_spec(_module_name("group_gemm", level), source_paths)


def gen_gemm_log_softmax_module() -> JitSpec:
    """Generate JIT spec for fused GEMM + log_softmax.

    Replaces ``F.log_softmax(linear(x), dim=-1)`` (e.g. the CTC head) with a
    single Python call; internally a CUTLASS GEMM and an online log_softmax
    kernel chain on the same stream.
    """
    return gen_jit_spec(
        "gemm_log_softmax",
        [
            env.OASR_CSRC_DIR / "gemm_log_softmax.cu",
            env.OASR_CSRC_DIR / "gemm_log_softmax_jit_binding.cu",
        ],
    )


# =============================================================================
# Default function name helpers
# =============================================================================


def gemm_func_name(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]) -> str:
    """Return the TVM-FFI export name for a GEMM variant."""
    return f"gemm_{cfg.compile_name}"


def gemm_activation_func_name(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]) -> str:
    """Return the TVM-FFI export name for a GEMM+activation variant."""
    return f"gemm_{cfg.compile_name}_activation"


def bmm_func_name(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]) -> str:
    """Return the TVM-FFI export name for a BMM variant."""
    return f"bmm_{cfg.compile_name}"


def group_gemm_func_name(cfg: Union[CutlassGemmConfig, CutlassGemmConfigSm90]) -> str:
    """Return the TVM-FFI export name for a grouped GEMM variant."""
    return f"group_gemm_{cfg.compile_name}"


# =============================================================================
# Default configs (used by non-autotuned paths in oasr/functionals/gemm.py)
# =============================================================================

_sm = _get_target_sm()


def default_config_for_sm(sm: int) -> Union[CutlassGemmConfig, CutlassGemmConfigSm90]:
    """The non-autotuned default config for *sm*.

    It **must** be one of the variants ``get_all_autotune_configs(sm)`` emits:
    the JIT module compiles exactly those, and the functional API looks the
    default up by ``compile_name``, so a default outside that set raises
    ``AttributeError: Module has no function ...`` on the first un-tuned call.
    ``tests/test_gemm_heuristic.py`` enforces the invariant for every target.
    """
    if sm < 90 or sm == 120:
        # SM120 uses the CUTLASS 2.x (SM<90) path for FP16/BF16 — see
        # ``_get_sm120_configs`` above.  Turing is the one target whose
        # kernel::DefaultGemm tensor-op specialisation is fixed at two pipeline
        # stages, which is why _get_sm75_configs emits only `_s2` variants.
        return CutlassGemmConfig(
            block_m=128,
            block_n=128,
            block_k=64,
            warp_m=64,
            warp_n=64,
            warp_k=64,
            kStages=2 if sm == 75 else 3,
            kSmVersion=sm,
        )
    # SM90 generates 1x2 and 2x1 clusters only, and pairs a 128x128 tile with the
    # pingpong schedule; SM100 does generate the 1x1 cooperative variant.
    cluster_n, pingpong = (2, True) if sm == 90 else (1, False)
    return CutlassGemmConfigSm90(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        cluster_m=1,
        cluster_n=cluster_n,
        pingpong=pingpong,
        is_dynamic_persistent=False,
        swap_ab=False,
        max_swizzle_size=8,
        use_tma_gather=False,
        kSMs=1,
        kStages=3,
        kSmVersion=sm,
    )


GEMM_DEFAULT: Union[CutlassGemmConfig, CutlassGemmConfigSm90] = default_config_for_sm(_sm)


# =============================================================================
# Tuning-DB codec -- a config as explicit parameters (oasr.tune.database)
# =============================================================================

#: The MMA lane each compiled SM family's GEMMs run on.  A tuning file records
#: its lane as a hard validator: a table measured on one lane means nothing on
#: another, whatever the SM number says.
GEMM_LANE_BY_SM: Dict[int, str] = {
    75: "sm80_mma",
    80: "sm80_mma",
    86: "sm80_mma",
    89: "sm80_mma",
    90: "sm90_wgmma",
    100: "sm100_tcgen05",
    120: "sm80_mma",
}

#: Non-config choices a rule can name, and the ids they are stored under.
#: ``"default"`` decodes to *this process's* :data:`GEMM_DEFAULT` -- the object
#: identity callers compare against -- not a parameter set.
_SENTINEL_IDS = ("torch", "fused", "default")

GemmChoice = Union[CutlassGemmConfig, CutlassGemmConfigSm90, str]


def gemm_config_id(choice: GemmChoice) -> str:
    """The stable, human-readable id a choice is stored under.

    The config's ``name`` without its ``sm<N>_`` prefix: the prefix is the
    file's arch, stated once in the file rather than on every id.
    """
    if isinstance(choice, str):
        if choice not in ("torch", "fused"):
            raise ValueError(f"unknown GEMM sentinel {choice!r}")
        return choice
    if choice is GEMM_DEFAULT:
        return "default"
    return choice.name.split("_", 1)[1]


def gemm_config_to_params(choice: GemmChoice, *, sentinel_default: bool = True) -> Dict[str, Any]:
    """A choice as the explicit parameter dict a tuning file stores.

    ``sentinel_default=False`` spells :data:`GEMM_DEFAULT` out as its parameters
    instead of the ``"default"`` sentinel -- what a measurement log wants, since
    a cost model needs the tile, not the name.
    """
    if isinstance(choice, str) or (sentinel_default and choice is GEMM_DEFAULT):
        return {"kind": gemm_config_id(choice)}
    if isinstance(choice, CutlassGemmConfig):
        return {
            "kind": "cutlass",
            "lane": "sm80_mma",
            "tile": [choice.block_m, choice.block_n, choice.block_k],
            "warp": [choice.warp_m, choice.warp_n, choice.warp_k],
            "stages": choice.kStages,
            "split_k": choice.split_k,
            "stream_k": bool(choice.stream_k),
            "parallel_split_k": bool(choice.parallel_split_k),
        }
    return {
        "kind": "cutlass",
        "lane": GEMM_LANE_BY_SM.get(choice.kSmVersion, "sm90_wgmma"),
        "tile": [choice.tile_m, choice.tile_n, choice.tile_k],
        "cluster": [choice.cluster_m, choice.cluster_n],
        "pingpong": bool(choice.pingpong),
        "persistent": bool(choice.is_dynamic_persistent),
        "swap_ab": bool(choice.swap_ab),
        "swizzle": choice.max_swizzle_size,
        "tma_gather": bool(choice.use_tma_gather),
        "sms": choice.kSMs,
        "stages": choice.kStages,
    }


def gemm_config_from_params(params: Dict[str, Any], sm: int) -> GemmChoice:
    """Inverse of :func:`gemm_config_to_params` for arch family *sm*."""
    kind = params.get("kind")
    if kind == "default":
        return GEMM_DEFAULT
    if kind in ("torch", "fused"):
        return str(kind)
    if kind != "cutlass":
        raise ValueError(f"unknown GEMM config kind {kind!r}")
    tile = [int(v) for v in params["tile"]]  # type: ignore[union-attr]
    if params.get("lane") == "sm80_mma":
        warp = [int(v) for v in params["warp"]]  # type: ignore[union-attr]
        return CutlassGemmConfig(
            block_m=tile[0],
            block_n=tile[1],
            block_k=tile[2],
            warp_m=warp[0],
            warp_n=warp[1],
            warp_k=warp[2],
            kStages=int(params["stages"]),  # type: ignore[arg-type]
            kSmVersion=int(sm),
            split_k=int(params.get("split_k", 1)),  # type: ignore[arg-type]
            stream_k=bool(params.get("stream_k", False)),
            parallel_split_k=bool(params.get("parallel_split_k", False)),
        )
    cluster = [int(v) for v in params.get("cluster", [1, 1])]  # type: ignore[union-attr]
    return CutlassGemmConfigSm90(
        tile_m=tile[0],
        tile_n=tile[1],
        tile_k=tile[2],
        cluster_m=cluster[0],
        cluster_n=cluster[1],
        pingpong=bool(params.get("pingpong", False)),
        is_dynamic_persistent=bool(params.get("persistent", False)),
        swap_ab=bool(params.get("swap_ab", False)),
        max_swizzle_size=int(params.get("swizzle", 8)),  # type: ignore[arg-type]
        use_tma_gather=bool(params.get("tma_gather", False)),
        kSMs=int(params.get("sms", 1)),  # type: ignore[arg-type]
        kStages=int(params.get("stages", 3)),  # type: ignore[arg-type]
        kSmVersion=int(sm),
    )


def gemm_impl_hash() -> str:
    """Identity of the GEMM-family kernel implementation a tuning result measured.

    The templates every variant is rendered from, plus the GEMM and shared
    headers they include.  A soft validator (see ``oasr.tune.database``): a
    change marks tuning files stale -- re-measure -- without discarding them.
    """
    from oasr.tune.database import hash_paths

    paths = sorted(env.OASR_TEMPLATE_DIR.glob("*gemm*_template*.jinja"))
    paths += sorted(env.OASR_TEMPLATE_DIR.glob("bmm_*_template*.jinja"))
    for sub in ("gemm", "common"):
        paths += sorted((env.OASR_INCLUDE_DIR / "oasr" / sub).rglob("*.h*"))
    paths.append(env.OASR_CSRC_DIR / "gemm_log_softmax.cu")
    return hash_paths(paths)


def _gemm_soft_validators() -> Dict[str, Any]:
    return {"impl_hash": gemm_impl_hash()}


def _register_with_tuning_db() -> None:
    from oasr.tune import database

    database.register_soft_validator("gemm", _gemm_soft_validators)


_register_with_tuning_db()


def gemm_sig_key(op: str, N: int, K: int, dtype_class: str = "half") -> str:
    """The tuning-DB key of a GEMM-family static signature.

    ``dtype_class`` is ``"half"`` for an entry that serves fp16 and bf16 alike
    -- every entry today: the rules were tuned in bf16 and verified to carry to
    fp16 within ~1% -- or ``"fp16"`` / ``"bf16"`` for a dtype-specific one,
    which the resolver prefers when both exist.
    """
    from oasr.tune.database import sig_key

    return sig_key("gemm", op=op, dt=dtype_class, N=int(N), K=int(K))


# =============================================================================
# Shape-aware selection: measured rule tables, read from the tuning DB
# =============================================================================
#
# The measured tables are data: ``oasr/tune/db/sm<family>/gemm.json`` is the
# shipped *system* tier, and ``~/.cache/oasr/tune/v2/sm<family>/gemm.json`` the
# *user* tier written by tuning runs on the machine that measured it, which wins
# on conflict.  They used to be Python literals that ``scripts/tune_asr_gemm.py``
# printed and somebody pasted here; ``oasr.tune.database`` has the format, the
# validator rules and the reasons.
#
# A file entry is keyed by the static signature -- ``(op, N, K)`` and a dtype
# class -- and its ascending ``(m_max, choice)`` regions are looked up by
# rounding M *up*, exactly as the literal tables were.

#: One arch's rules: ``(op, N, K) -> ascending [(m_max | None, choice), ...]``.
_RuleTable = Dict[Tuple[str, int, int], list]


def _table_from_file(tf, dtype_class: str) -> _RuleTable:
    """The ``dtype_class`` rules of tuning file *tf* as a ``(op, N, K)`` table."""
    from oasr.tune.database import parse_sig_key

    table: _RuleTable = {}
    decoded: Dict[str, GemmChoice] = {}
    for key, entry in tf.entries.items():
        family, fields = parse_sig_key(key)
        if family != "gemm" or fields.get("dt") != dtype_class or not entry.regions:
            continue
        rules = []
        for m_max, cid in entry.regions:
            choice = decoded.get(cid)
            if choice is None:
                choice = decoded[cid] = gemm_config_from_params(tf.configs[cid], tf.arch_family)
            rules.append((m_max, choice))
        table[(fields["op"], int(fields["N"]), int(fields["K"]))] = rules
    return table


#: ``(epoch, sm) -> [(tier, {dtype_class: table}), ...]``, highest tier first.
_TIER_VIEWS: Dict[Tuple[int, int], List[Tuple[str, Dict[str, _RuleTable]]]] = {}


def _tier_views(sm: int) -> List[Tuple[str, Dict[str, _RuleTable]]]:
    """The decoded user and system tables for arch family *sm*, this epoch."""
    from oasr.tune import database

    key = (database.epoch(), int(sm))
    views = _TIER_VIEWS.get(key)
    if views is not None:
        return views
    views = []
    for tier, tf in database.tiers("gemm", int(sm)).ordered():
        views.append((tier, {dt: _table_from_file(tf, dt) for dt in ("fp16", "bf16", "half")}))
    _TIER_VIEWS.clear()  # only the current epoch is ever consulted
    _TIER_VIEWS[key] = views
    return views


def _load_system_rules() -> Dict[int, _RuleTable]:
    from oasr.tune import database

    out: Dict[int, _RuleTable] = {}
    for sm in _TARGET_SMS:
        tf = database.tiers("gemm", sm).system
        if tf is not None:
            out[sm] = _table_from_file(tf, "half")
    return out


#: The shipped (system-tier) rule tables by compiled SM family
#: (``oasr.jit.core._SM_FAMILY``) -- a *view* of ``oasr/tune/db/sm*/gemm.json``.
#:
#: The heuristic used to be gated on ``sm != 120`` in ``select_default_config``,
#: which made "which architectures are tuned?" a control-flow question with one
#: possible answer.  Here it is data, and the answer is this dict's keys (plus
#: whatever a user-tier file adds on this machine).
#:
#: An architecture that is absent is neither an error nor a rule miss -- nobody
#: has measured it, and ``GEMM_DEFAULT`` computes the right answer -- but it is
#: not silence either: ``select_default_config`` counts the lookups in
#: ``_ARCH_INACTIVE`` and ``rule_miss_report`` names the arch.  That matters
#: because the fall-through is the *largest* gap the heuristic can have (every
#: shape, not one width) and was the only invisible one: the report used to say
#: "every shape this process issued had a tuned rule" on a box whose table was
#: never consulted.
#:
#: Populating one is a measurement, not a guess.  Rules that were reasoned about
#: rather than timed have shipped a 4.6x regression and an empty transcript in
#: this file's history; ``oasr tune build`` (or ``scripts/tune_asr_gemm.py``) on
#: the target card is the only supported way in.
_GEMM_HEURISTIC_RULES: Dict[int, _RuleTable] = _load_system_rules()

#: The SM120 table, kept under its historical name.
_GEMM_HEURISTIC_RULES_SM120: _RuleTable = _GEMM_HEURISTIC_RULES.get(120, {})


def _on_tuning_reload() -> None:
    global _GEMM_HEURISTIC_RULES, _GEMM_HEURISTIC_RULES_SM120
    _TIER_VIEWS.clear()
    _MODEL_VIEWS.clear()
    _GEMM_HEURISTIC_RULES = _load_system_rules()
    _GEMM_HEURISTIC_RULES_SM120 = _GEMM_HEURISTIC_RULES.get(120, {})
    _COMPILED_NAMES.clear()


def _register_reload_hook() -> None:
    from oasr.tune import database

    database.register_reload_hook(_on_tuning_reload)


_register_reload_hook()

#: ``sm -> compile names`` a production dispatch can reach (see :func:`_is_compiled`).
_COMPILED_NAMES: Dict[int, frozenset] = {}

#: ``sm -> compile names`` the production GEMM module of this process was built
#: with.  Frozen at generation: a later tuning-DB reload can name configs the
#: loaded module does not contain, and those must not reach dispatch.
_BUILT_PRODUCTION: Dict[int, frozenset] = {}

#: Families whose tuning module (the whole space) this process has loaded.
_TUNE_LOADED: set = set()


def mark_tuning_module_loaded(sm: int) -> None:
    """Every config of the tuning space is dispatchable from now on (the tuning
    module is loaded); called by the functional layer when it loads it."""
    _TUNE_LOADED.add(int(sm))
    _COMPILED_NAMES.pop(int(sm), None)


def _is_compiled(choice: GemmChoice, sm: int) -> bool:
    """Can *choice* be dispatched on arch family *sm*?  Sentinels always can."""
    if isinstance(choice, str) or choice is GEMM_DEFAULT:
        return True
    names = _COMPILED_NAMES.get(sm)
    if names is None:
        try:
            built = _BUILT_PRODUCTION.get(sm)
            found = set(built) if built is not None else set(get_production_configs(sm))
            if sm in _TUNE_LOADED:
                found |= set(get_unique_compile_configs(sm))
            names = frozenset(found)
        except ValueError:
            names = frozenset()
        _COMPILED_NAMES[sm] = names
    return getattr(choice, "kSmVersion", sm) == sm and choice.compile_name in names


# Half-precision dtype strings the rules apply to (the kernels + SMEM budgets
# assume 2-byte operands; fp32 keeps GEMM_DEFAULT).
_HEURISTIC_DTYPES = ("torch.float16", "torch.bfloat16")

# Rollback / A-B-parity switch (read once at import): set OASR_GEMM_HEURISTIC=0
# to force the legacy fixed-config path (every shape → GEMM_DEFAULT).
_HEURISTIC_ENABLED = os.environ.get("OASR_GEMM_HEURISTIC", "1") != "0"


class _RuleMiss:
    """How often an untuned ``(op, N, K)`` was asked for, and over what M."""

    __slots__ = ("calls", "m_min", "m_max")

    def __init__(self, M: int):
        self.calls = 1
        self.m_min = M
        self.m_max = M

    def add(self, M: int) -> None:
        self.calls += 1
        if M < self.m_min:
            self.m_min = M
        elif M > self.m_max:
            self.m_max = M


#: ``(op, N, K)`` with no tuned rule -> :class:`_RuleMiss`.  Bounded by the number
#: of distinct GEMM shapes a model has, so this cannot grow with request count.
_RULE_MISSES: Dict[Tuple[str, int, int], _RuleMiss] = {}

#: SM family -> lookups that found no rule table for it at all.
#:
#: Deliberately *not* folded into ``_RULE_MISSES``: a missing table is one gap
#: covering every shape, and recording it per ``(op, N, K)`` would name every
#: GEMM the process issued and drown the widths that are genuinely untuned on a
#: tuned arch.  One entry per architecture is the true cardinality of the fact.
_ARCH_INACTIVE: Dict[int, int] = {}


def rule_misses() -> Dict[Tuple[str, int, int], Tuple[int, int, int]]:
    """Untuned shapes this process asked about: ``(op, N, K) -> (calls, Mmin, Mmax)``."""
    return {k: (v.calls, v.m_min, v.m_max) for k, v in _RULE_MISSES.items()}


def heuristic_inactive() -> Dict[int, int]:
    """SM families this process asked about that have no rule table: ``sm -> calls``.

    Empty is the good case *and* the uninteresting one: it also covers a process
    that issued no half-precision GEMM at all.  Read it against
    :data:`_GEMM_HEURISTIC_RULES`, which says which architectures are tuned.
    """
    return dict(_ARCH_INACTIVE)


def reset_rule_misses() -> None:
    """Clear the miss tables (per-test isolation, per-benchmark accounting).

    Covers the untuned-arch counter as well, so ``reset`` -> run -> report stays
    one call for both halves of the coverage question.
    """
    _RULE_MISSES.clear()
    _ARCH_INACTIVE.clear()


def rule_miss_report() -> str:
    """Which GEMM shapes ran on the untuned fallback tile, and how often.

    A missing rule is not an error — ``GEMM_DEFAULT`` computes the right answer —
    which is exactly why it needs reporting.  The table is keyed on the exact
    ``(op, N, K)``, so it only ever covers model widths somebody tuned, and for a
    year it covered one architecture while five others silently took a fallback
    tile that is up to 4.6x off the best backend.  Nothing failed, no counter
    moved, and no log line was emitted; the only way to find out was to read the
    table and compare it against a capture by hand.

    Run a workload, print this, and the output is both the answer to "is this
    model covered?" and the shape list to feed ``scripts/tune_asr_gemm.py``.

    Two kinds of gap, reported separately because the fix differs.  A *shape*
    with no rule wants that shape tuned; an *architecture* with no table wants a
    sweep.  The second used to be unreportable — the arch fall-through returned
    before recording anything — so this function said "every shape this process
    issued had a tuned rule" on every non-SM120 box, which is the opposite of
    what happened there.
    """
    if not _HEURISTIC_ENABLED:
        return "GEMM heuristic disabled (OASR_GEMM_HEURISTIC=0) — every shape used GEMM_DEFAULT."
    lines: List[str] = []
    if _ARCH_INACTIVE:
        tuned = ", ".join(f"sm{s}" for s in sorted(_GEMM_HEURISTIC_RULES)) or "none"
        for sm_, calls in sorted(_ARCH_INACTIVE.items()):
            lines.append(
                f"GEMM heuristic inactive on sm{sm_}: no tuned rule table (tuned: {tuned}), so "
                f"all {calls} shape lookup(s) used GEMM_DEFAULT. "
                f"Tune this card with scripts/tune_asr_gemm.py."
            )
    if not _RULE_MISSES:
        if lines:
            return "\n".join(lines)
        return "GEMM heuristic: every shape this process issued had a tuned rule."
    lines += [
        f"GEMM heuristic: {len(_RULE_MISSES)} shape(s) had no tuned rule and used "
        f"GEMM_DEFAULT (tune with scripts/tune_asr_gemm.py):",
        f"    {'op':<18} {'N':>7} {'K':>7} {'calls':>8} {'M range':>19}",
    ]
    for (op, N, K), st in sorted(_RULE_MISSES.items(), key=lambda kv: -kv[1].calls):
        span = f"{st.m_min}" if st.m_min == st.m_max else f"{st.m_min}..{st.m_max}"
        lines.append(f"    {op:<18} {N:>7} {K:>7} {st.calls:>8} {span:>19}")
    return "\n".join(lines)


def select_default_config(op: str, M: int, N: int, K: int, dtype, sm: int):
    """Pick a GEMM config for the non-autotuned production path.

    ``op`` is one of ``"gemm"``, ``"gemm_activation"``, ``"bmm"``, or
    ``"gemm_log_softmax"``.  Returns a :class:`CutlassGemmConfig`, the string
    ``"torch"`` (dispatch to cuBLAS), the string ``"fused"`` (the single-call
    fused CUTLASS launcher — ``gemm_log_softmax`` only), or
    :data:`GEMM_DEFAULT`.  Pure function of the shape, so it is CUDA-graph
    safe (same choice on every capture/replay).  Unknown ops/shapes, untuned
    arches, and non-half dtypes fall back to ``GEMM_DEFAULT`` — i.e.
    byte-identical to the previous fixed behaviour.

    Which architectures are tuned is :data:`_GEMM_HEURISTIC_RULES`, not a
    condition here.  Both fall-throughs are counted, at the cardinality of the
    fact each one is: a shape with no rule in :func:`rule_misses`, an arch with
    no table in :func:`heuristic_inactive`.

    The dtype gate is checked first on purpose.  An fp32 lookup would take the
    fallback on a *tuned* arch too, so counting it as an untuned-architecture
    gap would overstate one — the counter means "this card has no table", not
    "this call missed".
    """
    if not _HEURISTIC_ENABLED or str(dtype) not in _HEURISTIC_DTYPES:
        return GEMM_DEFAULT
    from oasr.tune.database import record_tier

    sm = int(sm)
    views = _tier_views(sm)
    if not views:
        _ARCH_INACTIVE[sm] = _ARCH_INACTIVE.get(sm, 0) + 1
        record_tier("gemm", op, "default")
        return GEMM_DEFAULT
    key = (op, int(N), int(K))
    dt = _DTYPE_CLASS[str(dtype)]
    had_rules = False
    for tier, tables in views:
        rules = tables[dt].get(key) or tables["half"].get(key)
        if rules is None:
            continue
        had_rules = True
        for m_max, choice in rules:
            if m_max is None or M <= m_max:
                if _is_compiled(choice, sm):
                    record_tier("gemm", op, tier)
                    return choice
                _note_uncompiled(tier, key, choice, sm)
                break
    if not had_rules:
        st = _RULE_MISSES.get(key)
        if st is None:
            _RULE_MISSES[key] = _RuleMiss(int(M))
        else:
            st.add(int(M))
        choice = _model_choice(op, int(M), int(N), int(K), sm)
        if choice is not None:
            record_tier("gemm", op, "model")
            return choice
    record_tier("gemm", op, "default")
    return GEMM_DEFAULT


#: ``OASR_TUNE_MODEL_FALLBACK``: ``auto`` (default) ranks a shape no tuning entry
#: covers with the cost model the arch's tuning file ships, when it ships one;
#: ``0`` keeps every such shape on :data:`GEMM_DEFAULT` (the A/B and rollback).
_MODEL_FALLBACK = os.environ.get("OASR_TUNE_MODEL_FALLBACK", "auto").strip().lower()

#: A model pick must beat the default by this factor *in the model's own
#: prediction* to replace it -- a predicted tie keeps today's behaviour.
_MODEL_MIN_GAIN = 1.05

#: ``(epoch, sm) -> (model | None, {op: {config_id: (params, choice)}})``.
_MODEL_VIEWS: Dict[Tuple[int, int], tuple] = {}


def _model_view(sm: int):
    from oasr.tune import database

    key = (database.epoch(), int(sm))
    view = _MODEL_VIEWS.get(key)
    if view is not None:
        return view
    from oasr.tune.cost_model import GemmCostModel

    model = None
    for _tier, tf in database.tiers("gemm", int(sm)).ordered():
        d = (tf.model or {}).get("gemm")
        if d:
            model = GemmCostModel.from_json(d)
            # The analytic structure generalises across widths; per-config
            # calibration does not (leave-one-signature-out top-1 regret up to
            # ~50% on small-K widths vs 0.4% geomean for the prior), so the
            # runtime tier ranks with the prior.
            model.coeffs = {}
            if int(sm) == _get_target_sm():
                with contextlib.suppress(Exception):
                    import torch

                    props = torch.cuda.get_device_properties(torch.cuda.current_device())
                    model.num_sms = int(props.multi_processor_count)
            break
    cands: Dict[str, Dict[str, tuple]] = {"gemm": {}, "gemm_activation": {}}
    if model is not None and sm in _STAGE_DOMAIN:
        for cfg in get_all_autotune_configs(sm).values():
            if not _is_compiled(cfg, sm):
                continue
            params = gemm_config_to_params(cfg, sentinel_default=False)
            cid = gemm_config_id(cfg) if cfg != GEMM_DEFAULT else "default"
            choice = GEMM_DEFAULT if cfg == GEMM_DEFAULT else cfg
            # Never serial split-K from an *unmeasured* tier: it round-trips the
            # partials through the output dtype, one rounding per slice, which a
            # measured entry is allowed only after the numerics gate -- the model
            # tier has none, and bf16 at split 4 missed a 1e-2 tolerance on
            # (64, 128, 256).  Parallel split-K reduces in fp32 and stays.
            if getattr(cfg, "split_k", 1) > 1 and not getattr(cfg, "parallel_split_k", False):
                continue
            cands["gemm"][cid] = (params, choice)
            cands["gemm_activation"][cid] = (params, choice)
    view = (model, cands)
    _MODEL_VIEWS.clear()
    _MODEL_VIEWS[key] = view
    return view


def _model_choice(op: str, M: int, N: int, K: int, sm: int):
    """The cost model's pick among compiled configs for an uncovered shape, or ``None``.

    Only ``gemm`` / ``gemm_activation`` on the aligned lane: the CTC head's
    fallback is the fused launcher the model does not describe, and the tuned
    ``bmm`` lane is keyed without its batch count.  A pure function of the shape
    and the snapshot, so capture and eager agree (rule 11).
    """
    if _MODEL_FALLBACK in ("0", "off", "false", "no") or op not in ("gemm", "gemm_activation"):
        return None
    if N % 8 or K % 8:
        return None
    model, cands = _model_view(sm)
    if model is None or not cands.get(op):
        return None
    params = {cid: pc[0] for cid, pc in cands[op].items()}
    ranked = model.rank(params, M, N, K)
    if not ranked:
        return None
    cid, t = ranked[0]
    t_default = model.predict_ms(
        "default", gemm_config_to_params(default_config_for_sm(sm), sentinel_default=False), M, N, K
    )
    if not (t * _MODEL_MIN_GAIN <= t_default):
        return None
    return cands[op][cid][1]


def covering_tier(op: str, M: int, N: int, K: int, dtype, sm: int) -> Optional[str]:
    """The tuning tier (``"user"`` / ``"system"``) whose entry covers this shape, or ``None``.

    Covered means an entry exists for the signature and one of its regions
    contains M -- i.e. the shape would be served from measured data rather than
    the cost model or the default.  ``dtype`` may be a torch dtype or its name.
    """
    name = str(dtype)
    if not name.startswith("torch."):
        name = f"torch.{name}"
    if name not in _DTYPE_CLASS:
        return None
    key = (op, int(N), int(K))
    for tier, tables in _tier_views(int(sm)):
        rules = tables[_DTYPE_CLASS[name]].get(key) or tables["half"].get(key)
        if rules and any(m_max is None or M <= m_max for m_max, _ in rules):
            return tier
    return None


#: ``str(torch dtype) -> the dtype class a tuning-DB entry is keyed under``.
_DTYPE_CLASS = {"torch.float16": "fp16", "torch.bfloat16": "bf16"}

_UNCOMPILED_WARNED: set = set()


def _note_uncompiled(tier: str, key, choice, sm: int) -> None:
    """A rule named a config this process does not build: say so once, then fall through.

    Reachable from the user tier -- a file tuned against a larger compile set,
    say -- and never from a shipped file, which a test holds to its arch's
    emitted set.  Serving it would raise ``AttributeError: Module has no
    function`` at dispatch; skipping it silently would hide a tuning file that
    no longer matches the build.
    """
    wkey = (tier, key, getattr(choice, "compile_name", choice), sm)
    if wkey in _UNCOMPILED_WARNED:
        return
    _UNCOMPILED_WARNED.add(wkey)
    logger.warning(
        "GEMM %s-tier rule for %s names %s, which sm%d does not compile; falling through",
        tier,
        key,
        getattr(choice, "compile_name", choice),
        sm,
    )
