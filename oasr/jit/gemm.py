# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for GEMM kernels (FlashInfer-style).

Tile configurations are defined here in the JIT layer, and ALL variants are
compiled into a single shared library per kernel family.  The autotuner
selects which pre-compiled variant to call — no JIT during tuning.
"""

import itertools
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from . import env
from .core import _TARGET_SMS, JitSpec, _get_target_sm, gen_jit_spec

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


#: Pipeline depths the two K-decomposition families are built at, per SM family.
#:
#: They mirror each architecture's own base stage list — Stream-K keeps the
#: single depth it has always been curated at, parallel split-K the arch's full
#: list — with one entry that is a hardware fact rather than a preference:
#: **Turing has no 3-stage tensor-op GEMM at all.**  Compiled here, `sm_75` at
#: three or four stages fails identically to the plain path,
#:
#:     default_gemm_universal.h(214): error: incomplete type
#:       "cutlass::gemm::kernel::DefaultGemmUniversal<...>"
#:
#: which is the same `kernel::DefaultGemm` 2-stage-only specialisation that makes
#: `RecurrentArch<75>` set `kStages = 2`.  Measured for every (arch, depth, family)
#: cell: 80/86/89/120 build at 2, 3 and 4; sm_75 builds at 2 and nothing else.
#:
#: A 2.x family with no entry raises `KeyError` at config-generation time, which
#: is the intended failure: silently receiving no decompositions is how this
#: became an sm_120-only feature in the first place.
_SM_STREAMK_STAGES: Dict[int, List[int]] = {75: [2], 80: [3], 86: [3], 89: [3], 120: [3]}
_SM_SPLITK_PARALLEL_STAGES: Dict[int, List[int]] = {
    75: [2],
    80: [3, 4],
    86: [3],
    89: [3],
    120: [3, 4],
}


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


def _get_sm75_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM75 (Turing): kStages ∈ {2,3}, tiles from _GEMM_TILES.

    The ``[2, 3]`` is a live defect, not a description: Turing's
    ``kernel::DefaultGemm`` tensor-op specialisation exists at two stages and no
    other, so every ``_s3`` variant this emits fails to compile and takes the
    whole module with it (audit A10 — untouched here, a different concern).  The
    K-decompositions added below are deliberately *not* built at three stages for
    exactly that reason, so this change adds no new broken TU to Turing.
    """
    cfgs = _build_sm_lt90_configs(sm, _GEMM_TILES, [2, 3], _SPLIT_K_LIST, _SM_MAX_SMEM_BYTES[75])
    return _add_k_decompositions(cfgs, sm, _SM_MAX_SMEM_BYTES[75])


def _get_sm80_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM80 (Ampere A100): kStages ∈ {3,4}, tiles from _GEMM_TILES."""
    cfgs = _build_sm_lt90_configs(sm, _GEMM_TILES, [3, 4], _SPLIT_K_LIST, _SM_MAX_SMEM_BYTES[80])
    return _add_k_decompositions(cfgs, sm, _SM_MAX_SMEM_BYTES[80])


def _get_sm86_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM86 (Ampere RTX 30-series): kStages=3, tiles from _GEMM_TILES."""
    cfgs = _build_sm_lt90_configs(sm, _GEMM_TILES, [3], _SPLIT_K_LIST, _SM_MAX_SMEM_BYTES[86])
    return _add_k_decompositions(cfgs, sm, _SM_MAX_SMEM_BYTES[86])


def _get_sm89_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM89 (Ada Lovelace): kStages=3, tiles from _GEMM_TILES."""
    cfgs = _build_sm_lt90_configs(sm, _GEMM_TILES, [3], _SPLIT_K_LIST, _SM_MAX_SMEM_BYTES[89])
    return _add_k_decompositions(cfgs, sm, _SM_MAX_SMEM_BYTES[89])


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


def _get_sm120_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM120 (GeForce Blackwell / RTX 50 series) configs.

    The CUTLASS 3.x SM120 CollectiveBuilder supports only F8/F6/F4 MMA, so
    FP16/BF16 GEMM on SM120 is routed through the CUTLASS 2.x tensor-op path
    using the Sm80 forward-compatible instructions (mma.sync.aligned.m16n8k16).

    Also includes Stream-K and parallel split-K variants (gemm family only —
    see ``_render_all_variants`` and the backend registration, which confine
    them to GEMM).
    """
    cfgs = _build_sm_lt90_configs(sm, _GEMM_TILES, [3, 4], _SPLIT_K_LIST, _SM_MAX_SMEM_BYTES[120])
    return _add_k_decompositions(cfgs, sm, _SM_MAX_SMEM_BYTES[120])


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
    if sm == 75:
        return _get_sm75_configs(sm)  # type: ignore[return-value]
    elif sm == 80:
        return _get_sm80_configs(sm)  # type: ignore[return-value]
    elif sm == 86:
        return _get_sm86_configs(sm)  # type: ignore[return-value]
    elif sm == 89:
        return _get_sm89_configs(sm)  # type: ignore[return-value]
    elif sm == 90:
        return _get_sm90_configs(sm)  # type: ignore[return-value]
    elif sm == 100:
        return _get_sm100_configs(sm)  # type: ignore[return-value]
    elif sm == 120:
        return _get_sm120_configs(sm)  # type: ignore[return-value]
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


# =============================================================================
# Helper: render all tile variants for a given template
# =============================================================================


def _render_all_variants(
    template_name: str,
    template_sm90_name: str,
    family: str,
    *,
    with_activation: bool = False,
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
    unique_configs = get_unique_compile_configs(sm)
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


def gen_gemm_module() -> JitSpec:
    """Generate JIT spec for GEMM with ALL tile variants in one module.

    Each variant exports ``gemm_{config_name}`` and ``gemm_{config_name}_activation``
    as TVM-FFI functions.  The autotuner selects which to call; the default path
    uses ``GEMM_DEFAULT``.
    """
    source_paths = _render_all_variants(
        "gemm_cutlass_template.cu.jinja",
        "gemm_cutlass_template_sm90.cu.jinja",
        "gemm",
        with_activation=True,
    )
    # Plus the workspace-cache diagnostics (``ws_cache_keys`` /
    # ``ws_cache_bytes``).  One extra TU, not part of the rendered template,
    # which is compiled once per tile configuration.
    source_paths = source_paths + [env.OASR_CSRC_DIR / "gemm_ws_cache.cu"]
    return gen_jit_spec("gemm", source_paths)


def gen_bmm_module() -> JitSpec:
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
    )
    source_paths = (
        source_paths
        + _render_bmm_general_variants()
        + [
            env.OASR_CSRC_DIR / "bmm.cu",
            env.OASR_CSRC_DIR / "bmm_jit_binding.cu",
        ]
    )
    return gen_jit_spec("bmm", source_paths)


def gen_group_gemm_module() -> JitSpec:
    """Generate JIT spec for grouped GEMM with ALL tile variants in one module.

    Each variant exports ``group_gemm_{config_name}`` as a TVM-FFI function.
    """
    source_paths = _render_all_variants(
        "group_gemm_cutlass_template.cu.jinja",
        "group_gemm_cutlass_template_sm90.cu.jinja",
        "group_gemm",
    )
    return gen_jit_spec("group_gemm", source_paths)


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


# SM120 production rules generated by ``scripts/tune_asr_gemm.py``.
# Keys are exact ``(op, N, K)`` tuples; values are ascending ``(m_max, choice)``
# entries with an optional catch-all. Misses use ``GEMM_DEFAULT`` and are counted
# by ``rule_miss_report()`` because rules do not transfer across model widths.
#
# One table per SM family, registered in ``_GEMM_HEURISTIC_RULES`` below.  The
# tuner already emits this literal named for the arch it measured
# (``emit_rules`` writes ``_GEMM_HEURISTIC_RULES_SM<sm>``), so a second
# architecture is a paste plus one registry line -- not an edit to
# ``select_default_config``.
_GEMM_HEURISTIC_RULES_SM120: Dict[Tuple[str, int, int], list] = {
    # Thin-N contraction; a smaller tile avoids wasted columns.  Zipformer's
    # ConvNeXt pointwise contraction (384 -> 128), whose M is
    # ``batch * embed_frames * 19``.
    #
    # The M <= 8192 bucket used to be split, with M in (1024, 2048] assigned
    # ``b128x16x64_w32x16x64_s4``.  That tile is unbuildable (see
    # ``_epilogue_covers_warp``) and it was also *slower* than the tile on both
    # sides of it -- graph-captured at N=128 K=384, ``b32x64x64_s4`` runs
    # 2.87/2.97/3.22/3.52 us at M=1026/1254/1710/2014 against the thin tile's
    # 3.36/3.34/3.78/4.13, so the bucket is merged rather than re-tuned.  A rule
    # generated from timings alone could see neither problem, which is why
    # tests/kernels/test_gemm_heuristic.py now asks this shape's selection
    # whether the tile it picked is *addressable* and not only whether it is
    # compiled.  The timings are a four-point measurement recorded in
    # .artifacts/gemm_thin_n_tile_epilogue.md, not a test.
    ("gemm", 128, 384): [
        (
            8192,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (None, GEMM_DEFAULT),
    ],
    ("gemm", 256, 256): [
        (
            512,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~416: cutlass 0.0061ms (2.00x vs default)
        (
            1024,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~896: cutlass 0.0062ms (1.99x vs default)
        (
            2048,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~1992: cutlass 0.0082ms (1.50x vs default)
        (None, "torch"),  # M~15872: torch 0.0205ms (1.10x vs default)
    ],
    ("gemm", 256, 2048): [
        (64, "torch"),  # M~48: torch 0.0082ms (6.25x vs default)
        (
            128,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=8,
                parallel_split_k=True,
            ),
        ),  # M~128: cutlass 0.0088ms (5.84x vs default)
        (512, "torch"),  # M~416: torch 0.0102ms (5.00x vs default)
        (
            1024,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=2,
            ),
        ),  # M~896: cutlass 0.0143ms (3.57x vs default)
        (
            2048,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=2,
            ),
        ),  # M~1992: cutlass 0.0205ms (2.50x vs default)
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=128,
                block_k=64,
                warp_m=32,
                warp_n=64,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~15872: cutlass 0.0901ms (1.17x vs default)
    ],
    ("gemm", 256, 4864): [
        (
            16,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=8,
                parallel_split_k=True,
            ),
        ),  # M~16: cutlass 0.0107ms (10.54x vs default)
        (1024, "torch"),  # M~896: torch 0.0225ms (5.09x vs default)
        (
            2048,
            CutlassGemmConfig(
                block_m=64,
                block_n=128,
                block_k=64,
                warp_m=32,
                warp_n=64,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
                stream_k=True,
            ),
        ),  # M~1992: cutlass 0.0370ms (3.10x vs default)
        (
            None,
            CutlassGemmConfig(
                block_m=128,
                block_n=128,
                block_k=64,
                warp_m=64,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=2,
            ),
        ),  # M~15872: cutlass 0.2112ms (1.25x vs default)
    ],
    # Width-384 projections. Small AR steps and unaligned vocabulary heads bypass
    # these rules; boundaries cover measured fixed-window encoder batches.
    ("gemm", 384, 384): [
        (
            2048,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M 128/256/1500: 8.2/8.2/9.6us (2.00x/2.00x/1.70x vs default)
        (
            4096,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M 3000 (batch 2): 12.3us (1.50x vs default)
        # M 6000 (batch 4) is the one cell where the 128x128 default really is
        # the best tile of the candidate space — 18.4us vs 20.5 for cuBLAS.  The
        # entry exists to *stop* this M reaching the catch-all, not to change it.
        (8192, GEMM_DEFAULT),  # M 6000: 18.4us (1.11x vs torch)
        (None, "torch"),  # M 12000/24000/48000/96000: 1.06x/1.11x/1.13x/1.14x
    ],
    # Deep-K projection: use the lower-overhead launcher at small M and the faster
    # library backend once GPU work dominates dispatch cost.
    ("gemm", 384, 1536): [
        (256, "torch"),  # B=32/64 decoder prefill: 10.2/12.3us (4.60x/3.83x); GPU-bound
        (
            2048,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M 1500 (batch 1): 18.4us, == cuBLAS on GPU, ~5us/call cheaper to issue
        (
            4096,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M 3000 (batch 2): 30.7us, == cuBLAS on GPU (1.60x vs default)
        (8192, GEMM_DEFAULT),  # M 6000 (batch 4): 51.3us, best of the three
        (None, "torch"),  # M 12000/24000/48000/96000: 1.05x/1.13x/1.09x/1.07x
    ],
    ("gemm", 512, 256): [
        (
            1024,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~780: cutlass 0.0082ms (1.50x vs default)
        (
            2048,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~1680: cutlass 0.0082ms (1.49x vs default)
        (
            4096,
            CutlassGemmConfig(
                block_m=128,
                block_n=64,
                block_k=64,
                warp_m=64,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~2104: cutlass 0.0096ms (1.27x vs default)
        (16384, "torch"),  # M~10368: torch 0.0205ms (1.10x vs default)
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~16768: cutlass 0.0348ms (1.12x vs default)
    ],
    # whisper-tiny FF up-projection (see the (384, *) keys above).
    ("gemm", 1536, 384): [
        (
            256,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M 128/256: 8.2/8.2us (2.01x/2.00x vs default, 1.24x vs torch at 256)
        # Boundary entry: at M=1500 the default ties cuBLAS (18.4us both) and the
        # next rule's tile costs 20.5, so what this pins is the *edge*.
        (2048, GEMM_DEFAULT),  # M 1500 (batch 1): 18.4us
        (
            16384,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M 3000/6000/12000: 1.07x/1.15x/1.04x vs default
        (None, "torch"),  # M 24000/48000/96000: 1.07x/1.06x/1.11x
    ],
    # LSTM/RNN gate projection: N = gates * hidden, K = hidden.  This key was a
    # silent rule miss until 2026-08-23 -- ``oasr.tune.capture`` only rebound the
    # ``oasr`` package attributes and the recurrent functional imports ``gemm``
    # from its defining module, so the shape never appeared in a captured
    # workload and the whole M range sat on GEMM_DEFAULT's 128x128 tile at 17us.
    #
    # Measured GPU-only (a 100-call loop captured in one CUDA graph, arms
    # round-robined over 9 reps, sigma <= 0.05us; fp16 and bf16 agree to 3%).  A
    # back-to-back launch loop -- what ``scripts/tune_asr_gemm.py`` measures --
    # cannot resolve this: every arm here is faster than the ~9.6us it costs to
    # *issue* a GEMM call, so the loop reads 10-20us for all of them and reported
    # 2.0x where the truth is 4.6x, picking the wrong tile twice.
    ("gemm", 2560, 640): [
        (
            128,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M 64/128: 3.64/4.91us vs default 16.72/17.14 (4.6x/3.5x)
        (
            768,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M 256/384/512/640/768: 2.66x/1.74x/1.62x/1.23x/1.20x vs default.
        # At M=512 cuBLAS is 2.7% ahead (10.36 vs 10.65) -- a tie, and encoding
        # it would cost a boundary that can be wrong.  Above 768 the default's
        # 128-row tiles finally fill and cuBLAS leads it by only 1.01-1.05x, under
        # the 1.05x bar the tuner uses, so the catch-all stays.
        (None, GEMM_DEFAULT),  # M 1024/2048/4096: 1.01x/1.05x/1.03x for cuBLAS
    ],
    ("gemm_activation", 2048, 256): [
        (
            64,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~48: cutlass 0.0062ms (1.98x vs default)
        (
            128,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~128: cutlass 0.0062ms (1.98x vs default)
        (
            256,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~176: cutlass 0.0082ms (1.50x vs default)
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~416: cutlass 0.0102ms (1.39x vs default)
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~15872: cutlass 0.1044ms (1.18x vs default)
    ],
    ("gemm_log_softmax", 5008, 256): [
        (64, "fused"),  # M~48: cutlass_fused 0.0102ms (1.00x vs default)
        (
            128,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~128: cutlass 0.0123ms (1.17x vs default)
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),  # M~15872: cutlass 0.4054ms (1.80x vs default)
    ],
    # ── Zipformer widths, from the 2026-08-24 capture-driven sweep
    # (scripts/tune_asr_gemm.py over 165 representative shapes captured from real
    # checkpoints).  A coverage census found Zipformer running GEMM_DEFAULT on 49
    # distinct (op, N, K) keys — every one of its own widths — against 1 for
    # Conformer.  These are those keys.
    #
    # Each arm cleared the tuner's self-overlap gate: faster both back-to-back in
    # one graph AND on a single replay, so none is a low-occupancy tile that only
    # wins by overlapping with its own next launch.  Kernel-level 1.07-8.63x.
    #
    # End-to-end, offline batch 64, 64 LJSpeech utterances, 6 interleaved arms of
    # 25 reps: 1.0192x on min, 1.0168x on p25, 1.0151x on median, transcripts
    # identical.  Modest because the encoder is CPU-issue-bound — which is also
    # why the sweep's other 16 keys were measured and NOT kept: conformer 1.003x,
    # paraformer 1.006x, whisper 1.002x, nemotron 0.993-1.004x, all inside their
    # own sigma, and the nemotron ones moved three commas for no gain.
    ("gemm", 16, 48): [
        (256, GEMM_DEFAULT),
        (None, "torch"),
    ],
    ("gemm", 48, 192): [
        (
            512,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 48, 256): [
        (
            256,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 48, 512): [
        (
            None,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 96, 768): [
        (
            None,
            CutlassGemmConfig(
                block_m=16,
                block_n=128,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=4,
                parallel_split_k=True,
            ),
        ),
    ],
    ("gemm", 192, 48): [
        (512, "torch"),
        (None, GEMM_DEFAULT),
    ],
    ("gemm", 192, 144): [
        (
            512,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 192, 192): [
        (
            512,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 192, 384): [
        (
            512,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 192, 512): [
        (
            512,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 192, 640): [
        (
            512,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 192, 2432): [
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=2,
                parallel_split_k=True,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 256, 48): [
        (256, "torch"),
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 256, 192): [
        (
            256,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 256, 576): [
        (
            256,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 256, 768): [
        (
            256,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 256, 960): [
        (
            256,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 272, 192): [
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 272, 256): [
        (
            256,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 272, 512): [
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 384, 128): [
        (
            2048,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (8192, GEMM_DEFAULT),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 384, 192): [
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 432, 192): [
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 512, 48): [
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 512, 192): [
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 512, 384): [
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=128,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 512, 512): [
        (
            256,
            CutlassGemmConfig(
                block_m=16,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            1024,
            CutlassGemmConfig(
                block_m=32,
                block_n=128,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 512, 1152): [
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=128,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=4,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 512, 1536): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 512, 1920): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 544, 768): [
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 576, 256): [
        (
            256,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 640, 192): [
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (None, GEMM_DEFAULT),
    ],
    ("gemm", 768, 256): [
        (
            256,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            512,
            CutlassGemmConfig(
                block_m=32,
                block_n=128,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 768, 576): [
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 768, 768): [
        (
            None,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 768, 1536): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=2,
                parallel_split_k=True,
            ),
        ),
    ],
    ("gemm", 768, 2048): [
        (None, "torch"),
    ],
    ("gemm", 768, 2560): [
        (None, "torch"),
    ],
    ("gemm", 960, 256): [
        (
            256,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 1024, 512): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 1152, 512): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 1536, 512): [
        (
            1024,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (None, GEMM_DEFAULT),
    ],
    ("gemm", 1536, 768): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 1728, 768): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 1920, 512): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 2048, 768): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm", 2560, 768): [
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
    ("gemm_log_softmax", 504, 768): [
        (64, "fused"),
        (
            256,
            CutlassGemmConfig(
                block_m=32,
                block_n=64,
                block_k=64,
                warp_m=16,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
        (
            None,
            CutlassGemmConfig(
                block_m=64,
                block_n=64,
                block_k=64,
                warp_m=32,
                warp_n=32,
                warp_k=64,
                kStages=3,
                kSmVersion=120,
                split_k=1,
            ),
        ),
    ],
}

#: Tuned rule tables by compiled SM family (``oasr.jit.core._SM_FAMILY``).
#:
#: The heuristic used to be gated on ``sm != 120`` in ``select_default_config``,
#: which made "which architectures are tuned?" a control-flow question with one
#: possible answer.  Here it is data, and the answer is this dict's keys.
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
#: this file's history; ``scripts/tune_asr_gemm.py`` on the target card is the
#: only supported way in.
_GEMM_HEURISTIC_RULES: Dict[int, Dict[Tuple[str, int, int], list]] = {
    120: _GEMM_HEURISTIC_RULES_SM120,
}

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
    table = _GEMM_HEURISTIC_RULES.get(int(sm))
    if table is None:
        _ARCH_INACTIVE[int(sm)] = _ARCH_INACTIVE.get(int(sm), 0) + 1
        return GEMM_DEFAULT
    rules = table.get((op, int(N), int(K)))
    if rules is None:
        key = (op, int(N), int(K))
        st = _RULE_MISSES.get(key)
        if st is None:
            _RULE_MISSES[key] = _RuleMiss(int(M))
        else:
            st.add(int(M))
        return GEMM_DEFAULT
    for m_max, choice in rules:
        if m_max is None or M <= m_max:
            return choice
    return GEMM_DEFAULT
