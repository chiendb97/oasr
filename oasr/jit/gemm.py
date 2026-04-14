# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for GEMM kernels (FlashInfer-style).

Tile configurations are defined here in the JIT layer, and ALL variants are
compiled into a single shared library per kernel family.  The autotuner
selects which pre-compiled variant to call — no JIT during tuning.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from .core import gen_jit_spec, _get_target_sm, JitSpec
from . import env


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
    """
    block_m: int
    block_n: int
    block_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    kStages: int
    kSmVersion: int
    split_k: int = 1          # runtime split-K factor (1 = disabled)

    @property
    def name(self) -> str:
        """Unique identifier for this config (includes all params, split_k)."""
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.block_m}x{self.block_n}x{self.block_k}")
        parts.append(f"w{self.warp_m}x{self.warp_n}x{self.warp_k}")
        parts.append(f"s{self.kStages}")
        if self.split_k != 1:
            parts.append(f"spk{self.split_k}")
        return "_".join(parts)

    @property
    def compile_name(self) -> str:
        """Config name used as the compiled binary key.

        Encodes tile shape, warp shape, and kStages (all compile-time
        template parameters).  Excludes ``split_k`` (runtime argument)
        so variants differing only in split-K share a single compiled ``.so``.
        """
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.block_m}x{self.block_n}x{self.block_k}")
        parts.append(f"w{self.warp_m}x{self.warp_n}x{self.warp_k}")
        parts.append(f"s{self.kStages}")
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
    tile_k: int              # 128 for SM90/SM120 (WGMMA width)
    cluster_m: int
    cluster_n: int
    pingpong: bool           # True = Pingpong, False = Cooperative (SM90/SM120)
    is_dynamic_persistent: bool  # Dynamic persistent / CLC scheduler (SM100)
    swap_ab: bool            # Swap A and B operands
    max_swizzle_size: int    # Max swizzle size for SMEM layout
    use_tma_gather: bool     # TMA gather for A (SM100 only)
    kSMs: int                # 1 or 2 (SM100 only; always 1 for SM90/SM120)
    kStages: int             # Pipeline stages (typically 3)
    kSmVersion: int          # 90, 100, or 120

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
TileShapeConfigs: List[TileShape] = [
    TileShape(block_m=16,  block_n=128, block_k=64, warp_m=16,  warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=16,  block_k=64, warp_m=32,  warp_n=16, warp_k=64),
    TileShape(block_m=32,  block_n=128, block_k=64, warp_m=32,  warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=32,  block_k=64, warp_m=32,  warp_n=32, warp_k=64),
    TileShape(block_m=64,  block_n=128, block_k=64, warp_m=32,  warp_n=64, warp_k=64),
    TileShape(block_m=128, block_n=64,  block_k=64, warp_m=64,  warp_n=32, warp_k=64),
    TileShape(block_m=64,  block_n=128, block_k=64, warp_m=64,  warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=64,  block_k=64, warp_m=64,  warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=128, block_k=64, warp_m=64,  warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=128, block_k=64, warp_m=64,  warp_n=64, warp_k=64),
    TileShape(block_m=128, block_n=128, block_k=64, warp_m=128, warp_n=32, warp_k=64),
    TileShape(block_m=128, block_n=256, block_k=64, warp_m=64,  warp_n=64, warp_k=64),
    TileShape(block_m=256, block_n=128, block_k=64, warp_m=64,  warp_n=64, warp_k=64),
    TileShape(block_m=16,  block_n=256, block_k=64, warp_m=16,  warp_n=64, warp_k=64),
    TileShape(block_m=256, block_n=16,  block_k=64, warp_m=64,  warp_n=16, warp_k=64),
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


# Maximum shared memory per threadblock per architecture (bytes).
# CUTLASS 2.x opts in to the maximum via cudaFuncSetAttribute at runtime.
_SM_MAX_SMEM_BYTES: Dict[int, int] = {
    75: 64  * 1024,   # Turing
    80: 164 * 1024,   # Ampere A100
    86: 100 * 1024,   # Ampere RTX 30-series
    89: 100 * 1024,   # Ada Lovelace
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
    Two constraints are applied:

    1. **SMEM fit** — kStages×(block_m+block_n)×block_k×dtype_bytes ≤ smem_limit.
       Software-pipelined operand buffers must fit in shared memory.
    2. **split_k applicability** — split_k>1 is only registered when
       block_m≤128 and block_n≤128 (shapes likely to be K-bound).

    Divisibility and warp-count validity are guaranteed by ``TileShapeConfigs``.

    The returned dict is keyed by ``CutlassGemmConfig.name`` (which includes
    split_k) for use in autotuner registration.  Callers that need only the
    compiled-binary set should deduplicate by ``compile_name``.
    """
    seen: Dict[str, CutlassGemmConfig] = {}
    for tile in tiles:
        for kStages in stage_list:
            # 1. SMEM fit
            if _smem_bytes(tile.block_m, tile.block_n, tile.block_k, kStages) > smem_limit:
                continue
            for split_k in split_k_list:
                # 2. split_k applicability
                if split_k > 1 and (tile.block_m > 128 or tile.block_n > 128):
                    continue
                cfg = CutlassGemmConfig(
                    block_m=tile.block_m, block_n=tile.block_n, block_k=tile.block_k,
                    warp_m=tile.warp_m, warp_n=tile.warp_n, warp_k=tile.warp_k,
                    kStages=kStages,
                    kSmVersion=sm,
                    split_k=split_k,
                )
                key = cfg.name
                if key not in seen:
                    seen[key] = cfg
    return seen


def _get_sm75_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM75 (Turing): kStages ∈ {2,3}, split_k ∈ {1,2,4}, tiles from TileShapeConfigs."""
    return _build_sm_lt90_configs(sm, TileShapeConfigs, [2, 3], [1, 2, 4], _SM_MAX_SMEM_BYTES[75])


def _get_sm80_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM80 (Ampere A100): kStages ∈ {3,4}, split_k ∈ {1,2,4}, tiles from TileShapeConfigs."""
    return _build_sm_lt90_configs(sm, TileShapeConfigs, [3, 4], [1, 2, 4], _SM_MAX_SMEM_BYTES[80])


def _get_sm86_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM86 (Ampere RTX 30-series): kStages=3, split_k ∈ {1,2,4}, tiles from TileShapeConfigs."""
    return _build_sm_lt90_configs(sm, TileShapeConfigs, [3], [1, 2, 4], _SM_MAX_SMEM_BYTES[86])


def _get_sm89_configs(sm: int) -> Dict[str, CutlassGemmConfig]:
    """SM89 (Ada Lovelace): kStages=3, split_k ∈ {1,2,4}, tiles from TileShapeConfigs."""
    return _build_sm_lt90_configs(sm, TileShapeConfigs, [3], [1, 2, 4], _SM_MAX_SMEM_BYTES[89])


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

    # Cooperative (non-pingpong) tile shapes
    tile_mn_coop = [
        (256, 128), (256, 160), (256, 192), (256, 208),
        (128, 224), (128, 256),
    ]
    # Pingpong tile shapes
    tile_mn_pingpong = [
        (128, 128), (128, 160), (128, 192), (128, 208),
        (192, 128),
    ]
    tile_mn_vals = (
        [(m, n, False) for m, n in tile_mn_coop] +
        [(m, n, True) for m, n in tile_mn_pingpong]
    )
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
        # kSMs=2 enables 2-SM co-operative TileShape (BM*2 × BN) when cluster_m ≥ 2
        kSMs = 2 if cluster_m >= 2 else 1
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


def _get_sm120_configs(sm: int) -> Dict[str, CutlassGemmConfigSm90]:
    """SM120 (GeForce Blackwell) configs following Quack's ``_get_sm120_configs()`` pattern.

    Uses warp-level MMA (similar to SM90 but with warp-level MMA instead of
    WGMMA). Cluster is always 1×1. Both Cooperative and Pingpong variants.
    """
    tile_k = 128
    kStages = 3

    tile_mn_coop = [(128, 128), (128, 64), (64, 128), (128, 160), (128, 192)]
    tile_mn_pingpong = [(128, 128), (128, 64), (64, 128), (128, 160)]
    tile_mn_vals = (
        [(m, n, False) for m, n in tile_mn_coop] +
        [(m, n, True) for m, n in tile_mn_pingpong]
    )

    seen: Dict[str, CutlassGemmConfigSm90] = {}
    for tile_m, tile_n, pingpong in tile_mn_vals:
        cfg = CutlassGemmConfigSm90(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            cluster_m=1,
            cluster_n=1,
            pingpong=pingpong,
            is_dynamic_persistent=True,
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


def get_all_autotune_configs(
    sm: int,
) -> Dict[str, Union[CutlassGemmConfig, CutlassGemmConfigSm90]]:
    """Return the **full** autotuner config set for *sm* (keyed by ``name``).

    For SM < 90 this includes all split_k and kStages variants; for SM ≥ 90
    it matches the Quack-style set (split_k is not applicable there).
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
    else:
        return _get_sm120_configs(sm)  # type: ignore[return-value]


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
    from .templates import render_template
    from .cubin_loader import write_if_different

    sm = _get_target_sm()
    unique_configs = get_unique_compile_configs(sm)
    source_paths = []

    for config_name, cfg in unique_configs.items():
        func_name = f"{family}_{config_name}"
        variant_file_name = f"{family}_sm{sm}_{config_name}"

        if sm in [75, 80, 86, 89]:
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
    return gen_jit_spec("gemm", source_paths)


def gen_bmm_module() -> JitSpec:
    """Generate JIT spec for BMM with ALL tile variants in one module.

    Each variant exports ``bmm_{config_name}`` as a TVM-FFI function.
    """
    source_paths = _render_all_variants(
        "bmm_cutlass_template.cu.jinja",
        "bmm_cutlass_template_sm90.cu.jinja",
        "bmm",
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
# Default configs (used by non-autotuned paths in oasr/gemm.py)
# =============================================================================

_sm = _get_target_sm()

if _sm < 90:
    GEMM_DEFAULT: Union[CutlassGemmConfig, CutlassGemmConfigSm90] = CutlassGemmConfig(
        block_m=128, block_n=128, block_k=64, warp_m=64, warp_n=64, warp_k=64, kStages=3, kSmVersion=_sm
    )
else:
    GEMM_DEFAULT = CutlassGemmConfigSm90(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
        swap_ab=False,
        max_swizzle_size=8,
        use_tma_gather=False,
        kSMs=1,
        kStages=3,
        kSmVersion=_sm,
    )
