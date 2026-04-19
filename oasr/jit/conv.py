# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for convolution kernels.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from .core import gen_jit_spec, _get_target_sm, JitSpec
from .gemm import TileShape, TileShapeConfigs, _smem_bytes, _SM_MAX_SMEM_BYTES
from . import env


# =============================================================================
# Conv2D config dataclasses  (mirror CutlassGemmConfig / CutlassGemmConfigSm90)
# =============================================================================


@dataclass(frozen=True)
class CutlassConv2dConfig:
    """CUTLASS 2.x Conv2D config for SM75–89 (implicit GEMM).

    Mirrors ``CutlassGemmConfig`` field-for-field; the only difference is the
    absence of ``split_k`` — CUTLASS 2.x implicit GEMM does not provide a
    split-K splitter, so ``name == compile_name`` for Conv2D.
    """
    block_m: int
    block_n: int
    block_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    kStages: int
    kSmVersion: int

    @property
    def name(self) -> str:
        return self.compile_name

    @property
    def compile_name(self) -> str:
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.block_m}x{self.block_n}x{self.block_k}")
        parts.append(f"w{self.warp_m}x{self.warp_n}x{self.warp_k}")
        parts.append(f"s{self.kStages}")
        return "_".join(parts)

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        return tuple([
            ("block_m", self.block_m), ("block_n", self.block_n), ("block_k", self.block_k),
            ("warp_m", self.warp_m), ("warp_n", self.warp_n), ("warp_k", self.warp_k),
            ("kStages", self.kStages),
        ])


@dataclass(frozen=True)
class CutlassConv2dConfigSm90:
    """CUTLASS 3.x Conv2D config for SM90, SM100, and SM120.

    Mirrors ``CutlassGemmConfigSm90`` field-for-field.  Omitted GEMM-only
    fields: ``is_dynamic_persistent``, ``swap_ab``, ``max_swizzle_size``,
    ``use_tma_gather`` (no Conv2D equivalents).  No ``cluster_k`` — CK is
    always 1 for implicit-GEMM convolution and is hardcoded in the C++ struct.

    Conv2D has no runtime-only parameters, so ``name == compile_name``.
    """
    tile_m: int
    tile_n: int
    tile_k: int             # 128 for SM90/SM120 (WGMMA width), matching GEMM
    cluster_m: int
    cluster_n: int
    pingpong: bool          # True = Pingpong, False = Cooperative (SM90/SM120)
    kSMs: int               # 1 or 2 (SM100 only; always 1 for SM90/SM120)
    kStages: int
    kSmVersion: int         # 90, 100, or 120

    @property
    def name(self) -> str:
        return self.compile_name

    @property
    def compile_name(self) -> str:
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.tile_m}x{self.tile_n}x{self.tile_k}")
        parts.append(f"c{self.cluster_m}x{self.cluster_n}")
        parts.append(f"k{self.kSMs}")
        parts.append(f"s{self.kStages}")
        parts.append("pp" if self.pingpong else "coop")
        return "_".join(parts)

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        return tuple([
            ("tile_m", self.tile_m), ("tile_n", self.tile_n), ("tile_k", self.tile_k),
            ("cluster_m", self.cluster_m), ("cluster_n", self.cluster_n),
            ("pingpong", int(self.pingpong)),
            ("kSMs", self.kSMs), ("kStages", self.kStages),
        ])


# =============================================================================
# SM<90 config generation — SMEM-analysed per-SM tile × stage
#
# Uses TileShapeConfigs directly from gemm.py (same 15 tiles as GEMM) and the
# same _smem_bytes / _SM_MAX_SMEM_BYTES limits.
# =============================================================================


def _build_sm_lt90_conv2d_configs(
    sm: int,
    tiles: List[TileShape],
    stage_list: List[int],
    smem_limit: int,
) -> Dict[str, CutlassConv2dConfig]:
    """Build the full autotune config dict for a SM<90 Conv2D architecture.

    Identical logic to GEMM's ``_build_sm_lt90_configs`` but without split_k.
    """
    seen: Dict[str, CutlassConv2dConfig] = {}
    for tile in tiles:
        for kStages in stage_list:
            if _smem_bytes(tile.block_m, tile.block_n, tile.block_k, kStages) > smem_limit:
                continue
            cfg = CutlassConv2dConfig(
                block_m=tile.block_m, block_n=tile.block_n, block_k=tile.block_k,
                warp_m=tile.warp_m, warp_n=tile.warp_n, warp_k=tile.warp_k,
                kStages=kStages, kSmVersion=sm,
            )
            key = cfg.compile_name
            if key not in seen:
                seen[key] = cfg
    return seen


def _get_sm75_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfig]:
    """SM75 (Turing): kStages ∈ {2, 3}, tiles from TileShapeConfigs."""
    return _build_sm_lt90_conv2d_configs(sm, TileShapeConfigs, [2, 3], _SM_MAX_SMEM_BYTES[75])


def _get_sm80_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfig]:
    """SM80 (Ampere A100): kStages ∈ {3, 4}, tiles from TileShapeConfigs."""
    return _build_sm_lt90_conv2d_configs(sm, TileShapeConfigs, [3, 4], _SM_MAX_SMEM_BYTES[80])


def _get_sm86_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfig]:
    """SM86 (Ampere RTX 30-series): kStages = 3, tiles from TileShapeConfigs."""
    return _build_sm_lt90_conv2d_configs(sm, TileShapeConfigs, [3], _SM_MAX_SMEM_BYTES[86])


def _get_sm89_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfig]:
    """SM89 (Ada Lovelace): kStages = 3, tiles from TileShapeConfigs."""
    return _build_sm_lt90_conv2d_configs(sm, TileShapeConfigs, [3], _SM_MAX_SMEM_BYTES[89])


# =============================================================================
# Quack-style SM90 / SM100 / SM120 Conv2D config generation
#
# Tile sets and cluster shapes mirror GEMM exactly.  tile_k=128 matches GEMM's
# WGMMA width.  Schedule selection uses GemmScheduleSelector via
# CutlassConv2dConfigSm90 (cooperative / pingpong for SM90/SM120; 1SM / 2SM
# for SM100), identical to CutlassGemmConfigSm90.
# =============================================================================


def _get_sm90_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfigSm90]:
    """SM90 configs following GEMM's ``_get_sm90_configs()`` pattern."""
    tile_k = 128
    kStages = 3

    tile_mn_coop = [
        (256, 128), (256, 160), (256, 192), (256, 208),
        (128, 224), (128, 256),
    ]
    tile_mn_pingpong = [
        (128, 128), (128, 160), (128, 192), (128, 208),
        (192, 128),
    ]
    tile_mn_vals = (
        [(m, n, False) for m, n in tile_mn_coop] +
        [(m, n, True) for m, n in tile_mn_pingpong]
    )
    cluster_vals = [(1, 2), (2, 1)]

    seen: Dict[str, CutlassConv2dConfigSm90] = {}
    for (tile_m, tile_n, pingpong), (cluster_m, cluster_n) in itertools.product(
        tile_mn_vals, cluster_vals
    ):
        cfg = CutlassConv2dConfigSm90(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            cluster_m=cluster_m, cluster_n=cluster_n,
            pingpong=pingpong, kSMs=1, kStages=kStages, kSmVersion=sm,
        )
        key = cfg.compile_name
        if key not in seen:
            seen[key] = cfg
    return seen


def _get_sm100_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfigSm90]:
    """SM100 configs following GEMM's ``_get_sm100_configs()`` pattern."""
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

    seen: Dict[str, CutlassConv2dConfigSm90] = {}
    for tile_m, tile_n, (cluster_m, cluster_n) in tile_mn_cluster_vals:
        kSMs = 2 if cluster_m >= 2 else 1
        cfg = CutlassConv2dConfigSm90(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            cluster_m=cluster_m, cluster_n=cluster_n,
            pingpong=False, kSMs=kSMs, kStages=kStages, kSmVersion=sm,
        )
        key = cfg.compile_name
        if key not in seen:
            seen[key] = cfg
    return seen


def _get_sm120_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfig]:
    """SM120 (GeForce Blackwell / RTX 50 series) Conv2D configs.

    The CUTLASS 3.x SM120 CollectiveBuilder supports only F8/F6/F4 MMA, so
    FP16/BF16 Conv2D on SM120 is routed through the CUTLASS 2.x tensor-op path
    using the Sm80 forward-compatible instructions (mma.sync.aligned.m16n8k16).
    Mirrors GEMM's ``_get_sm120_configs()``.
    """
    return _build_sm_lt90_conv2d_configs(sm, TileShapeConfigs, [3], _SM_MAX_SMEM_BYTES[120])


def get_all_conv2d_autotune_configs(
    sm: int,
) -> Dict[str, Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]]:
    """Return the full autotune config set for *sm* (keyed by ``name``).

    Conv2D has no runtime-only parameters (no split-K), so ``name ==
    compile_name`` and this set equals the compile set.
    """
    if sm == 75:
        return _get_sm75_conv2d_configs(sm)  # type: ignore[return-value]
    elif sm == 80:
        return _get_sm80_conv2d_configs(sm)  # type: ignore[return-value]
    elif sm == 86:
        return _get_sm86_conv2d_configs(sm)  # type: ignore[return-value]
    elif sm == 89:
        return _get_sm89_conv2d_configs(sm)  # type: ignore[return-value]
    elif sm == 90:
        return _get_sm90_conv2d_configs(sm)  # type: ignore[return-value]
    elif sm == 100:
        return _get_sm100_conv2d_configs(sm)  # type: ignore[return-value]
    else:
        return _get_sm120_conv2d_configs(sm)  # type: ignore[return-value]


def get_unique_conv2d_compile_configs(
    sm: int,
) -> Dict[str, Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]]:
    """Return the compile-deduplicated config set for *sm* (keyed by ``compile_name``).

    For Conv2D, ``name == compile_name`` so this is identical to
    ``get_all_conv2d_autotune_configs``.  Provided for API parity with the GEMM
    layer (``get_unique_compile_configs``).
    """
    return get_all_conv2d_autotune_configs(sm)


# =============================================================================
# Default configs (used by non-autotuned paths in oasr/conv.py)
# =============================================================================

_sm = _get_target_sm()

if _sm < 90 or _sm == 120:
    # SM120 uses the CUTLASS 2.x (SM<90) path for FP16/BF16 — see
    # ``_get_sm120_conv2d_configs`` above.
    CONV2D_DEFAULT: Union[CutlassConv2dConfig, CutlassConv2dConfigSm90] = CutlassConv2dConfig(
        block_m=128, block_n=128, block_k=64, warp_m=64, warp_n=64, warp_k=64,
        kStages=3, kSmVersion=_sm,
    )
else:
    CONV2D_DEFAULT = CutlassConv2dConfigSm90(
        tile_m=128, tile_n=128, tile_k=128, cluster_m=1, cluster_n=1,
        pingpong=False, kSMs=1, kStages=3, kSmVersion=_sm,
    )


# =============================================================================
# Conv1D module (static sources, no variants)
# =============================================================================


def gen_conv_module() -> JitSpec:
    """Generate JIT spec for Conv1D kernels."""
    return gen_jit_spec(
        "conv",
        [
            env.OASR_CSRC_DIR / "conv.cu",
            env.OASR_CSRC_DIR / "conv_jit_binding.cu",
        ],
    )


# =============================================================================
# Conv2D module — ALL variants compiled into ONE .so
# =============================================================================


def _render_all_conv2d_variants() -> List:
    """Render Jinja templates for all unique Conv2D tile configs."""
    from .templates import render_template
    from .cubin_loader import write_if_different

    sm = _get_target_sm()
    unique_configs = get_unique_conv2d_compile_configs(sm)
    source_paths = []

    for config_name, cfg in unique_configs.items():
        func_name = f"conv2d_{config_name}"
        variant_file_name = f"conv2d_sm{sm}_{config_name}"

        if sm in [75, 80, 86, 89, 120]:
            rendered = render_template(
                "conv2d_cutlass_template.cu.jinja",
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
                with_activation=True,
            )
        else:
            rendered = render_template(
                "conv2d_cutlass_template_sm90.cu.jinja",
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
                with_activation=True,
            )

        gen_path = env.OASR_GEN_SRC_DIR / "conv2d" / f"{variant_file_name}.cu"
        write_if_different(gen_path, rendered)
        source_paths.append(gen_path)

    return source_paths


def gen_conv2d_module() -> JitSpec:
    """Generate JIT spec for Conv2D with ALL tile variants in one module.

    Each variant exports ``conv2d_{config_name}`` and
    ``conv2d_{config_name}_activation`` as TVM-FFI functions.
    """
    source_paths = _render_all_conv2d_variants()
    return gen_jit_spec("conv2d", source_paths)


# =============================================================================
# cuDNN Conv2D (unchanged)
# =============================================================================


def gen_cudnn_conv2d_module() -> JitSpec:
    """Generate JIT spec for cuDNN Conv2D kernels (small IC path)."""
    return gen_jit_spec(
        "cudnn_conv2d",
        [env.OASR_CSRC_DIR / "cudnn_conv2d_kernel_launcher.cu"],
        extra_ldflags=["-lcudnn"],
    )


# =============================================================================
# Function name helpers
# =============================================================================


def conv2d_func_name(cfg: Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]) -> str:
    """Return the TVM-FFI export name for a Conv2D variant."""
    return f"conv2d_{cfg.compile_name}"


def conv2d_activation_func_name(cfg: Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]) -> str:
    """Return the TVM-FFI export name for a Conv2D+activation variant."""
    return f"conv2d_{cfg.compile_name}_activation"
