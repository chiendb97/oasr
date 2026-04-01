# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for convolution kernels (FlashInfer-style).

Conv1D uses static source files.  Conv2D tile configurations are defined here
and ALL variants are compiled into a single shared library, matching the
FlashInfer pattern used by jit/gemm.py.

SM75–89: CUTLASS 2.x implicit GEMM (CutlassConv2dConfig)
SM90+:   CUTLASS 3.x TMA warp-specialized (CutlassConv2dConfigSm90)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from .core import gen_jit_spec, _get_target_sm, JitSpec
from .gemm import TileShape, TileShapeSm90, ClusterShape
from . import env


# =============================================================================
# Conv2D config dataclasses
# =============================================================================


@dataclass(frozen=True)
class CutlassConv2dConfig:
    """CUTLASS 2.x Conv2D config for SM75–89."""
    BM: int
    BN: int
    BK: int
    WM: int
    WN: int
    WK: int
    kStages: int
    kSmVersion: int

    @property
    def compile_name(self) -> str:
        return "_".join([
            f"sm{self.kSmVersion}",
            f"b{self.BM}x{self.BN}x{self.BK}",
            f"w{self.WM}x{self.WN}x{self.WK}",
            f"s{self.kStages}",
        ])

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        return tuple([
            ("BM", self.BM), ("BN", self.BN), ("BK", self.BK),
            ("WM", self.WM), ("WN", self.WN), ("WK", self.WK),
            ("kStages", self.kStages),
        ])


@dataclass(frozen=True)
class CutlassConv2dConfigSm90:
    """CUTLASS 3.x Conv2D config for SM90+ (mirrors CutlassGemmConfigSm90 structure)."""
    BM: int
    BN: int
    BK: int
    CM: int
    CN: int
    CK: int
    kSMs: int
    kStages: int
    kSmVersion: int

    @property
    def compile_name(self) -> str:
        return "_".join([
            f"sm{self.kSmVersion}",
            f"b{self.BM}x{self.BN}x{self.BK}",
            f"c{self.CM}x{self.CN}x{self.CK}",
            f"k{self.kSMs}",
            f"s{self.kStages}",
        ])

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        return tuple([
            ("BM", self.BM), ("BN", self.BN), ("BK", self.BK),
            ("CM", self.CM), ("CN", self.CN), ("CK", self.CK),
            ("kSMs", self.kSMs), ("kStages", self.kStages),
        ])


# =============================================================================
# Conv2D tile configurations
# =============================================================================

# SM75–89: threadblock tile (BM, BN, BK) + warp tile (WM, WN, WK)
CONV2D_TILE_CONFIGS: List[TileShape] = [
    TileShape(BM=64,  BN=64,  BK=64, WM=32, WN=32, WK=64),
    TileShape(BM=128, BN=128, BK=64, WM=64, WN=64, WK=64),  # default
    TileShape(BM=128, BN=128, BK=32, WM=64, WN=64, WK=32),
    TileShape(BM=256, BN=128, BK=32, WM=64, WN=64, WK=32),
    TileShape(BM=128, BN=256, BK=32, WM=64, WN=64, WK=32),
]

# SM90+: tile shape in GEMM view (M=N*P*Q, N=K, K=R*S*IC)
CONV2D_TILE_CONFIGS_SM90: List[TileShapeSm90] = [
    TileShapeSm90(BM=64,  BN=64,  BK=64),
    TileShapeSm90(BM=64,  BN=128, BK=64),
    TileShapeSm90(BM=128, BN=64,  BK=64),
    TileShapeSm90(BM=128, BN=128, BK=64),  # default
]

CONV2D_CLUSTER_CONFIGS_SM90: List[ClusterShape] = [
    ClusterShape(CM=1, CN=1, CK=1),
    ClusterShape(CM=2, CN=1, CK=1),  # default
    ClusterShape(CM=1, CN=2, CK=1),
]

# Default configs (used by non-autotuned paths)
CONV2D_DEFAULT = CutlassConv2dConfig(
    BM=128, BN=128, BK=64, WM=64, WN=64, WK=64, kStages=3, kSmVersion=80
)
CONV2D_DEFAULT_SM90 = CutlassConv2dConfigSm90(
    BM=128, BN=128, BK=64, CM=2, CN=1, CK=1, kSMs=1, kStages=3, kSmVersion=90
)


# =============================================================================
# Config deduplication
# =============================================================================


def _get_unique_conv2d_compile_configs(
    sm: int,
) -> Dict[str, Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]]:
    """Return deduplicated compile configs for the target SM."""
    seen: Dict[str, Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]] = {}

    if sm in [75, 80, 86, 89]:
        for cfg in CONV2D_TILE_CONFIGS:
            compile_cfg = CutlassConv2dConfig(
                BM=cfg.BM, BN=cfg.BN, BK=cfg.BK,
                WM=cfg.WM, WN=cfg.WN, WK=cfg.WK,
                kStages=3, kSmVersion=sm,
            )
            key = compile_cfg.compile_name
            if key not in seen:
                seen[key] = compile_cfg
    else:
        for tile in CONV2D_TILE_CONFIGS_SM90:
            for cluster in CONV2D_CLUSTER_CONFIGS_SM90:
                compile_cfg = CutlassConv2dConfigSm90(
                    BM=tile.BM, BN=tile.BN, BK=tile.BK,
                    CM=cluster.CM, CN=cluster.CN, CK=cluster.CK,
                    kSMs=1, kStages=3, kSmVersion=sm,
                )
                key = compile_cfg.compile_name
                if key not in seen:
                    seen[key] = compile_cfg
    return seen


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
    unique_configs = _get_unique_conv2d_compile_configs(sm)
    source_paths = []

    for config_name, cfg in unique_configs.items():
        func_name = f"conv2d_{config_name}"
        variant_file_name = f"conv2d_sm{sm}_{config_name}"

        if sm in [75, 80, 86, 89]:
            rendered = render_template(
                "conv2d_cutlass_template.cu.jinja",
                op_name=variant_file_name,
                func_name=func_name,
                tile_m=cfg.BM,
                tile_n=cfg.BN,
                tile_k=cfg.BK,
                warp_m=cfg.WM,
                warp_n=cfg.WN,
                warp_k=cfg.WK,
                stages=cfg.kStages,
                sm_version=sm,
                with_activation=True,
            )
        else:
            rendered = render_template(
                "conv2d_cutlass_template_sm90.cu.jinja",
                op_name=variant_file_name,
                func_name=func_name,
                tile_m=cfg.BM,
                tile_n=cfg.BN,
                tile_k=cfg.BK,
                cluster_m=cfg.CM,
                cluster_n=cfg.CN,
                cluster_k=cfg.CK,
                k_sms=cfg.kSMs,
                stages=cfg.kStages,
                sm_version=sm,
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
