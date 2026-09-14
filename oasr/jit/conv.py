# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for convolution kernels."""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from . import env
from .core import _TARGET_SMS, JitSpec, _get_target_sm, gen_jit_spec
from .gemm import _SM_MAX_SMEM_BYTES, TileShape, TileShapeConfigs, _tile_is_buildable

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
        return (
            ("block_m", self.block_m),
            ("block_n", self.block_n),
            ("block_k", self.block_k),
            ("warp_m", self.warp_m),
            ("warp_n", self.warp_n),
            ("warp_k", self.warp_k),
            ("kStages", self.kStages),
        )


@dataclass(frozen=True)
class CutlassConv2dConfigSm90:
    """CUTLASS 3.x implicit-GEMM Conv2D config for SM90 and SM100.

    Deliberately **not** a field-for-field mirror of ``CutlassGemmConfigSm90``.
    It used to be, and two of the borrowed fields were the reason no Conv2D
    kernel compiled on Hopper or datacenter Blackwell:

    * ``pingpong`` — conv has no persistent schedule on SM90. The tags exist but
      ``conv/dispatch_policy.hpp`` static_asserts on them, and CUTLASS's own
      auto-selector has the cooperative branch commented out. Enumerating the
      axis produced two identically-scheduled kernels under different names.
    * ``kSMs`` — GEMM doubles the M tile for a 2-SM SM100 atom. Conv does not:
      ``BM`` *is* the MMA tile M, and the 2-SM atom is chosen from the cluster.
      ``BM * 2`` trips "Invalid TileShape_M." (see :func:`_sm100_conv_tile_ok`).

    Omitted GEMM-only fields: ``is_dynamic_persistent``, ``swap_ab``,
    ``max_swizzle_size``, ``use_tma_gather``. No ``cluster_k`` — CK is always 1
    for implicit-GEMM convolution and is hardcoded in the C++ struct.

    Conv2D has no runtime-only parameters, so ``name == compile_name``.
    """

    tile_m: int
    tile_n: int
    tile_k: int  # the implicit-GEMM K tile (the filter's C extent)
    cluster_m: int
    cluster_n: int
    kStages: int
    kSmVersion: int  # 90 or 100

    @property
    def name(self) -> str:
        return self.compile_name

    @property
    def compile_name(self) -> str:
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.tile_m}x{self.tile_n}x{self.tile_k}")
        parts.append(f"c{self.cluster_m}x{self.cluster_n}")
        parts.append(f"s{self.kStages}")
        return "_".join(parts)

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        return (
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
            ("cluster_m", self.cluster_m),
            ("cluster_n", self.cluster_n),
            ("kStages", self.kStages),
        )


# =============================================================================
# SM<90 config generation — SMEM-analysed per-SM tile × stage
#
# Uses TileShapeConfigs directly from gemm.py (same 15 tiles as GEMM) and the
# same _tile_is_buildable / _SM_MAX_SMEM_BYTES limits.
# =============================================================================


def _build_sm_lt90_conv2d_configs(
    sm: int,
    tiles: List[TileShape],
    stage_list: List[int],
    smem_limit: int,
) -> Dict[str, CutlassConv2dConfig]:
    """Build the full autotune config dict for a SM<90 Conv2D architecture.

    Identical logic to GEMM's ``_build_sm_lt90_configs`` but without split_k —
    including the epilogue constraint, because implicit GEMM instantiates the
    same ``DefaultThreadMapTensorOp`` epilogue from the same tile list.  A
    ``block_n=16`` conv2d variant is wrong by the same 1.2x relative error its
    GEMM twin is; see :func:`oasr.jit.gemm._epilogue_covers_warp`.
    """
    seen: Dict[str, CutlassConv2dConfig] = {}
    for tile in tiles:
        for kStages in stage_list:
            if not _tile_is_buildable(tile, kStages, smem_limit):
                continue
            cfg = CutlassConv2dConfig(
                block_m=tile.block_m,
                block_n=tile.block_n,
                block_k=tile.block_k,
                warp_m=tile.warp_m,
                warp_n=tile.warp_n,
                warp_k=tile.warp_k,
                kStages=kStages,
                kSmVersion=sm,
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


# SM90+ Conv2D tile and cluster choices mirror the corresponding GEMM schedules.


#: One pipeline stage of the conv mainloop costs ``(BM + BN) * BK * dtype_bytes``
#: bytes of shared memory, and ``StageCountAutoCarveout`` has to fit **two** of
#: them plus the epilogue into Hopper's 227 KiB.  Measured by compiling the full
#: M x N x cluster grid for sm_90a: every tile at or below this builds, and the
#: next rung up (114,688 B, e.g. 256x192x128 or 192x256x128) fails with
#: "Specialization requires Stages set to value 1 or more" — the carveout leaves
#: no room for even one stage.  Stated as a budget rather than a tile list
#: because one unbuildable variant fails the whole JIT module, not just its own
#: tactic, so the space must be derived and not remembered.
_SM90_CONV_SMEM_PER_STAGE_MAX = 106_496


def _sm90_conv_tile_ok(tile_m: int, tile_n: int, tile_k: int, dtype_bytes: int = 2) -> bool:
    """Can SM90's conv mainloop pipeline this tile at all?"""
    return (tile_m + tile_n) * tile_k * dtype_bytes <= _SM90_CONV_SMEM_PER_STAGE_MAX


def _sm100_conv_tile_ok(tile_m: int, cluster_m: int) -> bool:
    """SM100 pairs a 256-row MMA tile only with the 2-SM atom.

    The atom is selected from the *cluster*, so a 256-row tile needs
    ``cluster_m == 2``; anything else is "Invalid TileShape_M."  Verified across
    the full M x N x cluster grid for sm_100a — M of 64 and 128 take every
    cluster, M of 256 takes only ``cluster_m == 2``.
    """
    return tile_m < 256 or cluster_m == 2


def _get_sm90_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfigSm90]:
    """SM90 (Hopper) conv configs.

    No pingpong/cooperative axis: conv has one schedule on SM90 and the config
    struct no longer carries the flag.  The M x N space is the GEMM tile ladder
    filtered by :func:`_sm90_conv_tile_ok`.
    """
    tile_k = 128
    kStages = 3

    # A starter ladder, not a tuned one: nothing has measured conv tile choice on
    # Hopper, so this spans the M range at two N widths rather than pretending to
    # a fitted optimum.  Deliberately close to the GEMM SM90 space in size (16
    # variants) -- each CUTLASS 3.x conv translation unit peaks around 3.7 GB in
    # `cicc`, and ninja defaults to nproc-way parallelism, so a wide space is a
    # first-call OOM on a memory-limited box, not just a slow build.  Widen it
    # from a measurement, and record the measurement in `.artifacts/`.
    tile_m_vals = [64, 128, 256]
    tile_n_vals = [128, 256]
    cluster_vals = [(1, 1), (1, 2), (2, 1)]

    seen: Dict[str, CutlassConv2dConfigSm90] = {}
    for (tile_m, tile_n), (cluster_m, cluster_n) in itertools.product(
        itertools.product(tile_m_vals, tile_n_vals), cluster_vals
    ):
        if not _sm90_conv_tile_ok(tile_m, tile_n, tile_k):
            continue
        cfg = CutlassConv2dConfigSm90(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            kStages=kStages,
            kSmVersion=sm,
        )
        seen[cfg.compile_name] = cfg
    return seen


def _get_sm100_conv2d_configs(sm: int) -> Dict[str, CutlassConv2dConfigSm90]:
    """SM100 (Blackwell data-center) conv configs.

    ``tile_m`` is the MMA tile M as the builder sees it — never scaled by a
    co-operating-SM count.  A 256-row tile is admissible only alongside
    ``cluster_m == 2``; see :func:`_sm100_conv_tile_ok`.
    """
    tile_k = 64
    kStages = 3

    # Same caveat as SM90: unmeasured, and sized against the module's build cost.
    tile_m_vals = [64, 128, 256]
    tile_n_vals = [128, 256]
    cluster_vals = [(1, 1), (1, 2), (2, 1), (2, 2)]

    seen: Dict[str, CutlassConv2dConfigSm90] = {}
    for (tile_m, tile_n), (cluster_m, cluster_n) in itertools.product(
        itertools.product(tile_m_vals, tile_n_vals), cluster_vals
    ):
        if not _sm100_conv_tile_ok(tile_m, cluster_m):
            continue
        cfg = CutlassConv2dConfigSm90(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            kStages=kStages,
            kSmVersion=sm,
        )
        seen[cfg.compile_name] = cfg
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

    An unrecognised SM **raises** rather than inheriting SM120's space; see
    :func:`oasr.jit.gemm.get_all_autotune_configs` for why that ``else`` was a
    defect rather than a convenience.
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
    elif sm == 120:
        return _get_sm120_conv2d_configs(sm)  # type: ignore[return-value]
    raise ValueError(
        f"no Conv2D config space for sm_{sm}; OASR compiles for "
        f"{', '.join(f'sm_{t}' for t in _TARGET_SMS)}"
    )


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
# Default configs (used by non-autotuned paths in oasr/functionals/conv.py)
# =============================================================================

_sm = _get_target_sm()

if _sm < 90 or _sm == 120:
    # SM120 uses the CUTLASS 2.x (SM<90) path for FP16/BF16 — see
    # ``_get_sm120_conv2d_configs`` above.
    CONV2D_DEFAULT: Union[CutlassConv2dConfig, CutlassConv2dConfigSm90] = CutlassConv2dConfig(
        block_m=128,
        block_n=128,
        block_k=64,
        warp_m=64,
        warp_n=64,
        warp_k=64,
        kStages=3,
        kSmVersion=_sm,
    )
else:
    # It **must** be one of the variants the arch's generator emits: the module
    # compiles exactly those and the functional API looks the default up by
    # ``compile_name``, so a default outside the set raises ``AttributeError:
    # Module has no function ...`` on the first un-tuned call.  SM90 and SM100
    # differ in the K tile they are generated at, so the default does too;
    # ``tests/kernels/test_jit.py`` enforces the invariant for both.
    CONV2D_DEFAULT = CutlassConv2dConfigSm90(
        tile_m=128,
        tile_n=128,
        tile_k=128 if _sm == 90 else 64,
        cluster_m=1,
        cluster_n=1,
        kStages=3,
        kSmVersion=_sm,
    )


def _sm120_conv1d_config(
    block_m: int, block_n: int, warp_m: int, warp_n: int
) -> CutlassConv2dConfig:
    return CutlassConv2dConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=64,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=64,
        kStages=3,
        kSmVersion=120,
    )


# Exact SM120 production shapes. Batch and sequence remain in the key because
# implicit-GEMM's M dimension changes the best block height.
_CONV1D_HEURISTIC_RULES_SM120: Dict[
    str, Dict[Tuple[int, ...], Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]]
] = {
    "torch.float16": {
        # Fixed-window frontend, width 384.
        (1, 3000, 80, 384, 3, 1, 1, 1): _sm120_conv1d_config(64, 128, 32, 64),
        (1, 3000, 384, 384, 3, 1, 2, 1): _sm120_conv1d_config(32, 128, 32, 32),
        # Fixed-window frontend, width 1280.
        (1, 3000, 128, 1280, 3, 1, 1, 1): _sm120_conv1d_config(64, 128, 32, 64),
        (1, 3000, 1280, 1280, 3, 1, 2, 1): _sm120_conv1d_config(128, 32, 32, 32),
        # Padded predictor convolution.
        (1, 502, 512, 512, 3, 0, 1, 1): _sm120_conv1d_config(16, 128, 16, 32),
    },
    "torch.bfloat16": {
        (1, 3000, 80, 384, 3, 1, 1, 1): _sm120_conv1d_config(128, 64, 64, 32),
        (1, 3000, 384, 384, 3, 1, 2, 1): _sm120_conv1d_config(128, 32, 32, 32),
        (1, 3000, 128, 1280, 3, 1, 1, 1): _sm120_conv1d_config(64, 128, 32, 64),
        (1, 3000, 1280, 1280, 3, 1, 2, 1): _sm120_conv1d_config(128, 32, 32, 32),
        (1, 502, 512, 512, 3, 0, 1, 1): _sm120_conv1d_config(16, 128, 16, 32),
    },
}

_CONV1D_ACTIVATION_HEURISTIC_RULES_SM120: Dict[
    str, Dict[Tuple[int, ...], Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]]
] = {
    # Predictor convolution with fused ReLU.
    "torch.float16": {
        (1, 502, 512, 512, 3, 0, 1, 1): _sm120_conv1d_config(16, 128, 16, 32),
    },
    "torch.bfloat16": {
        (1, 502, 512, 512, 3, 0, 1, 1): _sm120_conv1d_config(16, 128, 16, 32),
    },
}


def _select_conv1d_rule(
    rules_by_dtype: Dict[
        str, Dict[Tuple[int, ...], Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]]
    ],
    shape: Tuple[int, ...],
    dtype,
    sm: int,
) -> Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]:
    if sm != 120:
        return CONV2D_DEFAULT
    rules = rules_by_dtype.get(str(dtype))
    if rules is None:
        return CONV2D_DEFAULT
    return rules.get(tuple(int(value) for value in shape), CONV2D_DEFAULT)


def select_default_conv1d_config(
    batch: int,
    seq_len: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    stride: int,
    dilation: int,
    dtype,
    sm: int,
) -> Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]:
    """Pick a measured dense Conv1D tile for the non-autotuned path.

    The table is deliberately exact and currently covers FP16/BF16 on SM120.
    An unmeasured architecture, dtype, batch, or length retains
    :data:`CONV2D_DEFAULT`; callers can opt into the full runtime autotuner for
    additional shapes.
    """
    return _select_conv1d_rule(
        _CONV1D_HEURISTIC_RULES_SM120,
        (batch, seq_len, in_channels, out_channels, kernel_size, padding, stride, dilation),
        dtype,
        sm,
    )


def select_default_conv1d_activation_config(
    batch: int,
    seq_len: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    stride: int,
    dilation: int,
    dtype,
    sm: int,
) -> Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]:
    """Pick a measured fused-activation Conv1D tile, with a safe fallback."""
    return _select_conv1d_rule(
        _CONV1D_ACTIVATION_HEURISTIC_RULES_SM120,
        (batch, seq_len, in_channels, out_channels, kernel_size, padding, stride, dilation),
        dtype,
        sm,
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
    from .cubin_loader import write_if_different
    from .templates import render_template

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
                config_name=cfg.compile_name,
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
                config_name=cfg.compile_name,
                tile_m=cfg.tile_m,
                tile_n=cfg.tile_n,
                tile_k=cfg.tile_k,
                cluster_m=cfg.cluster_m,
                cluster_n=cfg.cluster_n,
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


def gen_grouped_conv2d_module() -> JitSpec:
    """Generate the direct NHWC grouped/depthwise Conv2D module.

    This module is deliberately separate from the many dense CUTLASS tile
    variants: grouped traffic has one direct implementation, so rebuilding it
    must not recompile every implicit-GEMM tactic.
    """
    return gen_jit_spec(
        "grouped_conv2d",
        [
            env.OASR_CSRC_DIR / "conv2d.cu",
            env.OASR_CSRC_DIR / "conv2d_jit_binding.cu",
        ],
        extra_cuda_cflags=["-DOASR_GROUPED_CONV2D_ONLY=1"],
    )


# =============================================================================
# cuDNN Conv2D (unchanged)
# =============================================================================


def _torch_cudnn_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Include dir and library **file** of the cuDNN torch ships, if it ships one.

    torch's CUDA wheels depend on ``nvidia-cudnn-cu12``, which installs headers
    and libraries under ``site-packages/nvidia/cudnn/``.  A *system* cuDNN is not
    guaranteed: a stock ``nvidia/cuda:*-devel`` image has none, and this module
    is the only place in the tree needing ``cudnn.h`` — so a bare ``-lcudnn``
    made cuDNN an undeclared build dependency that fails at first *call* (JIT),
    long after ``pip install`` said it was fine.

    The library is a file rather than a directory on purpose.  The wheel is a
    *runtime* distribution: it ships ``libcudnn.so.9`` and **not** the
    unversioned ``libcudnn.so`` symlink that ``-lcudnn`` resolves through, which
    comes from a system dev package.  So ``-L<wheeldir> -lcudnn`` still fails
    with "cannot find -lcudnn" on a box without one — it only appears to work
    where a system cuDNN is quietly supplying the symlink.  Naming the versioned
    file skips library search altogether.

    Preferring the wheel where both exist is deliberate: it is the copy torch
    itself loads, so the process ends up with one cuDNN rather than linking
    against one version and loading another.

    Returns ``(None, None)`` when the wheel is absent, which leaves the bare
    ``-lcudnn`` to find a system install exactly as before.
    """
    try:
        import nvidia
    except ImportError:  # pragma: no cover - depends on the torch wheel flavour
        return None, None
    if not getattr(nvidia, "__file__", None):
        return None, None
    root = Path(nvidia.__file__).resolve().parent / "cudnn"
    include = root / "include"
    lib_dir = root / "lib"

    lib: Optional[Path] = None
    if lib_dir.is_dir():
        unversioned = lib_dir / "libcudnn.so"
        if unversioned.is_file():
            lib = unversioned
        else:
            # Shortest name first: libcudnn.so.9 ahead of libcudnn.so.9.19.0,
            # i.e. the soname rather than the fully-qualified release.
            versioned = sorted(lib_dir.glob("libcudnn.so.*"), key=lambda p: (len(p.name), p.name))
            lib = versioned[0] if versioned else None

    return (include if (include / "cudnn.h").is_file() else None, lib)


def gen_cudnn_conv2d_module() -> JitSpec:
    """Generate JIT spec for cuDNN Conv2D kernels (small IC path)."""
    include, lib = _torch_cudnn_paths()
    if lib is not None:
        # The library by absolute path instead of `-lcudnn` (see above), plus
        # -rpath: the module is dlopen'd at first call, so the dynamic loader
        # has to find it at *load* time too, not only the linker at build time.
        ldflags = [str(lib), f"-Wl,-rpath,{lib.parent}"]
    else:
        ldflags = ["-lcudnn"]
    return gen_jit_spec(
        "cudnn_conv2d",
        [env.OASR_CSRC_DIR / "cudnn_conv2d_kernel_launcher.cu"],
        extra_cuda_cflags=[f"-I{include}"] if include is not None else None,
        extra_ldflags=ldflags,
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


def conv1d_func_name(cfg: Union[CutlassConv2dConfig, CutlassConv2dConfigSm90]) -> str:
    """Return the TVM-FFI export name for a dense BTC Conv1D variant.

    Dense Conv1D is the height-one specialization of the same CUTLASS
    implicit-GEMM kernel used by Conv2D.  Keeping a distinct export gives the
    public operation a strict three-dimensional contract without introducing
    a view/copy in Python.
    """
    return f"conv1d_{cfg.compile_name}"


def conv1d_activation_func_name(
    cfg: Union[CutlassConv2dConfig, CutlassConv2dConfigSm90],
) -> str:
    """Return the TVM-FFI export name for a dense Conv1D+activation variant."""
    return f"conv1d_{cfg.compile_name}_activation"
