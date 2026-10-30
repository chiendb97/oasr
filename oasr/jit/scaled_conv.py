# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the int8 implicit-GEMM convolution (``oasr.scaled_conv2d``).

``out[n, p, q, k] = act(scale[k] * conv(x, w)[n, p, q, k] + bias[k])`` over an int8
NHWC input and an int8 KRSC filter: the forward of a quantized ``Conv2d`` /
``Conv1d`` (a conv1d is the height-one problem).  CUTLASS 2.x Sm80 int8
``mma.sync`` with an exact int32 accumulator, through
``DefaultConv2dFpropWithBroadcast`` (``include/oasr/scaled_conv/``): the 2.x
convolutions have no EVT epilogue, so the per-channel scale rides the broadcast
vector and the bias a zero-stride source tensor.

The lane serves the SMs the int8 GEMM lane does, natively on sm80/86/89/120 and
as the mixed lane on sm90/sm100.  A family of its own, as the int8 GEMM is: only
a model with a quantized convolution compiles it.

The GEMM view: M = N * P * Q output pixels, N = K output channels, K = R * S * C.
The tile is picked like the int8 GEMM's (:data:`oasr.jit.scaled_mm._INT8_LADDER`):
the largest tile that still fills the GPU, a shallower pipeline with two
resident CTAs per SM where the launch covers every SM.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .core import GeneratedSource, JitSpec, _get_target_sm, gen_jit_spec
from .gemm import _SM_MAX_SMEM_BYTES, TileShape, _epilogue_covers_warp, _smem_bytes

#: Operand alignment: 16-byte int8 loads along C, 16-byte output stores along K.
C_ALIGNMENT = 16
K_ALIGNMENT = 8


@dataclass(frozen=True)
class ScaledConvConfig:
    """One CUTLASS 2.x int8 convolution tile."""

    block_m: int
    block_n: int
    block_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    stages: int

    @property
    def compile_name(self) -> str:
        return (
            f"b{self.block_m}x{self.block_n}x{self.block_k}_"
            f"w{self.warp_m}x{self.warp_n}x{self.warp_k}_s{self.stages}"
        )

    @property
    def tile(self) -> TileShape:
        return TileShape(
            self.block_m, self.block_n, self.block_k, self.warp_m, self.warp_n, self.warp_k
        )


_T64 = ScaledConvConfig(64, 64, 64, 32, 32, 64, 4)
_T64X128 = ScaledConvConfig(64, 128, 64, 32, 64, 64, 4)
_T128 = ScaledConvConfig(128, 128, 64, 64, 64, 64, 3)
_T128_SPARSE = ScaledConvConfig(128, 128, 64, 64, 64, 64, 4)
_T128X256 = ScaledConvConfig(128, 256, 64, 64, 64, 64, 3)

#: The production tiles, the int8 GEMM's occupancy-sized shapes.
_PRODUCTION: Tuple[ScaledConvConfig, ...] = (_T64, _T64X128, _T128, _T128_SPARSE, _T128X256)

#: ``(tile, minimum CTAs per SM, minimum GEMM K, tile when the CTAs do not cover
#: every SM once)``, largest first; :data:`_T64` serves everything below.  Measured
#: on an RTX 5090 over the U2++ subsampling conv (batch 1-64) and Whisper's conv1d
#: front-ends (tiny and large, batch 1-16), 2026-10-23: the 128x256 tile wins only
#: once it fills the GPU four times over (3-5 % at batch 32-64) and loses up to
#: 1.31x below that; with these rungs every case is within 4 % of the best tile.
_LADDER: Tuple[Tuple[ScaledConvConfig, float, int, "ScaledConvConfig | None"], ...] = (
    (_T128X256, 4.0, 2048, None),
    (_T128, 0.5, 0, _T128_SPARSE),
    (_T64X128, 0.5, 0, None),
)

#: SM families the int8 convolution lane runs on, and whether it is native there.
SCALED_CONV_LANE_BY_SM: Dict[int, bool] = {
    80: True,
    86: True,
    89: True,
    120: True,
    90: False,
    100: False,
}

#: The epilogue partitions fp16/bf16 output fragments, eight per access.
_EPILOGUE_ELEMENT_BITS = 16


def _buildable(cfg: ScaledConvConfig, sm: int) -> bool:
    limit = _SM_MAX_SMEM_BYTES.get(sm, 100 * 1024)
    if _smem_bytes(cfg.block_m, cfg.block_n, cfg.block_k, cfg.stages, dtype_bytes=1) > limit:
        return False
    return _epilogue_covers_warp(cfg.tile, element_bits=_EPILOGUE_ELEMENT_BITS)


def lane_available(sm: int) -> bool:
    """Whether the int8 convolution lane serves *sm*."""
    return sm in SCALED_CONV_LANE_BY_SM


def is_native_lane(sm: int) -> bool:
    return SCALED_CONV_LANE_BY_SM.get(sm, False)


def get_production_configs(sm: int | None = None) -> Dict[str, ScaledConvConfig]:
    """``{compile_name: config}`` for every production tile buildable on *sm*."""
    sm = _get_target_sm() if sm is None else sm
    return {c.compile_name: c for c in _PRODUCTION if _buildable(c, sm)}


@functools.lru_cache(maxsize=4096)
def _select(m: int, n: int, k: int, sm: int, num_sms: int) -> ScaledConvConfig:
    built = get_production_configs(sm)
    for tile, min_ctas_per_sm, min_k, sparse in _LADDER:
        if k < min_k or 2 * m <= tile.block_m:
            continue
        ctas = -(-m // tile.block_m) * -(-n // tile.block_n)
        if ctas < min_ctas_per_sm * num_sms:
            continue
        pick = sparse if sparse is not None and ctas < num_sms else tile
        if pick.compile_name in built:
            return pick
    return _T64


@functools.lru_cache(maxsize=1)
def _process_sm() -> int:
    return _get_target_sm()


@functools.lru_cache(maxsize=1)
def _process_num_sms() -> int:
    import torch

    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


def select_config(
    m: int, n: int, k: int, sm: int | None = None, num_sms: int | None = None
) -> ScaledConvConfig:
    """The tile for a convolution whose GEMM view is ``[m, k] @ [n, k]ᵀ``.

    A function of shape and device alone -- never of capture state (AGENTS.md
    rule 11) -- and memoised, so a repeated shape is one cache hit.
    """
    sm = _process_sm() if sm is None else int(sm)
    num_sms = _process_num_sms() if num_sms is None else int(num_sms)
    return _select(int(m), int(n), int(k), sm, num_sms)


def scaled_conv2d_func_name(cfg: ScaledConvConfig) -> str:
    """The TVM-FFI export name of a tile's launcher."""
    return f"scaled_conv2d_{cfg.compile_name}"


def _render_variants(configs: Dict[str, ScaledConvConfig]) -> List:
    from .templates import render_template

    sm = _get_target_sm()
    paths = []
    for name, cfg in configs.items():
        variant = f"scaled_conv2d_sm{sm}_{name}"
        rendered = render_template(
            "scaled_conv2d_cutlass_template.cu.jinja",
            op_name=variant,
            func_name=scaled_conv2d_func_name(cfg),
            tile_m=cfg.block_m,
            tile_n=cfg.block_n,
            tile_k=cfg.block_k,
            warp_m=cfg.warp_m,
            warp_n=cfg.warp_n,
            warp_k=cfg.warp_k,
            stages=cfg.stages,
            sm_version=sm,
        )
        paths.append(GeneratedSource(f"{variant}.cu", rendered))
    return paths


def gen_scaled_conv_module() -> JitSpec:
    """JIT spec for every production int8 convolution tile, one TU each."""
    return gen_jit_spec("scaled_conv", _render_variants(get_production_configs()))


def gen_scaled_conv_modules() -> List[JitSpec]:
    """The module, or nothing on an SM the lane does not serve (for the AOT list)."""
    return [gen_scaled_conv_module()] if lane_available(_get_target_sm()) else []


__all__ = [
    "C_ALIGNMENT",
    "K_ALIGNMENT",
    "SCALED_CONV_LANE_BY_SM",
    "ScaledConvConfig",
    "gen_scaled_conv_module",
    "gen_scaled_conv_modules",
    "get_production_configs",
    "is_native_lane",
    "lane_available",
    "scaled_conv2d_func_name",
    "select_config",
]
