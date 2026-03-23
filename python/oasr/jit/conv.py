# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for convolution kernels."""

from .core import gen_jit_spec, JitSpec
from . import env


def gen_conv_module() -> JitSpec:
    """Generate JIT spec for Conv1D kernels."""
    return gen_jit_spec(
        "conv",
        [
            env.OASR_CSRC_DIR / "conv.cu",
            env.OASR_CSRC_DIR / "conv_jit_binding.cu",
        ],
    )


def gen_conv2d_module() -> JitSpec:
    """Generate JIT spec for Conv2D kernels."""
    return gen_jit_spec(
        "conv2d",
        [
            env.OASR_CSRC_DIR / "conv2d.cu",
            env.OASR_CSRC_DIR / "conv2d_jit_binding.cu",
        ],
    )


def gen_conv2d_module_variant(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    stages: int,
) -> JitSpec:
    """Generate JIT spec for Conv2D with a custom tile configuration.

    Each unique set of tile parameters produces a separate compiled module,
    allowing the autotuner to profile different configurations.
    """
    variant_name = f"conv2d_t{tile_m}x{tile_n}x{tile_k}_w{warp_m}x{warp_n}x{warp_k}_s{stages}"
    tile_flags = [
        f"-DOASR_CONV2D_TILE_M={tile_m}",
        f"-DOASR_CONV2D_TILE_N={tile_n}",
        f"-DOASR_CONV2D_TILE_K={tile_k}",
        f"-DOASR_CONV2D_WARP_M={warp_m}",
        f"-DOASR_CONV2D_WARP_N={warp_n}",
        f"-DOASR_CONV2D_WARP_K={warp_k}",
        f"-DOASR_CONV2D_STAGES={stages}",
    ]
    return gen_jit_spec(
        variant_name,
        [
            env.OASR_CSRC_DIR / "conv2d.cu",
            env.OASR_CSRC_DIR / "conv2d_jit_binding.cu",
        ],
        extra_cuda_cflags=tile_flags,
    )


def gen_cudnn_conv2d_module() -> JitSpec:
    """Generate JIT spec for cuDNN Conv2D kernels (small IC path)."""
    return gen_jit_spec(
        "cudnn_conv2d",
        [env.OASR_CSRC_DIR / "cudnn_conv2d_kernel_launcher.cu"],
        extra_ldflags=["-lcudnn"],
    )
