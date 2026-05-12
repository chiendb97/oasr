# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GMEM / SMEM copy atom + tiled-copy factories.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/copy_utils.py``
but the implementations are OASR-native and only cover what the SM80 /
SM120 forward pass needs (128-bit cp.async for Q/K/V, universal-copy for the
O epilogue, ldmatrix.x4 for the QK and PV gemms).
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp


# 128-bit cp.async / universal copy: the widest contiguous load the
# Ampere LSU supports without crossing alignment boundaries.
UNIVERSAL_COPY_BITS = 128


def async_copy_elements(dtype) -> int:
    """Number of elements per 128-bit cp.async."""
    return UNIVERSAL_COPY_BITS // dtype.width


def make_cp_async_atom(dtype) -> cute.CopyAtom:
    """128-bit gmem->smem cp.async with the standard GLOBAL cache mode."""
    return cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        dtype,
        num_bits_per_copy=UNIVERSAL_COPY_BITS,
    )


def make_universal_copy_atom(dtype) -> cute.CopyAtom:
    """128-bit universal copy atom (used for the O epilogue smem->gmem)."""
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        dtype,
        num_bits_per_copy=UNIVERSAL_COPY_BITS,
    )


def make_ldmatrix_x4_atom(dtype, *, transpose: bool = False) -> cute.CopyAtom:
    """ldmatrix.x4 atom for smem->rmem load of MMA operands.

    ``transpose=True`` is used for the V operand (k-major) on the PV gemm.
    """
    return cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
        dtype,
    )


def make_qkv_tiled_copy(
    atom: cute.CopyAtom, num_threads: int, k_block_size: int, vec_elems: int,
) -> cute.TiledCopy:
    """Tile a cp.async atom across ``num_threads`` for Q/K/V loads.

    Layout: ``(num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1)`` where
    ``tQKV_shape_dim_1 = k_block_size // vec_elems``. Each thread issues
    one 128-bit cp.async covering ``vec_elems`` elements along the
    head-dim axis.
    """
    tQKV_shape_dim_1 = k_block_size // vec_elems
    t_layout = cute.make_layout(
        (num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
        stride=(tQKV_shape_dim_1, 1),
    )
    v_layout = cute.make_layout((1, vec_elems))
    return cute.make_tiled_copy_tv(atom, t_layout, v_layout)


def make_o_tiled_copy(
    atom: cute.CopyAtom, num_threads: int, k_block_size: int, vec_elems: int,
) -> cute.TiledCopy:
    """Tiled-copy for the O epilogue (smem->gmem universal copy).

    Shape matches ``make_qkv_tiled_copy`` so the same threadblock geometry
    walks O.
    """
    return make_qkv_tiled_copy(atom, num_threads, k_block_size, vec_elems)


__all__ = [
    "UNIVERSAL_COPY_BITS",
    "async_copy_elements",
    "make_cp_async_atom",
    "make_universal_copy_atom",
    "make_ldmatrix_x4_atom",
    "make_qkv_tiled_copy",
    "make_o_tiled_copy",
]
