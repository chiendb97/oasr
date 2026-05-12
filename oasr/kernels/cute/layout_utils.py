# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Smem layout helpers shared by the OASR attention kernels.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/layout_utils.py``
but scoped to what the SM80 / SM120 forward pass needs (swizzled smem atoms
for Q/K/V, ``tile_to_shape`` helper). The implementations are OASR-native.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

from typing import Tuple

import cutlass
import cutlass.cute as cute


def make_smem_swizzle_atom(
    dtype, head_dim_block: int,
) -> Tuple[cute.ComposedLayout, int]:
    """Build the 8x``head_dim_block`` smem swizzle atom for Q/K/V smem.

    Returns ``(atom, smem_k_block_size)`` where ``smem_k_block_size`` is the
    inner-dim tile width (64 or 32 depending on whether ``head_dim_block``
    is a multiple of 64). Swizzle bits are 3 for 64-wide and 2 for 32-wide,
    matching the FlashAttention / CUTLASS Ampere convention.

    Why this matters: a row-major (8, k) atom with the given swizzle yields
    bank-conflict-free 128-bit ldmatrix loads when the same atom is tiled
    out to the full ``(M_BLOCK, head_dim_padded)`` smem region.
    """
    smem_k_block_size = 64 if head_dim_block % 64 == 0 else 32
    swizzle_bits = 3 if smem_k_block_size == 64 else 2
    atom = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
    )
    return atom, smem_k_block_size


def make_smem_layout(
    atom: cute.ComposedLayout, rows: int, cols: int,
) -> cute.ComposedLayout:
    """Tile ``atom`` to cover the ``(rows, cols)`` smem region."""
    return cute.tile_to_shape(atom, (rows, cols), (0, 1))


def make_v_transpose_view(
    sV: cute.Tensor, head_dim_padded: int, n_block_size: int,
) -> cute.Tensor:
    """Return a logical k-major view of the V smem tile for the PV gemm.

    The V cp.async writes a row-major ``(N_BLOCK, head_dim)`` tile; the PV
    gemm wants it in ``(head_dim, N_BLOCK)`` form so the m16n8k16 atom can
    treat V's column axis (n_block) as the gemm-K. We don't materialize a
    transpose -- we compose a layout view over the same smem pointer.
    """
    return cute.composition(
        sV,
        cute.make_layout(
            (head_dim_padded, n_block_size),
            stride=(n_block_size, 1),
        ),
    )


__all__ = [
    "make_smem_swizzle_atom",
    "make_smem_layout",
    "make_v_transpose_view",
]
