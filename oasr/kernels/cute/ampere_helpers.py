# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SM80 gemm helpers for the OASR FMHA forward kernel.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/ampere_helpers.py``.
The implementation is OASR-native and only covers what the SM80 / SM120
forward pass needs:

* :func:`gemm_with_smem_prefetch` -- QK gemm where each ldmatrix x4 overlaps
  with the prior m16n8k16 mma instead of going strictly serial. Reduces the
  GEMM-K loop's critical path by ~one mma latency per inner iteration.
* :func:`gemm_rs` -- A-in-rmem PV gemm. ``rA`` (the softmaxed P matrix) is
  already in registers, so we only load B (= V) from smem and run the
  m16n8k16 chain with the same overlap trick.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

import cutlass
import cutlass.cute as cute


@cute.jit
def gemm_with_smem_prefetch(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    rA_for_mma: cute.Tensor,
    rB_for_mma: cute.Tensor,
    smem_tiled_copy_A: cute.TiledCopy,
    smem_tiled_copy_B: cute.TiledCopy,
    sA_partitioned: cute.Tensor,
    sB_partitioned: cute.Tensor,
    rA_copy_view: cute.Tensor,
    rB_copy_view: cute.Tensor,
):
    """QK gemm with inner-loop ldmatrix prefetch.

    Standard ldmatrix -> mma chain runs strictly serial. By kicking off the
    *next* iteration's ldmatrix before the *current* iteration's mma we let
    the LSU work in parallel with the mma pipe, saving one ldmatrix latency
    per inner iteration.

    The first iter's load is issued before the loop, then on iter ``k`` we
    issue iter ``(k+1) % K_tiles``'s load before the iter ``k`` mma. Wrap-
    around on the last iter is benign (the wrapped data is overwritten on
    the next outer call).
    """
    K_tiles = cute.size(sA_partitioned.shape[2])
    # Prologue: load the first k-tile.
    cute.copy(
        smem_tiled_copy_A,
        sA_partitioned[None, None, 0],
        rA_copy_view[None, None, 0],
    )
    cute.copy(
        smem_tiled_copy_B,
        sB_partitioned[None, None, 0],
        rB_copy_view[None, None, 0],
    )
    for k in cutlass.range_constexpr(K_tiles):
        k_next = (k + 1) % K_tiles
        # Prefetch the next-iter operands BEFORE the current mma.
        cute.copy(
            smem_tiled_copy_A,
            sA_partitioned[None, None, k_next],
            rA_copy_view[None, None, k_next],
        )
        cute.copy(
            smem_tiled_copy_B,
            sB_partitioned[None, None, k_next],
            rB_copy_view[None, None, k_next],
        )
        cute.gemm(
            tiled_mma, acc,
            rA_for_mma[None, None, k],
            rB_for_mma[None, None, k],
            acc,
        )


@cute.jit
def gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    rA_view: cute.Tensor,
    rB_for_mma: cute.Tensor,
    smem_tiled_copy_B: cute.TiledCopy,
    sB_partitioned: cute.Tensor,
    rB_copy_view: cute.Tensor,
):
    """A-in-rmem PV gemm with inner-loop ldmatrix prefetch for B (= V).

    ``rA_view`` is the (already-softmaxed) P matrix laid out for the m16n8k16
    A operand. We only need to ldmatrix B = V from smem; A is consumed from
    registers directly.
    """
    K_tiles = cute.size(rA_view.shape[2])
    cute.copy(
        smem_tiled_copy_B,
        sB_partitioned[None, None, 0],
        rB_copy_view[None, None, 0],
    )
    for k in cutlass.range_constexpr(K_tiles):
        k_next = (k + 1) % K_tiles
        cute.copy(
            smem_tiled_copy_B,
            sB_partitioned[None, None, k_next],
            rB_copy_view[None, None, k_next],
        )
        cute.gemm(
            tiled_mma, acc,
            rA_view[None, None, k],
            rB_for_mma[None, None, k],
            acc,
        )


__all__ = ["gemm_with_smem_prefetch", "gemm_rs"]
