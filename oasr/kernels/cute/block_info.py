# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-Q-tile block-range computation for the FMHA forward kernel.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/block_info.py``.
The implementation is OASR-native; only forward / SM80 cases are covered
(no backward, no SM90 producer/consumer block math).

Given a Q-tile coordinate ``m_block``, this module computes
``(n_block_min, n_block_max)`` -- the half-open range of K-tile indices the
kernel needs to walk. The range tightens when:

* **Causal**: only K-tiles whose column-range intersects ``[0, q_row+1)``.
* **Local window**: ``[q_row - W_left, q_row + W_right]``.
* **Sliding window**: same as local but symmetric.
* **Length mask**: ``seqlen_k`` (per-batch) bounds the upper limit.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

import cutlass
import cutlass.cute as cute


class BlockInfo:
    """Half-open K-tile range ``[n_block_min, n_block_max)`` for a Q-tile.

    Iterate descending in the forward kernel so the row-max state can be
    initialized with the rightmost (and largest-magnitude under causal)
    block first.
    """

    def __init__(
        self,
        *,
        n_block_min: cutlass.Int32,
        n_block_max: cutlass.Int32,
    ):
        self.n_block_min = n_block_min
        self.n_block_max = n_block_max

    @cute.jit
    def num_blocks(self) -> cutlass.Int32:
        return self.n_block_max - self.n_block_min

    @cute.jit
    def is_empty(self) -> cutlass.Boolean:
        return self.n_block_max <= self.n_block_min


@cute.jit
def make_block_info(
    m_block: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    m_block_size: cutlass.Constexpr,
    n_block_size: cutlass.Constexpr,
    causal: cutlass.Constexpr,
    window_left: cutlass.Constexpr,
    window_right: cutlass.Constexpr,
) -> BlockInfo:
    """Compute the half-open K-tile range for the given Q-tile.

    ``window_left == -1`` or ``window_right == -1`` means *unbounded* on
    that side. Causal == True is equivalent to ``window_right = 0`` plus
    the implicit ``window_left = +inf`` (unbounded past).
    """
    # Largest K-row this Q-tile attends to:
    #   * unbounded:              seqlen_k
    #   * causal:                 min(seqlen_k, (m_block+1)*M_BLOCK + window_right)
    #   * local (right window):   min(seqlen_k, (m_block+1)*M_BLOCK + window_right)
    if cutlass.const_expr(causal or (window_right >= 0)):
        wr = cutlass.Int32(window_right if window_right >= 0 else 0)
        upper_row = (m_block + 1) * m_block_size + wr
        n_max_row = cute.arch.min_s32(seqlen_k, upper_row)
    else:
        n_max_row = seqlen_k

    # Smallest K-row this Q-tile attends to:
    #   * unbounded / causal:     0
    #   * local (left window):    max(0, m_block*M_BLOCK - window_left)
    if cutlass.const_expr(window_left >= 0):
        wl = cutlass.Int32(window_left)
        lower_row = m_block * m_block_size - wl
        n_min_row = cute.arch.max_s32(cutlass.Int32(0), lower_row)
    else:
        n_min_row = cutlass.Int32(0)

    # Convert row bounds to tile bounds. n_block_max is exclusive.
    n_block_max = (n_max_row + (n_block_size - 1)) // n_block_size
    n_block_min = n_min_row // n_block_size
    # Empty Q-tile (past seqlen_q) -> empty K-range.
    is_q_oob = (m_block * m_block_size) >= seqlen_q
    if is_q_oob:
        n_block_max = n_block_min
    return BlockInfo(n_block_min=n_block_min, n_block_max=n_block_max)


__all__ = ["BlockInfo", "make_block_info"]
