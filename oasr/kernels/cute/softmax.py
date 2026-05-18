# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Online softmax block for the OASR FMHA forward kernel.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/softmax.py``.
The implementation is OASR-native; only the SM80-friendly path is covered
(no SM100 fused-mask hooks, no FP32x2 packed reductions -- SM80's
``shuffle_sync_bfly`` is per-lane fp32 which is what we use).

Algorithm (FA2 online softmax with rescale-after):

    For each K-tile (descending iteration), holding ``row_max`` and
    ``row_sum`` in rmem:

    1.  ``row_max_cur = max(row_max_prev, row_max_of_this_tile)``  (per row)
    2.  ``P = exp2((S - row_max_cur) * scale_log2)``                (per element)
    3.  ``row_sum_cur = row_sum_prev * exp2((row_max_prev - row_max_cur) * scale_log2)
                       + sum(P)``                                  (per row)
    4.  ``acc_O *= exp2((row_max_prev - row_max_cur) * scale_log2)`` (per row)

    After the last tile, ``acc_O /= row_sum``.

The ``softmax_scale`` is pre-multiplied by ``log2(e)`` so the kernel can
call ``exp2`` (faster than ``exp`` on Ampere). The factor is folded into
``S`` *and* the ``(row_max_prev - row_max_cur)`` delta -- both inside the
same ``exp2`` -- so the math is algebraically identical to a plain
``softmax`` over the unscaled logits.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

import cutlass
import cutlass.cute as cute

from .utils import make_acc_mn_view, quad_reduce_max, quad_reduce_sum


class Softmax:
    """Per-thread online-softmax state for one Q-tile.

    Owns the ``row_max`` (max-so-far, fp32) and ``row_sum`` (denominator
    accumulator, fp32) registers for each row this thread is responsible
    for. The shape follows the MMA-C accumulator partition.

    Use:

        sm = Softmax.make_from_acc_O(acc_O, softmax_scale_log2)
        sm.init()
        for n_block in ...:
            sm.online_softmax(acc_S, acc_O, is_first=...)
        sm.finalize(acc_O)
    """

    def __init__(
        self,
        *,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
    ):
        self.row_max = row_max
        self.row_sum = row_sum
        self.softmax_scale_log2 = softmax_scale_log2

    # --- Construction --------------------------------------------------------

    @staticmethod
    @cute.jit
    def make_from_acc_O(
        acc_O: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
    ) -> "Softmax":
        """Allocate row_max / row_sum tensors sized to the acc_O M-axis.

        ``acc_O`` is the (2,2)-leaf MMA-C accumulator with shape
        ``((2, 2), MMA_M, MMA_N)``. The per-row state has length
        ``acc_O.shape[0][0] * acc_O.shape[1]`` (= row-pair * MMA_M).
        """
        nrows = acc_O.shape[0][0] * acc_O.shape[1]
        row_max = cute.make_rmem_tensor(nrows, cutlass.Float32)
        row_sum = cute.make_rmem_tensor(nrows, cutlass.Float32)
        return Softmax(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

    @cute.jit
    def init(self):
        """Reset row_max to -inf and row_sum to 0 for a fresh Q-tile."""
        self.row_max.fill(-cutlass.Float32.inf)
        self.row_sum.fill(0.0)

    # --- Per-K-tile update ---------------------------------------------------

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        acc_O: cute.Tensor,
        is_first: cutlass.Constexpr,
    ):
        """Apply one K-tile's worth of softmax updates.

        On ``is_first=True`` (the first n-block this Q-tile sees) we skip
        the ``acc_O`` rescale -- it's all zero anyway -- which lets the
        JIT drop a register-rich branch.

        Caller must have already masked invalid columns/rows in ``acc_S``
        to ``-inf`` via :class:`AttentionMask`. The empty-row clamp
        (``row_max == -inf`` => set to 0 to avoid NaN from exp(-inf -
        -inf)) is applied here.
        """
        acc_S_mn = make_acc_mn_view(acc_S)
        acc_O_mn = make_acc_mn_view(acc_O)
        scale_log2 = self.softmax_scale_log2

        # Stash the previous row_max so we can compute the rescale factor.
        if cutlass.const_expr(not is_first):
            row_max_prev = cute.make_fragment_like(self.row_max, cutlass.Float32)
            cute.basic_copy(self.row_max, row_max_prev)
        else:
            row_max_prev = None

        for r in cutlass.range_constexpr(cute.size(self.row_max)):
            acc_S_row = acc_S_mn[r, None].load()

            # Row-max from this tile, reduced across the 4-thread row quad.
            row_max_cur = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max_cur = quad_reduce_max(row_max_cur)
            if cutlass.const_expr(not is_first):
                row_max_cur = cute.arch.fmax(row_max_prev[r], row_max_cur)

            # Empty-row clamp: if every column was masked to -inf, the
            # subsequent ``exp(-inf - -inf)`` is NaN. Replace by 0 so
            # ``exp(-inf - 0) = 0`` and row_sum stays 0; downstream
            # arithmetic stays finite.
            row_max_cur = (
                0.0 if row_max_cur == -cutlass.Float32.inf else row_max_cur
            )

            # P = exp2((S - row_max_cur) * scale_log2). Distributing
            # ``scale_log2`` inside exp2 is mathematically equivalent and
            # lets us share the constant across all elements.
            row_p = cute.math.exp2(
                acc_S_row * scale_log2 - row_max_cur * scale_log2,
                fastmath=True,
            )
            row_sum_cur = row_p.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )

            if cutlass.const_expr(not is_first):
                # Rescale prior row_sum + acc_O by the delta exp.
                delta_exp = cute.math.exp2(
                    row_max_prev[r] * scale_log2 - row_max_cur * scale_log2,
                    fastmath=True,
                )
                row_sum_cur = row_sum_cur + self.row_sum[r] * delta_exp
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * delta_exp

            self.row_max[r] = row_max_cur
            self.row_sum[r] = row_sum_cur
            acc_S_mn[r, None] = row_p

    # --- Epilogue ------------------------------------------------------------

    @cute.jit
    def finalize(self, acc_O: cute.Tensor):
        """Divide acc_O by row_sum (with 4-thread quad reduction)."""
        acc_O_mn = make_acc_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(self.row_sum)):
            self.row_sum[r] = quad_reduce_sum(self.row_sum[r])
            is_bad = self.row_sum[r] == 0.0 or self.row_sum[r] != self.row_sum[r]
            scale = 1.0 if is_bad else cute.arch.rcp_approx(self.row_sum[r])
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale


__all__ = ["Softmax"]
