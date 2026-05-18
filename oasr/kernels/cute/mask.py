# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention-mask + score-mod block for the OASR FMHA forward kernel.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/mask.py``. The
implementation is OASR-native; only the element-wise ``-inf`` mask path is
supported (no SM100 R2P bitmask). Causal, local (sliding-window), per-stream
length, and a generic ``score_mod`` callable are all supported.

The class operates on the mn-view of an MMA-C accumulator: ``acc_S_mn[r, c]``
is the (row, col) entry within the Q-tile x K-tile mma output. The tile's
absolute global coordinates come from an identity tensor ``tScS`` partitioned
through the same MMA partition; ``tScS_mn[r, c][2]`` is the absolute q_row
and ``tScS_mn[r, c][3]`` is the absolute k_col.

``score_mod`` (when provided) is invoked *before* the mask comparison so the
plug-in can add bias / softcap / temperature without the mask path having to
care about it. The kernel uses this to fold ``attn_bias`` into the inner
loop without a separate pass.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

import cutlass
import cutlass.cute as cute

from .utils import make_acc_mn_view


class AttentionMask:
    """Per-block mask + optional score_mod hook.

    All flags are compile-time constexprs so the JIT specializes one
    variant per (causal, window_left, window_right, has_seqlen_k) tuple.

    ``window_left == -1`` / ``window_right == -1`` mean *unbounded* on that
    side. ``causal == True`` is equivalent to ``window_right == 0`` plus an
    implicit unbounded left window; we keep ``causal`` as a separate flag
    purely for readability.
    """

    def __init__(
        self,
        *,
        causal: cutlass.Constexpr,
        window_left: cutlass.Constexpr,
        window_right: cutlass.Constexpr,
        has_seqlen_k: cutlass.Constexpr,
        has_seqlen_q: cutlass.Constexpr,
    ):
        self.causal = causal
        self.window_left = window_left
        self.window_right = window_right
        self.has_seqlen_k = has_seqlen_k
        self.has_seqlen_q = has_seqlen_q

    @cute.jit
    def apply(
        self,
        acc_S: cute.Tensor,
        tScS: cute.Tensor,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        score_mod=None,
    ):
        """Mutate ``acc_S`` in place: apply score_mod, then mask to -inf.

        ``score_mod`` (optional) is ``Callable[[s: f32, r: int, c: int],
        f32]`` and is invoked per element in (q_row, k_col) coordinates.
        The bias kernel uses it to add the divided-by-scale bias entry.
        """
        acc_S_mn = make_acc_mn_view(acc_S)
        tScS_mn = make_acc_mn_view(tScS)
        # Tuple indices into mQ.layout.shape: (B, H, T_q=2, D=3) and the
        # bias / mask tile uses (B, H, T_q=2, T_k=3). Both line up because
        # both partition through the same MMA layout.
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            q_row = tScS_mn[r, 0][2]
            row_oob = False
            if cutlass.const_expr(self.has_seqlen_q):
                row_oob = cute.elem_less(seqlen_q, q_row + 1)

            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                k_col = tScS_mn[r, c][3]
                if cutlass.const_expr(score_mod is not None):
                    acc_S_mn[r, c] = score_mod(acc_S_mn[r, c], r, c)

                # Compose mask conditions; any one => -inf.
                masked = row_oob
                if cutlass.const_expr(self.has_seqlen_k):
                    masked = masked or cute.elem_less(seqlen_k, k_col + 1)
                if cutlass.const_expr(self.causal):
                    # k_col > q_row  -> mask
                    masked = masked or cute.elem_less(q_row, k_col)
                if cutlass.const_expr(self.window_right >= 0):
                    masked = masked or cute.elem_less(
                        q_row + self.window_right, k_col
                    )
                if cutlass.const_expr(self.window_left >= 0):
                    masked = masked or cute.elem_less(
                        k_col + self.window_left, q_row
                    )
                if masked:
                    acc_S_mn[r, c] = -cutlass.Float32.inf


__all__ = ["AttentionMask"]
