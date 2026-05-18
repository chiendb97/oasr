# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Generic CuteDSL utilities used by the OASR attention kernels.

Notes
-----
The module shape (a single ``utils.py`` holding row-reductions, fastdiv, the
``log2(e)``-folded softmax scale, etc.) is structurally modelled on
FlashAttention's ``flash_attn/cute/utils.py``. The implementations here are
OASR-native and only cover what the SM80 / SM120 forward pass actually uses.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

from typing import Tuple

import cutlass
import cutlass.cute as cute


LOG2_E = 1.4426950408889634074


def compute_softmax_scale_log2(softmax_scale: cutlass.Float32) -> cutlass.Float32:
    """Pre-multiply ``softmax_scale`` by ``log2(e)``.

    Used so the online softmax can call ``exp2`` (a faster intrinsic on
    Ampere) instead of ``exp``: ``exp(x * scale) == exp2(x * scale * log2(e))``.
    """
    return softmax_scale * LOG2_E


# ---------------------------------------------------------------------------
# Warp-shuffle row reductions
# ---------------------------------------------------------------------------
# m16n8k16 lays out the accumulator so each row of the MMA-C tile is owned by
# a 4-thread quad (tid % 4 in {0,1,2,3} share the same row). The row-max /
# row-sum reductions therefore need a 2-step bfly shuffle (offsets 2 and 1).

def _quad_reduce(val, op):
    val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
    val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
    return val


def quad_reduce_max(val):
    """Two-step bfly max across a 4-thread row quad."""
    return _quad_reduce(val, lambda x, y: cute.arch.fmax(x, y))


def quad_reduce_sum(val):
    """Two-step bfly sum across a 4-thread row quad."""
    return _quad_reduce(val, lambda x, y: x + y)


# ---------------------------------------------------------------------------
# FastDiv (host-side magic-number generation)
# ---------------------------------------------------------------------------
# When a divisor is known at launch time but not at compile time (e.g.
# ``num_heads // num_kv_heads`` for GQA), we can replace ``/`` and ``%`` with
# a multiply-high-shift sequence. The kernel only needs ``(divisor, mul_hi,
# shift)`` -- evaluating them on the host avoids a 70-cycle integer divide
# inside the kernel.

class FastDivmod:
    """Holds the magic numbers for an unsigned 32-bit fast divmod.

    Host-side construction:
        fd = FastDivmod.make(divisor)
        # pass fd.divisor / fd.mul_hi / fd.shift into the kernel as scalars.

    Device-side use lives inline in the kernel because cute.jit's
    constexpr machinery doesn't let us return tuples through a free function.
    """

    __slots__ = ("divisor", "mul_hi", "shift")

    def __init__(self, divisor: int, mul_hi: int, shift: int):
        self.divisor = divisor
        self.mul_hi = mul_hi
        self.shift = shift

    @staticmethod
    def make(divisor: int) -> "FastDivmod":
        if divisor <= 0:
            raise ValueError(f"FastDivmod divisor must be positive, got {divisor}")
        if divisor == 1:
            return FastDivmod(1, 1, 0)
        # Classic Hacker's Delight 10-1 magic-number algorithm restricted to
        # unsigned 32-bit divisors. The kernel evaluates
        # ``q = ((n * mul_hi) >> 32) >> shift`` and ``r = n - q * divisor``.
        nc = ((1 << 32) // divisor) * divisor - 1
        for p in range(32, 64):
            if (1 << p) > nc * (divisor - (1 << p) % divisor):
                mul_hi = (((1 << p) + divisor - (1 << p) % divisor) // divisor) & 0xFFFFFFFF
                shift = p - 32
                return FastDivmod(divisor, mul_hi, shift)
        raise RuntimeError(f"FastDivmod magic generation failed for divisor={divisor}")


# ---------------------------------------------------------------------------
# Score-mod helpers
# ---------------------------------------------------------------------------

def softcap(x: cutlass.Float32, cap: cutlass.Float32) -> cutlass.Float32:
    """``softcap(x, c) = c * tanh(x / c)``.

    Used by Gemma / softcap-trained models. Kept here because it's a
    one-liner ``score_mod`` that the kernel can plug into ``AttentionMask``.
    """
    return cap * cute.math.tanh(x / cap, fastmath=True)


# ---------------------------------------------------------------------------
# mn-view of an MMA accumulator
# ---------------------------------------------------------------------------
# The m16n8k16 accumulator layout is ((2, 2), MMA_M, MMA_N) where the inner
# (2, 2) is (col-pair, row-pair). For per-row softmax we want an mn-view
# ((row_pair, MMA_M), (col_pair, MMA_N)) so ``acc[r, c]`` is the (row, col)
# entry within the tile. The shape/stride swizzle below is identical to the
# old _make_acc_tensor_mn_view in fmha_sm80.py but lives here so other
# kernels can reuse it.

def make_acc_mn_view(acc: cute.Tensor) -> cute.Tensor:
    """Re-layout an MMA-C accumulator into per-row indexing.

    Input shape: ``((2, 2), MMA_M, MMA_N)``.
    Output shape: ``((2, MMA_M), (2, MMA_N))`` i.e. ``(rows, cols)``.
    """
    acc_layout = cute.make_layout(acc.layout.shape)
    mn_layout = cute.make_layout(
        (
            (acc_layout.shape[0][1], acc_layout.shape[1]),
            (acc_layout.shape[0][0], acc_layout.shape[2]),
        ),
        stride=(
            (acc_layout.stride[0][1], acc_layout.stride[1]),
            (acc_layout.stride[0][0], acc_layout.stride[2]),
        ),
    )
    mn_layout = cute.composition(acc.layout, mn_layout)
    return cute.make_tensor(acc.iterator, mn_layout)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "LOG2_E",
    "compute_softmax_scale_log2",
    "quad_reduce_max",
    "quad_reduce_sum",
    "FastDivmod",
    "softcap",
    "make_acc_mn_view",
]
