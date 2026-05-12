# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GQA index helpers for the OASR FMHA forward kernel.

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/pack_gqa.py``.
The implementation is OASR-native and (in this revision) supports only the
*one Q head per CTA* mode -- the same dispatch the existing OASR kernel uses.

The packed-along-M optimization (where ``qhead_per_kvhead`` Q heads are
folded into the M axis so one CTA tile shares one KV head) is a future
phase-B optimization. The interface here is shaped so that switching to
the packed mode is a kernel-side change only -- callers can stay
parameterized on ``PackGQA``.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

import cutlass
import cutlass.cute as cute


class PackGQA:
    """Constexpr GQA descriptor.

    Fields:

    * ``qhead_per_kvhead`` -- ``num_heads // num_kv_heads``. Constexpr.
    * ``packed`` -- whether Q heads are packed along M. Currently always
                    False (per-Q-head CTA dispatch).
    """

    def __init__(
        self,
        *,
        qhead_per_kvhead: cutlass.Constexpr,
        packed: cutlass.Constexpr = False,
    ):
        self.qhead_per_kvhead = qhead_per_kvhead
        self.packed = packed

    @cute.jit
    def kv_head_of(self, q_head: cutlass.Int32) -> cutlass.Int32:
        """Map a Q head index to its KV head."""
        return q_head // self.qhead_per_kvhead


__all__ = ["PackGQA"]
