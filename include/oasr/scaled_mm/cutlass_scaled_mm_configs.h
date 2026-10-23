// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Tile configurations for the scaled GEMM (``D = act(a_scale ⊗ b_scale ·
// (A @ Bᵀ) + bias)`` over 8-bit A and B).
//
// Two lanes, one template (``scaled_mm_cutlass_template.h``), differing only in
// what a config names:
//
// * int8 -- CUTLASS 2.x ``mma.sync`` ``16x8x32`` s8 with an int32 accumulator.
//   The Sm80 kernel on every SM it serves: natively on sm80/86/89/120, and as
//   the mixed lane on sm90/sm100, exactly as the fp16 GEMM runs there
//   (``oasr/jit/gemm.py::_MIXED_LANE_SMS``).
// * fp8 (e4m3) -- CUTLASS 2.x ``mma.sync`` ``16x8x32`` e4m3 (Ada's instruction,
//   which sm_120 also executes) with an fp32 accumulator, on sm89/sm120.

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/float8.h>
#include <cutlass/gemm/gemm.h>

namespace oasr {
namespace scaled_mm {

template <int BM, int BN, int BK, int WM, int WN, int WK, int Stages>
struct CutlassScaledMmConfig {
    using ElementAB = int8_t;
    using ElementAccumulator = int32_t;
    using ThreadblockShape = cutlass::gemm::GemmShape<BM, BN, BK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using ArchTag = cutlass::arch::Sm80;
    // Saturating int8 MMA: the accumulator clamps instead of wrapping.  At
    // |q| <= 127 an int32 accumulator cannot overflow below K = 133,000, so
    // saturation never fires in practice -- it is the defined behaviour if it does.
    using Operator = cutlass::arch::OpMultiplyAddSaturate;
    static constexpr int kStages = Stages;
};

// fp8 e4m3 x e4m3 -> fp32.  ``OpMultiplyAddFastAccum``: every MMA accumulates
// straight into the running fp32 fragment, in K order.  (``OpMultiplyAdd`` would
// select CUTLASS's Sm89 *staged* accumulation instead -- each warp sums one
// K tile's MMAs into a zeroed fragment, then adds it in -- which makes the
// summation order, and so the bits, a function of the warp's K extent.)  With
// no K sliced across warps (``BK == WK``, asserted) and no split-K or Stream-K,
// fast accumulation sums every output in the same order whatever the tile, so
// every fp8 config computes the same bits and the lane is batch-invariant
// although its accumulator is not exact.  Measured on an RTX 5090 (2026-10-21):
// every tile bit-identical at K = 2048 / 4096 / 18944; 0.1 % (staged) vs 0.3-1 %
// (fast) of fp16 outputs one ulp off the correctly rounded exact value, mean
// error identical (the fp8 operands dominate); fast up to 10 % faster at large M.
template <int BM, int BN, int BK, int WM, int WN, int WK, int Stages>
struct CutlassScaledMmFp8Config {
    using ElementAB = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using ThreadblockShape = cutlass::gemm::GemmShape<BM, BN, BK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using ArchTag = cutlass::arch::Sm89;
    using Operator = cutlass::arch::OpMultiplyAddFastAccum;
    static constexpr int kStages = Stages;
    static_assert(BK == WK, "fp8 tiles must not slice K across warps (summation order)");
};

}  // namespace scaled_mm
}  // namespace oasr
