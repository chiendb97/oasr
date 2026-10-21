// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Tile configurations for the int8 scaled GEMM (``D = act(a_scale ⊗ b_scale ·
// (A_int8 @ B_int8ᵀ) + bias)``).
//
// One lane today: CUTLASS 2.x ``mma.sync`` int8 (``16x8x32``) with an int32
// accumulator.  It is the Sm80 kernel on every SM it serves — natively on
// sm80/86/89/120, and as the mixed lane on sm90/sm100, exactly as the fp16 GEMM
// runs there (``oasr/jit/gemm.py::_MIXED_LANE_SMS``).

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>

namespace oasr {
namespace scaled_mm {

template <int BM, int BN, int BK, int WM, int WN, int WK, int Stages>
struct CutlassScaledMmConfig {
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

}  // namespace scaled_mm
}  // namespace oasr
