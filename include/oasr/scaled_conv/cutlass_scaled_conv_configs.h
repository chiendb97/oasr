// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Tile configurations for the int8 implicit-GEMM convolution (``scaled_conv``).
//
// The GEMM view of an NHWC forward convolution: M = N * P * Q output pixels,
// N = K output channels, K = R * S * C.  int8 operands (16-byte loads: C % 16),
// an exact int32 accumulator (``mma.sync`` 16x8x32 s8), the Sm80 kernel on every
// SM the int8 GEMM lane serves (natively on sm80/86/89/120, as the mixed lane on
// sm90/sm100).  No split-K: one K pass per output tile, so no workspace.

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>

namespace oasr {
namespace scaled_conv {

template <int BM, int BN, int BK, int WM, int WN, int WK, int Stages>
struct CutlassScaledConvConfig {
    using ElementAB = int8_t;
    using ElementAccumulator = int32_t;
    using ThreadblockShape = cutlass::gemm::GemmShape<BM, BN, BK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using ArchTag = cutlass::arch::Sm80;
    // Saturating, as the int8 GEMM: |q| <= 127 cannot overflow an int32 sum below
    // K = R * S * C = 133,000 terms, so it never fires in practice.
    using Operator = cutlass::arch::OpMultiplyAddSaturate;
    static constexpr int kStages = Stages;
};

}  // namespace scaled_conv
}  // namespace oasr
