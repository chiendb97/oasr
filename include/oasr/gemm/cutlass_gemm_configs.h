// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Cutlass GEMM configuration templates
// Cutlass 2.x SM75/SM80/SM86/SM89 dispatch.
// Cutlass 3.x SM90+/SM100+/SM120+ dispatch.
//
#pragma once

#include <cassert>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

namespace oasr {
namespace gemm {

template <int SmVersion>
struct CutlassArch;

template <>
struct CutlassArch<75> {
    using Type = cutlass::arch::Sm75;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
};
template <>
struct CutlassArch<80> {
    using Type = cutlass::arch::Sm80;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};
template <>
struct CutlassArch<86> {
    using Type = cutlass::arch::Sm86;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};
template <>
struct CutlassArch<89> {
    using Type = cutlass::arch::Sm89;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};

template <>
struct CutlassArch<90> {
    using Type = cutlass::arch::Sm90;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
};

template <>
struct CutlassArch<100> {
    using Type = cutlass::arch::Sm100;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
};

template <>
struct CutlassArch<120> {
    using Type = cutlass::arch::Sm120;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
};

template <int BM, int BN, int BK, int WM, int WN, int WK, int kStages, int kSmVersion>
struct CutlassGemmConfig {
    using ThreadblockShape = cutlass::gemm::GemmShape<BM, BN, BK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;

    using SmArch = typename CutlassArch<kSmVersion>::Type;
    using InstructionShape = typename CutlassArch<kSmVersion>::InstructionShape;

    static constexpr int Stages = kStages;
};

//==============================================================================
// CutlassGemmConfigSm90 -- CUTLASS 3.x GEMM configuration for SM90+
//==============================================================================

template <int>
struct SMTypeAdapter;

template <>
struct SMTypeAdapter<1> {
    static constexpr int Scale = 1;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
};

template <>
struct SMTypeAdapter<2> {
    static constexpr int Scale = 2;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
};

template <int BM, int BN, int BK, int CM, int CN, int CK, int kSMs, int kStages, int kSmVersion>
struct CutlassGemmConfigSm90 {
    using TileShape = cute::Shape<cute::Int<BM * kSMs>, cute::Int<BN>, cute::Int<BK>>;
    using ClusterShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;
    using SmArch = typename CutlassArch<kSmVersion>::Type;
    using EpilogueSchedule = typename SMTypeAdapter<kSMs>::EpilogueSchedule;
    using MainloopSchedule = typename SMTypeAdapter<kSMs>::MainloopSchedule;
    static constexpr int Stages = kStages;
};

}  // namespace gemm
}  // namespace oasr
