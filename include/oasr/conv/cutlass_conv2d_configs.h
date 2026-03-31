// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS Conv2D configuration structs.
//   SM75–89: CutlassConv2dConfig  (CUTLASS 2.x, DefaultConv2dFprop)
//   SM90+:   CutlassConv2dConfigSm90  (CUTLASS 3.x, CollectiveBuilder — defined in
//             conv2d_cutlass_template_sm90.h)
//
// Member naming follows what conv2d_cutlass_template.h expects:
//   ThreadBlock, Warps, MMAShape, NumStages, SmArch

#pragma once

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/gemm/gemm.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/gemm/cutlass_gemm_configs.h>

namespace oasr {
namespace conv {

//==============================================================================
// CutlassConv2dConfig -- CUTLASS 2.x Conv2D configuration for SM75–89
//
// BM/BN/BK: threadblock tile shape (M = N*P*Q, N = K, K = R*S*IC in GEMM view)
// WM/WN/WK: warp tile shape
// kStages:  pipeline depth (typically 3–4)
// kSmVersion: 75 | 80 | 86 | 89
//==============================================================================

template <int BM, int BN, int BK, int WM, int WN, int WK, int kStages, int kSmVersion>
struct CutlassConv2dConfig {
    // Member names expected by conv2d_cutlass_template.h
    using ThreadBlock = cutlass::gemm::GemmShape<BM, BN, BK>;
    using Warps = cutlass::gemm::GemmShape<WM, WN, WK>;
    using MMAShape = typename gemm::CutlassArch<kSmVersion>::InstructionShape;
    using SmArch = typename gemm::CutlassArch<kSmVersion>::Type;
    static constexpr int NumStages = kStages;
};

template <int BM, int BN, int BK, int CM, int CN, int CK, int kSMs, int kStages, int kSmVersion>
struct CutlassConv2dConfigSm90 {
    using TileShape = cute::Shape<cute::Int<BM * kSMs>, cute::Int<BN>, cute::Int<BK>>;
    using ClusterShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>;
    using SmArch = typename gemm::CutlassArch<kSmVersion>::Type;
    using EpilogueSchedule = typename gemm::SMTypeAdapter<kSMs>::EpilogueSchedule;
    using MainloopSchedule = typename gemm::SMTypeAdapter<kSMs>::MainloopSchedule;
    static constexpr int Stages = kStages;
};

}  // namespace conv
}  // namespace oasr
