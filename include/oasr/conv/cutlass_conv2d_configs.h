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

template <int BM, int BN, int BK, int CM, int CN, int kSMs, int kStages, int kSmVersion,
          bool kPingpong = false>
struct CutlassConv2dConfigSm90 {
    // TileShape: for SM100 with kSMs=2 the tile M dimension is doubled so that
    // both SMs together cover BM*2 rows.  For SM90/SM120 kSMs=1 so no scaling.
    using TileShape = cute::Shape<cute::Int<BM * kSMs>, cute::Int<BN>, cute::Int<BK>>;
    // CK is always 1 for implicit-GEMM convolution.
    using ClusterShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<1>>;
    using SmArch = typename gemm::CutlassArch<kSmVersion>::Type;

    using _ScheduleSelector = gemm::GemmScheduleSelector<kSmVersion, kSMs, kPingpong>;
    using EpilogueSchedule = typename _ScheduleSelector::EpilogueSchedule;
    using MainloopSchedule = typename _ScheduleSelector::MainloopSchedule;

    static constexpr int Stages = kStages;
};

}  // namespace conv
}  // namespace oasr
