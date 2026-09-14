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

//==============================================================================
// CutlassConv2dConfigSm90 -- CUTLASS 3.x implicit-GEMM Conv2D config for SM90+
//
// Deliberately *not* a mirror of CutlassGemmConfigSm90.  Three differences, each
// of which was a compile failure on sm_90 and sm_100 while this struct borrowed
// GEMM's shapes and schedules:
//
//   1. The K mode is a **nested** `Shape<Int<BK>>`, not a flat `Int<BK>`.  The
//      implicit-GEMM K axis is the filter's (C, S, R...) modes, so the conv
//      mainloop builds its smem layout with `shape<2>(TileShape)` and hands the
//      result to an im2col TMA descriptor.  A flat K collapses that mode and the
//      failure surfaces ~40 frames down in `copy_traits_sm90_tma.hpp`, nowhere
//      near the tile.  CUTLASS's own conv tests spell it `Shape<_64,_64,Shape<_64>>`
//      for conv1d, conv2d and conv3d alike.
//   2. There is no pingpong / cooperative axis.  `KernelScheduleAuto` is the only
//      schedule conv has on SM90: the persistent variants exist as tags but
//      `conv/dispatch_policy.hpp` static_asserts on them ("Persistent schedules
//      not support for conv yet"), and CUTLASS's own auto-selector has the
//      cooperative branch commented out.  The schedules are named in
//      `conv2d_cutlass_template_sm90.h`, where the builder headers are included.
//   3. `BM` is the MMA tile M, never scaled.  GEMM doubles it for a 2-SM SM100
//      atom; conv does not -- CUTLASS's 2-SM conv tests pass `_256` directly
//      alongside cluster (2,1,1), and `BM * 2` trips "Invalid TileShape_M."
//      The 2-SM atom is selected from the *cluster*, which is why kSMs is gone
//      as a parameter: it carried no information the cluster did not.
//
// `oasr/jit/conv.py` owns the buildable tile space and the reasoning behind it.
//==============================================================================

template <int BM, int BN, int BK, int CM, int CN, int kStages, int kSmVersion>
struct CutlassConv2dConfigSm90 {
    using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Shape<cute::Int<BK>>>;
    // CK is always 1 for implicit-GEMM convolution.
    using ClusterShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<1>>;
    using SmArch = typename gemm::CutlassArch<kSmVersion>::Type;
    static constexpr int SmVersion = kSmVersion;

    // Advisory only: the mainloop uses StageCountAutoCarveout, which derives the
    // depth from what is left of shared memory after the epilogue.
    static constexpr int Stages = kStages;
};

}  // namespace conv
}  // namespace oasr
