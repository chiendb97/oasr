// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Tuneable traits wrapper — applies preprocessor overrides on top of ArchTraits.
//
// When compiled with -DOASR_GEMM_TILE_M=256 etc., the corresponding fields in
// ArchTraits are overridden.  When no overrides are specified, TuneableTraits
// is identical to ArchTraits (zero-cost abstraction).
//
// Kernel headers (gemm.cuh, bmm.cuh, conv2d.cuh, group_gemm.cuh) use
// TuneableTraits instead of ArchTraits directly so that the autotuner can
// JIT-compile multiple tile variants as separate shared libraries.

#pragma once

#include <oasr/common/arch_traits.h>

namespace oasr {

template <int SmVersion>
struct TuneableTraits {
    // Inherited from ArchTraits — not overridable
    using SmArch = typename ArchTraits<SmVersion>::SmArch;
    using MMAOp = typename ArchTraits<SmVersion>::MMAOp;
    static constexpr bool kSupportsBF16 = ArchTraits<SmVersion>::kSupportsBF16;

    // -------------------------------------------------------------------------
    // GEMM tuneables
    // -------------------------------------------------------------------------
    struct Gemm {
#ifdef OASR_GEMM_TILE_M
        using ThreadBlock = cutlass::gemm::GemmShape<OASR_GEMM_TILE_M,
                                                      OASR_GEMM_TILE_N,
                                                      OASR_GEMM_TILE_K>;
#else
        using ThreadBlock = typename ArchTraits<SmVersion>::Gemm::ThreadBlock;
#endif

#ifdef OASR_GEMM_WARP_M
        using Warps = cutlass::gemm::GemmShape<OASR_GEMM_WARP_M,
                                                OASR_GEMM_WARP_N,
                                                OASR_GEMM_WARP_K>;
#else
        using Warps = typename ArchTraits<SmVersion>::Gemm::Warps;
#endif

        // MMA shape is determined by hardware — not tuneable
        using MMAShape = typename ArchTraits<SmVersion>::Gemm::MMAShape;

#ifdef OASR_GEMM_STAGES
        static constexpr int NumStages = OASR_GEMM_STAGES;
#else
        static constexpr int NumStages = ArchTraits<SmVersion>::Gemm::NumStages;
#endif
    };

    // -------------------------------------------------------------------------
    // Conv2D tuneables
    // -------------------------------------------------------------------------
    struct Conv2d {
#ifdef OASR_CONV2D_TILE_M
        using ThreadBlock = cutlass::gemm::GemmShape<OASR_CONV2D_TILE_M,
                                                      OASR_CONV2D_TILE_N,
                                                      OASR_CONV2D_TILE_K>;
#else
        using ThreadBlock = typename ArchTraits<SmVersion>::Conv2d::ThreadBlock;
#endif

#ifdef OASR_CONV2D_WARP_M
        using Warps = cutlass::gemm::GemmShape<OASR_CONV2D_WARP_M,
                                                OASR_CONV2D_WARP_N,
                                                OASR_CONV2D_WARP_K>;
#else
        using Warps = typename ArchTraits<SmVersion>::Conv2d::Warps;
#endif

        // MMA shape is determined by hardware — not tuneable
        using MMAShape = typename ArchTraits<SmVersion>::Conv2d::MMAShape;

#ifdef OASR_CONV2D_STAGES
        static constexpr int NumStages = OASR_CONV2D_STAGES;
#else
        static constexpr int NumStages = ArchTraits<SmVersion>::Conv2d::NumStages;
#endif

        // These are architecture-dependent, not tuneable
        static constexpr int EpilogueAlignment = ArchTraits<SmVersion>::Conv2d::EpilogueAlignment;
        static constexpr auto IterAlgo = ArchTraits<SmVersion>::Conv2d::IterAlgo;
        static constexpr auto OutStride = ArchTraits<SmVersion>::Conv2d::OutStride;
    };
};

}  // namespace oasr
