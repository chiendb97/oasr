// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-architecture kernel configuration traits for CUTLASS 2.x API

#pragma once

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/conv/convolution.h>
#include <cutlass/gemm/gemm.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

namespace oasr {
namespace kernels {

// Primary template — unspecialized triggers a compile error
template <int SmVersion>
struct ArchTraits;

//==============================================================================
// SM70 — Volta
//==============================================================================

template <>
struct ArchTraits<70> {
    using SmArch = cutlass::arch::Sm70;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    static constexpr bool kSupportsBF16 = false;

    struct Gemm {
        using ThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
        using Warps = cutlass::gemm::GemmShape<64, 64, 32>;
        using MMAShape = cutlass::gemm::GemmShape<8, 8, 4>;
        static constexpr int NumStages = 2;
    };

    struct Conv2d {
        using ThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
        using Warps = cutlass::gemm::GemmShape<64, 64, 32>;
        using MMAShape = cutlass::gemm::GemmShape<8, 8, 4>;
        static constexpr int NumStages = 2;
        static constexpr int EpilogueAlignment = 8;
        static constexpr auto IterAlgo = cutlass::conv::IteratorAlgorithm::kAnalytic;
        static constexpr auto OutStride = cutlass::conv::StrideSupport::kStrided;
    };
};

//==============================================================================
// SM75 — Turing
//==============================================================================

template <>
struct ArchTraits<75> {
    using SmArch = cutlass::arch::Sm75;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    static constexpr bool kSupportsBF16 = false;

    struct Gemm {
        using ThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
        using Warps = cutlass::gemm::GemmShape<64, 64, 32>;
        using MMAShape = cutlass::gemm::GemmShape<16, 8, 8>;
        static constexpr int NumStages = 2;
    };

    struct Conv2d {
        using ThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
        using Warps = cutlass::gemm::GemmShape<64, 64, 32>;
        using MMAShape = cutlass::gemm::GemmShape<16, 8, 8>;
        static constexpr int NumStages = 2;
        static constexpr int EpilogueAlignment = 8;
        static constexpr auto IterAlgo = cutlass::conv::IteratorAlgorithm::kAnalytic;
        static constexpr auto OutStride = cutlass::conv::StrideSupport::kStrided;
    };
};

//==============================================================================
// SM80 — Ampere (also covers SM86, SM89 via resolveSmVersion)
//==============================================================================

template <>
struct ArchTraits<80> {
    using SmArch = cutlass::arch::Sm80;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    static constexpr bool kSupportsBF16 = true;

    struct Gemm {
        using ThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
        using Warps = cutlass::gemm::GemmShape<64, 64, 32>;
        using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
        static constexpr int NumStages = 2;
    };

    struct Conv2d {
        using ThreadBlock = cutlass::gemm::GemmShape<128, 128, 64>;
        using Warps = cutlass::gemm::GemmShape<64, 64, 64>;
        using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
        static constexpr int NumStages = 3;
        static constexpr int EpilogueAlignment = 8;
        static constexpr auto IterAlgo = cutlass::conv::IteratorAlgorithm::kOptimized;
        static constexpr auto OutStride = cutlass::conv::StrideSupport::kUnity;
    };
};

//==============================================================================
// SM90 — Hopper (CUTLASS 2.x API — uses Sm80 arch tag)
//==============================================================================

template <>
struct ArchTraits<90> {
    using SmArch = cutlass::arch::Sm80;  // Sm80 tag for CUTLASS 2.x API compatibility
    using MMAOp = cutlass::arch::OpClassTensorOp;
    static constexpr bool kSupportsBF16 = true;

    struct Gemm {
        using ThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
        using Warps = cutlass::gemm::GemmShape<64, 64, 32>;
        using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
        static constexpr int NumStages = 4;
    };

    struct Conv2d {
        using ThreadBlock = cutlass::gemm::GemmShape<128, 128, 64>;
        using Warps = cutlass::gemm::GemmShape<64, 64, 64>;
        using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
        static constexpr int NumStages = 4;
        static constexpr int EpilogueAlignment = 8;
        static constexpr auto IterAlgo = cutlass::conv::IteratorAlgorithm::kOptimized;
        static constexpr auto OutStride = cutlass::conv::StrideSupport::kUnity;
    };
};

}  // namespace kernels
}  // namespace oasr
