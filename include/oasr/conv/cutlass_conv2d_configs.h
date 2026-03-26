// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// FlashInfer-style Conv2D configuration system.
//
// Replaces ArchTraits<SM>::Conv2d and TuneableTraits<SM>::Conv2d with direct
// template-parameterized config structs and per-SM defaults.

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
namespace conv {

//==============================================================================
// Conv2dConfig -- direct template-parameterized tile configuration
//==============================================================================

template <int TileM_, int TileN_, int TileK_,
          int WarpM_, int WarpN_, int WarpK_,
          int Stages_,
          int EpilogueAlignment_ = 8,
          cutlass::conv::IteratorAlgorithm IterAlgo_ = cutlass::conv::IteratorAlgorithm::kOptimized,
          cutlass::conv::StrideSupport OutStride_ = cutlass::conv::StrideSupport::kUnity>
struct Conv2dConfig {
    static constexpr int TileM = TileM_;
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;
    static constexpr int WarpM = WarpM_;
    static constexpr int WarpN = WarpN_;
    static constexpr int WarpK = WarpK_;
    static constexpr int Stages = Stages_;
    static constexpr int EpilogueAlignment = EpilogueAlignment_;
    static constexpr cutlass::conv::IteratorAlgorithm IterAlgo = IterAlgo_;
    static constexpr cutlass::conv::StrideSupport OutStride = OutStride_;

    using ThreadBlock = cutlass::gemm::GemmShape<TileM_, TileN_, TileK_>;
    using Warps = cutlass::gemm::GemmShape<WarpM_, WarpN_, WarpK_>;
    static constexpr int NumStages = Stages_;
};

//==============================================================================
// SmConv2dTraits -- per-SM MMA traits for Conv2D (hardware-determined)
//==============================================================================

template <int SmVersion>
struct SmConv2dTraits;

template <>
struct SmConv2dTraits<70> {
    using MMAShape = cutlass::gemm::GemmShape<8, 8, 4>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm70;
};

template <>
struct SmConv2dTraits<75> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 8>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm75;
};

template <>
struct SmConv2dTraits<80> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmConv2dTraits<86> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmConv2dTraits<89> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmConv2dTraits<90> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmConv2dTraits<100> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmConv2dTraits<103> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmConv2dTraits<120> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

//==============================================================================
// Named config aliases
//==============================================================================

// SM70/75 -- analytic iterator, strided output
using Conv2dConfig_128x128x32_s2_analytic = Conv2dConfig<
    128, 128, 32, 64, 64, 32, 2, 8,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kStrided>;

// SM80+ -- optimized iterator, unity stride
using Conv2dConfig_128x128x64_s3 = Conv2dConfig<128, 128, 64, 64, 64, 64, 3>;
using Conv2dConfig_128x128x64_s4 = Conv2dConfig<128, 128, 64, 64, 64, 64, 4>;
using Conv2dConfig_128x128x32_s3 = Conv2dConfig<128, 128, 32, 64, 64, 32, 3>;
using Conv2dConfig_256x128x32_s3 = Conv2dConfig<256, 128, 32, 64, 64, 32, 3>;
using Conv2dConfig_128x256x32_s3 = Conv2dConfig<128, 256, 32, 64, 64, 32, 3>;
using Conv2dConfig_64x64x64_s4   = Conv2dConfig<64, 64, 64, 32, 32, 64, 4>;

//==============================================================================
// DefaultConv2dConfig -- per-SM default config
//==============================================================================

template <int SmVersion>
struct DefaultConv2dConfig;

template <>
struct DefaultConv2dConfig<70> { using type = Conv2dConfig_128x128x32_s2_analytic; };

template <>
struct DefaultConv2dConfig<75> { using type = Conv2dConfig_128x128x32_s2_analytic; };

template <>
struct DefaultConv2dConfig<80> { using type = Conv2dConfig_128x128x64_s3; };

template <>
struct DefaultConv2dConfig<86> { using type = Conv2dConfig_128x128x64_s3; };

template <>
struct DefaultConv2dConfig<89> { using type = Conv2dConfig_128x128x64_s3; };

template <>
struct DefaultConv2dConfig<90> { using type = Conv2dConfig_128x128x64_s4; };

template <>
struct DefaultConv2dConfig<100> { using type = Conv2dConfig_128x128x64_s4; };

template <>
struct DefaultConv2dConfig<103> { using type = Conv2dConfig_128x128x64_s4; };

template <>
struct DefaultConv2dConfig<120> { using type = Conv2dConfig_128x128x64_s4; };

//==============================================================================
// JIT compile-time config selection via -D flags
//==============================================================================

#ifdef OASR_CONV2D_TILE_M
using JitConv2dConfig = Conv2dConfig<OASR_CONV2D_TILE_M, OASR_CONV2D_TILE_N, OASR_CONV2D_TILE_K,
                                      OASR_CONV2D_WARP_M, OASR_CONV2D_WARP_N, OASR_CONV2D_WARP_K,
                                      OASR_CONV2D_STAGES>;
#endif

}  // namespace conv
}  // namespace oasr
