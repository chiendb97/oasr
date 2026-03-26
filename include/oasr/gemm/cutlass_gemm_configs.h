// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// FlashInfer-style GEMM configuration system.
//
// Replaces ArchTraits<SM>::Gemm and TuneableTraits<SM>::Gemm with direct
// template-parameterized config structs, per-SM MMA traits, and default
// config selection.

#pragma once

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

namespace oasr {
namespace gemm {

//==============================================================================
// GemmConfig — direct template-parameterized tile configuration
//==============================================================================

template <int TileM_, int TileN_, int TileK_,
          int WarpM_, int WarpN_, int WarpK_,
          int Stages_>
struct GemmConfig {
    static constexpr int TileM = TileM_;
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;
    static constexpr int WarpM = WarpM_;
    static constexpr int WarpN = WarpN_;
    static constexpr int WarpK = WarpK_;
    static constexpr int Stages = Stages_;

    using ThreadBlock = cutlass::gemm::GemmShape<TileM_, TileN_, TileK_>;
    using Warps = cutlass::gemm::GemmShape<WarpM_, WarpN_, WarpK_>;
    static constexpr int NumStages = Stages_;
};

//==============================================================================
// Named config aliases for common tile configurations
//==============================================================================

using GemmConfig_128x128x32_s2 = GemmConfig<128, 128, 32, 64, 64, 32, 2>;
using GemmConfig_128x128x32_s3 = GemmConfig<128, 128, 32, 64, 64, 32, 3>;
using GemmConfig_128x128x32_s4 = GemmConfig<128, 128, 32, 64, 64, 32, 4>;
using GemmConfig_256x128x32_s3 = GemmConfig<256, 128, 32, 64, 64, 32, 3>;
using GemmConfig_128x256x32_s3 = GemmConfig<128, 256, 32, 64, 64, 32, 3>;
using GemmConfig_64x64x32_s4   = GemmConfig<64, 64, 32, 32, 32, 32, 4>;
using GemmConfig_128x64x32_s3  = GemmConfig<128, 64, 32, 64, 32, 32, 3>;
using GemmConfig_64x128x32_s3  = GemmConfig<64, 128, 32, 32, 64, 32, 3>;
using GemmConfig_128x128x64_s3 = GemmConfig<128, 128, 64, 64, 64, 64, 3>;

//==============================================================================
// SmMMATraits — per-SM hardware-determined MMA traits (not tunable)
//==============================================================================

template <int SmVersion>
struct SmMMATraits;

// SM70 — Volta
template <>
struct SmMMATraits<70> {
    using MMAShape = cutlass::gemm::GemmShape<8, 8, 4>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm70;
    static constexpr bool kSupportsBF16 = false;
};

// SM75 — Turing
template <>
struct SmMMATraits<75> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 8>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm75;
    static constexpr bool kSupportsBF16 = false;
};

// SM80 — Ampere (GA100, e.g. A100)
template <>
struct SmMMATraits<80> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
    static constexpr bool kSupportsBF16 = true;
};

// SM86 — Ampere (GA102, e.g. RTX 3090, A40)
template <>
struct SmMMATraits<86> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;  // Shares SM80 MMA ISA
    static constexpr bool kSupportsBF16 = true;
};

// SM89 — Ada Lovelace (e.g. RTX 4090, L40)
template <>
struct SmMMATraits<89> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;  // Shares SM80 MMA ISA
    static constexpr bool kSupportsBF16 = true;
};

// SM90 — Hopper (e.g. H100, H200)
template <>
struct SmMMATraits<90> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;  // CUTLASS 2.x API compatibility
    static constexpr bool kSupportsBF16 = true;
};

// SM100 — Blackwell (e.g. B200, GB200)
template <>
struct SmMMATraits<100> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;  // CUTLASS 2.x API compatibility
    static constexpr bool kSupportsBF16 = true;
};

// SM103 — Blackwell consumer (e.g. RTX 5090)
template <>
struct SmMMATraits<103> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;  // CUTLASS 2.x API compatibility
    static constexpr bool kSupportsBF16 = true;
};

// SM120 — Next-generation
template <>
struct SmMMATraits<120> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;  // CUTLASS 2.x API compatibility
    static constexpr bool kSupportsBF16 = true;
};

//==============================================================================
// DefaultGemmConfig — per-SM default tile config selection
//==============================================================================

template <int SmVersion>
struct DefaultGemmConfig;

template <>
struct DefaultGemmConfig<70> { using type = GemmConfig_128x128x32_s2; };

template <>
struct DefaultGemmConfig<75> { using type = GemmConfig_128x128x32_s2; };

template <>
struct DefaultGemmConfig<80> { using type = GemmConfig_128x128x32_s2; };

template <>
struct DefaultGemmConfig<86> { using type = GemmConfig_128x128x32_s3; };

template <>
struct DefaultGemmConfig<89> { using type = GemmConfig_128x128x32_s3; };

template <>
struct DefaultGemmConfig<90> { using type = GemmConfig_128x128x32_s4; };

template <>
struct DefaultGemmConfig<100> { using type = GemmConfig_128x128x32_s4; };

template <>
struct DefaultGemmConfig<103> { using type = GemmConfig_128x128x32_s4; };

template <>
struct DefaultGemmConfig<120> { using type = GemmConfig_128x128x32_s4; };

//==============================================================================
// JitGemmConfig — compile-time config selection via -D flags
//==============================================================================

#ifdef OASR_GEMM_TILE_M
using JitGemmConfig = GemmConfig<OASR_GEMM_TILE_M, OASR_GEMM_TILE_N, OASR_GEMM_TILE_K,
                                  OASR_GEMM_WARP_M, OASR_GEMM_WARP_N, OASR_GEMM_WARP_K,
                                  OASR_GEMM_STAGES>;
#endif

}  // namespace gemm
}  // namespace oasr
