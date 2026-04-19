// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Grouped GEMM kernel -- pure CUDA + CUTLASS.
// Uses GemmConfig + SmMMATraits dispatch (FlashInfer-style).
// Requires SM >= 80 (Ampere or newer).

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/utils.h>
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_cutlass_template.h>

namespace oasr {
namespace gemm {



//==============================================================================
// Dispatch helpers
//==============================================================================

namespace detail {

template <int SM_VERSION, typename ElementA, typename ElementB,
          typename ElementCD>
static GemmStatus dispatchGroupGemmWithSmVersion(int problem_count, int K, int N,
                                               const ElementA* A_ptr, const ElementB* B_ptr,
                                               ElementCD* D_ptr, const int* offsets_host,
                                               cudaStream_t stream) {

    GroupedGemmProblemDesc<ElementA, ElementB, ElementCD> problem_desc(problem_count, K, N, A_ptr,
                                                                       B_ptr, D_ptr, offsets_host);
    if constexpr (SM_VERSION == 75) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 75>;
        return CutlassGroupGemmKernel<Config, ElementA, ElementB, ElementCD>::run(problem_desc, problem_count, stream);
    } else if constexpr (SM_VERSION == 80) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 80>;
        return CutlassGroupGemmKernel<Config, ElementA, ElementB, ElementCD>::run(problem_desc, problem_count, stream);
    } else if constexpr (SM_VERSION == 86) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 86>;
        return CutlassGroupGemmKernel<Config, ElementA, ElementB, ElementCD>::run(problem_desc, problem_count, stream);
    } else if constexpr (SM_VERSION == 89) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 89>;
        return CutlassGroupGemmKernel<Config, ElementA, ElementB, ElementCD>::run(problem_desc, problem_count, stream);
    } else if constexpr (SM_VERSION == 90) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 90>;
        return CutlassGroupGemmKernel<Config, ElementA, ElementB, ElementCD>::run(problem_desc, problem_count, stream);
    } else if constexpr (SM_VERSION == 100) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 100>;
        return CutlassGroupGemmKernel<Config, ElementA, ElementB, ElementCD>::run(problem_desc, problem_count, stream);
    } else if constexpr (SM_VERSION == 120) {
        // SM120 TMA warp-specialised builder supports only F8/F6/F4 MMA; fall
        // back to the CUTLASS 2.x path (Sm80 tensor op is forward-compatible).
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 120>;
        return CutlassGroupGemmKernel<Config, ElementA, ElementB, ElementCD>::run(problem_desc, problem_count, stream);
    } else {
        return GemmStatus::INVALID_ARGUMENT;
    }
}

}  // namespace detail

//==============================================================================
// Typed Launcher: GroupGemm
//==============================================================================

/**
 * @brief Execute grouped GEMM with variable problem sizes.
 *
 * Computes: D_i = A_i @ B_i for each problem i in [0, problem_count).
 * Requires SM >= 80 (Ampere or newer).
 */
template <typename ElementA, typename ElementB, typename ElementCD>
cudaError_t GroupGemm(const ElementA* A, const ElementB* B, ElementCD* D, int problem_count, int K,
                      int N, const int* offsets_host, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || D == nullptr || offsets_host == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (problem_count <= 0 || K <= 0 || N <= 0) {
        return cudaErrorInvalidValue;
    }

    GemmStatus status = GemmStatus::SUCCESS;

#ifdef OASR_TARGET_SM
    {
        constexpr int SM_VERSION = OASR_TARGET_SM;
        status = detail::dispatchGroupGemmWithSmVersion<SM_VERSION, ElementA, ElementB, ElementCD>(problem_count, K, N, A, B, D, offsets_host, stream);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        status = detail::dispatchGroupGemmWithSmVersion<SM_VERSION, ElementA, ElementB, ElementCD>(problem_count, K, N, A, B, D, offsets_host, stream);
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace gemm
}  // namespace oasr
