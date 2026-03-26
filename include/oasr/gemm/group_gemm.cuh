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
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_utils.h>

namespace oasr {
namespace gemm {

//==============================================================================
// Grouped GEMM Problem Descriptor
//==============================================================================

template <typename ElementA, typename ElementB, typename ElementCD>
struct GroupedGemmProblemDesc {
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problems_sizes_device;

    cutlass::DeviceAllocation<ElementA*> ptr_A_device;
    cutlass::DeviceAllocation<ElementB*> ptr_B_device;
    cutlass::DeviceAllocation<ElementCD*> ptr_D_device;

    cutlass::DeviceAllocation<int64_t> lda_device;
    cutlass::DeviceAllocation<int64_t> ldb_device;
    cutlass::DeviceAllocation<int64_t> ldd_device;

    GroupedGemmProblemDesc(int problem_count, int K, int N, const ElementA* A_ptr,
                           const ElementB* B_ptr, ElementCD* D_ptr, const int* offsets_host)
        : problem_sizes(problem_count),
          problems_sizes_device(problem_count),
          ptr_A_device(problem_count),
          ptr_B_device(problem_count),
          ptr_D_device(problem_count),
          lda_device(problem_count),
          ldb_device(problem_count),
          ldd_device(problem_count) {
        std::vector<ElementA*> ptr_A(problem_count);
        std::vector<ElementB*> ptr_B(problem_count);
        std::vector<ElementCD*> ptr_D(problem_count);
        std::vector<int64_t> lda(problem_count);
        std::vector<int64_t> ldb(problem_count);
        std::vector<int64_t> ldd(problem_count);

        int offset_M = 0;
        for (int i = 0; i < problem_count; ++i) {
            int next_offset_M = offsets_host[i];
            int M = next_offset_M - offset_M;
            problem_sizes[i] = cutlass::gemm::GemmCoord(M, N, K);
            lda[i] = K;
            ldb[i] = K;
            ldd[i] = N;

            ptr_A[i] = const_cast<ElementA*>(A_ptr) + static_cast<int64_t>(offset_M) * K;
            ptr_B[i] = const_cast<ElementB*>(B_ptr) + static_cast<int64_t>(i) * N * K;
            ptr_D[i] = D_ptr + static_cast<int64_t>(offset_M) * N;

            offset_M = next_offset_M;
        }

        problems_sizes_device.copy_from_host(problem_sizes.data());
        ptr_A_device.copy_from_host(ptr_A.data());
        ptr_B_device.copy_from_host(ptr_B.data());
        ptr_D_device.copy_from_host(ptr_D.data());
        lda_device.copy_from_host(lda.data());
        ldb_device.copy_from_host(ldb.data());
        ldd_device.copy_from_host(ldd.data());
    }
};

//==============================================================================
// CUTLASS Grouped GEMM Template
//==============================================================================

template <typename Config, typename MMATraits, typename ElementA, typename ElementB,
          typename ElementCD, typename LayoutA, typename LayoutB, typename LayoutCD>
struct CutlassGroupGemmKernel {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int kAlignmentCD = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = typename MMATraits::MMAOp;
    using SmArch = typename MMATraits::SmArch;

    using ShapeMMAThreadBlock = typename Config::ThreadBlock;
    using ShapeMMAWarps = typename Config::Warps;
    using ShapeMMAOp = typename MMATraits::MMAShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementCD, kAlignmentCD, ElementAccumulator,
                                                     ElementAccumulator>;

    static constexpr int NumStages = Config::NumStages;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, kAlignmentA, ElementB, LayoutB,
        cutlass::ComplexTransform::kNone, kAlignmentB, ElementCD, LayoutCD, ElementAccumulator,
        MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp, EpilogueOp,
        SwizzleThreadblock, NumStages>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    static GemmStatus run(GroupedGemmProblemDesc<ElementA, ElementB, ElementCD>& problem_desc,
                          int problem_count, cudaStream_t stream) {
        typename EpilogueOp::Params epilogue_params(1.0f, 0.0f);

        int threadblock_count = Gemm::sufficient(problem_desc.problem_sizes.data(), problem_count);
        typename Gemm::Arguments args(
            problem_desc.problems_sizes_device.get(), problem_count, threadblock_count,
            epilogue_params, problem_desc.ptr_A_device.get(), problem_desc.ptr_B_device.get(),
            problem_desc.ptr_D_device.get(), problem_desc.ptr_D_device.get(),
            problem_desc.lda_device.get(), problem_desc.ldb_device.get(),
            problem_desc.ldd_device.get(), problem_desc.ldd_device.get());

        Gemm gemm_op;
        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::NOT_SUPPORTED;
        }

        size_t workspace_size = gemm_op.get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        status = gemm_op.initialize(args, workspace.get(), stream);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::INTERNAL_ERROR;
        }

        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::CUTLASS_ERROR;
        }

        return GemmStatus::SUCCESS;
    }
};

//==============================================================================
// Dispatch helpers
//==============================================================================

namespace detail {

template <typename Config, typename MMATraits, typename ElementA, typename ElementB,
          typename ElementCD>
static GemmStatus dispatchGroupGemmWithConfig(int problem_count, int K, int N,
                                               const ElementA* A_ptr, const ElementB* B_ptr,
                                               ElementCD* D_ptr, const int* offsets_host,
                                               cudaStream_t stream) {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    GroupedGemmProblemDesc<ElementA, ElementB, ElementCD> problem_desc(problem_count, K, N, A_ptr,
                                                                       B_ptr, D_ptr, offsets_host);
    return CutlassGroupGemmKernel<Config, MMATraits, ElementA, ElementB, ElementCD, LayoutA,
                                   LayoutB, LayoutCD>::run(problem_desc, problem_count, stream);
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
        static_assert(SM_VERSION >= 80, "Grouped GEMM requires SM80 or newer");
#ifdef OASR_GEMM_TILE_M
        using Config = JitGemmConfig;
#else
        using Config = typename DefaultGemmConfig<SM_VERSION>::type;
#endif
        using MMA = SmMMATraits<SM_VERSION>;
        status = detail::dispatchGroupGemmWithConfig<Config, MMA>(problem_count, K, N, A, B, D,
                                                                   offsets_host, stream);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        if constexpr (SM_VERSION < 80) {
            throw std::runtime_error("Grouped GEMM requires SM80 or newer (current: SM" +
                                     std::to_string(SM_VERSION) + ")");
        } else {
            using Config = typename DefaultGemmConfig<SM_VERSION>::type;
            using MMA = SmMMATraits<SM_VERSION>;
            status = detail::dispatchGroupGemmWithConfig<Config, MMA>(problem_count, K, N, A, B, D,
                                                                       offsets_host, stream);
        }
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace gemm
}  // namespace oasr
