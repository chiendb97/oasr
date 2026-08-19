// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS GEMM kernel template — parameterized by GemmConfig + SmMMATraits.
//
// This is the core CUTLASS 2.x GEMM implementation. Config provides tile
// dimensions and pipeline stages; MMATraits provides hardware-specific MMA
// shape, op class, and SM architecture tag.

#pragma once

#include <cstdint>
#include <type_traits>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/conversion_op.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/reduction/thread/reduction_operators.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/epilogue_functors.h>
#include <oasr/common/graph_safe_workspace.h>
#include <oasr/common/utils.h>
#include <oasr/common/workspace_cache.h>
#include <oasr/gemm/cutlass_gemm_configs.h>

namespace oasr {
namespace gemm {

//==============================================================================
// CutlassGemmKernel — CUTLASS 2.x GEMM template
//==============================================================================

// Stream-K distributes thin reductions across multiprocessors. Parallel split-K
// stores fp32 partials and applies fused epilogues after reduction. Serial
// split-K reuses pre-zeroed semaphores and restores them in-kernel, avoiding a
// per-launch workspace clear.
template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD,
          ActivationType activation_type, bool kStreamK = false, bool kParallelSplitK = false>
struct CutlassGemmKernel {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = typename CutlassGemmConfig::SmArch;

    using ThreadblockShape = typename CutlassGemmConfig::ThreadblockShape;
    using WarpShape = typename CutlassGemmConfig::WarpShape;
    using InstructionShape = typename CutlassGemmConfig::InstructionShape;

    using EpilogueOp = typename FusionEpilogueOp<activation_type, AlignmentEpilogue, ElementCD,
                                                 ElementComputeEpilogue, ElementCD>::type;

    static constexpr int Stages = CutlassGemmConfig::Stages;

    // Data-parallel (identity swizzle) classic device GEMM.  SplitKSerial=true
    // enables the runtime ``split_k_slices`` argument (semaphore-serialized
    // K-partition reduction); with split_k_slices == 1 the kernel is identical
    // to the non-split-K instantiation.
    using GemmBasic =
        cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementCD, LayoutCD,
                                    ElementAccumulator, MMAOp, SmArch, ThreadblockShape, WarpShape,
                                    InstructionShape, EpilogueOp,
                                    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                    Stages, AlignmentA, AlignmentB, /*SplitKSerial=*/true>;

    // Stream-K (GemmUniversal): same template parameters, only the swizzle differs.
    using GemmStreamK = cutlass::gemm::device::GemmUniversal<
        ElementA, LayoutA, ElementB, LayoutB, ElementCD, LayoutCD, ElementAccumulator, MMAOp,
        SmArch, ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
        cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, Stages, AlignmentA, AlignmentB>;

    using Gemm = std::conditional_t<kStreamK, GemmStreamK, GemmBasic>;

    static GemmStatus run(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                          int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
                          ElementComputeEpilogue alpha, cudaStream_t stream,
                          int split_k_slices = 1) {
        float beta = (C == nullptr) ? 0.0f : 1.0f;

        if constexpr (kParallelSplitK) {
            return runParallelSplitK(A, B, C, D, M, N, K, lda, ldb, ldc, alpha, beta, stream,
                                     split_k_slices);
        } else if constexpr (kStreamK) {
            // GemmUniversal kGemm mode; the bias vector C[N] broadcasts over
            // rows via stride_c = 0.  avail_sms = -1 → use all device SMs for
            // Stream-K load balancing.
            typename Gemm::Arguments args(
                cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
                /*batch_count=*/1, {alpha, beta}, A, B, C, D,
                /*batch_stride_A=*/static_cast<int64_t>(M) * K,
                /*batch_stride_B=*/static_cast<int64_t>(N) * K,
                /*batch_stride_C=*/static_cast<int64_t>(0),
                /*batch_stride_D=*/static_cast<int64_t>(M) * N,
                /*stride_a=*/lda, /*stride_b=*/ldb, /*stride_c=*/static_cast<int64_t>(0),
                /*stride_d=*/ldc, /*avail_sms=*/-1);

            Gemm gemm_op;
            if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
                return GemmStatus::NOT_SUPPORTED;
            }

            // The Stream-K barrier workspace must be re-zeroed every launch
            // (initialize() does that), but the allocation itself is reusable —
            // prefer the persistent scratch buffer over per-call alloc/free.
            size_t workspace_size = Gemm::get_workspace_size(args);
            void* cached = getCachedWorkspace(workspace_size, stream, WorkspacePool::kScratch);
            if (cached != nullptr) {
                return runInitialized(gemm_op, args, cached, stream);
            }
            oasr::GraphSafeWorkspace workspace(workspace_size, stream);
            return runInitialized(gemm_op, args, workspace.get(), stream);
        } else {
            if (activation_type != ActivationType::IDENTITY && split_k_slices > 1) {
                // Serial split-K applies the epilogue per K-partition, which
                // would nest the activation around partial sums — wrong math.
                // (Parallel split-K variants handle fused activations.)
                return GemmStatus::NOT_SUPPORTED;
            }

            typename Gemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, 0}, {D, ldc},
                                          {alpha, beta}, split_k_slices);
            Gemm gemm_op;
            if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
                return GemmStatus::NOT_SUPPORTED;
            }

            if (split_k_slices > 1) {
                size_t workspace_size = Gemm::get_workspace_size(args);
                void* semaphore =
                    getCachedWorkspace(workspace_size, stream, WorkspacePool::kZeroedSemaphore);
                if (semaphore != nullptr) {
                    // Single-launch fast path: semaphores are pre-zeroed and
                    // self-restoring, so skip initialize()'s cudaMemsetAsync.
                    return runSerialSplitKPreZeroed(args, static_cast<int*>(semaphore), stream);
                }
                // Fallback (cache disabled, OOM, or first touch during CUDA
                // graph capture): legacy per-call workspace + memset.
                oasr::GraphSafeWorkspace workspace(workspace_size, stream);
                return runInitialized(gemm_op, args, workspace.get(), stream);
            }

            return runInitialized(gemm_op, args, nullptr, stream);
        }
    }

private:
    /// initialize() + run() with the given workspace (legacy CUTLASS flow).
    static GemmStatus runInitialized(Gemm& gemm_op, typename Gemm::Arguments const& args,
                                     void* workspace, cudaStream_t stream) {
        if (gemm_op.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
            return GemmStatus::INTERNAL_ERROR;
        }
        if (gemm_op(stream) != cutlass::Status::kSuccess) {
            return GemmStatus::CUTLASS_ERROR;
        }
        return GemmStatus::SUCCESS;
    }

    /// Serial split-K without the per-launch semaphore memset.  Mirrors
    /// ``device::Gemm::initialize`` + ``run`` (params construction, smem
    /// opt-in, kernel launch) minus the ``cudaMemsetAsync`` — valid because
    /// *semaphore* points at pre-zeroed memory and the kernel's final K-slice
    /// releases each tile's lock back to 0 (``cutlass/gemm/kernel/gemm.h``).
    static GemmStatus runSerialSplitKPreZeroed(typename GemmBasic::Arguments const& args,
                                               int* semaphore, cudaStream_t stream) {
        using GemmKernelT = typename GemmBasic::GemmKernel;

        typename GemmBasic::ThreadblockSwizzle threadblock_swizzle;
        cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
            args.problem_size, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            args.split_k_slices);

        typename GemmKernelT::Params params{args.problem_size,
                                            grid_shape,
                                            args.ref_A.non_const_ref(),
                                            args.ref_B.non_const_ref(),
                                            args.ref_C.non_const_ref(),
                                            args.ref_D,
                                            args.epilogue,
                                            semaphore,
                                            args.gather_A_indices,
                                            args.gather_B_indices,
                                            args.scatter_D_indices};

        dim3 grid = threadblock_swizzle.get_grid_shape(grid_shape);
        dim3 block(GemmKernelT::kThreadCount, 1, 1);
        int smem_size = static_cast<int>(sizeof(typename GemmKernelT::SharedStorage));
        if (smem_size >= (48 << 10)) {
            if (cudaFuncSetAttribute(cutlass::Kernel<GemmKernelT>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     smem_size) != cudaSuccess) {
                return GemmStatus::INTERNAL_ERROR;
            }
        }

        cutlass::Kernel<GemmKernelT><<<grid, block, smem_size, stream>>>(params);
        return (cudaGetLastError() == cudaSuccess) ? GemmStatus::SUCCESS
                                                   : GemmStatus::CUTLASS_ERROR;
    }

    /// Parallel split-K: fp32 partials + reduction kernel applying the
    /// (possibly activation-fused) epilogue exactly once.  Workspace needs no
    /// zeroing (partials are fully overwritten before the reduction reads).
    static GemmStatus runParallelSplitK(const ElementA* A, const ElementB* B, const ElementCD* C,
                                        ElementCD* D, int M, int N, int K, int64_t lda,
                                        int64_t ldb, int64_t ldc, float alpha, float beta,
                                        cudaStream_t stream, int split_k_slices) {
        if (split_k_slices <= 1) {
            return GemmStatus::NOT_SUPPORTED;  // only registered with slices > 1
        }

        using ConvertScaledOp =
            cutlass::epilogue::thread::Convert<ElementAccumulator, EpilogueOp::kCount,
                                               ElementAccumulator>;
        using ReductionOp = cutlass::reduction::thread::ReduceAdd<
            ElementAccumulator, typename EpilogueOp::ElementAccumulator, EpilogueOp::kCount>;
        using GemmPk = cutlass::gemm::device::GemmSplitKParallel<
            ElementA, LayoutA, ElementB, LayoutB, ElementCD, LayoutCD, ElementAccumulator, MMAOp,
            SmArch, ThreadblockShape, WarpShape, InstructionShape, EpilogueOp, ConvertScaledOp,
            ReductionOp, cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
            Stages, AlignmentA, AlignmentB>;

        typename GemmPk::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, 0}, {D, ldc},
                                        {alpha, beta}, split_k_slices);
        GemmPk gemm_op;
        if (GemmPk::can_implement(args) != cutlass::Status::kSuccess) {
            return GemmStatus::NOT_SUPPORTED;
        }

        size_t workspace_size = GemmPk::get_workspace_size(args);
        void* workspace = getCachedWorkspace(workspace_size, stream, WorkspacePool::kScratch);
        oasr::GraphSafeWorkspace fallback(workspace == nullptr ? workspace_size : 0, stream);
        if (workspace == nullptr) {
            workspace = fallback.get();
        }

        if (gemm_op.initialize(args, workspace) != cutlass::Status::kSuccess) {
            return GemmStatus::INTERNAL_ERROR;
        }
        if (gemm_op.run(stream) != cutlass::Status::kSuccess) {
            return GemmStatus::CUTLASS_ERROR;
        }
        return GemmStatus::SUCCESS;
    }
};

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD>
struct CutlassBmmKernel {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = typename CutlassGemmConfig::SmArch;

    using ThreadblockShape = typename CutlassGemmConfig::ThreadblockShape;
    using WarpShape = typename CutlassGemmConfig::WarpShape;
    using InstructionShape = typename CutlassGemmConfig::InstructionShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementCD, AlignmentEpilogue,
                                                     ElementComputeEpilogue, ElementComputeEpilogue,
                                                     cutlass::epilogue::thread::ScaleType::Default>;

    static constexpr int NumStages = CutlassGemmConfig::Stages;

    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementA, LayoutA, ElementB, LayoutB, ElementCD, LayoutCD, ElementAccumulator, MMAOp,
        SmArch, ThreadblockShape, WarpShape, InstructionShape, EpilogueOp, SwizzleThreadblock,
        NumStages, AlignmentA, AlignmentB>;

    static GemmStatus run(const ElementA* A, const ElementB* B, ElementCD* D, int batch_size, int M,
                          int N, int K, int64_t lda, int64_t ldb, int64_t ldd, int64_t stride_a,
                          int64_t stride_b, int64_t stride_d, float alpha, float beta,
                          cudaStream_t stream) {
        typename Gemm::Arguments args({M, N, K}, {A, lda}, stride_a, {B, ldb}, stride_b, {D, ldd},
                                      stride_d, {D, ldd}, stride_d, {alpha, beta}, batch_size);

        Gemm gemm_op;
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess)
            return GemmStatus::NOT_SUPPORTED;

        size_t ws_size = Gemm::get_workspace_size(args);
        oasr::GraphSafeWorkspace ws(ws_size, stream);

        if (gemm_op.initialize(args, ws.get(), stream) != cutlass::Status::kSuccess)
            return GemmStatus::INTERNAL_ERROR;

        return (gemm_op(stream) == cutlass::Status::kSuccess) ? GemmStatus::SUCCESS
                                                              : GemmStatus::CUTLASS_ERROR;
    }
};

//==============================================================================
// CUTLASS Grouped GEMM Template
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

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD>
struct CutlassGroupGemmKernel {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = typename CutlassGemmConfig::SmArch;

    using ThreadblockShape = typename CutlassGemmConfig::ThreadblockShape;
    using WarpShape = typename CutlassGemmConfig::WarpShape;
    using InstructionShape = typename CutlassGemmConfig::InstructionShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementCD, AlignmentEpilogue,
                                                     ElementAccumulator, ElementAccumulator>;

    static constexpr int NumStages = CutlassGemmConfig::Stages;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA, ElementB, LayoutB,
        cutlass::ComplexTransform::kNone, AlignmentB, ElementCD, LayoutCD, ElementAccumulator,
        MMAOp, SmArch, ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
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
        oasr::GraphSafeWorkspace workspace(workspace_size, stream);

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

}  // namespace gemm
}  // namespace oasr
