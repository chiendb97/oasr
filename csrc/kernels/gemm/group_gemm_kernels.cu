// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Grouped GEMM kernel implementations using CUTLASS
// Supports variable-sized problems with BF16 and FP16 precision

#include "group_gemm_kernels.h"
#include "gemm_utils.h"
#include "common/cuda_utils.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <stdexcept>
#include <sstream>
#include <vector>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Grouped GEMM Implementation (SM80)
//==============================================================================

template <typename DType>
struct CutlassGroupGemmSM80 {
    using ElementA = DType;
    using ElementB = DType;
    using ElementC = DType;
    using ElementAccumulator = float;
    
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    
    static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, kAlignmentA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, kAlignmentB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4  // Stages
    >::GemmKernel;
    
    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    
    static GemmStatus run(
        void* workspace, size_t workspace_size,
        cutlass::gemm::GemmCoord* problems,
        int num_problems,
        const DType* const* A_array,
        const DType* const* B_array,
        DType* const* D_array,
        const int64_t* lda_array,
        const int64_t* ldb_array,
        const int64_t* ldd_array,
        cudaStream_t stream)
    {
        typename EpilogueOutputOp::Params epilogue_params(1.0f, 0.0f);
        
        typename GemmGrouped::Arguments args(
            reinterpret_cast<cutlass::gemm::GemmCoord*>(problems),
            num_problems,
            4,  // threadblock count
            epilogue_params,
            A_array,
            B_array,
            D_array,
            D_array,
            lda_array,
            ldb_array,
            ldd_array,
            ldd_array
        );
        
        GemmGrouped gemm;
        
        cutlass::Status status = gemm.initialize(args, workspace, stream);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::INTERNAL_ERROR;
        }
        
        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::CUTLASS_ERROR;
        }
        
        return GemmStatus::SUCCESS;
    }
};

//==============================================================================
// Public API Implementations
//==============================================================================

GemmStatus invokeGroupGemm(const GroupGemmParams& params) {
    if (params.problems == nullptr || params.num_problems <= 0) {
        return GemmStatus::INVALID_ARGUMENT;
    }
    if (params.A_array == nullptr || params.B_array == nullptr || 
        params.D_array == nullptr) {
        return GemmStatus::INVALID_ARGUMENT;
    }
    
    // Allocate device memory for problem descriptors
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problems_device(params.num_problems);
    
    // Convert GemmProblemDesc to cutlass::gemm::GemmCoord on host, then copy
    std::vector<cutlass::gemm::GemmCoord> problems_host(params.num_problems);
    for (int i = 0; i < params.num_problems; ++i) {
        problems_host[i] = cutlass::gemm::GemmCoord(
            params.problems[i].M,
            params.problems[i].N,
            params.problems[i].K
        );
    }
    
    OASR_CUDA_CHECK(cudaMemcpyAsync(
        problems_device.get(),
        problems_host.data(),
        params.num_problems * sizeof(cutlass::gemm::GemmCoord),
        cudaMemcpyHostToDevice,
        params.stream
    ));
    
    if (params.dtype_a == DataType::FP16) {
        using DType = cutlass::half_t;
        return CutlassGroupGemmSM80<DType>::run(
            params.workspace_float, params.workspace_float_size,
            problems_device.get(),
            params.num_problems,
            reinterpret_cast<const DType* const*>(params.A_array),
            reinterpret_cast<const DType* const*>(params.B_array),
            reinterpret_cast<DType* const*>(params.D_array),
            params.lda_array,
            params.ldb_array,
            params.ldd_array,
            params.stream
        );
    }
    else if (params.dtype_a == DataType::BF16) {
        using DType = cutlass::bfloat16_t;
        return CutlassGroupGemmSM80<DType>::run(
            params.workspace_float, params.workspace_float_size,
            problems_device.get(),
            params.num_problems,
            reinterpret_cast<const DType* const*>(params.A_array),
            reinterpret_cast<const DType* const*>(params.B_array),
            reinterpret_cast<DType* const*>(params.D_array),
            params.lda_array,
            params.ldb_array,
            params.ldd_array,
            params.stream
        );
    }
    
    return GemmStatus::NOT_SUPPORTED;
}

template <>
GemmStatus invokeGroupGemmTyped<half, half>(
    void* workspace_float, size_t workspace_float_size,
    void* workspace_int, size_t workspace_int_size,
    const GemmProblemDesc* problems, int num_problems,
    const half* const* A_array, const half* const* B_array,
    half* const* D_array,
    const int64_t* lda_array, const int64_t* ldb_array, const int64_t* ldd_array,
    bool weight_column_major,
    cudaStream_t stream)
{
    (void)workspace_int;
    (void)workspace_int_size;
    (void)weight_column_major;
    
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problems_device(num_problems);
    std::vector<cutlass::gemm::GemmCoord> problems_host(num_problems);
    for (int i = 0; i < num_problems; ++i) {
        problems_host[i] = cutlass::gemm::GemmCoord(
            problems[i].M, problems[i].N, problems[i].K
        );
    }
    
    OASR_CUDA_CHECK(cudaMemcpyAsync(
        problems_device.get(),
        problems_host.data(),
        num_problems * sizeof(cutlass::gemm::GemmCoord),
        cudaMemcpyHostToDevice,
        stream
    ));
    
    using DType = cutlass::half_t;
    return CutlassGroupGemmSM80<DType>::run(
        workspace_float, workspace_float_size,
        problems_device.get(),
        num_problems,
        reinterpret_cast<const DType* const*>(A_array),
        reinterpret_cast<const DType* const*>(B_array),
        reinterpret_cast<DType* const*>(D_array),
        lda_array, ldb_array, ldd_array,
        stream
    );
}

template <>
GemmStatus invokeGroupGemmTyped<__nv_bfloat16, __nv_bfloat16>(
    void* workspace_float, size_t workspace_float_size,
    void* workspace_int, size_t workspace_int_size,
    const GemmProblemDesc* problems, int num_problems,
    const __nv_bfloat16* const* A_array, const __nv_bfloat16* const* B_array,
    __nv_bfloat16* const* D_array,
    const int64_t* lda_array, const int64_t* ldb_array, const int64_t* ldd_array,
    bool weight_column_major,
    cudaStream_t stream)
{
    (void)workspace_int;
    (void)workspace_int_size;
    (void)weight_column_major;
    
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problems_device(num_problems);
    std::vector<cutlass::gemm::GemmCoord> problems_host(num_problems);
    for (int i = 0; i < num_problems; ++i) {
        problems_host[i] = cutlass::gemm::GemmCoord(
            problems[i].M, problems[i].N, problems[i].K
        );
    }
    
    OASR_CUDA_CHECK(cudaMemcpyAsync(
        problems_device.get(),
        problems_host.data(),
        num_problems * sizeof(cutlass::gemm::GemmCoord),
        cudaMemcpyHostToDevice,
        stream
    ));
    
    using DType = cutlass::bfloat16_t;
    return CutlassGroupGemmSM80<DType>::run(
        workspace_float, workspace_float_size,
        problems_device.get(),
        num_problems,
        reinterpret_cast<const DType* const*>(A_array),
        reinterpret_cast<const DType* const*>(B_array),
        reinterpret_cast<DType* const*>(D_array),
        lda_array, ldb_array, ldd_array,
        stream
    );
}

//==============================================================================
// Workspace Size Query
//==============================================================================

void queryGroupGemmWorkspaceSize(
    int num_problems,
    const GemmProblemDesc* problems,
    DataType dtype,
    size_t& float_workspace_size,
    size_t& int_workspace_size,
    const GemmConfig& config)
{
    (void)num_problems; (void)problems; (void)dtype; (void)config;
    float_workspace_size = 16 * 1024 * 1024;  // 16MB
    int_workspace_size = 1 * 1024 * 1024;     // 1MB
}

//==============================================================================
// Grouped GEMM Runner Implementation
//==============================================================================

struct GroupGemmRunner::Impl {
    DataType dtype;
    int sm_version;
    
    Impl(DataType dt, int sm) : dtype(dt), sm_version(sm) {
        if (sm_version < 0) {
            sm_version = getSMVersion();
        }
    }
};

GroupGemmRunner::GroupGemmRunner(DataType dtype, int sm_version)
    : impl_(std::make_unique<Impl>(dtype, sm_version)) {}

GroupGemmRunner::~GroupGemmRunner() = default;

void GroupGemmRunner::getWorkspaceSize(int num_problems,
                                       size_t& float_workspace_size,
                                       size_t& int_workspace_size) const {
    queryGroupGemmWorkspaceSize(num_problems, nullptr, impl_->dtype,
                                float_workspace_size, int_workspace_size);
}

GemmStatus GroupGemmRunner::run(const GroupGemmParams& params) {
    return invokeGroupGemm(params);
}

GemmStatus GroupGemmRunner::run(
    const GemmProblemDesc* problems, int num_problems,
    const void* const* A_array, const void* const* B_array,
    void* const* D_array,
    const int64_t* lda_array, const int64_t* ldb_array, const int64_t* ldd_array,
    void* workspace_float, size_t workspace_float_size,
    void* workspace_int, size_t workspace_int_size,
    bool weight_column_major,
    cudaStream_t stream)
{
    GroupGemmParams params;
    params.problems = problems;
    params.num_problems = num_problems;
    params.A_array = A_array;
    params.B_array = B_array;
    params.D_array = D_array;
    params.lda_array = lda_array;
    params.ldb_array = ldb_array;
    params.ldd_array = ldd_array;
    params.weight_column_major = weight_column_major;
    params.dtype_a = impl_->dtype;
    params.dtype_b = impl_->dtype;
    params.dtype_d = impl_->dtype;
    params.workspace_float = workspace_float;
    params.workspace_float_size = workspace_float_size;
    params.workspace_int = workspace_int;
    params.workspace_int_size = workspace_int_size;
    params.stream = stream;
    
    return invokeGroupGemm(params);
}

std::vector<GemmConfig> GroupGemmRunner::getConfigs() const {
    if (impl_->sm_version >= 90) {
        return getDefaultSM90Configs();
    }
    return getDefaultSM80Configs();
}

//==============================================================================
// Segment GEMM Implementation
//==============================================================================

GemmStatus invokeSegmentGemm(const SegmentGemmParams& params) {
    (void)params;
    // TODO: Implement segment GEMM using grouped GEMM internally
    return GemmStatus::NOT_SUPPORTED;
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
