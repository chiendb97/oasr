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
        
        // CUTLASS Arguments expects non-const pointer types (Element* **); kernel only reads A/B/ld*
        typename GemmGrouped::Arguments args(
            problems,
            num_problems,
            4,  // threadblock count
            epilogue_params,
            const_cast<DType**>(A_array),
            const_cast<DType**>(B_array),
            const_cast<DType**>(D_array),
            const_cast<DType**>(D_array),
            const_cast<int64_t*>(lda_array),
            const_cast<int64_t*>(ldb_array),
            const_cast<int64_t*>(ldd_array),
            const_cast<int64_t*>(ldd_array),
            nullptr  // host_problem_sizes (optional)
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

GemmStatus invokeGroupGemm(const GemmProblemDesc* problems, int num_problems,
                           const void* const* A_array, const void* const* B_array,
                           void* const* D_array,
                           const int64_t* lda_array, const int64_t* ldb_array,
                           const int64_t* ldd_array,
                           DataType dtype,
                           void* workspace_float, size_t workspace_float_size,
                           cudaStream_t stream) {
    if (problems == nullptr || num_problems <= 0) return GemmStatus::INVALID_ARGUMENT;
    if (A_array == nullptr || B_array == nullptr || D_array == nullptr)
        return GemmStatus::INVALID_ARGUMENT;

    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problems_device(num_problems);

    // problems may be on device; copy to host first
    std::vector<GemmProblemDesc> problems_desc_host(num_problems);
    OASR_CUDA_CHECK(cudaMemcpyAsync(
        problems_desc_host.data(),
        problems,
        num_problems * sizeof(GemmProblemDesc),
        cudaMemcpyDeviceToHost,
        stream
    ));
    OASR_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<cutlass::gemm::GemmCoord> problems_host(num_problems);
    for (int i = 0; i < num_problems; ++i) {
        problems_host[i] = cutlass::gemm::GemmCoord(
            problems_desc_host[i].M,
            problems_desc_host[i].N,
            problems_desc_host[i].K
        );
    }

    OASR_CUDA_CHECK(cudaMemcpyAsync(
        problems_device.get(),
        problems_host.data(),
        num_problems * sizeof(cutlass::gemm::GemmCoord),
        cudaMemcpyHostToDevice,
        stream
    ));

    if (dtype == DataType::FP16) {
        using DType = cutlass::half_t;
        return CutlassGroupGemmSM80<DType>::run(
            workspace_float, workspace_float_size,
            problems_device.get(), num_problems,
            reinterpret_cast<const DType* const*>(A_array),
            reinterpret_cast<const DType* const*>(B_array),
            reinterpret_cast<DType* const*>(D_array),
            lda_array, ldb_array, ldd_array,
            stream
        );
    }
    if (dtype == DataType::BF16) {
        using DType = cutlass::bfloat16_t;
        return CutlassGroupGemmSM80<DType>::run(
            workspace_float, workspace_float_size,
            problems_device.get(), num_problems,
            reinterpret_cast<const DType* const*>(A_array),
            reinterpret_cast<const DType* const*>(B_array),
            reinterpret_cast<DType* const*>(D_array),
            lda_array, ldb_array, ldd_array,
            stream
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

GemmStatus GroupGemmRunner::run(const GemmProblemDesc* problems, int num_problems,
                                const void* const* A_array, const void* const* B_array,
                                void* const* D_array,
                                const int64_t* lda_array, const int64_t* ldb_array,
                                const int64_t* ldd_array,
                                void* workspace_float, size_t workspace_float_size,
                                void* workspace_int, size_t workspace_int_size,
                                bool weight_column_major,
                                cudaStream_t stream)
{
    (void)workspace_int;
    (void)workspace_int_size;
    (void)weight_column_major;
    return invokeGroupGemm(problems, num_problems, A_array, B_array, D_array,
                          lda_array, ldb_array, ldd_array, impl_->dtype,
                          workspace_float, workspace_float_size, stream);
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

GemmStatus invokeSegmentGemm(const void* A, const void* B, void* D,
                             const int* segment_offsets, int num_segments,
                             int K, int N, DataType dtype,
                             bool weight_column_major,
                             void* workspace, size_t workspace_size,
                             cudaStream_t stream) {
    (void)A; (void)B; (void)D; (void)segment_offsets; (void)num_segments;
    (void)K; (void)N; (void)dtype; (void)weight_column_major;
    (void)workspace; (void)workspace_size; (void)stream;
    return GemmStatus::NOT_SUPPORTED;  // TODO: Implement using grouped GEMM
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
