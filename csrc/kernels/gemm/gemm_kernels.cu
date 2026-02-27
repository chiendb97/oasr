// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel implementations using CUTLASS
// Supports BF16 and FP16 precision

#include <cstdint>
#include "gemm_kernels.h"
#include "gemm_utils.h"
#include "common/cuda_utils.h"
#include <torch/extension.h>

// Suppress warnings from CUTLASS headers
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <stdexcept>    
#include <sstream>
#include <limits>
#include <type_traits>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Standard GEMM Implementation (SM80)
//==============================================================================

template <typename ElementA, typename ElementB, typename ElementC,
          typename LayoutA, typename LayoutB, typename LayoutC>
struct CutlassGemmSM80 {
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        float,                                          // Accumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3  // Pipeline stages
    >;
    
    static GemmStatus run(
        const ElementA* A, const ElementB* B, const ElementC* C, ElementC* D,
        int M, int N, int K,
        int64_t lda, int64_t ldb, int64_t ldc,
        float alpha, float beta,
        cudaStream_t stream)
    {
        typename Gemm::Arguments args(
            {M, N, K},
            {A, lda},
            {B, ldb},
            {C ? C : D, ldc},
            {D, ldc},
            {alpha, beta}
        );
        
        Gemm gemm_op;
        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::NOT_SUPPORTED;
        }
        
        size_t workspace_size = Gemm::get_workspace_size(args);
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
// Public API Implementations
//==============================================================================


GemmStatus invokeGemm(const torch::Tensor& A, const torch::Tensor& B,
                      const torch::Tensor& C, torch::Tensor& D,
                      cudaStream_t stream) {

    const int K = A.size(-1);
    const int M = A.numel() / K;
    const int N = B.size(0);

    const void* A_ptr = A.data_ptr();
    const void* B_ptr = B.data_ptr();
    const void* C_ptr = C.defined() ? C.data_ptr() : nullptr;
    void* D_ptr = D.data_ptr();

    if (A_ptr == nullptr || B_ptr == nullptr || D_ptr == nullptr) {
        return GemmStatus::INVALID_ARGUMENT;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return GemmStatus::INVALID_ARGUMENT;
    }

    float alpha = 1.0f;
    float beta = (C_ptr == nullptr) ? 0.0f : 1.0f;

    uint64_t lda = K;
    uint64_t ldb = K;
    uint64_t ldc = N;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    if (A.scalar_type() == torch::kHalf) {
        return CutlassGemmSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA, LayoutB, LayoutC>::run(
            reinterpret_cast<const cutlass::half_t*>(A_ptr),
            reinterpret_cast<const cutlass::half_t*>(B_ptr),
            reinterpret_cast<const cutlass::half_t*>(C_ptr),
            reinterpret_cast<cutlass::half_t*>(D_ptr), M, N, K, lda, ldb, ldc, alpha, beta, stream);
    } else if (A.scalar_type() == torch::kBFloat16) {
        return CutlassGemmSM80<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, LayoutA, LayoutB, LayoutC>::run(
            reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
            reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
            reinterpret_cast<const cutlass::bfloat16_t*>(C_ptr),
            reinterpret_cast<cutlass::bfloat16_t*>(D_ptr), M, N, K, lda, ldb, ldc, alpha, beta, stream);
    } else {
        return GemmStatus::INVALID_ARGUMENT;
    }
}

//==============================================================================
// Fused GEMM + Activation (placeholder)
//==============================================================================

template <typename ElementA, typename ElementB, typename ElementC>
GemmStatus invokeGemmBiasActivation(
    const torch::Tensor& A, const torch::Tensor& B,
    const torch::Tensor& C, torch::Tensor& D,
    ActivationType activation,
    cudaStream_t stream)
{
    (void)activation;
    return invokeGemm(A, B, C, D, stream);
}


//==============================================================================
// Auto-Tuning
//==============================================================================

GemmConfig autoTuneGemm(int M, int N, int K, DataType dtype,
                        int num_warmup, int num_iter,
                        cudaStream_t stream) {
    int sm_version = getSMVersion();
    
    std::vector<GemmConfig> configs;
    if (sm_version >= 90) {
        configs = getDefaultSM90Configs();
    } else {
        configs = getDefaultSM80Configs();
    }
    
    if (configs.empty()) {
        return GemmConfig();
    }
    
    size_t size_a = M * K * getDataTypeSize(dtype);
    size_t size_b = N * K * getDataTypeSize(dtype);
    size_t size_d = M * N * getDataTypeSize(dtype);
    
    void *A = nullptr, *B = nullptr, *D = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&A, size_a));
    OASR_CUDA_CHECK(cudaMalloc(&B, size_b));
    OASR_CUDA_CHECK(cudaMalloc(&D, size_d));

    GemmConfig best_config = configs[0];
    (void)num_warmup;
    (void)num_iter;
    (void)stream;
    
    // for (const auto& config : configs) {
    //     try {
    //         (void)config;
    //         // Warmup
    //         for (int i = 0; i < num_warmup; ++i) {
    //             invokeGemm(A, B, nullptr, D, M, N, K, dtype, stream);
    //         }
    //         OASR_CUDA_CHECK(cudaStreamSynchronize(stream));
            
    //         // Timing
    //         cudaEvent_t start, stop;
    //         OASR_CUDA_CHECK(cudaEventCreate(&start));
    //         OASR_CUDA_CHECK(cudaEventCreate(&stop));
            
    //         OASR_CUDA_CHECK(cudaEventRecord(start, stream));
    //         for (int i = 0; i < num_iter; ++i) {
    //             invokeGemm(A, B, nullptr, D, M, N, K, dtype, stream);
    //         }
    //         OASR_CUDA_CHECK(cudaEventRecord(stop, stream));
    //         OASR_CUDA_CHECK(cudaEventSynchronize(stop));
            
    //         float elapsed_ms;
    //         OASR_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    //         elapsed_ms /= num_iter;
            
    //         if (elapsed_ms < best_time) {
    //             best_time = elapsed_ms;
    //             best_config = config;
    //         }
            
    //         OASR_CUDA_CHECK(cudaEventDestroy(start));
    //         OASR_CUDA_CHECK(cudaEventDestroy(stop));
    //     }
    //     catch (...) {
    //         continue;
    //     }
    // }
    
    OASR_CUDA_CHECK(cudaFree(A));
    OASR_CUDA_CHECK(cudaFree(B));
    OASR_CUDA_CHECK(cudaFree(D));
    
    return best_config;
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
