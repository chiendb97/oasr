// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel implementations using CUTLASS
// Supports BF16 and FP16 precision

#include "gemm_kernels.h"
#include "gemm_utils.h"
#include "common/cuda_utils.h"

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
        const ElementA* A, const ElementB* B, ElementC* D,
        int M, int N, int K,
        int64_t lda, int64_t ldb, int64_t ldd,
        float alpha, float beta,
        cudaStream_t stream)
    {
        typename Gemm::Arguments args(
            {M, N, K},
            {A, lda},
            {B, ldb},
            {D, ldd},
            {D, ldd},
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

GemmStatus invokeGemm(const GemmParams& params) {
    if (params.A == nullptr || params.B == nullptr || params.D == nullptr) {
        return GemmStatus::INVALID_ARGUMENT;
    }
    if (params.M <= 0 || params.N <= 0 || params.K <= 0) {
        return GemmStatus::INVALID_ARGUMENT;
    }
    
    if (params.dtype_a == DataType::FP16 && params.dtype_b == DataType::FP16) {
        using DType = cutlass::half_t;
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::RowMajor;
        using LayoutC = cutlass::layout::RowMajor;
        
        return CutlassGemmSM80<DType, DType, DType, LayoutA, LayoutB, LayoutC>::run(
            reinterpret_cast<const DType*>(params.A),
            reinterpret_cast<const DType*>(params.B),
            reinterpret_cast<DType*>(params.D),
            params.M, params.N, params.K,
            params.lda, params.ldb, params.ldd,
            params.alpha, params.beta,
            params.stream
        );
    }
    else if (params.dtype_a == DataType::BF16 && params.dtype_b == DataType::BF16) {
        using DType = cutlass::bfloat16_t;
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::RowMajor;
        using LayoutC = cutlass::layout::RowMajor;
        
        return CutlassGemmSM80<DType, DType, DType, LayoutA, LayoutB, LayoutC>::run(
            reinterpret_cast<const DType*>(params.A),
            reinterpret_cast<const DType*>(params.B),
            reinterpret_cast<DType*>(params.D),
            params.M, params.N, params.K,
            params.lda, params.ldb, params.ldd,
            params.alpha, params.beta,
            params.stream
        );
    }
    
    return GemmStatus::NOT_SUPPORTED;
}

template <>
GemmStatus invokeGemmTyped<half>(const GemmParams& params) {
    using DType = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    
    return CutlassGemmSM80<DType, DType, DType, LayoutA, LayoutB, LayoutC>::run(
        reinterpret_cast<const DType*>(params.A),
        reinterpret_cast<const DType*>(params.B),
        reinterpret_cast<DType*>(params.D),
        params.M, params.N, params.K,
        params.lda, params.ldb, params.ldd,
        params.alpha, params.beta,
        params.stream
    );
}

template <>
GemmStatus invokeGemmTyped<__nv_bfloat16>(const GemmParams& params) {
    using DType = cutlass::bfloat16_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    
    return CutlassGemmSM80<DType, DType, DType, LayoutA, LayoutB, LayoutC>::run(
        reinterpret_cast<const DType*>(params.A),
        reinterpret_cast<const DType*>(params.B),
        reinterpret_cast<DType*>(params.D),
        params.M, params.N, params.K,
        params.lda, params.ldb, params.ldd,
        params.alpha, params.beta,
        params.stream
    );
}

//==============================================================================
// Fused GEMM + Activation (placeholder)
//==============================================================================

GemmStatus invokeGemmBiasActivation(
    const void* A, const void* B, const void* bias, void* D,
    int M, int N, int K,
    ActivationType activation,
    DataType dtype,
    cudaStream_t stream)
{
    (void)bias;
    (void)activation;
    
    GemmParams params;
    params.A = A;
    params.B = B;
    params.D = D;
    params.M = M;
    params.N = N;
    params.K = K;
    params.lda = K;
    params.ldb = N;
    params.ldd = N;
    params.dtype_a = dtype;
    params.dtype_b = dtype;
    params.dtype_d = dtype;
    params.stream = stream;
    
    // TODO: Add epilogue fusion for activation
    return invokeGemm(params);
}

//==============================================================================
// Workspace Size Query
//==============================================================================

size_t queryGemmWorkspaceSize(int M, int N, int K, DataType dtype,
                              const GemmConfig& config) {
    (void)M; (void)N; (void)K; (void)dtype; (void)config;
    return 4 * 1024 * 1024;  // 4MB
}

//==============================================================================
// GEMM Runner Implementations
//==============================================================================

struct Fp16GemmRunner::Impl {
    int sm_version;
    Impl(int sm) : sm_version(sm) {}
};

Fp16GemmRunner::Fp16GemmRunner(int sm_version)
    : impl_(std::make_unique<Impl>(sm_version)) {}

Fp16GemmRunner::~Fp16GemmRunner() = default;

void Fp16GemmRunner::gemm(const void* A, const void* B, void* D,
                          int M, int N, int K,
                          const GemmConfig& config,
                          void* workspace, size_t workspace_size,
                          cudaStream_t stream) {
    (void)config; (void)workspace; (void)workspace_size;
    
    GemmParams params;
    params.A = A;
    params.B = B;
    params.D = D;
    params.M = M;
    params.N = N;
    params.K = K;
    params.lda = K;
    params.ldb = N;
    params.ldd = N;
    params.dtype_a = DataType::FP16;
    params.dtype_b = DataType::FP16;
    params.dtype_d = DataType::FP16;
    params.stream = stream;
    
    GemmStatus status = invokeGemmTyped<half>(params);
    if (status != GemmStatus::SUCCESS) {
        throw std::runtime_error(std::string("GEMM failed: ") + getGemmStatusString(status));
    }
}

size_t Fp16GemmRunner::getWorkspaceSize(int M, int N, int K) {
    return queryGemmWorkspaceSize(M, N, K, DataType::FP16);
}

std::vector<GemmConfig> Fp16GemmRunner::getConfigs() const {
    if (impl_->sm_version >= 90) {
        return getDefaultSM90Configs();
    }
    return getDefaultSM80Configs();
}

struct Bf16GemmRunner::Impl {
    int sm_version;
    Impl(int sm) : sm_version(sm) {}
};

Bf16GemmRunner::Bf16GemmRunner(int sm_version)
    : impl_(std::make_unique<Impl>(sm_version)) {}

Bf16GemmRunner::~Bf16GemmRunner() = default;

void Bf16GemmRunner::gemm(const void* A, const void* B, void* D,
                          int M, int N, int K,
                          const GemmConfig& config,
                          void* workspace, size_t workspace_size,
                          cudaStream_t stream) {
    (void)config; (void)workspace; (void)workspace_size;
    
    GemmParams params;
    params.A = A;
    params.B = B;
    params.D = D;
    params.M = M;
    params.N = N;
    params.K = K;
    params.lda = K;
    params.ldb = N;
    params.ldd = N;
    params.dtype_a = DataType::BF16;
    params.dtype_b = DataType::BF16;
    params.dtype_d = DataType::BF16;
    params.stream = stream;
    
    GemmStatus status = invokeGemmTyped<__nv_bfloat16>(params);
    if (status != GemmStatus::SUCCESS) {
        throw std::runtime_error(std::string("GEMM failed: ") + getGemmStatusString(status));
    }
}

size_t Bf16GemmRunner::getWorkspaceSize(int M, int N, int K) {
    return queryGemmWorkspaceSize(M, N, K, DataType::BF16);
}

std::vector<GemmConfig> Bf16GemmRunner::getConfigs() const {
    if (impl_->sm_version >= 90) {
        return getDefaultSM90Configs();
    }
    return getDefaultSM80Configs();
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
    
    size_t size_a = static_cast<size_t>(M) * K * getDataTypeSize(dtype);
    size_t size_b = static_cast<size_t>(K) * N * getDataTypeSize(dtype);
    size_t size_d = static_cast<size_t>(M) * N * getDataTypeSize(dtype);
    
    void *A, *B, *D;
    OASR_CUDA_CHECK(cudaMalloc(&A, size_a));
    OASR_CUDA_CHECK(cudaMalloc(&B, size_b));
    OASR_CUDA_CHECK(cudaMalloc(&D, size_d));
    
    GemmConfig best_config = configs[0];
    float best_time = std::numeric_limits<float>::max();
    
    for (const auto& config : configs) {
        try {
            GemmParams params;
            params.A = A;
            params.B = B;
            params.D = D;
            params.M = M;
            params.N = N;
            params.K = K;
            params.lda = K;
            params.ldb = N;
            params.ldd = N;
            params.dtype_a = dtype;
            params.dtype_b = dtype;
            params.dtype_d = dtype;
            params.config = config;
            params.stream = stream;
            
            // Warmup
            for (int i = 0; i < num_warmup; ++i) {
                invokeGemm(params);
            }
            OASR_CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // Timing
            cudaEvent_t start, stop;
            OASR_CUDA_CHECK(cudaEventCreate(&start));
            OASR_CUDA_CHECK(cudaEventCreate(&stop));
            
            OASR_CUDA_CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < num_iter; ++i) {
                invokeGemm(params);
            }
            OASR_CUDA_CHECK(cudaEventRecord(stop, stream));
            OASR_CUDA_CHECK(cudaEventSynchronize(stop));
            
            float elapsed_ms;
            OASR_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_ms /= num_iter;
            
            if (elapsed_ms < best_time) {
                best_time = elapsed_ms;
                best_config = config;
            }
            
            OASR_CUDA_CHECK(cudaEventDestroy(start));
            OASR_CUDA_CHECK(cudaEventDestroy(stop));
        }
        catch (...) {
            continue;
        }
    }
    
    OASR_CUDA_CHECK(cudaFree(A));
    OASR_CUDA_CHECK(cudaFree(B));
    OASR_CUDA_CHECK(cudaFree(D));
    
    return best_config;
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
