// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batched Matrix Multiplication (BMM) kernel implementations using CUTLASS

#include "bmm_kernels.h"
#include "gemm_utils.h"
#include "common/cuda_utils.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <stdexcept>
#include <limits>

namespace oasr {
namespace kernels {
namespace gemm {

template <typename ElementA, typename ElementB, typename ElementC,
          typename LayoutA, typename LayoutB, typename LayoutC>
struct CutlassBmmSM80 {
    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, float, float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 3>;
    
    static GemmStatus run(const ElementA* A, const ElementB* B, ElementC* D,
        int batch_size, int M, int N, int K,
        int64_t lda, int64_t ldb, int64_t ldd,
        int64_t stride_a, int64_t stride_b, int64_t stride_d,
        float alpha, float beta, cudaStream_t stream) {
        typename Gemm::Arguments args({M, N, K}, {A, lda}, stride_a,
            {B, ldb}, stride_b, {D, ldd}, stride_d, {D, ldd}, stride_d,
            {alpha, beta}, batch_size);
        
        Gemm gemm_op;
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess)
            return GemmStatus::NOT_SUPPORTED;
        
        size_t ws_size = Gemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> ws(ws_size);
        
        if (gemm_op.initialize(args, ws.get(), stream) != cutlass::Status::kSuccess)
            return GemmStatus::INTERNAL_ERROR;
        
        return (gemm_op(stream) == cutlass::Status::kSuccess) ? 
               GemmStatus::SUCCESS : GemmStatus::CUTLASS_ERROR;
    }
};

GemmStatus invokeBmm(const void* A, const void* B, void* D,
                     int batch_size, int M, int N, int K,
                     int64_t lda, int64_t ldb, int64_t ldd,
                     int64_t stride_a, int64_t stride_b, int64_t stride_d,
                     float alpha, float beta,
                     TransposeOp trans_a, TransposeOp trans_b,
                     DataType dtype,
                     cudaStream_t stream) {
    if (A == nullptr || B == nullptr || D == nullptr) return GemmStatus::INVALID_ARGUMENT;
    if (batch_size <= 0 || M <= 0 || N <= 0 || K <= 0) return GemmStatus::INVALID_ARGUMENT;
    if (trans_a != TransposeOp::NoTranspose || trans_b != TransposeOp::NoTranspose)
        return GemmStatus::NOT_SUPPORTED;

    if (dtype == DataType::FP16) {
        using DType = cutlass::half_t;
        using Layout = cutlass::layout::RowMajor;
        return CutlassBmmSM80<DType, DType, DType, Layout, Layout, Layout>::run(
            reinterpret_cast<const DType*>(A),
            reinterpret_cast<const DType*>(B),
            reinterpret_cast<DType*>(D),
            batch_size, M, N, K,
            lda, ldb, ldd,
            stride_a, stride_b, stride_d,
            alpha, beta, stream);
    }
    if (dtype == DataType::BF16) {
        using DType = cutlass::bfloat16_t;
        using Layout = cutlass::layout::RowMajor;
        return CutlassBmmSM80<DType, DType, DType, Layout, Layout, Layout>::run(
            reinterpret_cast<const DType*>(A),
            reinterpret_cast<const DType*>(B),
            reinterpret_cast<DType*>(D),
            batch_size, M, N, K,
            lda, ldb, ldd,
            stride_a, stride_b, stride_d,
            alpha, beta, stream);
    }
    return GemmStatus::NOT_SUPPORTED;
}

template <>
GemmStatus invokeBmmStrided<half>(const half* A, const half* B, half* D,
    int batch_size, int M, int N, int K,
    int64_t stride_a, int64_t stride_b, int64_t stride_d,
    float alpha, float beta, TransposeOp trans_a, TransposeOp trans_b,
    cudaStream_t stream) {
    if (trans_a != TransposeOp::NoTranspose || trans_b != TransposeOp::NoTranspose)
        return GemmStatus::NOT_SUPPORTED;
    using DType = cutlass::half_t;
    using Layout = cutlass::layout::RowMajor;
    return CutlassBmmSM80<DType, DType, DType, Layout, Layout, Layout>::run(
        reinterpret_cast<const DType*>(A), reinterpret_cast<const DType*>(B),
        reinterpret_cast<DType*>(D), batch_size, M, N, K, K, N, N,
        stride_a, stride_b, stride_d, alpha, beta, stream);
}

template <>
GemmStatus invokeBmmStrided<__nv_bfloat16>(const __nv_bfloat16* A, const __nv_bfloat16* B, 
    __nv_bfloat16* D, int batch_size, int M, int N, int K,
    int64_t stride_a, int64_t stride_b, int64_t stride_d,
    float alpha, float beta, TransposeOp trans_a, TransposeOp trans_b,
    cudaStream_t stream) {
    if (trans_a != TransposeOp::NoTranspose || trans_b != TransposeOp::NoTranspose)
        return GemmStatus::NOT_SUPPORTED;
    using DType = cutlass::bfloat16_t;
    using Layout = cutlass::layout::RowMajor;
    return CutlassBmmSM80<DType, DType, DType, Layout, Layout, Layout>::run(
        reinterpret_cast<const DType*>(A), reinterpret_cast<const DType*>(B),
        reinterpret_cast<DType*>(D), batch_size, M, N, K, K, N, N,
        stride_a, stride_b, stride_d, alpha, beta, stream);
}

template <>
GemmStatus invokeBmmArray<half>(const half* const*, const half* const*, half* const*,
    int, int, int, int, int64_t, int64_t, int64_t, float, float,
    TransposeOp, TransposeOp, cudaStream_t) { return GemmStatus::NOT_SUPPORTED; }

template <>
GemmStatus invokeBmmArray<__nv_bfloat16>(const __nv_bfloat16* const*, const __nv_bfloat16* const*,
    __nv_bfloat16* const*, int, int, int, int, int64_t, int64_t, int64_t,
    float, float, TransposeOp, TransposeOp, cudaStream_t) { return GemmStatus::NOT_SUPPORTED; }

size_t queryBmmWorkspaceSize(int, int, int, int, DataType, const GemmConfig&) {
    return 8 * 1024 * 1024;
}

struct BmmRunner::Impl {
    DataType dtype;
    int sm_version;
    Impl(DataType dt, int sm) : dtype(dt), sm_version(sm < 0 ? getSMVersion() : sm) {}
};

BmmRunner::BmmRunner(DataType dtype, int sm_version) : impl_(std::make_unique<Impl>(dtype, sm_version)) {}
BmmRunner::~BmmRunner() = default;

size_t BmmRunner::getWorkspaceSize(int batch, int M, int N, int K) const {
    return queryBmmWorkspaceSize(batch, M, N, K, impl_->dtype);
}

GemmStatus BmmRunner::runStrided(const void* A, const void* B, void* D,
    int batch, int M, int N, int K,
    int64_t stride_a, int64_t stride_b, int64_t stride_d,
    float alpha, float beta, void*, size_t, cudaStream_t stream) {
    return invokeBmm(A, B, D, batch, M, N, K, K, N, N,
                    stride_a, stride_b, stride_d,
                    alpha, beta, TransposeOp::NoTranspose, TransposeOp::NoTranspose,
                    impl_->dtype, stream);
}

GemmStatus BmmRunner::runArray(const void* const* A_array, const void* const* B_array,
    void* const* D_array, int batch, int M, int N, int K,
    float, float, void*, size_t, cudaStream_t stream) {
    (void)A_array; (void)B_array; (void)D_array; (void)batch; (void)M; (void)N; (void)K;
    (void)stream;
    return GemmStatus::NOT_SUPPORTED;  // Pointer array mode not implemented
}

std::vector<GemmConfig> BmmRunner::getConfigs() const {
    return impl_->sm_version >= 90 ? getDefaultSM90Configs() : getDefaultSM80Configs();
}

GemmConfig autoTuneBmm(int batch, int M, int N, int K, DataType dtype,
    int num_warmup, int num_iter, cudaStream_t stream) {
    auto configs = getSMVersion() >= 90 ? getDefaultSM90Configs() : getDefaultSM80Configs();
    if (configs.empty()) return GemmConfig();
    
    size_t elem_size = getDataTypeSize(dtype);
    void *A, *B, *D;
    OASR_CUDA_CHECK(cudaMalloc(&A, static_cast<size_t>(batch) * M * K * elem_size));
    OASR_CUDA_CHECK(cudaMalloc(&B, static_cast<size_t>(batch) * K * N * elem_size));
    OASR_CUDA_CHECK(cudaMalloc(&D, static_cast<size_t>(batch) * M * N * elem_size));
    
    GemmConfig best = configs[0];
    float best_time = std::numeric_limits<float>::max();
    
    int64_t stride_a = static_cast<int64_t>(M) * K;
    int64_t stride_b = static_cast<int64_t>(K) * N;
    int64_t stride_d = static_cast<int64_t>(M) * N;

    for (const auto& cfg : configs) {
        try {
            (void)cfg;
            for (int i = 0; i < num_warmup; ++i) {
                invokeBmm(A, B, D, batch, M, N, K, K, N, N,
                         stride_a, stride_b, stride_d,
                         1.0f, 0.0f, TransposeOp::NoTranspose, TransposeOp::NoTranspose,
                         dtype, stream);
            }
            OASR_CUDA_CHECK(cudaStreamSynchronize(stream));
            
            cudaEvent_t start, stop;
            OASR_CUDA_CHECK(cudaEventCreate(&start));
            OASR_CUDA_CHECK(cudaEventCreate(&stop));
            OASR_CUDA_CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < num_iter; ++i) {
                invokeBmm(A, B, D, batch, M, N, K, K, N, N,
                         stride_a, stride_b, stride_d,
                         1.0f, 0.0f, TransposeOp::NoTranspose, TransposeOp::NoTranspose,
                         dtype, stream);
            }
            OASR_CUDA_CHECK(cudaEventRecord(stop, stream));
            OASR_CUDA_CHECK(cudaEventSynchronize(stop));
            
            float ms;
            OASR_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms / num_iter < best_time) { best_time = ms / num_iter; best = cfg; }
            OASR_CUDA_CHECK(cudaEventDestroy(start));
            OASR_CUDA_CHECK(cudaEventDestroy(stop));
        } catch (...) { continue; }
    }
    
    OASR_CUDA_CHECK(cudaFree(A));
    OASR_CUDA_CHECK(cudaFree(B));
    OASR_CUDA_CHECK(cudaFree(D));
    return best;
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
