// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batched Matrix Multiplication (BMM) kernel implementations using CUTLASS

#include <stdexcept>

#include "bmm_impl.h"
#include "bmm_kernels.h"
#include "gemm_utils.h"
#include "kernels/common/arch_dispatch.h"
#include "kernels/common/arch_traits.h"

namespace oasr {
namespace kernels {
namespace gemm {

template <int SmVersion>
static GemmStatus dispatchBmmDtype(const torch::Tensor& A, const void* A_ptr, const void* B_ptr,
                                   void* D_ptr, int batch_size, int M, int N, int K, uint64_t lda,
                                   uint64_t ldb, uint64_t ldd, int64_t stride_a, int64_t stride_b,
                                   int64_t stride_d, float alpha, float beta, cudaStream_t stream) {
    using Traits = ArchTraits<SmVersion>;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    if (A.scalar_type() == torch::kHalf) {
        return CutlassBmm<Traits, cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA,
                           LayoutB,
                           LayoutCD>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr),
                                          reinterpret_cast<const cutlass::half_t*>(B_ptr),
                                          reinterpret_cast<cutlass::half_t*>(D_ptr), batch_size, M,
                                          N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha,
                                          beta, stream);
    } else if (A.scalar_type() == torch::kBFloat16) {
        if constexpr (Traits::kSupportsBF16) {
            return CutlassBmm<Traits, cutlass::bfloat16_t, cutlass::bfloat16_t,
                               cutlass::bfloat16_t, LayoutA, LayoutB,
                               LayoutCD>::run(reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
                                              reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
                                              reinterpret_cast<cutlass::bfloat16_t*>(D_ptr),
                                              batch_size, M, N, K, lda, ldb, ldd, stride_a,
                                              stride_b, stride_d, alpha, beta, stream);
        } else {
            throw std::runtime_error("BF16 not supported on SM" + std::to_string(SmVersion));
        }
    }
    throw std::invalid_argument("Unsupported data type");
}

torch::Tensor invokeBmm(const torch::Tensor& A, const torch::Tensor& B, cudaStream_t stream) {
    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int N = B.size(1);
    const int K = A.size(2);

    auto D = torch::empty({batch_size, M, N}, A.options());

    const void* A_ptr = A.data_ptr();
    const void* B_ptr = B.data_ptr();
    void* D_ptr = D.data_ptr();

    if (A_ptr == nullptr || B_ptr == nullptr || D_ptr == nullptr) {
        throw std::invalid_argument("Invalid input tensor");
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("Invalid input tensor dimensions");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    uint64_t lda = K;
    uint64_t ldb = K;
    uint64_t ldd = N;

    int64_t stride_a = M * K;
    int64_t stride_b = N * K;
    int64_t stride_d = M * N;

    GemmStatus status = GemmStatus::SUCCESS;

    int sm = getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        status = dispatchBmmDtype<SM_VERSION>(A, A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb,
                                              ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    });

    if (status != GemmStatus::SUCCESS) {
        throw std::runtime_error("BMM execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return D;
}

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
