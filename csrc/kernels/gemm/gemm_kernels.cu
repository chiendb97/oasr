// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel implementations using CUTLASS
// Supports BF16 and FP16 precision with architecture-aware dispatch

#include <cstdint>
#include <stdexcept>
#include <torch/extension.h>

#include "gemm_impl.h"
#include "gemm_kernels.h"
#include "gemm_utils.h"
#include "kernels/common/arch_dispatch.h"
#include "kernels/common/arch_traits.h"
#include "kernels/common/epilogue_functors.h"

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Arch-dispatched helpers (template context enables if constexpr)
//==============================================================================

template <int SmVersion, template <int, typename, typename> class EpilogueFn>
static GemmStatus dispatchGemmDtype(const torch::Tensor& A, const void* A_ptr, const void* B_ptr,
                                    const void* C_ptr, void* D_ptr, int M, int N, int K,
                                    uint64_t lda, uint64_t ldb, uint64_t ldc, float alpha,
                                    cudaStream_t stream) {
    using Traits = ArchTraits<SmVersion>;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    if (A.scalar_type() == torch::kHalf) {
        return CutlassGemm<Traits, cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA,
                           LayoutB, LayoutC,
                           EpilogueFn>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr),
                                            reinterpret_cast<const cutlass::half_t*>(B_ptr),
                                            reinterpret_cast<const cutlass::half_t*>(C_ptr),
                                            reinterpret_cast<cutlass::half_t*>(D_ptr), M, N, K, lda,
                                            ldb, ldc, alpha, stream);
    } else if (A.scalar_type() == torch::kBFloat16) {
        if constexpr (Traits::kSupportsBF16) {
            return CutlassGemm<Traits, cutlass::bfloat16_t, cutlass::bfloat16_t,
                               cutlass::bfloat16_t, LayoutA, LayoutB, LayoutC, EpilogueFn>::run(
                reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
                reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
                reinterpret_cast<const cutlass::bfloat16_t*>(C_ptr),
                reinterpret_cast<cutlass::bfloat16_t*>(D_ptr), M, N, K, lda, ldb, ldc, alpha,
                stream);
        } else {
            throw std::runtime_error("BF16 not supported on SM" + std::to_string(SmVersion));
        }
    }
    return GemmStatus::INVALID_ARGUMENT;
}

//==============================================================================
// Public API Implementations
//==============================================================================

torch::Tensor invokeGemm(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
                         cudaStream_t stream) {
    const int K = A.size(-1);
    const int M = A.numel() / K;
    const int N = B.numel() / K;

    auto out_sizes = A.sizes().vec();
    out_sizes.back() = N;

    auto D = torch::empty(out_sizes, A.options());

    const void* A_ptr = A.data_ptr();
    const void* B_ptr = B.data_ptr();
    const void* C_ptr = C.defined() ? C.data_ptr() : nullptr;
    void* D_ptr = D.data_ptr();

    if (A_ptr == nullptr || B_ptr == nullptr || D_ptr == nullptr) {
        throw std::invalid_argument("Invalid input tensor");
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("Invalid input tensor dimensions");
    }

    float alpha = 1.0f;

    uint64_t lda = K;
    uint64_t ldb = K;
    uint64_t ldc = N;

    GemmStatus status = GemmStatus::SUCCESS;

    int sm = getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        status = dispatchGemmDtype<SM_VERSION, EpilogueIdentity>(A, A_ptr, B_ptr, C_ptr, D_ptr, M,
                                                                 N, K, lda, ldb, ldc, alpha, stream);
    });

    if (status != GemmStatus::SUCCESS) {
        throw std::runtime_error("GEMM execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return D;
};

//==============================================================================
// Fused GEMM + Activation
//==============================================================================

torch::Tensor invokeGemmActivation(const torch::Tensor& A, const torch::Tensor& B,
                                   const torch::Tensor& C, ActivationType activation,
                                   cudaStream_t stream) {
    const int K = A.size(-1);
    const int M = A.numel() / K;
    const int N = B.numel() / K;

    auto out_sizes = A.sizes().vec();
    out_sizes.back() = N;
    auto D = torch::empty(out_sizes, A.options());

    const void* A_ptr = A.data_ptr();
    const void* B_ptr = B.data_ptr();
    const void* C_ptr = C.defined() ? C.data_ptr() : nullptr;
    void* D_ptr = D.data_ptr();

    if (A_ptr == nullptr || B_ptr == nullptr || D_ptr == nullptr) {
        throw std::invalid_argument("Invalid input tensor");
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("Invalid input tensor dimensions");
    }

    float alpha = 1.0f;

    uint64_t lda = K;
    uint64_t ldb = K;
    uint64_t ldc = N;

    GemmStatus status = GemmStatus::SUCCESS;

    int sm = getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        if (activation == ActivationType::RELU) {
            status = dispatchGemmDtype<SM_VERSION, EpilogueRelu>(A, A_ptr, B_ptr, C_ptr, D_ptr, M,
                                                                 N, K, lda, ldb, ldc, alpha, stream);
        } else if (activation == ActivationType::GELU) {
            status = dispatchGemmDtype<SM_VERSION, EpilogueGelu>(A, A_ptr, B_ptr, C_ptr, D_ptr, M,
                                                                 N, K, lda, ldb, ldc, alpha, stream);
        } else if (activation == ActivationType::SWISH) {
            status = dispatchGemmDtype<SM_VERSION, EpilogueSwish>(A, A_ptr, B_ptr, C_ptr, D_ptr, M,
                                                                  N, K, lda, ldb, ldc, alpha, stream);
        } else {
            status = GemmStatus::INVALID_ARGUMENT;
        }
    });

    if (status != GemmStatus::SUCCESS) {
        throw std::runtime_error("GEMM activation execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return D;
};

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
