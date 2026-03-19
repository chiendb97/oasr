// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel implementations using CUTLASS
// Supports BF16 and FP16 precision

#include <cstdint>
#include <stdexcept>
#include <torch/extension.h>

// Suppress warnings from CUTLASS headers
#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include "gemm_kernels.h"
#include "gemm_utils.h"

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Epilogue Op Aliases
//==============================================================================

template <typename ElementCD, typename ElementCompute>
struct EpilogueIdentity {
    using Op = cutlass::epilogue::thread::LinearCombination<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <typename ElementCD, typename ElementCompute>
struct EpilogueRelu {
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <typename ElementCD, typename ElementCompute>
struct EpilogueGelu {
    using Op = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <typename ElementCD, typename ElementCompute>
struct EpilogueSwish {
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

//==============================================================================
// Unified GEMM Implementation (SM80)
//==============================================================================

template <typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD, template <typename, typename> class EpilogueFunctor>
struct CutlassGemmSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarps = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp = typename EpilogueFunctor<ElementCD, ElementComputeEpilogue>::Op;

    static constexpr int NumStages = 2;

    using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementCD,
                                             LayoutCD, ElementAccumulator, MMAOp, SmArch,
                                             ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp,
                                             EpilogueOp, SwizzleThreadblock, NumStages>;

    static GemmStatus run(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                          int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
                          ElementComputeEpilogue alpha, cudaStream_t stream) {
        int split_k_slices = 1;
        float beta = (C == nullptr) ? 0.0f : 1.0f;
        typename Gemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, 0}, {D, ldc},
                                      {alpha, beta}, split_k_slices);

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

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    GemmStatus status = GemmStatus::SUCCESS;

    if (A.scalar_type() == torch::kHalf) {
        status =
            CutlassGemmSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA, LayoutB,
                            LayoutC,
                            EpilogueIdentity>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr),
                                                   reinterpret_cast<const cutlass::half_t*>(B_ptr),
                                                   reinterpret_cast<const cutlass::half_t*>(C_ptr),
                                                   reinterpret_cast<cutlass::half_t*>(D_ptr), M, N,
                                                   K, lda, ldb, ldc, alpha, stream);
    } else if (A.scalar_type() == torch::kBFloat16) {
        status = CutlassGemmSM80<
            cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, LayoutA, LayoutB,
            LayoutC, EpilogueIdentity>::run(reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
                                            reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
                                            reinterpret_cast<const cutlass::bfloat16_t*>(C_ptr),
                                            reinterpret_cast<cutlass::bfloat16_t*>(D_ptr), M, N, K,
                                            lda, ldb, ldc, alpha, stream);
    } else {
        status = GemmStatus::INVALID_ARGUMENT;
    }

    if (status != GemmStatus::SUCCESS) {
        // throw exception with status code
        throw std::runtime_error("GEMM execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return D;
};

//==============================================================================
// Fused GEMM + Activation (placeholder)
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

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    GemmStatus status = GemmStatus::SUCCESS;

    // Dispatch activation type, then dtype
#define DISPATCH_GEMM_ACTIVATION(EpilogueFn)                                                       \
    if (A.scalar_type() == torch::kHalf) {                                                         \
        status = CutlassGemmSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA,       \
                                 LayoutB, LayoutC,                                                 \
                                 EpilogueFn>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr), \
                                                  reinterpret_cast<const cutlass::half_t*>(B_ptr), \
                                                  reinterpret_cast<const cutlass::half_t*>(C_ptr), \
                                                  reinterpret_cast<cutlass::half_t*>(D_ptr), M, N, \
                                                  K, lda, ldb, ldc, alpha, stream);                \
    } else if (A.scalar_type() == torch::kBFloat16) {                                              \
        status =                                                                                   \
            CutlassGemmSM80<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,         \
                            LayoutA, LayoutB, LayoutC,                                             \
                            EpilogueFn>::run(reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),  \
                                             reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),  \
                                             reinterpret_cast<const cutlass::bfloat16_t*>(C_ptr),  \
                                             reinterpret_cast<cutlass::bfloat16_t*>(D_ptr), M, N,  \
                                             K, lda, ldb, ldc, alpha, stream);                     \
    } else {                                                                                       \
        throw std::invalid_argument("Invalid input tensor type");                                  \
    }

    if (activation == ActivationType::RELU) {
        DISPATCH_GEMM_ACTIVATION(EpilogueRelu)
    } else if (activation == ActivationType::GELU) {
        DISPATCH_GEMM_ACTIVATION(EpilogueGelu)
    } else if (activation == ActivationType::SWISH) {
        DISPATCH_GEMM_ACTIVATION(EpilogueSwish)
    } else {
        status = GemmStatus::INVALID_ARGUMENT;
    }

#undef DISPATCH_GEMM_ACTIVATION

    if (status != GemmStatus::SUCCESS) {
        // throw exception with status code
        throw std::runtime_error("GEMM activation execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return D;
};

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
