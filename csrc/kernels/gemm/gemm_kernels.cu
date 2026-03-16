// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel implementations using CUTLASS
// Supports BF16 and FP16 precision

#include <cstdint>
#include <torch/extension.h>

#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "gemm_kernels.h"
#include "gemm_utils.h"
#include "kernels/common/cuda_utils.h"

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
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Standard GEMM Implementation (SM80)
//==============================================================================

template <typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD>
struct CutlassGemmSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = cutlass::arch::OpClassTensorOp;

    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarps = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::epilogue::thread::ScaleType::Default>;

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

template <typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD>
struct CutlassGemmReluSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarps = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::epilogue::thread::ScaleType::Default>;

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

template <typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD>
struct CutlassGemmGeluSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarps = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::epilogue::thread::ScaleType::Default>;

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

template <typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD>
struct CutlassGemmSwishSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarps = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::epilogue::thread::ScaleType::Default>;

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
                            LayoutC>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr),
                                          reinterpret_cast<const cutlass::half_t*>(B_ptr),
                                          reinterpret_cast<const cutlass::half_t*>(C_ptr),
                                          reinterpret_cast<cutlass::half_t*>(D_ptr), M, N, K, lda,
                                          ldb, ldc, alpha, stream);
    } else if (A.scalar_type() == torch::kBFloat16) {
        status = CutlassGemmSM80<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
                                 LayoutA, LayoutB,
                                 LayoutC>::run(reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
                                               reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
                                               reinterpret_cast<const cutlass::bfloat16_t*>(C_ptr),
                                               reinterpret_cast<cutlass::bfloat16_t*>(D_ptr), M, N,
                                               K, lda, ldb, ldc, alpha, stream);
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

    if (A.scalar_type() == torch::kHalf) {
        if (activation == ActivationType::RELU) {
            status =
                CutlassGemmReluSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA,
                                    LayoutB,
                                    LayoutC>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr),
                                                  reinterpret_cast<const cutlass::half_t*>(B_ptr),
                                                  reinterpret_cast<const cutlass::half_t*>(C_ptr),
                                                  reinterpret_cast<cutlass::half_t*>(D_ptr), M, N,
                                                  K, lda, ldb, ldc, alpha, stream);
        } else if (activation == ActivationType::GELU) {
            status =
                CutlassGemmGeluSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA,
                                    LayoutB,
                                    LayoutC>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr),
                                                  reinterpret_cast<const cutlass::half_t*>(B_ptr),
                                                  reinterpret_cast<const cutlass::half_t*>(C_ptr),
                                                  reinterpret_cast<cutlass::half_t*>(D_ptr), M, N,
                                                  K, lda, ldb, ldc, alpha, stream);
        } else if (activation == ActivationType::SWISH) {
            status =
                CutlassGemmSwishSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA,
                                     LayoutB,
                                     LayoutC>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr),
                                                   reinterpret_cast<const cutlass::half_t*>(B_ptr),
                                                   reinterpret_cast<const cutlass::half_t*>(C_ptr),
                                                   reinterpret_cast<cutlass::half_t*>(D_ptr), M, N,
                                                   K, lda, ldb, ldc, alpha, stream);
        } else {
            status = GemmStatus::INVALID_ARGUMENT;
        }
    } else if (A.scalar_type() == torch::kBFloat16) {
        if (activation == ActivationType::RELU) {
            status = CutlassGemmReluSM80<
                cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, LayoutA, LayoutB,
                LayoutC>::run(reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
                              reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
                              reinterpret_cast<const cutlass::bfloat16_t*>(C_ptr),
                              reinterpret_cast<cutlass::bfloat16_t*>(D_ptr), M, N, K, lda, ldb, ldc,
                              alpha, stream);
        } else if (activation == ActivationType::GELU) {
            status = CutlassGemmGeluSM80<
                cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, LayoutA, LayoutB,
                LayoutC>::run(reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
                              reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
                              reinterpret_cast<const cutlass::bfloat16_t*>(C_ptr),
                              reinterpret_cast<cutlass::bfloat16_t*>(D_ptr), M, N, K, lda, ldb, ldc,
                              alpha, stream);
        } else if (activation == ActivationType::SWISH) {
            status = CutlassGemmSwishSM80<
                cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, LayoutA, LayoutB,
                LayoutC>::run(reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
                              reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
                              reinterpret_cast<const cutlass::bfloat16_t*>(C_ptr),
                              reinterpret_cast<cutlass::bfloat16_t*>(D_ptr), M, N, K, lda, ldb, ldc,
                              alpha, stream);
        } else {
            status = GemmStatus::INVALID_ARGUMENT;
        }
    } else {
        throw std::invalid_argument("Invalid input tensor type");
    }

    if (status != GemmStatus::SUCCESS) {
        // throw exception with status code
        throw std::runtime_error("GEMM activation execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return D;
};

//==============================================================================
// Auto-Tuning
//==============================================================================

GemmConfig autoTuneGemm(int M, int N, int K, DataType dtype, int num_warmup, int num_iter,
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

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
