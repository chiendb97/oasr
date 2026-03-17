// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batched Matrix Multiplication (BMM) kernel implementations using CUTLASS

#include "bmm_kernels.h"
#include "gemm_utils.h"
#include "kernels/common/cuda_utils.h"

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <limits>
#include <stdexcept>

namespace oasr {
namespace kernels {
namespace gemm {

template <typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD>
struct CutlassBmmSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = cutlass::arch::OpClassTensorOp;

    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarps = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::epilogue::thread::ScaleType::Default>;

    static constexpr int NumStages = 2;
    using Gemm = cutlass::gemm::device::GemmBatched<ElementA, LayoutA, ElementB, LayoutB, ElementCD,
                                                    LayoutCD, ElementAccumulator, MMAOp, SmArch,
                                                    ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp,
                                                    EpilogueOp, SwizzleThreadblock, NumStages>;

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
        cutlass::device_memory::allocation<uint8_t> ws(ws_size);

        if (gemm_op.initialize(args, ws.get(), stream) != cutlass::Status::kSuccess)
            return GemmStatus::INTERNAL_ERROR;

        return (gemm_op(stream) == cutlass::Status::kSuccess) ? GemmStatus::SUCCESS
                                                              : GemmStatus::CUTLASS_ERROR;
    }
};

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

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    GemmStatus status = GemmStatus::SUCCESS;

    if (A.scalar_type() == torch::kHalf) {
        status = CutlassBmmSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA, LayoutB,
                                LayoutCD>::run(reinterpret_cast<const cutlass::half_t*>(A_ptr),
                                               reinterpret_cast<const cutlass::half_t*>(B_ptr),
                                               reinterpret_cast<cutlass::half_t*>(D_ptr),
                                               batch_size, M, N, K, lda, ldb, ldd, stride_a,
                                               stride_b, stride_d, alpha, beta, stream);
    } else if (A.scalar_type() == torch::kBFloat16) {
        status = CutlassBmmSM80<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
                                LayoutA, LayoutB,
                                LayoutCD>::run(reinterpret_cast<const cutlass::bfloat16_t*>(A_ptr),
                                               reinterpret_cast<const cutlass::bfloat16_t*>(B_ptr),
                                               reinterpret_cast<cutlass::bfloat16_t*>(D_ptr),
                                               batch_size, M, N, K, lda, ldb, ldd, stride_a,
                                               stride_b, stride_d, alpha, beta, stream);
    } else {
        throw std::invalid_argument("Unsupported data type");
    }

    if (status != GemmStatus::SUCCESS) {
        throw std::runtime_error("BMM execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return D;
}

torch::Tensor invokeBmmArray(const torch::Tensor& A, const torch::Tensor& B,
                             cudaStream_t stream = nullptr) {
    return invokeBmm(A, B, stream);
}

GemmConfig autoTuneBmm(int batch, int M, int N, int K, DataType dtype, int num_warmup, int num_iter,
                       cudaStream_t stream) {
    auto configs = getSMVersion() >= 90 ? getDefaultSM90Configs() : getDefaultSM80Configs();
    if (configs.empty())
        return GemmConfig();

    size_t elem_size = getDataTypeSize(dtype);
    torch::Tensor A, B;
    A = torch::empty({batch, M, K}, A.options());
    B = torch::empty({batch, K, N}, B.options());

    GemmConfig best = configs[0];
    float best_time = std::numeric_limits<float>::max();

    for (const auto& cfg : configs) {
        try {
            (void)cfg;
            for (int i = 0; i < num_warmup; ++i) {
                auto D = invokeBmm(A, B, stream);
            }
            OASR_CUDA_CHECK(cudaStreamSynchronize(stream));

            cudaEvent_t start, stop;
            OASR_CUDA_CHECK(cudaEventCreate(&start));
            OASR_CUDA_CHECK(cudaEventCreate(&stop));
            OASR_CUDA_CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < num_iter; ++i) {
                auto D = invokeBmm(A, B, stream);
            }
            OASR_CUDA_CHECK(cudaEventRecord(stop, stream));
            OASR_CUDA_CHECK(cudaEventSynchronize(stop));

            float ms;
            OASR_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms / num_iter < best_time) {
                best_time = ms / num_iter;
                best = cfg;
            }
            OASR_CUDA_CHECK(cudaEventDestroy(start));
            OASR_CUDA_CHECK(cudaEventDestroy(stop));
        } catch (...) {
            continue;
        }
    }

    return best;
}

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
