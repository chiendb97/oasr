// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Grouped GEMM kernel interfaces using NVIDIA CUTLASS
// Supports variable-sized problems with BF16 and FP16 precision

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/util/device_memory.h>

#include <ATen/core/TensorBody.h>
#include <cstdint>
#include <torch/torch.h>
#include <vector>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Grouped GEMM Interface
//==============================================================================

// Grouped GEMM Problem Descriptor
template <typename ElementA, typename ElementB, typename ElementCD>
struct GroupedGemmProblemDesc {
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problems_sizes_device;

    cutlass::DeviceAllocation<ElementA*> ptr_A_device;
    cutlass::DeviceAllocation<ElementB*> ptr_B_device;
    cutlass::DeviceAllocation<ElementCD*> ptr_D_device;

    cutlass::DeviceAllocation<int64_t> lda_device;
    cutlass::DeviceAllocation<int64_t> ldb_device;
    cutlass::DeviceAllocation<int64_t> ldd_device;

    GroupedGemmProblemDesc(int problem_count, int L, int K, int N, const torch::Tensor& A,
                           const torch::Tensor& B, const torch::Tensor& offset,
                           const torch::Tensor& D)
        : problem_sizes(problem_count),
          problems_sizes_device(problem_count),
          ptr_A_device(problem_count),
          ptr_B_device(problem_count),
          ptr_D_device(problem_count),
          lda_device(problem_count),
          ldb_device(problem_count),
          ldd_device(problem_count) {
        std::vector<ElementA*> ptr_A(problem_count);
        std::vector<ElementB*> ptr_B(problem_count);
        std::vector<ElementCD*> ptr_D(problem_count);
        std::vector<int64_t> lda(problem_count);
        std::vector<int64_t> ldb(problem_count);
        std::vector<int64_t> ldd(problem_count);

        auto offset_host = offset.cpu();

        int offset_M = 0;
        for (int i = 0; i < problem_count; ++i) {
            int next_offset_M = offset_host[i].item<int>();
            int M = next_offset_M - offset_M;
            problem_sizes[i] = cutlass::gemm::GemmCoord(M, N, K);
            lda[i] = K;
            ldb[i] = K;
            ldd[i] = N;

            // A and D are laid out as concatenated [M_i, K] and [M_i, N] tiles.
            // Use data_ptr() (void*) and cast to avoid undefined symbol for data_ptr<cutlass::*>
            ptr_A[i] =
                reinterpret_cast<ElementA*>(A.data_ptr()) + static_cast<int64_t>(offset_M) * K;
            ptr_B[i] = reinterpret_cast<ElementB*>(B[i].data_ptr());
            ptr_D[i] =
                reinterpret_cast<ElementCD*>(D.data_ptr()) + static_cast<int64_t>(offset_M) * N;

            offset_M = next_offset_M;
        }

        problems_sizes_device.copy_from_host(problem_sizes.data());
        ptr_A_device.copy_from_host(ptr_A.data());
        ptr_B_device.copy_from_host(ptr_B.data());
        ptr_D_device.copy_from_host(ptr_D.data());
        lda_device.copy_from_host(lda.data());
        ldb_device.copy_from_host(ldb.data());
        ldd_device.copy_from_host(ldd.data());
    }
};

/**
 * @brief Execute grouped GEMM operation
 *
 * Computes: D[i] = alpha * A[i] @ B[i] + beta * D[i] for variable-sized problems
 *
 * @param A Input tensor [L, K]
 * @param B Input tensor [B, N, K]
 * @param offset Offset tensor [B]
 * @param stream CUDA stream
 * @return Output tensor [L, N]
 * @note L = sum(M_i for i in [0, B-1]), B is the batch size
 */
torch::Tensor invokeGroupGemm(const torch::Tensor& A, const torch::Tensor& B,
                              const torch::Tensor& offset, cudaStream_t stream = nullptr);

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
