// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA softmax kernel — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/limits>

#include <oasr/common/math.h>
#include <oasr/common/types.h>
#include <oasr/common/utils.h>
#include <oasr/common/vec_dtypes.h>
#include <oasr/reduction.cuh>

namespace oasr {
namespace softmax {
using namespace oasr::reduction;

// =============================================================================
// Softmax Kernel
// =============================================================================

// One block per row. Three passes: find max, compute exp(x-max) + sum, normalize.
template <typename T, int VecSize>
__global__ void softmaxKernel(const T* __restrict__ input, T* __restrict__ output, int num_cols) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * num_cols;
    T* row_output = output + row_idx * num_cols;

    const int vec_num_cols = num_cols / VecSize;

    __shared__ float smem[2];  // workspace for blockBroadcast: [max, sum]

    // Phase 1: find row maximum for numerical stability
    float local_max = cuda::std::numeric_limits<float>::lowest();
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            local_max = max(local_max, static_cast<float>(v[j]));
        }
    }
    float row_max = blockBroadcast(blockReduceMax(local_max), &smem[0]);

    // Phase 2: compute exp(x - max) for each element, store to output, accumulate sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = expf(static_cast<float>(v[j]) - row_max);
            local_sum += vals[j];
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
    float inv_sum = 1.0f / blockBroadcast(blockReduceSum(local_sum), &smem[1]);

    // Phase 3: normalize; each thread reads back its own phase-2 output
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_output + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v[j]) * inv_sum;
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Typed Launcher (raw pointer interface, returns cudaError_t)
// =============================================================================

template <typename T>
cudaError_t Softmax(const T* input, T* output, unsigned int num_rows, unsigned int num_cols,
                    cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (num_cols >= static_cast<unsigned int>(VecSize)) &&
                   (num_cols % VecSize == 0) && isAligned<T, VecSize>(input) &&
                   isAligned<T, VecSize>(output);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(num_cols) / VecSize);
        softmaxKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, output, static_cast<int>(num_cols));
    } else {
        int block_size = alignedBlockSize(static_cast<int>(num_cols));
        softmaxKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, output, static_cast<int>(num_cols));
    }
    return cudaGetLastError();
}

}  // namespace softmax
}  // namespace oasr
