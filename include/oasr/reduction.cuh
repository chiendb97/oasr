// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA activation kernels — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/limits>
#include <oasr/common/math.h>
#include <oasr/common/utils.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace reduction {
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    __shared__ T shared[32];  // One slot per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only first warp does the final reduction
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    T max_val = val;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return max_val;
}

template <typename T>
__device__ __forceinline__ T blockReduceMax(T val) {
    __shared__ T shared[32];  // One slot per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane]
                                                 : cuda::std::numeric_limits<T>::lowest();

    // Only first warp does the final reduction
    if (wid == 0) {
        val = warpReduceMax(val);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T blockBroadcast(T value, T* workspace) {
    if (threadIdx.x == 0) {
        *workspace = value;
    }
    __syncthreads();
    return *workspace;
}
}  // namespace reduction
}  // namespace oasr