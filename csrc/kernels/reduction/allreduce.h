// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace oasr {
namespace kernels {

// =============================================================================
// Constants
// =============================================================================

constexpr int WARP_SIZE = 32;

// =============================================================================
// Warp-level reductions
// =============================================================================

// Warp-level reduction for sum
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for max
template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for min
template <typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = min(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// =============================================================================
// Block-level reductions
// =============================================================================

// Block-level reduction for sum
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

// Block-level reduction for max
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
    
    // Only first warp does the final reduction
    // Use a very small value for threads that don't participate
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(-1e30);
    if (wid == 0) {
        val = warpReduceMax(val);
    }
    
    return val;
}

// Block-level reduction for min
template <typename T>
__device__ __forceinline__ T blockReduceMin(T val) {
    __shared__ T shared[32];  // One slot per warp
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceMin(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Only first warp does the final reduction
    // Use a very large value for threads that don't participate
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(1e30);
    if (wid == 0) {
        val = warpReduceMin(val);
    }
    
    return val;
}

} // namespace kernels
} // namespace oasr
