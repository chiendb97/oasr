// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Activation quantization kernels — no framework dependencies.
//
// Per-token symmetric int8: one scale per row, chosen from that row alone, so
// a row quantizes identically whatever else shares the batch (padding rows,
// co-batched utterances, capture buckets).  The numerics are pinned and the
// torch reference (``oasr.quantization.methods.w8a8.reference_quantize_per_token``)
// reproduces them bit for bit:
//
//     x'    = f32(x) * smooth_inv              (smoother applied BEFORE the amax)
//     amax  = max(max_j |x'_j|, eps)
//     inv   = 127 / amax                       (IEEE division: __fdiv_rn)
//     q_j   = clamp(rne(x'_j * inv), -127, 127)
//     scale = amax / 127                       (IEEE division: __fdiv_rn)
//
// OASR builds with ``--use_fast_math``, under which a bare ``/`` is an
// approximation; every division here is spelled ``__fdiv_rn`` so the kernel and
// the reference agree at rounding ties.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>

#include <oasr/common/reduction.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace quant {
using namespace oasr::reduction;

constexpr int kQuantMax = 127;

__device__ __forceinline__ int8_t quantizeInt8(float x, float inv) {
    int q = __float2int_rn(__fmul_rn(x, inv));
    q = q > kQuantMax ? kQuantMax : (q < -kQuantMax ? -kQuantMax : q);
    return static_cast<int8_t>(q);
}

template <typename T, int VecSize, bool kSmooth>
__device__ __forceinline__ void loadRowVec(const T* __restrict__ row_input,
                                           const float* __restrict__ smooth_inv, int vec_idx,
                                           float (&x)[VecSize]) {
    oasr::Vec<T, VecSize> v;
    if constexpr (VecSize == 1) {
        v[0] = row_input[vec_idx];
    } else {
        v.load(row_input + vec_idx * VecSize);
    }
#pragma unroll
    for (int j = 0; j < VecSize; ++j) {
        x[j] = static_cast<float>(v[j]);
        if constexpr (kSmooth) {
            x[j] = __fmul_rn(x[j], smooth_inv[vec_idx * VecSize + j]);
        }
    }
}

template <int VecSize>
__device__ __forceinline__ void storeRowVec(int8_t* __restrict__ row_output, int vec_idx,
                                            const float (&x)[VecSize], float inv) {
    int8_t q[VecSize];
#pragma unroll
    for (int j = 0; j < VecSize; ++j) {
        q[j] = quantizeInt8(x[j], inv);
    }
    if constexpr (VecSize == 8) {
        *reinterpret_cast<int2*>(row_output + vec_idx * VecSize) = *reinterpret_cast<int2*>(q);
    } else {
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            row_output[vec_idx * VecSize + j] = q[j];
        }
    }
}

// Narrow rows (up to 32 * VecsPerLane vectors): one warp per row, several rows
// per CTA, the row held in registers between the amax and the quantize — a
// single read of global memory and no shared-memory round trip.
template <typename T, int VecSize, int VecsPerLane, bool kSmooth>
__global__ void quantizePerTokenWarpKernel(const T* __restrict__ input,
                                           const float* __restrict__ smooth_inv,
                                           int8_t* __restrict__ output, float* __restrict__ scale,
                                           int64_t num_rows, int hidden_size, float eps) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (row >= num_rows) {
        return;
    }
    const T* row_input = input + row * hidden_size;
    int8_t* row_output = output + row * hidden_size;
    const int vec_count = hidden_size / VecSize;

    float x[VecsPerLane][VecSize];
    float local_max = 0.0f;
#pragma unroll
    for (int c = 0; c < VecsPerLane; ++c) {
        const int i = lane + c * WARP_SIZE;
        if (i < vec_count) {
            loadRowVec<T, VecSize, kSmooth>(row_input, smooth_inv, i, x[c]);
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                local_max = fmaxf(local_max, fabsf(x[c][j]));
            }
        }
    }
    const float amax = fmaxf(warpReduceMax(local_max), eps);
    const float inv = __fdiv_rn(static_cast<float>(kQuantMax), amax);
    if (lane == 0) {
        scale[row] = __fdiv_rn(amax, static_cast<float>(kQuantMax));
    }
#pragma unroll
    for (int c = 0; c < VecsPerLane; ++c) {
        const int i = lane + c * WARP_SIZE;
        if (i < vec_count) {
            storeRowVec<VecSize>(row_output, i, x[c], inv);
        }
    }
}

// Wide rows (up to blockDim * VecsPerThread vectors): one CTA per row, the row
// held in registers, a block-wide max.
template <typename T, int VecSize, int VecsPerThread, bool kSmooth>
__global__ void quantizePerTokenBlockKernel(const T* __restrict__ input,
                                            const float* __restrict__ smooth_inv,
                                            int8_t* __restrict__ output,
                                            float* __restrict__ scale, int hidden_size,
                                            float eps) {
    const int64_t row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    int8_t* row_output = output + row * hidden_size;
    const int vec_count = hidden_size / VecSize;

    __shared__ float smem_amax;

    float x[VecsPerThread][VecSize];
    float local_max = 0.0f;
#pragma unroll
    for (int c = 0; c < VecsPerThread; ++c) {
        const int i = threadIdx.x + c * blockDim.x;
        if (i < vec_count) {
            loadRowVec<T, VecSize, kSmooth>(row_input, smooth_inv, i, x[c]);
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                local_max = fmaxf(local_max, fabsf(x[c][j]));
            }
        }
    }
    float amax = blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        amax = fmaxf(amax, eps);
        smem_amax = amax;
        scale[row] = __fdiv_rn(amax, static_cast<float>(kQuantMax));
    }
    __syncthreads();
    const float inv = __fdiv_rn(static_cast<float>(kQuantMax), smem_amax);
#pragma unroll
    for (int c = 0; c < VecsPerThread; ++c) {
        const int i = threadIdx.x + c * blockDim.x;
        if (i < vec_count) {
            storeRowVec<VecSize>(row_output, i, x[c], inv);
        }
    }
}

// Fallback for any width: one CTA per row, two passes over global memory (the
// second from L1/L2).  Serves unaligned rows and rows too wide to hold.
template <typename T, int VecSize, bool kSmooth>
__global__ void quantizePerTokenKernel(const T* __restrict__ input,
                                       const float* __restrict__ smooth_inv,
                                       int8_t* __restrict__ output, float* __restrict__ scale,
                                       int hidden_size, float eps) {
    const int64_t row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    int8_t* row_output = output + row * hidden_size;
    const int vec_count = hidden_size / VecSize;

    __shared__ float smem_amax;

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        oasr::Vec<T, VecSize> v;
        if constexpr (VecSize == 1) {
            v[0] = row_input[i];
        } else {
            v.load(row_input + i * VecSize);
        }
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float x = static_cast<float>(v[j]);
            if constexpr (kSmooth) {
                x = __fmul_rn(x, smooth_inv[i * VecSize + j]);
            }
            local_max = fmaxf(local_max, fabsf(x));
        }
    }
    float amax = blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        amax = fmaxf(amax, eps);
        smem_amax = amax;
        scale[row] = __fdiv_rn(amax, static_cast<float>(kQuantMax));
    }
    __syncthreads();
    const float inv = __fdiv_rn(static_cast<float>(kQuantMax), smem_amax);

    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        oasr::Vec<T, VecSize> v;
        if constexpr (VecSize == 1) {
            v[0] = row_input[i];
        } else {
            v.load(row_input + i * VecSize);
        }
        int8_t q[VecSize];
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float x = static_cast<float>(v[j]);
            if constexpr (kSmooth) {
                x = __fmul_rn(x, smooth_inv[i * VecSize + j]);
            }
            q[j] = quantizeInt8(x, inv);
        }
        if constexpr (VecSize == 8) {
            *reinterpret_cast<int2*>(row_output + i * VecSize) = *reinterpret_cast<int2*>(q);
        } else {
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                row_output[i * VecSize + j] = q[j];
            }
        }
    }
}

// Threads per CTA: one warp per 8 vectors' worth of work, at least one warp,
// at most 1024 — a 256-wide row is one warp, a 2048-wide row eight.
inline int quantizeThreads(int vec_count) {
    int threads = ((vec_count + 31) / 32) * 32;
    threads = threads < 32 ? 32 : threads;
    return threads > 1024 ? 1024 : threads;
}

constexpr int kWarpRowsPerBlock = 8;
constexpr int kBlockRowThreads = 256;

template <typename T, int VecsPerLane, bool kSmooth>
void launchWarpRows(const T* input, const float* smooth_inv, int8_t* output, float* scale,
                    int64_t num_rows, int hidden_size, float eps, cudaStream_t stream) {
    dim3 block(kWarpRowsPerBlock * WARP_SIZE);
    dim3 grid(static_cast<unsigned int>((num_rows + kWarpRowsPerBlock - 1) / kWarpRowsPerBlock));
    quantizePerTokenWarpKernel<T, 8, VecsPerLane, kSmooth><<<grid, block, 0, stream>>>(
        input, smooth_inv, output, scale, num_rows, hidden_size, eps);
}

template <typename T, int VecsPerThread, bool kSmooth>
void launchBlockRows(const T* input, const float* smooth_inv, int8_t* output, float* scale,
                     int64_t num_rows, int hidden_size, float eps, cudaStream_t stream) {
    dim3 block(kBlockRowThreads);
    dim3 grid(static_cast<unsigned int>(num_rows));
    quantizePerTokenBlockKernel<T, 8, VecsPerThread, kSmooth>
        <<<grid, block, 0, stream>>>(input, smooth_inv, output, scale, hidden_size, eps);
}

// Register-resident dispatch for 8-aligned rows: a warp per row up to 1024
// columns, a 256-thread CTA per row up to 16384.  Returns false past that.
template <typename T, bool kSmooth>
bool launchResident(const T* input, const float* smooth_inv, int8_t* output, float* scale,
                    int64_t num_rows, int hidden_size, float eps, cudaStream_t stream) {
    const int vec_count = hidden_size / 8;
    if (vec_count <= WARP_SIZE) {
        launchWarpRows<T, 1, kSmooth>(input, smooth_inv, output, scale, num_rows, hidden_size,
                                      eps, stream);
    } else if (vec_count <= 2 * WARP_SIZE) {
        launchWarpRows<T, 2, kSmooth>(input, smooth_inv, output, scale, num_rows, hidden_size,
                                      eps, stream);
    } else if (vec_count <= 4 * WARP_SIZE) {
        launchWarpRows<T, 4, kSmooth>(input, smooth_inv, output, scale, num_rows, hidden_size,
                                      eps, stream);
    } else if (vec_count <= kBlockRowThreads) {
        launchBlockRows<T, 1, kSmooth>(input, smooth_inv, output, scale, num_rows, hidden_size,
                                       eps, stream);
    } else if (vec_count <= 2 * kBlockRowThreads) {
        launchBlockRows<T, 2, kSmooth>(input, smooth_inv, output, scale, num_rows, hidden_size,
                                       eps, stream);
    } else if (vec_count <= 4 * kBlockRowThreads) {
        launchBlockRows<T, 4, kSmooth>(input, smooth_inv, output, scale, num_rows, hidden_size,
                                       eps, stream);
    } else if (vec_count <= 8 * kBlockRowThreads) {
        launchBlockRows<T, 8, kSmooth>(input, smooth_inv, output, scale, num_rows, hidden_size,
                                       eps, stream);
    } else {
        return false;
    }
    return true;
}

template <typename T>
cudaError_t QuantizePerToken(const T* input, const float* smooth_inv, int8_t* output,
                             float* scale, int64_t num_rows, int hidden_size, float eps,
                             cudaStream_t stream) {
    if (num_rows == 0) {
        return cudaSuccess;
    }
    const bool vec8 = hidden_size % 8 == 0 && reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(output) % 8 == 0;
    dim3 grid(static_cast<unsigned int>(num_rows));
    if (vec8) {
        const bool resident =
            smooth_inv != nullptr
                ? launchResident<T, true>(input, smooth_inv, output, scale, num_rows,
                                          hidden_size, eps, stream)
                : launchResident<T, false>(input, smooth_inv, output, scale, num_rows,
                                           hidden_size, eps, stream);
        if (resident) {
            return cudaGetLastError();
        }
        dim3 block(quantizeThreads(hidden_size / 8));
        if (smooth_inv != nullptr) {
            quantizePerTokenKernel<T, 8, true>
                <<<grid, block, 0, stream>>>(input, smooth_inv, output, scale, hidden_size, eps);
        } else {
            quantizePerTokenKernel<T, 8, false>
                <<<grid, block, 0, stream>>>(input, smooth_inv, output, scale, hidden_size, eps);
        }
    } else {
        dim3 block(quantizeThreads(hidden_size));
        if (smooth_inv != nullptr) {
            quantizePerTokenKernel<T, 1, true>
                <<<grid, block, 0, stream>>>(input, smooth_inv, output, scale, hidden_size, eps);
        } else {
            quantizePerTokenKernel<T, 1, false>
                <<<grid, block, 0, stream>>>(input, smooth_inv, output, scale, hidden_size, eps);
        }
    }
    return cudaGetLastError();
}

// =============================================================================
// Static (calibrated) int8 quantization
//
//     q_j = clamp(rne(f32(x_j) * mul[j % K]), -127, 127)
//
// ``mul`` is the per-channel multiplier ``smooth_inv * (1 / input_scale)``
// (``oasr.functionals.quant.static_multiplier``), computed once when the layer is
// bound, so a static layer's quantize is one multiply per element: no amax, no
// division.  The same expression the int8-output GEMM epilogue evaluates
// (``MulRintClampInt8``), so the fused and unfused paths agree bit for bit.
// =============================================================================

template <typename T, int VecSize>
__global__ void quantizeStaticKernel(const T* __restrict__ input, const float* __restrict__ mul,
                                     int8_t* __restrict__ output, int64_t num_vecs,
                                     int vecs_per_row) {
    for (int64_t v = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; v < num_vecs;
         v += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int col_vec = static_cast<int>(v % vecs_per_row);
        oasr::Vec<T, VecSize> x;
        if constexpr (VecSize == 1) {
            x[0] = input[v];
        } else {
            x.load(input + v * VecSize);
        }
        int8_t q[VecSize];
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            q[j] = quantizeInt8(static_cast<float>(x[j]), mul[col_vec * VecSize + j]);
        }
        if constexpr (VecSize == 8) {
            *reinterpret_cast<int2*>(output + v * VecSize) = *reinterpret_cast<int2*>(q);
        } else {
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                output[v * VecSize + j] = q[j];
            }
        }
    }
}

template <typename T>
cudaError_t QuantizeStatic(const T* input, const float* mul, int8_t* output, int64_t num_rows,
                           int hidden_size, cudaStream_t stream) {
    const int64_t numel = num_rows * hidden_size;
    if (numel == 0) {
        return cudaSuccess;
    }
    const bool vec8 = hidden_size % 8 == 0 && reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(output) % 8 == 0;
    constexpr int kThreads = 256;
    // Enough CTAs to fill the device a few times over; the grid-stride loop
    // takes the rest.
    constexpr int64_t kMaxBlocks = 4096;
    if (vec8) {
        const int64_t num_vecs = numel / 8;
        const int64_t blocks = std::min<int64_t>((num_vecs + kThreads - 1) / kThreads, kMaxBlocks);
        quantizeStaticKernel<T, 8><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
            input, mul, output, num_vecs, hidden_size / 8);
    } else {
        const int64_t blocks = std::min<int64_t>((numel + kThreads - 1) / kThreads, kMaxBlocks);
        quantizeStaticKernel<T, 1><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
            input, mul, output, numel, hidden_size);
    }
    return cudaGetLastError();
}

}  // namespace quant
}  // namespace oasr
