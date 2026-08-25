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
#include <oasr/common/reduction.h>
#include <oasr/common/types.h>
#include <oasr/common/utils.h>
#include <oasr/common/vec_dtypes.h>

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
    }
    float inv_sum = 1.0f / blockBroadcast(blockReduceSum(local_sum), &smem[1]);

    // Phase 3: normalize; each thread reads back its own phase-2 output
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = expf(static_cast<float>(v[j]) - row_max) * inv_sum;
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// Online softmax pairwise merge over a warp.
// Treats lanes as partials (m, s) where s = sum_i exp(x_i - m); combines two
// partials (m1, s1), (m2, s2) into (max(m1,m2), s1*exp(m1-new_m) + s2*exp(m2-new_m)).
__device__ __forceinline__ float2 warpReduce(float2 val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xffffffff, val.x, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, val.y, offset);
        float new_max = max(val.x, other_max);
        val.y = val.y * expf(val.x - new_max) + other_sum * expf(other_max - new_max);
        val.x = new_max;
    }
    return val;
}

__device__ __forceinline__ float2 blockReduce(float2 val) {
    __shared__ float2 shared[32];  // One slot per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduce(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val.x = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane].x
                                                   : cuda::std::numeric_limits<float>::lowest();

    val.y = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane].y : 0.0f;
    // Only first warp does the final reduction
    if (wid == 0) {
        val = warpReduce(val);
    }
    return val;
}

// One block per row. Two passes: online (max, sum) accumulation, then emit
// log_softmax(x) = x - row_max - log(sum_exp). The first phase mirrors
// onlineSoftmaxKernel; the second writes the log form directly.
template <typename T, int VecSize>
__global__ void onlineLogSoftmaxKernel(const T* __restrict__ input, T* __restrict__ output,
                                       int num_cols) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * num_cols;
    T* row_output = output + row_idx * num_cols;

    const int vec_num_cols = num_cols / VecSize;

    __shared__ float2 smem;

    float2 local_val = make_float2(cuda::std::numeric_limits<float>::lowest(), 0.0f);
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        float vec_max = static_cast<float>(v[0]);
#pragma unroll
        for (int j = 1; j < VecSize; j++) {
            vec_max = max(vec_max, static_cast<float>(v[j]));
        }
        float new_max = max(local_val.x, vec_max);
        float vec_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vec_sum += expf(static_cast<float>(v[j]) - new_max);
        }
        local_val.y = local_val.y * expf(local_val.x - new_max) + vec_sum;
        local_val.x = new_max;
    }
    float2 row_val = blockBroadcast(blockReduce(local_val), &smem);
    const float row_max = row_val.x;
    const float log_norm = logf(row_val.y);  // log(sum_exp(x - row_max))

    // Phase 2: emit log_softmax(x) = (x - row_max) - log_norm.
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v[j]) - row_max - log_norm;
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// One block per row. Two passes: online (max, sum) accumulation, then normalize.
template <typename T, int VecSize>
__global__ void onlineSoftmaxKernel(const T* __restrict__ input, T* __restrict__ output,
                                    int num_cols) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * num_cols;
    T* row_output = output + row_idx * num_cols;

    const int vec_num_cols = num_cols / VecSize;

    __shared__ float2 smem;  // workspace for blockBroadcast

    // Phase 1: single-pass online (max, sum) over the row.
    // Per loaded vector, fold elements into the per-thread partial in two steps:
    //   1) update running max,
    //   2) rescale running sum by exp(old_max - new_max) and add sum_j exp(x_j - new_max).
    // This costs one extra exp per vector instead of one extra exp per element.
    float2 local_val = make_float2(cuda::std::numeric_limits<float>::lowest(), 0.0f);
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        float vec_max = static_cast<float>(v[0]);
#pragma unroll
        for (int j = 1; j < VecSize; j++) {
            vec_max = max(vec_max, static_cast<float>(v[j]));
        }
        float new_max = max(local_val.x, vec_max);
        float vec_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vec_sum += expf(static_cast<float>(v[j]) - new_max);
        }
        local_val.y = local_val.y * expf(local_val.x - new_max) + vec_sum;
        local_val.x = new_max;
    }
    float2 row_val = blockBroadcast(blockReduce(local_val), &smem);
    const float row_max = row_val.x;
    const float inv_sum = 1.0f / row_val.y;

    // Phase 2: normalize and write output.
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = expf(static_cast<float>(v[j]) - row_max) * inv_sum;
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

    bool use_vec = (num_cols >= static_cast<unsigned int>(VecSize)) && (num_cols % VecSize == 0) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(num_cols) / VecSize);
        onlineSoftmaxKernel<T, VecSize>
            <<<num_rows, block_size, 0, stream>>>(input, output, static_cast<int>(num_cols));
    } else {
        int block_size = alignedBlockSize(static_cast<int>(num_cols));
        onlineSoftmaxKernel<T, 1>
            <<<num_rows, block_size, 0, stream>>>(input, output, static_cast<int>(num_cols));
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t LogSoftmax(const T* input, T* output, unsigned int num_rows, unsigned int num_cols,
                       cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (num_cols >= static_cast<unsigned int>(VecSize)) && (num_cols % VecSize == 0) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(num_cols) / VecSize);
        onlineLogSoftmaxKernel<T, VecSize>
            <<<num_rows, block_size, 0, stream>>>(input, output, static_cast<int>(num_cols));
    } else {
        int block_size = alignedBlockSize(static_cast<int>(num_cols));
        onlineLogSoftmaxKernel<T, 1>
            <<<num_rows, block_size, 0, stream>>>(input, output, static_cast<int>(num_cols));
    }
    return cudaGetLastError();
}

// =============================================================================
// Fused Masked / Biased Softmax
// =============================================================================
//
// One kernel for what a relative-position attention actually runs:
//
//     softmax( cast<T>(scores + bias) where(mask) -> fill )
//
// `bias` and each mask are *broadcast strided views* of the score tensor, so no
// caller has to materialize one.  Zipformer is the reason both halves are
// strided rather than merely broadcast: its relative-position bias is a shifted
// `as_strided` window over an `(H, B, T, 2T-1)` product, and its key-padding
// mask arrives as a `[..., ::ds]` slice of the un-downsampled mask.  Copying
// either into place costs a full pass over a T^2 tensor -- which is the whole
// cost this kernel exists to remove.
//
// The biased score is rounded back to `T` before the reduction.  That is
// deliberate: it makes folding the operands in **numerically free**, so this
// kernel on `(scores, bias, mask)` equals this kernel on the materialized
// `(scores + bias).to(T).masked_fill(mask, fill)`, bit for bit.  Keeping fp32
// precision through the add instead would be *more* accurate and would move
// decoded tokens with nothing to attribute the move to.  It also keeps phase 1
// and phase 2 bit-identical on the recompute path.
//
// Against `Softmax` on that same materialized tensor the two agree to rounding
// but not always bit for bit, and the cause is the vector ladder below rather
// than the fusion: `Softmax` tries only the widest width before dropping to
// scalar, so at a row length that is 4- but not 8-divisible the two group the
// online reduction differently.  Measured at 500: one element in 3000 moves by
// one fp16 ulp, both the same distance from the fp32 reference.
//
// A fully masked row behaves exactly as `masked_fill` + `softmax` does: every
// element equals `fill`, so `row_max == fill`, every `exp` is 1, and the row
// comes out uniform.  A caller that passes `-inf` as the fill gets torch's NaN
// for such a row, on purpose -- diverging here would hide the caller's bug.

//! Shared-memory budget for the row cache, below the 48 KB no-opt-in limit.
constexpr size_t kRowCacheMaxBytes = 32 * 1024;

//! Block ceiling.  A row long enough to want more threads than this is already
//! bandwidth-bound, and 1024 threads would put the kernel over a block's
//! register budget.
constexpr int kMaxBlockSize = 512;

/*!
 * \brief A tensor broadcast against the score tensor, addressed by (row, col).
 *
 * The three leading strides are *per grid axis*, not per tensor axis: the
 * launcher maps the score tensor's leading extents onto `gridDim.{x,y,z}`, so a
 * block reads its own row base straight out of `blockIdx` with no division and
 * no array indexing.  That last part is not a micro-optimization -- indexing a
 * parameter array with a runtime subscript puts the whole parameter block in
 * *local memory*, which measured 20x slower than `oasr.softmax` on Zipformer's
 * shapes before this struct was flattened.
 *
 * A zero stride is a broadcast axis; `ptr == nullptr` means the operand is
 * absent.  Strides are in elements and otherwise arbitrary -- the point of the
 * struct is that a shifted, permuted or step-sliced view needs no copy.
 */
template <typename T>
struct BroadcastView {
    const T* ptr;
    int64_t col_stride;
    int64_t stride_x;
    int64_t stride_y;
    int64_t stride_z;
};

/// Resolves a broadcast view to the row this block owns, or `nullptr` if absent.
template <typename T>
__device__ __forceinline__ const T* resolveRow(const BroadcastView<T>& view) {
    if (view.ptr == nullptr) {
        return nullptr;
    }
    return view.ptr + static_cast<int64_t>(blockIdx.x) * view.stride_x +
           static_cast<int64_t>(blockIdx.y) * view.stride_y +
           static_cast<int64_t>(blockIdx.z) * view.stride_z;
}

/*!
 * \brief Materializes `VecSize` biased, masked scores from one row.
 *
 * Holds the *resolved* row base of each operand rather than the broadcast view
 * it came from.  Used by both phases of the recompute path, so the two agree
 * bit for bit.
 */
template <typename T, int VecSize>
struct MaskedRowLoader {
    const T* row_input;
    const T* bias_row;
    const uint8_t* mask_row;
    const uint8_t* mask2_row;
    int64_t bias_col_stride;
    int64_t mask_col_stride;
    int64_t mask2_col_stride;
    T fill;

    __device__ __forceinline__ void load(int first_col, T (&out)[VecSize]) const {
        oasr::Vec<T, VecSize> v;
        v.load(row_input + first_col);
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            const int col = first_col + j;
            float value = oasr::toFloat(v[j]);
            if (bias_row != nullptr) {
                value += oasr::toFloat(bias_row[col * bias_col_stride]);
            }
            T rounded = oasr::fromFloat<T>(value);
            if (mask_row != nullptr && mask_row[col * mask_col_stride] != 0) {
                rounded = fill;
            }
            if (mask2_row != nullptr && mask2_row[col * mask2_col_stride] != 0) {
                rounded = fill;
            }
            out[j] = rounded;
        }
    }
};

// One block per row, `gridDim` carrying the score tensor's leading extents.
// Phase 1 forms the biased/masked row and accumulates the online (max, sum);
// phase 2 normalizes.  `CacheRow` keeps phase 1's result in shared memory so
// global memory is read exactly once; without it phase 2 re-derives the row,
// which is what a row too wide for the cache falls back to.
template <typename T, int VecSize, bool CacheRow>
__global__ void __launch_bounds__(kMaxBlockSize)
    maskedSoftmaxKernel(const T* __restrict__ input, T* __restrict__ output, BroadcastView<T> bias,
                        BroadcastView<uint8_t> mask, BroadcastView<uint8_t> mask2, float mask_value,
                        int num_cols) {
    extern __shared__ __align__(16) char masked_softmax_smem[];
    T* row_cache = reinterpret_cast<T*>(masked_softmax_smem);

    const int row = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const int64_t row_base = static_cast<int64_t>(row) * num_cols;

    MaskedRowLoader<T, VecSize> loader;
    loader.row_input = input + row_base;
    loader.bias_row = resolveRow(bias);
    loader.mask_row = resolveRow(mask);
    loader.mask2_row = resolveRow(mask2);
    loader.bias_col_stride = bias.col_stride;
    loader.mask_col_stride = mask.col_stride;
    loader.mask2_col_stride = mask2.col_stride;
    loader.fill = oasr::fromFloat<T>(mask_value);

    T* row_output = output + row_base;
    const int vec_num_cols = num_cols / VecSize;

    __shared__ float2 smem;

    float2 local_val = make_float2(cuda::std::numeric_limits<float>::lowest(), 0.0f);
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        T vals[VecSize];
        loader.load(i * VecSize, vals);

        if constexpr (CacheRow) {
            oasr::Vec<T, VecSize> cached;
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                cached[j] = vals[j];
            }
            cached.store(row_cache + i * VecSize);
        }

        float vec_max = oasr::toFloat(vals[0]);
#pragma unroll
        for (int j = 1; j < VecSize; ++j) {
            vec_max = max(vec_max, oasr::toFloat(vals[j]));
        }
        const float new_max = max(local_val.x, vec_max);
        float vec_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            vec_sum += expf(oasr::toFloat(vals[j]) - new_max);
        }
        local_val.y = local_val.y * expf(local_val.x - new_max) + vec_sum;
        local_val.x = new_max;
    }
    float2 row_val = blockBroadcast(blockReduce(local_val), &smem);
    const float row_max = row_val.x;
    const float inv_sum = 1.0f / row_val.y;

    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        T vals[VecSize];
        if constexpr (CacheRow) {
            oasr::Vec<T, VecSize> cached;
            cached.load(row_cache + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                vals[j] = cached[j];
            }
        } else {
            loader.load(i * VecSize, vals);
        }

        oasr::Vec<float, VecSize> normalized;
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            normalized[j] = expf(oasr::toFloat(vals[j]) - row_max) * inv_sum;
        }
        oasr::vecCast<T>(normalized).store(row_output + i * VecSize);
    }
}

// -----------------------------------------------------------------------------
// Vector-width ladder
// -----------------------------------------------------------------------------
//
// The score tensor's row length is an attention extent, so it is whatever the
// audio made it -- 500, 501, 63.  Falling straight from the widest vector to
// scalar on the first odd length costs the common even ones eight loads each,
// so the dispatch walks 8 -> 4 -> 2 -> 1 and stops at the first width that
// divides the row and keeps both pointers aligned.

template <typename T, int VecSize>
inline void launchMaskedSoftmax(const T* input, T* output, const BroadcastView<T>& bias,
                                const BroadcastView<uint8_t>& mask,
                                const BroadcastView<uint8_t>& mask2, dim3 grid, float mask_value,
                                unsigned int num_cols, bool cache_row, size_t cache_bytes,
                                cudaStream_t stream) {
    const int block_size = alignedBlockSize(static_cast<int>(num_cols) / VecSize, kMaxBlockSize);
    if (cache_row) {
        maskedSoftmaxKernel<T, VecSize, true><<<grid, block_size, cache_bytes, stream>>>(
            input, output, bias, mask, mask2, mask_value, static_cast<int>(num_cols));
    } else {
        maskedSoftmaxKernel<T, VecSize, false><<<grid, block_size, 0, stream>>>(
            input, output, bias, mask, mask2, mask_value, static_cast<int>(num_cols));
    }
}

template <typename T, int VecSize>
struct MaskedSoftmaxVecLadder {
    static void run(const T* input, T* output, const BroadcastView<T>& bias,
                    const BroadcastView<uint8_t>& mask, const BroadcastView<uint8_t>& mask2,
                    dim3 grid, float mask_value, unsigned int num_cols, bool cache_row,
                    size_t cache_bytes, cudaStream_t stream) {
        if (num_cols % VecSize == 0 && isAligned<T, VecSize>(input) &&
            isAligned<T, VecSize>(output)) {
            launchMaskedSoftmax<T, VecSize>(input, output, bias, mask, mask2, grid, mask_value,
                                            num_cols, cache_row, cache_bytes, stream);
            return;
        }
        MaskedSoftmaxVecLadder<T, VecSize / 2>::run(input, output, bias, mask, mask2, grid,
                                                    mask_value, num_cols, cache_row, cache_bytes,
                                                    stream);
    }
};

template <typename T>
struct MaskedSoftmaxVecLadder<T, 1> {
    static void run(const T* input, T* output, const BroadcastView<T>& bias,
                    const BroadcastView<uint8_t>& mask, const BroadcastView<uint8_t>& mask2,
                    dim3 grid, float mask_value, unsigned int num_cols, bool cache_row,
                    size_t cache_bytes, cudaStream_t stream) {
        launchMaskedSoftmax<T, 1>(input, output, bias, mask, mask2, grid, mask_value, num_cols,
                                  cache_row, cache_bytes, stream);
    }
};

/*!
 * \brief `softmax(cast<T>(input + bias) where(mask | mask2) -> mask_value)`.
 *
 * \param input  Row-major scores; the softmax runs over `num_cols`.
 * \param output Same layout as `input`.
 * \param bias   Additive broadcast view, or `{nullptr, ...}` for none.
 * \param mask   Boolean broadcast view; true selects `mask_value`.
 * \param mask2  A second, independently broadcast boolean mask.
 * \param grid   The score tensor's leading extents, innermost on `x`.  Its
 *               product is the row count, and the broadcast views' strides are
 *               stated per grid axis.
 */
template <typename T>
cudaError_t MaskedSoftmax(const T* input, T* output, const BroadcastView<T>& bias,
                          const BroadcastView<uint8_t>& mask, const BroadcastView<uint8_t>& mask2,
                          dim3 grid, float mask_value, unsigned int num_cols, cudaStream_t stream) {
    if (grid.x == 0 || grid.y == 0 || grid.z == 0 || num_cols == 0) {
        return cudaSuccess;
    }
    const size_t cache_bytes = static_cast<size_t>(num_cols) * sizeof(T);
    const bool cache_row = cache_bytes <= kRowCacheMaxBytes;

    MaskedSoftmaxVecLadder<T, oasr::VecTypeTrait<T>::VecSize>::run(input, output, bias, mask, mask2,
                                                                   grid, mask_value, num_cols,
                                                                   cache_row, cache_bytes, stream);
    return cudaGetLastError();
}

}  // namespace softmax
}  // namespace oasr
