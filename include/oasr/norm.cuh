// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA normalization kernels — no framework dependencies.
// Includes LayerNorm, RMSNorm, BatchNorm1D, GroupNorm, AddLayerNorm,
// AddRMSNorm, and fused norm+activation variants.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <oasr/common/math.h>
#include <oasr/common/reduction.h>
#include <oasr/common/types.h>
#include <oasr/common/utils.h>
#include <oasr/common/vec_dtypes.h>
#include <oasr/quant/quantize.cuh>

namespace oasr {
namespace norm {
using namespace oasr::reduction;

// =============================================================================
// Per-token 8-bit output (the kQuantOut variants of the LayerNorm kernels)
//
// A LayerNorm whose only consumers are 8-bit GEMMs can hand them the quantized
// activation directly.  The contract is bit-exactness with the unfused pair:
//
//     LayerNormQuant(x)  ==  QuantizePerToken(LayerNorm(x))
//
// so the kQuantOut variant is the *same* kernel template with only the sink of
// the normalize pass swapped: every normalized value is rounded to T exactly as
// the store would round it, held until the row's amax is known, and the amax is
// taken over those T values.  The quantize then reads the held values back --
// not a recomputation, so it cannot drift from the amax -- and applies
// quantize.cuh's ``QuantTraits`` (int8, or fp8 e4m3 when ``QuantOutParams::fp8``)
// with its ``__fdiv_rn`` reciprocal and scale.  The format is a runtime flag
// read once per row, not a template parameter: it changes only the final store,
// and a second instantiation would double every quant-out kernel.  The
// mean/variance passes, the block size and the vector/scalar choice are
// untouched, which is what keeps the fp32 normalized values identical to the
// unfused kernel's (and the affine's rounding is spelled out in source, so the
// optimizer cannot round the two instantiations differently).
// =============================================================================

struct QuantOutParams {
    void* q;       // 8-bit rows (int8, or e4m3 when ``fp8``), laid out like the input
    float* scale;  // one fp32 scale per row
    float eps;     // floor on a row's amax (quantize.cuh's eps)
    bool fp8;      // e4m3 (Q = 448) instead of int8 (Q = 127)
};

// Where the normalize pass leaves a row's T values until its amax is known.
//
// A thread only ever reads back the vectors it produced, so no value crosses
// threads.  The launcher sizes the block to the row (one vector per thread up
// to 1024 vectors), so the first vector -- usually the only one -- stays in
// registers; any further ones go to dynamic shared memory, which the launcher
// sizes to exactly that overflow (``quantStageBytes``).
template <typename T, int VecSize>
struct QuantRowSink {
    using VecT = oasr::Vec<T, VecSize>;

    // The first vector as f32(T value): converted once, for the amax and the
    // quantize alike.  Further vectors are staged as T.
    float held[VecSize];
    T* overflow;
    float local_max = 0.0f;

    __device__ __forceinline__ QuantRowSink() {
        extern __shared__ __align__(16) unsigned char norm_quant_stage[];
        overflow = reinterpret_cast<T*>(norm_quant_stage);
    }

    // Keep vector ``i`` (already rounded to T) and fold it into the amax, which
    // is over f32(T value) -- exactly what the unfused quantize reads back.
    __device__ __forceinline__ void put(int i, const VecT& y) {
        const bool first = i == static_cast<int>(threadIdx.x);
        if (!first) {
            y.store(overflow + (i - static_cast<int>(blockDim.x)) * VecSize);
        }
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            const float f = static_cast<float>(y[j]);
            if (first) {
                held[j] = f;
            }
            local_max = fmaxf(local_max, fabsf(f));
        }
    }

    __device__ __forceinline__ void get(int i, float (&f)[VecSize]) const {
        if (i == static_cast<int>(threadIdx.x)) {
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                f[j] = held[j];
            }
            return;
        }
        VecT y;
        y.load(overflow + (i - static_cast<int>(blockDim.x)) * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            f[j] = static_cast<float>(y[j]);
        }
    }

    // The row's amax, its scale, and the 8-bit store of the held row.
    __device__ __forceinline__ void finish(int vec_hidden_size, int hidden_size,
                                          const QuantOutParams& quant) const {
        __shared__ float warp_max[WARP_SIZE];
        const int64_t row = blockIdx.x;

        // The max is exact in any order, so it needs no tree: every lane gets
        // its warp's max by shuffles, and past one warp each warp publishes it
        // and every thread folds the published values itself -- one barrier,
        // where a reduce-then-broadcast costs two (the rows are short enough
        // that barrier latency, not bandwidth, sets the time).
        float amax = warpReduceMax(local_max);
        if (blockDim.x > WARP_SIZE) {
            if (threadIdx.x % WARP_SIZE == 0) {
                warp_max[threadIdx.x / WARP_SIZE] = amax;
            }
            __syncthreads();
            const int num_warps = blockDim.x / WARP_SIZE;
            amax = warp_max[0];
            for (int w = 1; w < num_warps; ++w) {
                amax = fmaxf(amax, warp_max[w]);
            }
        }
        amax = fmaxf(amax, quant.eps);
        const float qmax = quant.fp8 ? quant::QuantTraits<__nv_fp8_e4m3>::kMax
                                     : quant::QuantTraits<int8_t>::kMax;
        if (threadIdx.x == 0) {
            quant.scale[row] = __fdiv_rn(amax, qmax);
        }
        const float inv = __fdiv_rn(qmax, amax);

        // Both formats are one byte; the bit patterns are written as such.
        uint8_t* row_q = static_cast<uint8_t*>(quant.q) + row * hidden_size;
        // An 8-wide row of bytes is one 8-byte store when the destination allows
        // it; the 8-bit tensor's alignment is independent of the input's, so it is
        // checked here rather than folded into the launcher's vector choice
        // (which must stay the unfused LayerNorm's).
        const bool q_vec = VecSize == 8 && (reinterpret_cast<uintptr_t>(row_q) % 8 == 0);
        for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
            float y[VecSize];
            get(i, y);
            alignas(8) uint8_t q[VecSize];
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                q[j] = quant.fp8 ? quant::QuantTraits<__nv_fp8_e4m3>::quantize(y[j], inv).__x
                                 : static_cast<uint8_t>(
                                       quant::QuantTraits<int8_t>::quantize(y[j], inv));
            }
            if (q_vec) {
                *reinterpret_cast<int2*>(row_q + i * VecSize) = *reinterpret_cast<const int2*>(q);
            } else {
#pragma unroll
                for (int j = 0; j < VecSize; ++j) {
                    row_q[i * VecSize + j] = q[j];
                }
            }
        }
    }
};

// Dynamic shared memory for the vectors a thread cannot keep in registers:
// every vector past the block's first round.
inline size_t quantStageBytes(int vec_hidden_size, int block_size, int vec_bytes) {
    const int overflow = vec_hidden_size > block_size ? vec_hidden_size - block_size : 0;
    return static_cast<size_t>(overflow) * static_cast<size_t>(vec_bytes);
}

// =============================================================================
// LayerNorm Kernel
// =============================================================================

template <typename T, int VecSize, bool kQuantOut = false>
__global__ void layerNormKernel(const T* __restrict__ input, T* __restrict__ output,
                                const T* __restrict__ weight, const T* __restrict__ bias,
                                int hidden_size, float eps, QuantOutParams quant) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem[2];

    // Phase 1: Compute mean using vectorized loads
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum += oasr::vecSum(v);
    }

    float mean =
        blockBroadcast(blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: Compute variance using vectorized loads
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v[j]) - mean;
            local_var = std::fmaf(diff, diff, local_var);
        }
    }

    float inv_std = rsqrtf(
        blockBroadcast(blockReduceSum(local_var) / static_cast<float>(hidden_size), &smem[1]) +
        eps);

    // kQuantOut: hold the T-rounded row instead of storing it.
    [[maybe_unused]] QuantRowSink<T, VecSize> sink;

    // Phase 3: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = (static_cast<float>(v_in[j]) - mean) * inv_std;
        }

        // The affine, with its rounding written down: ``n * w + b`` is one FMA.
        // Left to the optimizer, the multiply and the add are contracted only
        // when nvcc unswitches the bias branch, a per-instantiation decision it
        // took for the plain kernel and not for the kQuantOut one -- which then
        // differed from it by an ulp.  The FMA is what this kernel compiled to
        // before the contraction was spelled out.
        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] = fmaf(vals[j], static_cast<float>(v_weight[j]),
                               static_cast<float>(v_bias[j]));
            }
        } else {
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] *= static_cast<float>(v_weight[j]);
            }
        }

        if constexpr (kQuantOut) {
            sink.put(i, oasr::vecCast<T>(vals));
        } else {
            oasr::vecCast<T>(vals).store(row_output + i * VecSize);
        }
    }

    // Phase 4 (kQuantOut): per-token int8 of the held row.
    if constexpr (kQuantOut) {
        sink.finish(vec_hidden_size, hidden_size, quant);
    }
}

// =============================================================================
// RMSNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void rmsNormKernel(const T* __restrict__ input, T* __restrict__ output,
                              const T* __restrict__ weight, const T* __restrict__ bias,
                              int hidden_size, float eps) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem;

    // Phase 1: Compute mean of squares using vectorized loads
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum_sq += oasr::vecSumSquares(v);
    }

    float inv_rms = rsqrtf(
        blockBroadcast(blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size), &smem) +
        eps);

    // Phase 2: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v_in[j]) * inv_rms * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// BatchNorm1D Kernel (inference mode)
// =============================================================================

template <typename T, int VecSize>
__global__ void batchNorm1DKernel(const T* __restrict__ input, T* __restrict__ output,
                                  const T* __restrict__ weight, const T* __restrict__ bias,
                                  const T* __restrict__ running_mean,
                                  const T* __restrict__ running_var, int batch_size, int seq_len,
                                  int channels, float eps) {
    using VecT = oasr::Vec<T, VecSize>;

    const int total_elements = batch_size * seq_len * channels;

    // Each thread processes VecSize elements at a time
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / VecSize;
         idx += blockDim.x * gridDim.x) {
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % channels;

        VecT v_in, v_weight, v_bias, v_mean, v_var;
        v_in.load(input + flat_idx);
        v_weight.load(weight + c_start);
        v_bias.load(bias + c_start);
        v_mean.load(running_mean + c_start);
        v_var.load(running_var + c_start);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = (static_cast<float>(v_in[j]) - static_cast<float>(v_mean[j])) *
                          rsqrtf(static_cast<float>(v_var[j]) + eps) *
                          static_cast<float>(v_weight[j]) +
                      static_cast<float>(v_bias[j]);
        }

        oasr::vecCast<T>(vals).store(output + flat_idx);
    }
}

// =============================================================================
// GroupNorm Kernel
// =============================================================================

// One block per row.  Group g is reduced by threads
// [g * threads_per_group, (g + 1) * threads_per_group), a power of two (the
// shuffle reduction pairs lanes by xor); threads past the last group, which
// fill out the last warp, carry zeros through the shuffles and write nothing.
template <typename T, int VecSize>
__global__ void groupNormKernel(const T* __restrict__ input, T* __restrict__ output,
                                const T* __restrict__ weight, const T* __restrict__ bias,
                                int channels, int num_groups, int threads_per_group, float eps) {
    using VecT = oasr::Vec<T, VecSize>;

    int channels_per_group = channels / num_groups;
    float inv_channels_per_group = 1.0f / static_cast<float>(channels_per_group);

    const T* row = input + static_cast<int64_t>(blockIdx.x) * channels;
    T* out_row = output + static_cast<int64_t>(blockIdx.x) * channels;

    int my_group = threadIdx.x / threads_per_group;
    int local_tid = threadIdx.x % threads_per_group;
    const bool in_a_group = my_group < num_groups;

    int lane = threadIdx.x & (oasr::WARP_SIZE - 1);

    // Shared memory: [group_mean | group_inv_std | warp_buf (inter-warp case)]
    extern __shared__ float smem[];
    float* group_mean = smem;
    float* group_inv_std = smem + num_groups;

    // ---- Phase 1: thread-local accumulation for assigned group (vectorized) ----
    float psum = 0.0f, psq = 0.0f;
    if (in_a_group) {
        const T* gptr = row + my_group * channels_per_group;
        int vec_channels_per_group = channels_per_group / VecSize;
        for (int vec_i = local_tid; vec_i < vec_channels_per_group; vec_i += threads_per_group) {
            VecT v;
            v.load(gptr + vec_i * VecSize);
            psum += oasr::vecSum(v);
            psq += oasr::vecSumSquares(v);
        }
        for (int i = vec_channels_per_group * VecSize + local_tid; i < channels_per_group;
             i += threads_per_group) {
            float v = static_cast<float>(gptr[i]);
            psum += v;
            psq += v * v;
        }
    }

    // ---- Phase 2: Tree reduction ----
    int seg = (threads_per_group < oasr::WARP_SIZE) ? threads_per_group : oasr::WARP_SIZE;
    for (int off = seg >> 1; off > 0; off >>= 1) {
        psum += __shfl_xor_sync(0xffffffff, psum, off);
        psq += __shfl_xor_sync(0xffffffff, psq, off);
    }

    if (threads_per_group <= oasr::WARP_SIZE) {
        if (in_a_group && local_tid == 0) {
            float mean = psum * inv_channels_per_group;
            group_mean[my_group] = mean;
            group_inv_std[my_group] = rsqrtf(psq * inv_channels_per_group - mean * mean + eps);
        }
    } else {
        int warps_per_group = threads_per_group / oasr::WARP_SIZE;
        int warp_idx = local_tid / oasr::WARP_SIZE;
        float* wbuf = smem + 2 * num_groups;

        if (in_a_group && lane == 0) {
            int bi = my_group * warps_per_group + warp_idx;
            wbuf[bi] = psum;
            wbuf[num_groups * warps_per_group + bi] = psq;
        }
        __syncthreads();

        if (in_a_group && local_tid == 0) {
            float s = 0.0f, sq = 0.0f;
            int base = my_group * warps_per_group;
            int sq_base = num_groups * warps_per_group + base;
            for (int w = 0; w < warps_per_group; ++w) {
                s += wbuf[base + w];
                sq += wbuf[sq_base + w];
            }
            float mean = s * inv_channels_per_group;
            group_mean[my_group] = mean;
            group_inv_std[my_group] = rsqrtf(sq * inv_channels_per_group - mean * mean + eps);
        }
    }
    __syncthreads();

    // ---- Phase 3: normalize and apply per-channel affine (vectorized where possible) ----
    int vec_channels = channels / VecSize;

    for (int vec_c = threadIdx.x; vec_c < vec_channels; vec_c += blockDim.x) {
        int c = vec_c * VecSize;
        int g = c / channels_per_group;
        VecT v_in, v_weight, v_bias;
        v_in.load(row + c);
        v_weight.load(weight + c);
        v_bias.load(bias + c);
        float mean_g = group_mean[g];
        float inv_std_g = group_inv_std[g];
        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = (static_cast<float>(v_in[j]) - mean_g) * inv_std_g *
                          static_cast<float>(v_weight[j]) +
                      static_cast<float>(v_bias[j]);
        }
        oasr::vecCast<T>(vals).store(out_row + c);
    }
}

// =============================================================================
// Add + LayerNorm Fused Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void addLayerNormKernel(const T* __restrict__ input, const T* __restrict__ residual,
                                   T* __restrict__ output, const T* __restrict__ weight,
                                   const T* __restrict__ bias, int hidden_size, float eps,
                                   float alpha) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    const T* row_residual = residual + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem[2];

    // Phase 1: Compute mean using vectorized loads
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);

#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            local_sum += static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]);
        }
    }

    float mean =
        blockBroadcast(blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: Compute variance using vectorized loads
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);

#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]) - mean;
            local_var = std::fmaf(diff, diff, local_var);
        }
    }

    float inv_std = rsqrtf(
        blockBroadcast(blockReduceSum(local_var) / static_cast<float>(hidden_size), &smem[1]) +
        eps);

    // Phase 3: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_weight;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float sum = static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]);
            vals[j] = (sum - mean) * inv_std * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Fused Add + LayerNorm with residual passthrough
//
// Computes, per row:
//     s        = residual + alpha * input          (stored to residual_out)
//     output   = LayerNorm(s) * weight + bias
// ``residual_out`` carries the *unnormalized* sum so the caller can chain it
// as the residual for the next pre-norm sub-layer (Conformer / Transformer
// pre-norm pattern). The normalization math keeps the sum in float; only the
// residual output and normalized output stores round to T.  kQuantOut swaps
// the normalized store for the per-token int8 sink described at the top of
// this file; ``residual_out`` is stored exactly as before.
// =============================================================================

template <typename T, int VecSize, bool kQuantOut = false>
__global__ void addLayerNormResidualKernel(const T* __restrict__ input,
                                           const T* __restrict__ residual, T* __restrict__ output,
                                           T* __restrict__ residual_out,
                                           const T* __restrict__ weight, const T* __restrict__ bias,
                                           int hidden_size, float eps, float alpha,
                                           QuantOutParams quant) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    const T* row_residual = residual + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    T* row_res_out = residual_out + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem[2];

    // Phase 0: materialise the carried residual. The normalization passes
    // independently recompute the sum in float rather than reading this
    // rounded T value back.
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_s;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            v_s[j] =
                static_cast<T>(static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]));
        }
        v_s.store(row_res_out + i * VecSize);
    }

    // Phase 1: mean of the fp32 sum.
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            local_sum += static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]);
        }
    }
    float mean =
        blockBroadcast(blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: variance of the fp32 sum.
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]) - mean;
            local_var = std::fmaf(diff, diff, local_var);
        }
    }
    float inv_std = rsqrtf(
        blockBroadcast(blockReduceSum(local_var) / static_cast<float>(hidden_size), &smem[1]) +
        eps);

    // kQuantOut: hold the T-rounded row instead of storing it.
    [[maybe_unused]] QuantRowSink<T, VecSize> sink;

    // Phase 3: normalize + affine.
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_weight;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        v_weight.load(weight + i * VecSize);

        // The affine, with its rounding written down -- see layerNormKernel for
        // why.  Each spelling is what that build compiled to before: the scalar
        // kernel contracted ``n * w + b`` into one FMA, the vector kernel
        // multiplied by the weight and then rounded the bias add on its own
        // (``__fadd_rn`` is never contracted).
        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float sum = static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]);
            vals[j] = (sum - mean) * inv_std;
            if constexpr (VecSize > 1) {
                vals[j] *= static_cast<float>(v_weight[j]);
            }
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                if constexpr (VecSize == 1) {
                    vals[j] = fmaf(vals[j], static_cast<float>(v_weight[j]),
                                   static_cast<float>(v_bias[j]));
                } else {
                    vals[j] = __fadd_rn(vals[j], static_cast<float>(v_bias[j]));
                }
            }
        } else if constexpr (VecSize == 1) {
            vals[0] *= static_cast<float>(v_weight[0]);
        }

        if constexpr (kQuantOut) {
            sink.put(i, oasr::vecCast<T>(vals));
        } else {
            oasr::vecCast<T>(vals).store(row_output + i * VecSize);
        }
    }

    // Phase 4 (kQuantOut): per-token int8 of the held row.
    if constexpr (kQuantOut) {
        sink.finish(vec_hidden_size, hidden_size, quant);
    }
}

// =============================================================================
// Add + RMSNorm Fused Kernels
// =============================================================================

template <typename T, int VecSize>
__global__ void addRmsNormKernel(const T* __restrict__ input, const T* __restrict__ residual,
                                 T* __restrict__ output, const T* __restrict__ weight,
                                 const T* __restrict__ bias, int hidden_size, float eps,
                                 float alpha) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    const T* row_residual = residual + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem;

    // Keep the residual sum in float throughout the RMS reduction.
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float sum = static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]);
            local_sum_sq = std::fmaf(sum, sum, local_sum_sq);
        }
    }
    float inv_rms = rsqrtf(
        blockBroadcast(blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size), &smem) +
        eps);

    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_weight;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> values;
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float sum = static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]);
            values[j] = sum * inv_rms * static_cast<float>(v_weight[j]);
        }
        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                values[j] += static_cast<float>(v_bias[j]);
            }
        }
        oasr::vecCast<T>(values).store(row_output + i * VecSize);
    }
}

template <typename T, int VecSize>
__global__ void addRmsNormResidualKernel(const T* __restrict__ input,
                                         const T* __restrict__ residual, T* __restrict__ output,
                                         T* __restrict__ residual_out, const T* __restrict__ weight,
                                         const T* __restrict__ bias, int hidden_size, float eps,
                                         float alpha) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    const T* row_residual = residual + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    T* row_residual_out = residual_out + row_idx * hidden_size;
    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem;

    // Carry the unnormalized residual stream forward without a separate add
    // launch. The normalization passes recompute the sum in float rather than
    // reading the rounded T residual output back.
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_sum;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            v_sum[j] =
                static_cast<T>(static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]));
        }
        v_sum.store(row_residual_out + i * VecSize);
    }

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float sum = static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]);
            local_sum_sq = std::fmaf(sum, sum, local_sum_sq);
        }
    }
    float inv_rms = rsqrtf(
        blockBroadcast(blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size), &smem) +
        eps);

    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_weight;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> values;
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
            float sum = static_cast<float>(v_res[j]) + alpha * static_cast<float>(v_in[j]);
            values[j] = sum * inv_rms * static_cast<float>(v_weight[j]);
        }
        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                values[j] += static_cast<float>(v_bias[j]);
            }
        }
        oasr::vecCast<T>(values).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Activation Dispatch Helper
// =============================================================================

template <typename Fn>
void dispatchActivation(ActivationType act, Fn&& fn) {
    switch (act) {
        case ActivationType::RELU:
            fn(oasr::ReluActivation{});
            break;
        case ActivationType::GELU:
            fn(oasr::GeluActivation{});
            break;
        case ActivationType::SWISH:
            fn(oasr::SwishActivation{});
            break;
        default:
            break;
    }
}

// =============================================================================
// Fused LayerNorm + Activation Kernel
// =============================================================================

template <typename T, int VecSize, typename ActFn>
__global__ void layerNormActKernel(const T* __restrict__ input, T* __restrict__ output,
                                   const T* __restrict__ weight, const T* __restrict__ bias,
                                   int hidden_size, float eps) {
    using VecT = oasr::Vec<T, VecSize>;
    ActFn act;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem[2];

    // Phase 1: Compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum += oasr::vecSum(v);
    }

    float mean =
        blockBroadcast(blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v[j]) - mean;
            local_var = std::fmaf(diff, diff, local_var);
        }
    }

    float inv_std = rsqrtf(
        blockBroadcast(blockReduceSum(local_var) / static_cast<float>(hidden_size), &smem[1]) +
        eps);

    // Phase 3: Normalize, scale, and apply activation
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] =
                (static_cast<float>(v_in[j]) - mean) * inv_std * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = act(vals[j]);
        }

        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Fused RMSNorm + Activation Kernel
// =============================================================================

template <typename T, int VecSize, typename ActFn>
__global__ void rmsNormActKernel(const T* __restrict__ input, T* __restrict__ output,
                                 const T* __restrict__ weight, const T* __restrict__ bias,
                                 int hidden_size, float eps) {
    using VecT = oasr::Vec<T, VecSize>;
    ActFn act;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem;

    // Phase 1: Compute mean of squares
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum_sq += oasr::vecSumSquares(v);
    }

    float inv_rms = rsqrtf(
        blockBroadcast(blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size), &smem) +
        eps);

    // Phase 2: Normalize, scale, and apply activation
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v_in[j]) * inv_rms * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = act(vals[j]);
        }

        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Fused BatchNorm + Activation Kernel (vectorized)
// =============================================================================

template <typename T, int VecSize, typename ActFn>
__global__ void batchNormActKernel(const T* __restrict__ input, T* __restrict__ output,
                                   const T* __restrict__ weight, const T* __restrict__ bias,
                                   const T* __restrict__ running_mean,
                                   const T* __restrict__ running_var, int batch_size, int seq_len,
                                   int channels, float eps) {
    using VecT = oasr::Vec<T, VecSize>;
    ActFn act;

    int total_elements = batch_size * seq_len * channels;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / VecSize;
         idx += blockDim.x * gridDim.x) {
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % channels;

        VecT v_in, v_weight, v_bias, v_mean, v_var;
        v_in.load(input + flat_idx);
        v_weight.load(weight + c_start);
        v_bias.load(bias + c_start);
        v_mean.load(running_mean + c_start);
        v_var.load(running_var + c_start);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = act((static_cast<float>(v_in[j]) - static_cast<float>(v_mean[j])) *
                              rsqrtf(static_cast<float>(v_var[j]) + eps) *
                              static_cast<float>(v_weight[j]) +
                          static_cast<float>(v_bias[j]));
        }

        oasr::vecCast<T>(vals).store(output + flat_idx);
    }
}

// =============================================================================
// CMVN (Cepstral Mean and Variance Normalization) Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void cmvnKernel(const T* __restrict__ input, T* __restrict__ output,
                           const T* __restrict__ mean, const T* __restrict__ istd, int num_rows,
                           int num_cols) {
    using VecT = oasr::Vec<T, VecSize>;

    const int total_elements = num_rows * num_cols;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / VecSize;
         idx += blockDim.x * gridDim.x) {
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % num_cols;

        VecT v_in, v_mean, v_istd;
        v_in.load(input + flat_idx);
        v_mean.load(mean + c_start);
        v_istd.load(istd + c_start);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = (static_cast<float>(v_in[j]) - static_cast<float>(v_mean[j])) *
                      static_cast<float>(v_istd[j]);
        }

        oasr::vecCast<T>(vals).store(output + flat_idx);
    }
}

// =============================================================================
// BiasNorm Kernel (Zipformer)
// =============================================================================
//
// scales = mean((x - bias)^2, dim=-1, keepdim=True) ** -0.5 * exp(log_scale)
// output = x * scales
//
// ``bias`` is a per-channel (last-dim) vector; ``log_scale`` is a 1-element
// tensor.  No eps term (matches icefall's inference-time BiasNorm).

template <typename T, int VecSize>
__global__ void biasNormKernel(const T* __restrict__ input, T* __restrict__ output,
                               const T* __restrict__ bias, const T* __restrict__ log_scale,
                               int hidden_size) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem;

    // Phase 1: mean of (x - bias)^2 over the row (bias is shared across rows).
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_bias;
        v_in.load(row_input + i * VecSize);
        v_bias.load(bias + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v_in[j]) - static_cast<float>(v_bias[j]);
            local_sum_sq = std::fmaf(diff, diff, local_sum_sq);
        }
    }

    float mean_sq =
        blockBroadcast(blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size), &smem);
    float scale = rsqrtf(mean_sq) * expf(static_cast<float>(log_scale[0]));

    // Phase 2: output = x * scale.
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in;
        v_in.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v_in[j]) * scale;
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Typed Launcher Functions (raw pointer interface, returning cudaError_t)
// =============================================================================

// Whether every operand the vector path touches sits on a vector boundary.  It
// loads the per-channel parameters (weight, bias, running statistics)
// `VecSize` elements at a time just as it does the rows, so a parameter that is
// a slice of a larger buffer faults with `misaligned address` unless it is
// checked too.  An absent (null) operand passes.
template <typename T, int VecSize, typename... Ptrs>
inline bool allAligned(Ptrs... ptrs) {
    return (isAligned<T, VecSize>(ptrs) && ...);
}

// ---- LayerNorm ----

template <typename T>
cudaError_t LayerNorm(const T* input, const T* weight, const T* bias, T* output,
                      unsigned int num_rows, unsigned int hidden_size, float eps,
                      cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) &&
                   allAligned<T, VecSize>(input, output, weight, bias);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        layerNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, output, weight, bias, static_cast<int>(hidden_size), eps, QuantOutParams{});
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        layerNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, output, weight, bias, static_cast<int>(hidden_size), eps, QuantOutParams{});
    }
    return cudaGetLastError();
}

// ---- Quant-out launch ----

// One block per row at LayerNorm's block size (the block size is part of the
// numerics: it fixes the reduction order), plus the shared memory for the
// vectors a thread cannot keep in registers.  That overflow exists only past
// 1024 vectors a row; past the 48 KiB a kernel gets without asking (less what
// the reductions declare statically) it opts in -- a 16-bit row wider than
// ~28K elements on the vector path, ~21K on the scalar one.
template <typename Kernel, typename... Args>
inline cudaError_t launchQuantOut(Kernel kernel, unsigned int num_rows, int vec_hidden_size,
                                  int vec_bytes, cudaStream_t stream, Args... args) {
    constexpr size_t kDefaultDynamicSmem = 40 * 1024;
    const int block_size = alignedBlockSize(vec_hidden_size);
    const size_t stage_bytes = quantStageBytes(vec_hidden_size, block_size, vec_bytes);
    if (stage_bytes > kDefaultDynamicSmem) {
        cudaError_t err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(stage_bytes));
        if (err != cudaSuccess) {
            return err;
        }
    }
    kernel<<<num_rows, block_size, stage_bytes, stream>>>(args...);
    return cudaGetLastError();
}

// ---- LayerNormQuant ----

// ``QuantizePerToken(LayerNorm(x))`` in one launch: ``out_q`` 8-bit (int8, or e4m3
// when ``fp8``) shaped like the input, ``out_scale`` one fp32 per row, ``qeps`` the
// amax floor.
template <typename T>
cudaError_t LayerNormQuant(const T* input, const T* weight, const T* bias, void* out_q,
                           float* out_scale, unsigned int num_rows, unsigned int hidden_size,
                           float eps, float qeps, bool fp8, cudaStream_t stream) {
    if (num_rows == 0) {
        return cudaSuccess;
    }
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    // The vector/scalar choice fixes each thread's accumulation order, hence the
    // mean and variance bits, so it must be LayerNorm's for the same input.
    // LayerNorm also requires its T output aligned, which the freshly allocated
    // output of the unfused pair always is.
    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) && allAligned<T, VecSize>(input, weight, bias);

    const QuantOutParams quant{out_q, out_scale, qeps, fp8};
    const int hidden = static_cast<int>(hidden_size);
    T* no_output = nullptr;
    if (use_vec) {
        return launchQuantOut(layerNormKernel<T, VecSize, true>, num_rows, hidden / VecSize,
                              static_cast<int>(VecSize * sizeof(T)), stream, input, no_output,
                              weight, bias, hidden, eps, quant);
    }
    return launchQuantOut(layerNormKernel<T, 1, true>, num_rows, hidden,
                          static_cast<int>(sizeof(T)), stream, input, no_output, weight, bias,
                          hidden, eps, quant);
}

// ---- BiasNorm ----

template <typename T>
cudaError_t BiasNorm(const T* input, const T* bias, const T* log_scale, T* output,
                     unsigned int num_rows, unsigned int hidden_size, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) && isAligned<T, VecSize>(input) &&
                   isAligned<T, VecSize>(output) && isAligned<T, VecSize>(bias);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        biasNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, output, bias, log_scale, static_cast<int>(hidden_size));
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        biasNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(input, output, bias, log_scale,
                                                                  static_cast<int>(hidden_size));
    }
    return cudaGetLastError();
}

// ---- RMSNorm ----

template <typename T>
cudaError_t RMSNorm(const T* input, const T* weight, const T* bias, T* output,
                    unsigned int num_rows, unsigned int hidden_size, float eps,
                    cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) &&
                   allAligned<T, VecSize>(input, output, weight, bias);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        rmsNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, output, weight, bias, static_cast<int>(hidden_size), eps);
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        rmsNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, output, weight, bias, static_cast<int>(hidden_size), eps);
    }
    return cudaGetLastError();
}

// ---- BatchNorm1D ----

template <typename T>
cudaError_t BatchNorm1D(const T* input, const T* weight, const T* bias, const T* running_mean,
                        const T* running_var, T* output, unsigned int batch_size,
                        unsigned int seq_len, unsigned int channels, float eps,
                        cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    int total_elements = static_cast<int>(batch_size * seq_len * channels);
    int block_size = 256;

    bool use_vec = (channels >= static_cast<unsigned int>(VecSize)) && (channels % VecSize == 0) &&
                   allAligned<T, VecSize>(input, output, weight, bias, running_mean, running_var);

    if (use_vec) {
        int grid_size = (total_elements / VecSize + block_size - 1) / block_size;
        batchNorm1DKernel<T, VecSize><<<grid_size, block_size, 0, stream>>>(
            input, output, weight, bias, running_mean, running_var, static_cast<int>(batch_size),
            static_cast<int>(seq_len), static_cast<int>(channels), eps);
    } else {
        int grid_size = (total_elements + block_size - 1) / block_size;
        batchNorm1DKernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            input, output, weight, bias, running_mean, running_var, static_cast<int>(batch_size),
            static_cast<int>(seq_len), static_cast<int>(channels), eps);
    }
    return cudaGetLastError();
}

// ---- GroupNorm ----

template <typename T>
cudaError_t GroupNorm(const T* input, const T* weight, const T* bias, T* output,
                      unsigned int batch_size, unsigned int seq_len, unsigned int channels,
                      unsigned int num_groups, float eps, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    // 1 <= num_groups <= 1024 dividing channels (the launcher checks it).
    int channels_per_group = static_cast<int>(channels / num_groups);
    const int64_t num_blocks = static_cast<int64_t>(batch_size) * seq_len;
    if (num_blocks == 0) {
        return cudaSuccess;  // a zero-block grid is a launch error, not a no-op
    }
    // Each group takes the largest power of two of the row's threads that fits,
    // and the block is rounded up to whole warps for the full-mask shuffles.
    // The threads per group used to be `block / num_groups`, which is neither
    // for a count that does not divide the block: 5 groups of 16 channels got
    // 19 threads each, so the shuffles summed across groups, and the 96th
    // thread reduced a sixth group from the next row into group 0's statistics.
    // Wherever that division was exact and a power of two, this picks the same
    // geometry, so those results are unchanged.
    const int row_threads = alignedBlockSize(static_cast<int>(channels));
    int threads_per_group = 1;
    while (threads_per_group * 2 * static_cast<int>(num_groups) <= row_threads) {
        threads_per_group *= 2;
    }
    const int block_size =
        (threads_per_group * static_cast<int>(num_groups) + oasr::WARP_SIZE - 1) /
        oasr::WARP_SIZE * oasr::WARP_SIZE;
    int num_warps = block_size / oasr::WARP_SIZE;
    size_t smem_bytes = sizeof(float) * (2 * num_groups + 2 * static_cast<unsigned int>(num_warps));

    bool use_vec = (static_cast<int>(channels_per_group) >= VecSize) &&
                   (channels_per_group % VecSize == 0) && isAligned<T, VecSize>(input) &&
                   isAligned<T, VecSize>(output) && isAligned<T, VecSize>(weight) &&
                   isAligned<T, VecSize>(bias);

    const auto grid = static_cast<unsigned int>(num_blocks);
    if (use_vec) {
        groupNormKernel<T, VecSize><<<grid, block_size, smem_bytes, stream>>>(
            input, output, weight, bias, static_cast<int>(channels), static_cast<int>(num_groups),
            threads_per_group, eps);
    } else {
        groupNormKernel<T, 1><<<grid, block_size, smem_bytes, stream>>>(
            input, output, weight, bias, static_cast<int>(channels), static_cast<int>(num_groups),
            threads_per_group, eps);
    }
    return cudaGetLastError();
}

// ---- AddLayerNorm ----

template <typename T>
cudaError_t AddLayerNorm(const T* input, const T* residual, const T* weight, const T* bias,
                         T* output, unsigned int num_rows, unsigned int hidden_size, float eps,
                         float alpha, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) &&
                   allAligned<T, VecSize>(input, residual, output, weight, bias);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        addLayerNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, residual, output, weight, bias, static_cast<int>(hidden_size), eps, alpha);
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        addLayerNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, residual, output, weight, bias, static_cast<int>(hidden_size), eps, alpha);
    }
    return cudaGetLastError();
}

// ---- AddLayerNormResidual ----

template <typename T>
cudaError_t AddLayerNormResidual(const T* input, const T* residual, const T* weight, const T* bias,
                                 T* output, T* residual_out, unsigned int num_rows,
                                 unsigned int hidden_size, float eps, float alpha,
                                 cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) &&
                   allAligned<T, VecSize>(input, residual, output, residual_out, weight, bias);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        addLayerNormResidualKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, residual, output, residual_out, weight, bias, static_cast<int>(hidden_size), eps,
            alpha, QuantOutParams{});
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        addLayerNormResidualKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, residual, output, residual_out, weight, bias, static_cast<int>(hidden_size), eps,
            alpha, QuantOutParams{});
    }
    return cudaGetLastError();
}

// ---- AddLayerNormResidualQuant ----

// ``(QuantizePerToken(out), residual_out)`` for ``(out, residual_out) =
// AddLayerNormResidual(...)``, in one launch.
template <typename T>
cudaError_t AddLayerNormResidualQuant(const T* input, const T* residual, const T* weight,
                                      const T* bias, void* out_q, float* out_scale,
                                      T* residual_out, unsigned int num_rows,
                                      unsigned int hidden_size, float eps, float alpha, float qeps,
                                      bool fp8, cudaStream_t stream) {
    if (num_rows == 0) {
        return cudaSuccess;
    }
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    // AddLayerNormResidual's vector choice, less its (always aligned, freshly
    // allocated) T output -- see LayerNormQuant.
    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) &&
                   allAligned<T, VecSize>(input, residual, residual_out, weight, bias);

    const QuantOutParams quant{out_q, out_scale, qeps, fp8};
    const int hidden = static_cast<int>(hidden_size);
    T* no_output = nullptr;
    if (use_vec) {
        return launchQuantOut(addLayerNormResidualKernel<T, VecSize, true>, num_rows,
                              hidden / VecSize, static_cast<int>(VecSize * sizeof(T)), stream,
                              input, residual, no_output, residual_out, weight, bias, hidden, eps,
                              alpha, quant);
    }
    return launchQuantOut(addLayerNormResidualKernel<T, 1, true>, num_rows, hidden,
                          static_cast<int>(sizeof(T)), stream, input, residual, no_output,
                          residual_out, weight, bias, hidden, eps, alpha, quant);
}

// ---- AddRMSNorm ----

template <typename T>
cudaError_t AddRMSNorm(const T* input, const T* residual, const T* weight, const T* bias, T* output,
                       unsigned int num_rows, unsigned int hidden_size, float eps, float alpha,
                       cudaStream_t stream) {
    if (num_rows == 0)
        return cudaSuccess;

    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;
    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) && isAligned<T, VecSize>(input) &&
                   isAligned<T, VecSize>(residual) && isAligned<T, VecSize>(output) &&
                   isAligned<T, VecSize>(weight) &&
                   (bias == nullptr || isAligned<T, VecSize>(bias));

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        addRmsNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, residual, output, weight, bias, static_cast<int>(hidden_size), eps, alpha);
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        addRmsNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, residual, output, weight, bias, static_cast<int>(hidden_size), eps, alpha);
    }
    return cudaGetLastError();
}

// ---- AddRMSNormResidual ----

template <typename T>
cudaError_t AddRMSNormResidual(const T* input, const T* residual, const T* weight, const T* bias,
                               T* output, T* residual_out, unsigned int num_rows,
                               unsigned int hidden_size, float eps, float alpha,
                               cudaStream_t stream) {
    if (num_rows == 0)
        return cudaSuccess;

    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;
    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) && isAligned<T, VecSize>(input) &&
                   isAligned<T, VecSize>(residual) && isAligned<T, VecSize>(output) &&
                   isAligned<T, VecSize>(residual_out) && isAligned<T, VecSize>(weight) &&
                   (bias == nullptr || isAligned<T, VecSize>(bias));

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        addRmsNormResidualKernel<T, VecSize>
            <<<num_rows, block_size, 0, stream>>>(input, residual, output, residual_out, weight,
                                                  bias, static_cast<int>(hidden_size), eps, alpha);
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        addRmsNormResidualKernel<T, 1>
            <<<num_rows, block_size, 0, stream>>>(input, residual, output, residual_out, weight,
                                                  bias, static_cast<int>(hidden_size), eps, alpha);
    }
    return cudaGetLastError();
}

// ---- LayerNormActivation ----

template <typename T>
cudaError_t LayerNormActivation(const T* input, const T* weight, const T* bias, T* output,
                                unsigned int num_rows, unsigned int hidden_size, float eps,
                                ActivationType activation, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) &&
                   allAligned<T, VecSize>(input, output, weight, bias);

    cudaError_t err = cudaSuccess;
    dispatchActivation(activation, [&](auto act) {
        using ActFn = decltype(act);
        if (use_vec) {
            int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
            layerNormActKernel<T, VecSize, ActFn><<<num_rows, block_size, 0, stream>>>(
                input, output, weight, bias, static_cast<int>(hidden_size), eps);
        } else {
            int block_size = alignedBlockSize(static_cast<int>(hidden_size));
            layerNormActKernel<T, 1, ActFn><<<num_rows, block_size, 0, stream>>>(
                input, output, weight, bias, static_cast<int>(hidden_size), eps);
        }
        err = cudaGetLastError();
    });
    return err;
}

// ---- RMSNormActivation ----

template <typename T>
cudaError_t RMSNormActivation(const T* input, const T* weight, const T* bias, T* output,
                              unsigned int num_rows, unsigned int hidden_size, float eps,
                              ActivationType activation, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   (hidden_size % VecSize == 0) &&
                   allAligned<T, VecSize>(input, output, weight, bias);

    cudaError_t err = cudaSuccess;
    dispatchActivation(activation, [&](auto act) {
        using ActFn = decltype(act);
        if (use_vec) {
            int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
            rmsNormActKernel<T, VecSize, ActFn><<<num_rows, block_size, 0, stream>>>(
                input, output, weight, bias, static_cast<int>(hidden_size), eps);
        } else {
            int block_size = alignedBlockSize(static_cast<int>(hidden_size));
            rmsNormActKernel<T, 1, ActFn><<<num_rows, block_size, 0, stream>>>(
                input, output, weight, bias, static_cast<int>(hidden_size), eps);
        }
        err = cudaGetLastError();
    });
    return err;
}

// ---- CMVN ----

template <typename T>
cudaError_t CMVN(const T* input, const T* mean, const T* istd, T* output, unsigned int num_rows,
                 unsigned int num_cols, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    int total_elements = static_cast<int>(num_rows * num_cols);
    int block_size = 256;

    bool use_vec = (num_cols >= static_cast<unsigned int>(VecSize)) && (num_cols % VecSize == 0) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output) &&
                   isAligned<T, VecSize>(mean) && isAligned<T, VecSize>(istd);

    if (use_vec) {
        int grid_size = (total_elements / VecSize + block_size - 1) / block_size;
        cmvnKernel<T, VecSize><<<grid_size, block_size, 0, stream>>>(
            input, output, mean, istd, static_cast<int>(num_rows), static_cast<int>(num_cols));
    } else {
        int grid_size = (total_elements + block_size - 1) / block_size;
        cmvnKernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            input, output, mean, istd, static_cast<int>(num_rows), static_cast<int>(num_cols));
    }
    return cudaGetLastError();
}

// ---- BatchNormActivation ----

template <typename T>
cudaError_t BatchNormActivation(const T* input, const T* weight, const T* bias,
                                const T* running_mean, const T* running_var, T* output,
                                unsigned int batch_size, unsigned int seq_len,
                                unsigned int channels, float eps, ActivationType activation,
                                cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    int total_elements = static_cast<int>(batch_size * seq_len * channels);
    int block_size = 256;

    bool use_vec = (channels >= static_cast<unsigned int>(VecSize)) && (channels % VecSize == 0) &&
                   allAligned<T, VecSize>(input, output, weight, bias, running_mean, running_var);

    cudaError_t err = cudaSuccess;
    dispatchActivation(activation, [&](auto act) {
        using ActFn = decltype(act);
        if (use_vec) {
            int grid_size = (total_elements / VecSize + block_size - 1) / block_size;
            batchNormActKernel<T, VecSize, ActFn><<<grid_size, block_size, 0, stream>>>(
                input, output, weight, bias, running_mean, running_var,
                static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(channels),
                eps);
        } else {
            int grid_size = (total_elements + block_size - 1) / block_size;
            batchNormActKernel<T, 1, ActFn><<<grid_size, block_size, 0, stream>>>(
                input, output, weight, bias, running_mean, running_var,
                static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(channels),
                eps);
        }
        err = cudaGetLastError();
    });
    return err;
}

// ---- BatchNormSwish (convenience wrapper) ----

template <typename T>
cudaError_t BatchNormSwish(const T* input, const T* weight, const T* bias, const T* running_mean,
                           const T* running_var, T* output, unsigned int batch_size,
                           unsigned int seq_len, unsigned int channels, float eps,
                           cudaStream_t stream) {
    return BatchNormActivation<T>(input, weight, bias, running_mean, running_var, output,
                                  batch_size, seq_len, channels, eps, ActivationType::SWISH,
                                  stream);
}

}  // namespace norm
}  // namespace oasr
