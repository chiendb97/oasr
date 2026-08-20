// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fused inference kernels for one recurrent layer.  The whole affine
// recurrence, gate activation, and state update is one launch per timestep.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <oasr/common/math.h>
#include <oasr/common/reduction.h>
#include <oasr/common/utils.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace recurrent {

enum class RnnActivation : int { TANH = 0, RELU = 1 };

namespace detail {

// One CTA owns one (batch, hidden) state element.  Threads split the two dot
// products and accumulate all four gates together, so each input/state value
// is loaded once rather than four times.  Gate order matches PyTorch/cuDNN:
// input, forget, cell, output.
template <typename T, int VecSize>
__global__ void LstmStepKernel(T* __restrict__ output, T* __restrict__ cells,
                               T* __restrict__ final_h, T* __restrict__ final_c,
                               const T* __restrict__ input, const T* __restrict__ previous_h,
                               const T* __restrict__ previous_c, const T* __restrict__ weight_ih,
                               const T* __restrict__ weight_hh, const T* __restrict__ bias_ih,
                               const T* __restrict__ bias_hh, int batch_size, int input_size,
                               int hidden_size, int64_t input_batch_stride,
                               int64_t previous_batch_stride, int64_t output_batch_stride) {
    const int hidden = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    if (hidden >= hidden_size || batch >= batch_size)
        return;

    const T* x = input + static_cast<int64_t>(batch) * input_batch_stride;
    const T* h = previous_h + static_cast<int64_t>(batch) * previous_batch_stride;
    const T* w_ih[4] = {
        weight_ih + static_cast<int64_t>(hidden) * input_size,
        weight_ih + static_cast<int64_t>(hidden_size + hidden) * input_size,
        weight_ih + static_cast<int64_t>(2 * hidden_size + hidden) * input_size,
        weight_ih + static_cast<int64_t>(3 * hidden_size + hidden) * input_size,
    };
    const T* w_hh[4] = {
        weight_hh + static_cast<int64_t>(hidden) * hidden_size,
        weight_hh + static_cast<int64_t>(hidden_size + hidden) * hidden_size,
        weight_hh + static_cast<int64_t>(2 * hidden_size + hidden) * hidden_size,
        weight_hh + static_cast<int64_t>(3 * hidden_size + hidden) * hidden_size,
    };

    using VecT = oasr::Vec<T, VecSize>;
    float accum[4] = {};
    for (int k = threadIdx.x * VecSize; k < input_size; k += blockDim.x * VecSize) {
        VecT xv;
        xv.load(x + k);
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            VecT wv;
            wv.load(w_ih[gate] + k);
#pragma unroll
            for (int element = 0; element < VecSize; ++element) {
                accum[gate] = fmaf(toFloat(xv[element]), toFloat(wv[element]), accum[gate]);
            }
        }
    }

    for (int k = threadIdx.x * VecSize; k < hidden_size; k += blockDim.x * VecSize) {
        VecT hv;
        hv.load(h + k);
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            VecT wv;
            wv.load(w_hh[gate] + k);
#pragma unroll
            for (int element = 0; element < VecSize; ++element) {
                accum[gate] = fmaf(toFloat(hv[element]), toFloat(wv[element]), accum[gate]);
            }
        }
    }

    __shared__ float scratch[4 * 32];
    reduction::blockReduceSums(accum, scratch);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            const int offset = gate * hidden_size + hidden;
            if (bias_ih != nullptr)
                accum[gate] += toFloat(bias_ih[offset]);
            if (bias_hh != nullptr)
                accum[gate] += toFloat(bias_hh[offset]);
        }
        const float input_gate = fastSigmoid(accum[0]);
        const float forget_gate = fastSigmoid(accum[1]);
        const float cell_gate = tanhf(accum[2]);
        const float output_gate = fastSigmoid(accum[3]);
        // The cell ring, the initial cell and the final cell are all
        // (batch, hidden) contiguous, so one offset addresses every one of them.
        const int64_t cell_offset = static_cast<int64_t>(batch) * hidden_size + hidden;
        const int64_t output_offset = static_cast<int64_t>(batch) * output_batch_stride + hidden;
        const float cell = forget_gate * toFloat(previous_c[cell_offset]) + input_gate * cell_gate;
        const T cell_value = fromFloat<T>(cell);
        const T hidden_value = fromFloat<T>(output_gate * tanhf(cell));
        cells[cell_offset] = cell_value;
        output[output_offset] = hidden_value;
        if (final_h != nullptr) {
            final_h[cell_offset] = hidden_value;
            final_c[cell_offset] = cell_value;
        }
    }
}

template <typename T, int VecSize, RnnActivation Activation>
__global__ void RnnStepKernel(T* __restrict__ output, T* __restrict__ final_h,
                              const T* __restrict__ input, const T* __restrict__ previous_h,
                              const T* __restrict__ weight_ih, const T* __restrict__ weight_hh,
                              const T* __restrict__ bias_ih, const T* __restrict__ bias_hh,
                              int batch_size, int input_size, int hidden_size,
                              int64_t input_batch_stride, int64_t previous_batch_stride,
                              int64_t output_batch_stride) {
    const int hidden = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    if (hidden >= hidden_size || batch >= batch_size)
        return;

    const T* x = input + static_cast<int64_t>(batch) * input_batch_stride;
    const T* h = previous_h + static_cast<int64_t>(batch) * previous_batch_stride;
    const T* w_ih = weight_ih + static_cast<int64_t>(hidden) * input_size;
    const T* w_hh = weight_hh + static_cast<int64_t>(hidden) * hidden_size;
    using VecT = oasr::Vec<T, VecSize>;
    float accum[1] = {};
    for (int k = threadIdx.x * VecSize; k < input_size; k += blockDim.x * VecSize) {
        VecT xv, wv;
        xv.load(x + k);
        wv.load(w_ih + k);
#pragma unroll
        for (int element = 0; element < VecSize; ++element) {
            accum[0] = fmaf(toFloat(xv[element]), toFloat(wv[element]), accum[0]);
        }
    }

    for (int k = threadIdx.x * VecSize; k < hidden_size; k += blockDim.x * VecSize) {
        VecT hv, wv;
        hv.load(h + k);
        wv.load(w_hh + k);
#pragma unroll
        for (int element = 0; element < VecSize; ++element) {
            accum[0] = fmaf(toFloat(hv[element]), toFloat(wv[element]), accum[0]);
        }
    }

    __shared__ float scratch[32];
    reduction::blockReduceSums(accum, scratch);
    if (threadIdx.x == 0) {
        if (bias_ih != nullptr)
            accum[0] += toFloat(bias_ih[hidden]);
        if (bias_hh != nullptr)
            accum[0] += toFloat(bias_hh[hidden]);
        const float activated =
            Activation == RnnActivation::RELU ? fmaxf(accum[0], 0.0f) : tanhf(accum[0]);
        const T hidden_value = fromFloat<T>(activated);
        output[static_cast<int64_t>(batch) * output_batch_stride + hidden] = hidden_value;
        if (final_h != nullptr) {
            final_h[static_cast<int64_t>(batch) * hidden_size + hidden] = hidden_value;
        }
    }
}

// Cohort path: one warp owns a batch row while the CTA owns one hidden output.
// All warps reuse a single shared-memory copy of that output's weights.  This
// removes the batch-fold repetition of the GEMV kernel without paying the
// packing and intermediate-buffer costs of a tensor-core GEMM at T=1.
template <typename T, int VecSize>
__global__ void LstmCohortStepKernel(
    T* __restrict__ output, T* __restrict__ cells, T* __restrict__ final_h, T* __restrict__ final_c,
    const T* __restrict__ input, const T* __restrict__ previous_h, const T* __restrict__ previous_c,
    const T* __restrict__ weight_ih, const T* __restrict__ weight_hh, const T* __restrict__ bias_ih,
    const T* __restrict__ bias_hh, int batch_size, int input_size, int hidden_size,
    int64_t input_batch_stride, int64_t previous_batch_stride, int64_t output_batch_stride) {
    const int hidden = static_cast<int>(blockIdx.x);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int batch = static_cast<int>(blockIdx.y) * warps_per_block + warp;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    T* shared_weight_ih = reinterpret_cast<T*>(shared_bytes);
    T* shared_weight_hh = shared_weight_ih + 4 * input_size;

    using VecT = oasr::Vec<T, VecSize>;
    const int input_vectors = input_size / VecSize;
    const int hidden_vectors = hidden_size / VecSize;
    const int total_vectors = 4 * (input_vectors + hidden_vectors);
    for (int index = threadIdx.x; index < total_vectors; index += blockDim.x) {
        if (index < 4 * input_vectors) {
            const int gate = index / input_vectors;
            const int k = VecSize * (index - gate * input_vectors);
            VecT weight;
            weight.load(weight_ih + static_cast<int64_t>(gate * hidden_size + hidden) * input_size +
                        k);
            weight.store(shared_weight_ih + gate * input_size + k);
        } else {
            const int recurrent_index = index - 4 * input_vectors;
            const int gate = recurrent_index / hidden_vectors;
            const int k = VecSize * (recurrent_index - gate * hidden_vectors);
            VecT weight;
            weight.load(weight_hh +
                        static_cast<int64_t>(gate * hidden_size + hidden) * hidden_size + k);
            weight.store(shared_weight_hh + gate * hidden_size + k);
        }
    }
    __syncthreads();
    if (batch >= batch_size)
        return;

    const T* x = input + static_cast<int64_t>(batch) * input_batch_stride;
    const T* h = previous_h + static_cast<int64_t>(batch) * previous_batch_stride;
    float accum[4] = {};
    for (int k = VecSize * lane; k < input_size; k += 32 * VecSize) {
        VecT value;
        value.load(x + k);
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            VecT weight;
            weight.load(shared_weight_ih + gate * input_size + k);
#pragma unroll
            for (int element = 0; element < VecSize; ++element) {
                accum[gate] = fmaf(toFloat(value[element]), toFloat(weight[element]), accum[gate]);
            }
        }
    }
    for (int k = VecSize * lane; k < hidden_size; k += 32 * VecSize) {
        VecT value;
        value.load(h + k);
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            VecT weight;
            weight.load(shared_weight_hh + gate * hidden_size + k);
#pragma unroll
            for (int element = 0; element < VecSize; ++element) {
                accum[gate] = fmaf(toFloat(value[element]), toFloat(weight[element]), accum[gate]);
            }
        }
    }
#pragma unroll
    for (int gate = 0; gate < 4; ++gate)
        accum[gate] = reduction::warpReduceSumDown(accum[gate]);
    if (lane == 0) {
#pragma unroll
        for (int gate = 0; gate < 4; ++gate) {
            const int offset = gate * hidden_size + hidden;
            if (bias_ih != nullptr)
                accum[gate] += toFloat(bias_ih[offset]);
            if (bias_hh != nullptr)
                accum[gate] += toFloat(bias_hh[offset]);
        }
        const float input_gate = fastSigmoid(accum[0]);
        const float forget_gate = fastSigmoid(accum[1]);
        const float cell_gate = tanhf(accum[2]);
        const float output_gate = fastSigmoid(accum[3]);
        const int64_t cell_offset = static_cast<int64_t>(batch) * hidden_size + hidden;
        const int64_t output_offset = static_cast<int64_t>(batch) * output_batch_stride + hidden;
        const float cell = forget_gate * toFloat(previous_c[cell_offset]) + input_gate * cell_gate;
        const T cell_value = fromFloat<T>(cell);
        const T hidden_value = fromFloat<T>(output_gate * tanhf(cell));
        cells[cell_offset] = cell_value;
        output[output_offset] = hidden_value;
        if (final_h != nullptr) {
            final_h[cell_offset] = hidden_value;
            final_c[cell_offset] = cell_value;
        }
    }
}

template <typename T, int VecSize, RnnActivation Activation>
__global__ void RnnCohortStepKernel(T* __restrict__ output, T* __restrict__ final_h,
                                    const T* __restrict__ input, const T* __restrict__ previous_h,
                                    const T* __restrict__ weight_ih,
                                    const T* __restrict__ weight_hh, const T* __restrict__ bias_ih,
                                    const T* __restrict__ bias_hh, int batch_size, int input_size,
                                    int hidden_size, int64_t input_batch_stride,
                                    int64_t previous_batch_stride, int64_t output_batch_stride) {
    const int hidden = static_cast<int>(blockIdx.x);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int batch = static_cast<int>(blockIdx.y) * warps_per_block + warp;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    T* shared_weight_ih = reinterpret_cast<T*>(shared_bytes);
    T* shared_weight_hh = shared_weight_ih + input_size;

    using VecT = oasr::Vec<T, VecSize>;
    const int input_vectors = input_size / VecSize;
    const int hidden_vectors = hidden_size / VecSize;
    const int total_vectors = input_vectors + hidden_vectors;
    for (int index = threadIdx.x; index < total_vectors; index += blockDim.x) {
        if (index < input_vectors) {
            const int k = VecSize * index;
            VecT weight;
            weight.load(weight_ih + static_cast<int64_t>(hidden) * input_size + k);
            weight.store(shared_weight_ih + k);
        } else {
            const int k = VecSize * (index - input_vectors);
            VecT weight;
            weight.load(weight_hh + static_cast<int64_t>(hidden) * hidden_size + k);
            weight.store(shared_weight_hh + k);
        }
    }
    __syncthreads();
    if (batch >= batch_size)
        return;
    const T* x = input + static_cast<int64_t>(batch) * input_batch_stride;
    const T* h = previous_h + static_cast<int64_t>(batch) * previous_batch_stride;
    float accum = 0.0f;
    for (int k = VecSize * lane; k < input_size; k += 32 * VecSize) {
        VecT value, weight;
        value.load(x + k);
        weight.load(shared_weight_ih + k);
#pragma unroll
        for (int element = 0; element < VecSize; ++element) {
            accum = fmaf(toFloat(value[element]), toFloat(weight[element]), accum);
        }
    }
    for (int k = VecSize * lane; k < hidden_size; k += 32 * VecSize) {
        VecT value, weight;
        value.load(h + k);
        weight.load(shared_weight_hh + k);
#pragma unroll
        for (int element = 0; element < VecSize; ++element) {
            accum = fmaf(toFloat(value[element]), toFloat(weight[element]), accum);
        }
    }
    accum = reduction::warpReduceSumDown(accum);
    if (lane == 0) {
        if (bias_ih != nullptr)
            accum += toFloat(bias_ih[hidden]);
        if (bias_hh != nullptr)
            accum += toFloat(bias_hh[hidden]);
        const float activated =
            Activation == RnnActivation::RELU ? fmaxf(accum, 0.0f) : tanhf(accum);
        const T hidden_value = fromFloat<T>(activated);
        const int64_t output_offset = static_cast<int64_t>(batch) * output_batch_stride + hidden;
        output[output_offset] = hidden_value;
        if (final_h != nullptr)
            final_h[static_cast<int64_t>(batch) * hidden_size + hidden] = hidden_value;
    }
}

// Tensor-core path epilogues.  The input projection is computed once for the
// whole sequence and the recurrent projection once per timestep; these kernels
// fuse their sum with nonlinearities and state writes.  One thread owns one
// state element, so there is no reduction after CUTLASS has produced the gates.
template <typename T>
__global__ void LstmGateStepKernel(T* __restrict__ output, T* __restrict__ cells,
                                   T* __restrict__ final_h, T* __restrict__ final_c,
                                   const T* __restrict__ input_gates,
                                   const T* __restrict__ recurrent_gates,
                                   const T* __restrict__ previous_c, int batch_size,
                                   int hidden_size, int64_t input_batch_stride,
                                   int64_t previous_cell_batch_stride) {
    const int hidden = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int batch = static_cast<int>(blockIdx.y);
    if (hidden >= hidden_size || batch >= batch_size)
        return;

    const int64_t input_base = static_cast<int64_t>(batch) * input_batch_stride + hidden;
    const int64_t recurrent_base = static_cast<int64_t>(batch) * 4 * hidden_size + hidden;
    const float input_gate =
        fastSigmoid(toFloat(input_gates[input_base]) + toFloat(recurrent_gates[recurrent_base]));
    const float forget_gate = fastSigmoid(toFloat(input_gates[input_base + hidden_size]) +
                                          toFloat(recurrent_gates[recurrent_base + hidden_size]));
    const float cell_gate = tanhf(toFloat(input_gates[input_base + 2 * hidden_size]) +
                                  toFloat(recurrent_gates[recurrent_base + 2 * hidden_size]));
    const float output_gate =
        fastSigmoid(toFloat(input_gates[input_base + 3 * hidden_size]) +
                    toFloat(recurrent_gates[recurrent_base + 3 * hidden_size]));
    const int64_t previous_offset =
        static_cast<int64_t>(batch) * previous_cell_batch_stride + hidden;
    const int64_t output_offset = static_cast<int64_t>(batch) * hidden_size + hidden;
    const float cell = forget_gate * toFloat(previous_c[previous_offset]) + input_gate * cell_gate;
    const T cell_value = fromFloat<T>(cell);
    const T hidden_value = fromFloat<T>(output_gate * tanhf(cell));
    cells[output_offset] = cell_value;
    output[output_offset] = hidden_value;
    if (final_h != nullptr) {
        final_h[output_offset] = hidden_value;
        final_c[output_offset] = cell_value;
    }
}

// Finalizer for decomposition tactics whose CUTLASS kernel materializes gates
// in the recurrent-friendly [batch, hidden, gate] layout.  The ordinary PyTorch
// checkpoint layout is gate-major; tensor-core weights are packed once by the
// Python layer before entering this path.
template <typename T>
__global__ void LstmInterleavedGateStepKernel(T* __restrict__ output, T* __restrict__ cells,
                                              T* __restrict__ final_h, T* __restrict__ final_c,
                                              const T* __restrict__ gates,
                                              const T* __restrict__ previous_c, int batch_size,
                                              int hidden_size, int64_t previous_cell_batch_stride) {
    const int hidden = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int batch = static_cast<int>(blockIdx.y);
    if (hidden >= hidden_size || batch >= batch_size)
        return;

    const int64_t gate_base = (static_cast<int64_t>(batch) * hidden_size + hidden) * 4;
    const float input_gate = fastSigmoid(toFloat(gates[gate_base]));
    const float forget_gate = fastSigmoid(toFloat(gates[gate_base + 1]));
    const float cell_gate = tanhf(toFloat(gates[gate_base + 2]));
    const float output_gate = fastSigmoid(toFloat(gates[gate_base + 3]));
    const int64_t previous_offset =
        static_cast<int64_t>(batch) * previous_cell_batch_stride + hidden;
    const int64_t output_offset = static_cast<int64_t>(batch) * hidden_size + hidden;
    const float cell = forget_gate * toFloat(previous_c[previous_offset]) + input_gate * cell_gate;
    const T cell_value = fromFloat<T>(cell);
    const T hidden_value = fromFloat<T>(output_gate * tanhf(cell));
    cells[output_offset] = cell_value;
    output[output_offset] = hidden_value;
    if (final_h != nullptr) {
        final_h[output_offset] = hidden_value;
        final_c[output_offset] = cell_value;
    }
}

template <typename T, RnnActivation Activation>
__global__ void RnnGateStepKernel(T* __restrict__ output, T* __restrict__ final_h,
                                  const T* __restrict__ input_gates,
                                  const T* __restrict__ recurrent_gates, int batch_size,
                                  int hidden_size, int64_t input_batch_stride) {
    const int hidden = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int batch = static_cast<int>(blockIdx.y);
    if (hidden >= hidden_size || batch >= batch_size)
        return;
    const int64_t input_offset = static_cast<int64_t>(batch) * input_batch_stride + hidden;
    const int64_t output_offset = static_cast<int64_t>(batch) * hidden_size + hidden;
    const float value =
        toFloat(input_gates[input_offset]) + toFloat(recurrent_gates[output_offset]);
    const float activated = Activation == RnnActivation::RELU ? fmaxf(value, 0.0f) : tanhf(value);
    const T hidden_value = fromFloat<T>(activated);
    output[output_offset] = hidden_value;
    if (final_h != nullptr)
        final_h[output_offset] = hidden_value;
}

template <int VecSize>
inline int BlockSize(int input_size, int hidden_size) {
    const int vectors = (std::max(input_size, hidden_size) + VecSize - 1) / VecSize;
    return std::min(512, std::max(32, ((vectors + 31) / 32) * 32));
}

template <typename T, int VecSize>
inline bool CanVectorizeLayer(const T* output, const T* input, const T* initial_h,
                              const T* weight_ih, const T* weight_hh, int input_size,
                              int hidden_size, int64_t input_time_stride,
                              int64_t input_batch_stride, int64_t output_time_stride,
                              int64_t output_batch_stride) {
    static_assert(VecSize > 1, "vectorized dispatch requires VecSize > 1");
    return input_size >= VecSize && hidden_size >= VecSize && input_size % VecSize == 0 &&
           hidden_size % VecSize == 0 && input_time_stride % VecSize == 0 &&
           input_batch_stride % VecSize == 0 && output_time_stride % VecSize == 0 &&
           output_batch_stride % VecSize == 0 && oasr::isAligned<T, VecSize>(output) &&
           oasr::isAligned<T, VecSize>(input) && oasr::isAligned<T, VecSize>(initial_h) &&
           oasr::isAligned<T, VecSize>(weight_ih) && oasr::isAligned<T, VecSize>(weight_hh);
}

template <typename T, int VecSize>
cudaError_t LstmLayerImpl(T* output, T* cells, T* final_h, T* final_c, const T* input,
                          const T* initial_h, const T* initial_c, const T* weight_ih,
                          const T* weight_hh, const T* bias_ih, const T* bias_hh,
                          int sequence_length, int batch_size, int input_size, int hidden_size,
                          int64_t input_time_stride, int64_t input_batch_stride,
                          int64_t output_time_stride, int64_t output_batch_stride,
                          cudaStream_t stream) {
    const int block_size = BlockSize<VecSize>(input_size, hidden_size);
    const dim3 grid(hidden_size, batch_size);
    const int cohort_warps = std::min(32, batch_size);
    const size_t cohort_smem = 4 * static_cast<size_t>(input_size + hidden_size) * sizeof(T);
    const bool use_cohort = batch_size >= 8 && cohort_smem <= 48 * 1024;
    const dim3 cohort_grid(hidden_size, (batch_size + cohort_warps - 1) / cohort_warps);
    // Only cell[t-1] is ever read, so the cell history is a two-slice ring of
    // (batch, hidden) rather than the whole (sequence, batch, hidden) tensor.
    const int cell_ring = sequence_length > 1 ? 2 : 1;
    const int64_t cell_time_stride = static_cast<int64_t>(batch_size) * hidden_size;
    for (int timestep = 0; timestep < sequence_length; ++timestep) {
        T* output_t = output + static_cast<int64_t>(timestep) * output_time_stride;
        T* cell_t = cells + static_cast<int64_t>(timestep % cell_ring) * cell_time_stride;
        const T* input_t = input + static_cast<int64_t>(timestep) * input_time_stride;
        const T* previous_h =
            timestep == 0 ? initial_h
                          : output + static_cast<int64_t>(timestep - 1) * output_time_stride;
        const T* previous_c =
            timestep == 0
                ? initial_c
                : cells + static_cast<int64_t>((timestep - 1) % cell_ring) * cell_time_stride;
        const int64_t previous_batch_stride = timestep == 0 ? hidden_size : output_batch_stride;
        T* final_h_t = timestep + 1 == sequence_length ? final_h : nullptr;
        T* final_c_t = timestep + 1 == sequence_length ? final_c : nullptr;
        if (use_cohort) {
            LstmCohortStepKernel<T, VecSize>
                <<<cohort_grid, cohort_warps * 32, cohort_smem, stream>>>(
                    output_t, cell_t, final_h_t, final_c_t, input_t, previous_h, previous_c,
                    weight_ih, weight_hh, bias_ih, bias_hh, batch_size, input_size, hidden_size,
                    input_batch_stride, previous_batch_stride, output_batch_stride);
        } else {
            LstmStepKernel<T, VecSize><<<grid, block_size, 0, stream>>>(
                output_t, cell_t, final_h_t, final_c_t, input_t, previous_h, previous_c, weight_ih,
                weight_hh, bias_ih, bias_hh, batch_size, input_size, hidden_size,
                input_batch_stride, previous_batch_stride, output_batch_stride);
        }
    }
    return cudaGetLastError();
}

template <typename T, int VecSize, RnnActivation Activation>
cudaError_t RnnLayerImpl(T* output, T* final_h, const T* input, const T* initial_h,
                         const T* weight_ih, const T* weight_hh, const T* bias_ih, const T* bias_hh,
                         int sequence_length, int batch_size, int input_size, int hidden_size,
                         int64_t input_time_stride, int64_t input_batch_stride,
                         int64_t output_time_stride, int64_t output_batch_stride,
                         cudaStream_t stream) {
    const int block_size = BlockSize<VecSize>(input_size, hidden_size);
    const dim3 grid(hidden_size, batch_size);
    const int cohort_warps = std::min(32, batch_size);
    const size_t cohort_smem = static_cast<size_t>(input_size + hidden_size) * sizeof(T);
    const bool use_cohort = batch_size >= 8 && cohort_smem <= 48 * 1024;
    const dim3 cohort_grid(hidden_size, (batch_size + cohort_warps - 1) / cohort_warps);
    for (int timestep = 0; timestep < sequence_length; ++timestep) {
        T* output_t = output + static_cast<int64_t>(timestep) * output_time_stride;
        const T* input_t = input + static_cast<int64_t>(timestep) * input_time_stride;
        const T* previous_h =
            timestep == 0 ? initial_h
                          : output + static_cast<int64_t>(timestep - 1) * output_time_stride;
        const int64_t previous_batch_stride = timestep == 0 ? hidden_size : output_batch_stride;
        T* final_h_t = timestep + 1 == sequence_length ? final_h : nullptr;
        if (use_cohort) {
            RnnCohortStepKernel<T, VecSize, Activation>
                <<<cohort_grid, cohort_warps * 32, cohort_smem, stream>>>(
                    output_t, final_h_t, input_t, previous_h, weight_ih, weight_hh, bias_ih,
                    bias_hh, batch_size, input_size, hidden_size, input_batch_stride,
                    previous_batch_stride, output_batch_stride);
        } else {
            RnnStepKernel<T, VecSize, Activation><<<grid, block_size, 0, stream>>>(
                output_t, final_h_t, input_t, previous_h, weight_ih, weight_hh, bias_ih, bias_hh,
                batch_size, input_size, hidden_size, input_batch_stride, previous_batch_stride,
                output_batch_stride);
        }
    }
    return cudaGetLastError();
}

template <typename T, int VecSize>
cudaError_t RnnLayerImpl(T* output, T* final_h, const T* input, const T* initial_h,
                         const T* weight_ih, const T* weight_hh, const T* bias_ih, const T* bias_hh,
                         int sequence_length, int batch_size, int input_size, int hidden_size,
                         int64_t input_time_stride, int64_t input_batch_stride,
                         int64_t output_time_stride, int64_t output_batch_stride,
                         RnnActivation activation, cudaStream_t stream) {
    if (activation == RnnActivation::RELU) {
        return RnnLayerImpl<T, VecSize, RnnActivation::RELU>(
            output, final_h, input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh,
            sequence_length, batch_size, input_size, hidden_size, input_time_stride,
            input_batch_stride, output_time_stride, output_batch_stride, stream);
    }
    return RnnLayerImpl<T, VecSize, RnnActivation::TANH>(
        output, final_h, input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh, sequence_length,
        batch_size, input_size, hidden_size, input_time_stride, input_batch_stride,
        output_time_stride, output_batch_stride, stream);
}

}  // namespace detail

template <typename T>
cudaError_t LstmLayer(T* output, T* cells, T* final_h, T* final_c, const T* input,
                      const T* initial_h, const T* initial_c, const T* weight_ih,
                      const T* weight_hh, const T* bias_ih, const T* bias_hh, int sequence_length,
                      int batch_size, int input_size, int hidden_size, int64_t input_time_stride,
                      int64_t input_batch_stride, int64_t output_time_stride,
                      int64_t output_batch_stride, cudaStream_t stream) {
    if (sequence_length == 0 || batch_size == 0)
        return cudaSuccess;
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;
    if constexpr (VecSize > 1) {
        if (detail::CanVectorizeLayer<T, VecSize>(
                output, input, initial_h, weight_ih, weight_hh, input_size, hidden_size,
                input_time_stride, input_batch_stride, output_time_stride, output_batch_stride)) {
            return detail::LstmLayerImpl<T, VecSize>(
                output, cells, final_h, final_c, input, initial_h, initial_c, weight_ih, weight_hh,
                bias_ih, bias_hh, sequence_length, batch_size, input_size, hidden_size,
                input_time_stride, input_batch_stride, output_time_stride, output_batch_stride,
                stream);
        }
    }
    if constexpr (VecSize > 2) {
        if (detail::CanVectorizeLayer<T, 2>(
                output, input, initial_h, weight_ih, weight_hh, input_size, hidden_size,
                input_time_stride, input_batch_stride, output_time_stride, output_batch_stride)) {
            return detail::LstmLayerImpl<T, 2>(output, cells, final_h, final_c, input, initial_h,
                                               initial_c, weight_ih, weight_hh, bias_ih, bias_hh,
                                               sequence_length, batch_size, input_size, hidden_size,
                                               input_time_stride, input_batch_stride,
                                               output_time_stride, output_batch_stride, stream);
        }
    }
    return detail::LstmLayerImpl<T, 1>(
        output, cells, final_h, final_c, input, initial_h, initial_c, weight_ih, weight_hh, bias_ih,
        bias_hh, sequence_length, batch_size, input_size, hidden_size, input_time_stride,
        input_batch_stride, output_time_stride, output_batch_stride, stream);
}

template <typename T>
cudaError_t RnnLayer(T* output, T* final_h, const T* input, const T* initial_h, const T* weight_ih,
                     const T* weight_hh, const T* bias_ih, const T* bias_hh, int sequence_length,
                     int batch_size, int input_size, int hidden_size, int64_t input_time_stride,
                     int64_t input_batch_stride, int64_t output_time_stride,
                     int64_t output_batch_stride, RnnActivation activation, cudaStream_t stream) {
    if (sequence_length == 0 || batch_size == 0)
        return cudaSuccess;
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;
    if constexpr (VecSize > 1) {
        if (detail::CanVectorizeLayer<T, VecSize>(
                output, input, initial_h, weight_ih, weight_hh, input_size, hidden_size,
                input_time_stride, input_batch_stride, output_time_stride, output_batch_stride)) {
            return detail::RnnLayerImpl<T, VecSize>(
                output, final_h, input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh,
                sequence_length, batch_size, input_size, hidden_size, input_time_stride,
                input_batch_stride, output_time_stride, output_batch_stride, activation, stream);
        }
    }
    if constexpr (VecSize > 2) {
        if (detail::CanVectorizeLayer<T, 2>(
                output, input, initial_h, weight_ih, weight_hh, input_size, hidden_size,
                input_time_stride, input_batch_stride, output_time_stride, output_batch_stride)) {
            return detail::RnnLayerImpl<T, 2>(
                output, final_h, input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh,
                sequence_length, batch_size, input_size, hidden_size, input_time_stride,
                input_batch_stride, output_time_stride, output_batch_stride, activation, stream);
        }
    }
    return detail::RnnLayerImpl<T, 1>(output, final_h, input, initial_h, weight_ih, weight_hh,
                                      bias_ih, bias_hh, sequence_length, batch_size, input_size,
                                      hidden_size, input_time_stride, input_batch_stride,
                                      output_time_stride, output_batch_stride, activation, stream);
}

template <typename T>
cudaError_t LstmGateStep(T* output, T* cells, T* final_h, T* final_c, const T* input_gates,
                         const T* recurrent_gates, const T* previous_c, int batch_size,
                         int hidden_size, int64_t input_batch_stride,
                         int64_t previous_cell_batch_stride, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const dim3 grid((hidden_size + kBlockSize - 1) / kBlockSize, batch_size);
    detail::LstmGateStepKernel<T><<<grid, kBlockSize, 0, stream>>>(
        output, cells, final_h, final_c, input_gates, recurrent_gates, previous_c, batch_size,
        hidden_size, input_batch_stride, previous_cell_batch_stride);
    return cudaGetLastError();
}

template <typename T>
cudaError_t LstmInterleavedGateStep(T* output, T* cells, T* final_h, T* final_c, const T* gates,
                                    const T* previous_c, int batch_size, int hidden_size,
                                    int64_t previous_cell_batch_stride, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const dim3 grid((hidden_size + kBlockSize - 1) / kBlockSize, batch_size);
    detail::LstmInterleavedGateStepKernel<T>
        <<<grid, kBlockSize, 0, stream>>>(output, cells, final_h, final_c, gates, previous_c,
                                          batch_size, hidden_size, previous_cell_batch_stride);
    return cudaGetLastError();
}

template <typename T>
cudaError_t RnnGateStep(T* output, T* final_h, const T* input_gates, const T* recurrent_gates,
                        int batch_size, int hidden_size, int64_t input_batch_stride,
                        RnnActivation activation, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const dim3 grid((hidden_size + kBlockSize - 1) / kBlockSize, batch_size);
    if (activation == RnnActivation::RELU) {
        detail::RnnGateStepKernel<T, RnnActivation::RELU>
            <<<grid, kBlockSize, 0, stream>>>(output, final_h, input_gates, recurrent_gates,
                                              batch_size, hidden_size, input_batch_stride);
    } else {
        detail::RnnGateStepKernel<T, RnnActivation::TANH>
            <<<grid, kBlockSize, 0, stream>>>(output, final_h, input_gates, recurrent_gates,
                                              batch_size, hidden_size, input_batch_stride);
    }
    return cudaGetLastError();
}

}  // namespace recurrent
}  // namespace oasr
