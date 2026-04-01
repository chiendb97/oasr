// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA activation kernels — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include <oasr/common/math.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace activation {

// =============================================================================
// GLU (Gated Linear Unit) Kernel
// =============================================================================


template <typename T, int VecSize>
__global__ void gluKernel(const T* __restrict__ input,   // [batch, seq_len, 2 * channels]
                             T* __restrict__ output,         // [batch, seq_len, channels]
                             int batch_size, int seq_len, int channels) {
    const int total_elements = batch_size * seq_len * channels;
    const int total_vec_elements = total_elements / VecSize;
    const int vec_channels = channels / VecSize;

    for (int vid = blockIdx.x * blockDim.x + threadIdx.x; vid < total_vec_elements;
         vid += gridDim.x * blockDim.x) {
        const int vec_c = vid % vec_channels;
        const int pos = vid / vec_channels;

        const int c_offset = vec_c * VecSize;

        // Load value half: input[pos, c_offset]
        const int input_idx1 = pos * (2 * channels) + c_offset;
        Vec<T, VecSize> val_vec;
        val_vec.load(input + input_idx1);

        // Load gate half: input[pos, channels + c_offset]
        const int input_idx2 = pos * (2 * channels) + channels + c_offset;
        Vec<T, VecSize> gate_vec;
        gate_vec.load(input + input_idx2);

        // Compute x * sigmoid(gate) in float
        Vec<T, VecSize> out_vec;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            float x = static_cast<float>(val_vec[v]);
            float gate = static_cast<float>(gate_vec[v]);
            float sigmoid_gate = oasr::sigmoid(gate);
            out_vec[v] = static_cast<T>(x * sigmoid_gate);
        }

        // Store result
        const int out_idx = pos * channels + c_offset;
        out_vec.store(output + out_idx);
    }
}

// =============================================================================
// Swish Kernel (vectorized)
// =============================================================================

template <typename T, int VecSize>
__global__ void swishKernel(const T* __restrict__ input, T* __restrict__ output, int batch_size,
                            int seq_len, int channels) {
    const int total_elements = batch_size * seq_len * channels;
    const int total_vec_elements = total_elements / VecSize;

    for (int vid = blockIdx.x * blockDim.x + threadIdx.x; vid < total_vec_elements;
         vid += gridDim.x * blockDim.x) {
        Vec<T, VecSize> v_in;
        v_in.load(input + vid * VecSize);

        Vec<T, VecSize> v_out;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            float x = static_cast<float>(v_in[v]);
            v_out[v] = static_cast<T>(oasr::swish(x));
        }

        v_out.store(output + vid * VecSize);
    }
}

// =============================================================================
// Typed Launchers — raw pointer interface, returns cudaError_t
// =============================================================================

template <typename T>
cudaError_t GLU(const T* input, T* output, int batch_size, int seq_len, int channels,
                cudaStream_t stream) {
    const int total_elements = batch_size * seq_len * channels;

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    if (channels % kVecSize == 0) {
        const int total_vec_elements = total_elements / kVecSize;
        const int block_size = 256;
        int grid_size = (total_vec_elements + block_size - 1) / block_size;
        grid_size = std::min(grid_size, 65535);
        gluKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input, output, batch_size, seq_len, channels);
    } else {
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        gluKernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            input, output, batch_size, seq_len, channels);
    }

    return cudaGetLastError();
}

template <typename T>
cudaError_t Swish(const T* input, T* output, int batch_size, int seq_len, int channels,
                  cudaStream_t stream) {
    const int total_elements = batch_size * seq_len * channels;

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    if (channels % kVecSize == 0) {
        const int total_vec_elements = total_elements / kVecSize;
        const int block_size = 256;
        int grid_size = (total_vec_elements + block_size - 1) / block_size;
        grid_size = std::min(grid_size, 65535);
        swishKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input, output, batch_size, seq_len, channels);
    } else {
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        swishKernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            input, output, batch_size, seq_len, channels);
    }

    return cudaGetLastError();
}

}  // namespace activation
}  // namespace oasr
