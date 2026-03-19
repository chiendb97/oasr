// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <torch/extension.h>

#include "kernels/activation/activation_kernels.h"
#include "kernels/common/math.h"
#include "kernels/common/vec_dtypes.h"

namespace oasr {
namespace kernels {

// =============================================================================
// GLU (Gated Linear Unit) Kernel
// =============================================================================

template <typename T>
__global__ void gluKernel(const T* __restrict__ input,  // [batch, seq_len, 2 * channels]
                          T* __restrict__ output,       // [batch, seq_len, channels]
                          int batch_size, int seq_len, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;

    if (idx >= total_elements)
        return;

    int c = idx % channels;
    int pos = idx / channels;

    // input[:, :, :channels] * sigmoid(input[:, :, channels:])
    int input_idx1 = pos * (2 * channels) + c;
    int input_idx2 = pos * (2 * channels) + channels + c;

    float x = static_cast<float>(input[input_idx1]);
    float gate = static_cast<float>(input[input_idx2]);
    float sigmoid_gate = kernels::sigmoid(gate);

    output[idx] = static_cast<T>(x * sigmoid_gate);
}

// =============================================================================
// GLU (Gated Linear Unit) Kernel (vectorized)
// =============================================================================
// Each thread processes VecSize elements using 128-bit vector loads/stores.
// Requires channels % VecSize == 0.
//
// Grid:  ceil(total_elements / (blockDim.x * VecSize))
// Block: 256

template <typename T, int VecSize>
__global__ void gluVecKernel(const T* __restrict__ input,  // [batch, seq_len, 2 * channels]
                             T* __restrict__ output,       // [batch, seq_len, channels]
                             int batch_size, int seq_len, int channels) {
    const int total_elements = batch_size * seq_len * channels;
    const int total_vec_elements = total_elements / VecSize;
    const int vec_channels = channels / VecSize;

    for (int vid = blockIdx.x * blockDim.x + threadIdx.x; vid < total_vec_elements;
         vid += gridDim.x * blockDim.x) {
        // Decompose vid into (pos, vec_c) where pos = batch * seq_len + seq_pos
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
            float sigmoid_gate = kernels::sigmoid(gate);
            out_vec[v] = static_cast<T>(x * sigmoid_gate);
        }

        // Store result
        const int out_idx = pos * channels + c_offset;
        out_vec.store(output + out_idx);
    }
}

// =============================================================================
// Swish Kernel
// =============================================================================

template <typename T>
__global__ void swishKernel(const T* __restrict__ input, T* __restrict__ output, int batch_size,
                            int seq_len, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;

    if (idx >= total_elements)
        return;

    float x = static_cast<float>(input[idx]);
    float result = kernels::swish(x);
    output[idx] = static_cast<T>(result);
}

template <typename T>
void invokeGLUTyped(const torch::Tensor& input, torch::Tensor& output, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2) / 2;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());

    const int total_elements = batch_size * seq_len * channels;

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    if (channels % kVecSize == 0) {
        const int total_vec_elements = total_elements / kVecSize;
        const int block_size = 256;
        int grid_size = (total_vec_elements + block_size - 1) / block_size;
        // Cap grid size to avoid excessive launches
        grid_size = std::min(grid_size, 65535);
        gluVecKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input_ptr, output_ptr, batch_size, seq_len, channels);
    } else {
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        gluKernel<T><<<grid_size, block_size, 0, stream>>>(input_ptr, output_ptr, batch_size,
                                                           seq_len, channels);
    }
}

torch::Tensor invokeGLU(const torch::Tensor& input, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2) / 2;

    auto output = torch::empty({batch_size, seq_len, channels}, input.options());

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            invokeGLUTyped<float>(input, output, stream);
            break;
        case torch::ScalarType::Half:
            invokeGLUTyped<half>(input, output, stream);
            break;
        case torch::ScalarType::BFloat16:
            invokeGLUTyped<__nv_bfloat16>(input, output, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for GLU");
            break;
    }

    return output;
}

torch::Tensor invokeSwish(const torch::Tensor& input, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);

    auto output = torch::empty_like(input);

    const void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();

    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            swishKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr), static_cast<float*>(output_ptr), batch_size,
                seq_len, channels);
            break;
        case torch::ScalarType::Half:
            swishKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr), static_cast<half*>(output_ptr), batch_size,
                seq_len, channels);
            break;
        case torch::ScalarType::BFloat16:
            swishKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(output_ptr), batch_size, seq_len, channels);
            break;
        default:
            throw std::runtime_error("Unsupported data type for Swish");
            break;
    }

    return output;
}

}  // namespace kernels
}  // namespace oasr