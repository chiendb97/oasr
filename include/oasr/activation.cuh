// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA activation kernels — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <oasr/common/math.h>
#include <oasr/common/utils.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace activation {

// =============================================================================
// GLU (Gated Linear Unit) Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void gluKernel(const T* __restrict__ input,  // [batch, seq_len, 2 * channels]
                          T* __restrict__ output,       // [batch, seq_len, channels]
                          int batch_size, int seq_len, int channels) {
    const int total_elements = batch_size * seq_len * channels;
    const int total_vec_elements = total_elements / VecSize;
    const int vec_channels = channels / VecSize;

    for (int vid = blockIdx.x * blockDim.x + threadIdx.x; vid < total_vec_elements;
         vid += gridDim.x * blockDim.x) {
        const int vec_c = vid % vec_channels;
        const int pos = vid / vec_channels;

        const int c_offset = vec_c * VecSize;

        const int input_idx1 = pos * (2 * channels) + c_offset;
        Vec<T, VecSize> val_vec;
        val_vec.load(input + input_idx1);

        const int input_idx2 = pos * (2 * channels) + channels + c_offset;
        Vec<T, VecSize> gate_vec;
        gate_vec.load(input + input_idx2);

        Vec<T, VecSize> out_vec;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            float x = static_cast<float>(val_vec[v]);
            float gate = static_cast<float>(gate_vec[v]);
            float sigmoid_gate = oasr::sigmoid(gate);
            out_vec[v] = static_cast<T>(x * sigmoid_gate);
        }

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
// Unary Activation Kernel (vectorized, elementwise over a flat buffer)
// =============================================================================

template <typename T, int VecSize, typename Activation>
__global__ void elementwiseKernel(const T* __restrict__ input, T* __restrict__ output, int64_t n,
                                  Activation activation) {
    const int64_t total_vec_elements = n / VecSize;
    for (int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vid < total_vec_elements; vid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        Vec<T, VecSize> v_in;
        v_in.load(input + vid * VecSize);

        Vec<T, VecSize> v_out;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            v_out[v] = static_cast<T>(activation(static_cast<float>(v_in[v])));
        }
        v_out.store(output + vid * VecSize);
    }
}

template <typename T, int VecSize, typename Activation>
__global__ void elementwiseStridedRowsKernel(const T* __restrict__ input, T* __restrict__ output,
                                             int64_t rows, int64_t columns,
                                             int64_t input_row_stride, Activation activation) {
    const int64_t vec_columns = columns / VecSize;
    const int64_t total_vec_elements = rows * vec_columns;
    for (int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vid < total_vec_elements; vid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t row = vid / vec_columns;
        const int64_t column = (vid - row * vec_columns) * VecSize;

        Vec<T, VecSize> v_in;
        v_in.load(input + row * input_row_stride + column);

        Vec<T, VecSize> v_out;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            v_out[v] = static_cast<T>(activation(static_cast<float>(v_in[v])));
        }
        v_out.store(output + row * columns + column);
    }
}

// =============================================================================
// Swoosh-L / Swoosh-R Kernels (vectorized, elementwise over a flat buffer)
// =============================================================================

template <typename T, int VecSize>
__global__ void swooshLKernel(const T* __restrict__ input, T* __restrict__ output, int n) {
    const int total_vec_elements = n / VecSize;
    for (int vid = blockIdx.x * blockDim.x + threadIdx.x; vid < total_vec_elements;
         vid += gridDim.x * blockDim.x) {
        Vec<T, VecSize> v_in;
        v_in.load(input + vid * VecSize);

        Vec<T, VecSize> v_out;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            v_out[v] = static_cast<T>(oasr::swoosh_l(static_cast<float>(v_in[v])));
        }
        v_out.store(output + vid * VecSize);
    }
}

template <typename T, int VecSize>
__global__ void swooshRKernel(const T* __restrict__ input, T* __restrict__ output, int n) {
    const int total_vec_elements = n / VecSize;
    for (int vid = blockIdx.x * blockDim.x + threadIdx.x; vid < total_vec_elements;
         vid += gridDim.x * blockDim.x) {
        Vec<T, VecSize> v_in;
        v_in.load(input + vid * VecSize);

        Vec<T, VecSize> v_out;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            v_out[v] = static_cast<T>(oasr::swoosh_r(static_cast<float>(v_in[v])));
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
        gluKernel<T, kVecSize>
            <<<grid_size, block_size, 0, stream>>>(input, output, batch_size, seq_len, channels);
    } else {
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        gluKernel<T, 1>
            <<<grid_size, block_size, 0, stream>>>(input, output, batch_size, seq_len, channels);
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
        swishKernel<T, kVecSize>
            <<<grid_size, block_size, 0, stream>>>(input, output, batch_size, seq_len, channels);
    } else {
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        swishKernel<T, 1>
            <<<grid_size, block_size, 0, stream>>>(input, output, batch_size, seq_len, channels);
    }

    return cudaGetLastError();
}

template <typename T, typename Activation>
cudaError_t Elementwise(const T* input, T* output, int64_t n, Activation activation,
                        cudaStream_t stream) {
    if (n == 0) {
        return cudaSuccess;
    }

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;
    constexpr int kBlockSize = 256;

    if (n % kVecSize == 0 && oasr::isAligned<T, kVecSize>(input) &&
        oasr::isAligned<T, kVecSize>(output)) {
        const int64_t total_vec_elements = n / kVecSize;
        int64_t grid_size = (total_vec_elements + kBlockSize - 1) / kBlockSize;
        grid_size = std::min<int64_t>(grid_size, 65535);
        elementwiseKernel<T, kVecSize>
            <<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(input, output, n, activation);
    } else {
        int64_t grid_size = (n + kBlockSize - 1) / kBlockSize;
        grid_size = std::min<int64_t>(grid_size, 65535);
        elementwiseKernel<T, 1>
            <<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(input, output, n, activation);
    }

    return cudaGetLastError();
}

template <typename T, typename Activation>
cudaError_t ElementwiseStridedRows(const T* input, T* output, int64_t rows, int64_t columns,
                                   int64_t input_row_stride, Activation activation,
                                   cudaStream_t stream) {
    if (rows == 0 || columns == 0) {
        return cudaSuccess;
    }

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;
    constexpr int kBlockSize = 256;
    const bool aligned = columns % kVecSize == 0 && input_row_stride % kVecSize == 0 &&
                         oasr::isAligned<T, kVecSize>(input) &&
                         oasr::isAligned<T, kVecSize>(output);
    const int64_t elements_per_row = aligned ? columns / kVecSize : columns;
    int64_t grid_size = (rows * elements_per_row + kBlockSize - 1) / kBlockSize;
    grid_size = std::min<int64_t>(grid_size, 65535);

    if (aligned) {
        elementwiseStridedRowsKernel<T, kVecSize>
            <<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(input, output, rows, columns,
                                                                     input_row_stride, activation);
    } else {
        elementwiseStridedRowsKernel<T, 1><<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(
            input, output, rows, columns, input_row_stride, activation);
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t GeluErf(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, GeluErfActivation{}, stream);
}

template <typename T>
cudaError_t Sigmoid(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, SigmoidActivation{}, stream);
}

template <typename T>
cudaError_t Tanh(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, TanhActivation{}, stream);
}

template <typename T>
cudaError_t Relu(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, ReluActivation{}, stream);
}

template <typename T>
cudaError_t SwooshL(const T* input, T* output, int n, cudaStream_t stream) {
    constexpr int kVecSize = VecTypeTrait<T>::VecSize;
    const int block_size = 256;

    if (n % kVecSize == 0) {
        const int total_vec_elements = n / kVecSize;
        int grid_size = (total_vec_elements + block_size - 1) / block_size;
        grid_size = std::min(grid_size, 65535);
        swooshLKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(input, output, n);
    } else {
        int grid_size = (n + block_size - 1) / block_size;
        grid_size = std::min(grid_size, 65535);
        swooshLKernel<T, 1><<<grid_size, block_size, 0, stream>>>(input, output, n);
    }

    return cudaGetLastError();
}

template <typename T>
cudaError_t SwooshR(const T* input, T* output, int n, cudaStream_t stream) {
    constexpr int kVecSize = VecTypeTrait<T>::VecSize;
    const int block_size = 256;

    if (n % kVecSize == 0) {
        const int total_vec_elements = n / kVecSize;
        int grid_size = (total_vec_elements + block_size - 1) / block_size;
        grid_size = std::min(grid_size, 65535);
        swooshRKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(input, output, n);
    } else {
        int grid_size = (n + block_size - 1) / block_size;
        grid_size = std::min(grid_size, 65535);
        swooshRKernel<T, 1><<<grid_size, block_size, 0, stream>>>(input, output, n);
    }

    return cudaGetLastError();
}

}  // namespace activation
}  // namespace oasr
