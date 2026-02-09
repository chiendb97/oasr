// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/types.h"
#include <cuda_runtime.h>

namespace oasr {
namespace kernels {

/** Backend for pointwise (1x1) convolution: native CUDA kernel or CUTLASS GEMM. */
enum class PointwiseConvBackend {
    NATIVE = 0,
    CUTLASS = 1,
};

/**
 * @brief Parameters for 1D convolution kernels
 * 
 * Used for the convolution module in Conformer and similar architectures.
 * Supports depthwise separable convolutions, which are the primary conv type in ASR.
 */
struct Conv1DParams {
    // Input tensor [batch, seq_len, channels] or [batch, channels, seq_len]
    const void* input;
    
    // Output tensor (same layout as input)
    void* output;
    
    // Convolution weight [out_channels, in_channels/groups, kernel_size]
    const void* weight;
    
    // Optional bias [out_channels]
    const void* bias;
    
    // Dimensions
    int batch_size;
    int seq_len;
    int in_channels;
    int out_channels;
    int kernel_size;
    
    // Convolution parameters
    int stride;
    int padding;
    int dilation;
    int groups;                 // For depthwise: groups = in_channels
    
    // Configuration
    ConvType conv_type;
    DataType dtype;
    bool channels_last;         // If true: [batch, seq, channels], else [batch, channels, seq]
    bool is_causal;             // For causal/streaming convolution
    
    // Activation fused with convolution
    ActivationType activation;
    bool fuse_activation;
    
    // Stream
    cudaStream_t stream;
    
    Conv1DParams()
        : input(nullptr), output(nullptr), weight(nullptr), bias(nullptr)
        , batch_size(0), seq_len(0), in_channels(0), out_channels(0), kernel_size(0)
        , stride(1), padding(0), dilation(1), groups(1)
        , conv_type(ConvType::STANDARD), dtype(DataType::FP16)
        , channels_last(true), is_causal(false)
        , activation(ActivationType::SWISH), fuse_activation(false)
        , stream(nullptr)
    {}
};

/**
 * @brief Parameters for Conformer convolution module
 * 
 * The Conformer conv module consists of:
 * 1. Pointwise conv (expand channels)
 * 2. GLU activation
 * 3. Depthwise conv
 * 4. Batch norm
 * 5. Activation (Swish)
 * 6. Pointwise conv (project back)
 */
struct ConformerConvParams {
    // Input [batch, seq_len, d_model]
    const void* input;
    
    // Output [batch, seq_len, d_model]
    void* output;
    
    // Weights
    const void* pointwise_conv1_weight;     // [2 * d_model, d_model]
    const void* pointwise_conv1_bias;       // [2 * d_model]
    const void* depthwise_conv_weight;      // [d_model, 1, kernel_size]
    const void* depthwise_conv_bias;        // [d_model]
    const void* batch_norm_weight;          // [d_model]
    const void* batch_norm_bias;            // [d_model]
    const void* batch_norm_mean;            // [d_model] running mean
    const void* batch_norm_var;             // [d_model] running variance
    const void* pointwise_conv2_weight;     // [d_model, d_model]
    const void* pointwise_conv2_bias;       // [d_model]
    
    // Dimensions
    int batch_size;
    int seq_len;
    int d_model;
    int kernel_size;
    
    // Configuration
    DataType dtype;
    float batch_norm_eps;
    bool is_causal;             // For streaming inference
    
    // Optional: mask for variable length sequences
    const int* sequence_lengths;    // [batch]
    
    cudaStream_t stream;
    
    ConformerConvParams()
        : input(nullptr), output(nullptr)
        , pointwise_conv1_weight(nullptr), pointwise_conv1_bias(nullptr)
        , depthwise_conv_weight(nullptr), depthwise_conv_bias(nullptr)
        , batch_norm_weight(nullptr), batch_norm_bias(nullptr)
        , batch_norm_mean(nullptr), batch_norm_var(nullptr)
        , pointwise_conv2_weight(nullptr), pointwise_conv2_bias(nullptr)
        , batch_size(0), seq_len(0), d_model(0), kernel_size(31)
        , dtype(DataType::FP16), batch_norm_eps(1e-5f), is_causal(false)
        , sequence_lengths(nullptr), stream(nullptr)
    {}
};

/**
 * @brief State for causal/streaming convolution
 */
struct ConvState {
    void* buffer;               // [batch, kernel_size - 1, channels]
    int buffer_size;
    int channels;
    DataType dtype;
    
    ConvState()
        : buffer(nullptr), buffer_size(0), channels(0), dtype(DataType::FP16)
    {}
};

} // namespace kernels
} // namespace oasr
