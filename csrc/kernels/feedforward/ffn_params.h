// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/types.h"
#include <cuda_runtime.h>

namespace oasr {
namespace kernels {

/**
 * @brief Parameters for feed-forward network kernels
 * 
 * Supports various FFN architectures:
 * - Standard FFN: Linear -> Activation -> Linear
 * - Gated FFN: Linear -> (Activation * Gate) -> Linear
 * - Conformer FFN: Linear -> Swish -> Dropout -> Linear
 */
struct FFNParams {
    // Input [batch, seq_len, d_model]
    const void* input;
    
    // Output [batch, seq_len, d_model]
    void* output;
    
    // First linear layer
    const void* weight1;        // [d_ff, d_model] or [2 * d_ff, d_model] for gated
    const void* bias1;          // [d_ff] or [2 * d_ff]
    
    // Second linear layer  
    const void* weight2;        // [d_model, d_ff]
    const void* bias2;          // [d_model]
    
    // For gated variants (SwiGLU, etc.)
    const void* gate_weight;    // [d_ff, d_model] (alternative to fused weight1)
    const void* gate_bias;      // [d_ff]
    
    // Dimensions
    int batch_size;
    int seq_len;
    int d_model;
    int d_ff;                   // Intermediate dimension (usually 4 * d_model)
    
    // Configuration
    ActivationType activation;
    DataType dtype;
    float dropout_prob;
    bool use_gated;             // Use gated activation (SwiGLU, GeGLU, etc.)
    bool fused_gated_weights;   // If true, weight1 is [2*d_ff, d_model] containing both
    
    // Optional intermediate buffer (for non-fused execution)
    void* intermediate;         // [batch, seq_len, d_ff]
    
    cudaStream_t stream;
    
    FFNParams()
        : input(nullptr), output(nullptr)
        , weight1(nullptr), bias1(nullptr)
        , weight2(nullptr), bias2(nullptr)
        , gate_weight(nullptr), gate_bias(nullptr)
        , batch_size(0), seq_len(0), d_model(0), d_ff(0)
        , activation(ActivationType::SWISH), dtype(DataType::FP16)
        , dropout_prob(0.0f), use_gated(false), fused_gated_weights(false)
        , intermediate(nullptr), stream(nullptr)
    {}
};

/**
 * @brief Parameters for Conformer feed-forward module
 * 
 * Conformer uses a specific FFN structure with half-step residuals:
 * output = input + 0.5 * FFN(LayerNorm(input))
 */
struct ConformerFFNParams : public FFNParams {
    // Layer normalization parameters
    const void* ln_weight;      // [d_model]
    const void* ln_bias;        // [d_model]
    
    // Residual scaling
    float residual_scale;       // Usually 0.5 for Conformer
    bool fuse_residual;         // Fuse residual addition
    bool fuse_layernorm;        // Fuse layer normalization
    
    ConformerFFNParams()
        : FFNParams()
        , ln_weight(nullptr), ln_bias(nullptr)
        , residual_scale(0.5f), fuse_residual(true), fuse_layernorm(true)
    {
        activation = ActivationType::SWISH;
    }
};

} // namespace kernels
} // namespace oasr
