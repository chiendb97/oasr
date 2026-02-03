// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ffn_params.h"
#include "common/tensor.h"

namespace oasr {
namespace kernels {

/**
 * @brief Feed-forward network
 * 
 * Standard FFN: output = Linear2(Activation(Linear1(input)))
 * 
 * @param params FFN parameters
 */
void invokeFFN(const FFNParams& params);

/**
 * @brief Gated feed-forward network (SwiGLU, GeGLU, etc.)
 * 
 * Gated FFN: output = Linear2((Activation(Linear1(input)) * Gate(input)))
 * 
 * @param params FFN parameters (use_gated should be true)
 */
void invokeGatedFFN(const FFNParams& params);

/**
 * @brief Conformer feed-forward module
 * 
 * Full Conformer FFN with optional fused operations:
 * output = input + scale * Dropout(Linear2(Swish(Linear1(LayerNorm(input)))))
 * 
 * @param params Conformer FFN parameters
 */
void invokeConformerFFN(const ConformerFFNParams& params);

/**
 * @brief Fused Linear + Activation kernel
 * 
 * Computes: output = Activation(input @ weight.T + bias)
 * 
 * @param input Input tensor [batch, seq_len, in_features]
 * @param weight Weight tensor [out_features, in_features]
 * @param bias Optional bias [out_features]
 * @param output Output tensor [batch, seq_len, out_features]
 * @param activation Activation function to apply
 */
void invokeLinearActivation(const void* input, const void* weight, const void* bias,
                            void* output, int batch_size, int seq_len,
                            int in_features, int out_features,
                            ActivationType activation,
                            DataType dtype, cudaStream_t stream);

/**
 * @brief Fused Linear + Gated Activation kernel
 * 
 * Computes: output = (Activation(input @ W1.T + b1)) * (input @ Wg.T + bg)
 * Or with fused weights: 
 * [act_out, gate_out] = split(input @ W.T + b)
 * output = Activation(act_out) * gate_out
 * 
 * @param input Input [batch, seq, in_features]
 * @param weight Fused weight [2 * out_features, in_features]
 * @param bias Fused bias [2 * out_features]
 * @param output Output [batch, seq, out_features]
 * @param activation Activation for the non-gate path
 */
void invokeLinearGatedActivation(const void* input, const void* weight, const void* bias,
                                 void* output, int batch_size, int seq_len,
                                 int in_features, int out_features,
                                 ActivationType activation,
                                 DataType dtype, cudaStream_t stream);

/**
 * @brief Fused Add + Linear kernel (for residual connections)
 * 
 * Computes: output = residual + scale * (input @ weight.T + bias)
 */
void invokeResidualLinear(const void* input, const void* residual,
                          const void* weight, const void* bias,
                          void* output, int batch_size, int seq_len,
                          int in_features, int out_features,
                          float scale, DataType dtype, cudaStream_t stream);

// Activation kernels
void invokeReLU(const void* input, void* output, int size, DataType dtype, cudaStream_t stream);
void invokeGELU(const void* input, void* output, int size, DataType dtype, cudaStream_t stream);
void invokeSwish(const void* input, void* output, int size, DataType dtype, cudaStream_t stream);
void invokeSiLU(const void* input, void* output, int size, DataType dtype, cudaStream_t stream);

// Fused activation + multiplication (for gated units)
// output = activation(a) * b
void invokeGatedActivation(const void* a, const void* b, void* output,
                           int size, ActivationType activation,
                           DataType dtype, cudaStream_t stream);

// Template specializations
template <typename T>
void invokeFFNTyped(const FFNParams& params);

template <typename T>
void invokeGatedFFNTyped(const FFNParams& params);

} // namespace kernels
} // namespace oasr
