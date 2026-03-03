// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "attention_params.h"
#include <cuda_runtime.h>

namespace oasr {
namespace kernels {

/**
 * @brief Multi-head attention kernel
 * 
 * Computes: output = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
 * 
 * This dispatcher selects the optimal kernel based on:
 * - Data type (FP32, FP16, BF16)
 * - Attention type (standard, relative position, etc.)
 * - Hardware capabilities (FlashAttention support)
 * 
 * @param params Attention parameters
 */
void invokeAttention(const AttentionParams& params);

/**
 * @brief Relative position attention kernel (Conformer-style)
 * 
 * Computes attention with relative position embeddings:
 * A_ij = (q_i + u)^T k_j + (q_i + v)^T r_{i-j}
 * 
 * where u and v are learnable biases, and r_{i-j} is the relative position embedding.
 * 
 * @param params Relative position attention parameters
 */
void invokeRelativePositionAttention(const RelativePositionAttentionParams& params);

/**
 * @brief Compute Q @ K^T with optional relative position bias
 * 
 * @param q Query tensor [batch, heads, seq_q, head_dim]
 * @param k Key tensor [batch, heads, seq_k, head_dim]
 * @param output Output tensor [batch, heads, seq_q, seq_k]
 * @param scale Scaling factor
 * @param pos_bias Optional position bias [heads, seq_q, seq_k]
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeQKDot(const void* q, const void* k, void* output,
                 int batch_size, int num_heads, int seq_q, int seq_k, int head_dim,
                 float scale, const void* pos_bias,
                 DataType dtype, cudaStream_t stream);

/**
 * @brief Apply softmax to attention scores
 * 
 * @param input Attention scores [batch, heads, seq_q, seq_k]
 * @param output Softmax output (can be same as input for in-place)
 * @param mask Optional attention mask [batch, 1, seq_q, seq_k]
 * @param seq_lens Optional sequence lengths [batch]
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeSoftmax(const void* input, void* output,
                   int batch_size, int num_heads, int seq_q, int seq_k,
                   const void* mask, const int* seq_lens,
                   bool is_causal, DataType dtype, cudaStream_t stream);

/**
 * @brief Compute attention output: attn_weights @ V
 * 
 * @param attn_weights Attention weights [batch, heads, seq_q, seq_k]
 * @param v Value tensor [batch, heads, seq_k, head_dim]
 * @param output Output tensor [batch, heads, seq_q, head_dim]
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeAttentionValue(const void* attn_weights, const void* v, void* output,
                          int batch_size, int num_heads, int seq_q, int seq_k, int head_dim,
                          DataType dtype, cudaStream_t stream);

/**
 * @brief Fused QKV projection kernel
 * 
 * Projects input to Q, K, V using a single fused kernel
 * 
 * @param input Input tensor [batch, seq, hidden]
 * @param weight Weight tensor [3 * hidden, hidden] or [hidden, 3 * hidden]
 * @param bias Optional bias [3 * hidden]
 * @param q Output Q [batch, seq, num_heads, head_dim]
 * @param k Output K [batch, seq, num_kv_heads, head_dim]
 * @param v Output V [batch, seq, num_kv_heads, head_dim]
 */
void invokeQKVProjection(const void* input, const void* weight, const void* bias,
                         void* q, void* k, void* v,
                         int batch_size, int seq_len, int hidden_size,
                         int num_heads, int num_kv_heads, int head_dim,
                         DataType dtype, cudaStream_t stream);

/**
 * @brief Update KV cache for incremental decoding
 * 
 * @param cache KV cache structure
 * @param k New key tensor [batch, 1, num_kv_heads, head_dim]
 * @param v New value tensor [batch, 1, num_kv_heads, head_dim]
 * @param positions Position indices for each batch element [batch]
 * @param stream CUDA stream
 */
void invokeUpdateKVCache(KVCache& cache, const void* k, const void* v,
                         const int* positions, int batch_size,
                         int num_kv_heads, int head_dim,
                         DataType dtype, cudaStream_t stream);

/**
 * @brief Generate relative position embeddings
 * 
 * @param output Output embeddings [seq_len, head_dim] or [2*seq_len-1, head_dim]
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param method Embedding method (sinusoidal, learnable)
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeGenerateRelativePosEmb(void* output, int seq_len, int head_dim,
                                  int max_relative_position,
                                  DataType dtype, cudaStream_t stream);

// Template specializations for different data types
template <typename T>
void invokeAttentionTyped(const AttentionParams& params);

template <typename T>
void invokeRelativePositionAttentionTyped(const RelativePositionAttentionParams& params);

} // namespace kernels
} // namespace oasr
