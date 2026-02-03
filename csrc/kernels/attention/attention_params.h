// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/types.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace oasr {
namespace kernels {

/**
 * @brief Parameters for multi-head attention kernel
 * 
 * Supports various attention mechanisms used in ASR:
 * - Standard multi-head attention
 * - Relative positional attention (Conformer, Transformer-XL)
 * - Rotary positional embeddings (RoPE)
 */
struct AttentionParams {
    // Input tensors
    const void* q;              // Query: [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
    const void* k;              // Key: same shape as Q
    const void* v;              // Value: same shape as Q
    
    // Output tensor
    void* output;               // Output: same shape as Q
    
    // Optional: attention weights output (for debugging/visualization)
    void* attn_weights;         // [batch, num_heads, seq_len, seq_len] or nullptr
    
    // Relative position embeddings (for Conformer)
    const void* pos_emb;        // [seq_len, head_dim] or [2*seq_len-1, head_dim]
    const void* pos_bias_u;     // [num_heads, head_dim] - content bias
    const void* pos_bias_v;     // [num_heads, head_dim] - position bias
    
    // Attention mask
    const void* attention_mask; // [batch, 1, seq_len, seq_len] or nullptr
    const int* sequence_lengths;// [batch] - actual sequence lengths for each batch
    
    // Dimensions
    int batch_size;
    int seq_len;
    int num_heads;
    int head_dim;
    int num_kv_heads;           // For grouped-query attention (GQA)
    
    // Scaling
    float scale;                // Usually 1/sqrt(head_dim)
    float dropout_prob;         // Dropout probability (0 = no dropout)
    
    // Configuration
    AttentionType attention_type;
    DataType dtype;
    bool is_causal;             // Apply causal mask
    bool use_flash_attention;   // Use FlashAttention if available
    bool return_softmax;        // Return softmax weights
    
    // For streaming/incremental decoding
    bool is_incremental;        // Whether this is incremental decoding
    int cache_seq_len;          // Length of cached K/V
    
    // Memory layout
    bool qkv_interleaved;       // If true: [batch, seq, 3, heads, dim], else separate Q/K/V
    bool head_first;            // If true: [batch, heads, seq, dim], else [batch, seq, heads, dim]
    
    // Stream for async execution
    cudaStream_t stream;
    
    // Constructor with sensible defaults
    AttentionParams()
        : q(nullptr), k(nullptr), v(nullptr)
        , output(nullptr), attn_weights(nullptr)
        , pos_emb(nullptr), pos_bias_u(nullptr), pos_bias_v(nullptr)
        , attention_mask(nullptr), sequence_lengths(nullptr)
        , batch_size(0), seq_len(0), num_heads(0), head_dim(0), num_kv_heads(0)
        , scale(1.0f), dropout_prob(0.0f)
        , attention_type(AttentionType::MULTI_HEAD)
        , dtype(DataType::FP16)
        , is_causal(false), use_flash_attention(true), return_softmax(false)
        , is_incremental(false), cache_seq_len(0)
        , qkv_interleaved(false), head_first(false)
        , stream(nullptr)
    {}
};

/**
 * @brief Parameters for relative position attention (Conformer-style)
 */
struct RelativePositionAttentionParams : public AttentionParams {
    // Relative position embedding method
    enum class Method {
        SHAW,           // Shaw et al. - relative position representations
        XL,             // Transformer-XL style
        CONFORMER,      // Conformer style with content and position biases
    };
    
    Method method;
    int max_relative_position;  // Maximum relative position to consider
    bool clamp_positions;       // Clamp positions outside range
    
    RelativePositionAttentionParams()
        : AttentionParams()
        , method(Method::CONFORMER)
        , max_relative_position(0)
        , clamp_positions(true)
    {
        attention_type = AttentionType::RELATIVE_POSITION;
    }
};

/**
 * @brief KV Cache structure for incremental decoding
 */
struct KVCache {
    void* k_cache;              // [batch, max_seq_len, num_kv_heads, head_dim]
    void* v_cache;              // [batch, max_seq_len, num_kv_heads, head_dim]
    int max_seq_len;            // Maximum sequence length the cache can hold
    int current_len;            // Current valid length in cache
    DataType dtype;
    
    KVCache()
        : k_cache(nullptr), v_cache(nullptr)
        , max_seq_len(0), current_len(0)
        , dtype(DataType::FP16)
    {}
};

} // namespace kernels
} // namespace oasr
