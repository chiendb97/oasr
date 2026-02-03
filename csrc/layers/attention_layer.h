// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_layer.h"
#include "kernels/attention/attention_params.h"

namespace oasr {
namespace layers {

/**
 * @brief Configuration for attention layer
 */
struct AttentionConfig {
    int hidden_size;            // Model dimension
    int num_heads;              // Number of attention heads
    int num_kv_heads;           // Number of KV heads (for GQA, 0 = same as num_heads)
    int head_dim;               // Head dimension (0 = hidden_size / num_heads)
    
    AttentionType attention_type;
    bool use_bias;              // Use bias in projections
    float dropout_prob;
    
    // For relative position attention
    int max_relative_position;
    bool use_rotary;            // Use rotary position embeddings
    int rotary_dim;             // Dimension for rotary (0 = head_dim)
    
    AttentionConfig()
        : hidden_size(256), num_heads(4), num_kv_heads(0), head_dim(0)
        , attention_type(AttentionType::MULTI_HEAD)
        , use_bias(true), dropout_prob(0.0f)
        , max_relative_position(0), use_rotary(false), rotary_dim(0)
    {}
};

/**
 * @brief Multi-head attention layer
 */
class MultiHeadAttention : public StatefulLayer {
public:
    MultiHeadAttention(const std::string& name, const AttentionConfig& config,
                       DataType dtype = DataType::FP16, int device_id = 0);
    ~MultiHeadAttention() override;
    
    /**
     * @brief Forward pass for self-attention
     * 
     * @param input Input tensor [batch, seq_len, hidden_size]
     * @param attention_mask Optional mask [batch, 1, seq_len, seq_len]
     * @param stream CUDA stream
     * @return Output tensor [batch, seq_len, hidden_size]
     */
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    /**
     * @brief Forward with explicit mask
     */
    Tensor forward(const Tensor& input, const Tensor* attention_mask,
                   cudaStream_t stream = nullptr);
    
    /**
     * @brief Incremental forward for streaming inference
     * 
     * @param input New input token(s) [batch, 1, hidden_size]
     * @param stream CUDA stream
     * @return Output [batch, 1, hidden_size]
     */
    Tensor forwardIncremental(const Tensor& input, cudaStream_t stream = nullptr);
    
    // StatefulLayer interface
    void resetState() override;
    std::unordered_map<std::string, Tensor> getState() const override;
    void setState(const std::unordered_map<std::string, Tensor>& state) override;
    
    // BaseLayer interface
    std::vector<std::string> weightNames() const override;
    
    // Configuration access
    const AttentionConfig& config() const { return config_; }
    
    /**
     * @brief Initialize KV cache for incremental decoding
     * 
     * @param batch_size Batch size
     * @param max_seq_len Maximum sequence length to cache
     */
    void initKVCache(int batch_size, int max_seq_len);
    
private:
    AttentionConfig config_;
    kernels::KVCache kv_cache_;
    
    // Internal buffers
    TensorPtr q_buf_, k_buf_, v_buf_;
    TensorPtr attn_output_buf_;
    
    void allocateBuffers(int batch_size, int seq_len);
};

/**
 * @brief Conformer-style relative position attention
 */
class RelativePositionAttention : public MultiHeadAttention {
public:
    RelativePositionAttention(const std::string& name, const AttentionConfig& config,
                              DataType dtype = DataType::FP16, int device_id = 0);
    ~RelativePositionAttention() override;
    
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    std::vector<std::string> weightNames() const override;
    
    /**
     * @brief Generate or update position embeddings for given sequence length
     */
    void updatePositionEmbeddings(int seq_len, cudaStream_t stream);
    
private:
    TensorPtr pos_emb_;         // Position embeddings
    int cached_seq_len_ = 0;    // Cached position embedding length
};

} // namespace layers
} // namespace oasr
