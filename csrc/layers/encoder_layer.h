// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_layer.h"
#include "attention_layer.h"
#include "conv_layer.h"
#include "ffn_layer.h"

namespace oasr {
namespace layers {

/**
 * @brief Configuration for encoder layer
 */
struct EncoderLayerConfig {
    int d_model;                // Model dimension
    int num_heads;              // Number of attention heads
    int d_ff;                   // FFN intermediate dimension
    int conv_kernel_size;       // Convolution kernel size (Conformer)
    
    AttentionType attention_type;
    ActivationType ffn_activation;
    
    float dropout_prob;
    float attention_dropout_prob;
    
    bool use_macaron_ffn;       // Use Macaron-style FFN (Conformer)
    bool causal;                // Causal attention/conv for streaming
    
    EncoderLayerConfig()
        : d_model(256), num_heads(4), d_ff(2048), conv_kernel_size(31)
        , attention_type(AttentionType::RELATIVE_POSITION)
        , ffn_activation(ActivationType::SWISH)
        , dropout_prob(0.1f), attention_dropout_prob(0.0f)
        , use_macaron_ffn(true), causal(false)
    {}
};

/**
 * @brief Conformer encoder layer
 * 
 * Structure:
 * x = x + 0.5 * FFN(x)              # First half-step FFN
 * x = x + MHSA(x)                   # Multi-head self-attention
 * x = x + Conv(x)                   # Convolution module
 * x = x + 0.5 * FFN(x)              # Second half-step FFN
 * x = LayerNorm(x)                  # Final layer norm
 */
class ConformerEncoderLayer : public StatefulLayer {
public:
    ConformerEncoderLayer(const std::string& name, const EncoderLayerConfig& config,
                          DataType dtype = DataType::FP16, int device_id = 0);
    ~ConformerEncoderLayer() override;
    
    /**
     * @brief Forward pass
     * 
     * @param input Input tensor [batch, seq_len, d_model]
     * @param attention_mask Optional attention mask
     * @param stream CUDA stream
     * @return Output tensor [batch, seq_len, d_model]
     */
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    Tensor forward(const Tensor& input, const Tensor* attention_mask,
                   cudaStream_t stream = nullptr);
    
    /**
     * @brief Streaming forward for chunk-based processing
     */
    Tensor forwardStreaming(const Tensor& input, cudaStream_t stream = nullptr);
    
    // StatefulLayer interface
    void resetState() override;
    std::unordered_map<std::string, Tensor> getState() const override;
    void setState(const std::unordered_map<std::string, Tensor>& state) override;
    
    std::vector<std::string> weightNames() const override;
    void loadWeights(const std::unordered_map<std::string, Tensor>& weights) override;
    
    const EncoderLayerConfig& config() const { return config_; }
    
private:
    EncoderLayerConfig config_;
    
    // Sub-layers
    std::unique_ptr<ConformerFeedForward> ffn1_;
    std::unique_ptr<RelativePositionAttention> self_attn_;
    std::unique_ptr<ConformerConvModule> conv_module_;
    std::unique_ptr<ConformerFeedForward> ffn2_;
    
    // Layer norms
    TensorPtr ln_final_weight_, ln_final_bias_;
};

/**
 * @brief Transformer encoder layer
 * 
 * Standard transformer encoder:
 * x = x + MHSA(LayerNorm(x))
 * x = x + FFN(LayerNorm(x))
 */
class TransformerEncoderLayer : public StatefulLayer {
public:
    TransformerEncoderLayer(const std::string& name, const EncoderLayerConfig& config,
                            DataType dtype = DataType::FP16, int device_id = 0);
    ~TransformerEncoderLayer() override;
    
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    void resetState() override;
    std::unordered_map<std::string, Tensor> getState() const override;
    void setState(const std::unordered_map<std::string, Tensor>& state) override;
    
    std::vector<std::string> weightNames() const override;
    
private:
    EncoderLayerConfig config_;
    
    std::unique_ptr<MultiHeadAttention> self_attn_;
    std::unique_ptr<FeedForward> ffn_;
    
    TensorPtr ln1_weight_, ln1_bias_;
    TensorPtr ln2_weight_, ln2_bias_;
};

/**
 * @brief Branchformer encoder layer
 * 
 * Parallel branches:
 * x = x + Merge(MHSA(x), Conv(x))
 * x = x + FFN(LayerNorm(x))
 */
class BranchformerEncoderLayer : public StatefulLayer {
public:
    BranchformerEncoderLayer(const std::string& name, const EncoderLayerConfig& config,
                             DataType dtype = DataType::FP16, int device_id = 0);
    ~BranchformerEncoderLayer() override;
    
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    void resetState() override;
    std::unordered_map<std::string, Tensor> getState() const override;
    void setState(const std::unordered_map<std::string, Tensor>& state) override;
    
    std::vector<std::string> weightNames() const override;
    
    // Merge strategies
    enum class MergeMethod {
        CONCAT_LINEAR,          // Concat + Linear projection
        WEIGHTED_SUM,           // Learnable weighted sum
        GATED,                  // Gated merge
    };
    
private:
    EncoderLayerConfig config_;
    MergeMethod merge_method_ = MergeMethod::CONCAT_LINEAR;
    
    std::unique_ptr<MultiHeadAttention> self_attn_;
    std::unique_ptr<ConformerConvModule> conv_module_;
    std::unique_ptr<FeedForward> ffn_;
    std::unique_ptr<Linear> merge_linear_;
    
    TensorPtr ln1_weight_, ln1_bias_;
    TensorPtr ln2_weight_, ln2_bias_;
};

} // namespace layers
} // namespace oasr
