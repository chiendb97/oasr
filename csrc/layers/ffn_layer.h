// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_layer.h"
#include "kernels/feedforward/ffn_params.h"

namespace oasr {
namespace layers {

/**
 * @brief Configuration for feed-forward network
 */
struct FFNConfig {
    int d_model;                // Input/output dimension
    int d_ff;                   // Intermediate dimension (0 = 4 * d_model)
    ActivationType activation;
    float dropout_prob;
    bool use_bias;
    bool use_gated;             // Use gated activation (SwiGLU, etc.)
    
    FFNConfig()
        : d_model(256), d_ff(0)
        , activation(ActivationType::SWISH)
        , dropout_prob(0.0f), use_bias(true), use_gated(false)
    {}
};

/**
 * @brief Standard feed-forward network layer
 * 
 * FFN(x) = Linear2(Activation(Linear1(x)))
 */
class FeedForward : public BaseLayer {
public:
    FeedForward(const std::string& name, const FFNConfig& config,
                DataType dtype = DataType::FP16, int device_id = 0);
    ~FeedForward() override;
    
    /**
     * @brief Forward pass
     * 
     * @param input Input tensor [batch, seq_len, d_model]
     * @param stream CUDA stream
     * @return Output tensor [batch, seq_len, d_model]
     */
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    std::vector<std::string> weightNames() const override;
    
    const FFNConfig& config() const { return config_; }
    
private:
    FFNConfig config_;
    TensorPtr intermediate_buf_;
    
    void allocateBuffers(int batch_size, int seq_len);
};

/**
 * @brief Gated feed-forward network (SwiGLU, GeGLU, etc.)
 * 
 * GatedFFN(x) = Linear2(Activation(Linear1(x)) * Gate(x))
 */
class GatedFeedForward : public FeedForward {
public:
    GatedFeedForward(const std::string& name, const FFNConfig& config,
                     DataType dtype = DataType::FP16, int device_id = 0);
    
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    std::vector<std::string> weightNames() const override;
};

/**
 * @brief Conformer feed-forward module
 * 
 * Uses half-step residual: output = input + 0.5 * FFN(LayerNorm(input))
 */
class ConformerFeedForward : public BaseLayer {
public:
    ConformerFeedForward(const std::string& name, const FFNConfig& config,
                         DataType dtype = DataType::FP16, int device_id = 0);
    ~ConformerFeedForward() override;
    
    /**
     * @brief Forward pass with half-step residual
     * 
     * @param input Input tensor [batch, seq_len, d_model]
     * @param stream CUDA stream
     * @return Output tensor [batch, seq_len, d_model]
     */
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    /**
     * @brief Forward without residual (just the FFN part)
     */
    Tensor forwardNoResidual(const Tensor& input, cudaStream_t stream = nullptr);
    
    std::vector<std::string> weightNames() const override;
    
    const FFNConfig& config() const { return config_; }
    
private:
    FFNConfig config_;
    TensorPtr intermediate_buf_;
    TensorPtr ln_output_buf_;
    
    void allocateBuffers(int batch_size, int seq_len);
};

/**
 * @brief Linear projection layer
 */
class Linear : public BaseLayer {
public:
    Linear(const std::string& name, int in_features, int out_features,
           bool use_bias = true, DataType dtype = DataType::FP16, int device_id = 0);
    ~Linear() override;
    
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    std::vector<std::string> weightNames() const override;
    
    int inFeatures() const { return in_features_; }
    int outFeatures() const { return out_features_; }
    
private:
    int in_features_;
    int out_features_;
    bool use_bias_;
};

} // namespace layers
} // namespace oasr
