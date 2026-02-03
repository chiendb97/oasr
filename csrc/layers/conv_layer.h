// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_layer.h"
#include "kernels/convolution/conv_params.h"

namespace oasr {
namespace layers {

/**
 * @brief Configuration for convolution module
 */
struct ConvModuleConfig {
    int channels;               // Input/output channels
    int kernel_size;            // Convolution kernel size
    float dropout_prob;
    bool causal;                // Causal convolution for streaming
    float batch_norm_eps;
    
    ConvModuleConfig()
        : channels(256), kernel_size(31), dropout_prob(0.0f)
        , causal(false), batch_norm_eps(1e-5f)
    {}
};

/**
 * @brief Conformer convolution module
 * 
 * Implements the convolution module from Conformer:
 * 1. Pointwise Conv (expand to 2x channels)
 * 2. GLU activation
 * 3. Depthwise Conv
 * 4. BatchNorm
 * 5. Swish activation
 * 6. Pointwise Conv (project back)
 */
class ConformerConvModule : public StatefulLayer {
public:
    ConformerConvModule(const std::string& name, const ConvModuleConfig& config,
                        DataType dtype = DataType::FP16, int device_id = 0);
    ~ConformerConvModule() override;
    
    /**
     * @brief Forward pass
     * 
     * @param input Input tensor [batch, seq_len, channels]
     * @param stream CUDA stream
     * @return Output tensor [batch, seq_len, channels]
     */
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    /**
     * @brief Streaming forward with state management
     * 
     * @param input Input chunk [batch, chunk_len, channels]
     * @param stream CUDA stream
     * @return Output chunk [batch, chunk_len, channels]
     */
    Tensor forwardStreaming(const Tensor& input, cudaStream_t stream = nullptr);
    
    // StatefulLayer interface
    void resetState() override;
    std::unordered_map<std::string, Tensor> getState() const override;
    void setState(const std::unordered_map<std::string, Tensor>& state) override;
    
    // BaseLayer interface
    std::vector<std::string> weightNames() const override;
    
    // Configuration access
    const ConvModuleConfig& config() const { return config_; }
    
private:
    ConvModuleConfig config_;
    kernels::ConvState conv_state_;
    
    // Internal buffers
    TensorPtr intermediate_buf_;
    TensorPtr glu_buf_;
    TensorPtr depthwise_buf_;
    
    void allocateBuffers(int batch_size, int seq_len);
};

/**
 * @brief Generic 1D convolution layer
 */
class Conv1D : public BaseLayer {
public:
    Conv1D(const std::string& name, int in_channels, int out_channels,
           int kernel_size, int stride = 1, int padding = 0, int groups = 1,
           bool use_bias = true, DataType dtype = DataType::FP16, int device_id = 0);
    ~Conv1D() override;
    
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    std::vector<std::string> weightNames() const override;
    
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int groups_;
    bool use_bias_;
};

/**
 * @brief Depthwise separable convolution layer
 */
class DepthwiseSeparableConv1D : public BaseLayer {
public:
    DepthwiseSeparableConv1D(const std::string& name, int channels, int kernel_size,
                              int pointwise_out_channels = 0,
                              DataType dtype = DataType::FP16, int device_id = 0);
    ~DepthwiseSeparableConv1D() override;
    
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) override;
    
    std::vector<std::string> weightNames() const override;
    
private:
    int channels_;
    int kernel_size_;
    int pointwise_out_channels_;
};

} // namespace layers
} // namespace oasr
