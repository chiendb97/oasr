// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/tensor.h"
#include "common/cuda_utils.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace oasr {
namespace layers {

/**
 * @brief Base class for all layers
 * 
 * Provides common functionality for inference layers:
 * - Weight management
 * - Input/output buffer management
 * - Stream management
 */
class BaseLayer {
public:
    BaseLayer(const std::string& name, DataType dtype = DataType::FP16, int device_id = 0);
    virtual ~BaseLayer() = default;
    
    // Disable copy
    BaseLayer(const BaseLayer&) = delete;
    BaseLayer& operator=(const BaseLayer&) = delete;
    
    /**
     * @brief Forward pass
     * 
     * @param input Input tensor
     * @param stream CUDA stream for execution
     * @return Output tensor
     */
    virtual Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) = 0;
    
    /**
     * @brief Get layer name
     */
    const std::string& name() const { return name_; }
    
    /**
     * @brief Get data type
     */
    DataType dtype() const { return dtype_; }
    
    /**
     * @brief Get device ID
     */
    int deviceId() const { return device_id_; }
    
    /**
     * @brief Load weights from a map of name -> tensor
     */
    virtual void loadWeights(const std::unordered_map<std::string, Tensor>& weights);
    
    /**
     * @brief Get all weight names this layer expects
     */
    virtual std::vector<std::string> weightNames() const = 0;
    
    /**
     * @brief Get total number of parameters
     */
    size_t numParameters() const;
    
    /**
     * @brief Get total memory used by weights in bytes
     */
    size_t weightMemoryBytes() const;
    
    /**
     * @brief Convert weights to a different data type
     */
    void convertWeights(DataType new_dtype);
    
    /**
     * @brief Check if layer is ready for inference
     */
    bool isInitialized() const { return initialized_; }
    
protected:
    std::string name_;
    DataType dtype_;
    int device_id_;
    bool initialized_ = false;
    
    // Weight storage
    std::unordered_map<std::string, TensorPtr> weights_;
    
    // Helper to register a weight
    void registerWeight(const std::string& name, const std::vector<int64_t>& shape);
    
    // Helper to get a weight tensor
    Tensor& getWeight(const std::string& name);
    const Tensor& getWeight(const std::string& name) const;
};

/**
 * @brief Layer with internal state (for streaming/incremental inference)
 */
class StatefulLayer : public BaseLayer {
public:
    using BaseLayer::BaseLayer;
    
    /**
     * @brief Reset internal state
     */
    virtual void resetState() = 0;
    
    /**
     * @brief Get current state for checkpointing
     */
    virtual std::unordered_map<std::string, Tensor> getState() const = 0;
    
    /**
     * @brief Set state from checkpoint
     */
    virtual void setState(const std::unordered_map<std::string, Tensor>& state) = 0;
};

} // namespace layers
} // namespace oasr
