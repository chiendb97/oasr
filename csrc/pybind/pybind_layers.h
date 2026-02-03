// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "layers/base_layer.h"
#include "layers/attention_layer.h"
#include "layers/conv_layer.h"
#include "layers/ffn_layer.h"
#include "layers/encoder_layer.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

/**
 * @brief Register layer-related Python bindings
 */
inline void registerLayerBindings(py::module_& m) {
    // Create layers submodule
    py::module_ layers = m.def_submodule("layers", "Neural network layers");
    
    // ============== Configuration Classes ==============
    
    // AttentionConfig
    py::class_<layers::AttentionConfig>(layers, "AttentionConfig")
        .def(py::init<>())
        .def_readwrite("hidden_size", &layers::AttentionConfig::hidden_size)
        .def_readwrite("num_heads", &layers::AttentionConfig::num_heads)
        .def_readwrite("num_kv_heads", &layers::AttentionConfig::num_kv_heads)
        .def_readwrite("head_dim", &layers::AttentionConfig::head_dim)
        .def_readwrite("attention_type", &layers::AttentionConfig::attention_type)
        .def_readwrite("use_bias", &layers::AttentionConfig::use_bias)
        .def_readwrite("dropout_prob", &layers::AttentionConfig::dropout_prob)
        .def_readwrite("max_relative_position", &layers::AttentionConfig::max_relative_position)
        .def_readwrite("use_rotary", &layers::AttentionConfig::use_rotary)
        .def_readwrite("rotary_dim", &layers::AttentionConfig::rotary_dim);
    
    // ConvModuleConfig
    py::class_<layers::ConvModuleConfig>(layers, "ConvModuleConfig")
        .def(py::init<>())
        .def_readwrite("channels", &layers::ConvModuleConfig::channels)
        .def_readwrite("kernel_size", &layers::ConvModuleConfig::kernel_size)
        .def_readwrite("dropout_prob", &layers::ConvModuleConfig::dropout_prob)
        .def_readwrite("causal", &layers::ConvModuleConfig::causal)
        .def_readwrite("batch_norm_eps", &layers::ConvModuleConfig::batch_norm_eps);
    
    // FFNConfig
    py::class_<layers::FFNConfig>(layers, "FFNConfig")
        .def(py::init<>())
        .def_readwrite("d_model", &layers::FFNConfig::d_model)
        .def_readwrite("d_ff", &layers::FFNConfig::d_ff)
        .def_readwrite("activation", &layers::FFNConfig::activation)
        .def_readwrite("dropout_prob", &layers::FFNConfig::dropout_prob)
        .def_readwrite("use_bias", &layers::FFNConfig::use_bias)
        .def_readwrite("use_gated", &layers::FFNConfig::use_gated);
    
    // EncoderLayerConfig
    py::class_<layers::EncoderLayerConfig>(layers, "EncoderLayerConfig")
        .def(py::init<>())
        .def_readwrite("d_model", &layers::EncoderLayerConfig::d_model)
        .def_readwrite("num_heads", &layers::EncoderLayerConfig::num_heads)
        .def_readwrite("d_ff", &layers::EncoderLayerConfig::d_ff)
        .def_readwrite("conv_kernel_size", &layers::EncoderLayerConfig::conv_kernel_size)
        .def_readwrite("attention_type", &layers::EncoderLayerConfig::attention_type)
        .def_readwrite("ffn_activation", &layers::EncoderLayerConfig::ffn_activation)
        .def_readwrite("dropout_prob", &layers::EncoderLayerConfig::dropout_prob)
        .def_readwrite("attention_dropout_prob", &layers::EncoderLayerConfig::attention_dropout_prob)
        .def_readwrite("use_macaron_ffn", &layers::EncoderLayerConfig::use_macaron_ffn)
        .def_readwrite("causal", &layers::EncoderLayerConfig::causal);
    
    // ============== Base Layer Classes ==============
    
    // BaseLayer (abstract)
    py::class_<layers::BaseLayer, std::shared_ptr<layers::BaseLayer>>(layers, "BaseLayer")
        .def_property_readonly("name", &layers::BaseLayer::name)
        .def_property_readonly("dtype", &layers::BaseLayer::dtype)
        .def_property_readonly("device_id", &layers::BaseLayer::deviceId)
        .def_property_readonly("is_initialized", &layers::BaseLayer::isInitialized)
        .def("weight_names", &layers::BaseLayer::weightNames)
        .def("num_parameters", &layers::BaseLayer::numParameters)
        .def("weight_memory_bytes", &layers::BaseLayer::weightMemoryBytes)
        .def("convert_weights", &layers::BaseLayer::convertWeights, py::arg("dtype"));
    
    // StatefulLayer (abstract)
    py::class_<layers::StatefulLayer, layers::BaseLayer, std::shared_ptr<layers::StatefulLayer>>(
        layers, "StatefulLayer")
        .def("reset_state", &layers::StatefulLayer::resetState)
        .def("get_state", &layers::StatefulLayer::getState)
        .def("set_state", &layers::StatefulLayer::setState, py::arg("state"));
    
    // ============== Attention Layers ==============
    
    // MultiHeadAttention
    py::class_<layers::MultiHeadAttention, layers::StatefulLayer, 
               std::shared_ptr<layers::MultiHeadAttention>>(layers, "MultiHeadAttention")
        .def(py::init<const std::string&, const layers::AttentionConfig&, DataType, int>(),
             py::arg("name"),
             py::arg("config"),
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", py::overload_cast<const Tensor&, cudaStream_t>(
             &layers::MultiHeadAttention::forward),
             py::arg("input"), py::arg("stream") = nullptr)
        .def("forward_incremental", &layers::MultiHeadAttention::forwardIncremental,
             py::arg("input"), py::arg("stream") = nullptr)
        .def("init_kv_cache", &layers::MultiHeadAttention::initKVCache,
             py::arg("batch_size"), py::arg("max_seq_len"))
        .def_property_readonly("config", &layers::MultiHeadAttention::config);
    
    // RelativePositionAttention
    py::class_<layers::RelativePositionAttention, layers::MultiHeadAttention,
               std::shared_ptr<layers::RelativePositionAttention>>(
        layers, "RelativePositionAttention")
        .def(py::init<const std::string&, const layers::AttentionConfig&, DataType, int>(),
             py::arg("name"),
             py::arg("config"),
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("update_position_embeddings", &layers::RelativePositionAttention::updatePositionEmbeddings,
             py::arg("seq_len"), py::arg("stream") = nullptr);
    
    // ============== Convolution Layers ==============
    
    // ConformerConvModule
    py::class_<layers::ConformerConvModule, layers::StatefulLayer,
               std::shared_ptr<layers::ConformerConvModule>>(layers, "ConformerConvModule")
        .def(py::init<const std::string&, const layers::ConvModuleConfig&, DataType, int>(),
             py::arg("name"),
             py::arg("config"),
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", &layers::ConformerConvModule::forward,
             py::arg("input"), py::arg("stream") = nullptr)
        .def("forward_streaming", &layers::ConformerConvModule::forwardStreaming,
             py::arg("input"), py::arg("stream") = nullptr)
        .def_property_readonly("config", &layers::ConformerConvModule::config);
    
    // Conv1D
    py::class_<layers::Conv1D, layers::BaseLayer, std::shared_ptr<layers::Conv1D>>(
        layers, "Conv1D")
        .def(py::init<const std::string&, int, int, int, int, int, int, bool, DataType, int>(),
             py::arg("name"),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("stride") = 1,
             py::arg("padding") = 0,
             py::arg("groups") = 1,
             py::arg("use_bias") = true,
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", &layers::Conv1D::forward,
             py::arg("input"), py::arg("stream") = nullptr);
    
    // ============== FFN Layers ==============
    
    // FeedForward
    py::class_<layers::FeedForward, layers::BaseLayer, std::shared_ptr<layers::FeedForward>>(
        layers, "FeedForward")
        .def(py::init<const std::string&, const layers::FFNConfig&, DataType, int>(),
             py::arg("name"),
             py::arg("config"),
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", &layers::FeedForward::forward,
             py::arg("input"), py::arg("stream") = nullptr)
        .def_property_readonly("config", &layers::FeedForward::config);
    
    // ConformerFeedForward
    py::class_<layers::ConformerFeedForward, layers::BaseLayer,
               std::shared_ptr<layers::ConformerFeedForward>>(layers, "ConformerFeedForward")
        .def(py::init<const std::string&, const layers::FFNConfig&, DataType, int>(),
             py::arg("name"),
             py::arg("config"),
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", &layers::ConformerFeedForward::forward,
             py::arg("input"), py::arg("stream") = nullptr)
        .def("forward_no_residual", &layers::ConformerFeedForward::forwardNoResidual,
             py::arg("input"), py::arg("stream") = nullptr)
        .def_property_readonly("config", &layers::ConformerFeedForward::config);
    
    // Linear
    py::class_<layers::Linear, layers::BaseLayer, std::shared_ptr<layers::Linear>>(
        layers, "Linear")
        .def(py::init<const std::string&, int, int, bool, DataType, int>(),
             py::arg("name"),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("use_bias") = true,
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", &layers::Linear::forward,
             py::arg("input"), py::arg("stream") = nullptr)
        .def_property_readonly("in_features", &layers::Linear::inFeatures)
        .def_property_readonly("out_features", &layers::Linear::outFeatures);
    
    // ============== Encoder Layers ==============
    
    // ConformerEncoderLayer
    py::class_<layers::ConformerEncoderLayer, layers::StatefulLayer,
               std::shared_ptr<layers::ConformerEncoderLayer>>(layers, "ConformerEncoderLayer")
        .def(py::init<const std::string&, const layers::EncoderLayerConfig&, DataType, int>(),
             py::arg("name"),
             py::arg("config"),
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", py::overload_cast<const Tensor&, cudaStream_t>(
             &layers::ConformerEncoderLayer::forward),
             py::arg("input"), py::arg("stream") = nullptr)
        .def("forward_streaming", &layers::ConformerEncoderLayer::forwardStreaming,
             py::arg("input"), py::arg("stream") = nullptr)
        .def_property_readonly("config", &layers::ConformerEncoderLayer::config);
    
    // TransformerEncoderLayer
    py::class_<layers::TransformerEncoderLayer, layers::StatefulLayer,
               std::shared_ptr<layers::TransformerEncoderLayer>>(layers, "TransformerEncoderLayer")
        .def(py::init<const std::string&, const layers::EncoderLayerConfig&, DataType, int>(),
             py::arg("name"),
             py::arg("config"),
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", &layers::TransformerEncoderLayer::forward,
             py::arg("input"), py::arg("stream") = nullptr);
    
    // BranchformerEncoderLayer
    py::class_<layers::BranchformerEncoderLayer, layers::StatefulLayer,
               std::shared_ptr<layers::BranchformerEncoderLayer>>(layers, "BranchformerEncoderLayer")
        .def(py::init<const std::string&, const layers::EncoderLayerConfig&, DataType, int>(),
             py::arg("name"),
             py::arg("config"),
             py::arg("dtype") = DataType::FP16,
             py::arg("device_id") = 0)
        .def("forward", &layers::BranchformerEncoderLayer::forward,
             py::arg("input"), py::arg("stream") = nullptr);
}

} // namespace pybind
} // namespace oasr
