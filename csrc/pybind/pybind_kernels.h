// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/attention/attention_kernels.h"
#include "kernels/attention/attention_params.h"
#include "kernels/convolution/conv_kernels.h"
#include "kernels/convolution/conv_params.h"
#include "kernels/feedforward/ffn_kernels.h"
#include "kernels/feedforward/ffn_params.h"
#include "kernels/normalization/norm_kernels.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

/**
 * @brief Register kernel-related Python bindings
 */
inline void registerKernelBindings(py::module_ &m) {
  // Create kernels submodule
  py::module_ kernels = m.def_submodule("kernels", "Low-level CUDA kernels");

  // ============== Attention Kernels ==============
  py::module_ attn = kernels.def_submodule("attention", "Attention kernels");

  // AttentionParams
  py::class_<kernels::AttentionParams>(attn, "AttentionParams")
      .def(py::init<>())
      .def_readwrite("batch_size", &kernels::AttentionParams::batch_size)
      .def_readwrite("seq_len", &kernels::AttentionParams::seq_len)
      .def_readwrite("num_heads", &kernels::AttentionParams::num_heads)
      .def_readwrite("head_dim", &kernels::AttentionParams::head_dim)
      .def_readwrite("num_kv_heads", &kernels::AttentionParams::num_kv_heads)
      .def_readwrite("scale", &kernels::AttentionParams::scale)
      .def_readwrite("dropout_prob", &kernels::AttentionParams::dropout_prob)
      .def_readwrite("attention_type",
                     &kernels::AttentionParams::attention_type)
      .def_readwrite("dtype", &kernels::AttentionParams::dtype)
      .def_readwrite("is_causal", &kernels::AttentionParams::is_causal)
      .def_readwrite("use_flash_attention",
                     &kernels::AttentionParams::use_flash_attention);

  // RelativePositionAttentionParams
  py::class_<kernels::RelativePositionAttentionParams,
             kernels::AttentionParams>(attn, "RelativePositionAttentionParams")
      .def(py::init<>())
      .def_readwrite(
          "max_relative_position",
          &kernels::RelativePositionAttentionParams::max_relative_position)
      .def_readwrite(
          "clamp_positions",
          &kernels::RelativePositionAttentionParams::clamp_positions);

  // KVCache
  py::class_<kernels::KVCache>(attn, "KVCache")
      .def(py::init<>())
      .def_readwrite("max_seq_len", &kernels::KVCache::max_seq_len)
      .def_readwrite("current_len", &kernels::KVCache::current_len)
      .def_readwrite("dtype", &kernels::KVCache::dtype);

  // Attention kernel functions
  attn.def("attention", &kernels::invokeAttention, py::arg("params"),
           "Multi-head attention kernel");

  attn.def("relative_position_attention",
           &kernels::invokeRelativePositionAttention, py::arg("params"),
           "Relative position attention kernel");

  // ============== Convolution Kernels ==============
  py::module_ conv =
      kernels.def_submodule("convolution", "Convolution kernels");

  // Conv1DParams
  py::class_<kernels::Conv1DParams>(conv, "Conv1DParams")
      .def(py::init<>())
      .def_readwrite("batch_size", &kernels::Conv1DParams::batch_size)
      .def_readwrite("seq_len", &kernels::Conv1DParams::seq_len)
      .def_readwrite("in_channels", &kernels::Conv1DParams::in_channels)
      .def_readwrite("out_channels", &kernels::Conv1DParams::out_channels)
      .def_readwrite("kernel_size", &kernels::Conv1DParams::kernel_size)
      .def_readwrite("stride", &kernels::Conv1DParams::stride)
      .def_readwrite("padding", &kernels::Conv1DParams::padding)
      .def_readwrite("dilation", &kernels::Conv1DParams::dilation)
      .def_readwrite("groups", &kernels::Conv1DParams::groups)
      .def_readwrite("conv_type", &kernels::Conv1DParams::conv_type)
      .def_readwrite("dtype", &kernels::Conv1DParams::dtype)
      .def_readwrite("channels_last", &kernels::Conv1DParams::channels_last)
      .def_readwrite("is_causal", &kernels::Conv1DParams::is_causal);

  // ConformerConvParams
  py::class_<kernels::ConformerConvParams>(conv, "ConformerConvParams")
      .def(py::init<>())
      .def_readwrite("batch_size", &kernels::ConformerConvParams::batch_size)
      .def_readwrite("seq_len", &kernels::ConformerConvParams::seq_len)
      .def_readwrite("d_model", &kernels::ConformerConvParams::d_model)
      .def_readwrite("kernel_size", &kernels::ConformerConvParams::kernel_size)
      .def_readwrite("dtype", &kernels::ConformerConvParams::dtype)
      .def_readwrite("batch_norm_eps",
                     &kernels::ConformerConvParams::batch_norm_eps)
      .def_readwrite("is_causal", &kernels::ConformerConvParams::is_causal);

  // ConvState
  py::class_<kernels::ConvState>(conv, "ConvState")
      .def(py::init<>())
      .def_readwrite("buffer_size", &kernels::ConvState::buffer_size)
      .def_readwrite("channels", &kernels::ConvState::channels)
      .def_readwrite("dtype", &kernels::ConvState::dtype);

  // Convolution kernel functions
  conv.def("conv1d", &kernels::invokeConv1D, py::arg("params"),
           "1D convolution kernel");

  conv.def("conformer_conv_module", &kernels::invokeConformerConvModule,
           py::arg("params"), "Conformer convolution module");

  conv.def("glu", &kernels::invokeGLU, py::arg("input"), py::arg("output"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
           py::arg("dtype"), py::arg("stream"), "GLU activation kernel");

  conv.def("swish", &kernels::invokeSwish, py::arg("input"), py::arg("output"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
           py::arg("dtype"), py::arg("stream"), "Swish activation kernel");

  // ============== FFN Kernels ==============
  py::module_ ffn =
      kernels.def_submodule("feedforward", "Feed-forward network kernels");

  // FFNParams
  py::class_<kernels::FFNParams>(ffn, "FFNParams")
      .def(py::init<>())
      .def_readwrite("batch_size", &kernels::FFNParams::batch_size)
      .def_readwrite("seq_len", &kernels::FFNParams::seq_len)
      .def_readwrite("d_model", &kernels::FFNParams::d_model)
      .def_readwrite("d_ff", &kernels::FFNParams::d_ff)
      .def_readwrite("activation", &kernels::FFNParams::activation)
      .def_readwrite("dtype", &kernels::FFNParams::dtype)
      .def_readwrite("dropout_prob", &kernels::FFNParams::dropout_prob)
      .def_readwrite("use_gated", &kernels::FFNParams::use_gated);

  // ConformerFFNParams
  py::class_<kernels::ConformerFFNParams, kernels::FFNParams>(
      ffn, "ConformerFFNParams")
      .def(py::init<>())
      .def_readwrite("residual_scale",
                     &kernels::ConformerFFNParams::residual_scale)
      .def_readwrite("fuse_residual",
                     &kernels::ConformerFFNParams::fuse_residual)
      .def_readwrite("fuse_layernorm",
                     &kernels::ConformerFFNParams::fuse_layernorm);

  // FFN kernel functions
  ffn.def("ffn", &kernels::invokeFFN, py::arg("params"),
          "Feed-forward network kernel");

  ffn.def("gated_ffn", &kernels::invokeGatedFFN, py::arg("params"),
          "Gated feed-forward network kernel");

  ffn.def("conformer_ffn", &kernels::invokeConformerFFN, py::arg("params"),
          "Conformer feed-forward module");

  // ============== Normalization Kernels ==============
  py::module_ norm =
      kernels.def_submodule("normalization", "Normalization kernels");

  // NormParams
  py::class_<kernels::NormParams>(norm, "NormParams")
      .def(py::init<>())
      .def_readwrite("batch_size", &kernels::NormParams::batch_size)
      .def_readwrite("seq_len", &kernels::NormParams::seq_len)
      .def_readwrite("hidden_size", &kernels::NormParams::hidden_size)
      .def_readwrite("eps", &kernels::NormParams::eps)
      .def_readwrite("norm_type", &kernels::NormParams::norm_type)
      .def_readwrite("dtype", &kernels::NormParams::dtype);

  // Normalization kernel functions
  norm.def("layer_norm", &kernels::invokeLayerNorm, py::arg("input"),
           py::arg("output"), py::arg("gamma"), py::arg("beta"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
           py::arg("eps"), py::arg("dtype"), py::arg("stream"),
           "Layer normalization kernel");

  norm.def("rms_norm", &kernels::invokeRMSNorm, py::arg("input"),
           py::arg("output"), py::arg("gamma"), py::arg("batch_size"),
           py::arg("seq_len"), py::arg("hidden_size"), py::arg("eps"),
           py::arg("dtype"), py::arg("stream"), "RMS normalization kernel");

  norm.def("batch_norm_1d", &kernels::invokeBatchNorm1D, py::arg("input"),
           py::arg("output"), py::arg("gamma"), py::arg("beta"),
           py::arg("running_mean"), py::arg("running_var"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
           py::arg("eps"), py::arg("dtype"), py::arg("stream"),
           "1D batch normalization kernel");

  norm.def("add_layer_norm", &kernels::invokeAddLayerNorm, py::arg("input"),
           py::arg("residual"), py::arg("output"), py::arg("gamma"),
           py::arg("beta"), py::arg("batch_size"), py::arg("seq_len"),
           py::arg("hidden_size"), py::arg("eps"), py::arg("dtype"),
           py::arg("stream"), "Fused add + layer norm kernel");
}

} // namespace pybind
} // namespace oasr
