// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/attention/attention_kernels.h"
#include "kernels/attention/attention_params.h"
#include "kernels/conv/conv_kernels.h"
#include "kernels/norm/norm_kernels.h"

// GEMM kernels
#include "kernels/gemm/gemm_configs.h"
#include "kernels/gemm/gemm_kernels.h"
#include "kernels/gemm/bmm_kernels.h"
#include "kernels/gemm/group_gemm_kernels.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace py = pybind11;

namespace oasr {
namespace pybind {

/**
 * @brief Register only GEMM/BMM/GroupGEMM Python bindings on an existing kernels module.
 *        Call this from pybind_main after creating the kernels submodule.
 */
inline void registerGemmBindings(py::module_& kernels) {
  using namespace kernels::gemm;

  py::module_ gemm_mod =
      kernels.def_submodule("gemm", "GEMM/BMM/GroupGEMM kernels (CUTLASS)");

  // --- Enums ---
  py::enum_<TransposeOp>(gemm_mod, "TransposeOp")
      .value("NoTranspose", TransposeOp::NoTranspose)
      .value("Transpose", TransposeOp::Transpose)
      .value("ConjugateTranspose", TransposeOp::ConjugateTranspose);

  py::enum_<MatrixLayout>(gemm_mod, "MatrixLayout")
      .value("RowMajor", MatrixLayout::RowMajor)
      .value("ColumnMajor", MatrixLayout::ColumnMajor);

  py::enum_<EpilogueFusion>(gemm_mod, "EpilogueFusion")
      .value("NONE", EpilogueFusion::NONE)
      .value("BIAS", EpilogueFusion::BIAS)
      .value("BIAS_RELU", EpilogueFusion::BIAS_RELU)
      .value("BIAS_GELU", EpilogueFusion::BIAS_GELU)
      .value("BIAS_SWISH", EpilogueFusion::BIAS_SWISH)
      .value("GATED", EpilogueFusion::GATED);

  py::enum_<GemmStatus>(gemm_mod, "GemmStatus")
      .value("SUCCESS", GemmStatus::SUCCESS)
      .value("INVALID_ARGUMENT", GemmStatus::INVALID_ARGUMENT)
      .value("WORKSPACE_TOO_SMALL", GemmStatus::WORKSPACE_TOO_SMALL)
      .value("NOT_SUPPORTED", GemmStatus::NOT_SUPPORTED)
      .value("INTERNAL_ERROR", GemmStatus::INTERNAL_ERROR)
      .value("CUDA_ERROR", GemmStatus::CUDA_ERROR)
      .value("CUTLASS_ERROR", GemmStatus::CUTLASS_ERROR);

  py::enum_<SplitKStyle>(gemm_mod, "SplitKStyle")
      .value("NO_SPLIT_K", SplitKStyle::NO_SPLIT_K)
      .value("SPLIT_K_SERIAL", SplitKStyle::SPLIT_K_SERIAL)
      .value("STREAM_K", SplitKStyle::STREAM_K);

  py::enum_<MainloopScheduleType>(gemm_mod, "MainloopScheduleType")
      .value("AUTO", MainloopScheduleType::AUTO)
      .value("PINGPONG", MainloopScheduleType::PINGPONG)
      .value("COOPERATIVE", MainloopScheduleType::COOPERATIVE)
      .value("WARPSPECIALIZED", MainloopScheduleType::WARPSPECIALIZED);

  py::enum_<ClusterShape>(gemm_mod, "ClusterShape")
      .value("ClusterShape_1x1x1", ClusterShape::ClusterShape_1x1x1)
      .value("ClusterShape_2x1x1", ClusterShape::ClusterShape_2x1x1)
      .value("ClusterShape_1x2x1", ClusterShape::ClusterShape_1x2x1)
      .value("ClusterShape_2x2x1", ClusterShape::ClusterShape_2x2x1)
      .value("ClusterShape_1x4x1", ClusterShape::ClusterShape_1x4x1)
      .value("ClusterShape_4x1x1", ClusterShape::ClusterShape_4x1x1);

  py::class_<GemmConfig>(gemm_mod, "GemmConfig")
      .def(py::init<>())
      .def_readwrite("split_k_style", &GemmConfig::split_k_style)
      .def_readwrite("split_k_factor", &GemmConfig::split_k_factor)
      .def_readwrite("stages", &GemmConfig::stages)
      .def_readwrite("mainloop_schedule", &GemmConfig::mainloop_schedule)
      .def_readwrite("cluster_shape", &GemmConfig::cluster_shape)
      .def_readwrite("sm_version", &GemmConfig::sm_version)
      .def_readwrite("is_tma_warp_specialized", &GemmConfig::is_tma_warp_specialized)
      .def("to_string", &GemmConfig::toString)
      .def("__repr__", &GemmConfig::toString);

  py::class_<GemmProblemDesc>(gemm_mod, "GemmProblemDesc")
      .def(py::init<>())
      .def(py::init<int, int, int>(), py::arg("m"), py::arg("n"), py::arg("k"))
      .def_readwrite("M", &GemmProblemDesc::M)
      .def_readwrite("N", &GemmProblemDesc::N)
      .def_readwrite("K", &GemmProblemDesc::K);

  gemm_mod.def("invoke_gemm",
      [](intptr_t a, intptr_t b, intptr_t d, int M, int N, int K,
         int64_t lda, int64_t ldb, int64_t ldd, float alpha, float beta,
         TransposeOp trans_a, TransposeOp trans_b, DataType dtype, py::object stream) {
        cudaStream_t s = stream.is_none() ? nullptr
            : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
        return invokeGemm(
            reinterpret_cast<const void*>(a),
            reinterpret_cast<const void*>(b),
            reinterpret_cast<void*>(d),
            M, N, K, lda, ldb, ldd, alpha, beta,
            trans_a, trans_b, dtype, s);
      },
      py::arg("a"), py::arg("b"), py::arg("d"),
      py::arg("M"), py::arg("N"), py::arg("K"),
      py::arg("lda"), py::arg("ldb"), py::arg("ldd"),
      py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
      py::arg("trans_a") = TransposeOp::NoTranspose,
      py::arg("trans_b") = TransposeOp::NoTranspose,
      py::arg("dtype"), py::arg("stream") = py::none(),
      "Execute GEMM: D = alpha * A @ B + beta * D");

  gemm_mod.def("invoke_bmm",
      [](intptr_t a, intptr_t b, intptr_t d, int batch_size, int M, int N, int K,
         int64_t lda, int64_t ldb, int64_t ldd,
         int64_t stride_a, int64_t stride_b, int64_t stride_d,
         float alpha, float beta,
         TransposeOp trans_a, TransposeOp trans_b, DataType dtype, py::object stream) {
        cudaStream_t s = stream.is_none() ? nullptr
            : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
        return invokeBmm(
            reinterpret_cast<const void*>(a),
            reinterpret_cast<const void*>(b),
            reinterpret_cast<void*>(d),
            batch_size, M, N, K, lda, ldb, ldd,
            stride_a, stride_b, stride_d, alpha, beta,
            trans_a, trans_b, dtype, s);
      },
      py::arg("a"), py::arg("b"), py::arg("d"),
      py::arg("batch_size"), py::arg("M"), py::arg("N"), py::arg("K"),
      py::arg("lda"), py::arg("ldb"), py::arg("ldd"),
      py::arg("stride_a"), py::arg("stride_b"), py::arg("stride_d"),
      py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
      py::arg("trans_a") = TransposeOp::NoTranspose,
      py::arg("trans_b") = TransposeOp::NoTranspose,
      py::arg("dtype"), py::arg("stream") = py::none(),
      "Execute strided batched GEMM");

  gemm_mod.def("invoke_group_gemm",
      [](intptr_t problems, int num_problems,
         intptr_t a_array, intptr_t b_array, intptr_t d_array,
         intptr_t lda_array, intptr_t ldb_array, intptr_t ldd_array,
         DataType dtype, intptr_t workspace_float, size_t workspace_float_size,
         py::object stream) {
        cudaStream_t s = stream.is_none() ? nullptr
            : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
        return invokeGroupGemm(
            reinterpret_cast<const GemmProblemDesc*>(problems),
            num_problems,
            reinterpret_cast<const void* const*>(a_array),
            reinterpret_cast<const void* const*>(b_array),
            reinterpret_cast<void* const*>(d_array),
            reinterpret_cast<const int64_t*>(lda_array),
            reinterpret_cast<const int64_t*>(ldb_array),
            reinterpret_cast<const int64_t*>(ldd_array),
            dtype,
            reinterpret_cast<void*>(workspace_float),
            workspace_float_size, s);
      },
      py::arg("problems"), py::arg("num_problems"),
      py::arg("a_array"), py::arg("b_array"), py::arg("d_array"),
      py::arg("lda_array"), py::arg("ldb_array"), py::arg("ldd_array"),
      py::arg("dtype"), py::arg("workspace_float"), py::arg("workspace_float_size"),
      py::arg("stream") = py::none(),
      "Execute grouped GEMM (variable-sized problems)");

  gemm_mod.def("query_group_gemm_workspace_size",
      [](int num_problems, DataType dtype, const GemmConfig& config) {
        size_t float_ws = 0, int_ws = 0;
        queryGroupGemmWorkspaceSize(num_problems, nullptr, dtype,
                                    float_ws, int_ws, config);
        return py::make_tuple(float_ws, int_ws);
      },
      py::arg("num_problems"), py::arg("dtype"),
      py::arg("config") = GemmConfig(),
      "Query workspace sizes for grouped GEMM (returns float_ws, int_ws)");

  gemm_mod.def("get_sm_version", &getSMVersion, py::arg("device_id") = -1,
      "Get SM version of the current or specified device");
  gemm_mod.def("supports_tma", &supportsTMA, py::arg("sm_version"),
      "Check if SM version supports TMA");
  gemm_mod.def("supports_warp_specialization", &supportsWarpSpecialization,
      py::arg("sm_version"),
      "Check if SM version supports warp specialization");
  gemm_mod.def("get_gemm_status_string", &getGemmStatusString, py::arg("status"),
      "Convert GemmStatus to string");
  gemm_mod.def("query_gemm_workspace_size", &queryGemmWorkspaceSize,
      py::arg("M"), py::arg("N"), py::arg("K"), py::arg("dtype"),
      py::arg("config") = GemmConfig(),
      "Query workspace size for GEMM");
  gemm_mod.def("query_bmm_workspace_size", &queryBmmWorkspaceSize,
      py::arg("batch_size"), py::arg("M"), py::arg("N"), py::arg("K"),
      py::arg("dtype"), py::arg("config") = GemmConfig(),
      "Query workspace size for batched GEMM");
  gemm_mod.def("auto_tune_gemm",
      [](int M, int N, int K, DataType dtype, int num_warmup, int num_iter) {
        return autoTuneGemm(M, N, K, dtype, num_warmup, num_iter, nullptr);
      },
      py::arg("M"), py::arg("N"), py::arg("K"), py::arg("dtype"),
      py::arg("num_warmup") = 5, py::arg("num_iter") = 10,
      "Auto-tune and return best GEMM configuration");
  gemm_mod.def("auto_tune_bmm",
      [](int batch, int M, int N, int K, DataType dtype,
         int num_warmup, int num_iter) {
        return autoTuneBmm(batch, M, N, K, dtype, num_warmup, num_iter, nullptr);
      },
      py::arg("batch"), py::arg("M"), py::arg("N"), py::arg("K"),
      py::arg("dtype"), py::arg("num_warmup") = 5, py::arg("num_iter") = 10,
      "Auto-tune and return best batched GEMM configuration");
  gemm_mod.def("get_default_sm80_configs", &getDefaultSM80Configs,
      "Get default GEMM configurations for SM80");
  gemm_mod.def("get_default_sm90_configs", &getDefaultSM90Configs,
      "Get default GEMM configurations for SM90");
}

/**
 * @brief Register kernel-related Python bindings
 */
inline void registerKernelBindings(py::module_ &m) {
  // Create kernels submodule
  py::module_ kernels = m.def_submodule("kernels", "Low-level CUDA kernels");

  // ============== Attention Kernels ==============
  py::module_ attn = kernels.def_submodule("attention", "Attention kernels");

  // AttentionParams (stream as intptr_t to avoid cudaStream_t in pybind)
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
                     &kernels::AttentionParams::use_flash_attention)
      .def_property("stream",
          [](const kernels::AttentionParams& p) {
            return reinterpret_cast<intptr_t>(p.stream);
          },
          [](kernels::AttentionParams& p, intptr_t v) {
            p.stream = reinterpret_cast<cudaStream_t>(v);
          });

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
      kernels.def_submodule("conv", "Convolution kernels");

  conv.def("glu",
           [](const void* input, void* output, int batch_size, int seq_len,
              int channels, DataType dtype, intptr_t stream) {
             kernels::invokeGLU(input, output, batch_size, seq_len, channels,
                                dtype, reinterpret_cast<cudaStream_t>(stream));
           },
           py::arg("input"), py::arg("output"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
           py::arg("dtype"), py::arg("stream") = 0,
           "GLU activation kernel");

  conv.def("swish",
           [](const void* input, void* output, int batch_size, int seq_len,
              int channels, DataType dtype, intptr_t stream) {
             kernels::invokeSwish(input, output, batch_size, seq_len, channels,
                                 dtype, reinterpret_cast<cudaStream_t>(stream));
           },
           py::arg("input"), py::arg("output"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
           py::arg("dtype"), py::arg("stream") = 0,
           "Swish activation kernel");

  // ============== GEMM Kernels ==============
  py::module_ gemm_mod =
      kernels.def_submodule("gemm", "GEMM/BMM/GroupGEMM kernels (CUTLASS)");

  using namespace kernels::gemm;

  // --- Enums ---
  py::enum_<TransposeOp>(gemm_mod, "TransposeOp")
      .value("NoTranspose", TransposeOp::NoTranspose)
      .value("Transpose", TransposeOp::Transpose)
      .value("ConjugateTranspose", TransposeOp::ConjugateTranspose);

  py::enum_<MatrixLayout>(gemm_mod, "MatrixLayout")
      .value("RowMajor", MatrixLayout::RowMajor)
      .value("ColumnMajor", MatrixLayout::ColumnMajor);

  py::enum_<EpilogueFusion>(gemm_mod, "EpilogueFusion")
      .value("NONE", EpilogueFusion::NONE)
      .value("BIAS", EpilogueFusion::BIAS)
      .value("BIAS_RELU", EpilogueFusion::BIAS_RELU)
      .value("BIAS_GELU", EpilogueFusion::BIAS_GELU)
      .value("BIAS_SWISH", EpilogueFusion::BIAS_SWISH)
      .value("GATED", EpilogueFusion::GATED);

  py::enum_<GemmStatus>(gemm_mod, "GemmStatus")
      .value("SUCCESS", GemmStatus::SUCCESS)
      .value("INVALID_ARGUMENT", GemmStatus::INVALID_ARGUMENT)
      .value("WORKSPACE_TOO_SMALL", GemmStatus::WORKSPACE_TOO_SMALL)
      .value("NOT_SUPPORTED", GemmStatus::NOT_SUPPORTED)
      .value("INTERNAL_ERROR", GemmStatus::INTERNAL_ERROR)
      .value("CUDA_ERROR", GemmStatus::CUDA_ERROR)
      .value("CUTLASS_ERROR", GemmStatus::CUTLASS_ERROR);

  py::enum_<SplitKStyle>(gemm_mod, "SplitKStyle")
      .value("NO_SPLIT_K", SplitKStyle::NO_SPLIT_K)
      .value("SPLIT_K_SERIAL", SplitKStyle::SPLIT_K_SERIAL)
      .value("STREAM_K", SplitKStyle::STREAM_K);

  py::enum_<MainloopScheduleType>(gemm_mod, "MainloopScheduleType")
      .value("AUTO", MainloopScheduleType::AUTO)
      .value("PINGPONG", MainloopScheduleType::PINGPONG)
      .value("COOPERATIVE", MainloopScheduleType::COOPERATIVE)
      .value("WARPSPECIALIZED", MainloopScheduleType::WARPSPECIALIZED);

  py::enum_<ClusterShape>(gemm_mod, "ClusterShape")
      .value("ClusterShape_1x1x1", ClusterShape::ClusterShape_1x1x1)
      .value("ClusterShape_2x1x1", ClusterShape::ClusterShape_2x1x1)
      .value("ClusterShape_1x2x1", ClusterShape::ClusterShape_1x2x1)
      .value("ClusterShape_2x2x1", ClusterShape::ClusterShape_2x2x1)
      .value("ClusterShape_1x4x1", ClusterShape::ClusterShape_1x4x1)
      .value("ClusterShape_4x1x1", ClusterShape::ClusterShape_4x1x1);

  // --- GemmConfig ---
  py::class_<GemmConfig>(gemm_mod, "GemmConfig")
      .def(py::init<>())
      .def_readwrite("split_k_style", &GemmConfig::split_k_style)
      .def_readwrite("split_k_factor", &GemmConfig::split_k_factor)
      .def_readwrite("stages", &GemmConfig::stages)
      .def_readwrite("mainloop_schedule", &GemmConfig::mainloop_schedule)
      .def_readwrite("cluster_shape", &GemmConfig::cluster_shape)
      .def_readwrite("sm_version", &GemmConfig::sm_version)
      .def_readwrite("is_tma_warp_specialized", &GemmConfig::is_tma_warp_specialized)
      .def("to_string", &GemmConfig::toString)
      .def("__repr__", &GemmConfig::toString);

  // --- GemmProblemDesc ---
  py::class_<GemmProblemDesc>(gemm_mod, "GemmProblemDesc")
      .def(py::init<>())
      .def(py::init<int, int, int>(), py::arg("m"), py::arg("n"), py::arg("k"))
      .def_readwrite("M", &GemmProblemDesc::M)
      .def_readwrite("N", &GemmProblemDesc::N)
      .def_readwrite("K", &GemmProblemDesc::K);

  // --- Invoke kernels (direct parameters) ---
  gemm_mod.def("invoke_gemm",
      [](intptr_t a, intptr_t b, intptr_t d, int M, int N, int K,
         int64_t lda, int64_t ldb, int64_t ldd, float alpha, float beta,
         TransposeOp trans_a, TransposeOp trans_b, DataType dtype, py::object stream) {
        cudaStream_t s = stream.is_none() ? nullptr
            : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
        return invokeGemm(
            reinterpret_cast<const void*>(a),
            reinterpret_cast<const void*>(b),
            reinterpret_cast<void*>(d),
            M, N, K, lda, ldb, ldd, alpha, beta,
            trans_a, trans_b, dtype, s);
      },
      py::arg("a"), py::arg("b"), py::arg("d"),
      py::arg("M"), py::arg("N"), py::arg("K"),
      py::arg("lda"), py::arg("ldb"), py::arg("ldd"),
      py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
      py::arg("trans_a") = TransposeOp::NoTranspose,
      py::arg("trans_b") = TransposeOp::NoTranspose,
      py::arg("dtype"), py::arg("stream") = py::none(),
      "Execute GEMM: D = alpha * A @ B + beta * D");

  gemm_mod.def("invoke_bmm",
      [](intptr_t a, intptr_t b, intptr_t d, int batch_size, int M, int N, int K,
         int64_t lda, int64_t ldb, int64_t ldd,
         int64_t stride_a, int64_t stride_b, int64_t stride_d,
         float alpha, float beta,
         TransposeOp trans_a, TransposeOp trans_b, DataType dtype, py::object stream) {
        cudaStream_t s = stream.is_none() ? nullptr
            : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
        return invokeBmm(
            reinterpret_cast<const void*>(a),
            reinterpret_cast<const void*>(b),
            reinterpret_cast<void*>(d),
            batch_size, M, N, K, lda, ldb, ldd,
            stride_a, stride_b, stride_d, alpha, beta,
            trans_a, trans_b, dtype, s);
      },
      py::arg("a"), py::arg("b"), py::arg("d"),
      py::arg("batch_size"), py::arg("M"), py::arg("N"), py::arg("K"),
      py::arg("lda"), py::arg("ldb"), py::arg("ldd"),
      py::arg("stride_a"), py::arg("stride_b"), py::arg("stride_d"),
      py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
      py::arg("trans_a") = TransposeOp::NoTranspose,
      py::arg("trans_b") = TransposeOp::NoTranspose,
      py::arg("dtype"), py::arg("stream") = py::none(),
      "Execute strided batched GEMM");

  gemm_mod.def("invoke_group_gemm",
      [](intptr_t problems, int num_problems,
         intptr_t a_array, intptr_t b_array, intptr_t d_array,
         intptr_t lda_array, intptr_t ldb_array, intptr_t ldd_array,
         DataType dtype, intptr_t workspace_float, size_t workspace_float_size,
         py::object stream) {
        cudaStream_t s = stream.is_none() ? nullptr
            : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
        return invokeGroupGemm(
            reinterpret_cast<const GemmProblemDesc*>(problems),
            num_problems,
            reinterpret_cast<const void* const*>(a_array),
            reinterpret_cast<const void* const*>(b_array),
            reinterpret_cast<void* const*>(d_array),
            reinterpret_cast<const int64_t*>(lda_array),
            reinterpret_cast<const int64_t*>(ldb_array),
            reinterpret_cast<const int64_t*>(ldd_array),
            dtype,
            reinterpret_cast<void*>(workspace_float),
            workspace_float_size, s);
      },
      py::arg("problems"), py::arg("num_problems"),
      py::arg("a_array"), py::arg("b_array"), py::arg("d_array"),
      py::arg("lda_array"), py::arg("ldb_array"), py::arg("ldd_array"),
      py::arg("dtype"), py::arg("workspace_float"), py::arg("workspace_float_size"),
      py::arg("stream") = py::none(),
      "Execute grouped GEMM (variable-sized problems)");

  // --- Group GEMM workspace query ---
  gemm_mod.def("query_group_gemm_workspace_size",
      [](int num_problems, DataType dtype, const GemmConfig& config) {
        size_t float_ws = 0, int_ws = 0;
        queryGroupGemmWorkspaceSize(num_problems, nullptr, dtype,
                                    float_ws, int_ws, config);
        return py::make_tuple(float_ws, int_ws);
      },
      py::arg("num_problems"), py::arg("dtype"),
      py::arg("config") = GemmConfig(),
      "Query workspace sizes for grouped GEMM (returns float_ws, int_ws)");

  // --- Utility functions ---
  gemm_mod.def("get_sm_version", &getSMVersion, py::arg("device_id") = -1,
      "Get SM version of the current or specified device");

  gemm_mod.def("supports_tma", &supportsTMA, py::arg("sm_version"),
      "Check if SM version supports TMA");

  gemm_mod.def("supports_warp_specialization", &supportsWarpSpecialization,
      py::arg("sm_version"),
      "Check if SM version supports warp specialization");

  gemm_mod.def("get_gemm_status_string", &getGemmStatusString, py::arg("status"),
      "Convert GemmStatus to string");

  // --- Workspace queries ---
  gemm_mod.def("query_gemm_workspace_size", &queryGemmWorkspaceSize,
      py::arg("M"), py::arg("N"), py::arg("K"), py::arg("dtype"),
      py::arg("config") = GemmConfig(),
      "Query workspace size for GEMM");

  gemm_mod.def("query_bmm_workspace_size", &queryBmmWorkspaceSize,
      py::arg("batch_size"), py::arg("M"), py::arg("N"), py::arg("K"),
      py::arg("dtype"), py::arg("config") = GemmConfig(),
      "Query workspace size for batched GEMM");

  // --- Auto-tuning ---
  gemm_mod.def("auto_tune_gemm",
      [](int M, int N, int K, DataType dtype, int num_warmup, int num_iter) {
        return autoTuneGemm(M, N, K, dtype, num_warmup, num_iter, nullptr);
      },
      py::arg("M"), py::arg("N"), py::arg("K"), py::arg("dtype"),
      py::arg("num_warmup") = 5, py::arg("num_iter") = 10,
      "Auto-tune and return best GEMM configuration");

  gemm_mod.def("auto_tune_bmm",
      [](int batch, int M, int N, int K, DataType dtype,
         int num_warmup, int num_iter) {
        return autoTuneBmm(batch, M, N, K, dtype, num_warmup, num_iter, nullptr);
      },
      py::arg("batch"), py::arg("M"), py::arg("N"), py::arg("K"),
      py::arg("dtype"), py::arg("num_warmup") = 5, py::arg("num_iter") = 10,
      "Auto-tune and return best batched GEMM configuration");

  // --- Default configs ---
  gemm_mod.def("get_default_sm80_configs", &getDefaultSM80Configs,
      "Get default GEMM configurations for SM80");

  gemm_mod.def("get_default_sm90_configs", &getDefaultSM90Configs,
      "Get default GEMM configurations for SM90");

  // ============== Normalization Kernels ==============
  py::module_ norm =
      kernels.def_submodule("norm", "Normalization kernels");

  // Normalization kernel functions (stream as intptr_t to avoid cudaStream_t in pybind)
  norm.def("layer_norm",
           [](const void* input, void* output, const void* gamma, const void* beta,
              int batch_size, int seq_len, int hidden_size, float eps,
              DataType dtype, intptr_t stream) {
             kernels::invokeLayerNorm(input, output, gamma, beta,
                 batch_size, seq_len, hidden_size, eps, dtype,
                 reinterpret_cast<cudaStream_t>(stream));
           },
           py::arg("input"), py::arg("output"), py::arg("gamma"), py::arg("beta"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
           py::arg("eps"), py::arg("dtype"), py::arg("stream") = 0,
           "Layer normalization kernel");

  norm.def("rms_norm",
           [](const void* input, void* output, const void* gamma, const void* beta,
              int batch_size, int seq_len, int hidden_size, float eps,
              DataType dtype, intptr_t stream) {
             kernels::invokeRMSNorm(input, output, gamma, beta,
                 batch_size, seq_len, hidden_size, eps, dtype,
                 reinterpret_cast<cudaStream_t>(stream));
           },
           py::arg("input"), py::arg("output"), py::arg("gamma"), py::arg("beta"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
           py::arg("eps"), py::arg("dtype"), py::arg("stream") = 0,
           "RMS normalization kernel");

  norm.def("batch_norm_1d",
           [](const void* input, void* output, const void* gamma, const void* beta,
              const void* running_mean, const void* running_var,
              int batch_size, int seq_len, int channels, float eps,
              DataType dtype, intptr_t stream) {
             kernels::invokeBatchNorm1D(input, output, gamma, beta,
                 running_mean, running_var,
                 batch_size, seq_len, channels, eps, dtype,
                 reinterpret_cast<cudaStream_t>(stream));
           },
           py::arg("input"), py::arg("output"), py::arg("gamma"), py::arg("beta"),
           py::arg("running_mean"), py::arg("running_var"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
           py::arg("eps"), py::arg("dtype"), py::arg("stream") = 0,
           "1D batch normalization kernel");

  norm.def("add_layer_norm",
           [](const void* input, const void* residual, void* output,
              const void* gamma, const void* beta,
              int batch_size, int seq_len, int hidden_size, float eps,
              DataType dtype, intptr_t stream) {
             kernels::invokeAddLayerNorm(input, residual, output, gamma, beta,
                 batch_size, seq_len, hidden_size, eps, dtype,
                 reinterpret_cast<cudaStream_t>(stream));
           },
           py::arg("input"), py::arg("residual"), py::arg("output"),
           py::arg("gamma"), py::arg("beta"),
           py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
           py::arg("eps"), py::arg("dtype"), py::arg("stream") = 0,
           "Fused add + layer norm kernel");
}

} // namespace pybind
} // namespace oasr
