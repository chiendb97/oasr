// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace oasr {

// Data types supported by OASR
enum class DataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    INT32 = 4,
};

// Memory types
enum class MemoryType {
    CPU = 0,
    GPU = 1,
    PINNED = 2,   // Pinned host memory
    UNIFIED = 3,  // Unified memory
};

// Attention types for ASR models
enum class AttentionType {
    MULTI_HEAD = 0,
    RELATIVE_POSITION = 1,  // For Conformer
    RELATIVE_SHIFT = 2,     // Shaw et al.
    ROTARY = 3,             // RoPE
    ALiBi = 4,
};

// Convolution types
enum class ConvType {
    DEPTHWISE = 0,  // Depthwise separable
    POINTWISE = 1,
    CAUSAL = 2,  // Causal convolution for streaming
    STANDARD = 3,
};

// Activation types
enum class ActivationType {
    RELU = 0,
    GELU = 1,
    SWISH = 2,  // SiLU
    IDENTITY = 3,
};

// Normalization types
enum class NormType {
    LAYER_NORM = 0,
    RMS_NORM = 1,
    BATCH_NORM = 2,
    GROUP_NORM = 3,
};

// Model types
enum class ModelType {
    CONFORMER = 0,
    PARAFORMER = 1,
    BRANCHFORMER = 2,
    TRANSFORMER = 3,
    ZIPFORMER = 4,
};

// Helper to get size of data type
inline size_t getDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return 4;
        case DataType::FP16:
            return 2;
        case DataType::BF16:
            return 2;
        case DataType::INT8:
            return 1;
        case DataType::INT32:
            return 4;
        default:
            return 0;
    }
}

// Template type traits for CUDA types
template <DataType DT>
struct DataTypeTraits;

template <>
struct DataTypeTraits<DataType::FP32> {
    using Type = float;
    static constexpr const char* name = "float32";
};

template <>
struct DataTypeTraits<DataType::FP16> {
    using Type = half;
    static constexpr const char* name = "float16";
};

template <>
struct DataTypeTraits<DataType::BF16> {
    using Type = __nv_bfloat16;
    static constexpr const char* name = "bfloat16";
};

template <>
struct DataTypeTraits<DataType::INT8> {
    using Type = int8_t;
    static constexpr const char* name = "int8";
};

template <>
struct DataTypeTraits<DataType::INT32> {
    using Type = int32_t;
    static constexpr const char* name = "int32";
};

}  // namespace oasr
