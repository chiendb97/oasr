// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace oasr {

// =============================================================================
// Vector Type Traits
// =============================================================================

// Primary template - defaults to scalar
template <typename T, int VecSize>
struct Vec {
    T data[VecSize];

    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }
};

// =============================================================================
// Float Vector Specializations
// =============================================================================

template <>
struct Vec<float, 1> {
    float data;

    __device__ __forceinline__ float& operator[](int i) { return data; }
    __device__ __forceinline__ const float& operator[](int i) const { return data; }

    __device__ __forceinline__ void load(const float* ptr) { data = *ptr; }

    __device__ __forceinline__ void store(float* ptr) const { *ptr = data; }
};

template <>
struct Vec<float, 2> {
    float2 data;

    __device__ __forceinline__ float& operator[](int i) {
        return reinterpret_cast<float*>(&data)[i];
    }
    __device__ __forceinline__ const float& operator[](int i) const {
        return reinterpret_cast<const float*>(&data)[i];
    }

    __device__ __forceinline__ void load(const float* ptr) {
        data = *reinterpret_cast<const float2*>(ptr);
    }

    __device__ __forceinline__ void store(float* ptr) const {
        *reinterpret_cast<float2*>(ptr) = data;
    }
};

template <>
struct Vec<float, 4> {
    float4 data;

    __device__ __forceinline__ float& operator[](int i) {
        return reinterpret_cast<float*>(&data)[i];
    }
    __device__ __forceinline__ const float& operator[](int i) const {
        return reinterpret_cast<const float*>(&data)[i];
    }

    __device__ __forceinline__ void load(const float* ptr) {
        data = *reinterpret_cast<const float4*>(ptr);
    }

    __device__ __forceinline__ void store(float* ptr) const {
        *reinterpret_cast<float4*>(ptr) = data;
    }
};

// =============================================================================
// Half (FP16) Vector Specializations
// =============================================================================

template <>
struct Vec<half, 1> {
    half data;

    __device__ __forceinline__ half& operator[](int i) { return data; }
    __device__ __forceinline__ const half& operator[](int i) const { return data; }

    __device__ __forceinline__ void load(const half* ptr) { data = *ptr; }

    __device__ __forceinline__ void store(half* ptr) const { *ptr = data; }
};

template <>
struct Vec<half, 2> {
    half2 data;

    __device__ __forceinline__ half& operator[](int i) { return reinterpret_cast<half*>(&data)[i]; }
    __device__ __forceinline__ const half& operator[](int i) const {
        return reinterpret_cast<const half*>(&data)[i];
    }

    __device__ __forceinline__ void load(const half* ptr) {
        data = *reinterpret_cast<const half2*>(ptr);
    }

    __device__ __forceinline__ void store(half* ptr) const {
        *reinterpret_cast<half2*>(ptr) = data;
    }
};

template <>
struct Vec<half, 4> {
    half2 data[2];

    __device__ __forceinline__ half& operator[](int i) { return reinterpret_cast<half*>(data)[i]; }
    __device__ __forceinline__ const half& operator[](int i) const {
        return reinterpret_cast<const half*>(data)[i];
    }

    __device__ __forceinline__ void load(const half* ptr) {
        float2 tmp = *reinterpret_cast<const float2*>(ptr);
        data[0] = reinterpret_cast<half2*>(&tmp)[0];
        data[1] = reinterpret_cast<half2*>(&tmp)[1];
    }

    __device__ __forceinline__ void store(half* ptr) const {
        *reinterpret_cast<float2*>(ptr) = *reinterpret_cast<const float2*>(data);
    }
};

template <>
struct Vec<half, 8> {
    half2 data[4];

    __device__ __forceinline__ half& operator[](int i) { return reinterpret_cast<half*>(data)[i]; }
    __device__ __forceinline__ const half& operator[](int i) const {
        return reinterpret_cast<const half*>(data)[i];
    }

    __device__ __forceinline__ void load(const half* ptr) {
        float4 tmp = *reinterpret_cast<const float4*>(ptr);
        data[0] = reinterpret_cast<half2*>(&tmp)[0];
        data[1] = reinterpret_cast<half2*>(&tmp)[1];
        data[2] = reinterpret_cast<half2*>(&tmp)[2];
        data[3] = reinterpret_cast<half2*>(&tmp)[3];
    }

    __device__ __forceinline__ void store(half* ptr) const {
        *reinterpret_cast<float4*>(ptr) = *reinterpret_cast<const float4*>(data);
    }
};

// =============================================================================
// BFloat16 Vector Specializations
// =============================================================================

template <>
struct Vec<__nv_bfloat16, 1> {
    __nv_bfloat16 data;

    __device__ __forceinline__ __nv_bfloat16& operator[](int i) { return data; }
    __device__ __forceinline__ const __nv_bfloat16& operator[](int i) const { return data; }

    __device__ __forceinline__ void load(const __nv_bfloat16* ptr) { data = *ptr; }

    __device__ __forceinline__ void store(__nv_bfloat16* ptr) const { *ptr = data; }
};

template <>
struct Vec<__nv_bfloat16, 2> {
    __nv_bfloat162 data;

    __device__ __forceinline__ __nv_bfloat16& operator[](int i) {
        return reinterpret_cast<__nv_bfloat16*>(&data)[i];
    }
    __device__ __forceinline__ const __nv_bfloat16& operator[](int i) const {
        return reinterpret_cast<const __nv_bfloat16*>(&data)[i];
    }

    __device__ __forceinline__ void load(const __nv_bfloat16* ptr) {
        data = *reinterpret_cast<const __nv_bfloat162*>(ptr);
    }

    __device__ __forceinline__ void store(__nv_bfloat16* ptr) const {
        *reinterpret_cast<__nv_bfloat162*>(ptr) = data;
    }
};

template <>
struct Vec<__nv_bfloat16, 4> {
    __nv_bfloat162 data[2];

    __device__ __forceinline__ __nv_bfloat16& operator[](int i) {
        return reinterpret_cast<__nv_bfloat16*>(data)[i];
    }
    __device__ __forceinline__ const __nv_bfloat16& operator[](int i) const {
        return reinterpret_cast<const __nv_bfloat16*>(data)[i];
    }

    __device__ __forceinline__ void load(const __nv_bfloat16* ptr) {
        float2 tmp = *reinterpret_cast<const float2*>(ptr);
        data[0] = reinterpret_cast<__nv_bfloat162*>(&tmp)[0];
        data[1] = reinterpret_cast<__nv_bfloat162*>(&tmp)[1];
    }

    __device__ __forceinline__ void store(__nv_bfloat16* ptr) const {
        *reinterpret_cast<float2*>(ptr) = *reinterpret_cast<const float2*>(data);
    }
};

template <>
struct Vec<__nv_bfloat16, 8> {
    __nv_bfloat162 data[4];

    __device__ __forceinline__ __nv_bfloat16& operator[](int i) {
        return reinterpret_cast<__nv_bfloat16*>(data)[i];
    }
    __device__ __forceinline__ const __nv_bfloat16& operator[](int i) const {
        return reinterpret_cast<const __nv_bfloat16*>(data)[i];
    }

    __device__ __forceinline__ void load(const __nv_bfloat16* ptr) {
        float4 tmp = *reinterpret_cast<const float4*>(ptr);
        data[0] = reinterpret_cast<__nv_bfloat162*>(&tmp)[0];
        data[1] = reinterpret_cast<__nv_bfloat162*>(&tmp)[1];
        data[2] = reinterpret_cast<__nv_bfloat162*>(&tmp)[2];
        data[3] = reinterpret_cast<__nv_bfloat162*>(&tmp)[3];
    }

    __device__ __forceinline__ void store(__nv_bfloat16* ptr) const {
        *reinterpret_cast<float4*>(ptr) = *reinterpret_cast<const float4*>(data);
    }
};

// =============================================================================
// Vector Size Traits - Determines optimal vector size for each type
// =============================================================================

template <typename T>
struct VecTypeTrait {
    static constexpr int VecSize = 1;
};

template <>
struct VecTypeTrait<float> {
    static constexpr int VecSize = 4;  // 128 bits = 4 floats
};

template <>
struct VecTypeTrait<half> {
    static constexpr int VecSize = 8;  // 128 bits = 8 halfs
};

template <>
struct VecTypeTrait<__nv_bfloat16> {
    static constexpr int VecSize = 8;  // 128 bits = 8 bf16s
};

// =============================================================================
// Helper Functions for Vectorized Load/Store
// =============================================================================

// Load a vector from memory
template <typename T, int VecSize>
__device__ __forceinline__ Vec<T, VecSize> loadVec(const T* ptr) {
    Vec<T, VecSize> v;
    v.load(ptr);
    return v;
}

// Store a vector to memory
template <typename T, int VecSize>
__device__ __forceinline__ void storeVec(T* ptr, const Vec<T, VecSize>& v) {
    v.store(ptr);
}

// Convert vector elements to float for accumulation
template <typename T, int VecSize>
__device__ __forceinline__ void vecToFloat(const Vec<T, VecSize>& v, float* out) {
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
        out[i] = static_cast<float>(v[i]);
    }
}

// Convert float array back to vector type
template <typename T, int VecSize>
__device__ __forceinline__ void floatToVec(const float* in, Vec<T, VecSize>& v) {
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
        v[i] = static_cast<T>(in[i]);
    }
}

// Sum all elements in a vector
template <typename T, int VecSize>
__device__ __forceinline__ float vecSum(const Vec<T, VecSize>& v) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
        sum += static_cast<float>(v[i]);
    }
    return sum;
}

// Sum of squares of all elements in a vector
template <typename T, int VecSize>
__device__ __forceinline__ float vecSumSquares(const Vec<T, VecSize>& v) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
        float val = static_cast<float>(v[i]);
        sum += val * val;
    }
    return sum;
}

// Sum of squared differences from mean
template <typename T, int VecSize>
__device__ __forceinline__ float vecSumSquaredDiff(const Vec<T, VecSize>& v, float mean) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
        float diff = static_cast<float>(v[i]) - mean;
        sum += diff * diff;
    }
    return sum;
}

// =============================================================================
// Type Alias for Common Use Cases
// =============================================================================

template <typename T>
using Vec1 = Vec<T, 1>;

template <typename T>
using Vec2 = Vec<T, 2>;

template <typename T>
using Vec4 = Vec<T, 4>;

template <typename T>
using Vec8 = Vec<T, 8>;

// Optimal vector type for a given scalar type
template <typename T>
using OptVec = Vec<T, VecTypeTrait<T>::VecSize>;

}  // namespace oasr
