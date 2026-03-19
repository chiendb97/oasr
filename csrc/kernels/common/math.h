#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace oasr {
namespace kernels {

// =============================================================================
// Scalar Activation Functions
// =============================================================================

template <typename T>
__device__ __forceinline__ T sigmoid(T x) {
    return T(1.0f) / (T(1.0f) + expf(-float(x)));
}

template <typename T>
__device__ __forceinline__ T relu(T x) {
    return x > T(0) ? x : T(0);
}

template <typename T>
__device__ __forceinline__ T gelu(T x) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;
    float xf = float(x);
    float cdf = 0.5f * (1.0f + tanhf(kSqrt2OverPi * (xf + kCoeff * xf * xf * xf)));
    return T(xf * cdf);
}

template <typename T>
__device__ __forceinline__ T swish(T x) {
    return x * sigmoid(x);
}

// =============================================================================
// Activation Functors for Kernel Fusion
// =============================================================================

struct IdentityActivation {
    __device__ __forceinline__ float operator()(float x) const { return x; }
};

struct ReluActivation {
    __device__ __forceinline__ float operator()(float x) const { return relu(x); }
};

struct GeluActivation {
    __device__ __forceinline__ float operator()(float x) const { return gelu(x); }
};

struct SwishActivation {
    __device__ __forceinline__ float operator()(float x) const { return swish(x); }
};

}  // namespace kernels
}  // namespace oasr
