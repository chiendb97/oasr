#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace oasr {

template <typename T>
__device__ __forceinline__ void swap(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}

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
__device__ __forceinline__ T gelu_erf(T x) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    float xf = float(x);
    return T(0.5f * xf * (1.0f + erff(xf * kInvSqrt2)));
}

template <typename T>
__device__ __forceinline__ T swish(T x) {
    return x * sigmoid(x);
}

// Numerically stable softplus: log(1 + exp(x)) = max(x, 0) + log1p(exp(-|x|)).
template <typename T>
__device__ __forceinline__ T softplus(T x) {
    float xf = float(x);
    return T(fmaxf(xf, 0.0f) + log1pf(expf(-fabsf(xf))));
}

// Zipformer Swoosh-L: log(1 + exp(x - 4)) - 0.08 x - 0.035.
template <typename T>
__device__ __forceinline__ T swoosh_l(T x) {
    float xf = float(x);
    return T(float(softplus(xf - 4.0f)) - 0.08f * xf - 0.035f);
}

// Zipformer Swoosh-R: log(1 + exp(x - 1)) - 0.08 x - 0.313261687.
template <typename T>
__device__ __forceinline__ T swoosh_r(T x) {
    float xf = float(x);
    return T(float(softplus(xf - 1.0f)) - 0.08f * xf - 0.313261687f);
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

struct GeluErfActivation {
    __device__ __forceinline__ float operator()(float x) const { return gelu_erf(x); }
};

struct SwishActivation {
    __device__ __forceinline__ float operator()(float x) const { return swish(x); }
};

}  // namespace oasr
