#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace oasr {
namespace kernels {

// Sigmoid function
template <typename T>
__device__ __forceinline__ T sigmoid(T x) {
    return T(1.0f) / (T(1.0f) + expf(-float(x)));
}

// Swish activation: x * sigmoid(x)
template <typename T>
__device__ __forceinline__ T swish(T x) {
    return x * sigmoid(x);
}

}  // namespace kernels
}  // namespace oasr
