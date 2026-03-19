// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <torch/extension.h>

#include "kernels/common/types.h"

namespace oasr {
namespace kernels {

// GLU (Gated Linear Unit) activation
// output = input[:, :half] * sigmoid(input[:, half:])
// Dimensions derived: batch_size=input.size(0), seq_len=input.size(1), channels=input.size(2)/2
//
// @param input Input [batch, seq_len, channels]
// @param stream CUDA stream
// @return Output [batch, seq_len, channels]
torch::Tensor invokeGLU(const torch::Tensor& input, cudaStream_t stream);

// Swish activation: x * sigmoid(x)
// Dimensions derived: batch_size=input.size(0), seq_len=input.size(1), channels=input.size(2)
// @param input Input [batch, seq_len, channels]
// @param stream CUDA stream
// @return Output [batch, seq_len, channels]
torch::Tensor invokeSwish(const torch::Tensor& input, cudaStream_t stream);

}  // namespace kernels
}  // namespace oasr
