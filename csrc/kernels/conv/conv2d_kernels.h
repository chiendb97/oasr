// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <torch/extension.h>

#include "kernels/common/types.h"

namespace oasr {
namespace kernels {

/**
 * @brief 2D convolution using CUTLASS Ampere Tensor Core Implicit GEMM (SM80)
 *
 * Tensors must be in NHWC layout:
 *   input  [N, H, W, IC]
 *   filter [K, R, S, IC]
 *   output [N, P, Q, K]  where P = (H + 2*pad_h - dilation_h*(R-1) - 1) / stride_h + 1
 *                              Q = (W + 2*pad_w - dilation_w*(S-1) - 1) / stride_w + 1
 *
 * Alignment requirement: IC % 8 == 0 and K % 8 == 0 (128-bit vector loads).
 * Supports FP16 and BF16 dtypes.
 *
 * @param input      Input tensor [N, H, W, IC]
 * @param filter     Filter tensor [K, R, S, IC]
 * @param bias       Optional per-channel bias [K]; pass undefined tensor to skip
 * @param pad_h      Symmetric padding along the H dimension
 * @param pad_w      Symmetric padding along the W dimension
 * @param stride_h   Convolution stride along H
 * @param stride_w   Convolution stride along W
 * @param dilation_h Dilation along H
 * @param dilation_w Dilation along W
 * @param stream     CUDA stream
 * @return Output tensor [N, P, Q, K]
 */
torch::Tensor invokeConv2d(const torch::Tensor& input, const torch::Tensor& filter,
                           const torch::Tensor& bias, int pad_h, int pad_w, int stride_h,
                           int stride_w, int dilation_h, int dilation_w,
                           cudaStream_t stream = nullptr);

/**
 * @brief 2D convolution with fused activation using CUTLASS Ampere Tensor Core Implicit GEMM
 *
 * Computes output = activation(conv2d(input, filter)) + bias (if provided).
 * Same layout and alignment requirements as invokeConv2d.
 *
 * @param input      Input tensor [N, H, W, IC]
 * @param filter     Filter tensor [K, R, S, IC]
 * @param bias       Optional per-channel bias [K]; pass undefined tensor to skip
 * @param activation Fused activation: RELU, GELU, or SWISH
 * @param pad_h      Symmetric padding along H
 * @param pad_w      Symmetric padding along W
 * @param stride_h   Convolution stride along H
 * @param stride_w   Convolution stride along W
 * @param dilation_h Dilation along H
 * @param dilation_w Dilation along W
 * @param stream     CUDA stream
 * @return Output tensor [N, P, Q, K]
 */
torch::Tensor invokeConv2dActivation(const torch::Tensor& input, const torch::Tensor& filter,
                                     const torch::Tensor& bias, ActivationType activation,
                                     int pad_h, int pad_w, int stride_h, int stride_w,
                                     int dilation_h, int dilation_w, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace oasr
