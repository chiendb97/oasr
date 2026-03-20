// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/extension.h>

#include "conv2d_impl.h"
#include "kernels/common/arch_dispatch.h"
#include "kernels/common/arch_traits.h"
#include "kernels/common/epilogue_functors.h"
#include "kernels/conv/conv2d_kernels.h"
#include "kernels/gemm/gemm_utils.h"

namespace oasr {
namespace kernels {

// =============================================================================
// Arch-dispatched helpers (template context enables if constexpr)
// =============================================================================

template <int SmVersion, template <int, typename, typename> class EpilogueFn>
static gemm::GemmStatus dispatchConv2dDtype(const torch::Tensor& input, const void* input_ptr,
                                            const void* filter_ptr, const void* bias_ptr,
                                            void* output_ptr, int N, int H, int W, int IC, int K,
                                            int R, int S, int pad_h, int pad_w, int stride_h,
                                            int stride_w, int dilation_h, int dilation_w,
                                            cudaStream_t stream) {
    if constexpr (SmVersion < 80) {
        throw std::runtime_error("CUTLASS Conv2D requires SM80 or newer (current: SM" +
                                 std::to_string(SmVersion) + ")");
    } else {
        using Traits = ArchTraits<SmVersion>;

        if (input.scalar_type() == torch::kHalf) {
            return CutlassConv2dFprop<
                Traits, cutlass::half_t, cutlass::half_t, cutlass::half_t,
                EpilogueFn>::run(reinterpret_cast<const cutlass::half_t*>(input_ptr),
                                 reinterpret_cast<const cutlass::half_t*>(filter_ptr),
                                 reinterpret_cast<const cutlass::half_t*>(bias_ptr),
                                 reinterpret_cast<cutlass::half_t*>(output_ptr), N, H, W, IC, K, R,
                                 S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f,
                                 stream);
        } else if (input.scalar_type() == torch::kBFloat16) {
            return CutlassConv2dFprop<
                Traits, cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
                EpilogueFn>::run(reinterpret_cast<const cutlass::bfloat16_t*>(input_ptr),
                                 reinterpret_cast<const cutlass::bfloat16_t*>(filter_ptr),
                                 reinterpret_cast<const cutlass::bfloat16_t*>(bias_ptr),
                                 reinterpret_cast<cutlass::bfloat16_t*>(output_ptr), N, H, W, IC,
                                 K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                 1.0f, stream);
        }
        throw std::invalid_argument("Unsupported dtype for Conv2d: expected float16 or bfloat16");
    }
}

// =============================================================================
// Conv2D IC=1 via cuDNN channels_last
// =============================================================================

static torch::Tensor conv2dIC1ViaCuDNN(const torch::Tensor& input,   // [N, H, W, 1]  NHWC
                                       const torch::Tensor& filter,  // [K, R, S, 1]  KRSC
                                       const torch::Tensor& bias,    // [K] or undefined
                                       int pad_h, int pad_w, int stride_h, int stride_w,
                                       int dilation_h, int dilation_w) {
    auto input_nchw = input.permute({0, 3, 1, 2}).contiguous(torch::MemoryFormat::ChannelsLast);
    auto filter_nchw = filter.permute({0, 3, 1, 2}).contiguous(torch::MemoryFormat::ChannelsLast);

    auto out_nchw = torch::conv2d(input_nchw, filter_nchw, bias.defined() ? bias : torch::Tensor(),
                                  /*stride=*/{stride_h, stride_w},
                                  /*padding=*/{pad_h, pad_w},
                                  /*dilation=*/{dilation_h, dilation_w});

    return out_nchw.permute({0, 2, 3, 1}).contiguous();
}

static torch::Tensor conv2dIC1ActivationViaCuDNN(const torch::Tensor& input,
                                                 const torch::Tensor& filter,
                                                 const torch::Tensor& bias,
                                                 ActivationType activation, int pad_h, int pad_w,
                                                 int stride_h, int stride_w, int dilation_h,
                                                 int dilation_w) {
    auto out = conv2dIC1ViaCuDNN(input, filter, bias, pad_h, pad_w, stride_h, stride_w, dilation_h,
                                 dilation_w);

    switch (activation) {
        case ActivationType::RELU:
            return torch::relu(out);
        case ActivationType::GELU:
            return torch::gelu(out);
        case ActivationType::SWISH:
            return torch::silu(out);
        default:
            throw std::invalid_argument("Unsupported activation type");
    }
}

// =============================================================================
// Conv2D NHWC — public API implementations
// =============================================================================

torch::Tensor invokeConv2d(const torch::Tensor& input, const torch::Tensor& filter,
                           const torch::Tensor& bias, int pad_h, int pad_w, int stride_h,
                           int stride_w, int dilation_h, int dilation_w, cudaStream_t stream) {
    const int N = static_cast<int>(input.size(0));
    const int H = static_cast<int>(input.size(1));
    const int W = static_cast<int>(input.size(2));
    const int IC = static_cast<int>(input.size(3));
    const int K = static_cast<int>(filter.size(0));
    const int R = static_cast<int>(filter.size(1));
    const int S = static_cast<int>(filter.size(2));

    const int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    const int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    if (IC == 1) {
        return conv2dIC1ViaCuDNN(input, filter, bias, pad_h, pad_w, stride_h, stride_w, dilation_h,
                                 dilation_w);
    }

    auto output = torch::empty({N, P, Q, K}, input.options());

    const void* input_ptr = input.data_ptr();
    const void* filter_ptr = filter.data_ptr();
    const void* bias_ptr = bias.defined() ? bias.data_ptr() : nullptr;
    void* output_ptr = output.data_ptr();

    if (IC % 8 != 0 || K % 8 != 0) {
        throw std::invalid_argument(
            "Conv2d: IC and K must be multiples of 8 for 128-bit alignment (CUTLASS path). "
            "For IC==1 use the dedicated direct kernel (handled automatically).");
    }

    gemm::GemmStatus status = gemm::GemmStatus::SUCCESS;

    int sm = getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        status = dispatchConv2dDtype<SM_VERSION, EpilogueIdentity>(
            input, input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w, stream);
    });

    if (status != gemm::GemmStatus::SUCCESS) {
        throw std::runtime_error("Conv2d execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return output;
}

torch::Tensor invokeConv2dActivation(const torch::Tensor& input, const torch::Tensor& filter,
                                     const torch::Tensor& bias, ActivationType activation,
                                     int pad_h, int pad_w, int stride_h, int stride_w,
                                     int dilation_h, int dilation_w, cudaStream_t stream) {
    const int N = static_cast<int>(input.size(0));
    const int H = static_cast<int>(input.size(1));
    const int W = static_cast<int>(input.size(2));
    const int IC = static_cast<int>(input.size(3));
    const int K = static_cast<int>(filter.size(0));
    const int R = static_cast<int>(filter.size(1));
    const int S = static_cast<int>(filter.size(2));

    const int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    const int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    if (IC == 1) {
        return conv2dIC1ActivationViaCuDNN(input, filter, bias, activation, pad_h, pad_w, stride_h,
                                           stride_w, dilation_h, dilation_w);
    }

    auto output = torch::empty({N, P, Q, K}, input.options());

    const void* input_ptr = input.data_ptr();
    const void* filter_ptr = filter.data_ptr();
    const void* bias_ptr = bias.defined() ? bias.data_ptr() : nullptr;
    void* output_ptr = output.data_ptr();

    if (IC % 8 != 0 || K % 8 != 0) {
        throw std::invalid_argument(
            "Conv2dActivation: IC and K must be multiples of 8 for 128-bit alignment (CUTLASS "
            "path).");
    }

    gemm::GemmStatus status = gemm::GemmStatus::SUCCESS;

    int sm = getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        if (activation == ActivationType::RELU) {
            status = dispatchConv2dDtype<SM_VERSION, EpilogueRelu>(
                input, input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h,
                pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
        } else if (activation == ActivationType::GELU) {
            status = dispatchConv2dDtype<SM_VERSION, EpilogueGelu>(
                input, input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h,
                pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
        } else if (activation == ActivationType::SWISH) {
            status = dispatchConv2dDtype<SM_VERSION, EpilogueSwish>(
                input, input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h,
                pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
        } else {
            status = gemm::GemmStatus::INVALID_ARGUMENT;
        }
    });

    if (status != gemm::GemmStatus::SUCCESS) {
        throw std::runtime_error("Conv2dActivation execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return output;
}

}  // namespace kernels
}  // namespace oasr
