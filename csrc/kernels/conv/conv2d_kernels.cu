// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/extension.h>

// Suppress warnings from CUTLASS headers
#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include "kernels/common/epilogue_functors.h"
#include "kernels/conv/conv2d_kernels.h"
#include "kernels/gemm/gemm_utils.h"

namespace oasr {
namespace kernels {

// =============================================================================
// Constants and helpers
// =============================================================================

// =============================================================================
// CUTLASS SM80 Conv2D Fprop helpers (Alignment = 8, kOptimized)
// =============================================================================
//
// All variants operate on packed NHWC tensors:
//   Input  [N, H, W, IC]  (TensorNHWC)
//   Filter [K, R, S, IC]  (TensorNHWC)
//   Output [N, P, Q, K]   (TensorNHWC)
//
// Requires IC % 8 == 0  and  K % 8 == 0  (128-bit alignment, FP16/BF16).
// For IC == 1, use conv2dIC1Kernel instead (see below).
//
// Bias (optional per-channel [K]) is fused into the CUTLASS epilogue via a
// zero-stride broadcast: layout_bias = LayoutOutput(make_Coord(0,0,0)).

template <typename ElementA, typename ElementB, typename ElementCD,
          template <int, typename, typename> class EpilogueFunctor>
struct CutlassConv2dFpropSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

    static constexpr int EpilogueAlignment = 8;

    using LayoutInput = cutlass::layout::TensorNHWC;
    using LayoutFilter = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 64>;
    using ShapeMMAWarps = cutlass::gemm::GemmShape<64, 64, 64>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    static constexpr int NumStages = 3;
    static cutlass::conv::IteratorAlgorithm const IterAlgo =
        cutlass::conv::IteratorAlgorithm::kOptimized;
    static cutlass::conv::StrideSupport const OutStride = cutlass::conv::StrideSupport::kUnity;

    using EpilogueOp =
        typename EpilogueFunctor<EpilogueAlignment, ElementCD, ElementComputeEpilogue>::Op;

    using Conv2dKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementA, LayoutInput, ElementB, LayoutFilter, ElementCD, LayoutOutput, ElementAccumulator,
        MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp, EpilogueOp,
        SwizzleThreadBlock, NumStages, cutlass::arch::OpMultiplyAdd, IterAlgo, OutStride, 8,
        8>::Kernel;

    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernel>;

    static gemm::GemmStatus run(const ElementA* input, const ElementB* filter,
                                const ElementCD* bias, ElementCD* output, int N, int H, int W,
                                int IC, int K, int R, int S, int pad_h, int pad_w, int stride_h,
                                int stride_w, int dilation_h, int dilation_w,
                                ElementComputeEpilogue alpha, cudaStream_t stream) {
        int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
        int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

        cutlass::conv::Conv2dProblemSize problem_size(
            {N, H, W, IC}, {K, R, S, IC}, {pad_h, pad_h, pad_w, pad_w}, {stride_h, stride_w},
            {dilation_h, dilation_w}, {N, P, Q, K}, cutlass::conv::Mode::kCrossCorrelation, 1);

        auto layout_input = LayoutInput::packed({N, H, W, IC});
        auto layout_filter = LayoutFilter::packed({K, R, S, IC});
        auto layout_output = LayoutOutput::packed({N, P, Q, K});
        // Bias [K] broadcast across N, P, Q via zero strides
        auto layout_bias = LayoutOutput(cutlass::make_Coord(0, 0, 0));
        ElementComputeEpilogue beta =
            (bias == nullptr) ? ElementComputeEpilogue(0) : ElementComputeEpilogue(1);

        typename ImplicitGemm::Arguments args{
            problem_size,
            {const_cast<ElementA*>(input), layout_input},
            {const_cast<ElementB*>(filter), layout_filter},
            {const_cast<ElementCD*>(bias), layout_bias},
            {output, layout_output},
            {alpha, beta},
        };

        ImplicitGemm conv_op;
        cutlass::Status status = conv_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            return gemm::GemmStatus::NOT_SUPPORTED;
        }

        size_t workspace_size = ImplicitGemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        status = conv_op.initialize(args, workspace.get(), stream);
        if (status != cutlass::Status::kSuccess) {
            return gemm::GemmStatus::INTERNAL_ERROR;
        }

        status = conv_op(stream);
        if (status != cutlass::Status::kSuccess) {
            return gemm::GemmStatus::CUTLASS_ERROR;
        }

        return gemm::GemmStatus::SUCCESS;
    }
};

// =============================================================================
// Conv2D IC=1 via cuDNN channels_last
// =============================================================================

// =============================================================================
// Conv2D IC=1 helper: cuDNN channels_last path
// =============================================================================
//
// For IC==1 inputs (e.g. Conv2dSubsampling first layer), we delegate to
// torch::conv2d with channels_last memory format.  This invokes cuDNN's
// highly-optimised NHWC kernels:
//
//  1. NHWC [N,H,W,1] → view as NCHW [N,1,H,W] (free – same memory layout).
//  2. Mark as channels_last (metadata only, no copy).
//  3. torch::conv2d → cuDNN NHWC path.
//  4. Output is channels_last [N,K,P,Q] → permute to [N,P,Q,K] (free –
//     channels_last memory is already NHWC order).
//
// This is dramatically faster than a custom direct-conv kernel because cuDNN
// uses specialised algorithms (implicit GEMM, Winograd, etc.) with tuned
// tiling and memory access patterns.

static torch::Tensor conv2dIC1ViaCuDNN(const torch::Tensor& input,   // [N, H, W, 1]  NHWC
                                       const torch::Tensor& filter,  // [K, R, S, 1]  KRSC
                                       const torch::Tensor& bias,    // [K] or undefined
                                       int pad_h, int pad_w, int stride_h, int stride_w,
                                       int dilation_h, int dilation_w) {
    // View as NCHW and mark channels_last (no data movement for IC/OC=1).
    auto input_nchw = input.permute({0, 3, 1, 2}).contiguous(torch::MemoryFormat::ChannelsLast);
    auto filter_nchw = filter.permute({0, 3, 1, 2}).contiguous(torch::MemoryFormat::ChannelsLast);

    auto out_nchw = torch::conv2d(input_nchw, filter_nchw, bias.defined() ? bias : torch::Tensor(),
                                  /*stride=*/{stride_h, stride_w},
                                  /*padding=*/{pad_h, pad_w},
                                  /*dilation=*/{dilation_h, dilation_w});

    // channels_last [N,K,P,Q] → logical [N,P,Q,K]; memory is already NHWC.
    return out_nchw.permute({0, 2, 3, 1}).contiguous();
}

static torch::Tensor conv2dIC1ActivationViaCuDNN(const torch::Tensor& input,   // [N, H, W, 1]  NHWC
                                                 const torch::Tensor& filter,  // [K, R, S, 1]  KRSC
                                                 const torch::Tensor& bias,    // [K] or undefined
                                                 ActivationType activation, int pad_h, int pad_w,
                                                 int stride_h, int stride_w, int dilation_h,
                                                 int dilation_w) {
    // Conv2d via cuDNN
    auto out = conv2dIC1ViaCuDNN(input, filter, bias, pad_h, pad_w, stride_h, stride_w, dilation_h,
                                 dilation_w);

    // Apply activation (out is contiguous [N,P,Q,K])
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

    // General path: CUTLASS kOptimized (requires IC % 8 == 0 && K % 8 == 0).
    if (IC % 8 != 0 || K % 8 != 0) {
        throw std::invalid_argument(
            "Conv2d: IC and K must be multiples of 8 for 128-bit alignment (CUTLASS path). "
            "For IC==1 use the dedicated direct kernel (handled automatically).");
    }

    gemm::GemmStatus status = gemm::GemmStatus::SUCCESS;
    if (input.scalar_type() == torch::kHalf) {
        status = CutlassConv2dFpropSM80<
            cutlass::half_t, cutlass::half_t, cutlass::half_t,
            EpilogueIdentity>::run(reinterpret_cast<const cutlass::half_t*>(input_ptr),
                                   reinterpret_cast<const cutlass::half_t*>(filter_ptr),
                                   reinterpret_cast<const cutlass::half_t*>(bias_ptr),
                                   reinterpret_cast<cutlass::half_t*>(output_ptr), N, H, W, IC, K,
                                   R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                   1.0f, stream);
    } else if (input.scalar_type() == torch::kBFloat16) {
        status = CutlassConv2dFpropSM80<
            cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
            EpilogueIdentity>::run(reinterpret_cast<const cutlass::bfloat16_t*>(input_ptr),
                                   reinterpret_cast<const cutlass::bfloat16_t*>(filter_ptr),
                                   reinterpret_cast<const cutlass::bfloat16_t*>(bias_ptr),
                                   reinterpret_cast<cutlass::bfloat16_t*>(output_ptr), N, H, W, IC,
                                   K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h,
                                   dilation_w, 1.0f, stream);
    } else {
        throw std::invalid_argument("Unsupported dtype for Conv2d: expected float16 or bfloat16");
    }

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

    // General path: CUTLASS kOptimized (requires IC % 8 == 0 && K % 8 == 0).
    if (IC % 8 != 0 || K % 8 != 0) {
        throw std::invalid_argument(
            "Conv2dActivation: IC and K must be multiples of 8 for 128-bit alignment (CUTLASS "
            "path).");
    }

    // Dispatch activation type, then dtype
#define DISPATCH_CONV2D_ACTIVATION(EpilogueFn)                                                     \
    if (input.scalar_type() == torch::kHalf) {                                                     \
        status = CutlassConv2dFpropSM80<                                                           \
            cutlass::half_t, cutlass::half_t, cutlass::half_t,                                     \
            EpilogueFn>::run(reinterpret_cast<const cutlass::half_t*>(input_ptr),                  \
                             reinterpret_cast<const cutlass::half_t*>(filter_ptr),                 \
                             reinterpret_cast<const cutlass::half_t*>(bias_ptr),                   \
                             reinterpret_cast<cutlass::half_t*>(output_ptr), N, H, W, IC, K, R, S, \
                             pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f,       \
                             stream);                                                              \
    } else if (input.scalar_type() == torch::kBFloat16) {                                          \
        status = CutlassConv2dFpropSM80<                                                           \
            cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,                         \
            EpilogueFn>::run(reinterpret_cast<const cutlass::bfloat16_t*>(input_ptr),              \
                             reinterpret_cast<const cutlass::bfloat16_t*>(filter_ptr),             \
                             reinterpret_cast<const cutlass::bfloat16_t*>(bias_ptr),               \
                             reinterpret_cast<cutlass::bfloat16_t*>(output_ptr), N, H, W, IC, K,   \
                             R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, \
                             stream);                                                              \
    } else {                                                                                       \
        throw std::invalid_argument(                                                               \
            "Unsupported dtype for Conv2dActivation: expected float16 or bfloat16");               \
    }

    gemm::GemmStatus status = gemm::GemmStatus::SUCCESS;
    if (activation == ActivationType::RELU) {
        DISPATCH_CONV2D_ACTIVATION(EpilogueRelu)
    } else if (activation == ActivationType::GELU) {
        DISPATCH_CONV2D_ACTIVATION(EpilogueGelu)
    } else if (activation == ActivationType::SWISH) {
        DISPATCH_CONV2D_ACTIVATION(EpilogueSwish)
    } else {
        status = gemm::GemmStatus::INVALID_ARGUMENT;
    }

#undef DISPATCH_CONV2D_ACTIVATION

    if (status != gemm::GemmStatus::SUCCESS) {
        throw std::runtime_error("Conv2dActivation execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return output;
}

}  // namespace kernels
}  // namespace oasr