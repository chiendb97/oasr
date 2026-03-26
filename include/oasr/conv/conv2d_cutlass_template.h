// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS Conv2D forward propagation template -- parameterized by Conv2dConfig + SmConv2dTraits.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

#include <oasr/conv/cutlass_conv2d_configs.h>

namespace oasr {
namespace conv {

//==============================================================================
// Status Codes
//==============================================================================

enum class Conv2dStatus {
    SUCCESS = 0,
    INVALID_ARGUMENT,
    NOT_SUPPORTED,
    INTERNAL_ERROR,
    CUTLASS_ERROR,
};

inline const char* getConv2dStatusString(Conv2dStatus status) {
    switch (status) {
        case Conv2dStatus::SUCCESS:       return "SUCCESS";
        case Conv2dStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case Conv2dStatus::NOT_SUPPORTED: return "NOT_SUPPORTED";
        case Conv2dStatus::INTERNAL_ERROR: return "INTERNAL_ERROR";
        case Conv2dStatus::CUTLASS_ERROR: return "CUTLASS_ERROR";
        default:                          return "UNKNOWN";
    }
}

//==============================================================================
// CutlassConv2dFpropKernel -- parameterized by Config + MMATraits
//==============================================================================

template <typename Config, typename MMATraits, typename ElementA, typename ElementB,
          typename ElementCD, template <int, typename, typename> class EpilogueFunctor>
struct CutlassConv2dFpropKernel {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

    // Scalar alignment for maximum IC compatibility (conformer subsampling has IC=1)
    static constexpr int kAlignmentA = 1;
    static constexpr int kAlignmentB = 1;

    // cp_async fallback: scalar alignment with fp16/bf16 (2 bytes) < 4 bytes minimum
    static constexpr int kElementBytes = cutlass::sizeof_bits<ElementA>::value / 8;
    static constexpr bool kNeedsCpAsyncFallback =
        (kAlignmentA * kElementBytes < 4) && (Config::NumStages > 2);

    static constexpr int EpilogueAlignment = Config::EpilogueAlignment;

    using LayoutInput = cutlass::layout::TensorNHWC;
    using LayoutFilter = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    using MMAOp = typename MMATraits::MMAOp;
    using SmArch = typename MMATraits::SmArch;
    using ShapeMMAThreadBlock = typename Config::ThreadBlock;
    using ShapeMMAWarps = typename Config::Warps;
    using ShapeMMAOp = typename MMATraits::MMAShape;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    static constexpr int NumStages = kNeedsCpAsyncFallback ? 2 : Config::NumStages;
    static constexpr cutlass::conv::IteratorAlgorithm IterAlgo = kNeedsCpAsyncFallback
        ? cutlass::conv::IteratorAlgorithm::kAnalytic : Config::IterAlgo;
    static constexpr cutlass::conv::StrideSupport OutStride = kNeedsCpAsyncFallback
        ? cutlass::conv::StrideSupport::kStrided : Config::OutStride;

    using EpilogueOp =
        typename EpilogueFunctor<EpilogueAlignment, ElementCD, ElementComputeEpilogue>::Op;

    using Conv2dKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementA, LayoutInput, ElementB, LayoutFilter, ElementCD, LayoutOutput, ElementAccumulator,
        MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp, EpilogueOp,
        SwizzleThreadBlock, NumStages, cutlass::arch::OpMultiplyAdd, IterAlgo, OutStride,
        kAlignmentA, kAlignmentB>::Kernel;

    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernel>;

    static Conv2dStatus run(const ElementA* input, const ElementB* filter, const ElementCD* bias,
                            ElementCD* output, int N, int H, int W, int IC, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                            int dilation_w, ElementComputeEpilogue alpha, cudaStream_t stream) {
        int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
        int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

        cutlass::conv::Conv2dProblemSize problem_size(
            {N, H, W, IC}, {K, R, S, IC}, {pad_h, pad_h, pad_w, pad_w}, {stride_h, stride_w},
            {dilation_h, dilation_w}, {N, P, Q, K}, cutlass::conv::Mode::kCrossCorrelation, 1);

        auto layout_input = LayoutInput::packed({N, H, W, IC});
        auto layout_filter = LayoutFilter::packed({K, R, S, IC});
        auto layout_output = LayoutOutput::packed({N, P, Q, K});
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
            return Conv2dStatus::NOT_SUPPORTED;
        }

        size_t workspace_size = ImplicitGemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        status = conv_op.initialize(args, workspace.get(), stream);
        if (status != cutlass::Status::kSuccess) {
            return Conv2dStatus::INTERNAL_ERROR;
        }

        status = conv_op(stream);
        if (status != cutlass::Status::kSuccess) {
            return Conv2dStatus::CUTLASS_ERROR;
        }

        return Conv2dStatus::SUCCESS;
    }
};

}  // namespace conv
}  // namespace oasr
