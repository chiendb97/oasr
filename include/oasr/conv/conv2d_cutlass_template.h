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

#include <oasr/common/epilogue_functors.h>
#include <oasr/common/utils.h>

namespace oasr {
namespace conv {

//==============================================================================
// CutlassConv2dFpropKernel -- parameterized by Config + MMATraits
//==============================================================================

template <typename CutlassConv2dConfig, typename ElementA, typename ElementB, typename ElementCD,
          oasr::ActivationType activation_type>
struct CutlassConv2dFpropKernel {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

    // Scalar alignment for maximum IC compatibility (conformer subsampling has IC=1)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using LayoutInput = cutlass::layout::TensorNHWC;
    using LayoutFilter = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = typename CutlassConv2dConfig::SmArch;
    using ShapeMMAThreadBlock = typename CutlassConv2dConfig::ThreadBlock;
    using ShapeMMAWarps = typename CutlassConv2dConfig::Warps;
    using ShapeMMAOp = typename CutlassConv2dConfig::MMAShape;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    static constexpr int NumStages = CutlassConv2dConfig::NumStages;
    static constexpr cutlass::conv::IteratorAlgorithm IterAlgo =
        cutlass::conv::IteratorAlgorithm::kOptimized;
    static constexpr cutlass::conv::StrideSupport OutStride =
        cutlass::conv::StrideSupport::kStrided;

    using EpilogueOp = typename FusionEpilogueOp<activation_type, AlignmentEpilogue, ElementCD,
                                                 ElementComputeEpilogue, ElementCD>::type;

    using Conv2dKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementA, LayoutInput, ElementB, LayoutFilter, ElementCD, LayoutOutput, ElementAccumulator,
        MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp, EpilogueOp,
        SwizzleThreadBlock, NumStages, cutlass::arch::OpMultiplyAdd, IterAlgo, OutStride,
        AlignmentA, AlignmentB>::Kernel;

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

}  // namespace conv
}  // namespace oasr
