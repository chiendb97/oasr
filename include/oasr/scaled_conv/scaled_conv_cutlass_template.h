// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS 2.x int8 implicit-GEMM forward convolution:
//
//     out[n, p, q, k] = act(scale[k] * sum_{r, s, c} x[n, h, w, c] * w[k, r, s, c] + bias[k])
//
// NHWC int8 input (statically quantized), KRSC int8 filter (per-output-channel
// quantized), fp16/bf16 NHWC output.  The int32 sum is exact, so every tile
// computes the same bits and the kernel is batch-invariant.  The epilogue is
// ``ScaledConvOutputOp`` through ``DefaultConv2dFpropWithBroadcast``: the
// per-channel scale is the broadcast vector, the bias a zero-stride source.
// A conv1d over ``(B, T, C)`` is this kernel at height one (``R = 1``).

#pragma once

#include <cuda_runtime.h>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/scaled_conv/cutlass_scaled_conv_configs.h>
#include <oasr/scaled_conv/scaled_conv_epilogue.h>

namespace oasr {
namespace scaled_conv {

enum class ScaledConvStatus { SUCCESS = 0, CANNOT_IMPLEMENT, INITIALIZE_FAILED, RUN_FAILED };

/// The geometry of one forward convolution (NHWC input, KRSC filter).
struct ConvGeometry {
    int N, H, W, C;
    int K, R, S;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dilation_h, dilation_w;

    int P() const { return (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1; }
    int Q() const { return (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1; }
};

template <typename Config, typename ElementD, ActivationType act>
struct CutlassScaledConv2d {
    using ElementAB = typename Config::ElementAB;
    using ElementAccumulator = typename Config::ElementAccumulator;
    using Layout = cutlass::layout::TensorNHWC;

    static constexpr int kAlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;  // 16
    static constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;    // 8

    using OutputOp = ScaledConvOutputOp<ElementD, ElementAccumulator, kAlignmentD, act>;

    using Kernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
        ElementAB, Layout, ElementAB, Layout, ElementD, Layout, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, typename Config::ArchTag,
        typename Config::ThreadblockShape, typename Config::WarpShape,
        typename Config::InstructionShape, OutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Config::kStages,
        typename Config::Operator, cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided, kAlignmentAB, kAlignmentAB>::Kernel;

    using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

    static ScaledConvStatus run(ElementAB const* x, ElementAB const* w, float const* scale,
                                ElementD const* bias, ElementD* out, ConvGeometry const& g,
                                cudaStream_t stream) {
        const int P = g.P(), Q = g.Q();
        cutlass::conv::Conv2dProblemSize problem(
            {g.N, g.H, g.W, g.C}, {g.K, g.R, g.S, g.C}, {g.pad_h, g.pad_h, g.pad_w, g.pad_w},
            {g.stride_h, g.stride_w}, {g.dilation_h, g.dilation_w}, {g.N, P, Q, g.K},
            cutlass::conv::Mode::kCrossCorrelation, /*split_k_slices=*/1);

        auto layout_x = Layout::packed({g.N, g.H, g.W, g.C});
        auto layout_w = Layout::packed({g.K, g.R, g.S, g.C});
        auto layout_out = Layout::packed({g.N, P, Q, g.K});
        // Every output pixel reads bias[k]: a zero-stride source tensor.
        auto layout_bias = Layout(cutlass::make_Coord(0, 0, 0));

        typename Conv::Arguments args(
            problem, {const_cast<ElementAB*>(x), layout_x}, {const_cast<ElementAB*>(w), layout_w},
            {const_cast<ElementD*>(bias), layout_bias}, {out, layout_out},
            typename OutputOp::Params(bias != nullptr), cutlass::conv::SplitKMode::kSerial,
            const_cast<float*>(scale), /*ptr_Tensor=*/nullptr, /*ldr=*/0, /*ldt=*/0);

        Conv conv;
        if (conv.can_implement(args) != cutlass::Status::kSuccess) {
            return ScaledConvStatus::CANNOT_IMPLEMENT;
        }
        // One K pass, no split: no workspace, so the launch stays capturable.
        if (conv.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) {
            return ScaledConvStatus::INITIALIZE_FAILED;
        }
        if (conv(stream) != cutlass::Status::kSuccess) {
            return ScaledConvStatus::RUN_FAILED;
        }
        return ScaledConvStatus::SUCCESS;
    }
};

}  // namespace scaled_conv
}  // namespace oasr
