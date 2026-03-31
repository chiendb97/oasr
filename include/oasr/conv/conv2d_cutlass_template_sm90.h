// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// SM90 (Hopper) Conv2D fprop template -- CUTLASS 3.x TMA warp-specialized implicit GEMM.
//
// Uses CUTLASS 3.x CollectiveBuilder pattern with im2col TMA for activation and
// regular TMA for filter. Supports KernelImplicitTmaWarpSpecializedSm90 scheduling.
//
// Reference: NVIDIA/cutlass test/unit/conv/device_3x/fprop/
//            sm90_conv2d_fprop_implicit_gemm_f16_f16_f32_tensorop_f16.cu

#pragma once

#include <cstdint>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cutlass/conv/collective/collective_builder.hpp>
#include <cutlass/conv/convnd_problem_shape.hpp>
#include <cutlass/conv/device/conv_universal_adapter.hpp>
#include <cutlass/conv/dispatch_policy.hpp>
#include <cutlass/conv/kernel/conv_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>

#include <cute/tensor.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/epilogue_functors.h>
#include <oasr/common/utils.h>
#include <oasr/conv/cutlass_conv2d_configs.h>

namespace oasr {
namespace conv {

//==============================================================================
// CutlassConv2dFpropKernelSm90 -- CUTLASS 3.x Conv2D fprop for SM90+
//
// Parameterized by CutlassConv2dConfigSm90, element types, and activation type.
// Input/filter/output are all NHWC-packed (TensorNHWC layout).
//==============================================================================

template <typename CutlassConv2dConfig, typename ElementA, typename ElementB, typename ElementCD,
          oasr::ActivationType activation_type>
struct CutlassConv2dFpropKernelSm90 {
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ArchTag = typename CutlassConv2dConfig::SmArch;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    // NHWC layout for all tensors (activation, filter, output)
    using LayoutA = cutlass::layout::TensorNHWC;
    using LayoutB = cutlass::layout::TensorNHWC;
    using LayoutCD = cutlass::layout::TensorNHWC;

    // 128-bit alignment
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using TileShape = typename CutlassConv2dConfig::TileShape;
    using ClusterShape = typename CutlassConv2dConfig::ClusterShape;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    using KernelSchedule = cutlass::conv::collective::KernelScheduleAuto;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

    // 2D conv fprop problem shape
    using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 2>;

    // CUTLASS 3.x fusion operation for epilogue
    using FusionOp =
        typename FusionEpilogueOpSm90<activation_type, ElementCD, ElementCompute, ElementCD>::type;

    // Build epilogue collective
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementCD, LayoutCD, AlignmentEpilogue, ElementCD, LayoutCD,
        AlignmentEpilogue, EpilogueSchedule, FusionOp>::CollectiveOp;

    // Build mainloop collective (im2col TMA for activation, TMA for filter)
    using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
        ArchTag, OperatorClass, cutlass::conv::Operator::kFprop, ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    // Assemble conv kernel and device adapter
    using ConvKernel =
        cutlass::conv::kernel::ConvUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

    using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

    // Stride types -- derived from the NHWC layout mapping in CUTLASS 3.x:
    //   StrideA: Stride<Stride<int64_t, int64_t, int64_t>, Int<1>>   ((Q,H,N), C=1)
    //   StrideB: depends on filter layout (KRSC)
    //   StrideCD: Stride<Stride<int64_t, int64_t, int64_t>, Int<1>, Int<0>>  ((Q,P,N), C=1, L=0)
    using StrideCD = typename Conv::ConvKernel::StrideD;

    static gemm::GemmStatus run(const ElementA* input, const ElementB* filter,
                                const ElementCD* bias, ElementCD* output, int N, int H, int W,
                                int IC, int K, int R, int S, int pad_h, int pad_w, int stride_h,
                                int stride_w, int dilation_h, int dilation_w, ElementCompute alpha,
                                cudaStream_t stream) {
        // Compute output spatial dimensions
        int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
        int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

        // Build ConvProblemShape from user parameters
        // shape_act = [N, H, W, IC], shape_flt = [K, R, S, IC]
        // lower_padding = upper_padding = [pad_h, pad_w]
        // traversal_stride = [stride_h, stride_w], dilation = [dilation_h, dilation_w]
        ProblemShape problem_shape{
            cutlass::conv::Mode::kCrossCorrelation,
            {N, H, W, IC},             // shape_act: NHWC
            {K, R, S, IC},             // shape_flt: KRSC (NHWC convention for filters)
            {pad_h, pad_w},            // lower_padding
            {pad_h, pad_w},            // upper_padding (symmetric)
            {stride_h, stride_w},      // traversal stride
            {dilation_h, dilation_w},  // dilation
            1                          // groups
        };

        // Mainloop arguments: just the raw pointers (strides embedded in problem shape)
        typename CollectiveMainloop::Arguments mainloop_args{input, filter};

        // Epilogue output stride for NHWC-packed tensor (N, P, Q, K)
        // StrideCD maps to ((Q_stride, P_stride, N_stride), C=1, L=0)
        // For packed NHWC: Q_stride=K, P_stride=Q*K, N_stride=P*Q*K
        StrideCD stride_D{};
        cute::get<0, 0>(stride_D) = (int64_t)K;          // Q dim stride
        cute::get<0, 1>(stride_D) = (int64_t)Q * K;      // P dim stride
        cute::get<0, 2>(stride_D) = (int64_t)P * Q * K;  // N dim stride
        // Int<1> (C stride) and Int<0> (L/batch stride) are static

        // For bias: broadcast over spatial dims by setting M-stride to 0.
        // Output-channel stride = 1 (innermost, matches Int<1> in StrideCD).
        StrideCD stride_C{};
        cute::get<0, 0>(stride_C) = 0;  // Q dim: broadcast
        cute::get<0, 1>(stride_C) = 0;  // P dim: broadcast
        cute::get<0, 2>(stride_C) = 0;  // N dim: broadcast

        ElementCompute beta = (bias == nullptr) ? ElementCompute(0) : ElementCompute(1);

        // Use output as fallback for ptr_C when bias is null (beta=0 makes it a no-op).
        // Avoids creating a TMA descriptor with a nullptr address during initialize().
        const ElementCD* ptr_C = (bias != nullptr) ? bias : output;

        // Build epilogue arguments
        typename CollectiveEpilogue::Arguments epilogue_args{
            {alpha, beta},  // thread (FusionCallbacks::Arguments)
            ptr_C,          // ptr_C
            stride_C,       // dC (broadcast when bias is nullptr)
            output,         // ptr_D
            stride_D        // dD
        };

        typename Conv::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm, problem_shape,
                                           mainloop_args, epilogue_args};

        Conv conv_op;

        cutlass::Status status = conv_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            return gemm::GemmStatus::NOT_SUPPORTED;
        }

        size_t workspace_size = conv_op.get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        status = conv_op.initialize(arguments, workspace.get(), stream);
        if (status != cutlass::Status::kSuccess) {
            return gemm::GemmStatus::INTERNAL_ERROR;
        }

        status = conv_op.run(stream);
        if (status != cutlass::Status::kSuccess) {
            return gemm::GemmStatus::CUTLASS_ERROR;
        }

        return gemm::GemmStatus::SUCCESS;
    }
};

}  // namespace conv
}  // namespace oasr
