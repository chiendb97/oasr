// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA/CUTLASS conv2d kernels — no framework dependencies.
// Traits-parameterized Conv2D forward propagation using CUTLASS 2.x API
// with architecture-aware dispatch.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/arch_traits.h>
#include <oasr/common/epilogue_functors.h>
#include <oasr/common/tuneable_traits.h>
#include <oasr/common/types.h>

namespace oasr {
namespace conv {

// =============================================================================
// Status Codes
// =============================================================================

enum class Conv2dStatus {
    SUCCESS = 0,
    INVALID_ARGUMENT,
    NOT_SUPPORTED,
    INTERNAL_ERROR,
    CUTLASS_ERROR,
};

inline const char* getConv2dStatusString(Conv2dStatus status) {
    switch (status) {
        case Conv2dStatus::SUCCESS:
            return "SUCCESS";
        case Conv2dStatus::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case Conv2dStatus::NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case Conv2dStatus::INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case Conv2dStatus::CUTLASS_ERROR:
            return "CUTLASS_ERROR";
        default:
            return "UNKNOWN";
    }
}

// =============================================================================
// CUTLASS Conv2D Forward Propagation Template
// =============================================================================

template <typename Traits, typename ElementA, typename ElementB, typename ElementCD,
          template <int, typename, typename> class EpilogueFunctor>
struct CutlassConv2dFprop {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

    // Use scalar alignment for maximum compatibility with any channel count.
    // TensorOp with alignment>1 requires IC/K divisible by alignment, but
    // conformer subsampling has IC=1 which isn't aligned to 8.
    static constexpr int kAlignmentA = 1;
    static constexpr int kAlignmentB = 1;

    // cp_async (used by kOptimized iterator on SM80+ with 3+ stages) requires
    // minimum 4-byte copies. With scalar alignment and fp16/bf16 (2 bytes),
    // we must fall back to kAnalytic + 2 stages to avoid the cp_async assertion.
    static constexpr int kElementBytes = cutlass::sizeof_bits<ElementA>::value / 8;
    static constexpr bool kNeedsCpAsyncFallback =
        (kAlignmentA * kElementBytes < 4) && (Traits::Conv2d::NumStages > 2);

    static constexpr int EpilogueAlignment = Traits::Conv2d::EpilogueAlignment;

    using LayoutInput = cutlass::layout::TensorNHWC;
    using LayoutFilter = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    using MMAOp = typename Traits::MMAOp;
    using SmArch = typename Traits::SmArch;
    using ShapeMMAThreadBlock = typename Traits::Conv2d::ThreadBlock;
    using ShapeMMAWarps = typename Traits::Conv2d::Warps;
    using ShapeMMAOp = typename Traits::Conv2d::MMAShape;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    static constexpr int NumStages = kNeedsCpAsyncFallback ? 2 : Traits::Conv2d::NumStages;
    static constexpr cutlass::conv::IteratorAlgorithm IterAlgo = kNeedsCpAsyncFallback
        ? cutlass::conv::IteratorAlgorithm::kAnalytic : Traits::Conv2d::IterAlgo;
    static constexpr cutlass::conv::StrideSupport OutStride = kNeedsCpAsyncFallback
        ? cutlass::conv::StrideSupport::kStrided : Traits::Conv2d::OutStride;

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

// =============================================================================
// Arch-dispatched dtype helper
// =============================================================================

namespace detail {

template <int SmVersion, template <int, typename, typename> class EpilogueFn, typename ElementType>
static Conv2dStatus dispatchConv2dElement(const void* input_ptr, const void* filter_ptr,
                                          const void* bias_ptr, void* output_ptr, int N, int H,
                                          int W, int IC, int K, int R, int S, int pad_h, int pad_w,
                                          int stride_h, int stride_w, int dilation_h,
                                          int dilation_w, cudaStream_t stream) {
    if constexpr (SmVersion < 80) {
        return Conv2dStatus::NOT_SUPPORTED;
    } else {
        using Traits = TuneableTraits<SmVersion>;
        return CutlassConv2dFprop<Traits, ElementType, ElementType, ElementType, EpilogueFn>::run(
            reinterpret_cast<const ElementType*>(input_ptr),
            reinterpret_cast<const ElementType*>(filter_ptr),
            reinterpret_cast<const ElementType*>(bias_ptr),
            reinterpret_cast<ElementType*>(output_ptr), N, H, W, IC, K, R, S, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
    }
}

}  // namespace detail

// =============================================================================
// Typed Launchers — raw pointer interface, returns cudaError_t
// =============================================================================

/**
 * @brief 2D convolution using CUTLASS Tensor Core Implicit GEMM
 *
 * Tensors must be in NHWC layout:
 *   input  [N, H, W, IC]
 *   filter [K, R, S, IC]
 *   output [N, P, Q, K]  where P = (H + 2*pad_h - dilation_h*(R-1) - 1) / stride_h + 1
 *                              Q = (W + 2*pad_w - dilation_w*(S-1) - 1) / stride_w + 1
 *
 * Alignment requirement: IC % 8 == 0 and K % 8 == 0 (128-bit vector loads).
 * Requires SM80+.
 *
 * @tparam T  Element type — must be cutlass::half_t or cutlass::bfloat16_t
 * @param input      Input data pointer [N, H, W, IC]
 * @param filter     Filter data pointer [K, R, S, IC]
 * @param bias       Optional per-channel bias pointer [K], nullptr to skip
 * @param output     Output data pointer [N, P, Q, K]
 * @param N          Batch size
 * @param H          Input height
 * @param W          Input width
 * @param IC         Input channels
 * @param K          Output channels (number of filters)
 * @param R          Filter height
 * @param S          Filter width
 * @param pad_h      Symmetric padding along H
 * @param pad_w      Symmetric padding along W
 * @param stride_h   Convolution stride along H
 * @param stride_w   Convolution stride along W
 * @param dilation_h Dilation along H
 * @param dilation_w Dilation along W
 * @param stream     CUDA stream
 * @return cudaError_t  cudaSuccess on success, error code otherwise
 */
template <typename T>
cudaError_t Conv2D(const T* input, const T* filter, const T* bias, T* output, int N, int H, int W,
                   int IC, int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w,
                   int dilation_h, int dilation_w, cudaStream_t stream) {
    Conv2dStatus status = Conv2dStatus::SUCCESS;

    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        status = detail::dispatchConv2dElement<SM_VERSION, EpilogueIdentity, T>(
            input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, stream);
    });

    if (status != Conv2dStatus::SUCCESS) {
        return cudaErrorNotSupported;
    }

    return cudaGetLastError();
}

/**
 * @brief 2D convolution with fused activation using CUTLASS Tensor Core Implicit GEMM
 *
 * Computes output = activation(conv2d(input, filter) + bias).
 * Same layout and alignment requirements as Conv2D.
 *
 * @tparam T  Element type — must be cutlass::half_t or cutlass::bfloat16_t
 * @param input      Input data pointer [N, H, W, IC]
 * @param filter     Filter data pointer [K, R, S, IC]
 * @param bias       Optional per-channel bias pointer [K], nullptr to skip
 * @param output     Output data pointer [N, P, Q, K]
 * @param activation Fused activation: RELU, GELU, or SWISH
 * @param N          Batch size
 * @param H          Input height
 * @param W          Input width
 * @param IC         Input channels
 * @param K          Output channels (number of filters)
 * @param R          Filter height
 * @param S          Filter width
 * @param pad_h      Symmetric padding along H
 * @param pad_w      Symmetric padding along W
 * @param stride_h   Convolution stride along H
 * @param stride_w   Convolution stride along W
 * @param dilation_h Dilation along H
 * @param dilation_w Dilation along W
 * @param stream     CUDA stream
 * @return cudaError_t  cudaSuccess on success, error code otherwise
 */
template <typename T>
cudaError_t Conv2DActivation(const T* input, const T* filter, const T* bias, T* output,
                             ActivationType activation, int N, int H, int W, int IC, int K, int R,
                             int S, int pad_h, int pad_w, int stride_h, int stride_w,
                             int dilation_h, int dilation_w, cudaStream_t stream) {
    Conv2dStatus status = Conv2dStatus::SUCCESS;

    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        if (activation == ActivationType::RELU) {
            status = detail::dispatchConv2dElement<SM_VERSION, EpilogueRelu, T>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                stride_w, dilation_h, dilation_w, stream);
        } else if (activation == ActivationType::GELU) {
            status = detail::dispatchConv2dElement<SM_VERSION, EpilogueGelu, T>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                stride_w, dilation_h, dilation_w, stream);
        } else if (activation == ActivationType::SWISH) {
            status = detail::dispatchConv2dElement<SM_VERSION, EpilogueSwish, T>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                stride_w, dilation_h, dilation_w, stream);
        } else {
            status = Conv2dStatus::INVALID_ARGUMENT;
        }
    });

    if (status != Conv2dStatus::SUCCESS) {
        return cudaErrorNotSupported;
    }

    return cudaGetLastError();
}

}  // namespace conv
}  // namespace oasr
