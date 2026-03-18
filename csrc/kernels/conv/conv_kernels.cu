// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <torch/extension.h>
#include <type_traits>

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
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include "kernels/common/vec_dtypes.h"
#include "kernels/conv/conv_kernels.h"
#include "kernels/gemm/gemm_kernels.h"
#include "kernels/gemm/gemm_utils.h"

namespace oasr {
namespace kernels {

// =============================================================================
// Constants and helpers
// =============================================================================

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

template <typename ElementA, typename ElementB, typename ElementCD>
struct CutlassConv2dFpropSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

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
        cutlass::epilogue::thread::LinearCombination<ElementCD, 8, ElementAccumulator,
                                                     ElementComputeEpilogue,
                                                     cutlass::epilogue::thread::ScaleType::Default>;

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

template <typename ElementA, typename ElementB, typename ElementCD>
struct CutlassConv2dFpropReluSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

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

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementCD, 8, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::Default>;

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

template <typename ElementA, typename ElementB, typename ElementCD>
struct CutlassConv2dFpropGeluSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

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

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementCD, 8, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::Default>;

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

template <typename ElementA, typename ElementB, typename ElementCD>
struct CutlassConv2dFpropSwishSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

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

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementCD, 8, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::Default>;

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
// Depthwise 1D Convolution Kernel
// =============================================================================

template <typename T>
__global__ void depthwiseConv1DKernel(const T* __restrict__ input,   // [batch, seq_len, channels]
                                      const T* __restrict__ weight,  // [kernel_size, channels]
                                      const T* __restrict__ bias,    // [channels] or nullptr
                                      T* __restrict__ output,        // [batch, seq_len, channels]
                                      int batch_size, int seq_len, int channels, int kernel_size,
                                      int padding) {
    int c_id = threadIdx.x;
    int s_id = blockIdx.x;
    int b_id = blockIdx.y;
    int o_id = (blockIdx.y * gridDim.x + blockIdx.x) * channels + c_id;

    int s_start = s_id - padding;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding - s_id, 0);

    input += b_id * seq_len * channels + c_id;
    weight += c_id;

    float val = 0.0f;
    for (int i = s_start; i < s_end; i++) {
        val += (float)input[i * channels] * (float)weight[(k_start + i - s_start) * channels];
    }

    if (bias != nullptr) {
        val += (float)bias[c_id];
    }

    output[o_id] = (T)val;
}

// =============================================================================
// Depthwise 1D Convolution Kernel (vectorized)
// =============================================================================
// Each thread processes VecSize channels using 128-bit vector loads/stores.
// Requires channels % VecSize == 0.
//
// Grid:  (out_len, batch_size)
// Block: (channels / VecSize)

template <typename T, int VecSize>
__global__ void depthwiseConv1DVecKernel(const T* __restrict__ input,  // [batch, seq_len, channels]
                                         const T* __restrict__ weight,  // [kernel_size, channels]
                                         const T* __restrict__ bias,    // [channels] or nullptr
                                         T* __restrict__ output,  // [batch, out_len, channels]
                                         int batch_size, int seq_len, int channels, int kernel_size,
                                         int padding) {
    // Thread ID in the vectorized channel dimension
    const int vec_id = threadIdx.x;  // which vector chunk [0, channels/VecSize)
    const int s_id = blockIdx.x;     // output sequence position
    const int b_id = blockIdx.y;     // batch index

    const int c_offset = vec_id * VecSize;  // starting channel for this thread

    // Compute the valid input range for this output position
    int s_start = s_id - padding;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding - s_id, 0);

    // Pointers for this batch element, offset to the vector chunk
    const T* input_base = input + b_id * seq_len * channels + c_offset;
    const T* weight_base = weight + c_offset;

    // Accumulate in float for numerical stability
    float acc[VecSize];
#pragma unroll
    for (int v = 0; v < VecSize; v++) {
        acc[v] = 0.0f;
    }

    // Main convolution loop
    for (int i = s_start; i < s_end; i++) {
        Vec<T, VecSize> in_vec;
        in_vec.load(input_base + i * channels);

        Vec<T, VecSize> w_vec;
        w_vec.load(weight_base + (k_start + i - s_start) * channels);

#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(in_vec[v]) * static_cast<float>(w_vec[v]);
        }
    }

    // Add bias
    if (bias != nullptr) {
        Vec<T, VecSize> bias_vec;
        bias_vec.load(bias + c_offset);
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(bias_vec[v]);
        }
    }

    // Store result
    const int out_offset = (b_id * gridDim.x + s_id) * channels + c_offset;
    Vec<T, VecSize> out_vec;
#pragma unroll
    for (int v = 0; v < VecSize; v++) {
        out_vec[v] = static_cast<T>(acc[v]);
    }
    out_vec.store(output + out_offset);
}

// =============================================================================
// Fused Depthwise 1D Convolution + SiLU Kernel
// =============================================================================

template <typename T>
__global__ void depthwiseConv1DSiluKernel(
    const T* __restrict__ input,   // [batch, seq_len, channels]
    const T* __restrict__ weight,  // [kernel_size, channels]
    const T* __restrict__ bias,    // [channels] or nullptr
    T* __restrict__ output,        // [batch, seq_len, channels]
    int batch_size, int seq_len, int channels, int kernel_size, int padding) {
    int c_id = threadIdx.x;
    int s_id = blockIdx.x;
    int b_id = blockIdx.y;
    int o_id = (blockIdx.y * gridDim.x + blockIdx.x) * channels + c_id;

    int s_start = s_id - padding;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding - s_id, 0);

    input += b_id * seq_len * channels + c_id;
    weight += c_id;

    float val = 0.0f;
    for (int i = s_start; i < s_end; i++) {
        val += (float)input[i * channels] * (float)weight[(k_start + i - s_start) * channels];
    }

    if (bias != nullptr) {
        val += (float)bias[c_id];
    }

    val = swish(val);

    output[o_id] = (T)val;
}

// =============================================================================
// Depthwise 1D Convolution + SiLU Kernel (vectorized)
// =============================================================================
// Each thread processes VecSize channels using 128-bit vector loads/stores.
// Requires channels % VecSize == 0.
//
// Grid:  (out_len, batch_size)
// Block: (channels / VecSize)

template <typename T, int VecSize>
__global__ void depthwiseConv1DVecSiluKernel(
    const T* __restrict__ input,   // [batch, seq_len, channels]
    const T* __restrict__ weight,  // [kernel_size, channels]
    const T* __restrict__ bias,    // [channels] or nullptr
    T* __restrict__ output,        // [batch, out_len, channels]
    int batch_size, int seq_len, int channels, int kernel_size, int padding) {
    // Thread ID in the vectorized channel dimension
    const int vec_id = threadIdx.x;  // which vector chunk [0, channels/VecSize)
    const int s_id = blockIdx.x;     // output sequence position
    const int b_id = blockIdx.y;     // batch index

    const int c_offset = vec_id * VecSize;  // starting channel for this thread

    // Compute the valid input range for this output position
    int s_start = s_id - padding;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding - s_id, 0);

    // Pointers for this batch element, offset to the vector chunk
    const T* input_base = input + b_id * seq_len * channels + c_offset;
    const T* weight_base = weight + c_offset;

    // Accumulate in float for numerical stability
    float acc[VecSize];
#pragma unroll
    for (int v = 0; v < VecSize; v++) {
        acc[v] = 0.0f;
    }

    // Main convolution loop
    for (int i = s_start; i < s_end; i++) {
        Vec<T, VecSize> in_vec;
        in_vec.load(input_base + i * channels);

        Vec<T, VecSize> w_vec;
        w_vec.load(weight_base + (k_start + i - s_start) * channels);

#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(in_vec[v]) * static_cast<float>(w_vec[v]);
        }
    }

    // Add bias
    if (bias != nullptr) {
        Vec<T, VecSize> bias_vec;
        bias_vec.load(bias + c_offset);
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(bias_vec[v]);
        }
    }

    // Apply SiLU activation
    for (int v = 0; v < VecSize; v++) {
        acc[v] = swish(acc[v]);
    }

    // Store result
    const int out_offset = (b_id * gridDim.x + s_id) * channels + c_offset;
    Vec<T, VecSize> out_vec;
#pragma unroll
    for (int v = 0; v < VecSize; v++) {
        out_vec[v] = static_cast<T>(acc[v]);
    }
    out_vec.store(output + out_offset);
}

// =============================================================================
// GLU (Gated Linear Unit) Kernel
// =============================================================================

template <typename T>
__global__ void gluKernel(const T* __restrict__ input,  // [batch, seq_len, 2 * channels]
                          T* __restrict__ output,       // [batch, seq_len, channels]
                          int batch_size, int seq_len, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;

    if (idx >= total_elements)
        return;

    int c = idx % channels;
    int pos = idx / channels;

    // input[:, :, :channels] * sigmoid(input[:, :, channels:])
    int input_idx1 = pos * (2 * channels) + c;
    int input_idx2 = pos * (2 * channels) + channels + c;

    float x = static_cast<float>(input[input_idx1]);
    float gate = static_cast<float>(input[input_idx2]);
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));

    output[idx] = static_cast<T>(x * sigmoid_gate);
}

// =============================================================================
// GLU (Gated Linear Unit) Kernel (vectorized)
// =============================================================================
// Each thread processes VecSize elements using 128-bit vector loads/stores.
// Requires channels % VecSize == 0.
//
// Grid:  ceil(total_elements / (blockDim.x * VecSize))
// Block: 256

template <typename T, int VecSize>
__global__ void gluVecKernel(const T* __restrict__ input,  // [batch, seq_len, 2 * channels]
                             T* __restrict__ output,       // [batch, seq_len, channels]
                             int batch_size, int seq_len, int channels) {
    const int total_elements = batch_size * seq_len * channels;
    const int total_vec_elements = total_elements / VecSize;
    const int vec_channels = channels / VecSize;

    for (int vid = blockIdx.x * blockDim.x + threadIdx.x; vid < total_vec_elements;
         vid += gridDim.x * blockDim.x) {
        // Decompose vid into (pos, vec_c) where pos = batch * seq_len + seq_pos
        const int vec_c = vid % vec_channels;
        const int pos = vid / vec_channels;

        const int c_offset = vec_c * VecSize;

        // Load value half: input[pos, c_offset]
        const int input_idx1 = pos * (2 * channels) + c_offset;
        Vec<T, VecSize> val_vec;
        val_vec.load(input + input_idx1);

        // Load gate half: input[pos, channels + c_offset]
        const int input_idx2 = pos * (2 * channels) + channels + c_offset;
        Vec<T, VecSize> gate_vec;
        gate_vec.load(input + input_idx2);

        // Compute x * sigmoid(gate) in float
        Vec<T, VecSize> out_vec;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            float x = static_cast<float>(val_vec[v]);
            float gate = static_cast<float>(gate_vec[v]);
            float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
            out_vec[v] = static_cast<T>(x * sigmoid_gate);
        }

        // Store result
        const int out_idx = pos * channels + c_offset;
        out_vec.store(output + out_idx);
    }
}

// =============================================================================
// Swish Kernel
// =============================================================================

template <typename T>
__global__ void swishKernel(const T* __restrict__ input, T* __restrict__ output, int batch_size,
                            int seq_len, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;

    if (idx >= total_elements)
        return;

    float x = static_cast<float>(input[idx]);
    float result = x / (1.0f + expf(-x));
    output[idx] = static_cast<T>(result);
}

// =============================================================================
// Fused BatchNorm + Swish Kernel
// =============================================================================

template <typename T>
__global__ void batchNormSwishKernel(const T* __restrict__ input, T* __restrict__ output,
                                     const T* __restrict__ weight, const T* __restrict__ bias,
                                     const T* __restrict__ running_mean,
                                     const T* __restrict__ running_var, int batch_size, int seq_len,
                                     int channels, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;

    if (idx >= total_elements)
        return;

    int c = idx % channels;

    float x = static_cast<float>(input[idx]);
    float mean = static_cast<float>(running_mean[c]);
    float var = static_cast<float>(running_var[c]);
    float g = static_cast<float>(weight[c]);
    float b = static_cast<float>(bias[c]);

    // BatchNorm
    float inv_std = rsqrtf(var + eps);
    float normalized = (x - mean) * inv_std * g + b;

    // Swish
    float result = normalized / (1.0f + expf(-normalized));

    output[idx] = static_cast<T>(result);
}

// =============================================================================
// Causal Conv1D with State Kernel
// =============================================================================

template <typename T>
__global__ void causalConv1DKernel(const T* __restrict__ input,  // [batch, chunk_len, channels]
                                   T* __restrict__ state,        // [batch, kernel_size-1, channels]
                                   const T* __restrict__ weight,  // [channels, 1, kernel_size]
                                   const T* __restrict__ bias,    // [channels] or nullptr
                                   T* __restrict__ output,        // [batch, chunk_len, channels]
                                   int batch_size, int chunk_len, int channels, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * chunk_len * channels;

    if (idx >= total_elements)
        return;

    int c = idx % channels;
    int t = (idx / channels) % chunk_len;
    int b = idx / (channels * chunk_len);

    int state_len = kernel_size - 1;

    float sum = 0.0f;

    // Compute convolution using state and current input
    for (int k = 0; k < kernel_size; k++) {
        int input_pos = t - (kernel_size - 1) + k;
        float val;

        if (input_pos < 0) {
            // Read from state buffer
            int state_pos = state_len + input_pos;  // Maps -state_len to 0, etc.
            int state_idx = b * state_len * channels + state_pos * channels + c;
            val = static_cast<float>(state[state_idx]);
        } else {
            // Read from current input
            int input_idx = b * chunk_len * channels + input_pos * channels + c;
            val = static_cast<float>(input[input_idx]);
        }

        int weight_idx = c * kernel_size + k;
        sum += val * static_cast<float>(weight[weight_idx]);
    }

    if (bias != nullptr) {
        sum += static_cast<float>(bias[c]);
    }

    output[idx] = static_cast<T>(sum);
}

// Update state buffer after processing chunk
template <typename T>
__global__ void updateConvStateKernel(const T* __restrict__ input,  // [batch, chunk_len, channels]
                                      T* __restrict__ state,        // [batch, state_len, channels]
                                      int batch_size, int chunk_len, int channels, int state_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * state_len * channels;

    if (idx >= total_elements)
        return;

    int c = idx % channels;
    int s = (idx / channels) % state_len;
    int b = idx / (channels * state_len);

    // New state comes from:
    // - Old state shifted left (if chunk_len < state_len)
    // - Or entirely from new input (if chunk_len >= state_len)
    int source_pos;
    if (chunk_len >= state_len) {
        // Take from input: last state_len positions
        source_pos = chunk_len - state_len + s;
        int input_idx = b * chunk_len * channels + source_pos * channels + c;
        state[idx] = input[input_idx];
    } else {
        // Mix: shift old state and add new input
        int shift = state_len - chunk_len;
        if (s < shift) {
            // From old state
            int old_state_idx = b * state_len * channels + (s + chunk_len) * channels + c;
            state[idx] = state[old_state_idx];
        } else {
            // From new input
            source_pos = s - shift;
            int input_idx = b * chunk_len * channels + source_pos * channels + c;
            state[idx] = input[input_idx];
        }
    }
}

// =============================================================================
// Dispatcher functions
// =============================================================================

template <typename T>
void invokeDepthwiseConv1DTyped(const torch::Tensor& input, const torch::Tensor& weight,
                                const torch::Tensor& bias, torch::Tensor& output, int padding,
                                cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(0);

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;
    T* output_ptr = static_cast<T*>(output.data_ptr());

    const int out_len = seq_len + 2 * padding - kernel_size + 1;
    dim3 grid_size(out_len, batch_size);

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    // Use vectorized kernel when channels are aligned to VecSize
    // and the thread count fits within hardware limits
    if (channels % kVecSize == 0 && (channels / kVecSize) <= 1024) {
        dim3 block_size(channels / kVecSize);
        depthwiseConv1DVecKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, seq_len, channels, kernel_size,
            padding);
    } else {
        dim3 block_size(channels);
        depthwiseConv1DKernel<T><<<grid_size, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, seq_len, channels, kernel_size,
            padding);
    }
}

template <typename T>
void invokeDepthwiseConv1DSiluTyped(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, torch::Tensor& output, int padding,
                                    cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(0);

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;
    T* output_ptr = static_cast<T*>(output.data_ptr());

    const int out_len = seq_len + 2 * padding - kernel_size + 1;
    dim3 grid_size(out_len, batch_size);

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    // Use vectorized kernel when channels are aligned to VecSize
    // and the thread count fits within hardware limits
    if (channels % kVecSize == 0 && (channels / kVecSize) <= 1024) {
        dim3 block_size(channels / kVecSize);
        depthwiseConv1DVecSiluKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, seq_len, channels, kernel_size,
            padding);
    } else {
        dim3 block_size(channels);
        depthwiseConv1DSiluKernel<T><<<grid_size, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, seq_len, channels, kernel_size,
            padding);
    }
}

// =============================================================================
// Public API implementations
// =============================================================================

torch::Tensor invokeDepthwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, int padding, cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);
    auto output = torch::empty({batch_size, seq_len + 2 * padding - kernel_size + 1, channels},
                               input.options());
    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            invokeDepthwiseConv1DTyped<float>(input, weight, bias, output, padding, stream);
            break;
        case torch::ScalarType::Half:
            invokeDepthwiseConv1DTyped<half>(input, weight, bias, output, padding, stream);
            break;
        case torch::ScalarType::BFloat16:
            invokeDepthwiseConv1DTyped<__nv_bfloat16>(input, weight, bias, output, padding, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for DepthwiseConv1D");
    }
    return output;
}

torch::Tensor invokeDepthwiseConv1DSilu(const torch::Tensor& input, const torch::Tensor& weight,
                                        const torch::Tensor& bias, int padding,
                                        cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);
    auto output = torch::empty({batch_size, seq_len + 2 * padding - kernel_size + 1, channels},
                               input.options());

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            invokeDepthwiseConv1DSiluTyped<float>(input, weight, bias, output, padding, stream);
            break;
        case torch::ScalarType::Half:
            invokeDepthwiseConv1DSiluTyped<half>(input, weight, bias, output, padding, stream);
            break;
        case torch::ScalarType::BFloat16:
            invokeDepthwiseConv1DSiluTyped<__nv_bfloat16>(input, weight, bias, output, padding,
                                                          stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for DepthwiseConv1DSilu");
            break;
    }
    return output;
}

torch::Tensor invokePointwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int out_channels = weight.size(0);

    using namespace oasr::kernels::gemm;

    auto output = gemm::invokeGemm(input, weight, bias, stream);
    output = output.view({batch_size, seq_len, out_channels});
    return output;
}

torch::Tensor invokePointwiseConv1DActivation(const torch::Tensor& input,
                                              const torch::Tensor& weight,
                                              const torch::Tensor& bias, ActivationType activation,
                                              cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int out_channels = weight.size(0);

    using namespace oasr::kernels::gemm;

    auto output = gemm::invokeGemmActivation(input, weight, bias, activation, stream);
    output = output.view({batch_size, seq_len, out_channels});
    return output;
}

template <typename T>
void invokeGLUTyped(const torch::Tensor& input, torch::Tensor& output, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2) / 2;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());

    const int total_elements = batch_size * seq_len * channels;

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    if (channels % kVecSize == 0) {
        const int total_vec_elements = total_elements / kVecSize;
        const int block_size = 256;
        int grid_size = (total_vec_elements + block_size - 1) / block_size;
        // Cap grid size to avoid excessive launches
        grid_size = min(grid_size, 65535);
        gluVecKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input_ptr, output_ptr, batch_size, seq_len, channels);
    } else {
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        gluKernel<T><<<grid_size, block_size, 0, stream>>>(input_ptr, output_ptr, batch_size,
                                                           seq_len, channels);
    }
}

torch::Tensor invokeGLU(const torch::Tensor& input, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2) / 2;

    auto output = torch::empty({batch_size, seq_len, channels}, input.options());

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            invokeGLUTyped<float>(input, output, stream);
            break;
        case torch::ScalarType::Half:
            invokeGLUTyped<half>(input, output, stream);
            break;
        case torch::ScalarType::BFloat16:
            invokeGLUTyped<__nv_bfloat16>(input, output, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for GLU");
            break;
    }

    return output;
}

torch::Tensor invokeSwish(const torch::Tensor& input, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);

    auto output = torch::empty_like(input);

    const void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();

    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            swishKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr), static_cast<float*>(output_ptr), batch_size,
                seq_len, channels);
            break;
        case torch::ScalarType::Half:
            swishKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr), static_cast<half*>(output_ptr), batch_size,
                seq_len, channels);
            break;
        case torch::ScalarType::BFloat16:
            swishKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(output_ptr), batch_size, seq_len, channels);
            break;
        default:
            throw std::runtime_error("Unsupported data type for Swish");
            break;
    }

    return output;
}

torch::Tensor invokeBatchNormSwish(const torch::Tensor& input, const torch::Tensor& weight,
                                   const torch::Tensor& bias, const torch::Tensor& running_mean,
                                   const torch::Tensor& running_var, float eps,
                                   cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);

    auto output = torch::empty_like(input);

    const void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();
    const void* weight_ptr = weight.data_ptr();
    const void* bias_ptr = bias.data_ptr();
    const void* running_mean_ptr = running_mean.data_ptr();
    const void* running_var_ptr = running_var.data_ptr();

    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            batchNormSwishKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr), static_cast<float*>(output_ptr),
                static_cast<const float*>(weight_ptr), static_cast<const float*>(bias_ptr),
                static_cast<const float*>(running_mean_ptr),
                static_cast<const float*>(running_var_ptr), batch_size, seq_len, channels, eps);
            break;
        case torch::ScalarType::Half:
            batchNormSwishKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr), static_cast<half*>(output_ptr),
                static_cast<const half*>(weight_ptr), static_cast<const half*>(bias_ptr),
                static_cast<const half*>(running_mean_ptr),
                static_cast<const half*>(running_var_ptr), batch_size, seq_len, channels, eps);
            break;
        case torch::ScalarType::BFloat16:
            batchNormSwishKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(output_ptr),
                static_cast<const __nv_bfloat16*>(weight_ptr),
                static_cast<const __nv_bfloat16*>(bias_ptr),
                static_cast<const __nv_bfloat16*>(running_mean_ptr),
                static_cast<const __nv_bfloat16*>(running_var_ptr), batch_size, seq_len, channels,
                eps);
            break;
        default:
            throw std::runtime_error("Unsupported data type for BatchNormSwish");
            break;
    }

    return output;
}

torch::Tensor invokeCausalConv1D(const torch::Tensor& input, void* state_buffer,
                                 const torch::Tensor& weight, const torch::Tensor& bias,
                                 cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int chunk_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(-1);

    auto output = torch::empty_like(input);

    const void* input_ptr = input.data_ptr();
    const void* weight_ptr = weight.data_ptr();
    const void* bias_ptr = bias.defined() ? bias.data_ptr() : nullptr;
    void* output_ptr = output.data_ptr();

    int total_elements = batch_size * chunk_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    int state_len = kernel_size - 1;

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            causalConv1DKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr), static_cast<float*>(state_buffer),
                static_cast<const float*>(weight_ptr), static_cast<const float*>(bias_ptr),
                static_cast<float*>(output_ptr), batch_size, chunk_len, channels, kernel_size);
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<float><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const float*>(input_ptr), static_cast<float*>(state_buffer),
                    batch_size, chunk_len, channels, state_len);
            }
            break;
        case torch::ScalarType::Half:
            causalConv1DKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr), static_cast<half*>(state_buffer),
                static_cast<const half*>(weight_ptr), static_cast<const half*>(bias_ptr),
                static_cast<half*>(output_ptr), batch_size, chunk_len, channels, kernel_size);
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<half><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const half*>(input_ptr), static_cast<half*>(state_buffer),
                    batch_size, chunk_len, channels, state_len);
            }
            break;
        case torch::ScalarType::BFloat16:
            causalConv1DKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(state_buffer),
                static_cast<const __nv_bfloat16*>(weight_ptr),
                static_cast<const __nv_bfloat16*>(bias_ptr),
                static_cast<__nv_bfloat16*>(output_ptr), batch_size, chunk_len, channels,
                kernel_size);
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<__nv_bfloat16><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input_ptr),
                    static_cast<__nv_bfloat16*>(state_buffer), batch_size, chunk_len, channels,
                    state_len);
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type for CausalConv1D");
            break;
    }

    return output;
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
        status = CutlassConv2dFpropSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t>::run(
            reinterpret_cast<const cutlass::half_t*>(input_ptr),
            reinterpret_cast<const cutlass::half_t*>(filter_ptr),
            reinterpret_cast<const cutlass::half_t*>(bias_ptr),
            reinterpret_cast<cutlass::half_t*>(output_ptr), N, H, W, IC, K, R, S, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
    } else if (input.scalar_type() == torch::kBFloat16) {
        status =
            CutlassConv2dFpropSM80<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t>::
                run(reinterpret_cast<const cutlass::bfloat16_t*>(input_ptr),
                    reinterpret_cast<const cutlass::bfloat16_t*>(filter_ptr),
                    reinterpret_cast<const cutlass::bfloat16_t*>(bias_ptr),
                    reinterpret_cast<cutlass::bfloat16_t*>(output_ptr), N, H, W, IC, K, R, S, pad_h,
                    pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
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

    gemm::GemmStatus status = gemm::GemmStatus::SUCCESS;
    if (input.scalar_type() == torch::kHalf) {
        if (activation == ActivationType::RELU) {
            status =
                CutlassConv2dFpropReluSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t>::run(
                    reinterpret_cast<const cutlass::half_t*>(input_ptr),
                    reinterpret_cast<const cutlass::half_t*>(filter_ptr),
                    reinterpret_cast<const cutlass::half_t*>(bias_ptr),
                    reinterpret_cast<cutlass::half_t*>(output_ptr), N, H, W, IC, K, R, S, pad_h,
                    pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
        } else if (activation == ActivationType::GELU) {
            status =
                CutlassConv2dFpropGeluSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t>::run(
                    reinterpret_cast<const cutlass::half_t*>(input_ptr),
                    reinterpret_cast<const cutlass::half_t*>(filter_ptr),
                    reinterpret_cast<const cutlass::half_t*>(bias_ptr),
                    reinterpret_cast<cutlass::half_t*>(output_ptr), N, H, W, IC, K, R, S, pad_h,
                    pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
        } else {
            status =
                CutlassConv2dFpropSwishSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t>::run(
                    reinterpret_cast<const cutlass::half_t*>(input_ptr),
                    reinterpret_cast<const cutlass::half_t*>(filter_ptr),
                    reinterpret_cast<const cutlass::half_t*>(bias_ptr),
                    reinterpret_cast<cutlass::half_t*>(output_ptr), N, H, W, IC, K, R, S, pad_h,
                    pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
        }
    } else if (input.scalar_type() == torch::kBFloat16) {
        if (activation == ActivationType::RELU) {
            status = CutlassConv2dFpropReluSM80<
                cutlass::bfloat16_t, cutlass::bfloat16_t,
                cutlass::bfloat16_t>::run(reinterpret_cast<const cutlass::bfloat16_t*>(input_ptr),
                                          reinterpret_cast<const cutlass::bfloat16_t*>(filter_ptr),
                                          reinterpret_cast<const cutlass::bfloat16_t*>(bias_ptr),
                                          reinterpret_cast<cutlass::bfloat16_t*>(output_ptr), N, H,
                                          W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
                                          dilation_h, dilation_w, 1.0f, stream);
        } else if (activation == ActivationType::GELU) {
            status = CutlassConv2dFpropGeluSM80<
                cutlass::bfloat16_t, cutlass::bfloat16_t,
                cutlass::bfloat16_t>::run(reinterpret_cast<const cutlass::bfloat16_t*>(input_ptr),
                                          reinterpret_cast<const cutlass::bfloat16_t*>(filter_ptr),
                                          reinterpret_cast<const cutlass::bfloat16_t*>(bias_ptr),
                                          reinterpret_cast<cutlass::bfloat16_t*>(output_ptr), N, H,
                                          W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
                                          dilation_h, dilation_w, 1.0f, stream);
        } else {
            status = CutlassConv2dFpropSwishSM80<
                cutlass::bfloat16_t, cutlass::bfloat16_t,
                cutlass::bfloat16_t>::run(reinterpret_cast<const cutlass::bfloat16_t*>(input_ptr),
                                          reinterpret_cast<const cutlass::bfloat16_t*>(filter_ptr),
                                          reinterpret_cast<const cutlass::bfloat16_t*>(bias_ptr),
                                          reinterpret_cast<cutlass::bfloat16_t*>(output_ptr), N, H,
                                          W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
                                          dilation_h, dilation_w, 1.0f, stream);
        }
    } else {
        throw std::invalid_argument(
            "Unsupported dtype for Conv2dActivation: expected float16 or bfloat16");
    }

    if (status != gemm::GemmStatus::SUCCESS) {
        throw std::runtime_error("Conv2dActivation execution failed: " +
                                 std::to_string(static_cast<int>(status)));
    }

    return output;
}

// Explicit template instantiations
template void invokeDepthwiseConv1DTyped<float>(const torch::Tensor&, const torch::Tensor&,
                                                const torch::Tensor&, torch::Tensor&, int,
                                                cudaStream_t);
template void invokeDepthwiseConv1DTyped<half>(const torch::Tensor&, const torch::Tensor&,
                                               const torch::Tensor&, torch::Tensor&, int,
                                               cudaStream_t);
template void invokeDepthwiseConv1DTyped<__nv_bfloat16>(const torch::Tensor&, const torch::Tensor&,
                                                        const torch::Tensor&, torch::Tensor&, int,
                                                        cudaStream_t);

}  // namespace kernels
}  // namespace oasr