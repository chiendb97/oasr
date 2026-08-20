// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for fused recurrent kernels.

#include <cstdint>
#include <limits>
#include <oasr/gemm/gemm.cuh>
#include <oasr/recurrent/recurrent.cuh>
#include <oasr/recurrent/recurrent_cutlass.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

namespace {

template <typename T>
struct ToCutlassType {
    using type = T;
};

template <>
struct ToCutlassType<half> {
    using type = cutlass::half_t;
};

template <>
struct ToCutlassType<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};

bool SameDtype(const TensorView& a, const TensorView& b) {
    return a.dtype().code == b.dtype().code && a.dtype().bits == b.dtype().bits &&
           a.dtype().lanes == b.dtype().lanes;
}

void CheckTensorLike(const TensorView& tensor, const TensorView& reference, const char* name) {
    CHECK_INPUT(tensor);
    CHECK_DEVICE(tensor, reference);
    CHECK_CONTIGUOUS_INPUT(tensor);
    TVM_FFI_ICHECK(SameDtype(tensor, reference)) << name << " dtype must match input";
}

struct RecurrentShape {
    int sequence_length;
    int batch_size;
    int input_size;
    int hidden_size;
    int64_t input_time_stride;
    int64_t input_batch_stride;
    int64_t output_time_stride;
    int64_t output_batch_stride;
};

struct GemmRecurrentShape {
    int sequence_length;
    int batch_size;
    int hidden_size;
    int64_t input_time_stride;
    int64_t input_batch_stride;
};

enum class RecurrentGemmTactic : int64_t {
    FUSED_16X64 = 0,
    FUSED_32X64 = 1,
    FUSED_64X64 = 2,
    STREAM_K = 3,
    PARALLEL_SPLIT_K = 4,
    SERIAL_SPLIT_K = 5,
};

void CheckRecurrentTactic(int64_t tactic, int64_t split_k_slices) {
    TVM_FFI_ICHECK(tactic >= static_cast<int64_t>(RecurrentGemmTactic::FUSED_16X64) &&
                   tactic <= static_cast<int64_t>(RecurrentGemmTactic::SERIAL_SPLIT_K))
        << "unknown recurrent GEMM tactic " << tactic;
    TVM_FFI_ICHECK(split_k_slices >= 1 && split_k_slices <= 16)
        << "split_k_slices must be in [1, 16], got " << split_k_slices;
    if (tactic == static_cast<int64_t>(RecurrentGemmTactic::PARALLEL_SPLIT_K) ||
        tactic == static_cast<int64_t>(RecurrentGemmTactic::SERIAL_SPLIT_K)) {
        TVM_FFI_ICHECK(split_k_slices > 1)
            << "a split-K recurrent tactic requires split_k_slices > 1";
    }
}

template <typename Config, typename CutlassType>
gemm::GemmStatus RunLstmFused(CutlassType* output, CutlassType* cells, CutlassType* final_h,
                              CutlassType* final_c, const CutlassType* previous_h,
                              const CutlassType* weight_hh, const CutlassType* input_gates,
                              const CutlassType* previous_c, CutlassType* split_k_workspace,
                              int batch_size, int hidden_size, int64_t input_gate_batch_stride,
                              int64_t previous_cell_batch_stride, cudaStream_t stream,
                              int split_k_slices = 1) {
    return recurrent::LstmStateGemm<Config, CutlassType>(
        output, cells, final_h, final_c, previous_h, weight_hh, input_gates, previous_c,
        split_k_workspace, batch_size, hidden_size, input_gate_batch_stride,
        previous_cell_batch_stride, stream, split_k_slices);
}

template <typename Config, bool kStreamK, bool kParallelSplitK, typename CutlassType,
          typename CudaType>
cudaError_t RunLstmMaterialized(CudaType* output, CudaType* cells, CudaType* final_h,
                                CudaType* final_c, const CutlassType* previous_h,
                                const CutlassType* weight_hh, const CutlassType* input_gates,
                                const CudaType* previous_c, CudaType* workspace, int batch_size,
                                int hidden_size, int64_t input_gate_batch_stride,
                                int64_t previous_cell_batch_stride, cudaStream_t stream,
                                int split_k_slices) {
    gemm::GemmStatus status =
        gemm::CutlassGemmKernel<Config, CutlassType, CutlassType, CutlassType,
                                ActivationType::IDENTITY, kStreamK,
                                kParallelSplitK>::run(previous_h, weight_hh, input_gates,
                                                      reinterpret_cast<CutlassType*>(workspace),
                                                      batch_size, 4 * hidden_size, hidden_size,
                                                      hidden_size, hidden_size,
                                                      input_gate_batch_stride, 1.0f, stream,
                                                      split_k_slices, false, 4 * hidden_size);
    if (status != gemm::GemmStatus::SUCCESS)
        return cudaErrorUnknown;
    return recurrent::LstmInterleavedGateStep<CudaType>(output, cells, final_h, final_c, workspace,
                                                        previous_c, batch_size, hidden_size,
                                                        previous_cell_batch_stride, stream);
}

template <typename Config, recurrent::RnnActivation Activation, bool kStreamK, bool kParallelSplitK,
          typename CutlassType>
gemm::GemmStatus RunRnnFused(CutlassType* output, const CutlassType* previous_h,
                             const CutlassType* weight_hh, const CutlassType* input_gates,
                             int batch_size, int hidden_size, int64_t input_gate_batch_stride,
                             cudaStream_t stream, int split_k_slices) {
    return recurrent::RnnStateGemm<Config, CutlassType, Activation, kStreamK, kParallelSplitK>(
        output, previous_h, weight_hh, input_gates, batch_size, hidden_size,
        input_gate_batch_stride, stream, split_k_slices);
}

template <typename CutlassType, recurrent::RnnActivation Activation>
gemm::GemmStatus RunRnnTactic(RecurrentGemmTactic tactic, CutlassType* output,
                              const CutlassType* previous_h, const CutlassType* weight_hh,
                              const CutlassType* input_gates, int batch_size, int hidden_size,
                              int64_t input_gate_batch_stride, cudaStream_t stream,
                              int split_k_slices) {
    switch (tactic) {
        case RecurrentGemmTactic::FUSED_16X64:
            return RunRnnFused<recurrent::RecurrentConfig16x64, Activation, false, false>(
                output, previous_h, weight_hh, input_gates, batch_size, hidden_size,
                input_gate_batch_stride, stream, split_k_slices);
        case RecurrentGemmTactic::FUSED_32X64:
            return RunRnnFused<recurrent::RecurrentConfig32x64, Activation, false, false>(
                output, previous_h, weight_hh, input_gates, batch_size, hidden_size,
                input_gate_batch_stride, stream, split_k_slices);
        case RecurrentGemmTactic::FUSED_64X64:
            return RunRnnFused<recurrent::RecurrentConfig64x64, Activation, false, false>(
                output, previous_h, weight_hh, input_gates, batch_size, hidden_size,
                input_gate_batch_stride, stream, split_k_slices);
        case RecurrentGemmTactic::STREAM_K:
            return RunRnnFused<recurrent::RecurrentConfig32x64, Activation, true, false>(
                output, previous_h, weight_hh, input_gates, batch_size, hidden_size,
                input_gate_batch_stride, stream, split_k_slices);
        case RecurrentGemmTactic::PARALLEL_SPLIT_K:
            return RunRnnFused<recurrent::RecurrentConfig32x64, Activation, false, true>(
                output, previous_h, weight_hh, input_gates, batch_size, hidden_size,
                input_gate_batch_stride, stream, split_k_slices);
        case RecurrentGemmTactic::SERIAL_SPLIT_K:
            return gemm::GemmStatus::NOT_SUPPORTED;
    }
    return gemm::GemmStatus::NOT_SUPPORTED;
}

RecurrentShape CheckCommon(TensorView output, TensorView final_h, TensorView input,
                           TensorView initial_h, TensorView weight_ih, TensorView weight_hh,
                           Optional bias_ih, Optional bias_hh, int gate_count, bool batch_first) {
    CHECK_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_DIM(3, input);
    CheckTensorLike(output, input, "output");
    CheckTensorLike(final_h, input, "final_h");
    CheckTensorLike(initial_h, input, "initial_h");
    CheckTensorLike(weight_ih, input, "weight_ih");
    CheckTensorLike(weight_hh, input, "weight_hh");
    CHECK_DIM(3, output);
    CHECK_DIM(2, final_h);
    CHECK_DIM(2, initial_h);
    CHECK_DIM(2, weight_ih);
    CHECK_DIM(2, weight_hh);

    const int64_t sequence_length = input.size(batch_first ? 1 : 0);
    const int64_t batch_size = input.size(batch_first ? 0 : 1);
    const int64_t input_size = input.size(2);
    const int64_t hidden_size = weight_hh.size(1);
    TVM_FFI_ICHECK(sequence_length > 0 && batch_size > 0 && input_size > 0 && hidden_size > 0)
        << "recurrent input dimensions must be positive, got sequence=" << sequence_length
        << " batch=" << batch_size << " input_size=" << input_size
        << " hidden_size=" << hidden_size;
    TVM_FFI_ICHECK(sequence_length <= std::numeric_limits<int>::max() && batch_size <= 65535 &&
                   input_size <= std::numeric_limits<int>::max() && hidden_size <= 65535)
        << "recurrent shape exceeds CUDA grid/index limits";

    TVM_FFI_ICHECK(weight_ih.size(0) == gate_count * hidden_size && weight_ih.size(1) == input_size)
        << "weight_ih must have shape (" << gate_count * hidden_size << ", " << input_size << ")";
    TVM_FFI_ICHECK(weight_hh.size(0) == gate_count * hidden_size)
        << "weight_hh must have shape (" << gate_count * hidden_size << ", " << hidden_size << ")";
    TVM_FFI_ICHECK(initial_h.size(0) == batch_size && initial_h.size(1) == hidden_size)
        << "initial_h must have shape (" << batch_size << ", " << hidden_size << ")";
    TVM_FFI_ICHECK(final_h.size(0) == batch_size && final_h.size(1) == hidden_size)
        << "final_h must have shape (" << batch_size << ", " << hidden_size << ")";
    TVM_FFI_ICHECK(output.size(batch_first ? 0 : 1) == batch_size &&
                   output.size(batch_first ? 1 : 0) == sequence_length &&
                   output.size(2) == hidden_size)
        << "output leading dimensions must match input and its last dimension must be "
        << hidden_size;

    for (const auto& named_bias : {std::pair<const char*, Optional>{"bias_ih", bias_ih},
                                   std::pair<const char*, Optional>{"bias_hh", bias_hh}}) {
        if (!named_bias.second.has_value())
            continue;
        TensorView bias = named_bias.second.value();
        CheckTensorLike(bias, input, named_bias.first);
        CHECK_DIM(1, bias);
        TVM_FFI_ICHECK(bias.size(0) == gate_count * hidden_size)
            << named_bias.first << " must have " << gate_count * hidden_size << " elements";
    }

    const int time_dim = batch_first ? 1 : 0;
    const int batch_dim = batch_first ? 0 : 1;
    return {
        static_cast<int>(sequence_length),
        static_cast<int>(batch_size),
        static_cast<int>(input_size),
        static_cast<int>(hidden_size),
        input.stride(time_dim),
        input.stride(batch_dim),
        output.stride(time_dim),
        output.stride(batch_dim),
    };
}

GemmRecurrentShape CheckGemmCommon(TensorView output, TensorView final_h, TensorView workspace,
                                   TensorView input_gates, TensorView initial_h,
                                   TensorView weight_hh, Optional bias_hh, int gate_count,
                                   bool input_batch_first) {
    CHECK_INPUT(input_gates);
    CHECK_CONTIGUOUS_INPUT(input_gates);
    CHECK_DIM(3, input_gates);
    CheckTensorLike(output, input_gates, "output");
    CheckTensorLike(final_h, input_gates, "final_h");
    CheckTensorLike(workspace, input_gates, "workspace");
    CheckTensorLike(initial_h, input_gates, "initial_h");
    CheckTensorLike(weight_hh, input_gates, "weight_hh");
    CHECK_DIM(3, output);
    CHECK_DIM(2, final_h);
    CHECK_DIM(2, workspace);
    CHECK_DIM(2, initial_h);
    CHECK_DIM(2, weight_hh);

    const int64_t sequence_length = input_gates.size(input_batch_first ? 1 : 0);
    const int64_t batch_size = input_gates.size(input_batch_first ? 0 : 1);
    const int64_t hidden_size = weight_hh.size(1);
    TVM_FFI_ICHECK(sequence_length > 0 && batch_size > 0 && hidden_size > 0)
        << "recurrent input dimensions must be positive";
    TVM_FFI_ICHECK(sequence_length <= std::numeric_limits<int>::max() && batch_size <= 65535 &&
                   hidden_size <= 65535)
        << "recurrent shape exceeds CUDA grid/index limits";
    TVM_FFI_ICHECK(hidden_size % 8 == 0)
        << "tensor-core recurrent path requires hidden_size divisible by 8, got " << hidden_size;
    TVM_FFI_ICHECK(input_gates.size(2) == gate_count * hidden_size)
        << "input_gates last dimension must be " << gate_count * hidden_size;
    TVM_FFI_ICHECK(weight_hh.size(0) == gate_count * hidden_size)
        << "weight_hh must have shape (" << gate_count * hidden_size << ", " << hidden_size << ")";
    TVM_FFI_ICHECK(initial_h.size(0) == batch_size && initial_h.size(1) == hidden_size)
        << "initial_h must have shape (" << batch_size << ", " << hidden_size << ")";
    TVM_FFI_ICHECK(final_h.size(0) == batch_size && final_h.size(1) == hidden_size)
        << "final_h must have shape (" << batch_size << ", " << hidden_size << ")";
    TVM_FFI_ICHECK(output.size(0) == sequence_length && output.size(1) == batch_size &&
                   output.size(2) == hidden_size)
        << "tensor-core recurrent output must be time-major (sequence, batch, hidden)";
    TVM_FFI_ICHECK(workspace.size(0) == batch_size && workspace.size(1) == gate_count * hidden_size)
        << "workspace must have shape (" << batch_size << ", " << gate_count * hidden_size << ")";
    if (bias_hh.has_value()) {
        TensorView bias = bias_hh.value();
        CheckTensorLike(bias, input_gates, "bias_hh");
        CHECK_DIM(1, bias);
        TVM_FFI_ICHECK(bias.size(0) == gate_count * hidden_size)
            << "bias_hh must have " << gate_count * hidden_size << " elements";
    }
    const int time_dim = input_batch_first ? 1 : 0;
    const int batch_dim = input_batch_first ? 0 : 1;
    return {
        static_cast<int>(sequence_length), static_cast<int>(batch_size),
        static_cast<int>(hidden_size),     input_gates.stride(time_dim),
        input_gates.stride(batch_dim),
    };
}

}  // namespace

void lstm_layer(TensorView output, TensorView final_h, TensorView final_c, TensorView cells,
                TensorView input, TensorView initial_h, TensorView initial_c, TensorView weight_ih,
                TensorView weight_hh, Optional bias_ih, Optional bias_hh, bool batch_first) {
    const RecurrentShape shape = CheckCommon(output, final_h, input, initial_h, weight_ih,
                                             weight_hh, bias_ih, bias_hh, 4, batch_first);
    CheckTensorLike(cells, input, "cells");
    CheckTensorLike(final_c, input, "final_c");
    CheckTensorLike(initial_c, input, "initial_c");
    CHECK_DIM(3, cells);
    CHECK_DIM(2, final_c);
    CHECK_DIM(2, initial_c);
    for (int dim = 0; dim < 3; ++dim) {
        TVM_FFI_ICHECK(cells.size(dim) == output.size(dim))
            << "cells must have the same shape as output";
    }
    TVM_FFI_ICHECK(initial_c.size(0) == shape.batch_size && initial_c.size(1) == shape.hidden_size)
        << "initial_c must have shape (" << shape.batch_size << ", " << shape.hidden_size << ")";
    TVM_FFI_ICHECK(final_c.size(0) == shape.batch_size && final_c.size(1) == shape.hidden_size)
        << "final_c must have shape (" << shape.batch_size << ", " << shape.hidden_size << ")";

    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ih_ptr =
            bias_ih.has_value() ? static_cast<const c_type*>(bias_ih.value().data_ptr()) : nullptr;
        const c_type* bias_hh_ptr =
            bias_hh.has_value() ? static_cast<const c_type*>(bias_hh.value().data_ptr()) : nullptr;
        cudaError_t status = recurrent::LstmLayer<c_type>(
            static_cast<c_type*>(output.data_ptr()), static_cast<c_type*>(cells.data_ptr()),
            static_cast<c_type*>(final_h.data_ptr()), static_cast<c_type*>(final_c.data_ptr()),
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(initial_h.data_ptr()),
            static_cast<const c_type*>(initial_c.data_ptr()),
            static_cast<const c_type*>(weight_ih.data_ptr()),
            static_cast<const c_type*>(weight_hh.data_ptr()), bias_ih_ptr, bias_hh_ptr,
            shape.sequence_length, shape.batch_size, shape.input_size, shape.hidden_size,
            shape.input_time_stride, shape.input_batch_stride, shape.output_time_stride,
            shape.output_batch_stride, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "LSTM kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void rnn_layer(TensorView output, TensorView final_h, TensorView input, TensorView initial_h,
               TensorView weight_ih, TensorView weight_hh, Optional bias_ih, Optional bias_hh,
               int64_t activation, bool batch_first) {
    TVM_FFI_ICHECK(activation == 0 || activation == 1)
        << "RNN activation must be 0 (tanh) or 1 (relu), got " << activation;
    const RecurrentShape shape = CheckCommon(output, final_h, input, initial_h, weight_ih,
                                             weight_hh, bias_ih, bias_hh, 1, batch_first);
    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ih_ptr =
            bias_ih.has_value() ? static_cast<const c_type*>(bias_ih.value().data_ptr()) : nullptr;
        const c_type* bias_hh_ptr =
            bias_hh.has_value() ? static_cast<const c_type*>(bias_hh.value().data_ptr()) : nullptr;
        cudaError_t status = recurrent::RnnLayer<c_type>(
            static_cast<c_type*>(output.data_ptr()), static_cast<c_type*>(final_h.data_ptr()),
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(initial_h.data_ptr()),
            static_cast<const c_type*>(weight_ih.data_ptr()),
            static_cast<const c_type*>(weight_hh.data_ptr()), bias_ih_ptr, bias_hh_ptr,
            shape.sequence_length, shape.batch_size, shape.input_size, shape.hidden_size,
            shape.input_time_stride, shape.input_batch_stride, shape.output_time_stride,
            shape.output_batch_stride, static_cast<recurrent::RnnActivation>(activation), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "RNN kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void lstm_gemm_layer(TensorView output, TensorView final_h, TensorView final_c, TensorView cells,
                     TensorView workspace, TensorView input_gates, TensorView initial_h,
                     TensorView initial_c, TensorView weight_hh, Optional bias_hh,
                     bool input_batch_first, int64_t tactic, int64_t split_k_slices) {
#if defined(OASR_TARGET_SM) && OASR_TARGET_SM < 80
    TVM_FFI_ICHECK(false) << "tensor-core recurrent tactics require SM80 or newer";
#else
    CheckRecurrentTactic(tactic, split_k_slices);
    TVM_FFI_ICHECK(!bias_hh.has_value())
        << "tensor-core LSTM expects bias_ih + bias_hh in the packed input projection";
    const GemmRecurrentShape shape =
        CheckGemmCommon(output, final_h, workspace, input_gates, initial_h, weight_hh, bias_hh, 4,
                        input_batch_first);
    CheckTensorLike(cells, input_gates, "cells");
    CheckTensorLike(final_c, input_gates, "final_c");
    CheckTensorLike(initial_c, input_gates, "initial_c");
    CHECK_DIM(3, cells);
    CHECK_DIM(2, final_c);
    CHECK_DIM(2, initial_c);
    for (int dim = 0; dim < 3; ++dim) {
        TVM_FFI_ICHECK(cells.size(dim) == output.size(dim))
            << "cells must have the same shape as output";
    }
    TVM_FFI_ICHECK(initial_c.size(0) == shape.batch_size && initial_c.size(1) == shape.hidden_size)
        << "initial_c shape must match initial_h";
    TVM_FFI_ICHECK(final_c.size(0) == shape.batch_size && final_c.size(1) == shape.hidden_size)
        << "final_c shape must match final_h";

    cudaStream_t stream = get_stream(input_gates.device());
    DISPATCH_DLPACK_HALF_DTYPE(input_gates.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;
        for (int timestep = 0; timestep < shape.sequence_length; ++timestep) {
            c_type* output_t =
                static_cast<c_type*>(output.data_ptr()) +
                static_cast<int64_t>(timestep) * shape.batch_size * shape.hidden_size;
            c_type* cell_t = static_cast<c_type*>(cells.data_ptr()) +
                             static_cast<int64_t>(timestep) * shape.batch_size * shape.hidden_size;
            const c_type* input_t = static_cast<const c_type*>(input_gates.data_ptr()) +
                                    static_cast<int64_t>(timestep) * shape.input_time_stride;
            const c_type* previous_h =
                timestep == 0
                    ? static_cast<const c_type*>(initial_h.data_ptr())
                    : static_cast<const c_type*>(output.data_ptr()) +
                          static_cast<int64_t>(timestep - 1) * shape.batch_size * shape.hidden_size;
            const c_type* previous_c =
                timestep == 0
                    ? static_cast<const c_type*>(initial_c.data_ptr())
                    : static_cast<const c_type*>(cells.data_ptr()) +
                          static_cast<int64_t>(timestep - 1) * shape.batch_size * shape.hidden_size;
            c_type* final_h_ptr = timestep + 1 == shape.sequence_length
                                      ? static_cast<c_type*>(final_h.data_ptr())
                                      : nullptr;
            c_type* final_c_ptr = timestep + 1 == shape.sequence_length
                                      ? static_cast<c_type*>(final_c.data_ptr())
                                      : nullptr;
            gemm::GemmStatus fused_status = gemm::GemmStatus::NOT_SUPPORTED;
            cudaError_t materialized_status = cudaSuccess;
            const auto selected = static_cast<RecurrentGemmTactic>(tactic);
            if (selected == RecurrentGemmTactic::FUSED_16X64) {
                fused_status = RunLstmFused<recurrent::RecurrentConfig16x64, CutlassType>(
                    reinterpret_cast<CutlassType*>(output_t),
                    reinterpret_cast<CutlassType*>(cell_t),
                    reinterpret_cast<CutlassType*>(final_h_ptr),
                    reinterpret_cast<CutlassType*>(final_c_ptr),
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t),
                    reinterpret_cast<const CutlassType*>(previous_c),
                    reinterpret_cast<CutlassType*>(workspace.data_ptr()), shape.batch_size,
                    shape.hidden_size, shape.input_batch_stride, shape.hidden_size, stream);
            } else if (selected == RecurrentGemmTactic::FUSED_32X64) {
                fused_status = RunLstmFused<recurrent::RecurrentConfig32x64, CutlassType>(
                    reinterpret_cast<CutlassType*>(output_t),
                    reinterpret_cast<CutlassType*>(cell_t),
                    reinterpret_cast<CutlassType*>(final_h_ptr),
                    reinterpret_cast<CutlassType*>(final_c_ptr),
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t),
                    reinterpret_cast<const CutlassType*>(previous_c),
                    reinterpret_cast<CutlassType*>(workspace.data_ptr()), shape.batch_size,
                    shape.hidden_size, shape.input_batch_stride, shape.hidden_size, stream);
            } else if (selected == RecurrentGemmTactic::FUSED_64X64) {
                fused_status = RunLstmFused<recurrent::RecurrentConfig64x64, CutlassType>(
                    reinterpret_cast<CutlassType*>(output_t),
                    reinterpret_cast<CutlassType*>(cell_t),
                    reinterpret_cast<CutlassType*>(final_h_ptr),
                    reinterpret_cast<CutlassType*>(final_c_ptr),
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t),
                    reinterpret_cast<const CutlassType*>(previous_c),
                    reinterpret_cast<CutlassType*>(workspace.data_ptr()), shape.batch_size,
                    shape.hidden_size, shape.input_batch_stride, shape.hidden_size, stream);
            } else if (selected == RecurrentGemmTactic::SERIAL_SPLIT_K) {
                fused_status = RunLstmFused<recurrent::RecurrentConfig16x64, CutlassType>(
                    reinterpret_cast<CutlassType*>(output_t),
                    reinterpret_cast<CutlassType*>(cell_t),
                    reinterpret_cast<CutlassType*>(final_h_ptr),
                    reinterpret_cast<CutlassType*>(final_c_ptr),
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t),
                    reinterpret_cast<const CutlassType*>(previous_c),
                    reinterpret_cast<CutlassType*>(workspace.data_ptr()), shape.batch_size,
                    shape.hidden_size, shape.input_batch_stride, shape.hidden_size, stream,
                    static_cast<int>(split_k_slices));
            } else if (selected == RecurrentGemmTactic::STREAM_K) {
                materialized_status = RunLstmMaterialized<recurrent::RecurrentConfig32x64, true,
                                                          false, CutlassType, c_type>(
                    output_t, cell_t, final_h_ptr, final_c_ptr,
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t), previous_c,
                    static_cast<c_type*>(workspace.data_ptr()), shape.batch_size, shape.hidden_size,
                    shape.input_batch_stride, shape.hidden_size, stream, 1);
            } else {
                materialized_status = RunLstmMaterialized<recurrent::RecurrentConfig32x64, false,
                                                          true, CutlassType, c_type>(
                    output_t, cell_t, final_h_ptr, final_c_ptr,
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t), previous_c,
                    static_cast<c_type*>(workspace.data_ptr()), shape.batch_size, shape.hidden_size,
                    shape.input_batch_stride, shape.hidden_size, stream,
                    static_cast<int>(split_k_slices));
            }
            if (selected == RecurrentGemmTactic::STREAM_K ||
                selected == RecurrentGemmTactic::PARALLEL_SPLIT_K) {
                TVM_FFI_ICHECK(materialized_status == cudaSuccess)
                    << "LSTM decomposed GEMM/finalizer failed: "
                    << cudaGetErrorString(materialized_status);
            } else {
                TVM_FFI_ICHECK(fused_status == gemm::GemmStatus::SUCCESS)
                    << "fused LSTM recurrent GEMM failed";
            }
        }
        return true;
    });
#endif
}

void rnn_gemm_layer(TensorView output, TensorView final_h, TensorView workspace,
                    TensorView input_gates, TensorView initial_h, TensorView weight_hh,
                    Optional bias_hh, int64_t activation, bool input_batch_first, int64_t tactic,
                    int64_t split_k_slices) {
#if defined(OASR_TARGET_SM) && OASR_TARGET_SM < 80
    TVM_FFI_ICHECK(false) << "tensor-core recurrent tactics require SM80 or newer";
#else
    CheckRecurrentTactic(tactic, split_k_slices);
    TVM_FFI_ICHECK(tactic != static_cast<int64_t>(RecurrentGemmTactic::SERIAL_SPLIT_K))
        << "serial split-K cannot safely apply an RNN activation to intermediate partitions";
    TVM_FFI_ICHECK(!bias_hh.has_value())
        << "tensor-core RNN expects bias_ih + bias_hh in the input projection";
    TVM_FFI_ICHECK(activation == 0 || activation == 1)
        << "RNN activation must be 0 (tanh) or 1 (relu), got " << activation;
    const GemmRecurrentShape shape =
        CheckGemmCommon(output, final_h, workspace, input_gates, initial_h, weight_hh, bias_hh, 1,
                        input_batch_first);
    cudaStream_t stream = get_stream(input_gates.device());
    DISPATCH_DLPACK_HALF_DTYPE(input_gates.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;
        for (int timestep = 0; timestep < shape.sequence_length; ++timestep) {
            c_type* output_t =
                static_cast<c_type*>(output.data_ptr()) +
                static_cast<int64_t>(timestep) * shape.batch_size * shape.hidden_size;
            const c_type* input_t = static_cast<const c_type*>(input_gates.data_ptr()) +
                                    static_cast<int64_t>(timestep) * shape.input_time_stride;
            const c_type* previous_h =
                timestep == 0
                    ? static_cast<const c_type*>(initial_h.data_ptr())
                    : static_cast<const c_type*>(output.data_ptr()) +
                          static_cast<int64_t>(timestep - 1) * shape.batch_size * shape.hidden_size;
            const auto selected = static_cast<RecurrentGemmTactic>(tactic);
            gemm::GemmStatus status;
            if (activation == static_cast<int64_t>(recurrent::RnnActivation::RELU)) {
                status = RunRnnTactic<CutlassType, recurrent::RnnActivation::RELU>(
                    selected, reinterpret_cast<CutlassType*>(output_t),
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t), shape.batch_size,
                    shape.hidden_size, shape.input_batch_stride, stream,
                    static_cast<int>(split_k_slices));
            } else {
                status = RunRnnTactic<CutlassType, recurrent::RnnActivation::TANH>(
                    selected, reinterpret_cast<CutlassType*>(output_t),
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t), shape.batch_size,
                    shape.hidden_size, shape.input_batch_stride, stream,
                    static_cast<int>(split_k_slices));
            }
            TVM_FFI_ICHECK(status == gemm::GemmStatus::SUCCESS)
                << "fused RNN recurrent GEMM failed";
        }
        const c_type* last_output =
            static_cast<const c_type*>(output.data_ptr()) +
            static_cast<int64_t>(shape.sequence_length - 1) * shape.batch_size * shape.hidden_size;
        TVM_FFI_ICHECK(cudaMemcpyAsync(final_h.data_ptr(), last_output,
                                       static_cast<size_t>(shape.batch_size) * shape.hidden_size *
                                           sizeof(c_type),
                                       cudaMemcpyDeviceToDevice, stream) == cudaSuccess)
            << "RNN final-state copy failed";
        return true;
    });
#endif
}
