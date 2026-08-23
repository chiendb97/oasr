// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for fused recurrent kernels.

#include <cstdint>
#include <limits>
#include <oasr/gemm/gemm.cuh>
#include <oasr/recurrent/recurrent.cuh>
#include <oasr/recurrent/recurrent_cutlass.cuh>

// The CUTLASS 3.x recurrent compositions exist for the two targets whose
// OpClassTensorOp builders accept FP16/BF16.  SM120's does not (F8/F6/F4 only),
// so GeForce Blackwell keeps the 2.x tactics and nothing else changes.
#if defined(OASR_TARGET_SM) && (OASR_TARGET_SM == 90 || OASR_TARGET_SM == 100)
    #define OASR_RECURRENT_HAS_TMA 1
    #include <oasr/recurrent/recurrent_cutlass_sm90.cuh>
#else
    #define OASR_RECURRENT_HAS_TMA 0
#endif

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

// Only cell[t-1] is ever read, so the cell history is a two-slice ring (one
// slice for a single-timestep sequence) rather than the whole sequence.  It is
// always (slices, batch, hidden) contiguous, whatever layout the input uses,
// which is what lets the kernels address the ring, the initial cell and the
// final cell with a single offset.
int CellRing(int sequence_length) {
    return sequence_length > 1 ? 2 : 1;
}

// Validates the LSTM cell buffers shared by both paths: the ring plus the
// initial and final cell, all (batch, hidden) contiguous slices.
void CheckCellBuffers(const TensorView& cells, const TensorView& initial_c,
                      const TensorView& final_c, const TensorView& reference, int sequence_length,
                      int batch_size, int hidden_size) {
    CheckTensorLike(cells, reference, "cells");
    CheckTensorLike(initial_c, reference, "initial_c");
    CheckTensorLike(final_c, reference, "final_c");
    CHECK_DIM(3, cells);
    CHECK_DIM(2, initial_c);
    CHECK_DIM(2, final_c);
    const int ring = CellRing(sequence_length);
    TVM_FFI_ICHECK(cells.size(0) == ring && cells.size(1) == batch_size &&
                   cells.size(2) == hidden_size)
        << "cells must be the (" << ring << ", " << batch_size << ", " << hidden_size
        << ") cell ring, got (" << cells.size(0) << ", " << cells.size(1) << ", " << cells.size(2)
        << ")";
    TVM_FFI_ICHECK(initial_c.size(0) == batch_size && initial_c.size(1) == hidden_size)
        << "initial_c must have shape (" << batch_size << ", " << hidden_size << ")";
    TVM_FFI_ICHECK(final_c.size(0) == batch_size && final_c.size(1) == hidden_size)
        << "final_c must have shape (" << batch_size << ", " << hidden_size << ")";
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
    // CUTLASS 3.x TMA warp-specialized, SM90 / SM100 only.
    TMA_64 = 6,
    TMA_128 = 7,
};

bool IsTmaTactic(RecurrentGemmTactic tactic) {
    return tactic == RecurrentGemmTactic::TMA_64 || tactic == RecurrentGemmTactic::TMA_128;
}

void CheckRecurrentTactic(int64_t tactic, int64_t split_k_slices) {
    TVM_FFI_ICHECK(tactic >= static_cast<int64_t>(RecurrentGemmTactic::FUSED_16X64) &&
                   tactic <= static_cast<int64_t>(RecurrentGemmTactic::TMA_128))
        << "unknown recurrent GEMM tactic " << tactic;
    TVM_FFI_ICHECK(split_k_slices >= 1 && split_k_slices <= 16)
        << "split_k_slices must be in [1, 16], got " << split_k_slices;
    if (tactic == static_cast<int64_t>(RecurrentGemmTactic::PARALLEL_SPLIT_K) ||
        tactic == static_cast<int64_t>(RecurrentGemmTactic::SERIAL_SPLIT_K)) {
        TVM_FFI_ICHECK(split_k_slices > 1)
            << "a split-K recurrent tactic requires split_k_slices > 1";
    }
    if (IsTmaTactic(static_cast<RecurrentGemmTactic>(tactic))) {
#if OASR_RECURRENT_HAS_TMA
        // The collective mainloop carries its own K pipeline; there is no
        // split-K knob to honour, so refuse rather than ignore one.
        TVM_FFI_ICHECK(split_k_slices == 1)
            << "a TMA warp-specialized recurrent tactic has no split-K; got split_k_slices "
            << split_k_slices;
#else
        TVM_FFI_ICHECK(false)
            << "recurrent tactic " << tactic
            << " is CUTLASS 3.x TMA warp-specialized and exists only for SM90 and SM100 targets";
#endif
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
        case RecurrentGemmTactic::TMA_64:
#if OASR_RECURRENT_HAS_TMA
            return recurrent::RnnStateGemmSm90<recurrent::RecurrentSm90Config64, CutlassType,
                                               Activation>(output, previous_h, weight_hh,
                                                           input_gates, batch_size, hidden_size,
                                                           input_gate_batch_stride, stream);
#else
            return gemm::GemmStatus::NOT_SUPPORTED;
#endif
        case RecurrentGemmTactic::TMA_128:
#if OASR_RECURRENT_HAS_TMA
            return recurrent::RnnStateGemmSm90<recurrent::RecurrentSm90Config128, CutlassType,
                                               Activation>(output, previous_h, weight_hh,
                                                           input_gates, batch_size, hidden_size,
                                                           input_gate_batch_stride, stream);
#else
            return gemm::GemmStatus::NOT_SUPPORTED;
#endif
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

GemmRecurrentShape CheckGemmCommon(TensorView output, TensorView final_h, TensorView input_gates,
                                   TensorView initial_h, TensorView weight_hh, Optional bias_hh,
                                   int gate_count, bool input_batch_first) {
    CHECK_INPUT(input_gates);
    CHECK_CONTIGUOUS_INPUT(input_gates);
    CHECK_DIM(3, input_gates);
    CheckTensorLike(output, input_gates, "output");
    CheckTensorLike(final_h, input_gates, "final_h");
    CheckTensorLike(initial_h, input_gates, "initial_h");
    CheckTensorLike(weight_hh, input_gates, "weight_hh");
    CHECK_DIM(3, output);
    CHECK_DIM(2, final_h);
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
    CheckCellBuffers(cells, initial_c, final_c, input, shape.sequence_length, shape.batch_size,
                     shape.hidden_size);

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
#if defined(OASR_TARGET_SM) && OASR_TARGET_SM < 75
    TVM_FFI_ICHECK(false) << "tensor-core recurrent tactics require SM75 or newer";
#else
    CheckRecurrentTactic(tactic, split_k_slices);
    TVM_FFI_ICHECK(!bias_hh.has_value())
        << "tensor-core LSTM expects bias_ih + bias_hh in the packed input projection";
    const GemmRecurrentShape shape = CheckGemmCommon(output, final_h, input_gates, initial_h,
                                                     weight_hh, bias_hh, 4, input_batch_first);
    CheckCellBuffers(cells, initial_c, final_c, input_gates, shape.sequence_length,
                     shape.batch_size, shape.hidden_size);
    CheckTensorLike(workspace, input_gates, "workspace");
    CHECK_DIM(2, workspace);
    TVM_FFI_ICHECK(workspace.size(0) == shape.batch_size &&
                   workspace.size(1) == 4 * shape.hidden_size)
        << "workspace must have shape (" << shape.batch_size << ", " << 4 * shape.hidden_size
        << ")";

    cudaStream_t stream = get_stream(input_gates.device());
    const int cell_ring = CellRing(shape.sequence_length);
    DISPATCH_DLPACK_HALF_DTYPE(input_gates.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;
        const int64_t state_stride = static_cast<int64_t>(shape.batch_size) * shape.hidden_size;
        for (int timestep = 0; timestep < shape.sequence_length; ++timestep) {
            c_type* output_t = static_cast<c_type*>(output.data_ptr()) +
                               static_cast<int64_t>(timestep) * state_stride;
            c_type* cell_t = static_cast<c_type*>(cells.data_ptr()) +
                             static_cast<int64_t>(timestep % cell_ring) * state_stride;
            const c_type* input_t = static_cast<const c_type*>(input_gates.data_ptr()) +
                                    static_cast<int64_t>(timestep) * shape.input_time_stride;
            const c_type* previous_h = timestep == 0
                                           ? static_cast<const c_type*>(initial_h.data_ptr())
                                           : static_cast<const c_type*>(output.data_ptr()) +
                                                 static_cast<int64_t>(timestep - 1) * state_stride;
            const c_type* previous_c =
                timestep == 0 ? static_cast<const c_type*>(initial_c.data_ptr())
                              : static_cast<const c_type*>(cells.data_ptr()) +
                                    static_cast<int64_t>((timestep - 1) % cell_ring) * state_stride;
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
            } else if (selected == RecurrentGemmTactic::TMA_64) {
    #if OASR_RECURRENT_HAS_TMA
                materialized_status = recurrent::LstmStateGemmSm90<recurrent::RecurrentSm90Config64,
                                                                   CutlassType, c_type>(
                    output_t, cell_t, final_h_ptr, final_c_ptr,
                    reinterpret_cast<const CutlassType*>(previous_h),
                    reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                    reinterpret_cast<const CutlassType*>(input_t), previous_c,
                    static_cast<c_type*>(workspace.data_ptr()), shape.batch_size, shape.hidden_size,
                    shape.input_batch_stride, shape.hidden_size, stream);
    #endif
            } else if (selected == RecurrentGemmTactic::TMA_128) {
    #if OASR_RECURRENT_HAS_TMA
                materialized_status =
                    recurrent::LstmStateGemmSm90<recurrent::RecurrentSm90Config128, CutlassType,
                                                 c_type>(
                        output_t, cell_t, final_h_ptr, final_c_ptr,
                        reinterpret_cast<const CutlassType*>(previous_h),
                        reinterpret_cast<const CutlassType*>(weight_hh.data_ptr()),
                        reinterpret_cast<const CutlassType*>(input_t), previous_c,
                        static_cast<c_type*>(workspace.data_ptr()), shape.batch_size,
                        shape.hidden_size, shape.input_batch_stride, shape.hidden_size, stream);
    #endif
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
                selected == RecurrentGemmTactic::PARALLEL_SPLIT_K || IsTmaTactic(selected)) {
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

void rnn_gemm_layer(TensorView output, TensorView final_h, TensorView input_gates,
                    TensorView initial_h, TensorView weight_hh, Optional bias_hh,
                    int64_t activation, bool input_batch_first, int64_t tactic,
                    int64_t split_k_slices) {
#if defined(OASR_TARGET_SM) && OASR_TARGET_SM < 75
    TVM_FFI_ICHECK(false) << "tensor-core recurrent tactics require SM75 or newer";
#else
    CheckRecurrentTactic(tactic, split_k_slices);
    TVM_FFI_ICHECK(tactic != static_cast<int64_t>(RecurrentGemmTactic::SERIAL_SPLIT_K))
        << "serial split-K cannot safely apply an RNN activation to intermediate partitions";
    TVM_FFI_ICHECK(!bias_hh.has_value())
        << "tensor-core RNN expects bias_ih + bias_hh in the input projection";
    TVM_FFI_ICHECK(activation == 0 || activation == 1)
        << "RNN activation must be 0 (tanh) or 1 (relu), got " << activation;
    const GemmRecurrentShape shape = CheckGemmCommon(output, final_h, input_gates, initial_h,
                                                     weight_hh, bias_hh, 1, input_batch_first);
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

namespace {

// Shape/dtype contract shared by both slot steps.  `state_h` is the (2, slots,
// hidden) ring; `state_slots` is int64 on the same device.
struct SlotStepShape {
    int batch_size;
    int input_size;
    int hidden_size;
    int slot_count;
    int64_t input_batch_stride;
    int64_t output_batch_stride;
};

SlotStepShape CheckSlotStep(TensorView output, TensorView state_h, TensorView input,
                            TensorView state_slots, TensorView read_parity, TensorView weight_ih,
                            TensorView weight_hh, Optional bias_ih, Optional bias_hh,
                            int gate_count) {
    CHECK_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_DIM(2, input);
    CheckTensorLike(output, input, "output");
    CheckTensorLike(state_h, input, "state_h");
    CheckTensorLike(weight_ih, input, "weight_ih");
    CheckTensorLike(weight_hh, input, "weight_hh");
    CHECK_DIM(2, output);
    CHECK_DIM(3, state_h);
    CHECK_DIM(2, weight_ih);
    CHECK_DIM(2, weight_hh);
    CHECK_INPUT(state_slots);
    CHECK_CONTIGUOUS_INPUT(state_slots);
    CHECK_DIM(1, state_slots);
    CHECK_DEVICE(state_slots, input);
    TVM_FFI_ICHECK(state_slots.dtype().code == kDLInt && state_slots.dtype().bits == 64)
        << "state_slots must be int64";
    CHECK_INPUT(read_parity);
    CHECK_CONTIGUOUS_INPUT(read_parity);
    CHECK_DIM(1, read_parity);
    CHECK_DEVICE(read_parity, input);
    TVM_FFI_ICHECK(read_parity.dtype().code == kDLInt && read_parity.dtype().bits == 32)
        << "read_parity must be int32";

    const int64_t batch_size = input.size(0);
    const int64_t input_size = input.size(1);
    const int64_t hidden_size = weight_hh.size(1);
    const int64_t slot_count = state_h.size(1);
    TVM_FFI_ICHECK(batch_size > 0 && input_size > 0 && hidden_size > 0 && slot_count > 0)
        << "slot-step dimensions must be positive, got batch=" << batch_size
        << " input_size=" << input_size << " hidden_size=" << hidden_size
        << " slots=" << slot_count;
    TVM_FFI_ICHECK(batch_size <= 65535 && hidden_size <= 65535)
        << "slot-step shape exceeds CUDA grid limits";
    TVM_FFI_ICHECK(state_h.size(0) == 2)
        << "state_h must be a two-slice ring, got leading dimension " << state_h.size(0);
    TVM_FFI_ICHECK(state_h.size(2) == hidden_size)
        << "state_h last dimension must be " << hidden_size;
    TVM_FFI_ICHECK(state_slots.size(0) == batch_size)
        << "state_slots must have " << batch_size << " entries";
    TVM_FFI_ICHECK(read_parity.size(0) == batch_size)
        << "read_parity must have " << batch_size << " entries";
    TVM_FFI_ICHECK(weight_ih.size(0) == gate_count * hidden_size && weight_ih.size(1) == input_size)
        << "weight_ih must have shape (" << gate_count * hidden_size << ", " << input_size << ")";
    TVM_FFI_ICHECK(weight_hh.size(0) == gate_count * hidden_size)
        << "weight_hh must have shape (" << gate_count * hidden_size << ", " << hidden_size << ")";
    TVM_FFI_ICHECK(output.size(0) == batch_size && output.size(1) == hidden_size)
        << "output must have shape (" << batch_size << ", " << hidden_size << ")";
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
    return {static_cast<int>(batch_size),
            static_cast<int>(input_size),
            static_cast<int>(hidden_size),
            static_cast<int>(slot_count),
            input.stride(0),
            output.stride(0)};
}

}  // namespace

void lstm_slot_step(TensorView output, TensorView state_h, TensorView state_c, TensorView input,
                    TensorView state_slots, TensorView read_parity, TensorView weight_ih,
                    TensorView weight_hh, Optional bias_ih, Optional bias_hh) {
    const SlotStepShape shape = CheckSlotStep(output, state_h, input, state_slots, read_parity,
                                              weight_ih, weight_hh, bias_ih, bias_hh, 4);
    CheckTensorLike(state_c, input, "state_c");
    CHECK_DIM(2, state_c);
    TVM_FFI_ICHECK(state_c.size(0) == shape.slot_count && state_c.size(1) == shape.hidden_size)
        << "state_c must have shape (" << shape.slot_count << ", " << shape.hidden_size << ")";

    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ih_ptr =
            bias_ih.has_value() ? static_cast<const c_type*>(bias_ih.value().data_ptr()) : nullptr;
        const c_type* bias_hh_ptr =
            bias_hh.has_value() ? static_cast<const c_type*>(bias_hh.value().data_ptr()) : nullptr;
        cudaError_t status = recurrent::LstmSlotStep<c_type>(
            static_cast<c_type*>(output.data_ptr()), static_cast<c_type*>(state_h.data_ptr()),
            static_cast<c_type*>(state_c.data_ptr()), static_cast<const c_type*>(input.data_ptr()),
            static_cast<const int64_t*>(state_slots.data_ptr()),
            static_cast<const c_type*>(weight_ih.data_ptr()),
            static_cast<const c_type*>(weight_hh.data_ptr()), bias_ih_ptr, bias_hh_ptr,
            shape.batch_size, shape.input_size, shape.hidden_size, shape.slot_count,
            static_cast<const int32_t*>(read_parity.data_ptr()), shape.input_batch_stride,
            shape.output_batch_stride, stream);
        // cudaErrorInvalidValue is the declared "one unit's weights do not fit in
        // shared memory" signal, and the Python caller routes around it rather
        // than silently producing nothing.
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "LSTM slot step failed: " << cudaGetErrorString(status);
        return true;
    });
}

void rnn_slot_step(TensorView output, TensorView state_h, TensorView input, TensorView state_slots,
                   TensorView read_parity, TensorView weight_ih, TensorView weight_hh,
                   Optional bias_ih, Optional bias_hh, int64_t activation) {
    TVM_FFI_ICHECK(activation == 0 || activation == 1)
        << "RNN activation must be 0 (tanh) or 1 (relu), got " << activation;
    const SlotStepShape shape = CheckSlotStep(output, state_h, input, state_slots, read_parity,
                                              weight_ih, weight_hh, bias_ih, bias_hh, 1);
    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ih_ptr =
            bias_ih.has_value() ? static_cast<const c_type*>(bias_ih.value().data_ptr()) : nullptr;
        const c_type* bias_hh_ptr =
            bias_hh.has_value() ? static_cast<const c_type*>(bias_hh.value().data_ptr()) : nullptr;
        cudaError_t status = recurrent::RnnSlotStep<c_type>(
            static_cast<c_type*>(output.data_ptr()), static_cast<c_type*>(state_h.data_ptr()),
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const int64_t*>(state_slots.data_ptr()),
            static_cast<const c_type*>(weight_ih.data_ptr()),
            static_cast<const c_type*>(weight_hh.data_ptr()), bias_ih_ptr, bias_hh_ptr,
            shape.batch_size, shape.input_size, shape.hidden_size, shape.slot_count,
            static_cast<const int32_t*>(read_parity.data_ptr()), shape.input_batch_stride,
            shape.output_batch_stride, static_cast<recurrent::RnnActivation>(activation), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "RNN slot step failed: " << cudaGetErrorString(status);
        return true;
    });
}

// The gate-major finalizers are the decomposition tactic for a checkpoint-order
// gate layout: the recurrent GEMM leaves its C operand out and the finalizer
// sums the two gate tensors itself.  No launcher routes through them yet, and a
// function template that is never instantiated is never type-checked, so pin
// both served dtypes here — dead code that does not compile is worse than dead
// code.
template cudaError_t oasr::recurrent::LstmGateStep<half>(half*, half*, half*, half*, const half*,
                                                         const half*, const half*, int, int,
                                                         int64_t, int64_t, cudaStream_t);
template cudaError_t oasr::recurrent::LstmGateStep<__nv_bfloat16>(
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, int, int, int64_t, int64_t, cudaStream_t);
template cudaError_t oasr::recurrent::RnnGateStep<half>(half*, half*, const half*, const half*, int,
                                                        int, int64_t,
                                                        oasr::recurrent::RnnActivation,
                                                        cudaStream_t);
template cudaError_t oasr::recurrent::RnnGateStep<__nv_bfloat16>(
    __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int64_t,
    oasr::recurrent::RnnActivation, cudaStream_t);
