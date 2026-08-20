// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Recurrent-specific CUTLASS compositions.  Unlike the general GEMM waist,
// these epilogues consume a matrix-valued C operand (the sequence-wide input
// projection) and write the next recurrent state directly.

#pragma once

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_conversion.h>

#include <cstdint>
#include <oasr/common/math.h>
#include <oasr/common/workspace_cache.h>
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_cutlass_template.h>
#include <oasr/recurrent/recurrent.cuh>

namespace oasr {
namespace recurrent {

#if defined(OASR_TARGET_SM) && OASR_TARGET_SM == 75
constexpr int kRecurrentCutlassArch = 75;
#else
// mma.sync.m16n8k16 remains available on Hopper and Blackwell.  Using the
// CUTLASS 2.x SM80 composition also gives the recurrent epilogue one stable
// output-thread mapping across Ampere through Blackwell.
constexpr int kRecurrentCutlassArch = 80;
#endif

using RecurrentConfig16x64 =
    gemm::CutlassGemmConfig<16, 64, 64, 16, 32, 64, 3, kRecurrentCutlassArch>;
using RecurrentConfig32x64 =
    gemm::CutlassGemmConfig<32, 64, 64, 16, 32, 64, 3, kRecurrentCutlassArch>;
using RecurrentConfig64x64 =
    gemm::CutlassGemmConfig<64, 64, 64, 32, 32, 64, 3, kRecurrentCutlassArch>;

namespace detail {

// CUTLASS's ordinary output functors are elementwise.  An LSTM output cell,
// however, consumes four adjacent gate values.  We pack the gate dimension as
// [hidden, gate], then use the output iterator's own thread map to recover the
// logical coordinate of each 128-bit epilogue access.  This applies the state
// transition after the complete K reduction and stores h/c directly, without
// a second gate read or state-finalizer launch.
template <typename ElementOutput_, typename OutputTileIterator_, typename ThreadblockShape_,
          typename ThreadblockSwizzle_>
class LstmStateEpilogue {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using OutputTileIterator = OutputTileIterator_;
    using ThreadblockShape = ThreadblockShape_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using ThreadMap = typename OutputTileIterator::ThreadMap;

    static constexpr int kCount = 128 / cutlass::sizeof_bits<ElementOutput>::value;
    static constexpr bool kIsHeavy = true;
    static_assert(kCount % 4 == 0, "an epilogue access must contain complete LSTM gates");
    static_assert(kCount == OutputTileIterator::kElementsPerAccess,
                  "LSTM epilogue and output iterator access widths must match");

    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentSource = FragmentOutput;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using FragmentCompute = cutlass::Array<ElementCompute, kCount>;

    struct Params {
        ElementOutput* output;
        ElementOutput* cells;
        ElementOutput* final_h;
        ElementOutput* final_c;
        const ElementOutput* input_gates;
        const ElementOutput* previous_c;
        int batch_size;
        int hidden_size;
        int problem_n;
        int64_t input_gate_batch_stride;
        int64_t previous_cell_batch_stride;
        int swizzle_log_tile;

        CUTLASS_HOST_DEVICE
        Params(ElementOutput* output_ = nullptr, ElementOutput* cells_ = nullptr,
               ElementOutput* final_h_ = nullptr, ElementOutput* final_c_ = nullptr,
               const ElementOutput* input_gates_ = nullptr,
               const ElementOutput* previous_c_ = nullptr, int batch_size_ = 0,
               int hidden_size_ = 0, int64_t input_gate_batch_stride_ = 0,
               int64_t previous_cell_batch_stride_ = 0, int swizzle_log_tile_ = 0)
            : output(output_),
              cells(cells_),
              final_h(final_h_),
              final_c(final_c_),
              input_gates(input_gates_),
              previous_c(previous_c_),
              batch_size(batch_size_),
              hidden_size(hidden_size_),
              problem_n(4 * hidden_size_),
              input_gate_batch_stride(input_gate_batch_stride_),
              previous_cell_batch_stride(previous_cell_batch_stride_),
              swizzle_log_tile(swizzle_log_tile_) {}
    };

private:
    Params params_;
    mutable OutputTileIterator coordinate_iterator_;
    mutable int access_index_;
    mutable bool apply_state_;

    CUTLASS_DEVICE
    static cutlass::MatrixCoord threadblock_offset(const Params& params) {
        ThreadblockSwizzle swizzle;
        const cutlass::gemm::GemmCoord tile = swizzle.get_tile_offset(params.swizzle_log_tile);
        return {tile.m() * ThreadblockShape::kM, tile.n() * ThreadblockShape::kN};
    }

public:
    CUTLASS_DEVICE
    explicit LstmStateEpilogue(const Params& params)
        : params_(params),
          coordinate_iterator_(typename OutputTileIterator::Params(
                                   cutlass::layout::RowMajor(params.input_gate_batch_stride)),
                               const_cast<ElementOutput*>(params.input_gates),
                               cutlass::MatrixCoord(params.batch_size, params.problem_n),
                               static_cast<int>(threadIdx.x), threadblock_offset(params)),
          access_index_(0),
          apply_state_(true) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return true; }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int k_partition_count) {
        // Serial split-K invokes the output functor once per partition.  Linear
        // partials are written every time, but the nonlinear state transition
        // must run exactly once, after the final partition has accumulated.
        apply_state_ = (k_partition + 1 == k_partition_count);
    }

    CUTLASS_DEVICE
    FragmentOutput operator()(const FragmentAccumulator& accumulator,
                              const FragmentSource& source) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount> convert_accum;
        cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount> convert_source;
        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount> convert_output;
        FragmentCompute values = convert_accum(accumulator);
        const FragmentCompute source_values = convert_source(source);

#pragma unroll
        for (int element = 0; element < kCount; ++element)
            values[element] += source_values[element];

        const cutlass::MatrixCoord access_offset = ThreadMap::iteration_offset(access_index_);
        const cutlass::MatrixCoord access_coord =
            coordinate_iterator_.thread_start() + access_offset;

        if (apply_state_ && access_coord.row() < params_.batch_size) {
#pragma unroll
            for (int cell_index = 0; cell_index < kCount / 4; ++cell_index) {
                const int gate_column = access_coord.column() + 4 * cell_index;
                const int hidden = gate_column / 4;
                if (hidden < params_.hidden_size) {
                    const int gate = 4 * cell_index;
                    const float input_gate = fastSigmoid(values[gate]);
                    const float forget_gate = fastSigmoid(values[gate + 1]);
                    const float cell_gate = tanhf(values[gate + 2]);
                    const float output_gate = fastSigmoid(values[gate + 3]);
                    const int64_t previous_offset = static_cast<int64_t>(access_coord.row()) *
                                                        params_.previous_cell_batch_stride +
                                                    hidden;
                    const int64_t output_offset =
                        static_cast<int64_t>(access_coord.row()) * params_.hidden_size + hidden;
                    const float cell =
                        forget_gate * static_cast<float>(params_.previous_c[previous_offset]) +
                        input_gate * cell_gate;
                    cutlass::NumericConverter<ElementOutput, float> convert_scalar;
                    const ElementOutput cell_value = convert_scalar(cell);
                    const ElementOutput hidden_value = convert_scalar(output_gate * tanhf(cell));
                    params_.cells[output_offset] = cell_value;
                    params_.output[output_offset] = hidden_value;
                    if (params_.final_h != nullptr) {
                        params_.final_h[output_offset] = hidden_value;
                        params_.final_c[output_offset] = cell_value;
                    }
                }
            }
        }

        ++access_index_;
        constexpr int kAccessesPerFragment =
            OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;
        if (access_index_ == kAccessesPerFragment) {
            access_index_ = 0;
            ++coordinate_iterator_;
        }
        return convert_output(values);
    }

    CUTLASS_DEVICE
    FragmentOutput operator()(const FragmentAccumulator& accumulator) const {
        FragmentSource source;
        source.clear();
        return (*this)(accumulator, source);
    }
};

template <typename Config, typename Element>
struct LstmStateGemm {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;
    using Accumulator = float;
    using MmaOp = cutlass::arch::OpClassTensorOp;
    using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using ThreadblockShape = typename Config::ThreadblockShape;
    using WarpShape = typename Config::WarpShape;
    using InstructionShape = typename Config::InstructionShape;
    using Arch = typename Config::SmArch;
    static constexpr int kAlignment = 128 / cutlass::sizeof_bits<Element>::value;

    using DummyOutputOp =
        cutlass::epilogue::thread::LinearCombination<Element, kAlignment, Accumulator, Accumulator>;
    using DummyGemm =
        cutlass::gemm::device::Gemm<Element, LayoutA, Element, LayoutB, Element, LayoutCD,
                                    Accumulator, MmaOp, Arch, ThreadblockShape, WarpShape,
                                    InstructionShape, DummyOutputOp, Swizzle, Config::Stages,
                                    kAlignment, kAlignment, true>;
    using OutputTileIterator = typename DummyGemm::GemmKernel::Epilogue::OutputTileIterator;
    using OutputOp = LstmStateEpilogue<Element, OutputTileIterator, ThreadblockShape, Swizzle>;
    using Gemm = cutlass::gemm::device::Gemm<Element, LayoutA, Element, LayoutB, Element, LayoutCD,
                                             Accumulator, MmaOp, Arch, ThreadblockShape, WarpShape,
                                             InstructionShape, OutputOp, Swizzle, Config::Stages,
                                             kAlignment, kAlignment, true>;

    static gemm::GemmStatus run(Element* output, Element* cells, Element* final_h, Element* final_c,
                                const Element* previous_h, const Element* weight_hh,
                                const Element* input_gates, const Element* previous_c,
                                Element* split_k_workspace, int batch_size, int hidden_size,
                                int64_t input_gate_batch_stride, int64_t previous_cell_batch_stride,
                                cudaStream_t stream, int split_k_slices = 1) {
        const cutlass::gemm::GemmCoord problem(batch_size, 4 * hidden_size, hidden_size);
        Swizzle swizzle;
        const cutlass::gemm::GemmCoord tiled_shape = swizzle.get_tiled_shape(
            problem, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            split_k_slices);
        typename OutputOp::Params output_params(
            output, cells, final_h, final_c, input_gates, previous_c, batch_size, hidden_size,
            input_gate_batch_stride, previous_cell_batch_stride, swizzle.get_log_tile(tiled_shape));

        // CUTLASS epilogues always issue a D store. Keep that linear output in
        // the caller's reusable tile so the runner is idempotent during
        // autotuning. The custom epilogue writes h/c directly, so the tile is
        // never read by a state-finalizer kernel. For serial split-K it retains
        // the partial between K partitions.
        Element* destination = split_k_workspace;
        typename Gemm::Arguments args(problem, {previous_h, hidden_size}, {weight_hh, hidden_size},
                                      {input_gates, input_gate_batch_stride},
                                      {destination, 4 * hidden_size}, output_params,
                                      split_k_slices);
        Gemm gemm_op;
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess)
            return gemm::GemmStatus::NOT_SUPPORTED;

        size_t workspace_size = 0;
        if (split_k_slices > 1) {
            workspace_size = Gemm::get_workspace_size(args);
            void* cached =
                getCachedWorkspace(workspace_size, stream, WorkspacePool::kZeroedSemaphore);
            if (cached != nullptr) {
                return gemm::runSerialSplitKPreZeroed<Gemm, ThreadblockShape>(
                    args, static_cast<int*>(cached), stream);
            }
        }
        GraphSafeWorkspace fallback(workspace_size, stream);
        void* workspace = fallback.get();
        if (gemm_op.initialize(args, workspace, stream) != cutlass::Status::kSuccess)
            return gemm::GemmStatus::INTERNAL_ERROR;
        if (gemm_op(stream) != cutlass::Status::kSuccess)
            return gemm::GemmStatus::CUTLASS_ERROR;
        return gemm::GemmStatus::SUCCESS;
    }
};

}  // namespace detail

template <typename Config, typename Element>
gemm::GemmStatus LstmStateGemm(Element* output, Element* cells, Element* final_h, Element* final_c,
                               const Element* previous_h, const Element* weight_hh,
                               const Element* input_gates, const Element* previous_c,
                               Element* split_k_workspace, int batch_size, int hidden_size,
                               int64_t input_gate_batch_stride, int64_t previous_cell_batch_stride,
                               cudaStream_t stream, int split_k_slices = 1) {
    return detail::LstmStateGemm<Config, Element>::run(
        output, cells, final_h, final_c, previous_h, weight_hh, input_gates, previous_c,
        split_k_workspace, batch_size, hidden_size, input_gate_batch_stride,
        previous_cell_batch_stride, stream, split_k_slices);
}

template <typename Config, typename Element, RnnActivation Activation, bool kStreamK = false,
          bool kParallelSplitK = false>
gemm::GemmStatus RnnStateGemm(Element* output, const Element* previous_h, const Element* weight_hh,
                              const Element* input_gates, int batch_size, int hidden_size,
                              int64_t input_gate_batch_stride, cudaStream_t stream,
                              int split_k_slices = 1) {
    constexpr ActivationType kActivation =
        Activation == RnnActivation::RELU ? ActivationType::RELU : ActivationType::TANH;
    return gemm::CutlassGemmKernel<Config, Element, Element, Element, kActivation, kStreamK,
                                   kParallelSplitK>::run(previous_h, weight_hh, input_gates, output,
                                                         batch_size, hidden_size, hidden_size,
                                                         hidden_size, hidden_size,
                                                         input_gate_batch_stride, 1.0f, stream,
                                                         split_k_slices, false, hidden_size);
}

}  // namespace recurrent
}  // namespace oasr
