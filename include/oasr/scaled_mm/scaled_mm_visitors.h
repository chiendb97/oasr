// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Epilogue visitors the int8 scaled GEMM needs beyond CUTLASS's own.
//
// ``OasrVisitorColOrScalarBroadcast`` is CUTLASS's ``VisitorColBroadcast`` (one
// value per output row) with a runtime switch to a single value for every row:
// the activation scale of a W8A8 GEMM is per-token when the activation was
// quantized dynamically and per-tensor when it carries a calibrated static
// scale.  A runtime flag rather than a second template keeps the per-tensor
// case from doubling every instantiation.  The scalar is read from device
// memory, never passed by value, so a captured CUDA graph sees a scale written
// after capture (vLLM's ``VisitorColOrScalarBroadcast`` makes the same choice).
//
// Defined in CUTLASS's own namespace, as its visitors are, for the cute names
// they use unqualified.

#pragma once

#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

namespace cutlass::epilogue::threadblock {

using namespace cute;

template <class ThreadMap, class Element, class StrideMNL = Stride<_1, _0, _0>>
struct OasrVisitorColOrScalarBroadcast {
    struct Arguments {
        Element const* ptr_col = nullptr;
        // false: ``ptr_col`` points at one value that every row takes.
        bool col_broadcast = true;
        StrideMNL dCol = {};
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                    Arguments const& args, void* workspace) {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
        return 0;
    }

    struct SharedStorage {};

    CUTLASS_HOST_DEVICE
    OasrVisitorColOrScalarBroadcast() {}

    CUTLASS_HOST_DEVICE
    OasrVisitorColOrScalarBroadcast(Params const& params, SharedStorage const& shared_storage)
        : params_ptr(&params) {}

    Params const* params_ptr;

    template <class GTensor, class RTensor, class CTensor, class ProblemShape>
    struct Callbacks : EmptyCallbacks {
        CUTLASS_DEVICE
        Callbacks(GTensor&& tC_gCol, RTensor&& tC_rCol, CTensor&& tC_cCol,
                  ProblemShape problem_shape, Params const* params_ptr)
            : tC_gCol(cute::forward<GTensor>(tC_gCol)),
              tC_rCol(cute::forward<RTensor>(tC_rCol)),
              tC_cCol(cute::forward<CTensor>(tC_cCol)),
              m(get<0>(problem_shape)),
              params_ptr(params_ptr) {}

        GTensor tC_gCol;
        RTensor tC_rCol;
        CTensor tC_cCol;
        int m;
        Params const* params_ptr;

        CUTLASS_DEVICE void begin_epilogue() {
            if (!params_ptr->col_broadcast) {
                fill(tC_rCol, *(params_ptr->ptr_col));
                return;
            }
            clear(tC_rCol);
            Tensor tC_pCol =
                cute::lazy::transform(tC_cCol, [&](auto const& c) { return get<0>(c) < m; });
            copy_if(tC_pCol, tC_gCol, tC_rCol);
        }

        template <class ElementAccumulator, int FragmentSize>
        CUTLASS_DEVICE auto visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
                                  Array<ElementAccumulator, FragmentSize> const& frg_acc) {
            Array<Element, FragmentSize> frg_col;
            frg_col.fill(tC_rCol(row_idx, iter_idx));
            return frg_col;
        }
    };

    template <class ProblemShape>
    CUTLASS_DEVICE auto get_callbacks(gemm::GemmCoord threadblock_tile_offset, int thread_idx,
                                      ProblemShape problem_shape) {
        Tensor mCol = make_tensor(make_gmem_ptr(params_ptr->ptr_col), problem_shape,
                                  params_ptr->dCol);

        // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP,
        // ITERATION_CLUSTER
        Tensor tC_gCol = group_modes<1, 4>(
            ThreadMap::partition(mCol, thread_idx, threadblock_tile_offset)(_0{}, _0{}, _, _, _, _));
        Tensor tC_rCol = make_tensor_like(tC_gCol);

        Tensor cCol = make_identity_tensor(mCol.shape());
        Tensor tC_cCol = group_modes<1, 4>(
            ThreadMap::partition(cCol, thread_idx, threadblock_tile_offset)(_0{}, _0{}, _, _, _, _));

        return Callbacks<decltype(tC_gCol), decltype(tC_rCol), decltype(tC_cCol), ProblemShape>(
            cute::move(tC_gCol), cute::move(tC_rCol), cute::move(tC_cCol), problem_shape,
            params_ptr);
    }
};

}  // namespace cutlass::epilogue::threadblock
