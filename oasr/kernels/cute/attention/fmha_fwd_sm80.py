# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SM80 fused multi-head attention forward kernel (CuteDSL).

Notes
-----
Structurally modelled on FlashAttention's ``flash_attn/cute/flash_fwd.py``
(SM80 path) but the implementation is OASR-native. Helpers under
``oasr.kernels.cute`` (``softmax``, ``mask``, etc.) carry the reusable
pieces; this file is just the prologue + mainloop + epilogue that composes
them. Paged-KV gather lives on the kernel class itself as
``_paged_load_kv_tile`` -- treating it as a method lets cute.jit capture
``self._block_size`` / ``self._head_dim_padded`` as class-level constexprs
across runtime if-regions, which a free-standing helper struct cannot do.

Key differences from the previous OASR SM80 kernel:

* ``num_stages = 2`` for **both** K and V cp.async (dual ``sK0/sK1`` *and*
  dual ``sV0/sV1``). The previous kernel pipelined K only and reloaded V
  into a single buffer each iter, which serialized V loads with the QK
  gemm. Smem grows from 32 KB to 40 KB at ``M=N=64, D=64`` -- still well
  under SM80's 163 KB and SM120's 99 KB caps.
* :class:`AttentionMask` carries the per-stream length, causal, and
  sliding-window mask in one constexpr-flagged loop instead of an inline
  branch tree. Causal / window-size knobs are wired through ``FmhaBase``
  and default off, preserving current behavior.
* Online softmax now lives in :class:`Softmax` (state + helpers). The
  empty-row clamp (``row_max == -inf`` => 0) carries over.
* Paged KV uses ``_paged_load_kv_tile`` on this class. Same
  block-aligned-``n_block`` constraint as before
  (``N_BLOCK % block_size == 0``).
* Varlen is scaffolded (constexpr ``self._varlen`` in feasibility), but the
  kernel-side packed-tensor indexing lands in a follow-up phase along with
  the Python wrapper rework.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import cuda.bindings.driver as cuda
from cutlass.cute.nvgpu import warp

from .base import FmhaBase
from ..copy_utils import (
    UNIVERSAL_COPY_BITS,
    async_copy_elements,
    make_cp_async_atom,
    make_ldmatrix_x4_atom,
    make_qkv_tiled_copy,
    make_universal_copy_atom,
)
from ..layout_utils import (
    make_smem_layout,
    make_smem_swizzle_atom,
    make_v_transpose_view,
)
from ..ampere_helpers import gemm_rs, gemm_with_smem_prefetch
from ..mask import AttentionMask
from ..named_barrier import make_cta_sync_barrier
from ..pack_gqa import PackGQA
from ..softmax import Softmax
from ..utils import LOG2_E, make_acc_mn_view


class FmhaForwardSm80(FmhaBase):
    """SM80 / SM120 FMHA forward.

    Tile choices match the previous kernel's Conformer-sized defaults:

    * ``M_BLOCK = 64`` -- ``(M_BLOCK * 2) % num_threads == 0`` with
      num_threads=128. T_q=8 still uses one m-block per ``(B, H)`` and
      predicates 56 unused rows.
    * ``N_BLOCK = 64`` -- fits SM120's 99 KB smem cap with 40 KB total
      smem (sQ + 2*sK + 2*sV at D=64).
    * ``num_threads = 128`` (4 warps).
    """

    arch = 80

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: Any,
        num_heads: int,
        num_kv_heads: int,
        has_bias: bool = False,
        paged: bool = False,
        block_size: int = 0,
        causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        varlen: bool = False,
        m_block_size: int = 64,
        n_block_size: int = 64,
        num_threads: int = 128,
        q_in_regs: bool = False,
    ):
        super().__init__(
            head_dim=head_dim,
            dtype=dtype,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            has_bias=has_bias,
            paged=paged,
            block_size=block_size,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            varlen=varlen,
        )
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        # Pad head_dim up to a multiple of 32 so m16n8k16 MMA k-stride works.
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        # Tier 2: if True, ldmatrix Q to rmem once in the prologue and skip
        # the per-iter Q ldmatrix in ``_compute_one_n_block``. Saves K_tiles
        # ldmatrix instructions per n-block iter, which adds up across many
        # iters at large T_q / T_k.
        self._q_in_regs = q_in_regs

        if paged:
            if n_block_size % block_size != 0:
                raise ValueError(
                    f"{type(self).__name__}: paged kernel requires "
                    f"n_block_size ({n_block_size}) % block_size ({block_size}) == 0"
                )
            self._blocks_per_n_tile = n_block_size // block_size
        else:
            self._blocks_per_n_tile = 1

        self.cta_sync_barrier = make_cta_sync_barrier(num_threads)

    # ------------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------------
    _smem_arch_str = "sm_80"

    @classmethod
    def _smem_capacity_in_bytes(cls) -> int:
        return cutlass_utils.get_smem_capacity_in_bytes(cls._smem_arch_str)

    @classmethod
    def can_implement(
        cls,
        dtype,
        head_dim: int,
        m_block_size: int = 64,
        n_block_size: int = 64,
        num_threads: int = 128,
        has_bias: bool = False,
        paged: bool = False,
        block_size: int = 0,
        causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        varlen: bool = False,
        **_kwargs,
    ) -> bool:
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (m_block_size * 2) % num_threads != 0:
            return False
        if paged:
            if block_size <= 0:
                return False
            if n_block_size % block_size != 0:
                return False
            # Paged path skips per-element head-dim predication; require
            # head_dim to fall on the MMA k-stride boundary.
            if head_dim % 32 != 0:
                return False
            if varlen:
                return False
        if causal and window_size_right == -1:
            window_size_right = 0  # causal implies right-window 0
        # Smem budget: sQ + 2*sK + 2*sV. Bias (when has_bias) is gmem-direct,
        # not staged through smem.
        del has_bias  # not part of the smem budget
        del causal
        del window_size_left
        del window_size_right
        del varlen
        smem_bytes = (
            m_block_size * head_dim
            + n_block_size * head_dim * 4   # 2*sK + 2*sV
        ) * 2                                # fp16/bf16 = 2B
        if smem_bytes > cls._smem_capacity_in_bytes():
            return False
        return True

    # ------------------------------------------------------------------------
    # Host launcher
    # ------------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mBias: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mBlockTable: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type
                 == mV.element_type == mO.element_type)
        ):
            raise TypeError("Q/K/V/O must share dtype")
        if cutlass.const_expr(
            not (mQ.element_type == cutlass.Float16
                 or mQ.element_type == cutlass.BFloat16)
        ):
            raise TypeError("Only fp16 and bf16 are supported")
        self._dtype = mQ.element_type

        # ---- Smem layouts ----------------------------------------------------
        sQ_atom, _smem_k_block_size = make_smem_swizzle_atom(
            self._dtype, self._head_dim_padded,
        )
        sQ_layout = make_smem_layout(
            sQ_atom, self._m_block_size, self._head_dim_padded,
        )
        sKV_layout = make_smem_layout(
            sQ_atom, self._n_block_size, self._head_dim_padded,
        )
        sO_layout = sQ_layout

        # 2-stage K + 2-stage V smem ring (dual ping-pong on both axes).
        # No double-buffer alias between sQ and sV in this revision; that's
        # a follow-on Q_in_regs optimization.
        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK0: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sK1: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV0: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV1: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]

        # ---- Copy atoms ------------------------------------------------------
        async_elems = async_copy_elements(self._dtype)
        atom_async = make_cp_async_atom(self._dtype)
        atom_universal = make_universal_copy_atom(self._dtype)
        smem_k_block_size = sQ_atom.outer.shape[1]
        gmem_tiled_copy_QKV = make_qkv_tiled_copy(
            atom_async, self._num_threads, smem_k_block_size, async_elems,
        )
        gmem_tiled_copy_O = make_qkv_tiled_copy(
            atom_universal, self._num_threads, smem_k_block_size, async_elems,
        )

        # ---- Tiled MMA (m16n8k16 fp16/bf16) ----------------------------------
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # ---- Grid ------------------------------------------------------------
        grid_dim = (
            cute.ceil_div(mQ.shape[2], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[1]),
        )

        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            mQ, mK, mV, mO, mBias, mCacheSeqlens, mBlockTable,
            softmax_scale, softmax_scale_log2,
            sQ_layout, sKV_layout, sO_layout,
            gmem_tiled_copy_QKV, gmem_tiled_copy_O,
            tiled_mma, SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    # ------------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mBias: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mBlockTable: cute.Tensor,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()

        # ---- GQA ------------------------------------------------------------
        pack_gqa = PackGQA(qhead_per_kvhead=self._gqa_ratio)
        kv_head = pack_gqa.kv_head_of(num_head)

        # ---- KV extent -------------------------------------------------------
        # ``t_kv_logical`` is the full addressable K extent (dense T_k or
        # ``max_blocks_per_seq * block_size`` for paged). The identity-tensor
        # bookkeeping inside the mainloop uses it so absolute k_col coords
        # are correct.
        if cutlass.const_expr(self._paged):
            t_kv_logical = mBlockTable.shape[1] * self._block_size
        else:
            t_kv_logical = mK.shape[2]

        # Tier 1A: per-CTA tight n_block_max from the stream's seqlen_k.
        # The wrapper always passes a (B,) ``mCacheSeqlens`` tensor; in
        # offline mode it's filled with T_k, so this collapses to the old
        # ``ceil_div(t_kv_logical, N_BLOCK)`` bound. In streaming / paged
        # streaming the per-batch ``seqlen_k`` can be far smaller, and
        # tightening the loop skips the wasted cp.async + QK gemm + PV gemm
        # on tiles whose every column would have been masked to -inf anyway.
        # Precondition: ``seqlen_k >= 1`` (the wrapper synthesizes T_k in
        # offline mode; the streaming engine never dispatches empty streams,
        # and paged mode would have an undefined block_table for them).
        seqlen_k = mCacheSeqlens[batch_size]
        n_block_max = cute.ceil_div(seqlen_k, self._n_block_size)

        # ---- Gmem tiles ------------------------------------------------------
        gQ = cute.local_tile(
            mQ[batch_size, num_head, None, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        if cutlass.const_expr(not self._paged):
            gK = cute.local_tile(
                mK[batch_size, kv_head, None, None],
                (self._n_block_size, self._head_dim_padded),
                (None, 0),
            )
            gV = cute.local_tile(
                mV[batch_size, kv_head, None, None],
                (self._n_block_size, self._head_dim_padded),
                (None, 0),
            )
        else:
            gK = None
            gV = None

        # ---- Smem ------------------------------------------------------------
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK0 = storage.sK0.get_tensor(sKV_layout)
        sK1 = storage.sK1.get_tensor(sKV_layout)
        sV0 = storage.sV0.get_tensor(sKV_layout)
        sV1 = storage.sV1.get_tensor(sKV_layout)
        sVt0 = make_v_transpose_view(sV0, self._head_dim_padded, self._n_block_size)
        sVt1 = make_v_transpose_view(sV1, self._head_dim_padded, self._n_block_size)

        # ---- Partitions ------------------------------------------------------
        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        tKsK0 = gmem_thr_copy_QKV.partition_D(sK0)
        tKsK1 = gmem_thr_copy_QKV.partition_D(sK1)
        tVsV0 = gmem_thr_copy_QKV.partition_D(sV0)
        tVsV1 = gmem_thr_copy_QKV.partition_D(sV1)
        if cutlass.const_expr(not self._paged):
            tKgK = gmem_thr_copy_QKV.partition_S(gK)
            tVgV = gmem_thr_copy_QKV.partition_S(gV)
        else:
            tKgK = None
            tVgV = None

        # ---- MMA partitions / accumulators -----------------------------------
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK0 = thr_mma.make_fragment_B(thr_mma.partition_B(sK0))
        tSrK1 = thr_mma.make_fragment_B(thr_mma.partition_B(sK1))
        tOrVt0 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt0))
        tOrVt1 = thr_mma.make_fragment_B(thr_mma.partition_B(sVt1))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ---- Smem copy atoms -------------------------------------------------
        smem_copy_atom_Q = make_ldmatrix_x4_atom(self._dtype, transpose=False)
        smem_copy_atom_K = make_ldmatrix_x4_atom(self._dtype, transpose=False)
        smem_copy_atom_V = make_ldmatrix_x4_atom(self._dtype, transpose=True)
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK0 = smem_thr_copy_K.partition_S(sK0)
        tSsK1 = smem_thr_copy_K.partition_S(sK1)
        tSrK0_copy_view = smem_thr_copy_K.retile(tSrK0)
        tSrK1_copy_view = smem_thr_copy_K.retile(tSrK1)
        tOsVt0 = smem_thr_copy_V.partition_S(sVt0)
        tOsVt1 = smem_thr_copy_V.partition_S(sVt1)
        tOrVt0_copy_view = smem_thr_copy_V.retile(tOrVt0)
        tOrVt1_copy_view = smem_thr_copy_V.retile(tOrVt1)

        # ---- Predicates ------------------------------------------------------
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_size, num_head, None, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        if cutlass.const_expr(not self._paged):
            mcKV = cute.make_identity_tensor(mK.layout.shape)
            cKV = cute.local_tile(
                mcKV[batch_size, kv_head, None, None],
                (self._n_block_size, self._head_dim_padded),
                (n_block_max - 1, 0),
            )
            tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
            tKVpKV = cute.make_rmem_tensor(
                cute.make_layout(
                    (tKsK0.shape[0][1], cute.size(tKsK0, mode=[1]), cute.size(tKsK0, mode=[2])),
                    stride=(cute.size(tKsK0, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
                for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                    tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                        tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                    )
        else:
            tKVcKV = None
            tKVpKV = None

        # ---- Prefetch Q + first K -------------------------------------------
        # Q (with predicates) -> sQ
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][2], mQ.layout.shape[2]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                tQsQ[None, m, None].fill(0)

        # First K -> sK0 (rightmost N-tile)
        n_block = n_block_max - 1
        self._load_k_tile(
            mK, sK0, tKsK0, tKgK, tKVpKV, tKVcKV,
            mBlockTable, batch_size, kv_head, n_block,
            gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
        )
        cute.arch.cp_async_commit_group()

        # Tier 2: when ``q_in_regs`` is enabled, drain the prologue cp.async
        # group and ldmatrix Q into the rmem fragment once. Subsequent
        # n-block iters then skip the per-iter Q ldmatrix entirely (the
        # gemm helper switches to ``gemm_rs`` which only loads B = K).
        # Draining here is harmless -- iter 0's ``wait_group(0)`` becomes a
        # no-op since K0 is also drained, and the sync barrier ordering
        # holds. K is reloaded in the mainloop as usual.
        if cutlass.const_expr(self._q_in_regs):
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            K_tiles_q = cute.size(tSsQ.shape[2])
            for k in cutlass.range_constexpr(K_tiles_q):
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, k],
                    tSrQ_copy_view[None, None, k],
                )

        # ---- Softmax + mask --------------------------------------------------
        softmax = Softmax.make_from_acc_O(acc_O, softmax_scale_log2)
        softmax.init()
        mask = AttentionMask(
            causal=self._causal,
            window_left=self._window_size_left,
            window_right=self._window_size_right,
            has_seqlen_k=True,
            has_seqlen_q=True,
        )

        # Per-batch length (``seqlen_k`` already loaded above for the
        # tight-n_block_max calc).
        seqlen_q = mQ.shape[2]

        # ---- 3-phase mainloop ------------------------------------------------
        # Iter 0 (rightmost N-tile) is peeled out so the JIT can drop the
        # "is_first" branches in softmax. When causal / window are off the
        # masked-tail / unmasked-inner distinction collapses; for now every
        # iter applies the per-stream length mask (cheap when seqlen_k is
        # block-aligned -- the AttentionMask is constexpr-flagged and the
        # length compare is a single Int32 elem_less per element).
        inv_scale = 1.0 / softmax_scale

        self._compute_one_n_block(
            mQ, mK, mV, mBias, mBlockTable,
            tKgK, tVgV,
            sK0, sK1, sV0, sV1,
            tKsK0, tKsK1, tVsV0, tVsV1,
            tSsQ, tSsK0, tSsK1, tSrQ, tSrQ_copy_view,
            tSrK0, tSrK1, tSrK0_copy_view, tSrK1_copy_view,
            tOrVt0, tOrVt1, tOrVt0_copy_view, tOrVt1_copy_view,
            tOsVt0, tOsVt1,
            smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
            gmem_tiled_copy_QKV, gmem_thr_copy_QKV,
            tKVcKV, tKVpKV,
            tiled_mma, thr_mma, acc_O,
            softmax, mask,
            batch_size, kv_head, num_head,
            t_kv_logical=t_kv_logical,
            seqlen_q=seqlen_q, seqlen_k=seqlen_k,
            inv_scale=inv_scale, m_block=m_block,
            n_block=n_block_max - 1,
            is_first=True,
            curr_stage=0,
        )
        # ``curr_stage`` is a constexpr (0 or 1); dispatch via if/else so the
        # JIT specializes both branches at the call site. ``n_tile`` is a
        # runtime value, so we cannot pass ``n_tile % 2`` as a constexpr.
        for n_tile in range(1, n_block_max, 1):
            n_block_cur = n_block_max - n_tile - 1
            if (n_tile % 2) == 1:
                self._compute_one_n_block(
                    mQ, mK, mV, mBias, mBlockTable,
                    tKgK, tVgV,
                    sK0, sK1, sV0, sV1,
                    tKsK0, tKsK1, tVsV0, tVsV1,
                    tSsQ, tSsK0, tSsK1, tSrQ, tSrQ_copy_view,
                    tSrK0, tSrK1, tSrK0_copy_view, tSrK1_copy_view,
                    tOrVt0, tOrVt1, tOrVt0_copy_view, tOrVt1_copy_view,
                    tOsVt0, tOsVt1,
                    smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
                    gmem_tiled_copy_QKV, gmem_thr_copy_QKV,
                    tKVcKV, tKVpKV,
                    tiled_mma, thr_mma, acc_O,
                    softmax, mask,
                    batch_size, kv_head, num_head,
                    t_kv_logical=t_kv_logical,
                    seqlen_q=seqlen_q, seqlen_k=seqlen_k,
                    inv_scale=inv_scale, m_block=m_block,
                    n_block=n_block_cur,
                    is_first=False,
                    curr_stage=1,
                )
            else:
                self._compute_one_n_block(
                    mQ, mK, mV, mBias, mBlockTable,
                    tKgK, tVgV,
                    sK0, sK1, sV0, sV1,
                    tKsK0, tKsK1, tVsV0, tVsV1,
                    tSsQ, tSsK0, tSsK1, tSrQ, tSrQ_copy_view,
                    tSrK0, tSrK1, tSrK0_copy_view, tSrK1_copy_view,
                    tOrVt0, tOrVt1, tOrVt0_copy_view, tOrVt1_copy_view,
                    tOsVt0, tOsVt1,
                    smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
                    gmem_tiled_copy_QKV, gmem_thr_copy_QKV,
                    tKVcKV, tKVpKV,
                    tiled_mma, thr_mma, acc_O,
                    softmax, mask,
                    batch_size, kv_head, num_head,
                    t_kv_logical=t_kv_logical,
                    seqlen_q=seqlen_q, seqlen_k=seqlen_k,
                    inv_scale=inv_scale, m_block=m_block,
                    n_block=n_block_cur,
                    is_first=False,
                    curr_stage=0,
                )

        # ---- Epilogue --------------------------------------------------------
        softmax.finalize(acc_O)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        gO = cute.local_tile(
            mO[batch_size, num_head, None, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)

        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO = cute.local_tile(
            mcO[batch_size, num_head, None, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][2], mO.layout.shape[2]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

    # ------------------------------------------------------------------------
    # Per-N-tile compute
    # ------------------------------------------------------------------------
    @cute.jit
    def _compute_one_n_block(
        self,
        mQ, mK, mV, mBias, mBlockTable,
        tKgK, tVgV,
        sK0, sK1, sV0, sV1,
        tKsK0, tKsK1, tVsV0, tVsV1,
        tSsQ, tSsK0, tSsK1, tSrQ, tSrQ_copy_view,
        tSrK0, tSrK1, tSrK0_copy_view, tSrK1_copy_view,
        tOrVt0, tOrVt1, tOrVt0_copy_view, tOrVt1_copy_view,
        tOsVt0, tOsVt1,
        smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
        gmem_tiled_copy_QKV, gmem_thr_copy_QKV,
        tKVcKV, tKVpKV,
        tiled_mma, thr_mma, acc_O,
        softmax, mask,
        batch_size, kv_head, num_head,
        t_kv_logical: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        inv_scale: cutlass.Float32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        is_first: cutlass.Constexpr,
        curr_stage: cutlass.Constexpr,
    ):
        # Select per-stage K / V partition views.
        if cutlass.const_expr(curr_stage == 0):
            tSsK_curr = tSsK0
            tSrK_curr = tSrK0
            tSrK_curr_copy_view = tSrK0_copy_view
            tKsK_next = tKsK1
            sK_next = sK1
            tVsV_curr = tVsV0
            tOsVt_curr = tOsVt0
            tOrVt_curr = tOrVt0
            tOrVt_curr_copy_view = tOrVt0_copy_view
            sV_curr = sV0
        else:
            tSsK_curr = tSsK1
            tSrK_curr = tSrK1
            tSrK_curr_copy_view = tSrK1_copy_view
            tKsK_next = tKsK0
            sK_next = sK0
            tVsV_curr = tVsV1
            tOsVt_curr = tOsVt1
            tOrVt_curr = tOrVt1
            tOrVt_curr_copy_view = tOrVt1_copy_view
            sV_curr = sV1

        # ---- Wait for K(curr); stage V(curr) + K(next) cp.async --------
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # V(curr) load (first-tile predicate on K-residue).
        if cutlass.const_expr(not self._paged):
            if is_first:
                for n in cutlass.range_constexpr(cute.size(tVsV_curr.shape[1])):
                    if cute.elem_less(tKVcKV[0, n, 0][2], mK.layout.shape[2]):
                        cute.copy(
                            gmem_tiled_copy_QKV,
                            tVgV[None, n, None, n_block],
                            tVsV_curr[None, n, None],
                            pred=tKVpKV[None, n, None],
                        )
                    else:
                        tVsV_curr[None, n, None].fill(0.0)
            else:
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tVgV[None, None, None, n_block],
                    tVsV_curr,
                    pred=tKVpKV,
                )
        else:
            self._paged_load_kv_tile(
                mV, sV_curr, mBlockTable,
                batch_size, kv_head, n_block,
                gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
            )
        cute.arch.cp_async_commit_group()

        # K(next) prefetch — runs concurrently with the QK gemm below.
        if n_block > 0:
            if cutlass.const_expr(not self._paged):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKgK[None, None, None, n_block - 1],
                    tKsK_next,
                    pred=tKVpKV,
                )
            else:
                self._paged_load_kv_tile(
                    mK, sK_next, mBlockTable,
                    batch_size, kv_head, n_block - 1,
                    gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
                )
            cute.arch.cp_async_commit_group()

        # ---- QK gemm (reads sK_curr; overlaps with V + K(next) cp.async) ----
        acc_shape_S = thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)
        if cutlass.const_expr(self._q_in_regs):
            # Q is pre-populated in rmem (kernel prologue ldmatrix). Only
            # need to load K from smem.
            gemm_rs(
                tiled_mma, acc_S,
                rA_view=tSrQ, rB_for_mma=tSrK_curr,
                smem_tiled_copy_B=smem_tiled_copy_K,
                sB_partitioned=tSsK_curr,
                rB_copy_view=tSrK_curr_copy_view,
            )
        else:
            gemm_with_smem_prefetch(
                tiled_mma, acc_S,
                rA_for_mma=tSrQ, rB_for_mma=tSrK_curr,
                smem_tiled_copy_A=smem_tiled_copy_Q,
                smem_tiled_copy_B=smem_tiled_copy_K,
                sA_partitioned=tSsQ, sB_partitioned=tSsK_curr,
                rA_copy_view=tSrQ_copy_view,
                rB_copy_view=tSrK_curr_copy_view,
            )

        # ---- Wait for V(curr) [+ K(next) still in flight] -------------------
        if n_block > 0:
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # ---- Bias add (gmem-direct, with bounds check) ----------------------
        if cutlass.const_expr(self._has_bias):
            self._add_bias_tile(
                mBias, batch_size, num_head,
                m_block, n_block, thr_mma, acc_S, inv_scale,
            )

        # ---- Mask + online softmax ------------------------------------------
        # Identity tile for (q_row, k_col) bounds. We synthesize a (B, H,
        # T_q, T_kv_logical) shape so ``tScS_mn[r, c][2]`` is the absolute
        # q_row and ``[3]`` is the absolute k_col regardless of whether
        # KV lives in a dense tensor or a paged pool.
        mcS = cute.make_identity_tensor(
            (mQ.shape[0], mQ.shape[1], mQ.shape[2], t_kv_logical)
        )
        cS = cute.local_tile(
            mcS[batch_size, num_head, None, None],
            (self._m_block_size, self._n_block_size),
            (m_block, n_block),
        )
        tScS = thr_mma.partition_C(cS)
        mask.apply(acc_S, tScS, seqlen_q=seqlen_q, seqlen_k=seqlen_k)
        softmax.online_softmax(acc_S, acc_O, is_first=is_first)

        # ---- PV gemm --------------------------------------------------------
        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # Layout transform (4, MMA_M, MMA_N) -> ((4,2), MMA_M, MMA_N/2).
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
        gemm_rs(
            tiled_mma, acc_O,
            rA_view=tOrS, rB_for_mma=tOrVt_curr,
            smem_tiled_copy_B=smem_tiled_copy_V,
            sB_partitioned=tOsVt_curr,
            rB_copy_view=tOrVt_curr_copy_view,
        )

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------
    @cute.jit
    def _load_k_tile(
        self,
        mK, sK_dst, tKsK_dst, tKgK, tKVpKV, tKVcKV,
        mBlockTable, batch_size, kv_head, n_block,
        gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
    ):
        """Load one K N-tile into ``sK_dst``. Dispatches dense vs paged."""
        if cutlass.const_expr(not self._paged):
            for n in cutlass.range_constexpr(cute.size(tKsK_dst.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][2], mK.layout.shape[2]):
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tKgK[None, n, None, n_block],
                        tKsK_dst[None, n, None],
                        pred=tKVpKV[None, n, None],
                    )
                else:
                    tKsK_dst[None, n, None].fill(0)
        else:
            self._paged_load_kv_tile(
                mK, sK_dst, mBlockTable, batch_size, kv_head, n_block,
                gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
            )

    @cute.jit
    def _paged_load_kv_tile(
        self,
        mKV: cute.Tensor,
        sKV: cute.Tensor,
        mBlockTable: cute.Tensor,
        batch_idx: cutlass.Int32,
        kv_head: cutlass.Int32,
        n_block: cutlass.Int32,
        gmem_thr_copy: cute.TiledCopy,
        gmem_tiled_copy: cute.TiledCopy,
    ):
        """Gather one N-tile of K (or V) from the paged pool into smem.

        ``mKV`` is ``(num_blocks, block_size, H_kv, D)`` (the per-layer pool
        view). ``sKV`` is the destination smem with ``(N_BLOCK, D)`` shape.
        We slice the smem into ``blocks_per_n_tile`` sub-tiles and issue
        one cp.async per logical block. Lives on ``self`` so cute.jit
        captures ``self._block_size`` / ``self._head_dim_padded`` /
        ``self._blocks_per_n_tile`` as class-level constexprs and avoids
        flattening a free-standing PagedKVManager into runtime if-regions.
        """
        n_logical_base = n_block * self._blocks_per_n_tile
        for i in cutlass.range_constexpr(self._blocks_per_n_tile):
            phys = mBlockTable[batch_idx, n_logical_base + i]
            gKV_block = cute.local_tile(
                mKV[phys, None, kv_head, None],
                (self._block_size, self._head_dim_padded),
                (0, 0),
            )
            sKV_sub = cute.local_tile(
                sKV,
                (self._block_size, self._head_dim_padded),
                (i, 0),
            )
            tKVgKV = gmem_thr_copy.partition_S(gKV_block)
            tKVsKV = gmem_thr_copy.partition_D(sKV_sub)
            cute.copy(gmem_tiled_copy, tKVgKV, tKVsKV)

    @cute.jit
    def _add_bias_tile(
        self,
        mBias, batch_size, num_head,
        m_block, n_block, thr_mma, acc_S, inv_scale,
    ):
        """Add gmem-direct bias to acc_S, divided by softmax_scale.

        SDPA semantics: bias is treated as a post-scale logit. The softmax
        baked into ``exp2((acc_S + bias_scaled) * scale_log2)`` collapses to
        ``exp(scale * acc_S + bias)`` once we pre-divide bias by scale.
        """
        gBias = cute.local_tile(
            mBias[batch_size, num_head, None, None],
            (self._m_block_size, self._n_block_size),
            (m_block, n_block),
        )
        tBias = thr_mma.partition_C(gBias)
        tBias_mn = make_acc_mn_view(tBias)
        acc_S_mn = make_acc_mn_view(acc_S)

        # Bounds tensor for q_row / k_col so we don't read past mBias.
        mcS = cute.make_identity_tensor(mBias.layout.shape)
        cS = cute.local_tile(
            mcS[batch_size, num_head, None, None],
            (self._m_block_size, self._n_block_size),
            (m_block, n_block),
        )
        tScS = thr_mma.partition_C(cS)
        tScS_mn = make_acc_mn_view(tScS)

        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            row_ok = cute.elem_less(tScS_mn[r, 0][2], mBias.shape[2])
            if row_ok:
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    if cute.elem_less(tScS_mn[r, c][3], mBias.shape[3]):
                        acc_S_mn[r, c] = (
                            acc_S_mn[r, c]
                            + cutlass.Float32(tBias_mn[r, c]) * inv_scale
                        )
