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

Key features (Tier 2 cp.async ring matching FA's flash_fwd SM80 path):

* ``num_stages`` cp.async ring for **both** K and V, default 3. Single 3D
  ``sK`` and ``sV`` smem tensors of shape ``(N_BLOCK, head_dim, num_stages)``;
  the iter loop indexes the active stage at runtime via ``smem_pipe_read``
  / ``smem_pipe_write`` ``Int32`` counters that cycle mod num_stages via
  :meth:`FmhaSm80._advance_pipeline`. Smem grows to 56 KB at
  ``M=N=64, D=64, num_stages=3`` — still well under SM120's 99 KB cap.
* Optional ``q_in_regs`` mode: ldmatrix Q once in the prologue, then
  reuse sQ's smem region as ``sV`` via ``cute.recast_ptr``. Saves
  ``num_stages * N * D * dtype_bytes`` of smem.
* FIFO drain magic: prologue commits each stage's K then V so V's commit
  lands between successive K commits. The iter loop's
  ``wait_group(num_stages * 2 - 2)`` drains the oldest K right before its
  QK gemm consumes it; later in the iter another wait drains the V right
  before its PV gemm.
* :class:`AttentionMask` carries the per-stream length, causal, and
  sliding-window mask in one constexpr-flagged loop. Causal / window-size
  knobs are wired through ``FmhaBase`` and default off.
* Online softmax lives in :class:`Softmax`. Empty-row clamp
  (``row_max == -inf`` => 0) carries over.
* Paged KV uses ``_paged_load_kv_tile`` on this class. Same
  block-aligned-``n_block`` constraint as before
  (``N_BLOCK % block_size == 0``).
* Varlen is scaffolded (constexpr ``self._varlen`` in feasibility), but
  the kernel-side packed-tensor indexing lands in a follow-up phase.
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


class FmhaSm80(FmhaBase):
    """SM80 / SM120 FMHA.

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
        num_stages: int = 3,
        q_in_regs: bool = False,
        bias_aligned: bool = False,
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
        # Tier 2: cp.async ring depth for both K and V. Structurally tracks
        # FlashAttention's ``num_stages`` knob: sK / sV are a single 3D
        # smem tensor of shape ``(N, D, num_stages)``, indexed cyclically
        # via Int32 ``smem_pipe_read`` / ``smem_pipe_write`` counters that
        # the iter loop advances mod ``num_stages``. ``num_stages = 2``
        # matches the prior 2-stage behavior; bumping to ``3`` keeps three
        # K + (num_stages - 1) V loads inflight at steady state for deeper
        # cp.async latency hiding (FA's expected 2-5% on compute-bound
        # shapes). Default stays at 2 so the OASR perf envelope doesn't
        # shift under the user.
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        self._num_stages = num_stages
        # Tier 2 (paired with the num_stages cp.async ring): when True,
        # ldmatrix Q to rmem once in the prologue and skip the per-iter
        # Q ldmatrix. Also opts into the sQ/sV0 smem alias path — sQ's
        # MemRange is sized to hold the multi-stage V ring and sV is a
        # ``cute.recast_ptr`` view of the same memory. Saves
        # ``num_stages * N * D * dtype_bytes`` of smem (= 24 KB at
        # num_stages=3, M=N=64, D=64, fp16). Default off because the added
        # rmem-Q register pressure regresses small-shape latency; turn on
        # via the constructor when smem capacity matters (e.g. D >= 128 or
        # SM80 with num_stages >= 4).
        self._q_in_regs = q_in_regs
        # Bias path optimization knob (only checked when has_bias=True).
        # When True, the kernel assumes the bias trailing dim T_k is even
        # (= row stride divisible by 4 B for fp16) and can issue unpredicated
        # vectorized ``autovec_copy`` (b32 col-pair loads) into rmem.
        # When False, falls back to a predicated element-wise ``basic_copy_if``
        # that's slower but safe on odd T_k (e.g. real audio frame counts
        # like 33 / 249). The wrapper computes this from
        # ``attn_bias.size(-1) % 2 == 0``.
        self._bias_aligned = bias_aligned

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
        num_stages: int = 3,
        has_bias: bool = False,
        paged: bool = False,
        block_size: int = 0,
        causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        varlen: bool = False,
        bias_aligned: bool = False,
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
        if num_stages < 1:
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
        # Smem budget: sQ + num_stages * (sK + sV). Bias (when has_bias) is
        # gmem-direct, not staged through smem.
        del has_bias  # not part of the smem budget
        del causal
        del window_size_left
        del window_size_right
        del varlen
        smem_bytes = (
            m_block_size * head_dim
            + n_block_size * head_dim * num_stages * 2  # num_stages * (sK + sV)
        ) * 2                                            # fp16/bf16 = 2B
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
        # sQ stays 2D ``(M_BLOCK, head_dim)``. sK and sV are 3D
        # ``(N_BLOCK, head_dim, num_stages)`` so the iter loop can index
        # the ring stage at runtime via ``smem_pipe_read`` / ``write``.
        # Structurally tracks ``FlashAttentionForwardSm80._get_shared_storage_cls``.
        sQ_atom, _smem_k_block_size = make_smem_swizzle_atom(
            self._dtype, self._head_dim_padded,
        )
        sQ_layout = make_smem_layout(
            sQ_atom, self._m_block_size, self._head_dim_padded,
        )
        sK_layout = cute.tile_to_shape(
            sQ_atom,
            (self._n_block_size, self._head_dim_padded, self._num_stages),
            (0, 1, 2),
        )
        sV_layout = sK_layout
        sO_layout = sQ_layout

        # When ``q_in_regs`` is on, sQ and sV share one ``MemRange`` sized to
        # max(cosize(sQ), cosize(sV)). Q is written/read first (cp.async +
        # ldmatrix), then a barrier flips ownership and the same smem holds
        # sV's multi-stage ring. Matches FA's ``SharedStorageSharedQV``;
        # saves ``num_stages * N * D * dtype_bytes`` of smem.
        if cutlass.const_expr(self._q_in_regs):
            cosize_sQV = max(cute.cosize(sQ_layout), cute.cosize(sV_layout))

            @cute.struct
            class SharedStorage:
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cosize_sQV], 1024
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024
                ]
        else:
            @cute.struct
            class SharedStorage:
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024
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
            sQ_layout, sK_layout, sV_layout, sO_layout,
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
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
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
        # Single sK / sV smem tensor with a trailing ``num_stages`` dim;
        # the iter loop indexes the stage at runtime via the
        # smem_pipe_read / smem_pipe_write Int32 counters. When q_in_regs is
        # on, sV reuses sQ's smem region via ``cute.recast_ptr`` — sized to
        # hold the multi-stage V ring (max of sQ-cosize, sV-cosize).
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if cutlass.const_expr(self._q_in_regs):
            sV = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=self._dtype),
                sV_layout,
            )
        else:
            sV = storage.sV.get_tensor(sV_layout)
        sVt = make_v_transpose_view(
            sV, self._head_dim_padded, self._n_block_size,
            num_stages=self._num_stages,
        )

        # ---- Partitions ------------------------------------------------------
        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        # tKsK / tVsV are 4D: (CPY_Atom, CPY_N, CPY_K, num_stages). Slicing
        # the last dim with the pipe counter at issue / load time picks the
        # active stage's smem region.
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)
        if cutlass.const_expr(not self._paged):
            tKgK = gmem_thr_copy_QKV.partition_S(gK)
            tVgV = gmem_thr_copy_QKV.partition_S(gV)
        else:
            tKgK = None
            tVgV = None

        # ---- MMA partitions / accumulators -----------------------------------
        thr_mma = tiled_mma.get_slice(tidx)
        # Use stage 0 slice to size the rmem fragments (same shape per stage).
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(
            thr_mma.partition_B(sK[None, None, 0])
        )
        tOrVt = thr_mma.make_fragment_B(
            thr_mma.partition_B(sVt[None, None, 0])
        )
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
        # 4D ldmatrix partitions; slice the trailing dim by the pipe
        # counter inside the gemm.
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

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
                    (tKsK.shape[0][1], cute.size(tKsK, mode=[1]), cute.size(tKsK, mode=[2])),
                    stride=(cute.size(tKsK, mode=[2]), 0, 1),
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

        # ---- Prologue (FA-style cp.async ring fill) -------------------------
        # Pattern mirrors ``FlashAttentionForwardSm80.__call__``'s prologue.
        # FIFO drain trick: each stage commits K **before** V so V's commit
        # lands between K's; the iter loop's
        # ``wait_group(num_stages * 2 - 2)`` then drains exactly the oldest
        # K (which we're about to consume), leaving the more recent K/V
        # loads inflight.
        #
        # Q_in_regs ordering: when sQ/sV alias, V's cp.async writes the same
        # smem region as Q's. So Q must be (a) drained, (b) ldmatrix'd to
        # rmem, and (c) cross-warp synced before any V issue. We split the
        # prologue accordingly:
        #
        #   1. Q -> sQ; commit
        #   2. K[N-1] -> sK[0]; commit
        #   3. (q_in_regs only) wait_group(num_stages * 2 - 1) drains Q;
        #      barrier; ldmatrix Q; barrier  — sQ is now safe to alias.
        #   4. For stage in [0, num_stages):
        #        - if (q_in_regs and stage>0) or not q_in_regs: load_K + commit
        #        - if stage < num_stages-1: load_V + commit
        #   5. (no-q_in_regs) wait_group(num_stages * 2 - 1) drains Q.
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
        cute.arch.cp_async_commit_group()

        n_block_init = n_block_max - 1

        # Q_in_regs: load K[N-1] into sK[0] before the wait/ldmatrix Q step,
        # then ldmatrix Q to rmem, barrier, and **only then** issue V loads
        # (which will write to the sQ-aliased smem region).
        if cutlass.const_expr(self._q_in_regs):
            self._load_kv_tile(
                mK, sK, tKsK, tKgK, tKVpKV, tKVcKV,
                mBlockTable, batch_size, kv_head,
                n_block_init, 0,
                gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
                need_predicates=True,
            )
            cute.arch.cp_async_commit_group()
            # Drain Q + K[N-1]. With 2 commits inflight, wait_group(0) is
            # the only safe choice (we need Q drained; K[N-1] will also
            # drain, which is fine — its data is in sK[0] in smem).
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            K_tiles_q = cute.size(tSsQ.shape[2])
            for k in cutlass.range_constexpr(K_tiles_q):
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, k],
                    tSrQ_copy_view[None, None, k],
                )
            # Ensure all warps finished reading sQ before any V cp.async writes.
            self.cta_sync_barrier.arrive_and_wait()

        # Stage-loop: load K[N-1-stage] and V[N-1-stage] for each ring stage.
        for stage in cutlass.range_constexpr(self._num_stages):
            # K: stage 0 already loaded above in the q_in_regs path.
            if cutlass.const_expr(not self._q_in_regs) or stage > 0:
                if stage == 0 or n_block_init - stage >= 0:
                    self._load_kv_tile(
                        mK, sK, tKsK, tKgK, tKVpKV, tKVcKV,
                        mBlockTable, batch_size, kv_head,
                        n_block_init - stage, stage,
                        gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
                        need_predicates=(stage == 0),
                    )
                cute.arch.cp_async_commit_group()
            # V: load num_stages - 1 V stages here; the last V stage is
            # loaded by iter 0's body.
            if cutlass.const_expr(stage < self._num_stages - 1):
                if stage == 0 or n_block_init - stage >= 0:
                    self._load_kv_tile(
                        mV, sV, tVsV, tVgV, tKVpKV, tKVcKV,
                        mBlockTable, batch_size, kv_head,
                        n_block_init - stage, stage,
                        gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
                        need_predicates=(stage == 0),
                    )
                cute.arch.cp_async_commit_group()

        # No-Q_in_regs: drain Q after all K/V issues. Total commits =
        # 1 (Q) + num_stages (K) + (num_stages - 1) (V) = 2 * num_stages.
        # wait_group(2 * num_stages - 1) drains 1 (= Q) and leaves the
        # K/V's inflight for the iter loop to consume.
        if cutlass.const_expr(not self._q_in_regs):
            cute.arch.cp_async_wait_group(2 * self._num_stages - 1)
            self.cta_sync_barrier.arrive_and_wait()

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
        seqlen_q = mQ.shape[2]
        inv_scale = 1.0 / softmax_scale

        # ---- Mainloop (pipe-counter cycling per FA) --------------------------
        # ``smem_pipe_read`` selects which sK / sV stage iter n consumes;
        # ``smem_pipe_write`` selects the slot for the next-stage's-worth
        # of load issues. Both cycle mod num_stages via ``_advance_pipeline``.
        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(self._num_stages - 1)

        self._compute_one_n_block(
            mQ, mK, mV, mBias, mBlockTable,
            tKgK, tVgV, sK, sV,
            tKsK, tVsV,
            tSsQ, tSsK, tSrQ, tSrQ_copy_view,
            tSrK, tSrK_copy_view,
            tOrVt, tOrVt_copy_view,
            tOsVt,
            smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V,
            gmem_tiled_copy_QKV, gmem_thr_copy_QKV,
            tKVcKV, tKVpKV,
            tiled_mma, thr_mma, acc_O,
            softmax, mask,
            batch_size, kv_head, num_head,
            t_kv_logical=t_kv_logical,
            seqlen_q=seqlen_q, seqlen_k=seqlen_k,
            inv_scale=inv_scale, m_block=m_block,
            n_block=n_block_init,
            smem_pipe_read=smem_pipe_read,
            smem_pipe_write=smem_pipe_write,
            is_first=True,
        )
        smem_pipe_read = self._advance_pipeline(smem_pipe_read)
        smem_pipe_write = self._advance_pipeline(smem_pipe_write)

        for n_tile in range(1, n_block_max, 1):
            n_block_cur = n_block_max - n_tile - 1
            self._compute_one_n_block(
                mQ, mK, mV, mBias, mBlockTable,
                tKgK, tVgV, sK, sV,
                tKsK, tVsV,
                tSsQ, tSsK, tSrQ, tSrQ_copy_view,
                tSrK, tSrK_copy_view,
                tOrVt, tOrVt_copy_view,
                tOsVt,
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
                smem_pipe_read=smem_pipe_read,
                smem_pipe_write=smem_pipe_write,
                is_first=False,
            )
            smem_pipe_read = self._advance_pipeline(smem_pipe_read)
            smem_pipe_write = self._advance_pipeline(smem_pipe_write)

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

    @cute.jit
    def _advance_pipeline(self, pipeline_index):
        """``(pipeline_index + 1) % num_stages`` for the cp.async ring."""
        return (
            pipeline_index + 1
            if pipeline_index < self._num_stages - 1
            else cutlass.Int32(0)
        )

    # ------------------------------------------------------------------------
    # Per-N-tile compute
    # ------------------------------------------------------------------------
    @cute.jit
    def _compute_one_n_block(
        self,
        mQ, mK, mV, mBias, mBlockTable,
        tKgK, tVgV, sK, sV,
        tKsK, tVsV,
        tSsQ, tSsK, tSrQ, tSrQ_copy_view,
        tSrK, tSrK_copy_view,
        tOrVt, tOrVt_copy_view,
        tOsVt,
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
        smem_pipe_read: cutlass.Int32,
        smem_pipe_write: cutlass.Int32,
        is_first: cutlass.Constexpr,
    ):
        """One N-tile of S/O processing.

        Structurally tracks ``FlashAttentionForwardSm80.compute_one_n_block``:
        the iter has two ``wait_group(num_stages*2 - 2)`` syncs — one
        before the QK gemm to drain the K we're about to consume, one
        before the PV gemm to drain the V we're about to consume. Between
        the QK gemm and the PV gemm the iter issues the V_next (look-ahead
        ``num_stages - 1`` blocks ahead) and the K_next (look-ahead
        ``num_stages`` blocks ahead) cp.asyncs into the slots picked by
        ``smem_pipe_write``.
        """
        # Sync helper — leaves (num_stages*2 - 2) groups inflight, drains 1.
        # For num_stages=2: wait_group(2) keeps 2 groups inflight. For
        # num_stages=3: wait_group(4) keeps 4. The FIFO order guarantees
        # the drained group is the one whose data we're about to read.
        wait_count = self._num_stages * 2 - 2

        acc_shape_S = thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # Drain the K[n_block] cp.async (oldest in flight).
        cute.arch.cp_async_wait_group(wait_count)
        self.cta_sync_barrier.arrive_and_wait()

        # Issue V_next = V[n_block - (num_stages - 1)] into sV[smem_pipe_write].
        # This V is consumed `num_stages - 1` iters from now; the lead lets
        # the cp.async overlap with the QK gemm below + the subsequent
        # softmax. Skipped when the look-ahead would go past block 0.
        if cutlass.const_expr(self._num_stages == 1) or (
            n_block - self._num_stages + 1 >= 0
        ):
            n_block_v_next = (
                n_block
                if cutlass.const_expr(self._num_stages == 1)
                else n_block - self._num_stages + 1
            )
            self._load_kv_tile(
                mV, sV, tVsV, tVgV, tKVpKV, tKVcKV,
                mBlockTable, batch_size, kv_head,
                n_block_v_next, smem_pipe_write,
                gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
                need_predicates=(is_first and self._num_stages == 1),
            )
        cute.arch.cp_async_commit_group()

        # ---- QK gemm: reads sK at the smem_pipe_read stage. ---------------
        if cutlass.const_expr(self._q_in_regs):
            gemm_rs(
                tiled_mma, acc_S,
                rA_view=tSrQ, rB_for_mma=tSrK,
                smem_tiled_copy_B=smem_tiled_copy_K,
                sB_partitioned=tSsK[None, None, None, smem_pipe_read],
                rB_copy_view=tSrK_copy_view,
            )
        else:
            gemm_with_smem_prefetch(
                tiled_mma, acc_S,
                rA_for_mma=tSrQ, rB_for_mma=tSrK,
                smem_tiled_copy_A=smem_tiled_copy_Q,
                smem_tiled_copy_B=smem_tiled_copy_K,
                sA_partitioned=tSsQ,
                sB_partitioned=tSsK[None, None, None, smem_pipe_read],
                rA_copy_view=tSrQ_copy_view,
                rB_copy_view=tSrK_copy_view,
            )

        # Advance smem_pipe_write for the K_next issue (K writes to the
        # slot one ahead of where V went, mirroring the prologue's K-then-V
        # ordering inside each stage).
        smem_pipe_write = self._advance_pipeline(smem_pipe_write)

        # Drain V[n_block] for the PV gemm. (Same wait count — by now
        # V[n_block]'s commit is the oldest remaining.) Also issues
        # K_next = K[n_block - num_stages] for the iter that's num_stages
        # ahead, keeping the ring full.
        if cutlass.const_expr(self._num_stages == 1):
            cute.arch.cp_async_wait_group(wait_count)
            self.cta_sync_barrier.arrive_and_wait()
            if n_block - 1 >= 0:
                self._load_kv_tile(
                    mK, sK, tKsK, tKgK, tKVpKV, tKVcKV,
                    mBlockTable, batch_size, kv_head,
                    n_block - 1, smem_pipe_write,
                    gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
                    need_predicates=False,
                )
            cute.arch.cp_async_commit_group()

        # ---- Bias add (in-rmem) -------------------------------------------
        if cutlass.const_expr(self._has_bias):
            self._add_bias_tile(
                mBias, batch_size, num_head,
                m_block, n_block, thr_mma, acc_S, inv_scale,
            )

        # ---- Mask + online softmax ----------------------------------------
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

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
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

        # Drain V[n_block] before PV. Same wait pattern — by now V's commit
        # is the oldest. Also issue K_next here for num_stages > 1 (so it
        # overlaps with PV gemm instead of with QK).
        if cutlass.const_expr(self._num_stages > 1):
            cute.arch.cp_async_wait_group(wait_count)
            self.cta_sync_barrier.arrive_and_wait()
            if n_block - self._num_stages >= 0:
                self._load_kv_tile(
                    mK, sK, tKsK, tKgK, tKVpKV, tKVcKV,
                    mBlockTable, batch_size, kv_head,
                    n_block - self._num_stages, smem_pipe_write,
                    gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
                    need_predicates=False,
                )
            cute.arch.cp_async_commit_group()

        # ---- PV gemm: reads sV at the smem_pipe_read stage. ---------------
        gemm_rs(
            tiled_mma, acc_O,
            rA_view=tOrS, rB_for_mma=tOrVt,
            smem_tiled_copy_B=smem_tiled_copy_V,
            sB_partitioned=tOsVt[None, None, None, smem_pipe_read],
            rB_copy_view=tOrVt_copy_view,
        )

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------
    @cute.jit
    def _load_kv_tile(
        self,
        mKV, sKV, tKVsKV_3D, tKVgKV, tKVpKV, tKVcKV,
        mBlockTable, batch_size, kv_head, n_block, stage,
        gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
        need_predicates: cutlass.Constexpr,
    ):
        """Load one K (or V) N-tile into the ring's ``stage`` slot.

        ``tKVsKV_3D`` is the 4D partitioned smem destination
        ``(Atom, n, K_tile, num_stages)``; the inner stage slice
        ``[..., stage]`` selects the active ring slot. ``need_predicates``
        is True only for the rightmost (residue-prone) tile — inner tiles
        can issue an unpredicated bulk cp.async.
        """
        if cutlass.const_expr(not self._paged):
            tKVsKV = tKVsKV_3D[None, None, None, stage]
            if cutlass.const_expr(need_predicates):
                for n in cutlass.range_constexpr(cute.size(tKVsKV.shape[1])):
                    if cute.elem_less(tKVcKV[0, n, 0][2], mKV.layout.shape[2]):
                        cute.copy(
                            gmem_tiled_copy_QKV,
                            tKVgKV[None, n, None, n_block],
                            tKVsKV[None, n, None],
                            pred=tKVpKV[None, n, None],
                        )
                    else:
                        tKVsKV[None, n, None].fill(0)
            else:
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKVgKV[None, None, None, n_block],
                    tKVsKV,
                    pred=tKVpKV,
                )
        else:
            self._paged_load_kv_tile(
                mKV, sKV[None, None, stage], mBlockTable,
                batch_size, kv_head, n_block,
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

        Implementation note: bulk-loads the per-thread bias fragment from
        gmem into rmem in one pass, then runs the add as a pure-compute
        pass. Concentrating the gmem reads in one copy lets the JIT emit
        ILP-friendly loads where the per-thread MMA C-partition lets it
        (col-pair adjacent elements fold into b32 reads).

        Why this matters: Triton's ``do_bench`` (used by our bench harness)
        clears L2 between iters, so cold gmem latency on the 4 MB bias
        tile dominates if reads are serialized inside the read-modify-write
        of the add. The old per-element form burned ~60 us per call on
        the (1,8,8,256,1024,64) shape.

        Two paths, dispatched at compile time on the constexpr
        ``self._bias_aligned``:

        * **aligned** (``T_k`` is even — every real-world bias tensor we
          ship): unpredicated ``cute.autovec_copy`` emits b32 col-pair loads.
          OOB rows/cols (when ``T_q`` / ``T_k`` don't divide the tile) read
          stale gmem; the values are harmless because ``AttentionMask.apply``
          overwrites those acc_S slots with -inf afterwards.
        * **unaligned** (``T_k`` is odd, e.g. 33-frame audio batches in the
          test suite): falls back to predicated ``cute.basic_copy_if`` —
          slower because the per-element predicate breaks vectorization,
          but safe against the b32-vs-2-byte-aligned-row-stride fault that
          unpredicated autovec hits.
        """
        gBias = cute.local_tile(
            mBias[batch_size, num_head, None, None],
            (self._m_block_size, self._n_block_size),
            (m_block, n_block),
        )
        tBias = thr_mma.partition_C(gBias)

        rBias = cute.make_fragment_like(tBias, self._dtype)
        if cutlass.const_expr(self._bias_aligned):
            # Fast path: T_k % 2 == 0 guarantees the col-pair (2 fp16) is
            # 4-byte aligned at every (q_row, k_col) start, so a b32 load
            # never faults. OOB read residues are masked to -inf later.
            cute.autovec_copy(tBias, rBias)
        else:
            # Safe path: build a per-element bounds predicate and use
            # predicated copy so we never read OOB / misaligned addresses.
            mcS = cute.make_identity_tensor(mBias.layout.shape)
            cS = cute.local_tile(
                mcS[batch_size, num_head, None, None],
                (self._m_block_size, self._n_block_size),
                (m_block, n_block),
            )
            tScS = thr_mma.partition_C(cS)
            tScS_mn = make_acc_mn_view(tScS)

            rBiasPred = cute.make_fragment_like(tBias, cutlass.Boolean)
            rBiasPred_mn = make_acc_mn_view(rBiasPred)
            for r in cutlass.range_constexpr(cute.size(rBiasPred_mn.shape[0])):
                row_ok = cute.elem_less(tScS_mn[r, 0][2], mBias.shape[2])
                for c in cutlass.range_constexpr(cute.size(rBiasPred_mn.shape[1])):
                    rBiasPred_mn[r, c] = row_ok and cute.elem_less(
                        tScS_mn[r, c][3], mBias.shape[3]
                    )
            rBias.fill(0)
            cute.basic_copy_if(rBiasPred, tBias, rBias)

        # Pure-compute add. On the aligned path, OOB rBias entries hold
        # stale gmem (will be -inf masked later). On the unaligned path,
        # OOB entries are 0 and the add is a no-op.
        rBias_mn = make_acc_mn_view(rBias)
        acc_S_mn = make_acc_mn_view(acc_S)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                acc_S_mn[r, c] = (
                    acc_S_mn[r, c]
                    + cutlass.Float32(rBias_mn[r, c]) * inv_scale
                )

