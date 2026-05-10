# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SM80 fused multi-head attention (CuteDSL).

This kernel is structurally a port of CUTLASS's bundled
``examples/python/CuTeDSL/ampere/flash_attention_v2.py``
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

from types import SimpleNamespace
from typing import Any

# --- Deferred imports ---
# Importing cutlass.cute is expensive (drags in MLIR/compiler infrastructure).
# We import lazily so module import on a non-CUDA host is cheap.

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import cuda.bindings.driver as cuda
from cutlass.cute.nvgpu import cpasync, warp

from .base import FmhaBase


class FmhaSm80(FmhaBase):
    """SM80 FMHA kernel.

    Tile choices for Conformer-sized problems (T_q small, T_k modest):

    * ``M_BLOCK = 64`` -- m-block big enough to satisfy
      ``(M_BLOCK * 2) % num_threads == 0`` with num_threads=128. For T_q=8
      the kernel processes one m-block per ``(B, H)`` and predicates the
      unused 56 rows. Wasteful at T_q=8 but correct; tuning is a follow-up
      (smaller M_BLOCK with fewer threads).
    * ``N_BLOCK = 64`` -- k-tile width; small enough to fit even SM120's
      99 KB smem cap with double-buffering disabled.
    * ``num_threads = 128`` (4 warps).
    * Smem usage per CTA (no double-buffer):
        sQ + sK + sV = 3 * 64 * 64 * 2B = 24 KB
      Well within SM80 (163 KB) and SM120 (99 KB). Bias is read directly
      from gmem so no sBias allocation is needed.

    Subclasses override :meth:`can_implement` to plug in their per-arch
    smem-capacity check. The compute body is shared verbatim.
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
        m_block_size: int = 64,
        n_block_size: int = 64,
        num_threads: int = 128,
    ):
        super().__init__(
            head_dim=head_dim,
            dtype=dtype,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            has_bias=has_bias,
            paged=paged,
            block_size=block_size,
        )
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        # Pad head_dim up to a multiple of 32 (so the m16n8k16 MMA k-stride works).
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        if paged:
            # Each n-tile of N_BLOCK rows reads ``blocks_per_n_tile`` consecutive
            # logical block-IDs from the block table. Validated as a constexpr
            # below; the load loop in kernel() walks 0..blocks_per_n_tile.
            if block_size <= 0:
                raise ValueError(
                    f"{type(self).__name__}: paged=True requires block_size > 0"
                )
            if n_block_size % block_size != 0:
                raise ValueError(
                    f"{type(self).__name__}: paged kernel requires "
                    f"n_block_size ({n_block_size}) % block_size ({block_size}) == 0"
                )
            self._blocks_per_n_tile = n_block_size // block_size
        else:
            self._blocks_per_n_tile = 1

        import cutlass.pipeline as pipeline
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    # ------------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------------
    # Subclasses override ``_smem_capacity_in_bytes`` to plug in a different
    # arch-string (e.g. SM120's 99 KB cap). The body of ``can_implement``
    # otherwise mirrors the Dao-AILab flash_fwd_sm80 pattern.
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
            # Paged mode skips per-element head_dim predication for sub-tile
            # cp.async loads -- require head_dim to fall on the MMA k-stride
            # boundary so head_dim == head_dim_padded.
            if head_dim % 32 != 0:
                return False
        # Smem budget: sQ + 2 * sK + sV. Bias (when has_bias) is gmem-direct,
        # not staged through smem. The 2x sK is for the K cp.async ping-pong
        # buffer.
        del has_bias  # not part of the smem budget
        smem_bytes = (
            m_block_size * head_dim
            + n_block_size * head_dim * 3
        ) * 2  # fp16/bf16 = 2B
        if smem_bytes > cls._smem_capacity_in_bytes():
            return False
        return True

    # ------------------------------------------------------------------------
    # Host launcher (configures layouts/atoms, launches device kernel)
    # ------------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,                              # (B, H,    T_q, D)
        mK: cute.Tensor,                              # (B, H_kv, T_k, D) or paged
        mV: cute.Tensor,                              # (B, H_kv, T_k, D) or paged
        mO: cute.Tensor,                              # (B, H,    T_q, D)
        mBias: cute.Tensor,                           # (B, H, T_q, T_k) or zero-rank dummy
        mCacheSeqlens: cute.Tensor,                   # (B,) int32 or zero-rank dummy
        mBlockTable: cute.Tensor,                     # (B, max_blocks_per_seq) int32 (paged) or 0-rank dummy
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Type checks.
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

        # ---- Smem layouts (Q, K, V, optional bias) ---------------------------
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1),
        )
        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom, (self._n_block_size, self._head_dim_padded), (0, 1),
        )
        sO_layout = sQ_layout

        # NB: bias is read directly from gmem inside ``_add_bias_tile`` -- we
        # do *not* stage it through smem. Each bias element is read once, so
        # the gmem cost is identical, and skipping the smem path keeps the
        # SharedStorage struct (and the kernel signature) free of an
        # ``if has_bias`` branch that the @cute.kernel preprocessor used to
        # garble into a misleading "argument #18 (SharedStorage)" error.

        # 2-stage K pipelining: dual sK0/sK1 buffers so the next-K cp.async
        # can start *before* the current QK gemm finishes (writes to the
        # opposite stage's buffer, no race with the in-flight smem read).
        # Smem grows from 24 KB to 32 KB for D=64 (still fits SM80 163 KB
        # and SM120 99 KB caps).
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
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]

        # ---- GMEM tiled copy (cp.async for Q/K/V, universal for O) ----------
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        tO_layout = tQKV_layout
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # ---- Tiled MMA (m16n8k16 fp16/bf16) -----------------------------------
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # ---- Grid: (m_block, B, H) -------------------------------------------
        # OASR layout is (B, H, T_q, D); m-axis is T_q (mQ.shape[2]).
        grid_dim = (
            cute.ceil_div(mQ.shape[2], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[1]),
        )

        LOG2_E = 1.4426950408889634074
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

        # GQA: head index in K/V tensors.
        kv_head = num_head // self._gqa_ratio

        # ---- Logical KV extent ----------------------------------------------
        # Dense: T_kv_logical == mK.shape[2] (the contiguous T_k axis).
        # Paged: mK is (num_blocks, block_size, H_kv, D), so the logical kv
        # extent is determined by how many logical blocks are addressable per
        # stream (``mBlockTable.shape[1] * block_size``). The per-stream length
        # mask in softmax_rescale_O still narrows this to each stream's actual
        # ``cache_seqlens[b]``.
        if cutlass.const_expr(self._paged):
            t_kv_logical = mBlockTable.shape[1] * self._block_size
        else:
            t_kv_logical = mK.shape[2]

        # Number of K-tiles to process for this stream. With per-stream length
        # mask, every block walks the full K-range and masks dynamically.
        n_block_max = cute.ceil_div(t_kv_logical, self._n_block_size)
        n_block = n_block_max - 1

        # ---- Slice gmem tiles for this (b, h) --------------------------------
        # mQ/mO: (B, H, T_q, D). Slice on (b, h) -> (T_q, D).
        gQ = cute.local_tile(
            mQ[batch_size, num_head, None, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        # K/V slicing.
        # Dense: mK/mV is (B, H_kv, T_k, D). gK/gV is the per-CTA tile chain.
        # Paged: mK/mV is (num_blocks, block_size, H_kv, D). We do NOT build a
        # contiguous gK/gV view -- per n_block loads happen via
        # _load_kv_tile_paged which gathers from mBlockTable.
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

        # NB: bias is read directly from gmem inside ``_add_bias_tile`` -- no
        # gBias slice or sBias allocation is needed at the kernel scope.

        # ---- Smem allocation -------------------------------------------------
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK0 = storage.sK0.get_tensor(sKV_layout)
        sK1 = storage.sK1.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)

        # Transpose view of sV for tiled MMA (K-major).
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        # ---- Thread partitions -----------------------------------------------
        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        # Per-stage K smem destination partitions for cp.async.
        tKsK0 = gmem_thr_copy_QKV.partition_D(sK0)
        tKsK1 = gmem_thr_copy_QKV.partition_D(sK1)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)
        # gmem-side partitions only exist in dense mode -- paged mode loads
        # K/V via per-sub-tile cp.async issued from the block_table lookup.
        if cutlass.const_expr(not self._paged):
            tKgK = gmem_thr_copy_QKV.partition_S(gK)
            tVgV = gmem_thr_copy_QKV.partition_S(gV)
        else:
            tKgK = None
            tVgV = None


        # ---- MMA partitions and accumulators ---------------------------------
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        # Two MMA fragments for K, one per stage.
        tSrK0 = thr_mma.make_fragment_B(thr_mma.partition_B(sK0))
        tSrK1 = thr_mma.make_fragment_B(thr_mma.partition_B(sK1))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ---- Smem copy atoms (ldmatrix.x4) -----------------------------------
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype,
        )
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
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # ---- Predicates for head_dim & seqlen --------------------------------
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
        # Q head_dim predicates (idx[3] -> head_dim axis in (B,H,T,D)).
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )

        # K/V predicates only exist in dense mode. Paged mode constrains
        # head_dim % 32 == 0 (so head_dim == head_dim_padded -- no
        # head-axis predicate needed) and reads block_table-resolved
        # physical blocks unconditionally; stale tail data is masked out
        # by softmax_rescale_O via cache_seqlens.
        if cutlass.const_expr(not self._paged):
            mcKV = cute.make_identity_tensor(mK.layout.shape)
            cKV = cute.local_tile(
                mcKV[batch_size, kv_head, None, None],
                (self._n_block_size, self._head_dim_padded),
                (n_block, 0),
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

        # NB: per-stream length mask is applied inside softmax_rescale_O using
        # ``mCacheSeqlens`` directly (when rank > 0) or ``mK.shape[2]`` for the
        # tail-residue mask. We do *not* wrap either value in
        # ``cutlass.Int32(...)`` -- doing so converts a coord-type value into
        # a register value that ``cute.elem_less`` rejects with silent NaN
        # propagation through the masked rows.

        # ---- Prefetch Q + first K tile ---------------------------------------
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
        # Initial K prefetch goes into stage 0 (sK0). Subsequent prefetches
        # alternate stages, with each compute_one_n_block_pipelined call
        # issuing the *next* iteration's K into the opposite stage.
        if cutlass.const_expr(not self._paged):
            for n in cutlass.range_constexpr(cute.size(tKsK0.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][2], mK.layout.shape[2]):
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tKgK[None, n, None, n_block],
                        tKsK0[None, n, None],
                        pred=tKVpKV[None, n, None],
                    )
                else:
                    tKsK0[None, n, None].fill(0)
        else:
            self._paged_load_kv_tile(
                mK, sK0, mBlockTable, batch_size, kv_head, n_block,
                gmem_thr_copy_QKV, gmem_tiled_copy_QKV,
            )
        cute.arch.cp_async_commit_group()

        # ---- Online softmax state --------------------------------------------
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        basic_params = SimpleNamespace(
            m_block=m_block, n_block=n_block,
            mQ=mQ, mK=mK, mV=mV, mO=mO, mBias=mBias,
            mCacheSeqlens=mCacheSeqlens, mBlockTable=mBlockTable,
            t_kv_logical=t_kv_logical,
            batch_size=batch_size, num_head=num_head, kv_head=kv_head,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma, tiled_mma=tiled_mma,
            tSrQ=tSrQ, tSrK0=tSrK0, tSrK1=tSrK1,
            tOrVt=tOrVt, acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            gmem_thr_copy_QKV=gmem_thr_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK, tKsK0=tKsK0, tKsK1=tKsK1,
            sK0=sK0, sK1=sK1,
            tVgV=tVgV, tVsV=tVsV, sV=sV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ, tSrQ_copy_view=tSrQ_copy_view,
            tSsK0=tSsK0, tSsK1=tSsK1,
            tSrK0_copy_view=tSrK0_copy_view,
            tSrK1_copy_view=tSrK1_copy_view,
            tOsVt=tOsVt, tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max, row_sum=row_sum,
            softmax_scale=softmax_scale,
            softmax_scale_log2=softmax_scale_log2,
        )
        # ---- N-tile loop -----------------------------------------------------
        # The first iteration is peeled out so ``is_first_n_block`` is a
        # compile-time constant -- ``softmax_rescale_O`` uses ``const_expr`` on
        # it to skip the previous-row state copy on the first tile, and the JIT
        # path can only emit that branch if the value is constexpr.
        #
        # 2-stage K pipelining: the initial K prefetch (above) lands in stage 0
        # (sK0), so iter 0 reads stage 0 while issuing the next-K prefetch into
        # stage 1. Subsequent iterations alternate. We dispatch ``curr_stage``
        # as a constexpr by branching on the runtime ``n_tile`` parity inside
        # the loop -- both paths get JIT-compiled.
        basic_params.n_block = n_block_max - 1
        self.compute_one_n_block(
            basic_params, mma_params, gmem_copy_params, smem_copy_params,
            softmax_params,
            is_first_n_block=True,
            curr_stage=0,
        )
        for n_tile in range(1, n_block_max, 1):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            if (n_tile % 2) == 1:
                self.compute_one_n_block(
                    basic_params, mma_params, gmem_copy_params,
                    smem_copy_params, softmax_params,
                    is_first_n_block=False,
                    curr_stage=1,
                )
            else:
                self.compute_one_n_block(
                    basic_params, mma_params, gmem_copy_params,
                    smem_copy_params, softmax_params,
                    is_first_n_block=False,
                    curr_stage=0,
                )

        # ---- Epilogue: normalize, write back ---------------------------------
        self.normalize_softmax(acc_O, row_sum)
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
    # Inner per-N-tile compute (QK MMA + bias add + softmax + V MMA)
    # ------------------------------------------------------------------------
    # 2-stage K cp.async pipelining: ``curr_stage`` (constexpr 0/1) selects
    # which sK buffer holds the current K data; the next-K prefetch is issued
    # *before* the QK gemm into the opposite stage's buffer so cp.async
    # overlaps with the MMA. The host launcher alternates stages per
    # iteration (peeled iter 0 = stage 0, then 1, 0, 1, 0, ...).
    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        curr_stage: cutlass.Constexpr,
    ):
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # Pick per-stage K partitions.
        if cutlass.const_expr(curr_stage == 0):
            tSsK_curr = smem_copy_params.tSsK0
            tSrK_curr = mma_params.tSrK0
            tSrK_curr_copy_view = smem_copy_params.tSrK0_copy_view
            tKsK_next = gmem_copy_params.tKsK1
            sK_next = gmem_copy_params.sK1
        else:
            tSsK_curr = smem_copy_params.tSsK1
            tSrK_curr = mma_params.tSrK1
            tSrK_curr_copy_view = smem_copy_params.tSrK1_copy_view
            tKsK_next = gmem_copy_params.tKsK0
            sK_next = gmem_copy_params.sK0

        # Wait for K(curr) -- the cp.async issued in the previous iteration
        # (or the initial prefetch for iter 0). At this point only the K
        # group is pending so wait_group(0) drains everything.
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Stage V for this n_block; on the first iteration we predicate K-residue.
        if cutlass.const_expr(not self._paged):
            if is_first_n_block:
                for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                    if cute.elem_less(
                        gmem_copy_params.tKVcKV[0, n, 0][2],
                        basic_params.mK.layout.shape[2],
                    ):
                        cute.copy(
                            gmem_copy_params.gmem_tiled_copy_QKV,
                            gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                            gmem_copy_params.tVsV[None, n, None],
                            pred=gmem_copy_params.tKVpKV[None, n, None],
                        )
                    else:
                        gmem_copy_params.tVsV[None, n, None].fill(0.0)
            else:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                    gmem_copy_params.tVsV,
                    pred=gmem_copy_params.tKVpKV,
                )
        else:
            self._paged_load_kv_tile(
                basic_params.mV, gmem_copy_params.sV,
                basic_params.mBlockTable,
                basic_params.batch_size, basic_params.kv_head,
                basic_params.n_block,
                gmem_copy_params.gmem_thr_copy_QKV,
                gmem_copy_params.gmem_tiled_copy_QKV,
            )
        cute.arch.cp_async_commit_group()

        # Issue NEXT K prefetch BEFORE the QK gemm. Writes go to the opposite
        # stage's sK buffer (sK_next), so there's no race with the QK gemm
        # reading sK_curr. This is the FA2 2-stage pipelining trick: the
        # cp.async runs concurrently with the MMA below.
        if basic_params.n_block > 0:
            if cutlass.const_expr(not self._paged):
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                    tKsK_next,
                    pred=gmem_copy_params.tKVpKV,
                )
            else:
                self._paged_load_kv_tile(
                    basic_params.mK, sK_next,
                    basic_params.mBlockTable,
                    basic_params.batch_size, basic_params.kv_head,
                    basic_params.n_block - 1,
                    gmem_copy_params.gmem_thr_copy_QKV,
                    gmem_copy_params.gmem_tiled_copy_QKV,
                )
            cute.arch.cp_async_commit_group()

        # ---- QK GEMM (reads sK_curr, runs concurrently with cp.asyncs) ------
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            tSsK_curr[None, None, 0],
            tSrK_curr_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                tSsK_curr[None, None, k_next],
                tSrK_curr_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma, acc_S,
                mma_params.tSrQ[None, None, k],
                tSrK_curr[None, None, k], acc_S,
            )

        # Wait for V. If next-K was also issued, leave it in flight
        # (wait_group(1) keeps the K group pending for the next iter).
        # If not (last iter, n_block == 0), drain everything.
        if basic_params.n_block > 0:
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # ---- Bias add (per-tile, after QK MMA, before softmax) --------------
        # Kernel matches SDPA semantics: ``attn_bias`` is treated as a
        # post-scale logit. _add_bias_tile divides the bias by softmax_scale
        # so the eventual ``exp2((acc_S + bias_scaled) * softmax_scale_log2)``
        # collapses to ``exp(softmax_scale * acc_S + bias)``.
        if cutlass.const_expr(self._has_bias):
            self._add_bias_tile(basic_params, mma_params, softmax_params, acc_S)

        # ---- Online softmax + rescale acc_O ---------------------------------
        self.softmax_rescale_O(
            basic_params, mma_params, softmax_params, acc_S,
            is_first_n_block=is_first_n_block,
        )

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))

        # Layout transform: (4, MMA_M, MMA_N) -> ((4,2), MMA_M, MMA_N/2) for SV gemm.
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

        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma, mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k], mma_params.acc_O,
            )

    # ------------------------------------------------------------------------
    # Paged K/V load: gather ``blocks_per_n_tile`` consecutive logical
    # block-IDs from the block table and issue per-sub-block cp.async copies
    # into ``sKV``. ``head_dim % 32 == 0`` is enforced by ``can_implement``
    # so head_dim == head_dim_padded and no per-element head-axis predicate
    # is required.
    # ------------------------------------------------------------------------
    @cute.jit
    def _paged_load_kv_tile(
        self,
        mKV: cute.Tensor,             # (num_blocks, block_size, H_kv, D)
        sKV: cute.Tensor,             # (N_BLOCK, head_dim_padded) smem
        mBlockTable: cute.Tensor,     # (B, max_blocks_per_seq) int32
        batch_size: cutlass.Int32,
        kv_head: cutlass.Int32,
        n_block: cutlass.Int32,
        gmem_thr_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_QKV: cute.TiledCopy,
    ):
        block_size = self._block_size
        head_dim_padded = self._head_dim_padded
        n_logical_base = n_block * self._blocks_per_n_tile
        for i in cutlass.range_constexpr(self._blocks_per_n_tile):
            phys = mBlockTable[batch_size, n_logical_base + i]
            # Slice (block_size, head_dim) view of one physical block for this
            # KV head. With head_dim % 32 == 0 the gmem tile shape exactly
            # matches the smem sub-tile so no head_dim predicate is needed.
            gKV_block = cute.local_tile(
                mKV[phys, None, kv_head, None],
                (block_size, head_dim_padded),
                (0, 0),
            )
            sKV_sub = cute.local_tile(
                sKV,
                (block_size, head_dim_padded),
                (i, 0),
            )
            tKVgKV = gmem_thr_copy_QKV.partition_S(gKV_block)
            tKVsKV = gmem_thr_copy_QKV.partition_D(sKV_sub)
            cute.copy(gmem_tiled_copy_QKV, tKVgKV, tKVsKV)

    # ------------------------------------------------------------------------
    # Bias tile: read gmem bias[b, h, q_row, k_col] and add into acc_S.
    # ------------------------------------------------------------------------
    # We do a direct gmem read per accumulator element instead of staging
    # bias through smem first. Each element is read once, so the gmem read
    # cost is identical -- the smem path is purely a latency-hiding tool and
    # a future optimization. The simple form makes the (B,H,T_q,T_k) coord
    # math obvious and dodges the partition_S-into-smem boilerplate.
    @cute.jit
    def _add_bias_tile(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
    ):
        # Partition the (M_BLOCK, N_BLOCK) bias tile through the same MMA
        # accumulator layout as acc_S so element (r, c) on each thread maps
        # to the same global (q_row, k_col) for both bias and acc_S.
        gBias = cute.local_tile(
            basic_params.mBias[
                basic_params.batch_size, basic_params.num_head, None, None
            ],
            (self._m_block_size, self._n_block_size),
            (basic_params.m_block, basic_params.n_block),
        )
        tBias = mma_params.thr_mma.partition_C(gBias)
        tBias_mn = self._make_acc_tensor_mn_view(tBias)
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)

        # Per-tile (q_row, k_col) coords for bounds checking. Out-of-bounds
        # entries (q_row >= T_q or k_col >= T_kv_logical) are skipped -- the
        # softmax masking path will set these acc_S slots to -inf anyway.
        # mBias.shape[3] is the logical kv extent in both modes (caller
        # always sizes it that way), so it is the correct bound regardless
        # of whether mK is dense or paged.
        mcS = cute.make_identity_tensor(
            (basic_params.mQ.shape[0],
             basic_params.mQ.shape[1],
             basic_params.mQ.shape[2],
             basic_params.mBias.shape[3])
        )
        cS = cute.local_tile(
            mcS[basic_params.batch_size, basic_params.num_head, None, None],
            (self._m_block_size, self._n_block_size),
            (basic_params.m_block, basic_params.n_block),
        )
        tScS = mma_params.thr_mma.partition_C(cS)
        tScS_mn = self._make_acc_tensor_mn_view(tScS)

        # The kernel applies ``softmax_scale`` *inside* exp2 (via
        # ``softmax_scale_log2``), so what reaches the user as the post-scale
        # logit is ``softmax_scale * (acc_S + bias)`` -- but SDPA semantics
        # treat ``attn_bias`` as added AFTER scaling. Pre-divide bias by
        # ``softmax_scale`` here so the math collapses to
        # ``softmax_scale * acc_S + bias`` once exp2 is applied.
        inv_scale = 1.0 / softmax_params.softmax_scale
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            row_in_bounds = cute.elem_less(
                tScS_mn[r, 0][2], basic_params.mQ.shape[2]
            )
            if row_in_bounds:
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    if cute.elem_less(
                        tScS_mn[r, c][3], basic_params.mBias.shape[3]
                    ):
                        acc_S_mn[r, c] = (
                            acc_S_mn[r, c]
                            + cutlass.Float32(tBias_mn[r, c]) * inv_scale
                        )

    # ------------------------------------------------------------------------
    # Online softmax + rescale of acc_O
    # ------------------------------------------------------------------------
    @cute.jit
    def softmax_rescale_O(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
    ):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(
                softmax_params.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_params.row_max, row_max_prev)

        # Per-tile (q_row, k_col) coords for length masking. The kv axis
        # uses the *logical* kv extent (``t_kv_logical``) so that
        # ``tScS_mn[..., 3]`` is the logical k position regardless of
        # whether K/V live in a dense tensor or a paged pool.
        mcS = cute.make_identity_tensor(
            (basic_params.mQ.shape[0],   # B
             basic_params.mQ.shape[1],   # H
             basic_params.mQ.shape[2],   # T_q
             basic_params.t_kv_logical)  # logical T_kv
        )
        cS = cute.local_tile(
            mcS[basic_params.batch_size, basic_params.num_head, None, None],
            (self._m_block_size, self._n_block_size),
            (basic_params.m_block, basic_params.n_block),
        )
        tScS = mma_params.thr_mma.partition_C(cS)
        tScS_mn = self._make_acc_tensor_mn_view(tScS)

        # Per-stream length mask. Always-on: the column compare also serves
        # as the T_k tail-residue mask (in offline mode, ``mCacheSeqlens``
        # is filled with T_k so columns >= T_k get masked to -inf).
        # NB: ``cute.elem_less`` rejects register values like
        # ``cutlass.Int32(...)`` -- pass coord-type values directly.
        cache_len = basic_params.mCacheSeqlens[basic_params.batch_size]
        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            # tScS_mn[0, c][3] is the absolute k-column index in T_k.
            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                if cute.elem_less(cache_len, tScS_mn[0, c][3] + 1):
                    acc_S_mn[r, c] = -cutlass.Float32.inf
            # Mask q-rows beyond T_q (m-residue padding).
            if cute.elem_less(
                basic_params.mQ.shape[2], tScS_mn[r, 0][2] + 1
            ):
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            # If every column in the tile was masked out (per-stream length
            # mask + tail-residue mask), row_max is -inf -> exp(-inf - -inf)
            # is NaN. Replace -inf with 0; exp(-inf - 0) = 0 still gives no
            # contribution and row_sum stays 0, but downstream arithmetic
            # avoids the NaN. The reference applies the same clamp for the
            # causal-mask path.
            row_max_cur_row = (
                0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
            )

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                )
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(self, acc_O: cute.Tensor, row_sum: cute.Tensor):
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = 1.0 if is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    # ------------------------------------------------------------------------
    # Helpers (mn-view, warp-shuffle reductions)
    # ------------------------------------------------------------------------
    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val, op):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)
