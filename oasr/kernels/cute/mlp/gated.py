# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused gated MLP (SwiGLU / GeGLU) as a CuTeDSL dual-B tensor-core GEMM.

The two thirds of a gated feed-forward block that can share a CTA, in one launch::

    out[m, n] = activation(x[m, :] @ w_gate[n, :] + b_gate[n])
                         * (x[m, :] @ w_up[n, :]   + b_up[n])

``x`` is ``(M, K)``, both weights are ``(N, K)`` -- ``nn.Linear``'s own layout, so
a checkpoint is read where it lies -- and the output is ``(M, N)``.
"""

# PEP 563 (deferred annotations) breaks CuteDSL Constexpr detection;
# do not enable.

from functools import lru_cache
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cute.nvgpu import warp

from oasr.kernels.cute.ampere_helpers import gemm_dual_with_smem_prefetch
from oasr.kernels.cute.copy_utils import (
    async_copy_elements,
    make_cp_async_atom,
    make_qkv_tiled_copy,
    make_universal_copy_atom,
)
from oasr.kernels.cute.layout_utils import make_smem_swizzle_atom

__all__ = ["GatedMlpCute", "ACTIVATIONS"]

#: Gate activations this kernel implements, by the name ``oasr.layers`` uses.
#: ``gelu`` is the exact-erf form and ``gelu_tanh`` the tanh approximation --
#: they stay separate names here for the same reason they do in the waist: they
#: are numerically different epilogues and a checkpoint means one of them.
ACTIVATIONS = ("silu", "swish", "relu", "gelu", "gelu_tanh", "identity")

#: The driver reserves a slice of every block's shared memory (1 KB on the parts
#: measured here).  A tile sized to the raw opt-in maximum fails at *launch*, so
#: budget against the opt-in limit minus this.  Same reserve as the recurrent
#: step; see ``oasr/kernels/cute/recurrent/step.py``.
_DRIVER_SMEM_RESERVE = 1024

#: Conservative floor for hosts where the attribute query fails.
_FALLBACK_SMEM_CAPACITY = 64 * 1024


@lru_cache(maxsize=None)
def smem_capacity(device_index: int = 0) -> int:
    """Dynamic shared memory a single block can actually be given, in bytes."""
    try:
        import ctypes

        # cudaDevAttrMaxSharedMemoryPerBlockOptin
        value = ctypes.c_int()
        if (
            ctypes.CDLL("libcudart.so").cudaDeviceGetAttribute(
                ctypes.byref(value), 97, device_index
            )
            != 0
            or value.value <= 0
        ):
            return _FALLBACK_SMEM_CAPACITY
        return value.value - _DRIVER_SMEM_RESERVE
    except Exception:
        return _FALLBACK_SMEM_CAPACITY


def _swizzle_width(extent: int) -> int:
    """Inner width of the smem swizzle atom :func:`make_smem_swizzle_atom` picks."""
    return 64 if extent % 64 == 0 else 32


class GatedMlpCute:
    """``activation(x @ w_gateᵀ + b_gate) * (x @ w_upᵀ + b_up)``, tensor-core.

    Parameters
    ----------
    dtype :
        ``cutlass.Float16`` or ``cutlass.BFloat16``.  The accumulator is FP32.
    activation : str
        One of :data:`ACTIVATIONS`, applied to the **gate** half only.
    has_bias : bool
        Compiled in.  With ``False`` the bias tensors are never dereferenced, so
        a caller with no bias passes a one-element dummy rather than a zero
        vector it would otherwise stream once per CTA.
    m_block, n_block, k_block : int
        CTA tile over ``(M, N, K)``.
    num_stages : int
        Depth of the cp.async ring over the K loop.  The ring carries A **and
        both** Bs, so a stage costs ``(m_block + 2 * n_block) * k_block``
        elements -- half again what a plain GEMM's does, which is the tuning
        pressure that makes ``k_block = 32`` a real option here.
    num_threads : int
        CTA size; a multiple of 32.
    warps_n : int
        How many of the warps tile N rather than M.  A decode-shaped call has
        ``M`` in single digits and ``N`` in the tens of thousands, so spending
        every warp on M would leave the N axis to one warp.
    """

    def __init__(
        self,
        *,
        dtype,
        activation: str = "silu",
        has_bias: bool = False,
        m_block: int = 64,
        n_block: int = 64,
        k_block: int = 64,
        num_stages: int = 3,
        num_threads: int = 128,
        warps_n: int = 1,
    ) -> None:
        self._dtype = dtype
        self._activation = activation
        self._has_bias = bool(has_bias)
        self._m_block = int(m_block)
        self._n_block = int(n_block)
        self._k_block = int(k_block)
        self._num_stages = int(num_stages)
        self._num_threads = int(num_threads)
        self._warps_n = int(warps_n)
        self._warps_m = self._num_threads // 32 // self._warps_n
        if not self.can_implement(
            dtype=dtype,
            activation=activation,
            has_bias=has_bias,
            m_block=m_block,
            n_block=n_block,
            k_block=k_block,
            num_stages=num_stages,
            num_threads=num_threads,
            warps_n=warps_n,
        ):
            raise ValueError(
                f"unsupported GatedMlpCute config: activation={activation!r} "
                f"tile=({m_block},{n_block},{k_block}) stages={num_stages} "
                f"threads={num_threads} warps_n={warps_n}"
            )

    # ------------------------------------------------------------------
    # Static contract
    # ------------------------------------------------------------------

    @staticmethod
    def can_implement(
        *,
        dtype,
        activation: str = "silu",
        has_bias: bool = False,
        m_block: int = 64,
        n_block: int = 64,
        k_block: int = 64,
        num_stages: int = 3,
        num_threads: int = 128,
        warps_n: int = 1,
        capacity: Optional[int] = None,
    ) -> bool:
        """Would this configuration compile, fit, and address only its own tiles?

        Refuses rather than degrading: a routing table that asks for an
        impossible tile finds out at compile time, not by reading out of bounds.
        """
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if activation not in ACTIVATIONS:
            return False
        if num_threads % 32 or num_threads < 32 or num_threads > 1024:
            return False
        num_warps = num_threads // 32
        if warps_n < 1 or num_warps % warps_n:
            return False
        warps_m = num_warps // warps_n
        # The MMA is tiled (warps_m, warps_n) over a 16x16 permutation of the
        # m16n8k16 atom, so each axis must cover a whole number of those.
        if m_block % (16 * warps_m) or n_block % (16 * warps_n):
            return False
        if k_block % 32 or num_stages < 2:
            return False
        # gmem -> smem: the thread layout walks `num_threads / (atom_w / vec)`
        # rows per pass.  A pass wider than the tile addresses outside it -- an
        # illegal access, not a predicated no-op, because the *partition* is out
        # of range.  Same trap as the recurrent step's `rows_per_pass` check.
        vec = 128 // dtype.width
        k_atom_w = _swizzle_width(k_block)
        rows_per_pass = num_threads // (k_atom_w // vec)
        if rows_per_pass == 0 or m_block % rows_per_pass or n_block % rows_per_pass:
            return False
        # ...and the same again for the epilogue, whose atom is sized on N.
        n_atom_w = _swizzle_width(n_block)
        o_rows_per_pass = num_threads // (n_atom_w // vec)
        if o_rows_per_pass == 0 or m_block % o_rows_per_pass:
            return False
        elem_bytes = dtype.width // 8
        ring = num_stages * (m_block + 2 * n_block) * k_block * elem_bytes
        # The epilogue stages the rounded result over the ring's memory, so it
        # has to fit there -- aliasing a larger tensor over a smaller allocation
        # would corrupt whatever the allocator handed out next.
        if m_block * n_block * elem_bytes > ring:
            return False
        return ring <= (smem_capacity() if capacity is None else capacity)

    @staticmethod
    def smem_bytes(*, dtype, m_block: int, n_block: int, k_block: int, num_stages: int) -> int:
        """Shared memory the kernel will request, in bytes."""
        elem_bytes = dtype.width // 8
        return num_stages * (m_block + 2 * n_block) * k_block * elem_bytes

    # ------------------------------------------------------------------
    # Device-side activation
    # ------------------------------------------------------------------

    @cute.jit
    def _activate(self, x: cutlass.Float32) -> cutlass.Float32:
        """The gate activation, in FP32, matching ``include/oasr/common/math.h``."""
        if cutlass.const_expr(self._activation in ("silu", "swish")):
            return x / (1.0 + cute.math.exp(-x, fastmath=True))
        elif cutlass.const_expr(self._activation == "relu"):
            return cute.math.max(x, cutlass.Float32(0.0))
        elif cutlass.const_expr(self._activation == "gelu_tanh"):
            inner = 0.7978845608 * (x + 0.044715 * x * x * x)
            return 0.5 * x * (1.0 + cute.math.tanh(inner, fastmath=True))
        elif cutlass.const_expr(self._activation == "gelu"):
            return 0.5 * x * (1.0 + cute.math.erf(x * 0.70710678118654752440))
        else:
            return x

    # ------------------------------------------------------------------
    # Host entry
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, K)
        mWg: cute.Tensor,  # (N, K)
        mWu: cute.Tensor,  # (N, K)
        mBg: cute.Tensor,  # (N,)  -- dummy when has_bias is False
        mBu: cute.Tensor,  # (N,)  -- dummy when has_bias is False
        mOut: cute.Tensor,  # (M, N)
        stream: cuda.CUstream,
    ):
        elem = self._dtype
        async_elems = async_copy_elements(elem)

        # Swizzled smem so the ldmatrix that follows is bank-conflict free.  The
        # atom is chosen from the K tile because K is the contiguous axis of X
        # (row-major) and of both weights (``nn.Linear``'s (out, in) layout).
        atom_ab, smem_k_block = make_smem_swizzle_atom(elem, self._k_block)
        sX_layout = cute.tile_to_shape(
            atom_ab, (self._m_block, self._k_block, self._num_stages), (0, 1, 2)
        )
        sW_layout = cute.tile_to_shape(
            atom_ab, (self._n_block, self._k_block, self._num_stages), (0, 1, 2)
        )
        # The epilogue's own atom is sized on N, the output's contiguous axis.
        atom_o, smem_n_block = make_smem_swizzle_atom(elem, self._n_block)
        sO_layout = cute.tile_to_shape(atom_o, (self._m_block, self._n_block), (0, 1))

        atom_cp = make_cp_async_atom(elem)
        gmem_tiled_copy_X = make_qkv_tiled_copy(
            atom_cp, self._num_threads, smem_k_block, async_elems
        )
        gmem_tiled_copy_W = gmem_tiled_copy_X
        gmem_tiled_copy_O = make_qkv_tiled_copy(
            make_universal_copy_atom(elem), self._num_threads, smem_n_block, async_elems
        )

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(elem, cutlass.Float32, (16, 8, 16)),
            (self._warps_m, self._warps_n, 1),
            permutation_mnk=(self._warps_m * 16, self._warps_n * 16, 16),
        )

        @cute.struct
        class SharedStorage:
            sX: cute.struct.Align[cute.struct.MemRange[elem, cute.cosize(sX_layout)], 1024]
            sWg: cute.struct.Align[cute.struct.MemRange[elem, cute.cosize(sW_layout)], 1024]
            sWu: cute.struct.Align[cute.struct.MemRange[elem, cute.cosize(sW_layout)], 1024]

        grid = (
            cute.ceil_div(cute.size(mOut.shape[1]), self._n_block),
            cute.ceil_div(cute.size(mX.shape[0]), self._m_block),
            1,
        )
        self.kernel(
            mX,
            mWg,
            mWu,
            mBg,
            mBu,
            mOut,
            sX_layout,
            sW_layout,
            sO_layout,
            gmem_tiled_copy_X,
            gmem_tiled_copy_W,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid,
            block=(self._num_threads, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mWg: cute.Tensor,
        mWu: cute.Tensor,
        mBg: cute.Tensor,
        mBu: cute.Tensor,
        mOut: cute.Tensor,
        sX_layout: cute.ComposedLayout,
        sW_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_X: cute.TiledCopy,
        gmem_tiled_copy_W: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        n_block, m_block, _ = cute.arch.block_idx()

        M = cute.size(mX.shape[0])
        K = cute.size(mX.shape[1])
        N = cute.size(mOut.shape[1])

        # ---- CTA tiles ------------------------------------------------------
        gX = cute.local_tile(mX, (self._m_block, self._k_block), (m_block, None))
        gWg = cute.local_tile(mWg, (self._n_block, self._k_block), (n_block, None))
        gWu = cute.local_tile(mWu, (self._n_block, self._k_block), (n_block, None))

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sX = storage.sX.get_tensor(sX_layout)
        sWg = storage.sWg.get_tensor(sW_layout)
        sWu = storage.sWu.get_tensor(sW_layout)

        gmem_thr_copy_X = gmem_tiled_copy_X.get_slice(tidx)
        gmem_thr_copy_W = gmem_tiled_copy_W.get_slice(tidx)
        tXgX = gmem_thr_copy_X.partition_S(gX)
        tXsX = gmem_thr_copy_X.partition_D(sX)
        tGgG = gmem_thr_copy_W.partition_S(gWg)
        tGsG = gmem_thr_copy_W.partition_D(sWg)
        tUgU = gmem_thr_copy_W.partition_S(gWu)
        tUsU = gmem_thr_copy_W.partition_D(sWu)

        # Row predicates: M and N are runtime extents and need not be tile
        # multiples.  K is looped exactly, so only the row axis is predicated.
        cX = cute.local_tile(
            cute.make_identity_tensor(mX.layout.shape),
            (self._m_block, self._k_block),
            (m_block, 0),
        )
        cW = cute.local_tile(
            cute.make_identity_tensor(mWg.layout.shape),
            (self._n_block, self._k_block),
            (n_block, 0),
        )
        tXcX = gmem_thr_copy_X.partition_S(cX)
        tWcW = gmem_thr_copy_W.partition_S(cW)
        predX = cute.make_rmem_tensor(
            cute.make_layout((cute.size(tXsX.shape[1]),)), cutlass.Boolean
        )
        predW = cute.make_rmem_tensor(
            cute.make_layout((cute.size(tGsG.shape[1]),)), cutlass.Boolean
        )
        for i in cutlass.range_constexpr(cute.size(predX)):
            predX[i] = tXcX[0, i, 0][0] < M
        for i in cutlass.range_constexpr(cute.size(predW)):
            predW[i] = tWcW[0, i, 0][0] < N

        # ---- MMA fragments --------------------------------------------------
        thr_mma = tiled_mma.get_slice(tidx)
        tCrX = thr_mma.make_fragment_A(thr_mma.partition_A(sX[None, None, 0]))
        tCrG = thr_mma.make_fragment_B(thr_mma.partition_B(sWg[None, None, 0]))
        tCrU = thr_mma.make_fragment_B(thr_mma.partition_B(sWu[None, None, 0]))
        acc_g = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block, self._n_block)), cutlass.Float32
        )
        acc_u = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block, self._n_block)), cutlass.Float32
        )
        acc_g.fill(0.0)
        acc_u.fill(0.0)

        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
        tCsX = smem_thr_copy_A.partition_S(sX)
        tCsG = smem_thr_copy_B.partition_S(sWg)
        tCsU = smem_thr_copy_B.partition_S(sWu)
        tCrX_view = smem_thr_copy_A.retile(tCrX)
        tCrG_view = smem_thr_copy_B.retile(tCrG)
        tCrU_view = smem_thr_copy_B.retile(tCrU)

        # ---- cp.async prologue ----------------------------------------------
        k_tiles = cute.ceil_div(K, self._k_block)
        stages = cutlass.const_expr(self._num_stages)
        for s in cutlass.range_constexpr(stages - 1):
            if s < k_tiles:
                for i in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
                    if predX[i]:
                        cute.copy(gmem_tiled_copy_X, tXgX[None, i, None, s], tXsX[None, i, None, s])
                for i in cutlass.range_constexpr(cute.size(tGsG.shape[1])):
                    if predW[i]:
                        cute.copy(gmem_tiled_copy_W, tGgG[None, i, None, s], tGsG[None, i, None, s])
                        cute.copy(gmem_tiled_copy_W, tUgU[None, i, None, s], tUsU[None, i, None, s])
            cute.arch.cp_async_commit_group()

        # ---- Mainloop -------------------------------------------------------
        # One committed group per K tile, so waiting on ``stages - 2`` leaves the
        # deepest in-flight group still landing while this tile is consumed.
        read_stage = cutlass.Int32(0)
        write_stage = cutlass.Int32(stages - 1)
        for tile in cutlass.range(k_tiles, unroll=1):
            cute.arch.cp_async_wait_group(stages - 2)
            cute.arch.barrier()

            gemm_dual_with_smem_prefetch(
                tiled_mma,
                acc_g,
                acc_u,
                tCrX,
                tCrG,
                tCrU,
                smem_tiled_copy_A,
                smem_tiled_copy_B,
                tCsX[None, None, None, read_stage],
                tCsG[None, None, None, read_stage],
                tCsU[None, None, None, read_stage],
                tCrX_view,
                tCrG_view,
                tCrU_view,
            )

            # Refill the stage just consumed with the tile ``stages - 1`` ahead.
            next_tile = tile + stages - 1
            cute.arch.barrier()
            if next_tile < k_tiles:
                for i in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
                    if predX[i]:
                        cute.copy(
                            gmem_tiled_copy_X,
                            tXgX[None, i, None, next_tile],
                            tXsX[None, i, None, write_stage],
                        )
                for i in cutlass.range_constexpr(cute.size(tGsG.shape[1])):
                    if predW[i]:
                        cute.copy(
                            gmem_tiled_copy_W,
                            tGgG[None, i, None, next_tile],
                            tGsG[None, i, None, write_stage],
                        )
                        cute.copy(
                            gmem_tiled_copy_W,
                            tUgU[None, i, None, next_tile],
                            tUsU[None, i, None, write_stage],
                        )
            cute.arch.cp_async_commit_group()
            read_stage = (read_stage + 1) % stages
            write_stage = (write_stage + 1) % stages

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ---- Epilogue -------------------------------------------------------
        # bias -> activation -> gate multiply, all FP32, one rounding at the end.
        n0 = n_block * self._n_block
        tCcO = thr_mma.partition_C(cute.make_identity_tensor((self._m_block, self._n_block)))
        rO = cute.make_rmem_tensor(acc_g.layout, self._dtype)
        for i in cutlass.range_constexpr(cute.size(acc_g)):
            g = acc_g[i]
            u = acc_u[i]
            if cutlass.const_expr(self._has_bias):
                col = n0 + tCcO[i][1]
                if col < N:
                    g = g + mBg[col].to(cutlass.Float32)
                    u = u + mBu[col].to(cutlass.Float32)
            rO[i] = (self._activate(g) * u).to(self._dtype)

        # Stage through smem so the gmem store is 128-bit and coalesced: the
        # m16n8k16 C layout gives a thread two adjacent columns, which on its own
        # would write the tile in 4-byte pieces.
        sO = cute.make_tensor(sX.iterator, sO_layout)
        smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype)
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        cute.copy(smem_copy_atom_O, smem_thr_copy_O.retile(rO), smem_thr_copy_O.partition_D(sO))

        gO = cute.local_tile(mOut, (self._m_block, self._n_block), (m_block, n_block))
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)

        cute.arch.barrier()
        cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

        cO = cute.local_tile(
            cute.make_identity_tensor(mOut.layout.shape),
            (self._m_block, self._n_block),
            (m_block, n_block),
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
                tOpO[rest_v, 0, rest_n] = tOcO[(0, rest_v), 0, rest_n][1] < N
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if tOcO[0, rest_m, 0][0] < M:
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )
