# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused recurrent step as a CuTeDSL tensor-core GEMM.

One LSTM or vanilla-RNN timestep, in one launch:

    gates[m, n] = sum_k previous_h[m, k] * weight_hh[n, k] + input_gates[m, n]
    c[m, i]     = sigmoid(g1) * previous_c[m, i] + sigmoid(g0) * tanh(g2)
    h[m, i]     = sigmoid(g3) * tanh(c[m, i])

with ``N = gates * hidden`` and the gate dimension *interleaved*, so column ``n``
is ``(hidden n // gates, gate n % gates)``.  The interleaving is what keeps the
epilogue inside one CTA tile: a whole hidden unit's gates are adjacent columns,
so no cross-tile reduction is needed to apply the nonlinearity.

Why this exists alongside `include/oasr/recurrent/recurrent.cuh`
----------------------------------------------------------------
The scalar cohort kernel there splits K across a warp's lanes, which is the right
decomposition when the batch is small: at B=16, H=640 one step is 52 MFLOP against
3.28 MB of weights -- 16 FLOP/byte, under the ~18 FLOP/byte non-MMA balance point
-- so the step is memory bound before tensor cores are even considered, and MMA
measured 1.4-2.4x *slower* there because forming tiles costs the CTA parallelism
that was hiding the latency.

This kernel targets the other end: batches where MMA's ceiling is reachable, and
where the shipped alternative is the CUTLASS 2.x path that materializes an
intermediate gate tensor and pays a second launch for the state transition.  This
one fuses both away.  Routing between the three lives in `oasr/jit/recurrent_cute.py`
and is measured, not assumed.

Structure
---------
Ampere-style throughout: a ``num_stages`` cp.async ring on A and B, swizzled
shared memory, ``ldmatrix.x4`` into MMA fragments, warp-level ``mma.sync``
m16n8k16, and the accumulator staged back through shared memory so the epilogue
can gather a hidden unit's gates regardless of the MMA thread-value layout.  That
composition is portable from SM80 through SM120 -- GeForce Blackwell's own
CuTeDSL GEMM uses the same ``cute.nvgpu.warp.MmaF16BF16Op``, because SM120 has no
FP16 tcgen05 path.  See the module docstring in ``__init__.py`` for what is
deliberately *not* here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cute.nvgpu import warp

from oasr.kernels.cute.ampere_helpers import gemm_with_smem_prefetch
from oasr.kernels.cute.copy_utils import (
    async_copy_elements,
    make_cp_async_atom,
    make_qkv_tiled_copy,
)
from oasr.kernels.cute.layout_utils import make_smem_swizzle_atom

__all__ = ["RecurrentStepCute"]

#: Gate order matches PyTorch/cuDNN: input, forget, cell, output.
LSTM_GATES = 4
RNN_GATES = 1

#: The driver takes a slice of every block's shared memory for itself -- 1 KB on
#: the parts measured here, visible as "Driver Shared Memory Per Block" in ncu.
#: A tile sized to the raw opt-in maximum therefore fails at *launch*, which is
#: how `num_stages=5` configs got past `can_implement` and then died with an
#: empty error.  Budget against the opt-in limit minus this.
_DRIVER_SMEM_RESERVE = 1024

#: Conservative floor for hosts where the attribute query fails: SM80 and later
#: all offer at least this much opt-in shared memory per block.
_FALLBACK_SMEM_CAPACITY = 64 * 1024


@lru_cache(maxsize=None)
def smem_capacity(device_index: int = 0) -> int:
    """Dynamic shared memory a single block can actually be given, in bytes.

    Queried rather than assumed: the opt-in maximum is 101376 on SM120, 166912 on
    SM90, and 101376 on SM80/86/89, so a constant would either refuse tiles that
    fit or accept tiles that do not launch.
    """
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


class RecurrentStepCute:
    """One fused recurrent timestep, tensor-core.

    Parameters
    ----------
    dtype :
        ``cutlass.Float16`` or ``cutlass.BFloat16``.  The accumulator is always
        FP32, which is what makes a 640-term dot product safe in half precision.
    gate_count : int
        4 for LSTM, 1 for a vanilla RNN.
    activation : str
        ``"lstm"``, ``"tanh"`` or ``"relu"``.  Only the RNN forms read this.
    m_block, n_block, k_block : int
        CTA tile.  ``n_block`` must be a multiple of ``8 * gate_count`` so a tile
        holds whole hidden units and the epilogue never straddles a tile edge.
    num_stages : int
        Depth of the cp.async ring over the K loop.
    num_threads : int
        CTA size; must be a multiple of 32 and divide ``m_block`` into warps.
    """

    def __init__(
        self,
        *,
        dtype,
        gate_count: int,
        activation: str = "lstm",
        m_block: int = 64,
        n_block: int = 64,
        k_block: int = 64,
        num_stages: int = 3,
        num_threads: int = 128,
        warps_n: int = 1,
    ) -> None:
        self._dtype = dtype
        self._gate_count = int(gate_count)
        self._activation = activation
        self._m_block = int(m_block)
        self._n_block = int(n_block)
        self._k_block = int(k_block)
        self._num_stages = int(num_stages)
        self._num_threads = int(num_threads)
        self._warps_n = int(warps_n)
        self._warps_m = self._num_threads // 32 // self._warps_n
        if not self.can_implement(
            dtype=dtype,
            gate_count=gate_count,
            activation=activation,
            m_block=m_block,
            n_block=n_block,
            k_block=k_block,
            num_stages=num_stages,
            num_threads=num_threads,
            warps_n=warps_n,
        ):
            raise ValueError(
                f"unsupported RecurrentStepCute config: gates={gate_count} "
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
        gate_count: int,
        activation: str = "lstm",
        m_block: int = 64,
        n_block: int = 64,
        k_block: int = 64,
        num_stages: int = 3,
        num_threads: int = 128,
        warps_n: int = 1,
        capacity: Optional[int] = None,
    ) -> bool:
        """Would this configuration compile and fit?

        Refuses rather than silently degrading, so a routing table that asks for
        an impossible tile finds out at compile time instead of producing a
        kernel that runs but reads out of bounds.
        """
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if gate_count not in (LSTM_GATES, RNN_GATES):
            return False
        if activation not in ("lstm", "tanh", "relu"):
            return False
        if (activation == "lstm") != (gate_count == LSTM_GATES):
            return False
        if num_threads % 32 or num_threads < 32 or num_threads > 512:
            return False
        num_warps = num_threads // 32
        if warps_n < 1 or num_warps % warps_n:
            return False
        warps_m = num_warps // warps_n
        # The MMA is tiled (warps_m, warps_n) over a 16x16 permutation of the
        # m16n8k16 atom, so each axis must cover a whole number of those.
        if m_block % (16 * warps_m) or n_block % (16 * warps_n):
            return False
        # A tile must hold whole hidden units, and the MMA atom is 8 wide.
        if n_block % (8 * gate_count) or n_block % 8:
            return False
        if k_block % 32 or num_stages < 2:
            return False
        # The gmem->smem thread layout walks `num_threads * async / k_block` rows
        # per pass.  If that exceeds a tile's row extent the surplus threads
        # address outside the tile -- an illegal access, not a predicated no-op,
        # because the copy partition itself is out of range.
        async_elems = 128 // dtype.width
        rows_per_pass = num_threads * async_elems // k_block
        if rows_per_pass == 0:
            return False
        if m_block % rows_per_pass or n_block % rows_per_pass:
            return False
        # A + B ring, plus the epilogue accumulator staging that reuses neither.
        elem_bytes = dtype.width // 8
        ring = num_stages * (m_block + n_block) * k_block * elem_bytes
        # The epilogue stages the FP32 accumulator into the A/B ring's memory, so
        # it has to fit there -- aliasing a larger tensor over a smaller
        # allocation would corrupt whatever the allocator handed out next.
        stage_c = m_block * (n_block + 1) * 4
        if stage_c > ring:
            return False
        return ring <= (smem_capacity() if capacity is None else capacity)

    @staticmethod
    def smem_bytes(*, dtype, m_block: int, n_block: int, k_block: int, num_stages: int) -> int:
        """Shared memory the kernel will request, in bytes."""
        elem_bytes = dtype.width // 8
        return num_stages * (m_block + n_block) * k_block * elem_bytes

    # ------------------------------------------------------------------
    # Host entry
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # previous_h      (M, K)
        mB: cute.Tensor,  # weight_hh       (N, K), gate-interleaved rows
        mC: cute.Tensor,  # input_gates     (M, N)
        mPrevC: cute.Tensor,  # previous_c  (M, H)   -- rank-0 dummy for RNN
        mOutH: cute.Tensor,  # h            (M, H)
        mOutC: cute.Tensor,  # c            (M, H)   -- rank-0 dummy for RNN
        stream: cuda.CUstream,
    ):
        elem = self._dtype
        async_elems = async_copy_elements(elem)

        # Swizzled smem so the ldmatrix that follows is bank-conflict free.  The
        # atom is chosen from the K tile because K is the contiguous axis of both
        # A (row-major, k fastest) and B (row-major over n, k fastest).
        atom_ab, smem_k_block = make_smem_swizzle_atom(elem, self._k_block)
        sA_layout = cute.tile_to_shape(
            atom_ab, (self._m_block, self._k_block, self._num_stages), (0, 1, 2)
        )
        sB_layout = cute.tile_to_shape(
            atom_ab, (self._n_block, self._k_block, self._num_stages), (0, 1, 2)
        )
        # FP32 accumulator staging for the epilogue.  Padded by one float per row
        # so the strided read a hidden unit's four gates perform is conflict-free.
        sAcc_layout = cute.make_layout(
            (self._m_block, self._n_block),
            stride=(self._n_block + 1, 1),
        )

        atom_cp = make_cp_async_atom(elem)
        gmem_tiled_copy_A = make_qkv_tiled_copy(
            atom_cp, self._num_threads, smem_k_block, async_elems
        )
        gmem_tiled_copy_B = gmem_tiled_copy_A

        # Warps tile M *and* N.  A recurrent step has a small M (the cohort) and a
        # large N (gates * hidden), so spending every warp on M -- as an attention
        # kernel does -- would force m_block up to num_warps*16 and leave the N
        # axis to a single warp.  warps_n lets a 16-row cohort still use 4 warps.
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(elem, cutlass.Float32, (16, 8, 16)),
            (self._warps_m, self._warps_n, 1),
            permutation_mnk=(self._warps_m * 16, self._warps_n * 16, 16),
        )

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[cute.struct.MemRange[elem, cute.cosize(sA_layout)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[elem, cute.cosize(sB_layout)], 1024]

        grid = (
            cute.ceil_div(cute.size(mC.shape[1]), self._n_block),
            cute.ceil_div(cute.size(mA.shape[0]), self._m_block),
            1,
        )
        smem_bytes = SharedStorage.size_in_bytes()
        self.kernel(
            mA,
            mB,
            mC,
            mPrevC,
            mOutH,
            mOutC,
            sA_layout,
            sB_layout,
            sAcc_layout,
            gmem_tiled_copy_A,
            gmem_tiled_copy_B,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid,
            block=(self._num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mPrevC: cute.Tensor,
        mOutH: cute.Tensor,
        mOutC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sAcc_layout: cute.Layout,
        gmem_tiled_copy_A: cute.TiledCopy,
        gmem_tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        n_block, m_block, _ = cute.arch.block_idx()

        M = cute.size(mA.shape[0])
        K = cute.size(mA.shape[1])
        N = cute.size(mC.shape[1])
        gates = cutlass.const_expr(self._gate_count)

        # ---- CTA tiles ------------------------------------------------------
        gA = cute.local_tile(mA, (self._m_block, self._k_block), (m_block, None))
        gB = cute.local_tile(mB, (self._n_block, self._k_block), (n_block, None))

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)

        gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(tidx)
        gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(tidx)
        tAgA = gmem_thr_copy_A.partition_S(gA)
        tAsA = gmem_thr_copy_A.partition_D(sA)
        tBgB = gmem_thr_copy_B.partition_S(gB)
        tBsB = gmem_thr_copy_B.partition_D(sB)

        # Row predicates: M and N are runtime extents and need not be tile
        # multiples.  K is looped exactly, so only the row axis is predicated.
        cA = cute.local_tile(
            cute.make_identity_tensor(mA.layout.shape),
            (self._m_block, self._k_block),
            (m_block, 0),
        )
        cB = cute.local_tile(
            cute.make_identity_tensor(mB.layout.shape),
            (self._n_block, self._k_block),
            (n_block, 0),
        )
        tAcA = gmem_thr_copy_A.partition_S(cA)
        tBcB = gmem_thr_copy_B.partition_S(cB)
        predA = cute.make_rmem_tensor(
            cute.make_layout((cute.size(tAsA.shape[1]),)), cutlass.Boolean
        )
        predB = cute.make_rmem_tensor(
            cute.make_layout((cute.size(tBsB.shape[1]),)), cutlass.Boolean
        )
        for i in cutlass.range_constexpr(cute.size(predA)):
            predA[i] = tAcA[0, i, 0][0] < M
        for i in cutlass.range_constexpr(cute.size(predB)):
            predB[i] = tBcB[0, i, 0][0] < N

        # ---- MMA fragments --------------------------------------------------
        thr_mma = tiled_mma.get_slice(tidx)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0]))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0]))
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block, self._n_block)), cutlass.Float32
        )
        acc.fill(0.0)

        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
        tCsA = smem_thr_copy_A.partition_S(sA)
        tCsB = smem_thr_copy_B.partition_S(sB)
        tCrA_view = smem_thr_copy_A.retile(tCrA)
        tCrB_view = smem_thr_copy_B.retile(tCrB)

        # ---- cp.async prologue ----------------------------------------------
        k_tiles = cute.ceil_div(K, self._k_block)
        stages = cutlass.const_expr(self._num_stages)
        for s in cutlass.range_constexpr(stages - 1):
            if s < k_tiles:
                for i in cutlass.range_constexpr(cute.size(tAsA.shape[1])):
                    if predA[i]:
                        cute.copy(
                            gmem_tiled_copy_A,
                            tAgA[None, i, None, s],
                            tAsA[None, i, None, s],
                        )
                for i in cutlass.range_constexpr(cute.size(tBsB.shape[1])):
                    if predB[i]:
                        cute.copy(
                            gmem_tiled_copy_B,
                            tBgB[None, i, None, s],
                            tBsB[None, i, None, s],
                        )
            cute.arch.cp_async_commit_group()

        # ---- Mainloop -------------------------------------------------------
        # One committed group per K tile, so waiting on ``stages - 2`` leaves the
        # deepest in-flight group still landing while this tile is consumed.
        read_stage = cutlass.Int32(0)
        write_stage = cutlass.Int32(stages - 1)
        for tile in cutlass.range(k_tiles, unroll=1):
            cute.arch.cp_async_wait_group(stages - 2)
            cute.arch.barrier()

            # Interleaved ldmatrix: iteration k's mma is issued only after
            # k+1's smem load, so the LSU works alongside the MMA pipe instead
            # of the whole k-block's loads draining before the first mma.
            gemm_with_smem_prefetch(
                tiled_mma,
                acc,
                tCrA,
                tCrB,
                smem_tiled_copy_A,
                smem_tiled_copy_B,
                tCsA[None, None, None, read_stage],
                tCsB[None, None, None, read_stage],
                tCrA_view,
                tCrB_view,
            )

            # Refill the stage just consumed with the tile ``stages - 1`` ahead.
            next_tile = tile + stages - 1
            cute.arch.barrier()
            if next_tile < k_tiles:
                for i in cutlass.range_constexpr(cute.size(tAsA.shape[1])):
                    if predA[i]:
                        cute.copy(
                            gmem_tiled_copy_A,
                            tAgA[None, i, None, next_tile],
                            tAsA[None, i, None, write_stage],
                        )
                for i in cutlass.range_constexpr(cute.size(tBsB.shape[1])):
                    if predB[i]:
                        cute.copy(
                            gmem_tiled_copy_B,
                            tBgB[None, i, None, next_tile],
                            tBsB[None, i, None, write_stage],
                        )
            cute.arch.cp_async_commit_group()
            read_stage = (read_stage + 1) % stages
            write_stage = (write_stage + 1) % stages

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # ---- Epilogue -------------------------------------------------------
        # The accumulator goes back through shared memory before the state
        # transition.  An LSTM cell consumes four adjacent gate columns, and the
        # m16n8k16 thread-value layout gives one thread only two of them; staging
        # makes the gather independent of that layout instead of encoding it.
        sAcc = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=cutlass.Float32), sAcc_layout)
        tCacc = thr_mma.partition_C(cute.make_identity_tensor((self._m_block, self._n_block)))
        for i in cutlass.range_constexpr(cute.size(acc)):
            coord = tCacc[i]
            sAcc[coord[0], coord[1]] = acc[i]
        cute.arch.barrier()

        # One thread per (row, hidden unit) in the tile; ``gates`` columns each.
        units = cutlass.const_expr(self._n_block // gates)
        total = self._m_block * units
        hidden0 = n_block * units
        row0 = m_block * self._m_block
        H = cute.size(mOutH.shape[1])

        for slot in cutlass.range(tidx, total, self._num_threads, unroll=1):
            r = slot // units
            u = slot % units
            m = row0 + r
            hid = hidden0 + u
            if m < M and hid < H:
                base = u * gates
                if cutlass.const_expr(gates == LSTM_GATES):
                    g0 = sAcc[r, base + 0] + mC[m, hid * gates + 0].to(cutlass.Float32)
                    g1 = sAcc[r, base + 1] + mC[m, hid * gates + 1].to(cutlass.Float32)
                    g2 = sAcc[r, base + 2] + mC[m, hid * gates + 2].to(cutlass.Float32)
                    g3 = sAcc[r, base + 3] + mC[m, hid * gates + 3].to(cutlass.Float32)
                    i_gate = 1.0 / (1.0 + cute.math.exp(-g0, fastmath=True))
                    f_gate = 1.0 / (1.0 + cute.math.exp(-g1, fastmath=True))
                    o_gate = 1.0 / (1.0 + cute.math.exp(-g3, fastmath=True))
                    c_new = f_gate * mPrevC[m, hid].to(cutlass.Float32) + i_gate * cute.math.tanh(
                        g2, fastmath=True
                    )
                    mOutC[m, hid] = c_new.to(self._dtype)
                    mOutH[m, hid] = (o_gate * cute.math.tanh(c_new, fastmath=True)).to(self._dtype)
                else:
                    v = sAcc[r, base] + mC[m, hid].to(cutlass.Float32)
                    if cutlass.const_expr(self._activation == "relu"):
                        act = cute.math.max(v, cutlass.Float32(0.0))
                    else:
                        act = cute.math.tanh(v, fastmath=True)
                    mOutH[m, hid] = act.to(self._dtype)
