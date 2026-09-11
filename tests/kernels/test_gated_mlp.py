# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused gated MLP (SwiGLU / GeGLU) — the CuTeDSL kernel, the functional and the layer.

The oracle is an FP32 matmul plus the gate equations in torch, so the kernel is
checked against the *definition* rather than against another OASR kernel.  Kept
in its own file because every test here needs CuTeDSL, which is an optional extra.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

cutlass = pytest.importorskip("cutlass", reason="CuTeDSL (nvidia-cutlass-dsl) not installed")

import oasr  # noqa: E402
from oasr.jit import mlp as jit_mlp  # noqa: E402
from oasr.jit.cute_runtime import current_stream  # noqa: E402
from oasr.layers import GatedMLP  # noqa: E402
from oasr.layers._backend import policy_hits, reset_backend_stats  # noqa: E402

pytestmark = pytest.mark.cuda


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in jit_mlp._SUPPORTED_SM


requires_cute = pytest.mark.skipif(
    not _supported(), reason="no CuTeDSL gated-MLP kernel for this arch"
)

_ACT = {
    "silu": F.silu,
    "swish": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_tanh": lambda t: F.gelu(t, approximate="tanh"),
    "identity": lambda t: t,
}

#: fp16 carries ~3 decimal digits and the oracle accumulates in fp32, so the
#: comparison is against the *relative* size of the output, not an absolute
#: epsilon that a large K would blow through.
_REL_TOL = {torch.float16: 3e-3, torch.bfloat16: 2e-2}


def _operands(M, N, K, dtype, bias, seed=17):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", dtype=dtype, generator=g)  # noqa: E731
    x = r(M, K) * 0.3
    wg = r(N, K) * K**-0.5
    wu = r(N, K) * K**-0.5
    bg = r(N) * 0.1 if bias else None
    bu = r(N) * 0.1 if bias else None
    return x, wg, wu, bg, bu


def _oracle(x, wg, wu, bg, bu, activation):
    gate = x.float() @ wg.float().T
    up = x.float() @ wu.float().T
    if bg is not None:
        gate = gate + bg.float()
        up = up + bu.float()
    return _ACT[activation](gate) * up


def _assert_close(out, ref, dtype):
    assert not torch.isnan(out).any(), "kernel produced NaN"
    scale = max(ref.abs().max().item(), 1e-6)
    rel = (out.float() - ref).abs().max().item() / scale
    assert rel < _REL_TOL[dtype], f"max relative error {rel:.5f}"


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@requires_cute
class TestKernel:
    """Direct against :class:`~oasr.kernels.cute.mlp.GatedMlpCute`, tile by tile."""

    @staticmethod
    def _run(M, N, K, tile, dtype, activation, bias):
        from oasr.kernels.cute.mlp import GatedMlpCute

        cute_dtype = cutlass.Float16 if dtype is torch.float16 else cutlass.BFloat16
        m, n, k, stages, threads, warps_n = tile
        kwargs = {
            "dtype": cute_dtype,
            "activation": activation,
            "has_bias": bias,
            "m_block": m,
            "n_block": n,
            "k_block": k,
            "num_stages": stages,
            "num_threads": threads,
            "warps_n": warps_n,
        }
        if not GatedMlpCute.can_implement(**kwargs):
            pytest.skip(f"tile {tile} does not fit here")
        dtype_str = "float16" if dtype is torch.float16 else "bfloat16"
        fn = jit_mlp._compiled_gated_mlp(
            torch.cuda.get_device_capability(), dtype_str, activation, bias, tile
        )
        x, wg, wu, bg, bu = _operands(M, N, K, dtype, bias)
        out = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
        dummy = torch.zeros(jit_mlp.ALIGNMENT, device="cuda", dtype=dtype)
        fn(
            x,
            wg,
            wu,
            bg if bias else dummy,
            bu if bias else dummy,
            out,
            current_stream(),
        )
        _assert_close(out, _oracle(x, wg, wu, bg, bu, activation), dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("activation", sorted(_ACT))
    def test_activations(self, dtype, activation):
        self._run(64, 256, 128, (16, 64, 64, 3, 128, 4), dtype, activation, False)

    @pytest.mark.parametrize("bias", [False, True])
    def test_bias(self, bias):
        self._run(40, 200, 256, (32, 64, 64, 3, 128, 2), torch.float16, "silu", bias)

    @pytest.mark.parametrize("M", [1, 3, 16, 17, 63, 65, 130])
    def test_ragged_rows(self, M):
        """M need not be a tile multiple; the row axis is predicated on both
        the gmem loads and the epilogue store."""
        self._run(M, 200, 128, (16, 64, 64, 3, 128, 4), torch.float16, "silu", False)

    @pytest.mark.parametrize("N", [8, 24, 64, 72, 136, 200])
    def test_ragged_cols(self, N):
        """N need not be a tile multiple either — but it *is* 8-aligned, which is
        the 128-bit store contract ``gated_mlp_shape_supported`` enforces."""
        self._run(33, N, 128, (16, 64, 64, 3, 128, 4), torch.float16, "silu", False)

    @pytest.mark.parametrize(
        "tile",
        [
            (16, 64, 64, 3, 128, 4),
            (16, 64, 64, 4, 128, 4),
            (16, 64, 32, 4, 64, 2),
            (32, 64, 64, 4, 128, 4),
            (32, 128, 64, 2, 256, 4),
            (64, 64, 64, 4, 256, 4),
            (64, 128, 64, 2, 512, 4),
            (128, 64, 64, 3, 256, 2),
            (128, 128, 64, 2, 512, 4),
        ],
    )
    def test_tiles(self, tile):
        self._run(70, 304, 256, tile, torch.float16, "silu", False)

    def test_can_implement_refuses_overrunning_thread_layout(self):
        """A gmem->smem pass wider than the tile is an illegal access, not a
        predicated no-op — the partition itself is out of range.  It has to be
        refused statically, which is why every tile above is checked first."""
        from oasr.kernels.cute.mlp import GatedMlpCute

        # 128 threads x 8 elements over a 32-wide K tile walks 32 rows per pass,
        # twice a 16-row M tile.
        assert not GatedMlpCute.can_implement(
            dtype=cutlass.Float16, m_block=16, n_block=64, k_block=32, num_threads=128, warps_n=4
        )
        assert GatedMlpCute.can_implement(
            dtype=cutlass.Float16, m_block=16, n_block=64, k_block=32, num_threads=64, warps_n=2
        )

    def test_can_implement_refuses_oversized_smem(self):
        from oasr.kernels.cute.mlp import GatedMlpCute

        assert not GatedMlpCute.can_implement(
            dtype=cutlass.Float16,
            m_block=128,
            n_block=256,
            k_block=64,
            num_stages=4,
            num_threads=512,
            warps_n=4,
        )


# ---------------------------------------------------------------------------
# Routing — what the kernel refuses, and why refusing matters
# ---------------------------------------------------------------------------


@requires_cute
class TestRouting:
    """``gated_mlp_shape_supported`` is the guard between "fused" and "silently wrong"."""

    @pytest.mark.parametrize(
        "n,k,ok",
        [
            (18944, 3584, True),
            (200, 128, True),
            (204, 128, False),  # N not 8-aligned: the 128-bit store misaligns
            (200, 132, False),  # K not 8-aligned: so does the 128-bit cp.async
            (200, 96, False),  # K not a whole number of 64-wide K tiles
            (200, 32, False),  # ...and a K *shorter* than one tile reads past x
        ],
    )
    def test_shape_contract(self, n, k, ok):
        """Each False here was observed to fault or return garbage before the
        guard existed: K=96 gave a max relative error of 83, K=32 gave NaN, and
        an N of 129 raised ``misaligned address``."""
        assert jit_mlp.gated_mlp_shape_supported(rows=8, n=n, k=k, k_block=64) is ok

    def test_unsupported_shape_is_not_available(self):
        x = torch.randn(8, 96, device="cuda", dtype=torch.float16)
        w = torch.randn(200, 96, device="cuda", dtype=torch.float16)
        assert not oasr.gated_mlp_available(x, w)

    def test_fp32_and_cpu_are_not_available(self):
        x32 = torch.randn(8, 128, device="cuda", dtype=torch.float32)
        w32 = torch.randn(256, 128, device="cuda", dtype=torch.float32)
        assert not oasr.gated_mlp_available(x32, w32)
        assert not oasr.gated_mlp_available(x32.cpu(), w32.cpu())

    def test_above_band_is_not_available(self):
        x = torch.randn(jit_mlp._BAND_MAX_ROWS + 1, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(256, 128, device="cuda", dtype=torch.float16)
        assert not oasr.gated_mlp_available(x, w)

    def test_mode_off_disables_everything(self):
        x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(256, 128, device="cuda", dtype=torch.float16)
        try:
            jit_mlp.set_gated_mlp_mode("off")
            assert not oasr.gated_mlp_available(x, w)
            jit_mlp.set_gated_mlp_mode("auto")
            assert oasr.gated_mlp_available(x, w)
        finally:
            jit_mlp.set_gated_mlp_mode("auto")

    def test_always_mode_reaches_above_the_band(self):
        rows = jit_mlp._BAND_MAX_ROWS + 64
        x = torch.randn(rows, 128, device="cuda", dtype=torch.float16) * 0.3
        wg = torch.randn(256, 128, device="cuda", dtype=torch.float16) * 128**-0.5
        wu = torch.randn(256, 128, device="cuda", dtype=torch.float16) * 128**-0.5
        try:
            jit_mlp.set_gated_mlp_mode("always")
            assert oasr.gated_mlp_available(x, wg)
            out = oasr.gated_mlp(x, wg, wu)
            _assert_close(out, _oracle(x, wg, wu, None, None, "silu"), torch.float16)
        finally:
            jit_mlp.set_gated_mlp_mode("auto")


# ---------------------------------------------------------------------------
# The functional
# ---------------------------------------------------------------------------


@requires_cute
class TestFunctional:
    def test_matches_oracle(self):
        x, wg, wu, _, _ = _operands(37, 18944, 3584, torch.float16, False)
        out = oasr.gated_mlp(x, wg, wu)
        _assert_close(out, _oracle(x, wg, wu, None, None, "silu"), torch.float16)

    def test_leading_rank_is_preserved(self):
        wg = torch.randn(512, 256, device="cuda", dtype=torch.float16) * 256**-0.5
        wu = torch.randn(512, 256, device="cuda", dtype=torch.float16) * 256**-0.5
        x = torch.randn(2, 8, 256, device="cuda", dtype=torch.float16) * 0.3
        out = oasr.gated_mlp(x, wg, wu)
        assert out.shape == (2, 8, 512)
        _assert_close(out, _oracle(x, wg, wu, None, None, "silu"), torch.float16)

    def test_destination_passing(self):
        x, wg, wu, _, _ = _operands(8, 512, 256, torch.float16, False)
        dst = torch.empty(8, 512, device="cuda", dtype=torch.float16)
        got = oasr.gated_mlp(x, wg, wu, out=dst)
        assert got.data_ptr() == dst.data_ptr()
        _assert_close(got, _oracle(x, wg, wu, None, None, "silu"), torch.float16)

    def test_refuses_rather_than_falling_back(self):
        """A silent reroute to torch is what makes a missing kernel invisible."""
        x = torch.randn(8, 96, device="cuda", dtype=torch.float16)
        w = torch.randn(200, 96, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="cannot serve"):
            oasr.gated_mlp(x, w, w)

    def test_one_bias_is_an_error(self):
        x, wg, wu, bg, _ = _operands(8, 512, 256, torch.float16, True)
        with pytest.raises(ValueError, match="both biases or neither"):
            oasr.gated_mlp(x, wg, wu, bg, None)

    def test_capturable_in_a_cuda_graph(self):
        """The decoder step is graph-captured, so this is the production path.

        ``--enable-tvm-ffi`` is what makes it safe: the legacy per-call DLPack
        wrapper produced capsules whose ownership was freed between replays.
        """
        x, wg, wu, _, _ = _operands(4, 512, 256, torch.float16, False)
        dst = torch.empty(4, 512, device="cuda", dtype=torch.float16)
        oasr.gated_mlp(x, wg, wu, out=dst)  # compile + warm outside the capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                oasr.gated_mlp(x, wg, wu, out=dst)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            oasr.gated_mlp(x, wg, wu, out=dst)
        dst.zero_()
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(dst, _oracle(x, wg, wu, None, None, "silu"), torch.float16)


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------


@requires_cute
class TestGatedMLPLayer:
    @staticmethod
    def _layer(activation, bias, d_model=256, hidden=512):
        torch.manual_seed(3)
        return GatedMLP(d_model, hidden, activation=activation, bias=bias).cuda().half()

    @pytest.mark.parametrize("activation", ["silu", "relu", "gelu_tanh", "gelu"])
    @pytest.mark.parametrize("bias", [False, True])
    def test_fused_matches_two_gemm_path(self, activation, bias):
        """Not bit-exact on purpose: the fused epilogue keeps the gate and the
        multiply in FP32 and rounds once, where the two-GEMM path rounds the
        activated gate and the up projection before multiplying."""
        layer = self._layer(activation, bias)
        x = torch.randn(6, 256, device="cuda", dtype=torch.float16) * 0.3
        layer.fuse_gate_up = True
        fused = layer(x)
        layer.fuse_gate_up = False
        unfused = layer(x)
        torch.testing.assert_close(fused, unfused, rtol=6e-3, atol=6e-3)

    def test_out_of_band_call_is_a_policy_hit_not_a_gap(self):
        """The alternative to fusing is two OASR kernels, not torch — so a
        decline is counted under performance policy and never as a kernel gap."""
        layer = self._layer("silu", False)
        x = torch.randn(jit_mlp._BAND_MAX_ROWS + 8, 256, device="cuda", dtype=torch.float16)
        reset_backend_stats()
        layer(x)
        assert policy_hits().get("gated-mlp-unfused") == 1

    def test_in_band_call_takes_no_policy_hit(self):
        layer = self._layer("silu", False)
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        reset_backend_stats()
        layer(x)
        assert "gated-mlp-unfused" not in policy_hits()

    def test_fuse_gate_up_false_never_asks(self):
        layer = self._layer("silu", False)
        layer.fuse_gate_up = False
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        reset_backend_stats()
        layer(x)
        assert "gated-mlp-unfused" not in policy_hits()
        assert "fuse_gate_up=False" in repr(layer)

    def test_torch_backend_bypasses_the_kernel(self):
        from oasr.layers._backend import layers_backend_override

        layer = self._layer("silu", False)
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16) * 0.3
        fused = layer(x)
        with layers_backend_override("torch"):
            torch_out = layer(x)
        torch.testing.assert_close(fused, torch_out, rtol=6e-3, atol=6e-3)

    def test_rollback_switch_is_not_counted_as_policy(self):
        """``OASR_GATED_MLP_CUTE=0`` is an operator decision, not a shape one.

        Counting it under performance policy would make every A/B run report a
        table of declines that says nothing about any shape.
        """
        layer = self._layer("silu", False)
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        try:
            jit_mlp.set_gated_mlp_mode("off")
            reset_backend_stats()
            layer(x)
            assert "gated-mlp-unfused" not in policy_hits()
        finally:
            jit_mlp.set_gated_mlp_mode("auto")


@requires_cute
class TestPreconditions:
    """Each of these is a *silent* wrong answer if the check is removed."""

    @staticmethod
    def _ops(M=8, N=512, K=256):
        return _operands(M, N, K, torch.float16, False)

    def test_non_contiguous_x_is_not_available(self):
        """``reshape`` on a non-contiguous ``x`` copies, and the 2-D kernel would
        then read a buffer whose strides are not the ones it was promised."""
        _, wg, _, _, _ = self._ops()
        strided = torch.randn(8, 512, device="cuda", dtype=torch.float16)[:, ::2]
        assert not strided.is_contiguous() and strided.shape[-1] == wg.shape[1]
        assert not oasr.gated_mlp_available(strided, wg)

    def test_k_mismatch_is_not_available(self):
        """``K`` comes from ``x``; a weight of a different width would be read at
        the wrong stride rather than raising."""
        x, _, _, _, _ = self._ops()
        wrong = torch.randn(512, 128, device="cuda", dtype=torch.float16)
        assert not oasr.gated_mlp_available(x, wrong)

    def test_non_contiguous_w_up_raises(self):
        x, wg, wu, _, _ = self._ops()
        strided = torch.randn(512, 512, device="cuda", dtype=torch.float16)[:, ::2]
        with pytest.raises(ValueError, match="w_up"):
            oasr.gated_mlp(x, wg, strided)
        with pytest.raises(ValueError, match="w_up"):
            oasr.gated_mlp(x, wg, wu[: wu.shape[0] // 2])

    def test_non_contiguous_out_raises(self):
        """A copy would swallow the result: the kernel writes the copy, the
        caller reads the original."""
        x, wg, wu, _, _ = self._ops()
        dst = torch.empty(8, 1024, device="cuda", dtype=torch.float16)[:, ::2]
        with pytest.raises(ValueError, match="contiguous out="):
            oasr.gated_mlp(x, wg, wu, out=dst)

    def test_non_contiguous_bias_raises(self):
        x, wg, wu, _, _ = self._ops()
        bias = torch.randn(1024, device="cuda", dtype=torch.float16)[::2]
        with pytest.raises(ValueError, match="contiguous biases"):
            oasr.gated_mlp(x, wg, wu, bias, bias)
