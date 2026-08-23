# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL fused recurrent step.

The oracle is an FP32 matmul plus the gate equations in torch, so these check the
kernel against the *definition* rather than against another OASR kernel.  Kept in
its own file because every test here needs CuTeDSL, which is an optional extra.
"""

from __future__ import annotations

import pytest
import torch

cutlass = pytest.importorskip("cutlass", reason="CuTeDSL (nvidia-cutlass-dsl) not installed")

from oasr.jit import recurrent_cute  # noqa: E402

pytestmark = pytest.mark.cuda


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in recurrent_cute._SUPPORTED_SM


requires_cute = pytest.mark.skipif(
    not _supported(), reason="no CuTeDSL recurrent-step kernel for this arch"
)


def _oracle(a, weight, c, prev_c, gates):
    """FP32 reference: the equations, not another kernel."""
    acc = a.float() @ weight.float().T + c.float()
    if gates == 1:
        return torch.tanh(acc), None
    gv = acc.view(a.shape[0], -1, 4)
    cell = torch.sigmoid(gv[..., 1]) * prev_c.float() + torch.sigmoid(gv[..., 0]) * torch.tanh(
        gv[..., 2]
    )
    return torch.sigmoid(gv[..., 3]) * torch.tanh(cell), cell


def _run(dtype, hidden, batch, gates, activation, tile):
    from oasr.kernels.cute.recurrent import RecurrentStepCute

    n = gates * hidden
    g = torch.Generator(device="cuda").manual_seed(17)
    a = torch.randn(batch, hidden, device="cuda", dtype=dtype, generator=g) * 0.3
    weight = torch.randn(n, hidden, device="cuda", dtype=dtype, generator=g) * hidden**-0.5
    c = torch.randn(batch, n, device="cuda", dtype=dtype, generator=g) * 0.3
    prev_c = torch.randn(batch, hidden, device="cuda", dtype=dtype, generator=g) * 0.3
    out_h = torch.zeros(batch, hidden, device="cuda", dtype=dtype)
    out_c = torch.zeros(batch, hidden, device="cuda", dtype=dtype)

    dtype_str = "float16" if dtype is torch.float16 else "bfloat16"
    cute_dtype = cutlass.Float16 if dtype is torch.float16 else cutlass.BFloat16
    m, nb, k, stages, threads, warps_n = tile
    if not RecurrentStepCute.can_implement(
        dtype=cute_dtype,
        gate_count=gates,
        activation=activation,
        m_block=m,
        n_block=nb,
        k_block=k,
        num_stages=stages,
        num_threads=threads,
        warps_n=warps_n,
    ):
        pytest.skip(f"tile {tile} not implementable")
    step = recurrent_cute._compiled_step(
        torch.cuda.get_device_capability(), dtype_str, gates, activation, tile
    )
    step(a, weight, c, prev_c, out_h, out_c, recurrent_cute.current_stream())
    torch.cuda.synchronize()
    ref_h, ref_c = _oracle(a, weight, c, prev_c, gates)
    return out_h, out_c, ref_h, ref_c


@requires_cute
class TestRecurrentStepCute:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("hidden,batch", [(256, 64), (640, 32), (640, 130), (1024, 8)])
    def test_lstm_matches_fp32_equations(self, device, dtype, hidden, batch):
        tile = recurrent_cute.select_tile(hidden, batch)
        assert tile is not None
        out_h, out_c, ref_h, ref_c = _run(dtype, hidden, batch, 4, "lstm", tile)
        # FP16 accumulation over K=hidden against an FP32 oracle; 2e-2 is the
        # tolerance the rest of the recurrent suite uses for the same comparison.
        torch.testing.assert_close(out_h.float(), ref_h, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_c.float(), ref_c, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("activation", ["tanh"])
    def test_rnn_matches_fp32_equations(self, device, activation):
        out_h, _, ref_h, _ = _run(torch.float16, 640, 64, 1, activation, (32, 64, 64, 3, 128, 2))
        torch.testing.assert_close(out_h.float(), ref_h, rtol=2e-2, atol=2e-2)

    def test_batch_need_not_be_a_tile_multiple(self, device):
        """A ragged M must be predicated, not rounded up into other rows' memory."""
        for batch in (1, 7, 33):
            out_h, out_c, ref_h, ref_c = _run(
                torch.float16, 640, batch, 4, "lstm", (32, 64, 64, 3, 128, 2)
            )
            torch.testing.assert_close(out_h.float(), ref_h, rtol=2e-2, atol=2e-2)

    def test_can_implement_rejects_oversized_copy_tiles(self):
        """The gmem thread layout must not walk past a tile's row extent.

        ``num_threads * 8 // k_block`` rows are touched per copy pass; when that
        exceeds ``n_block`` the surplus threads address outside the tile, which is
        an illegal access rather than a predicated no-op.  This exact tile faulted
        before the constraint existed.
        """
        from oasr.kernels.cute.recurrent import RecurrentStepCute

        common = {
            "dtype": cutlass.Float16,
            "gate_count": 4,
            "activation": "lstm",
            "num_stages": 3,
        }
        assert not RecurrentStepCute.can_implement(
            m_block=128, n_block=32, k_block=32, num_threads=256, **common
        )
        assert RecurrentStepCute.can_implement(
            m_block=128, n_block=32, k_block=32, num_threads=128, **common
        )

    def test_can_implement_rejects_mismatched_gate_and_activation(self):
        from oasr.kernels.cute.recurrent import RecurrentStepCute

        assert not RecurrentStepCute.can_implement(
            dtype=cutlass.Float16, gate_count=4, activation="tanh"
        )
        assert not RecurrentStepCute.can_implement(
            dtype=cutlass.Float16, gate_count=1, activation="lstm"
        )
        assert not RecurrentStepCute.can_implement(
            dtype=cutlass.Float32, gate_count=4, activation="lstm"
        )

    def test_epilogue_staging_fits_the_ring_it_aliases(self):
        """The FP32 accumulator staging aliases the A/B ring, so it must fit in it."""
        from oasr.kernels.cute.recurrent import RecurrentStepCute

        # 128x128 of FP32 is 66 KB; a 2-stage 32-deep FP16 ring is only 32 KB.
        assert not RecurrentStepCute.can_implement(
            dtype=cutlass.Float16,
            gate_count=4,
            activation="lstm",
            m_block=128,
            n_block=128,
            k_block=32,
            num_stages=2,
            num_threads=128,
        )


class TestRecurrentCuteRouting:
    """The routing table and its gate -- these need no GPU."""

    def test_default_is_auto(self, monkeypatch):
        """Default is band routing, which the layer-level measurement earned."""
        monkeypatch.delenv("OASR_RECURRENT_CUTE", raising=False)
        assert recurrent_cute._read_mode() == "auto"

    @pytest.mark.parametrize(
        "raw,expected",
        [("1", "always"), ("always", "always"), ("0", "off"), ("off", "off"), ("auto", "auto")],
    )
    def test_env_gate(self, monkeypatch, raw, expected):
        monkeypatch.setenv("OASR_RECURRENT_CUTE", raw)
        assert recurrent_cute._read_mode() == expected

    def test_unknown_mode_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("OASR_RECURRENT_CUTE", "banana")
        assert recurrent_cute._read_mode() == "auto"

    def test_every_band_shape_has_a_tile(self):
        """A shape the band admits must have a tile; otherwise routing dead-ends."""
        for width, (low, high) in recurrent_cute._LSTM_BANDS:
            hidden = width if width < (1 << 20) else 2048
            for batch in (low, min(high, 512)):
                assert recurrent_cute.select_tile(hidden, batch) is not None, (hidden, batch)

    def test_tiles_cover_every_width_and_batch(self):
        """No (hidden, batch) may fall through the table -- the last row is a catch-all."""
        for hidden in (16, 256, 257, 640, 1024, 1536, 2048, 4096):
            for batch in (1, 3, 16, 64, 129, 512, 4096):
                assert recurrent_cute.select_tile(hidden, batch) is not None, (hidden, batch)

    def test_every_tabled_tile_is_implementable(self):
        """The table cannot contain a tile the kernel would refuse to build."""
        from oasr.kernels.cute.recurrent import RecurrentStepCute

        for _, _, tile in recurrent_cute._TILES:
            m, n, k, stages, threads, warps_n = tile
            assert RecurrentStepCute.can_implement(
                dtype=cutlass.Float16,
                gate_count=4,
                activation="lstm",
                m_block=m,
                n_block=n,
                k_block=k,
                num_stages=stages,
                num_threads=threads,
                warps_n=warps_n,
            ), tile

    def test_rnn_is_not_routed_in_auto(self):
        """Declared, not guessed: the RNN has no matched reference measurement."""
        previous = recurrent_cute.get_mode()
        try:
            recurrent_cute.set_mode("auto")
            if recurrent_cute._probe() is None:
                pytest.skip("no CuTeDSL device")
            assert not recurrent_cute.should_use(1, 640, 64)
            assert recurrent_cute.should_use(4, 640, 64)
        finally:
            recurrent_cute.set_mode(previous)
