#!/usr/bin/env python3
"""Correctness and contract tests for custom LSTM and vanilla RNN layers."""

from __future__ import annotations

import pytest
import torch
from helpers import device_sm, requires_cute, requires_sm, tol
from torch import nn

import oasr
from oasr.layers import LSTM, RNN, layers_backend_override


def _copy_to_reference(ours: nn.Module, reference: nn.Module) -> None:
    reference.load_state_dict(ours.state_dict())


#: The CUTLASS 3.x recurrent arms are compiled only for these two targets.
_TMA_TACTICS = [(6, 1), (7, 1)]
_requires_tma = requires_sm(90, 100, what="TMA warp-specialized recurrent tactic")


class TestRecurrentCpu:
    @pytest.mark.parametrize("batch_first", [False, True])
    @pytest.mark.parametrize("bias", [False, True])
    def test_lstm_matches_pytorch(self, batch_first, bias):
        torch.manual_seed(0)
        ours = LSTM(7, 11, num_layers=2, bias=bias, batch_first=batch_first)
        reference = nn.LSTM(7, 11, num_layers=2, bias=bias, batch_first=batch_first)
        _copy_to_reference(ours, reference)
        shape = (3, 5, 7) if batch_first else (5, 3, 7)
        x = torch.randn(shape)
        state = (torch.randn(2, 3, 11), torch.randn(2, 3, 11))
        got = ours(x, state)
        expected = reference(x, state)
        torch.testing.assert_close(got[0], expected[0])
        torch.testing.assert_close(got[1][0], expected[1][0])
        torch.testing.assert_close(got[1][1], expected[1][1])

    @pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
    def test_rnn_matches_pytorch(self, nonlinearity):
        torch.manual_seed(1)
        ours = RNN(9, 13, num_layers=3, nonlinearity=nonlinearity, batch_first=True)
        reference = nn.RNN(9, 13, num_layers=3, nonlinearity=nonlinearity, batch_first=True)
        _copy_to_reference(ours, reference)
        x = torch.randn(4, 6, 9)
        state = torch.randn(3, 4, 13)
        got = ours(x, state)
        expected = reference(x, state)
        torch.testing.assert_close(got[0], expected[0])
        torch.testing.assert_close(got[1], expected[1])

    def test_unbatched_lstm_matches_pytorch(self):
        torch.manual_seed(2)
        ours = LSTM(5, 8, num_layers=2, batch_first=True)
        reference = nn.LSTM(5, 8, num_layers=2, batch_first=True)
        _copy_to_reference(ours, reference)
        x = torch.randn(7, 5)
        state = (torch.randn(2, 8), torch.randn(2, 8))
        got = ours(x, state)
        expected = reference(x, state)
        torch.testing.assert_close(got[0], expected[0])
        torch.testing.assert_close(got[1][0], expected[1][0])
        torch.testing.assert_close(got[1][1], expected[1][1])

    @pytest.mark.parametrize("ours_cls,torch_cls", [(LSTM, nn.LSTM), (RNN, nn.RNN)])
    def test_parameter_layout_matches_pytorch(self, ours_cls, torch_cls):
        ours = ours_cls(8, 16, num_layers=2, bias=True)
        reference = torch_cls(8, 16, num_layers=2, bias=True)
        assert {name: tuple(parameter.shape) for name, parameter in ours.named_parameters()} == {
            name: tuple(parameter.shape) for name, parameter in reference.named_parameters()
        }

    def test_rejects_unsupported_variants(self):
        with pytest.raises(ValueError, match="bidirectional"):
            LSTM(8, 16, bidirectional=True)
        with pytest.raises(ValueError, match="proj_size"):
            LSTM(8, 16, proj_size=8)
        with pytest.raises(ValueError, match="nonlinearity"):
            RNN(8, 16, nonlinearity="sigmoid")

    def test_lstm_packed_parameter_layout_and_cache_invalidation(self):
        hidden_size, input_size = 3, 2
        module = LSTM(input_size, hidden_size)
        with torch.no_grad():
            module.weight_ih_l0.copy_(torch.arange(24).reshape(12, 2))
            module.weight_hh_l0.copy_(torch.arange(36).reshape(12, 3))
            module.bias_ih_l0.copy_(torch.arange(12))
            module.bias_hh_l0.copy_(10 + torch.arange(12))

        packed_ih, packed_hh, packed_bias = module._packed_lstm_parameters(0)
        expected_ih = module.weight_ih_l0.reshape(4, hidden_size, input_size).permute(1, 0, 2)
        expected_hh = module.weight_hh_l0.reshape(4, hidden_size, hidden_size).permute(1, 0, 2)
        expected_bias = (module.bias_ih_l0 + module.bias_hh_l0).reshape(4, hidden_size).t()
        torch.testing.assert_close(packed_ih.reshape(hidden_size, 4, input_size), expected_ih)
        torch.testing.assert_close(packed_hh.reshape(hidden_size, 4, hidden_size), expected_hh)
        torch.testing.assert_close(packed_bias.reshape(hidden_size, 4), expected_bias)

        cached = module._packed_lstm_parameters(0)
        assert cached[0].data_ptr() == packed_ih.data_ptr()
        with torch.no_grad():
            module.weight_hh_l0.add_(1)
        repacked = module._packed_lstm_parameters(0)
        assert repacked[0].data_ptr() != packed_ih.data_ptr()


def _lstm_layer_formula(sequence, hidden, cell, weight_ih, weight_hh, bias_ih, bias_hh):
    """The LSTM recurrence, written out, one timestep at a time.

    An independent statement of the rule ``LSTM._torch_layer`` implements, kept
    here rather than in the layer for the same reason ``tests/test_alignment_cpp.py``
    restates the alignment rule: the fast path calls ``torch.lstm``, so without a
    second, separately written source of truth the layer would only ever be
    checked against itself.  ``sequence`` is time-major ``(T, B, input)``.
    """
    outputs = []
    for timestep in range(sequence.shape[0]):
        gates = torch.nn.functional.linear(sequence[timestep], weight_ih, bias_ih)
        gates = gates + torch.nn.functional.linear(hidden, weight_hh, bias_hh)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(cell_gate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        outputs.append(hidden)
    return torch.stack(outputs), hidden, cell


class TestTorchLayerFormula:
    """``LSTM._torch_layer`` against the recurrence written out by hand.

    This is what makes the fused call safe to have taken.  The gate order
    ``(i, f, g, o)`` packed into one ``4H`` row block is not a convention the
    layer may pick for itself -- ``convert_silero_state_dict`` and every
    PyTorch-shaped checkpoint depend on it -- so it is pinned against a formula
    that spells it out rather than against another library call.
    """

    @pytest.mark.parametrize("batch_first", [False, True])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("timesteps", [1, 7])
    def test_fused_layer_matches_the_written_out_recurrence(self, batch_first, bias, timesteps):
        torch.manual_seed(3)
        batch, input_size, hidden_size = 4, 6, 10
        module = LSTM(input_size, hidden_size, bias=bias, batch_first=batch_first)
        shape = (batch, timesteps, input_size) if batch_first else (timesteps, batch, input_size)
        x = torch.randn(shape)
        h0 = torch.randn(batch, hidden_size)
        c0 = torch.randn(batch, hidden_size)
        bias_ih, bias_hh = module._biases(0)

        got = LSTM._torch_layer(
            x, h0, c0, module.weight_ih_l0, module.weight_hh_l0, bias_ih, bias_hh, batch_first
        )
        expected = _lstm_layer_formula(
            x.transpose(0, 1) if batch_first else x,
            h0,
            c0,
            module.weight_ih_l0,
            module.weight_hh_l0,
            bias_ih,
            bias_hh,
        )
        expected_output = expected[0].transpose(0, 1) if batch_first else expected[0]
        torch.testing.assert_close(got[0], expected_output)
        torch.testing.assert_close(got[1], expected[1])
        torch.testing.assert_close(got[2], expected[2])

    def test_the_gate_order_is_i_f_g_o(self):
        """A permuted gate block must fail, or the test above proves nothing."""
        torch.manual_seed(4)
        module = LSTM(6, 10, batch_first=True)
        x = torch.randn(4, 7, 6)
        h0, c0 = torch.randn(4, 10), torch.randn(4, 10)
        bias_ih, bias_hh = module._biases(0)
        swapped = module.weight_ih_l0.detach().clone()
        # Swap the input and forget blocks -- the classic layout confusion.
        swapped[:10], swapped[10:20] = module.weight_ih_l0[10:20], module.weight_ih_l0[:10]
        got = LSTM._torch_layer(x, h0, c0, swapped, module.weight_hh_l0, bias_ih, bias_hh, True)
        expected = _lstm_layer_formula(
            x.transpose(0, 1), h0, c0, module.weight_ih_l0, module.weight_hh_l0, bias_ih, bias_hh
        )
        assert not torch.allclose(got[0], expected[0].transpose(0, 1), atol=1e-4)


@pytest.mark.cuda
class TestWideUnitsReachTheCohortPath:
    """A unit whose weights need more than 48 KiB of shared memory.

    The cohort path used to be gated on ``cohort_smem <= 48 * 1024`` -- the
    budget a block gets without asking, identical on every architecture -- so an
    LSTM with ``input + hidden > 6144`` at half precision fell back to the
    per-unit kernel on a card with 164 KiB available.  Measured on an A30, the
    cohort kernel is **1.5-2.5x faster** across that whole region, so the
    fallback was not a trade-off, just an unasked question.

    Shapes here are chosen to land above the old ceiling: ``4 * (I + H) * 2``
    bytes is 50 KiB at I = H = 3200.
    """

    @pytest.mark.parametrize("hidden", [3200, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_torch_above_the_old_48kib_ceiling(self, hidden, dtype):
        import oasr

        smem = 4 * (hidden + hidden) * torch.tensor([], dtype=dtype).element_size()
        assert smem > 48 * 1024, "this shape no longer tests the widened path"
        if smem > torch.cuda.get_device_properties(0).shared_memory_per_block_optin:
            pytest.skip(f"{smem} B exceeds this device's opt-in shared memory")

        torch.manual_seed(hidden)
        batch, seq, i_size = 8, 3, hidden
        x = torch.randn(seq, batch, i_size, device="cuda", dtype=dtype) * 0.1
        h0 = torch.randn(batch, hidden, device="cuda", dtype=dtype) * 0.1
        c0 = torch.randn(batch, hidden, device="cuda", dtype=dtype) * 0.1
        w_ih = torch.randn(4 * hidden, i_size, device="cuda", dtype=dtype) / i_size**0.5
        w_hh = torch.randn(4 * hidden, hidden, device="cuda", dtype=dtype) / hidden**0.5
        b_ih = torch.randn(4 * hidden, device="cuda", dtype=dtype) * 0.05
        b_hh = torch.randn(4 * hidden, device="cuda", dtype=dtype) * 0.05

        out, _, _ = oasr.lstm_layer(x, h0, c0, w_ih, w_hh, b_ih, b_hh)

        ref = torch.nn.LSTM(i_size, hidden).to("cuda", dtype)
        with torch.no_grad():
            ref.weight_ih_l0.copy_(w_ih)
            ref.weight_hh_l0.copy_(w_hh)
            ref.bias_ih_l0.copy_(b_ih)
            ref.bias_hh_l0.copy_(b_hh)
            expected, _ = ref(x, (h0.unsqueeze(0), c0.unsqueeze(0)))
        torch.testing.assert_close(out, expected, **tol(dtype))

    def test_a_wide_slot_step_is_served_rather_than_refused(self):
        """The capability half, and the one that is behavioural.

        ``SlotStepImpl`` returns ``cudaErrorInvalidValue`` -- the declared "this
        unit's weights do not fit in shared memory" signal -- when the weights
        exceed the budget.  That budget was the 48 KiB every block gets without
        asking, so an LSTM with ``input + hidden > 6144`` at half precision was
        refused on a card with three times the shared memory free.  Asking the
        device turns the refusal into a launch.
        """
        import oasr

        hidden = i_size = 3200
        smem = 4 * (i_size + hidden) * 2
        assert smem > 48 * 1024, "this shape no longer tests the widened path"
        if smem > torch.cuda.get_device_properties(0).shared_memory_per_block_optin:
            pytest.skip(f"{smem} B exceeds this device's opt-in shared memory")

        torch.manual_seed(0)
        batch = slots = 4
        x = torch.randn(batch, i_size, device="cuda", dtype=torch.float16) * 0.1
        state_h = torch.zeros(2, slots, hidden, device="cuda", dtype=torch.float16)
        state_c = torch.zeros(slots, hidden, device="cuda", dtype=torch.float16)
        state_slots = torch.arange(batch, device="cuda", dtype=torch.int64)
        read_parity = torch.zeros(batch, device="cuda", dtype=torch.int32)
        w_ih = torch.randn(4 * hidden, i_size, device="cuda", dtype=torch.float16) / i_size**0.5
        w_hh = torch.randn(4 * hidden, hidden, device="cuda", dtype=torch.float16) / hidden**0.5

        out = oasr.lstm_slot_step(x, state_h, state_c, state_slots, read_parity, w_ih, w_hh)
        torch.cuda.synchronize()
        assert out.shape == (batch, hidden)
        assert torch.isfinite(out).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="recurrent kernels need CUDA")
class TestRecurrentCuda:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "batch,sequence,input_size,hidden_size,layers,batch_first",
        [
            (1, 1, 64, 64, 1, True),
            (3, 5, 48, 64, 2, True),
            (8, 3, 64, 64, 2, True),
            (8, 2, 63, 65, 1, True),  # widths that are not a multiple of 8
            (4, 7, 64, 96, 2, False),
            (16, 3, 1024, 1024, 1, True),
        ],
    )
    def test_lstm_matches_cudnn(
        self, dtype, batch, sequence, input_size, hidden_size, layers, batch_first
    ):
        torch.manual_seed(3)
        ours = (
            LSTM(input_size, hidden_size, num_layers=layers, batch_first=batch_first)
            .cuda()
            .to(dtype)
        )
        reference = (
            nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=batch_first)
            .cuda()
            .to(dtype)
        )
        _copy_to_reference(ours, reference)
        shape = (batch, sequence, input_size) if batch_first else (sequence, batch, input_size)
        x = torch.randn(shape, device="cuda", dtype=dtype)
        state = (
            torch.randn(layers, batch, hidden_size, device="cuda", dtype=dtype),
            torch.randn(layers, batch, hidden_size, device="cuda", dtype=dtype),
        )
        with torch.no_grad():
            got = ours(x, state)
            expected = reference(x, state)
        torch.testing.assert_close(got[0], expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1][0], expected[1][0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1][1], expected[1][1], rtol=3e-2, atol=3e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
    def test_rnn_matches_cudnn(self, dtype, nonlinearity):
        torch.manual_seed(4)
        ours = (
            RNN(80, 128, num_layers=2, nonlinearity=nonlinearity, batch_first=True).cuda().to(dtype)
        )
        reference = (
            nn.RNN(80, 128, num_layers=2, nonlinearity=nonlinearity, batch_first=True)
            .cuda()
            .to(dtype)
        )
        _copy_to_reference(ours, reference)
        x = torch.randn(5, 9, 80, device="cuda", dtype=dtype)
        state = torch.randn(2, 5, 128, device="cuda", dtype=dtype)
        with torch.no_grad():
            got = ours(x, state)
            expected = reference(x, state)
        torch.testing.assert_close(got[0], expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1], expected[1], rtol=3e-2, atol=3e-2)

    # The module-level tensor-core gate needs batch >= 16 and hidden >= 1024, so
    # these are the only parametrisations that reach lstm_gemm_layer /
    # rnn_gemm_layer through the layer rather than through an explicit tactic.
    # num_layers > 1 also covers the batch-first handoff: the first layer
    # consumes BTC and every later one consumes the TBC output of its
    # predecessor.
    @pytest.mark.parametrize("layers", [1, 2])
    @pytest.mark.parametrize("batch_first", [False, True])
    def test_lstm_tensor_core_layer_matches_cudnn(self, layers, batch_first):
        torch.manual_seed(7)
        batch, sequence, hidden_size = 16, 3, 1024
        ours = LSTM(hidden_size, hidden_size, num_layers=layers, batch_first=batch_first)
        ours = ours.cuda().half().eval()
        reference = nn.LSTM(hidden_size, hidden_size, num_layers=layers, batch_first=batch_first)
        reference = reference.cuda().half().eval()
        _copy_to_reference(ours, reference)
        shape = (batch, sequence, hidden_size) if batch_first else (sequence, batch, hidden_size)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        state = (
            torch.randn(layers, batch, hidden_size, device="cuda", dtype=torch.float16),
            torch.randn(layers, batch, hidden_size, device="cuda", dtype=torch.float16),
        )
        with torch.no_grad():
            got = ours(x, state)
            expected = reference(x, state)
        assert got[0].shape == expected[0].shape
        torch.testing.assert_close(got[0], expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1][0], expected[1][0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1][1], expected[1][1], rtol=3e-2, atol=3e-2)

    @pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
    @pytest.mark.parametrize("layers,batch_first", [(1, False), (2, True)])
    def test_rnn_tensor_core_layer_matches_cudnn(self, nonlinearity, layers, batch_first):
        torch.manual_seed(8)
        batch, sequence, hidden_size = 16, 3, 1024
        ours = RNN(
            hidden_size,
            hidden_size,
            num_layers=layers,
            nonlinearity=nonlinearity,
            batch_first=batch_first,
        )
        ours = ours.cuda().half().eval()
        reference = nn.RNN(
            hidden_size,
            hidden_size,
            num_layers=layers,
            nonlinearity=nonlinearity,
            batch_first=batch_first,
        )
        reference = reference.cuda().half().eval()
        _copy_to_reference(ours, reference)
        shape = (batch, sequence, hidden_size) if batch_first else (sequence, batch, hidden_size)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        state = torch.randn(layers, batch, hidden_size, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            got = ours(x, state)
            expected = reference(x, state)
        assert got[0].shape == expected[0].shape
        torch.testing.assert_close(got[0], expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1], expected[1], rtol=3e-2, atol=3e-2)

    # The cell history is a two-slice ring, so anything past t=1 wraps it.  A
    # ring indexed as if it were the whole sequence reads a stale slice.
    @pytest.mark.parametrize("sequence", [1, 2, 9])  # 1: no wrap, 2: first wrap, 9: many
    @pytest.mark.parametrize("hidden_size", [64, 1024])
    def test_lstm_cell_ring_matches_cudnn(self, sequence, hidden_size):
        torch.manual_seed(9)
        batch = 16
        ours = LSTM(hidden_size, hidden_size, batch_first=True).cuda().half().eval()
        reference = nn.LSTM(hidden_size, hidden_size, batch_first=True).cuda().half().eval()
        _copy_to_reference(ours, reference)
        x = torch.randn(batch, sequence, hidden_size, device="cuda", dtype=torch.float16)
        state = (
            torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16),
            torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16),
        )
        with torch.no_grad():
            got = ours(x, state)
            expected = reference(x, state)
        torch.testing.assert_close(got[0], expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1][0], expected[1][0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1][1], expected[1][1], rtol=3e-2, atol=3e-2)

    @pytest.mark.parametrize("module_cls,torch_cls", [(LSTM, nn.LSTM), (RNN, nn.RNN)])
    def test_non_contiguous_input_matches_pytorch(self, module_cls, torch_cls):
        """``nn.LSTM``/``nn.RNN`` accept any view; the launcher needs a
        contiguous tensor, so the layer must materialize one rather than refuse.
        """
        torch.manual_seed(10)
        batch, sequence, hidden_size = 4, 3, 64
        ours = module_cls(hidden_size, hidden_size, batch_first=True).cuda().half().eval()
        reference = torch_cls(hidden_size, hidden_size, batch_first=True).cuda().half().eval()
        _copy_to_reference(ours, reference)
        # A BTC view of a TBC tensor: right shape, wrong strides.
        x = torch.randn(sequence, batch, hidden_size, device="cuda", dtype=torch.float16)
        x = x.transpose(0, 1)
        assert not x.is_contiguous()
        state = torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16)
        hx = (state, state.clone()) if module_cls is LSTM else state
        with torch.no_grad():
            got = ours(x, hx)
            expected = reference(x, hx)
        torch.testing.assert_close(got[0], expected[0], rtol=3e-2, atol=3e-2)

    def test_lstm_accepts_non_contiguous_state_view(self):
        """``unstack_states`` hands back a batch window of a wider cohort, which
        is non-contiguous across the layer axis.  Compared against the same call
        on a materialized copy, because ``nn.LSTM`` refuses such a state outright.
        """
        torch.manual_seed(11)
        hidden_size = 64
        ours = LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True).cuda().half().eval()
        cohort_h = torch.randn(2, 8, hidden_size, device="cuda", dtype=torch.float16)
        cohort_c = torch.randn(2, 8, hidden_size, device="cuda", dtype=torch.float16)
        view = (cohort_h[:, 2:5], cohort_c[:, 2:5])
        assert not view[0].is_contiguous()
        x = torch.randn(3, 1, hidden_size, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            from_view = ours(x, view)
            from_copy = ours(x, (view[0].contiguous(), view[1].contiguous()))
        torch.testing.assert_close(from_view[0], from_copy[0])
        torch.testing.assert_close(from_view[1][0], from_copy[1][0])
        torch.testing.assert_close(from_view[1][1], from_copy[1][1])

    def test_functional_destination_passing(self):
        batch, sequence, input_size, hidden_size = 2, 3, 32, 48
        x = torch.randn(batch, sequence, input_size, device="cuda", dtype=torch.float16)
        h = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float16)
        c = torch.randn_like(h)
        weight_ih = torch.randn(4 * hidden_size, input_size, device="cuda", dtype=x.dtype)
        weight_hh = torch.randn(4 * hidden_size, hidden_size, device="cuda", dtype=x.dtype)
        out = torch.empty(batch, sequence, hidden_size, device="cuda", dtype=x.dtype)
        final_h = torch.empty_like(h)
        final_c = torch.empty_like(c)
        result = oasr.lstm_layer(
            x,
            h,
            c,
            weight_ih,
            weight_hh,
            batch_first=True,
            out=out,
            final_h=final_h,
            final_c=final_c,
        )
        assert result[0].data_ptr() == out.data_ptr()
        assert result[1].data_ptr() == final_h.data_ptr()
        assert result[2].data_ptr() == final_c.data_ptr()

    def test_lstm_misaligned_input_matches_cudnn(self):
        torch.manual_seed(6)
        batch, sequence, input_size, hidden_size = 8, 2, 64, 64
        ours = LSTM(input_size, hidden_size, batch_first=True).cuda().half().eval()
        reference = nn.LSTM(input_size, hidden_size, batch_first=True).cuda().half().eval()
        _copy_to_reference(ours, reference)

        storage = torch.randn(batch * sequence * input_size + 1, device="cuda", dtype=torch.float16)
        x = storage[1:].view(batch, sequence, input_size)
        assert x.is_contiguous()
        assert x.data_ptr() % 16 != 0
        state = (
            torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16),
            torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16),
        )

        with torch.no_grad():
            got = ours(x, state)
            expected = reference(x, state)
        torch.testing.assert_close(got[0], expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1][0], expected[1][0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1][1], expected[1][1], rtol=3e-2, atol=3e-2)

    def test_lstm_cuda_graph_capture_replay(self):
        module = LSTM(64, 64, num_layers=2, batch_first=True).cuda().half().eval()
        x = torch.randn(2, 1, 64, device="cuda", dtype=torch.float16)
        state = (
            torch.randn(2, 2, 64, device="cuda", dtype=torch.float16),
            torch.randn(2, 2, 64, device="cuda", dtype=torch.float16),
        )
        stream = torch.cuda.Stream()
        with torch.no_grad(), torch.cuda.stream(stream):
            module(x, state)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
            captured = module(x, state)
        x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        with torch.no_grad(), layers_backend_override("torch"):
            expected = module(x, state)
        torch.testing.assert_close(captured[0], expected[0], rtol=3e-2, atol=3e-2)

    #: ``batch_first`` is a transpose in the Python wrapper and cannot
    #: interact with the CUTLASS tactic, so it is covered once
    #: (:meth:`test_lstm_tensor_core_layer_matches_cudnn` sweeps it) rather
    #: than doubling every tactic.
    @pytest.mark.parametrize("tactic", [(0, 1), (1, 1), (2, 1), (3, 1), (4, 4), (5, 4)])
    @pytest.mark.parametrize("batch_first", [True])
    def test_lstm_cutlass_tactics_match_pytorch(self, tactic, batch_first):
        torch.manual_seed(5)
        batch, sequence, hidden_size = 16, 3, 64
        ours = LSTM(hidden_size, hidden_size, batch_first=batch_first).cuda().half().eval()
        reference = nn.LSTM(hidden_size, hidden_size, batch_first=batch_first).cuda().half().eval()
        _copy_to_reference(ours, reference)
        shape = (batch, sequence, hidden_size) if batch_first else (sequence, batch, hidden_size)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        h = torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16)
        c = torch.randn_like(h)
        with torch.no_grad():
            got = oasr.lstm_gemm_layer(
                x,
                h[0],
                c[0],
                ours.weight_ih_l0,
                ours.weight_hh_l0,
                ours.bias_ih_l0,
                ours.bias_hh_l0,
                batch_first=batch_first,
                _packed_parameters=ours._packed_lstm_parameters(0),
                _tactic=tactic,
            )
            expected = reference(x, (h, c))
        actual_output = got[0].transpose(0, 1) if batch_first else got[0]
        torch.testing.assert_close(actual_output, expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1], expected[1][0][0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[2], expected[1][1][0], rtol=3e-2, atol=3e-2)

    @pytest.mark.parametrize("tactic", [(0, 1), (1, 1), (2, 1), (3, 1), (4, 4)])
    @pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
    @pytest.mark.parametrize("batch_first", [True])
    def test_rnn_cutlass_tactics_match_pytorch(self, tactic, nonlinearity, batch_first):
        torch.manual_seed(6)
        batch, sequence, hidden_size = 16, 3, 64
        ours = (
            RNN(
                hidden_size,
                hidden_size,
                nonlinearity=nonlinearity,
                batch_first=batch_first,
            )
            .cuda()
            .half()
            .eval()
        )
        reference = (
            nn.RNN(
                hidden_size,
                hidden_size,
                nonlinearity=nonlinearity,
                batch_first=batch_first,
            )
            .cuda()
            .half()
            .eval()
        )
        _copy_to_reference(ours, reference)
        shape = (batch, sequence, hidden_size) if batch_first else (sequence, batch, hidden_size)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        h = torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            got = oasr.rnn_gemm_layer(
                x,
                h[0],
                ours.weight_ih_l0,
                ours.weight_hh_l0,
                ours.bias_ih_l0,
                ours.bias_hh_l0,
                nonlinearity=nonlinearity,
                batch_first=batch_first,
                _combined_input_bias=ours._combined_rnn_bias(0),
                _tactic=tactic,
            )
            expected = reference(x, h)
        actual_output = got[0].transpose(0, 1) if batch_first else got[0]
        torch.testing.assert_close(actual_output, expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1], expected[1][0], rtol=3e-2, atol=3e-2)

    @_requires_tma
    @pytest.mark.parametrize("tactic", _TMA_TACTICS)
    @pytest.mark.parametrize("batch_first", [True])
    def test_lstm_tma_tactics_match_pytorch(self, tactic, batch_first):
        torch.manual_seed(12)
        batch, sequence, hidden_size = 128, 3, 256
        ours = LSTM(hidden_size, hidden_size, batch_first=batch_first).cuda().half().eval()
        reference = nn.LSTM(hidden_size, hidden_size, batch_first=batch_first).cuda().half().eval()
        _copy_to_reference(ours, reference)
        shape = (batch, sequence, hidden_size) if batch_first else (sequence, batch, hidden_size)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        h = torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16)
        c = torch.randn_like(h)
        with torch.no_grad():
            got = oasr.lstm_gemm_layer(
                x,
                h[0],
                c[0],
                ours.weight_ih_l0,
                ours.weight_hh_l0,
                ours.bias_ih_l0,
                ours.bias_hh_l0,
                batch_first=batch_first,
                _packed_parameters=ours._packed_lstm_parameters(0),
                _tactic=tactic,
            )
            expected = reference(x, (h, c))
        actual = got[0].transpose(0, 1) if batch_first else got[0]
        torch.testing.assert_close(actual, expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1], expected[1][0][0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[2], expected[1][1][0], rtol=3e-2, atol=3e-2)

    @_requires_tma
    @pytest.mark.parametrize("tactic", _TMA_TACTICS)
    @pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
    def test_rnn_tma_tactics_match_pytorch(self, tactic, nonlinearity):
        torch.manual_seed(13)
        batch, sequence, hidden_size = 128, 3, 256
        ours = (
            RNN(hidden_size, hidden_size, nonlinearity=nonlinearity, batch_first=True)
            .cuda()
            .half()
            .eval()
        )
        reference = (
            nn.RNN(hidden_size, hidden_size, nonlinearity=nonlinearity, batch_first=True)
            .cuda()
            .half()
            .eval()
        )
        _copy_to_reference(ours, reference)
        x = torch.randn(batch, sequence, hidden_size, device="cuda", dtype=torch.float16)
        h = torch.randn(1, batch, hidden_size, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            got = oasr.rnn_gemm_layer(
                x,
                h[0],
                ours.weight_ih_l0,
                ours.weight_hh_l0,
                ours.bias_ih_l0,
                ours.bias_hh_l0,
                nonlinearity=nonlinearity,
                batch_first=True,
                _combined_input_bias=ours._combined_rnn_bias(0),
                _tactic=tactic,
            )
            expected = reference(x, h)
        torch.testing.assert_close(got[0].transpose(0, 1), expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(got[1], expected[1][0], rtol=3e-2, atol=3e-2)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or device_sm() in (90, 100),
        reason="covers the refusal on targets that do not build the TMA arms",
    )
    @pytest.mark.parametrize("tactic", _TMA_TACTICS)
    def test_tma_tactics_declared_unavailable_off_sm90(self, tactic):
        """A tactic the target did not compile must refuse, not silently reroute."""
        batch, sequence, hidden_size = 16, 2, 64
        x = torch.randn(sequence, batch, hidden_size, device="cuda", dtype=torch.float16)
        h = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float16)
        weight_ih = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float16)
        weight_hh = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="TMA warp-specialized"):
            oasr.rnn_gemm_layer(x, h, weight_ih, weight_hh, _tactic=tactic)

    def test_rejects_unknown_tactic(self):
        batch, sequence, hidden_size = 16, 2, 64
        x = torch.randn(sequence, batch, hidden_size, device="cuda", dtype=torch.float16)
        h = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float16)
        weight_ih = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float16)
        weight_hh = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="unknown recurrent GEMM tactic"):
            oasr.rnn_gemm_layer(x, h, weight_ih, weight_hh, _tactic=(8, 1))

    def test_rnn_gemm_rejects_serial_split_k(self):
        """Applying tanh/ReLU to an intermediate K partition is wrong, so the
        launcher refuses the tactic instead of producing a plausible answer.
        """
        batch, sequence, hidden_size = 16, 3, 64
        x = torch.randn(sequence, batch, hidden_size, device="cuda", dtype=torch.float16)
        h = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float16)
        weight_ih = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float16)
        weight_hh = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="serial split-K"):
            oasr.rnn_gemm_layer(x, h, weight_ih, weight_hh, _tactic=(5, 4))


class TestRecurrentValidation:
    def test_functional_rejects_rank_before_jit(self):
        with pytest.raises(ValueError, match="3-D"):
            oasr.lstm_layer(
                torch.randn(2, 8),
                torch.randn(2, 8),
                torch.randn(2, 8),
                torch.randn(32, 8),
                torch.randn(32, 8),
            )

    def test_module_rejects_wrong_input_width(self):
        with pytest.raises(RuntimeError, match="input_size"):
            LSTM(8, 16)(torch.randn(3, 2, 7))


class TestRecurrentSlotStep:
    """Slot-addressed single timestep -- the continuous-batching primitive.

    The oracle is the dense path: gather the same rows and run the validated
    ``lstm_layer`` / ``rnn_layer`` at T=1.  Both compute the same equation with
    the same reduction order, so agreement here should be exact, not approximate.
    """

    @staticmethod
    def _weights(gates, hidden, input_size, dtype):
        g = torch.Generator(device="cuda").manual_seed(11)
        k = input_size**-0.5
        return (
            torch.randn(gates * hidden, input_size, device="cuda", dtype=dtype, generator=g) * k,
            torch.randn(gates * hidden, hidden, device="cuda", dtype=dtype, generator=g)
            * hidden**-0.5,
            torch.randn(gates * hidden, device="cuda", dtype=dtype, generator=g) * 0.1,
            torch.randn(gates * hidden, device="cuda", dtype=dtype, generator=g) * 0.1,
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("hidden,input_size", [(640, 640), (256, 320), (64, 64)])
    def test_lstm_slot_step_matches_dense(self, device, dtype, hidden, input_size):
        rows, slots = 6, 8
        wih, whh, bih, bhh = self._weights(4, hidden, input_size, dtype)
        g = torch.Generator(device="cuda").manual_seed(5)
        x = torch.randn(rows, input_size, device="cuda", dtype=dtype, generator=g) * 0.5
        h0 = torch.randn(slots, hidden, device="cuda", dtype=dtype, generator=g) * 0.3
        c0 = torch.randn(slots, hidden, device="cuda", dtype=dtype, generator=g) * 0.3
        slot_ids = torch.tensor([5, 0, 7, 2, 1, 6], device="cuda", dtype=torch.int64)
        # Mixed parities: half the rows currently live in the other ring slice.
        parity = torch.tensor([1, 0, 1, 0, 1, 0], device="cuda", dtype=torch.int32)
        ring = torch.zeros(2, slots, hidden, device="cuda", dtype=dtype)
        for p, s in zip(parity.tolist(), slot_ids.tolist()):
            ring[p, s] = h0[s]
        cells = c0.clone()

        out = oasr.lstm_slot_step(x, ring, cells, slot_ids, parity, wih, whh, bih, bhh)
        ref_out, ref_h, ref_c = oasr.lstm_layer(
            x.unsqueeze(0), h0[slot_ids].contiguous(), c0[slot_ids].contiguous(), wih, whh, bih, bhh
        )
        written = torch.stack([ring[1 - p, s] for p, s in zip(parity.tolist(), slot_ids.tolist())])
        torch.testing.assert_close(out, ref_out[0], rtol=0, atol=0)
        torch.testing.assert_close(written, ref_h, rtol=0, atol=0)
        torch.testing.assert_close(cells[slot_ids], ref_c, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
    def test_rnn_slot_step_matches_dense(self, device, dtype, nonlinearity):
        hidden = input_size = 512
        rows, slots = 5, 8
        wih, whh, bih, bhh = self._weights(1, hidden, input_size, dtype)
        g = torch.Generator(device="cuda").manual_seed(6)
        x = torch.randn(rows, input_size, device="cuda", dtype=dtype, generator=g) * 0.5
        h0 = torch.randn(slots, hidden, device="cuda", dtype=dtype, generator=g) * 0.3
        slot_ids = torch.tensor([3, 0, 6, 1, 7], device="cuda", dtype=torch.int64)
        parity = torch.tensor([0, 1, 1, 0, 1], device="cuda", dtype=torch.int32)
        ring = torch.zeros(2, slots, hidden, device="cuda", dtype=dtype)
        for p, s in zip(parity.tolist(), slot_ids.tolist()):
            ring[p, s] = h0[s]

        out = oasr.rnn_slot_step(
            x, ring, slot_ids, parity, wih, whh, bih, bhh, nonlinearity=nonlinearity
        )
        ref_out, ref_h = oasr.rnn_layer(
            x.unsqueeze(0),
            h0[slot_ids].contiguous(),
            wih,
            whh,
            bih,
            bhh,
            nonlinearity=nonlinearity,
        )
        written = torch.stack([ring[1 - p, s] for p, s in zip(parity.tolist(), slot_ids.tolist())])
        torch.testing.assert_close(out, ref_out[0], rtol=0, atol=0)
        torch.testing.assert_close(written, ref_h, rtol=0, atol=0)

    def test_slot_step_leaves_inactive_slots_alone(self, device):
        """A slot not named this tick must not move -- an idle stream keeps its state."""
        hidden = input_size = 128
        slots = 8
        wih, whh, bih, bhh = self._weights(4, hidden, input_size, torch.float16)
        g = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn(3, input_size, device="cuda", dtype=torch.float16, generator=g)
        ring = torch.randn(2, slots, hidden, device="cuda", dtype=torch.float16, generator=g) * 0.2
        cells = torch.randn(slots, hidden, device="cuda", dtype=torch.float16, generator=g) * 0.2
        before_ring, before_cells = ring.clone(), cells.clone()
        slot_ids = torch.tensor([1, 4, 6], device="cuda", dtype=torch.int64)
        parity = torch.zeros(3, device="cuda", dtype=torch.int32)

        oasr.lstm_slot_step(x, ring, cells, slot_ids, parity, wih, whh, bih, bhh)

        untouched = [s for s in range(slots) if s not in slot_ids.tolist()]
        torch.testing.assert_close(ring[:, untouched], before_ring[:, untouched], rtol=0, atol=0)
        torch.testing.assert_close(cells[untouched], before_cells[untouched], rtol=0, atol=0)
        # The read slice of the *active* slots is untouched too; only 1-parity moves.
        torch.testing.assert_close(ring[0, slot_ids], before_ring[0, slot_ids], rtol=0, atol=0)

    def test_slot_step_rejects_bad_metadata(self, device):
        hidden = input_size = 64
        wih, whh, _, _ = self._weights(4, hidden, input_size, torch.float16)
        x = torch.randn(2, input_size, device="cuda", dtype=torch.float16)
        ring = torch.zeros(2, 4, hidden, device="cuda", dtype=torch.float16)
        cells = torch.zeros(4, hidden, device="cuda", dtype=torch.float16)
        slot_ids = torch.zeros(2, device="cuda", dtype=torch.int64)
        parity = torch.zeros(2, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="int32"):
            oasr.lstm_slot_step(x, ring, cells, slot_ids, parity.long(), wih, whh)
        with pytest.raises(ValueError, match="int64"):
            oasr.lstm_slot_step(x, ring, cells, slot_ids.int(), parity, wih, whh)
        with pytest.raises(ValueError, match=r"\(2, slots, hidden\) ring"):
            oasr.lstm_slot_step(x, ring[0], cells, slot_ids, parity, wih, whh)
        with pytest.raises(ValueError, match="slot ids"):
            oasr.lstm_slot_step(x, ring, cells, slot_ids[:1], parity, wih, whh)
        with pytest.raises(ValueError, match="read_parity"):
            oasr.lstm_slot_step(x, ring, cells, slot_ids, parity[:1], wih, whh)


class TestRecurrentInferenceTensors:
    """A module built or moved inside ``torch.inference_mode()``.

    ``Tensor._version`` raises on an inference tensor, and the packed-parameter
    cache read it on every forward — so constructing the layer inside
    ``inference_mode`` took down the *first forward* with
    ``RuntimeError: Inference tensors do not track version counter``, far from the
    construction that caused it.  An inference tensor cannot be mutated in place,
    so there is nothing for the counter to guard.
    """

    def test_layer_built_inside_inference_mode_still_runs(self, device):
        from oasr.layers.recurrent import LSTM

        with torch.inference_mode():
            layer = LSTM(16, 16, num_layers=1).to(device, torch.float16).eval()
            x = torch.randn(2, 3, 16, dtype=torch.float16, device=device)
            out, (h, c) = layer(x)
        assert out.shape == (2, 3, 16)
        assert torch.isfinite(out).all()

    def test_packed_parameters_are_reused_across_steps(self, device):
        """The fast slot must be a hit, not a rebuild, on the second call."""
        from oasr.layers.recurrent import LSTM

        layer = LSTM(16, 16, num_layers=2).to(device, torch.float16).eval()
        first = layer._packed_lstm_parameters(0)
        second = layer._packed_lstm_parameters(0)
        assert all(a is b for a, b in zip(first, second) if a is not None)

    def test_an_in_place_weight_edit_invalidates(self, device):
        """The version guard is what makes the fast slot safe to keep."""
        from oasr.layers.recurrent import LSTM

        layer = LSTM(16, 16, num_layers=1).to(device, torch.float16).eval()
        before = layer._packed_lstm_parameters(0)[0].clone()
        with torch.no_grad():
            layer.weight_ih_l0.add_(1.0)
        after = layer._packed_lstm_parameters(0)[0]
        assert not torch.equal(before, after), "an in-place weight edit was served from cache"

    def test_moving_the_module_invalidates(self, device):
        from oasr.layers.recurrent import LSTM

        layer = LSTM(16, 16, num_layers=1).to(device, torch.float16).eval()
        packed = layer._packed_lstm_parameters(0)[0]
        assert packed.dtype is torch.float16
        layer.to(torch.float32)
        assert layer._packed_lstm_parameters(0)[0].dtype is torch.float32


class TestPackWarning:
    """A direct functional caller repacks the weights on every call.

    ``_pack_lstm_parameters`` is two permute-copies of the whole weight matrix —
    40.9 us at LSTM(640, 640), more than the timestep it feeds — and is meant to
    run once per weight set.  ``oasr.layers.LSTM`` caches it and threads the
    result in; a caller reaching for ``oasr.lstm_gemm_layer`` directly gets no
    cache, and nothing used to say so.
    """

    def _reset(self):
        import oasr.functionals.recurrent as fr

        fr._PACK_WARNED = False

    def test_a_direct_caller_is_warned_once(self, device, caplog):
        import logging

        import oasr

        self._reset()
        H = 16
        wih = torch.randn(4 * H, H, dtype=torch.float16, device=device) * 0.02
        whh = torch.randn(4 * H, H, dtype=torch.float16, device=device) * 0.02
        x = torch.randn(1, 4, H, dtype=torch.float16, device=device)
        h = torch.zeros(4, H, dtype=torch.float16, device=device)
        c = torch.zeros(4, H, dtype=torch.float16, device=device)
        with caplog.at_level(logging.WARNING, logger="oasr.functionals.recurrent"):
            for _ in range(3):
                oasr.lstm_gemm_layer(x, h, c, wih, whh)
        hits = [r for r in caplog.records if "_pack_lstm_parameters" in r.getMessage()]
        assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"
        assert "_packed_parameters" in hits[0].getMessage(), "the warning must name the fix"

    def test_the_layer_is_not_warned(self, device, caplog):
        import logging

        from oasr.layers.recurrent import LSTM

        self._reset()
        layer = LSTM(16, 16, num_layers=2).to(device, torch.float16).eval()
        x = torch.randn(1, 4, 16, dtype=torch.float16, device=device)
        with caplog.at_level(logging.WARNING, logger="oasr.functionals.recurrent"):
            for _ in range(3):
                layer(x)
        hits = [r for r in caplog.records if "_pack_lstm_parameters" in r.getMessage()]
        assert not hits, "the layer caches the packed parameters and must stay quiet"


# ---------------------------------------------------------------------------
# The CuTeDSL fused recurrent step
#
# A second backend for the same recurrence, selected by ``OASR_RECURRENT_CUTE``
# and the routing table below. It lives with the CUTLASS arms because the
# oracle is the same written-out recurrence and a routing change moves work
# between them.
# ---------------------------------------------------------------------------

cutlass = pytest.importorskip("cutlass", reason="CuTeDSL (nvidia-cutlass-dsl) not installed")

from oasr.jit import recurrent_cute  # noqa: E402

_requires_cute = requires_cute(recurrent_cute, "recurrent-step")


def _cute_oracle(a, weight, c, prev_c, gates):
    """FP32 reference: the equations, not another kernel."""
    acc = a.float() @ weight.float().T + c.float()
    if gates == 1:
        return torch.tanh(acc), None
    gv = acc.view(a.shape[0], -1, 4)
    cell = torch.sigmoid(gv[..., 1]) * prev_c.float() + torch.sigmoid(gv[..., 0]) * torch.tanh(
        gv[..., 2]
    )
    return torch.sigmoid(gv[..., 3]) * torch.tanh(cell), cell


def _cute_run(dtype, hidden, batch, gates, activation, tile):
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
    ref_h, ref_c = _cute_oracle(a, weight, c, prev_c, gates)
    return out_h, out_c, ref_h, ref_c


@_requires_cute
@pytest.mark.cuda
class TestRecurrentStepCute:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("hidden,batch", [(256, 64), (640, 32), (640, 130), (1024, 8)])
    def test_lstm_matches_fp32_equations(self, device, dtype, hidden, batch):
        tile = recurrent_cute.select_tile(hidden, batch)
        assert tile is not None
        out_h, out_c, ref_h, ref_c = _cute_run(dtype, hidden, batch, 4, "lstm", tile)
        # FP16 accumulation over K=hidden against an FP32 oracle; 2e-2 is the
        # tolerance the rest of the recurrent suite uses for the same comparison.
        torch.testing.assert_close(out_h.float(), ref_h, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_c.float(), ref_c, rtol=2e-2, atol=2e-2)

    def test_rnn_matches_fp32_equations(self, device):
        # ``tanh`` was a one-element parametrize; it is the only nonlinearity
        # the CuTe step compiles, so it is a constant, not an axis.
        out_h, _, ref_h, _ = _cute_run(torch.float16, 640, 64, 1, "tanh", (32, 64, 64, 3, 128, 2))
        torch.testing.assert_close(out_h.float(), ref_h, rtol=2e-2, atol=2e-2)

    def test_batch_need_not_be_a_tile_multiple(self, device):
        """A ragged M must be predicated, not rounded up into other rows' memory."""
        for batch in (1, 7, 33):
            out_h, out_c, ref_h, ref_c = _cute_run(
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


@pytest.mark.cuda
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


@pytest.mark.cuda
class TestRoutedStepMemo:
    """``routed_step`` memoises band + arch probe + compile behind one lookup.

    That removed 1.18 us per layer per timestep (two table scans and a
    ``functools.cache`` key build, twice over for a two-layer predictor).  The
    hazard it introduces is staleness: a memo that survives ``set_mode`` would
    make ``OASR_RECURRENT_CUTE=off`` -- the rollback switch -- do nothing.
    """

    def test_set_mode_invalidates_the_memo(self, device):
        from oasr.jit import recurrent_cute as rc

        before = rc.get_mode()
        try:
            rc.set_mode("auto")
            routed = rc.routed_step(
                dtype_str="float16", gate_count=4, activation="lstm", hidden=256, batch=128
            )
            if routed is None:
                pytest.skip("this shape is not routed on this device")
            rc.set_mode("off")
            assert (
                rc.routed_step(
                    dtype_str="float16", gate_count=4, activation="lstm", hidden=256, batch=128
                )
                is None
            ), "mode=off was served from the route memo"
            rc.set_mode("auto")
            assert (
                rc.routed_step(
                    dtype_str="float16", gate_count=4, activation="lstm", hidden=256, batch=128
                )
                is not None
            )
        finally:
            rc.set_mode(before)

    def test_declines_outside_the_band_without_compiling(self, device):
        from oasr.jit import recurrent_cute as rc

        before = rc.get_mode()
        try:
            rc.set_mode("auto")
            # gate_count 1 (vanilla RNN) is never routed under auto.
            assert (
                rc.routed_step(
                    dtype_str="float16", gate_count=1, activation="tanh", hidden=256, batch=128
                )
                is None
            )
        finally:
            rc.set_mode(before)
