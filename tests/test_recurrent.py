#!/usr/bin/env python3
"""Correctness and contract tests for custom LSTM and vanilla RNN layers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import oasr
from oasr.layers import LSTM, RNN, layers_backend_override


def _copy_to_reference(ours: nn.Module, reference: nn.Module) -> None:
    reference.load_state_dict(ours.state_dict())


def _device_sm() -> int:
    """Compute capability as the JIT spells it, e.g. 90 for Hopper."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


#: The CUTLASS 3.x recurrent arms are compiled only for these two targets.
_TMA_TACTICS = [(6, 1), (7, 1)]
_requires_tma = pytest.mark.skipif(
    not torch.cuda.is_available() or _device_sm() not in (90, 100),
    reason="the TMA warp-specialized recurrent tactics are built for SM90/SM100 only",
)


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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="recurrent kernels need CUDA")
class TestRecurrentCuda:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "batch,sequence,input_size,hidden_size,layers,batch_first",
        [
            (1, 1, 64, 64, 1, True),
            (3, 5, 48, 64, 2, True),
            (8, 3, 64, 64, 2, True),
            (8, 2, 66, 70, 1, True),
            (8, 2, 63, 65, 1, True),
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
    @pytest.mark.parametrize("layers", [1, 2])
    @pytest.mark.parametrize("batch_first", [False, True])
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
    @pytest.mark.parametrize("sequence", [1, 2, 3, 9])
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

    @pytest.mark.parametrize("tactic", [(0, 1), (1, 1), (2, 1), (3, 1), (4, 4), (5, 4)])
    @pytest.mark.parametrize("batch_first", [False, True])
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
    @pytest.mark.parametrize("batch_first", [False, True])
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
    @pytest.mark.parametrize("batch_first", [False, True])
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
        not torch.cuda.is_available() or _device_sm() in (90, 100),
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


class TestRecurrentContinuousBatching:
    """Timestep-granular continuous batching against a per-sequence oracle."""

    LENGTHS = [7, 3, 11, 2, 9, 5, 13, 4, 6, 8]

    def _drive(self, module, cache, batcher, sequences):
        collected = {key: [] for key in sequences}
        ticks = 0
        while True:
            plan = batcher.next_step()
            if plan is None:
                break
            out = module.step(plan.frames, cache, plan.slot_ids, plan.read_parity)
            assert out.shape == (len(plan.stream_ids), module.hidden_size)
            for row, key in enumerate(plan.stream_ids):
                collected[key].append(out[row].clone())
            batcher.commit(plan)
            ticks += 1
        return collected, ticks

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("cls,layers", [(LSTM, 2), (RNN, 1)])
    def test_matches_per_sequence_forward(self, device, dtype, cls, layers):
        from oasr.cache import RecurrentContinuousBatcher, RecurrentStateCache

        hidden = width = 256
        slots = 4
        module = cls(width, hidden, layers).cuda().to(dtype)
        cache = RecurrentStateCache(
            layers, hidden, slots, torch.device("cuda"), dtype, cell=(cls is LSTM)
        )
        batcher = RecurrentContinuousBatcher(cache, width)
        g = torch.Generator(device="cuda").manual_seed(21)
        sequences = {
            i: torch.randn(n, width, device="cuda", dtype=dtype, generator=g) * 0.5
            for i, n in enumerate(self.LENGTHS)
        }
        for key, frames in sequences.items():
            batcher.submit(key, frames)

        collected, ticks = self._drive(module, cache, batcher, sequences)

        # Every frame was stepped exactly once, and no cohort ran past its length.
        assert ticks >= max(self.LENGTHS)
        assert sum(len(v) for v in collected.values()) == sum(self.LENGTHS)
        for key, frames in sequences.items():
            assert len(collected[key]) == frames.shape[0]
            expected = module(frames.unsqueeze(1))[0][:, 0]
            torch.testing.assert_close(torch.stack(collected[key]), expected, rtol=2e-2, atol=2e-2)

    def test_slots_are_recycled_not_leaked(self, device):
        from oasr.cache import RecurrentContinuousBatcher, RecurrentStateCache

        cache = RecurrentStateCache(1, 32, 2, torch.device("cuda"), torch.float16)
        batcher = RecurrentContinuousBatcher(cache, 32)
        module = LSTM(32, 32, 1).cuda().half()
        for i, n in enumerate([1, 2, 3, 1, 2]):
            batcher.submit(i, torch.randn(n, 32, device="cuda", dtype=torch.float16))
        seen_peak = 0
        while True:
            plan = batcher.next_step()
            if plan is None:
                break
            module.step(plan.frames, cache, plan.slot_ids, plan.read_parity)
            batcher.commit(plan)
            seen_peak = max(seen_peak, len(plan.stream_ids))
        # Five streams through two slots: capacity was reused, never exceeded.
        assert seen_peak == 2
        assert batcher.active == 0 and batcher.pending == 0
        assert not batcher

    def test_fresh_slot_starts_from_zero_state(self, device):
        """An admitted stream must see zero h/c, not the retired stream's tail."""
        from oasr.cache import RecurrentContinuousBatcher, RecurrentStateCache

        hidden = width = 64
        module = LSTM(width, hidden, 1).cuda().half()
        cache = RecurrentStateCache(1, hidden, 1, torch.device("cuda"), torch.float16)
        batcher = RecurrentContinuousBatcher(cache, width)
        g = torch.Generator(device="cuda").manual_seed(31)
        first = torch.randn(4, width, device="cuda", dtype=torch.float16, generator=g)
        second = torch.randn(3, width, device="cuda", dtype=torch.float16, generator=g)
        batcher.submit("first", first)
        batcher.submit("second", second)
        out = {"first": [], "second": []}
        while True:
            plan = batcher.next_step()
            if plan is None:
                break
            y = module.step(plan.frames, cache, plan.slot_ids, plan.read_parity)
            out[plan.stream_ids[0]].append(y[0].clone())
            batcher.commit(plan)
        # The second stream ran on the slot the first one released.
        expected = module(second.unsqueeze(1))[0][:, 0]
        torch.testing.assert_close(torch.stack(out["second"]), expected, rtol=2e-2, atol=2e-2)

    def test_step_rejects_geometry_and_dtype_mismatch(self, device):
        from oasr.cache import RecurrentStateCache

        cache = RecurrentStateCache(1, 32, 2, torch.device("cuda"), torch.float16)
        module = LSTM(32, 32, 1).cuda().half()
        slot_ids = torch.zeros(1, device="cuda", dtype=torch.int64)
        parity = torch.zeros(1, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="step frames"):
            module.step(
                torch.randn(1, 7, device="cuda", dtype=torch.float16), cache, slot_ids, parity
            )
        with pytest.raises(ValueError, match="does not match"):
            LSTM(32, 64, 1).cuda().half().step(
                torch.randn(1, 32, device="cuda", dtype=torch.float16), cache, slot_ids, parity
            )
        with pytest.raises(NotImplementedError, match="no torch fallback"):
            module.float().step(
                torch.randn(1, 32, device="cuda", dtype=torch.float32), cache, slot_ids, parity
            )
        rnn_cache = RecurrentStateCache(1, 32, 2, torch.device("cuda"), torch.float16, cell=False)
        with pytest.raises(ValueError, match="cell state"):
            LSTM(32, 32, 1).cuda().half().step(
                torch.randn(1, 32, device="cuda", dtype=torch.float16),
                rnn_cache,
                slot_ids,
                parity,
            )

    def test_long_run_compacts_retired_frames(self, device):
        """A batcher that outlives its streams must not retain every one of them.

        Retired frames stay addressable until half the packed buffer is dead, so
        the buffer is bounded by the live corpus rather than by everything ever
        submitted -- and the streams that ran after a compaction must still be
        correct, which is what would break if a base offset went stale.
        """
        from oasr.cache import RecurrentContinuousBatcher, RecurrentStateCache

        hidden = width = 64
        slots = 4
        module = LSTM(width, hidden, 1).cuda().half()
        cache = RecurrentStateCache(1, hidden, slots, torch.device("cuda"), torch.float16)
        batcher = RecurrentContinuousBatcher(cache, width)
        g = torch.Generator(device="cuda").manual_seed(41)
        lengths = [3, 9, 5, 12, 4, 7, 6, 11, 2, 8] * 6
        sequences = {
            i: torch.randn(n, width, device="cuda", dtype=torch.float16, generator=g) * 0.4
            for i, n in enumerate(lengths)
        }
        # Submitted in waves, as a server receives them -- all-up-front would make
        # the first pack hold the whole corpus by definition and prove nothing.
        pending = list(sequences.items())
        collected = {key: [] for key in sequences}
        peak_packed = 0
        while pending or batcher:
            for key, frames in pending[:10]:
                batcher.submit(key, frames)
            pending = pending[10:]
            for _ in range(20):
                plan = batcher.next_step()
                if plan is None:
                    break
                out = module.step(plan.frames, cache, plan.slot_ids, plan.read_parity)
                for row, key in enumerate(plan.stream_ids):
                    collected[key].append(out[row].clone())
                batcher.commit(plan)
                peak_packed = max(peak_packed, batcher._packed.shape[0])

        assert sum(len(v) for v in collected.values()) == sum(lengths)
        # Compaction happened: the buffer never had to hold every submission.
        assert peak_packed < sum(lengths)
        # The last few streams ran entirely after compactions rebased the buffer.
        for key in list(sequences)[-4:]:
            expected = module(sequences[key].unsqueeze(1))[0][:, 0]
            torch.testing.assert_close(torch.stack(collected[key]), expected, rtol=2e-2, atol=2e-2)


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
