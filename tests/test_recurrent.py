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
