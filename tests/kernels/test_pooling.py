#!/usr/bin/env python3
"""Correctness and contract tests for AvgPool1D."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import oasr


def _reference(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
) -> torch.Tensor:
    return (
        F.avg_pool1d(
            x.transpose(-2, -1),
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        )
        .transpose(-2, -1)
        .contiguous()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="AvgPool1D kernel needs CUDA")
class TestAvgPool1d:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize(
        "shape,kernel,stride,padding,ceil_mode,count_include_pad",
        [
            ((2, 16, 128), 2, 2, 0, False, True),
            ((2, 17, 13), 2, 2, 0, True, True),
            ((2, 17, 128), 3, 2, 1, False, True),
            ((2, 16, 128), 3, 2, 1, True, False),
            ((19, 64), 5, 3, 2, True, False),
        ],
    )
    def test_matches_torch(
        self,
        dtype,
        shape,
        kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    ):
        torch.manual_seed(0)
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        got = oasr.avg_pool1d(
            x,
            kernel,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        )
        ref = _reference(x, kernel, stride, padding, ceil_mode, count_include_pad)
        tolerance = 1e-5 if dtype is torch.float32 else 2e-2
        torch.testing.assert_close(got, ref, rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_production_shape_and_destination_passing(self, dtype):
        x = torch.randn(1, 1500, 1280, device="cuda", dtype=dtype)
        out = torch.empty(1, 750, 1280, device="cuda", dtype=dtype)
        result = oasr.avg_pool1d(x, 2, stride=2, out=out)
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result, _reference(x, 2, 2, 0, False, True))

    def test_rejects_wrong_destination_shape(self):
        x = torch.randn(2, 16, 32, device="cuda", dtype=torch.float16)
        out = torch.empty(2, 9, 32, device="cuda", dtype=torch.float16)
        with pytest.raises(Exception, match="output must have T=8 C=32"):
            oasr.avg_pool1d(x, 2, 2, out=out)

    def test_rejects_noncontiguous_input(self):
        x = torch.randn(2, 32, 16, device="cuda", dtype=torch.float16).transpose(1, 2)
        assert not x.is_contiguous()
        with pytest.raises(Exception, match="contiguous"):
            oasr.avg_pool1d(x, 2, 2)

    def test_cuda_graph_capture_replay(self):
        x = torch.randn(2, 32, 64, device="cuda", dtype=torch.float16)
        out = torch.empty(2, 16, 64, device="cuda", dtype=torch.float16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            oasr.avg_pool1d(x, 2, 2, out=out)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            oasr.avg_pool1d(x, 2, 2, out=out)

        x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, _reference(x, 2, 2, 0, False, True))


class TestAvgPool1dValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"kernel_size": 0}, "kernel_size must be positive"),
            ({"kernel_size": 2, "stride": 0}, "stride must be positive"),
            ({"kernel_size": 2, "padding": 2}, "at most half"),
            ({"kernel_size": (2, 3)}, "one-element tuple"),
        ],
    )
    def test_invalid_arguments_fail_before_jit(self, kwargs, match):
        with pytest.raises((TypeError, ValueError), match=match):
            oasr.avg_pool1d(torch.randn(2, 8, 4), **kwargs)

    def test_invalid_rank_fails_before_jit(self):
        with pytest.raises(ValueError, match="TC or BTC"):
            oasr.avg_pool1d(torch.randn(8), 2)

    def test_invalid_output_length_fails_before_jit(self):
        with pytest.raises(ValueError, match="invalid output length"):
            oasr.avg_pool1d(torch.randn(2, 1, 4), 4)
