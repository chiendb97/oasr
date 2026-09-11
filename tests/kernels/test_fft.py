#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.rfft`` / ``oasr.rfft_power`` (``oasr/functionals/fft.py``).

``n_fft`` is the only axis that selects a radix decomposition, so the full
power-of-two ladder is swept once, for ``rfft``.  ``rfft_power`` is
``|rfft|**2`` over the same transform, so it re-tests the ladder's ends rather
than all of it; the leading batch dims are a flatten in the wrapper.
"""

import pytest
import torch
from helpers import assert_dest_passing

import oasr

pytestmark = pytest.mark.cuda


N_FFT_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


class TestRfft:
    """Tests for oasr.rfft (real-to-complex FFT)."""

    @pytest.mark.parametrize("n_fft", N_FFT_VALUES)
    def test_rfft_1d(self, n_fft):
        torch.manual_seed(0)
        x = torch.randn(n_fft, device="cuda", dtype=torch.float32)

        out = oasr.rfft(x)
        ref = torch.fft.rfft(x)

        assert out.shape == ref.shape
        assert out.dtype == torch.complex64
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("n_fft", [256, 1024])
    @pytest.mark.parametrize("batch_shape", [(4,), (3, 5, 7)])
    def test_rfft_batched(self, n_fft, batch_shape):
        torch.manual_seed(0)
        x = torch.randn(*batch_shape, n_fft, device="cuda", dtype=torch.float32)

        out = oasr.rfft(x)
        ref = torch.fft.rfft(x)

        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("frame_length,n_fft", [(400, 512), (160, 256), (200, 256)])
    def test_rfft_pad(self, frame_length, n_fft):
        """rfft should zero-pad when input length < n_fft (matches torch.fft.rfft)."""
        torch.manual_seed(0)
        x = torch.randn(8, frame_length, device="cuda", dtype=torch.float32)

        out = oasr.rfft(x, n=n_fft)
        ref = torch.fft.rfft(x, n=n_fft)

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)

    def test_rfft_destination_passing(self):
        x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
        out = torch.empty(4, 257, device="cuda", dtype=torch.complex64)
        assert_dest_passing(oasr.rfft, x, out=out)
        torch.testing.assert_close(out, torch.fft.rfft(x), rtol=1e-4, atol=1e-3)

    def test_rfft_invalid_n_fft(self):
        x = torch.randn(8, 500, device="cuda", dtype=torch.float32)
        with pytest.raises(ValueError):
            oasr.rfft(x, n=500)  # not a power of two
        with pytest.raises(ValueError):
            oasr.rfft(x, n=4)  # below the supported minimum
        with pytest.raises(ValueError):
            oasr.rfft(x, n=4096)  # above the supported maximum

    def test_rfft_dtype_error(self):
        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError):
            oasr.rfft(x)

    def test_rfft_cpu_error(self):
        x = torch.randn(4, 256, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="CUDA tensor"):
            oasr.rfft(x)


class TestRfftPower:
    """Tests for oasr.rfft_power (real-FFT power spectrum)."""

    @pytest.mark.parametrize("n_fft", [8, 512, 2048])
    def test_power_1d(self, n_fft):
        """The ends and the middle of the radix ladder.

        ``rfft_power`` is ``|rfft|**2``, and :meth:`TestRfft.test_rfft_1d`
        already walks every width -- re-walking it here would re-test the same
        transform through a squaring epilogue.
        """
        torch.manual_seed(0)
        x = torch.randn(n_fft, device="cuda", dtype=torch.float32)

        power = oasr.rfft_power(x)
        ref = torch.fft.rfft(x).abs().pow(2)

        assert power.shape == ref.shape
        assert power.dtype == torch.float32
        torch.testing.assert_close(power, ref, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("n_fft", [512])
    @pytest.mark.parametrize("batch_shape", [(4,), (4, 100)])  # (4, 100): the fbank shape
    def test_power_batched(self, n_fft, batch_shape):
        torch.manual_seed(0)
        x = torch.randn(*batch_shape, n_fft, device="cuda", dtype=torch.float32)

        power = oasr.rfft_power(x)
        ref = torch.fft.rfft(x).abs().pow(2)

        torch.testing.assert_close(power, ref, rtol=1e-3, atol=1e-2)

    def test_power_destination_passing(self):
        x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
        out = torch.empty(4, 257, device="cuda", dtype=torch.float32)
        assert_dest_passing(oasr.rfft_power, x, out=out)
        torch.testing.assert_close(out, torch.fft.rfft(x).abs().pow(2), rtol=1e-3, atol=1e-2)
