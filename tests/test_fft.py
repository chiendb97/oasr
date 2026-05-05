#!/usr/bin/env python3
"""Unit tests for oasr.rfft / oasr.rfft_power."""

import pytest
import torch

import oasr


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

    @pytest.mark.parametrize("n_fft", [256, 512, 1024])
    @pytest.mark.parametrize("batch_shape", [(4,), (2, 8), (3, 5, 7)])
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
        result = oasr.rfft(x, out=out)
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result, torch.fft.rfft(x), rtol=1e-4, atol=1e-3)

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
        with pytest.raises(Exception):
            oasr.rfft(x)


class TestRfftPower:
    """Tests for oasr.rfft_power (real-FFT power spectrum)."""

    @pytest.mark.parametrize("n_fft", N_FFT_VALUES)
    def test_power_1d(self, n_fft):
        torch.manual_seed(0)
        x = torch.randn(n_fft, device="cuda", dtype=torch.float32)

        power = oasr.rfft_power(x)
        ref = torch.fft.rfft(x).abs().pow(2)

        assert power.shape == ref.shape
        assert power.dtype == torch.float32
        torch.testing.assert_close(power, ref, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("n_fft", [256, 512])
    @pytest.mark.parametrize("batch_shape", [(4,), (2, 16)])
    def test_power_batched(self, n_fft, batch_shape):
        torch.manual_seed(0)
        x = torch.randn(*batch_shape, n_fft, device="cuda", dtype=torch.float32)

        power = oasr.rfft_power(x)
        ref = torch.fft.rfft(x).abs().pow(2)

        torch.testing.assert_close(power, ref, rtol=1e-3, atol=1e-2)

    def test_power_fbank_shape(self):
        """Shape that matches the FBANK pipeline: (B, num_frames, n_fft)."""
        torch.manual_seed(0)
        B, F, n_fft = 4, 100, 512
        x = torch.randn(B, F, n_fft, device="cuda", dtype=torch.float32)

        power = oasr.rfft_power(x)
        ref = torch.fft.rfft(x).abs().pow(2)

        assert power.shape == (B, F, n_fft // 2 + 1)
        torch.testing.assert_close(power, ref, rtol=1e-3, atol=1e-2)

    def test_power_destination_passing(self):
        x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
        out = torch.empty(4, 257, device="cuda", dtype=torch.float32)
        result = oasr.rfft_power(x, out=out)
        assert result.data_ptr() == out.data_ptr()
        ref = torch.fft.rfft(x).abs().pow(2)
        torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-2)
