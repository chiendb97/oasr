# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pluggable streaming-encoder backend seam.

* registry dispatch (paged / stateful / unknown),
* StatefulStreamingBackend orchestration: it must thread the per-request encoder
  state across chunks exactly like a manual ``model.streaming_forward`` loop.

The stateful test builds a tiny causal Zipformer (OASR-only, random weights) and
compares the backend's output against a hand-rolled streaming loop over the same
chunks — validating orchestration independent of the absolute fbank geometry.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from oasr.engine.request import Request
from oasr.engine.streaming_backend import build_streaming_backend
from oasr.engine.streaming_backend.base import _REGISTRY


def test_registry_has_builtin_backends():
    assert "paged" in _REGISTRY
    assert "stateful" in _REGISTRY


def test_build_unknown_backend_raises():
    with pytest.raises(NotImplementedError, match="No streaming backend"):
        build_streaming_backend("does-not-exist", None, None, None)


def _make_request(stream_id: int, feature_buffer: torch.Tensor) -> Request:
    req = Request(None, streaming=True)
    req.stream_id = stream_id
    req.feature_buffer = feature_buffer
    req.feature_frames = int(feature_buffer.size(0))
    req.feature_cursor = 0
    req.offset = 0
    return req


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")
class TestStatefulStreamingBackend:
    def _build_model(self):
        from oasr.models.zipformer.config import (
            ZipformerEncoderConfig,
            ZipformerModelConfig,
        )
        from oasr.models.zipformer.model import ZipformerModel

        enc_cfg = ZipformerEncoderConfig(
            feature_dim=80,
            downsampling_factor=(1, 2),
            encoder_dim=(64, 96),
            num_encoder_layers=(1, 1),
            query_head_dim=(8,),
            pos_head_dim=(4,),
            value_head_dim=(6,),
            num_heads=(4, 4),
            feedforward_dim=(64, 96),
            cnn_module_kernel=(15, 15),
            pos_dim=16,
            causal=True,
            chunk_size=(16,),
            left_context_frames=(32,),
        )
        torch.manual_seed(0)
        model = ZipformerModel(ZipformerModelConfig(encoder=enc_cfg, vocab_size=32))
        return model.half().cuda().eval()

    def test_streaming_kind_and_window(self):
        model = self._build_model()
        assert model.encoder.streaming_kind == "stateful"
        # chunk_size 16 * 2x embed = 32 input frames per steady-state chunk.
        assert model.encoder.streaming_chunk_frames == 32

    def test_state_threading_matches_manual_loop(self):
        model = self._build_model()
        cfg = SimpleNamespace(device="cuda", dtype=torch.float16, chunk_size=16)
        backend = build_streaming_backend("stateful", model, cfg, None)
        window = backend.decoding_window
        assert window == 32

        n_chunks = 3
        feats = {
            sid: torch.randn(window * n_chunks, 80, dtype=torch.float16, device="cuda")
            for sid in (0, 1)
        }
        reqs = [_make_request(sid, feats[sid]) for sid in (0, 1)]
        for r in reqs:
            backend.allocate(r)

        # Backend: n_chunks ticks, one window consumed per tick.
        backend_out = {sid: [] for sid in (0, 1)}
        for _ in range(n_chunks):
            out = backend.forward_step(reqs)
            for r in reqs:
                backend_out[r.stream_id].append(out[r.request_id].clone())

        # Manual reference: independent streaming_forward loop with the same chunks.
        with torch.no_grad():
            for sid in (0, 1):
                state = model.get_streaming_init_states(1, device="cuda", dtype=torch.float16)
                for k in range(n_chunks):
                    chunk = feats[sid][k * window : (k + 1) * window].unsqueeze(0)
                    lens = torch.tensor([window], dtype=torch.int32, device="cuda")
                    lp, _ol, state = model.streaming_forward(chunk, lens, state)
                    torch.testing.assert_close(backend_out[sid][k], lp)

        # Free releases per-request state.
        for r in reqs:
            backend.free(r)
        assert not backend._states  # noqa: SLF001
