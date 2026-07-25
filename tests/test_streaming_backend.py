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

        # Manual reference: independent streaming_forward loop with the same
        # chunks.  The backend batches same-length streams into one B=N
        # forward, so fp16 log-probs may differ from the B=1 reference by
        # kernel reduction order (~one ulp, measured flat across chunks — a
        # state bug would compound); the decode-relevant argmax must match
        # exactly.
        with torch.no_grad():
            for sid in (0, 1):
                state = model.get_streaming_init_states(1, device="cuda", dtype=torch.float16)
                for k in range(n_chunks):
                    chunk = feats[sid][k * window : (k + 1) * window].unsqueeze(0)
                    lens = torch.tensor([window], dtype=torch.int32, device="cuda")
                    lp, _ol, state = model.streaming_forward(chunk, lens, state)
                    torch.testing.assert_close(backend_out[sid][k], lp, atol=1e-2, rtol=1e-2)
                    assert torch.equal(backend_out[sid][k].argmax(-1), lp.argmax(-1))

        # Free releases per-request state.
        for r in reqs:
            backend.free(r)
        assert not backend._states  # noqa: SLF001

    def test_batched_grouping_with_short_tail(self):
        """Mixed chunk lengths: full-window streams batch together; a stream
        on its final short tail runs in its own singleton group.  Every
        stream's output must match its independent B=1 reference loop."""
        model = self._build_model()
        cfg = SimpleNamespace(device="cuda", dtype=torch.float16, chunk_size=16)
        backend = build_streaming_backend("stateful", model, cfg, None)
        window = backend.decoding_window

        # Streams 0/1: two full windows.  Stream 2: one full window + a
        # half-window tail (audio_final so the tail is deemed ready).
        lengths = {0: 2 * window, 1: 2 * window, 2: window + window // 2}
        feats = {
            sid: torch.randn(n, 80, dtype=torch.float16, device="cuda")
            for sid, n in lengths.items()
        }
        reqs = [_make_request(sid, feats[sid]) for sid in feats]
        for r in reqs:
            r.audio_final = True
            backend.allocate(r)

        # Spy on the singleton path: only the tail chunk should take it.
        singles: list = []
        orig_forward_one = backend._forward_one  # noqa: SLF001

        def spy(req):
            singles.append(req.stream_id)
            return orig_forward_one(req)

        backend._forward_one = spy  # noqa: SLF001

        backend_out = {sid: [] for sid in feats}
        for _ in range(2):
            out = backend.forward_step(reqs)
            for r in reqs:
                if r.request_id in out:
                    backend_out[r.stream_id].append(out[r.request_id].clone())
        # Tick 1: all three at a full window → one batched B=3 forward.
        # Tick 2: streams 0/1 batch (B=2); stream 2's short tail is singleton.
        assert singles == [2]

        with torch.no_grad():
            for sid, total in lengths.items():
                state = model.get_streaming_init_states(1, device="cuda", dtype=torch.float16)
                cursor = 0
                for lp_backend in backend_out[sid]:
                    t = min(window, total - cursor)
                    chunk = feats[sid][cursor : cursor + t].unsqueeze(0)
                    lens = torch.tensor([t], dtype=torch.int32, device="cuda")
                    lp, _ol, state = model.streaming_forward(chunk, lens, state)
                    torch.testing.assert_close(lp_backend, lp, atol=1e-2, rtol=1e-2)
                    assert torch.equal(lp_backend.argmax(-1), lp.argmax(-1))
                    cursor += t
        for r in reqs:
            backend.free(r)

    def test_stack_unstack_roundtrip(self):
        """Encoder state stack/unstack must be mutually inverse and produce
        the same shapes as a natively batched init."""
        model = self._build_model()
        enc = model.encoder
        singles = [
            model.get_streaming_init_states(1, device="cuda", dtype=torch.float16) for _ in range(3)
        ]
        # Give each stream distinct state contents.
        for i, st in enumerate(singles):
            for t in st:
                t.fill_(float(i + 1))
        stacked = enc.stack_streaming_states(singles)
        native = model.get_streaming_init_states(3, device="cuda", dtype=torch.float16)
        assert [t.shape for t in stacked] == [t.shape for t in native]
        unstacked = enc.unstack_streaming_states(stacked)
        assert len(unstacked) == 3
        for st, ref in zip(unstacked, singles):
            for a, b in zip(st, ref):
                assert torch.equal(a, b)

    def test_hidden_mode_routes_to_encoder(self):
        """consumes='hidden' must thread chunks through the *encoder-only*
        streaming forward (raw hidden states, no CTC head)."""
        model = self._build_model()
        cfg = SimpleNamespace(device="cuda", dtype=torch.float16, chunk_size=16)
        backend = build_streaming_backend("stateful", model, cfg, None, consumes="hidden")
        window = backend.decoding_window

        feats = torch.randn(window * 2, 80, dtype=torch.float16, device="cuda")
        req = _make_request(0, feats)
        backend.allocate(req)

        got = []
        for _ in range(2):
            out = backend.forward_step([req])
            got.append(out[req.request_id].clone())

        with torch.no_grad():
            state = model.get_streaming_init_states(1, device="cuda", dtype=torch.float16)
            for k in range(2):
                chunk = feats[k * window : (k + 1) * window].unsqueeze(0)
                lens = torch.tensor([window], dtype=torch.int32, device="cuda")
                hidden, _ol, state = model.encoder.streaming_forward(chunk, lens, state)
                torch.testing.assert_close(got[k], hidden)
        # Hidden = encoder dim (max stack dim 96), not the 32-entry vocab.
        assert got[0].shape[-1] == 96
        backend.free(req)


# --------------------------------------------------------------------------- #
# consumes routing (constructor-level; no CUDA / no forward needed)
# --------------------------------------------------------------------------- #


class _RoutingEncoderStub:
    streaming_kind = "paged"
    subsampling_rate = 4
    right_context = 6


class _RoutingModelStub:
    """Just enough surface for PagedStreamingBackend.__init__ routing checks."""

    encoder = _RoutingEncoderStub()

    def forward_chunk_paged(self, *a, **k):  # fused encoder+head
        raise AssertionError("not called in this test")

    def encode_chunk_paged(self, *a, **k):  # encoder-only
        raise AssertionError("not called in this test")


def _paged_backend(consumes):
    from oasr.cache import CacheConfig
    from oasr.engine.streaming_backend.paged import PagedStreamingBackend

    cache_cfg = CacheConfig(
        num_layers=2,
        n_kv_head=4,
        head_dim=16,
        hidden_dim=64,
        kernel_size=15,
        chunk_size=16,
        num_left_chunks=-1,
        block_size_frames=16,
        max_num_blocks=32,
        max_blocks_per_seq=8,
        max_batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cfg = SimpleNamespace(
        device="cpu",
        dtype=torch.float32,
        chunk_size=16,
        use_cuda_graphs=True,  # must still be disabled by hidden routing / cpu
        feature_config=SimpleNamespace(output_dim=80),
    )
    model = _RoutingModelStub()
    return (
        PagedStreamingBackend(model, cfg, cache_cfg, consumes=consumes),
        model,
    )


def test_paged_backend_consumes_routing():
    backend, model = _paged_backend("hidden")
    assert backend._chunk_forward.__func__ is _RoutingModelStub.encode_chunk_paged  # noqa: SLF001
    assert backend._graph_cache is None  # noqa: SLF001

    backend, model = _paged_backend("log_probs")
    assert backend._chunk_forward.__func__ is _RoutingModelStub.forward_chunk_paged  # noqa: SLF001


def test_paged_backend_gates_capacity_exhausted_stream():
    """An out-of-cache stream must be flagged, not dispatched into the forward.

    Regression: with unlimited history the pool eventually has no free block, and
    the allocator raised ``BlockPool exhausted`` from inside the encoder forward.
    The model stub's ``forward_chunk_paged`` raises, so reaching it fails the test.
    """
    backend, _model = _paged_backend("log_probs")
    req = _make_request(stream_id=0, feature_buffer=torch.zeros(1024, 80))
    req.feature_frames = 1024
    req.feature_cursor = 0
    backend.allocate(req)

    # Drain the pool behind the manager's back so the next chunk has nowhere to go.
    pool = backend.block_pool
    pool.allocate(pool.num_free_blocks)

    results = backend.forward_step([req])
    assert results == {}
    assert req.cache_exhausted is True


class TestOfflineModeSkipsTheStreamingBackend:
    """``service_mode="offline"`` must not build the paged backend (H13).

    ``service_mode`` pins the engine to one executor for its lifetime and rejects
    mismatched requests at admission, so an offline engine can never reach a
    streaming forward — yet ``ModelRunner`` used to build the real backend anyway,
    which constructs ``BlockPool`` + ``AttentionCacheManager`` + ``CnnCacheManager``
    (~0.4 GB of VRAM at the defaults) and holds them for the process lifetime.  That
    lands on exactly the offline / speech-LLM deployments where VRAM is tightest.
    """

    def _spy(self, monkeypatch):
        from oasr.engine import model_runner as mr

        seen = {}

        def _fake_build(kind, model, config, cache_config, **kw):
            seen["kind"] = kind
            seen["cache_config"] = cache_config
            return SimpleNamespace(decoding_window=0, stride=0)

        monkeypatch.setattr(mr, "build_streaming_backend", _fake_build)
        return mr, seen

    def _model(self, streaming_kind):
        return SimpleNamespace(encoder=SimpleNamespace(streaming_kind=streaming_kind))

    @pytest.mark.parametrize("encoder_kind", ["paged", "stateful"])
    def test_offline_mode_selects_the_no_op_backend(self, monkeypatch, encoder_kind):
        mr, seen = self._spy(monkeypatch)
        cfg = SimpleNamespace(service_mode="offline")
        mr.ModelRunner(self._model(encoder_kind), cfg, object())
        assert seen["kind"] == "none"

    @pytest.mark.parametrize("encoder_kind", ["paged", "stateful"])
    def test_streaming_mode_still_selects_the_encoders_own_backend(self, monkeypatch, encoder_kind):
        mr, seen = self._spy(monkeypatch)
        cfg = SimpleNamespace(service_mode="streaming")
        mr.ModelRunner(self._model(encoder_kind), cfg, object())
        assert seen["kind"] == encoder_kind

    def test_offline_only_encoder_is_unchanged(self, monkeypatch):
        mr, seen = self._spy(monkeypatch)
        cfg = SimpleNamespace(service_mode="streaming")
        mr.ModelRunner(self._model("none"), cfg, None)
        assert seen["kind"] == "none"
        assert seen["cache_config"] is None


def test_no_streaming_backend_accepts_a_missing_cache_config():
    """An offline-only encoder reports ``cache_spec is None``, so the engine has no
    ``CacheConfig`` to hand down — the placeholder backend must take that."""
    backend = build_streaming_backend("none", object(), object(), None)
    assert backend.decoding_window == 0
    assert backend.stride == 0
    with pytest.raises(NotImplementedError, match="does not support streaming"):
        backend.forward_step([])
