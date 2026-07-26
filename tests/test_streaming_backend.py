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


def _paged_backend(consumes, max_batch_size=2):
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
        max_batch_size=max_batch_size,
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

    backend, model = _paged_backend("log_probs")
    assert backend._chunk_forward.__func__ is _RoutingModelStub.forward_chunk_paged  # noqa: SLF001


def test_graph_capture_follows_the_device_not_the_consumes_mode():
    """Graph capture is gated on CUDA + the flag, never on what the strategy eats.

    It used to also require ``consumes == "log_probs"``, which silently forfeited
    capture for every hidden-mode family (streaming transducer ran ~200 eager
    kernel launches per chunk).  The capture machinery takes the chunk-forward
    callable and never inspects its output, so both modes qualify; here the CPU
    device is what disables it, for both.
    """
    for consumes in ("hidden", "log_probs"):
        backend, _ = _paged_backend(consumes)
        # ``use_cuda_graphs=True`` in the stub config, but device is CPU.
        assert backend._use_cuda_graphs is False, consumes  # noqa: SLF001
        assert backend._graph_cache is None, consumes  # noqa: SLF001


class TestGraphReplayBufferIsNotHandedOutTwice:
    """Two replays of one shape key in a step must not share an output buffer.

    ``GraphedEncoderForward`` reuses one pre-allocated output buffer per
    ``(B, T_input, cache_t1_bucket)`` key, and ``forward_step`` can replay the
    same key more than once: a full-window *final* chunk goes through
    ``_forward_single`` at ``B=1``, so two streams finalizing in the same step
    at the same offset bucket collide — as does a ``B=1`` batched cohort
    alongside one such final.  Every earlier caller's tensor was then silently
    rewritten before the decoder read it.

    Observed on the CTC path before the fix: three lockstep streams each ending
    on a full window produced two transcripts whose tails were the *third*
    stream's final chunk.  This predates hidden-mode capture; extending capture
    to ``consumes="hidden"`` widened its reach, which is how it surfaced.
    """

    def _backend_and_reqs(self, n):
        backend, _ = _paged_backend("log_probs", max_batch_size=n)
        reqs = [_make_request(sid, torch.zeros(1024, 80)) for sid in range(n)]
        for r in reqs:
            backend.allocate(r)
        return backend, reqs

    def _record_detach(self, backend):
        """Replace both forwards with recorders; return the recorded flags."""
        seen = {"batched": [], "single": []}

        def batched(group, window, stride, context, results, detach=False):
            seen["batched"].append(detach)
            for r in group:
                results[r.request_id] = torch.zeros(1, 4, 8)
                r.feature_cursor += stride

        def single(req, window, stride, context, results, detach=False):
            seen["single"].append(detach)
            results[req.request_id] = torch.zeros(1, 4, 8)
            req.feature_cursor = req.feature_frames

        backend._forward_batched_paged = batched  # noqa: SLF001
        backend._forward_single = single  # noqa: SLF001
        return seen

    def test_only_the_last_graph_consumer_of_a_step_may_alias(self):
        backend, reqs = self._backend_and_reqs(4)
        seen = self._record_detach(backend)

        # One mid-stream (batchable) + three finalizing (fallback).  A B=1
        # cohort is the only width that can collide with a single.
        for r in reqs[1:]:
            r.audio_final = True
            r.feature_frames = r.feature_cursor + backend.decoding_window
        backend.forward_step(reqs)

        assert seen["batched"] == [True]
        # Among the singles, only the last one may keep the live buffer.
        assert seen["single"] == [True, True, False]

    def test_a_wide_cohort_never_collides_with_a_single(self):
        """``_forward_single`` always replays at B=1, so B>1 cannot share a key.

        Detaching a wide cohort would copy a ``(B, chunk, V)`` tensor on the
        steady-state path for a collision that is impossible by construction.
        """
        backend, reqs = self._backend_and_reqs(4)
        seen = self._record_detach(backend)
        for r in reqs[2:]:  # two batchable, two finalizing
            r.audio_final = True
            r.feature_frames = r.feature_cursor + backend.decoding_window
        backend.forward_step(reqs)

        assert seen["batched"] == [False]
        assert seen["single"] == [True, False]

    def test_a_lone_batched_cohort_still_aliases(self):
        """The steady state must stay copy-free — this is the hot path."""
        backend, reqs = self._backend_and_reqs(4)
        seen = self._record_detach(backend)
        backend.forward_step(reqs)
        assert seen["batched"] == [False]
        assert seen["single"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
def test_graph_replay_reuses_one_output_buffer_per_shape_key():
    """The mechanism the detach policy above defends against.

    Documents the ``GraphedEncoderForward.replay`` contract explicitly: the
    returned tensor **aliases** the capture's output buffer, so a second replay
    at the same key rewrites it in place.  If this ever stops being true the
    detach guard becomes dead weight and should go with it.
    """
    from oasr.engine.graph_cache import GraphedEncoderForward

    captured = {}

    def chunk_forward(xs, offset, caches, cnn_cache, cache_t1=0):
        return xs.sum(-1, keepdim=True) * 2.0

    gc = GraphedEncoderForward(
        chunk_forward,
        att_mgr=_StubAttMgr(),
        cnn_mgr=SimpleNamespace(buffer=torch.zeros(1, 4, 2, 8, device="cuda")),
        device=torch.device("cuda"),
    )
    dev = torch.device("cuda")
    slot = torch.zeros(1, dtype=torch.long, device=dev)
    off = torch.zeros(1, dtype=torch.int32, device=dev)
    a = torch.ones(1, 4, 8, device=dev)
    b = torch.full((1, 4, 8), 5.0, device=dev)

    out_a = gc.replay(1, 4, 0, xs=a, slot_ids=slot, offsets=off)
    first = out_a.clone()
    out_b = gc.replay(1, 4, 0, xs=b, slot_ids=slot, offsets=off)

    captured["same_buffer"] = out_a.data_ptr() == out_b.data_ptr()
    assert captured["same_buffer"], "replay no longer aliases; drop the detach guard"
    assert not torch.equal(out_a, first), "the first handle survived a second replay"


#: Geometries exercised by the capture-determinism tests below, as
#: ``(model_dim, n_heads)``.  ``head_dim = dim // heads`` is the axis that
#: matters; the widths are there to show it is not a width effect.
_HD32_GEOMETRIES = [(64, 2), (128, 4), (256, 8)]
_HD64_GEOMETRIES = [(256, 4), (512, 8)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
@pytest.mark.parametrize("dim,heads", _HD64_GEOMETRIES)
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_capture_is_deterministic_and_exact_at_head_dim_64(dim, heads, batch):
    """The working case, and the regression guard for H3.

    At head_dim 64 two independent captures of the same shape agree **bit for
    bit** with each other and with eager, at every batch width.  That is the
    property graph capture is supposed to have, and the contrast with head_dim
    32 below is what localises the defect to the kernel rather than to the
    capture machinery.
    """
    runs = _drive_capture(dim, heads, batch)
    assert runs["graph_vs_eager"] == 0.0, f"graph diverged from eager: {runs}"
    assert runs["graph_vs_graph"] == 0.0, f"capture is not reproducible: {runs}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: CuteDSL FMHA reads uninitialised pool memory at head_dim 32",
)
@pytest.mark.parametrize("dim,heads", _HD32_GEOMETRIES)
def test_graph_capture_is_reproducible_at_head_dim_32(dim, heads):
    """Known defect: at head_dim 32, capture is not even self-consistent.

    Two independent captures of the *same* shape, fed the *same* input, disagree
    by ~1e-1 in log-probs.  Nothing in the computation is random, so a
    difference between two runs can only come from reading memory the capture
    never initialised — the over-read that
    ``PagedStreamingBackend._forward_single`` already documents for sub-window
    chunks, benign in eager mode because adjacent allocations happen to be
    mapped, stale once the graph carves its own pool.

    Asserting **self**-consistency rather than agreement-with-eager is what makes
    this test deterministic.  A graph-vs-eager assertion is *flaky* here: whether
    the stale bytes happen to differ from the correct ones depends on allocator
    history, so it XPASSes in some processes.  Self-inconsistency needs no oracle
    and cannot be accidentally satisfied.

    Measured axes (`B` = streams in the cohort):

    ==========  ==========================  ==================
    head_dim    B = 1                       B >= 2
    ==========  ==========================  ==================
    32          marginal (sometimes exact)  always diverges
    64          exact                       exact
    ==========  ==========================  ==================

    So it needs a **batched** cohort to manifest reliably — which is the
    production streaming shape — and it is **not** a hidden-mode problem: it
    reproduces on the fused CTC path, captured since long before H3.  Any
    head_dim-32 checkpoint has therefore been streaming non-reproducible
    log-probs with the default ``use_cuda_graphs=True``.  It surfaced only
    because extending capture to hidden mode (H3) put the transducer fixture's
    64/2 geometry on the captured path and made its token-identity gate flaky.

    Fixing it belongs in the FMHA kernel, alongside the masked-tile NaN bug.
    """
    runs = _drive_capture(dim, heads, batch=2)
    assert runs["graph_vs_graph"] == 0.0, f"capture is not reproducible: {runs}"


def _drive_capture(dim: int, heads: int, batch: int, n_chunks: int = 5) -> dict:
    """Run one tiny conformer three ways: eager, graphed, graphed again.

    Returns ``{"graph_vs_eager": float, "graph_vs_graph": float}`` — the second
    is the oracle-free signal, since two identical captures of a deterministic
    computation must agree.
    """
    from oasr.cache.types import CacheConfig
    from oasr.engine.streaming_backend.paged import PagedStreamingBackend
    from oasr.models.conformer.config import ConformerEncoderConfig, ConformerModelConfig
    from oasr.models.conformer.model import ConformerModel

    dtype = torch.float16
    enc = ConformerEncoderConfig(
        input_size=80,
        output_size=dim,
        num_blocks=2,
        attention_heads=heads,
        linear_units=dim * 2,
        cnn_module_kernel=15,
        causal=True,
        embed_layer_norm=False,
    )
    torch.manual_seed(7)
    model = (
        ConformerModel.from_config(ConformerModelConfig(encoder=enc, vocab_size=32))
        .eval()
        .to(device="cuda", dtype=dtype)
    )
    spec = model.cache_spec
    cache_cfg = CacheConfig(
        num_layers=spec.num_layers,
        n_kv_head=spec.n_kv_head,
        head_dim=spec.head_dim,
        hidden_dim=spec.hidden_dim,
        kernel_size=spec.conv_kernel_size,
        chunk_size=16,
        num_left_chunks=-1,
        block_size_frames=16,
        max_num_blocks=64 * batch,
        max_blocks_per_seq=64,
        max_batch_size=batch,
        device=torch.device("cuda"),
        dtype=dtype,
    )
    window = (16 - 1) * model.encoder.subsampling_rate + model.encoder.right_context + 1
    torch.manual_seed(21)
    feats = [
        torch.randn(window * (n_chunks + 1), 80, dtype=dtype, device="cuda") * 0.5
        for _ in range(batch)
    ]

    def drive(graphs):
        cfg = SimpleNamespace(
            device="cuda",
            dtype=dtype,
            chunk_size=16,
            use_cuda_graphs=graphs,
            finalize_silence_pad=True,
            feature_config=SimpleNamespace(output_dim=80),
        )
        backend = PagedStreamingBackend(model, cfg, cache_cfg, consumes="log_probs")
        reqs = [_make_request(sid, f) for sid, f in enumerate(feats)]
        for r in reqs:
            backend.allocate(r)
        got = {r.stream_id: [] for r in reqs}
        for _ in range(n_chunks):
            res = backend.forward_step(reqs)
            for r in reqs:
                if r.request_id in res:
                    got[r.stream_id].append(res[r.request_id].detach().float().cpu().clone())
        for r in reqs:
            backend.free(r)
        return got

    def worst(a, b):
        return max((x - y).abs().max().item() for sid in a for x, y in zip(a[sid], b.get(sid, [])))

    eager, g1, g2 = drive(False), drive(True), drive(True)
    return {"graph_vs_eager": worst(eager, g1), "graph_vs_graph": worst(g1, g2)}


class _StubAttMgr:
    """Minimal ``AttentionCacheManager`` surface for a capture-only test."""

    num_layers = 1

    def __init__(self):
        dev = torch.device("cuda")
        self.block_table = torch.zeros(4, 8, dtype=torch.int32, device=dev)
        self.cache_seqlens = torch.zeros(4, dtype=torch.int32, device=dev)
        self._persistent_caches = [
            SimpleNamespace(
                k_cache=torch.zeros(8, 16, 1, 8, device=dev),
                v_cache=torch.zeros(8, 16, 1, 8, device=dev),
                block_size=16,
            )
        ]


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
