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

    def streaming_geometry(self, chunk_size):
        """``None`` = "use the generic window formula", the ``BaseEncoder`` default.

        Spelled out rather than left off: the backend calls this at construction to
        let an encoder declare a front-end the formula does not describe (and to
        refuse a chunk size it cannot serve), so a stub that omits it is not a stub
        of the current contract.
        """
        del chunk_size
        return None


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
    """A step's earlier graph results must survive everything that follows them.

    Two independent mechanisms invalidate a ``GraphedEncoderForward`` result, and
    each one cost real transcripts before it was understood:

    **Same-key reuse.**  One pre-allocated output buffer per
    ``(B, T_input, cache_t1_bucket)`` key, and ``forward_step`` can replay one key
    twice: a full-window *final* chunk goes through ``_forward_single`` at ``B=1``,
    so two streams finalizing in the same step collide — as does a ``B=1`` batched
    cohort alongside one such final.  Observed on the CTC path: three lockstep
    streams each ending on a full window produced two transcripts whose tails were
    the *third* stream's final chunk.

    **A later capture.**  Captures share one memory pool, so a *first* capture at a
    new key may be handed the block an earlier capture's output buffer occupies —
    which invalidates results from keys that were never replayed again.  Observed
    on Nemotron streaming: 5 of 40 LJSpeech utterances lost their trailing words,
    deterministically, only with graphs enabled, and only for streams that
    finalized in the same step as a fresh capture.  This is why a **wide** cohort
    must detach too, even though no single can share its key.
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

    def test_a_wide_cohort_detaches_when_anything_follows_it(self):
        """Not for a key collision — a single always replays at ``B=1`` — but because
        a following single may **capture**, and a capture can reuse the pool block
        this cohort's output buffer sits in.  Cheaper to test than to rediscover.
        """
        backend, reqs = self._backend_and_reqs(4)
        seen = self._record_detach(backend)
        for r in reqs[2:]:  # two batchable, two finalizing
            r.audio_final = True
            r.feature_frames = r.feature_cursor + backend.decoding_window
        backend.forward_step(reqs)

        assert seen["batched"] == [True]
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

    from oasr.cache.state import SlotStateCache, StreamStateSpec

    gc = GraphedEncoderForward(
        chunk_forward,
        att_mgr=_StubAttMgr(),
        state_mgr=SlotStateCache(
            [StreamStateSpec("conv", (1, 2, 8), slot_axis=1)],
            max_batch_size=4,
            device=torch.device("cuda"),
            dtype=torch.float32,
        ),
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
@pytest.mark.parametrize("dim,heads", _HD32_GEOMETRIES)
def test_graph_capture_is_reproducible_at_head_dim_32(dim, heads):
    """Regression guard for a fixed defect: capture must be self-consistent.

    Until 2026-07-27 two independent captures of the *same* shape, fed the *same*
    input, disagreed by ~1e-1 in log-probs at head_dim 32.  Nothing in the
    computation is random, so a difference between two runs meant unsynchronised
    concurrency.

    **Generalised by** :func:`test_paged_load_is_race_free_across_block_sizes`:
    the defect was in the paged K/V load and depended on ``head_dim`` *and*
    ``block_size_frames`` together, not on ``head_dim`` alone.  This test is the
    narrower historical case, kept because it is the shape the bug was found on.

    **It was a shared-memory write-after-write race** (not, as recorded for a
    while, a read of uninitialised memory)::

        compute-sanitizer --tool racecheck --racecheck-report hazard \\
            pytest "tests/test_streaming_backend.py::\\
    test_graph_capture_is_reproducible_at_head_dim_32[64-2]"

        Error: Potential WAW hazard detected (invalid memcpy_async
        synchronization) at __shared__ 0x1800 in block (0,1,1):
            Write Thread (64,0,0) at ...FmhaSm120...+0x990
            Write Thread  (0,0,0) at ...FmhaSm120...+0x9c0
        RACECHECK SUMMARY: 176 errors

    The two writers are in ``_paged_load_kv_tile``, which used to slice smem into
    ``(block_size, head_dim)`` sub-tiles and re-partition each one; the copy's
    32-row per-pass extent then spilled 16 rows into the next page.  0x1800 is
    ``sQ``'s 0x1000 plus 0x800, i.e. ``sK`` stage 0 row 32 — exactly where page
    1's spill meets page 2's own write, and the reported threads are 0 and 64,
    exactly the pair the arithmetic predicts.  (It is **not** the cp.async ring's
    ``num_stages`` / ``cp_async_wait_group`` counts; those match FlashAttention's
    SM80 path instruction for instruction.)

    Three facts that bound it:

    * It is **not head_dim-32-specific**, and this test's name is therefore too
      narrow — it fires whenever the copy's per-pass row extent exceeds
      ``block_size_frames``.  head_dim 64 with ``block_size_frames=8`` races too,
      and every shipped checkpoint is head_dim 64; only the default page height
      of 16 kept it out of production.  See
      :func:`test_paged_load_is_race_free_across_block_sizes`.
    * ``OASR_ATTN_BACKEND=sdpa`` makes this test XPASS, which is what localises
      the defect to the cute kernel rather than the capture machinery.  It is
      also a 1.75 s discriminator versus a 12 s cute run.
    * ``--tool initcheck`` reports **0** uninitialised reads and the test XPASSes
      under it, because the sanitizer serialises execution.  That is what refuted
      the original stale-memory reading.

    Asserting **self**-consistency rather than agreement-with-eager is what makes
    this test deterministic.  A graph-vs-eager assertion is *flaky* here: whether
    the racing write lands before or after the read depends on scheduling, so it
    XPASSes in some processes.  Self-inconsistency needs no oracle and cannot be
    accidentally satisfied.

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

    **Fixed 2026-07-27.** ``_paged_load_kv_tile`` now partitions the whole
    ``(N_BLOCK, D)`` tile once and varies the *gmem source per row*, mirroring
    ``flash_attn/cute/paged_kv.py::PagedKVManager.load_KV``, so every smem element
    has exactly one writer for any ``(head_dim, block_size)`` pair.  Verified:
    racecheck went **176 errors -> 0**, all three head_dim-32 geometries went from
    xfail to pass, and reverting only the loader makes
    :func:`test_paged_load_is_race_free_across_block_sizes` fail on exactly the two
    spilling parametrisations and pass on the two exact-fit ones.  Streaming
    throughput is unchanged (the cp.async count per thread is identical).

    Ruled out, recorded so it is not retried: adding a CTA barrier before the
    epilogue's ``sO``-into-``sQ`` write (FlashAttention's
    ``if (Share_Q_K_smem) __syncthreads()``).  The alias here *is*
    unconditional, but racecheck's hazard count is unchanged by the barrier
    (176 before and after) and reports no ``sQ`` hazard at all, so the epilogue
    is already correctly synchronised and the barrier only costs the hot path.
    """
    runs = _drive_capture(dim, heads, batch=2)
    assert runs["graph_vs_graph"] == 0.0, f"capture is not reproducible: {runs}"


def _drive_capture(
    dim: int, heads: int, batch: int, n_chunks: int = 5, block_size: int = 16
) -> dict:
    """Run one tiny conformer four ways: eager twice, graphed twice.

    Returns ``{"graph_vs_eager", "graph_vs_graph", "eager_vs_eager"}``.  The two
    self-comparisons are the oracle-free signals: nothing in the computation is
    random, so two runs of the *same* path must agree bit for bit.
    ``eager_vs_eager`` catches a data race even with capture switched off, which
    is what distinguishes a race from a capture-pool artefact.

    ``block_size`` is the paged-KV page height (``CacheConfig.block_size_frames``);
    it is a parameter because the paged loader's correctness depends on how it
    compares with the gmem copy's per-pass row extent — see
    :func:`test_paged_load_is_race_free_across_block_sizes`.  The encoder chunk
    follows it: ``CacheConfig`` requires ``chunk_size <= block_size_frames`` (one
    page is allocated per chunk), so a narrow page implies a narrow chunk.
    """
    from oasr.cache.types import CacheConfig
    from oasr.engine.streaming_backend.paged import PagedStreamingBackend
    from oasr.models.conformer.config import ConformerEncoderConfig, ConformerModelConfig
    from oasr.models.conformer.model import ConformerModel

    # CacheConfig requires chunk_size <= block_size_frames.
    chunk = min(16, block_size)
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
        chunk_size=chunk,
        num_left_chunks=-1,
        block_size_frames=block_size,
        max_num_blocks=64 * batch,
        max_blocks_per_seq=64,
        max_batch_size=batch,
        device=torch.device("cuda"),
        dtype=dtype,
    )
    window = (chunk - 1) * model.encoder.subsampling_rate + model.encoder.right_context + 1
    torch.manual_seed(21)
    feats = [
        torch.randn(window * (n_chunks + 1), 80, dtype=dtype, device="cuda") * 0.5
        for _ in range(batch)
    ]

    def drive(graphs):
        cfg = SimpleNamespace(
            device="cuda",
            dtype=dtype,
            chunk_size=chunk,
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

    e1, e2, g1, g2 = drive(False), drive(False), drive(True), drive(True)
    return {
        "graph_vs_eager": worst(e1, g1),
        "graph_vs_graph": worst(g1, g2),
        "eager_vs_eager": worst(e1, e2),
    }


#: ``(model_dim, n_heads, block_size)`` combinations for the paged-load race
#: guard below.  The first entry of each pair is a geometry where the gmem
#: copy's per-pass row extent *exceeds* the paged page height, which is the
#: condition that used to corrupt smem; the second is the exact-fit control.
_PAGED_RACE_GEOMETRIES = [
    # head_dim 32 -> smem_k_block 32 -> 128*8/32 = 32 rows per pass.
    (64, 2, 16),  # 32 rows vs page 16 -> used to spill 16 rows
    (64, 2, 32),  # exact fit
    # head_dim 64 -> smem_k_block 64 -> 128*8/64 = 16 rows per pass.
    (256, 4, 8),  # 16 rows vs page 8 -> used to spill 8 rows (production width!)
    (256, 4, 16),  # exact fit, the shipped default
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
@pytest.mark.parametrize("dim,heads,block_size", _PAGED_RACE_GEOMETRIES)
def test_paged_load_is_race_free_across_block_sizes(dim, heads, block_size):
    """The paged K/V smem load must not depend on page height vs copy extent.

    ``_paged_load_kv_tile`` used to slice smem into ``(block_size, head_dim)``
    sub-tiles and re-run ``partition_D`` on each one.  The gmem tiled copy covers
    ``num_threads * async_elems // smem_k_block_size`` **rows** per pass — a
    function of ``head_dim``, not of ``block_size`` — so whenever that exceeded
    ``block_size`` the surplus threads wrote rows past the end of their sub-tile,
    straight into the next page's rows, which that page's own copy also wrote.
    Two writers, *different* source pages, no synchronisation: the values were
    wrong, not merely non-reproducible.

    ==========  =============  ==========  =========================
    head_dim    rows per pass  page height  before the fix
    ==========  =============  ==========  =========================
    32          32             16           16 rows spilled
    64          16             8            8 rows spilled
    64          16             16           exact fit (shipped default)
    ==========  =============  ==========  =========================

    Note the ``head_dim 64 / block_size 8`` row: the defect was **not**
    head_dim-32-specific, and every shipped checkpoint is head_dim 64.  Reaching
    it needs ``block_size_frames < 16``, and since ``CacheConfig`` also requires
    ``chunk_size <= block_size_frames`` that means a low-latency streaming
    deployment (chunk 8) rather than an arbitrary one — plausible, but not the
    default.  So the shipped default of 16 is the only thing that kept this out
    of production, which is a thinner margin than "head_dim 64 is clean" implied.

    Asserted as **self**-consistency (two eager runs, two captured runs) rather
    than against a reference: the computation is deterministic, so any run-to-run
    difference is a race, and that needs no oracle.  ``eager_vs_eager`` is the
    load-bearing assertion — it holds with capture switched off, which is what
    separates a genuine race from a graph-pool artefact.
    """
    runs = _drive_capture(dim, heads, batch=2, block_size=block_size)
    assert runs["eager_vs_eager"] == 0.0, f"eager path is not deterministic: {runs}"
    assert runs["graph_vs_graph"] == 0.0, f"capture is not reproducible: {runs}"
    assert runs["graph_vs_eager"] == 0.0, f"graph diverged from eager: {runs}"


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
