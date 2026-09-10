# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/engine/input_processor.py`` -- everything between the waveform and the encoder.

One module, one set of buffers, one file.  It used to be five, across two CI
families: the feature ring's growth and compaction, the C++ twin of that ring,
the host-staging helper, the sample-rate contract, and -- filed under
*features* -- the pinned-audio and collate buffers plus the captured feature
graph.  ``test_features.py::TestStreamingFeatureAppend`` and
``test_feature_buffer_growth.py`` were pinning the same function,
``InputProcessor._plan_append_features``, from two different families.

What is genuinely ``oasr/features`` -- the extractors and the kernels under
them -- stayed in ``tests/features/``.
"""

from __future__ import annotations

import random

import pytest
import torch

from oasr.engine.config import EngineConfig
from oasr.engine.input_processor import (
    _FEATURE_HEADROOM_APPENDS,
    _FEATURE_HEADROOM_MAX,
    InputProcessor,
)
from oasr.engine.request import Request
from oasr.features import FeatureConfig
from oasr.utils.staging import to_device

# --------------------------------------------------------------------------
# The feature ring: growth, compaction, and the absolute frame index
# --------------------------------------------------------------------------


FEAT_DIM = 8
N_NEW = 62  # one encoder window's worth of new frames, as streaming produces


class _RingReq:
    """The attributes ``_plan_append_features`` reads and writes."""

    def __init__(self):
        self.feature_buffer = None
        self.feature_frames = 0
        self.feature_cursor = 0
        self.feature_base = 0


def _steady_state(ticks: int, n_new: int = N_NEW, consume: int | None = None):
    """Append ``n_new`` frames per tick and consume as many, as a live stream does.

    Returns ``(reallocations, appends, final_capacity)``.
    """
    consume = n_new if consume is None else consume
    req = _RingReq()
    reallocs = 0
    for _ in range(ticks):
        dsts: list = []
        srcs: list = []
        before = req.feature_buffer
        InputProcessor._plan_append_features(
            None, req, torch.zeros(n_new, FEAT_DIM), FEAT_DIM, dsts, srcs
        )
        if req.feature_buffer is not before:
            reallocs += 1
        for d, s in zip(dsts, srcs):
            d.copy_(s)
        req.feature_cursor = min(req.feature_cursor + consume, req.feature_frames)
    return reallocs, ticks, req.feature_buffer.size(0)


class TestSteadyStateDoesNotThrash:
    def test_reallocation_is_rare(self):
        reallocs, appends, _ = _steady_state(200)
        rate = reallocs / appends
        # The old rule reallocated on ~91 % of appends; anything near that is the
        # thrash returning.
        assert rate < 0.15, f"{rate:.1%} of appends reallocated (was ~91 % before)"

    def test_capacity_exceeds_the_live_window_by_real_headroom(self):
        """``2 x live`` is what made the loop self-perpetuating."""
        _, _, cap = _steady_state(200)
        assert cap >= 4 * N_NEW, f"capacity {cap} leaves no room to amortise"

    def test_buffer_does_not_grow_without_bound(self):
        """Compaction must still happen — ``have`` grows every tick forever."""
        _, _, cap = _steady_state(2000)
        assert cap < 40 * N_NEW, f"capacity {cap} suggests compaction stopped"


class TestFrameBookkeepingSurvivesCompaction:
    def test_absolute_frame_index_is_preserved(self):
        """``feature_base + feature_cursor`` is the stream's absolute input-frame
        index — the gate reads it to decide which seconds a window covers, so a
        compaction that rebases the cursor must move the same amount into the
        base."""
        req = _RingReq()
        for tick in range(120):
            dsts: list = []
            srcs: list = []
            before_abs = req.feature_base + req.feature_cursor
            InputProcessor._plan_append_features(
                None, req, torch.zeros(N_NEW, FEAT_DIM), FEAT_DIM, dsts, srcs
            )
            for d, s in zip(dsts, srcs):
                d.copy_(s)
            assert (
                req.feature_base + req.feature_cursor == before_abs
            ), f"tick {tick}: compaction moved the absolute frame index"
            req.feature_cursor = min(req.feature_cursor + N_NEW, req.feature_frames)

    def test_live_frames_survive_a_compaction_bit_exactly(self):
        """Whatever a compaction keeps must be the same frames, in order."""
        req = _RingReq()
        counter = 0.0
        expected: list[float] = []
        for _ in range(80):
            dsts: list = []
            srcs: list = []
            block = (
                torch.arange(counter, counter + N_NEW, dtype=torch.float32)
                .unsqueeze(1)
                .expand(N_NEW, FEAT_DIM)
                .contiguous()
            )
            counter += N_NEW
            InputProcessor._plan_append_features(None, req, block, FEAT_DIM, dsts, srcs)
            for d, s in zip(dsts, srcs):
                d.copy_(s)
            expected.extend(block[:, 0].tolist())
            # Consume all but a 40-frame tail, so something real is always live.
            keep = 40
            req.feature_cursor = max(0, req.feature_frames - keep)
            live = req.feature_buffer[req.feature_cursor : req.feature_frames, 0]
            assert live.tolist() == expected[-live.numel() :], "live frames corrupted"


class TestUnconsumedStreamStillGrows:
    def test_a_backlogged_stream_keeps_every_frame(self):
        """A stream nobody consumes must not lose frames to the new rule."""
        reallocs, _, cap = _steady_state(60, consume=0)
        assert cap >= 60 * N_NEW, "buffer failed to grow for an unconsumed stream"
        assert reallocs < 30, f"{reallocs} reallocations to grow 60 appends"


class TestFixedWindowFrontendHeadroomIsBounded:
    def test_a_3000_frame_window_does_not_reserve_tens_of_thousands(self):
        """``whisper_logmel`` hands over a whole 30 s window at once; headroom is
        capped in absolute frames so that does not multiply."""
        _, _, cap = _steady_state(6, n_new=3000)
        assert cap < 3000 * 4, f"capacity {cap} for a 3000-frame window"


# --------------------------------------------------------------------------
# The same rule in C++ (oasr._C.streaming.append_features)
# --------------------------------------------------------------------------


_C_streaming = None
try:
    from oasr import _C  # type: ignore[attr-defined]

    _C_streaming = _C.streaming
except (ImportError, AttributeError):  # pragma: no cover
    _C_streaming = None

needs_native = pytest.mark.skipif(_C_streaming is None, reason="oasr._C.streaming not built")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FEAT_DIM = 8


class _NativeReq:
    """The four attributes the append reads and writes."""

    __slots__ = ("feature_buffer", "feature_frames", "feature_cursor", "feature_base")

    def __init__(self):
        self.feature_buffer = None
        self.feature_frames = 0
        self.feature_cursor = 0
        self.feature_base = 0

    def state(self):
        return (self.feature_frames, self.feature_cursor, self.feature_base)


def _python_step(reqs, feats, lens):
    dsts: list = []
    srcs: list = []
    for i, r in enumerate(reqs):
        if lens[i] > 0:
            InputProcessor._plan_append_features(
                None, r, feats[i, : lens[i], :], FEAT_DIM, dsts, srcs
            )
    if len(dsts) == 1:
        dsts[0].copy_(srcs[0])
    elif dsts:
        torch._foreach_copy_(dsts, srcs)


def _native_step(reqs, feats, lens):
    buffers, frames, cursors, base_delta, n_realloc = _C_streaming.append_features(
        [r.feature_buffer for r in reqs],
        [r.feature_frames for r in reqs],
        [r.feature_cursor for r in reqs],
        feats,
        list(lens),
        FEAT_DIM,
        _FEATURE_HEADROOM_APPENDS,
        _FEATURE_HEADROOM_MAX,
    )
    for r, b, f, c, d in zip(reqs, buffers, frames, cursors, base_delta):
        r.feature_buffer, r.feature_frames, r.feature_cursor = b, f, c
        r.feature_base += d
    return n_realloc


def _feats(B, T, start, device):
    return torch.arange(
        start, start + B * T * FEAT_DIM, dtype=torch.float32, device=device
    ).reshape(B, T, FEAT_DIM)


@needs_native
@needs_cuda
class TestMatchesThePythonReference:
    @pytest.mark.parametrize("seed", range(6))
    def test_random_streaming_history(self, seed):
        """Drive both over the same history and compare at every step."""
        dev = torch.device("cuda")
        rng = random.Random(seed)
        B = rng.randint(1, 6)
        py_reqs = [_NativeReq() for _ in range(B)]
        cpp_reqs = [_NativeReq() for _ in range(B)]
        counter = 0.0

        for step in range(30):
            # Zero-length appends are the common case for a stream that has no
            # full window this tick; they must pass through untouched.
            lens = [rng.choice([0, 1, 62, rng.randint(1, 70)]) for _ in range(B)]
            T = max(1, max(lens))
            feats = _feats(B, T, counter, dev)
            counter += B * T * FEAT_DIM

            _python_step(py_reqs, feats, lens)
            _native_step(cpp_reqs, feats, lens)

            for i, (a, b) in enumerate(zip(py_reqs, cpp_reqs)):
                where = f"seed {seed} step {step} stream {i}"
                assert a.state() == b.state(), f"{where}: bookkeeping diverged"
                assert (a.feature_buffer is None) == (b.feature_buffer is None), where
                if a.feature_buffer is not None:
                    assert a.feature_buffer.size(0) == b.feature_buffer.size(0), (
                        f"{where}: capacity {a.feature_buffer.size(0)} vs "
                        f"{b.feature_buffer.size(0)}"
                    )
                    n = a.feature_frames
                    assert torch.equal(
                        a.feature_buffer[:n], b.feature_buffer[:n]
                    ), f"{where}: live frames differ"

            # Consume like the engine does, keeping both sides in lockstep.
            for a, b in zip(py_reqs, cpp_reqs):
                a.feature_cursor = min(a.feature_cursor + rng.randint(0, 62), a.feature_frames)
                b.feature_cursor = a.feature_cursor


@needs_native
@needs_cuda
class TestContract:
    def test_a_stream_with_no_new_frames_is_untouched(self):
        dev = torch.device("cuda")
        reqs = [_NativeReq(), _NativeReq()]
        _native_step(reqs, _feats(2, 4, 0, dev), [4, 0])
        assert reqs[0].feature_frames == 4
        assert reqs[1].feature_frames == 0 and reqs[1].feature_buffer is None

    def test_compaction_preserves_the_absolute_frame_index(self):
        """``feature_base + feature_cursor`` is what the speech gate reads, so a
        rebase must move exactly what it removed."""
        dev = torch.device("cuda")
        r = _NativeReq()
        for _ in range(200):
            before = r.feature_base + r.feature_cursor
            _native_step([r], _feats(1, 62, 0, dev), [62])
            assert r.feature_base + r.feature_cursor == before
            r.feature_cursor = min(r.feature_cursor + 62, r.feature_frames)

    def test_live_frames_survive_compaction_bit_exactly(self):
        dev = torch.device("cuda")
        r = _NativeReq()
        expected: list = []
        counter = 0.0
        for _ in range(60):
            feats = _feats(1, 62, counter, dev)
            counter += 62 * FEAT_DIM
            _native_step([r], feats, [62])
            expected.extend(feats[0, :, 0].tolist())
            r.feature_cursor = max(0, r.feature_frames - 40)
            live = r.feature_buffer[r.feature_cursor : r.feature_frames, 0]
            assert live.tolist() == expected[-live.numel() :]

    def test_reallocation_is_rare_at_steady_state(self):
        """The same property ``test_feature_buffer_growth.py`` pins for Python."""
        dev = torch.device("cuda")
        r = _NativeReq()
        reallocs = 0
        for _ in range(200):
            reallocs += _native_step([r], _feats(1, 62, 0, dev), [62])
            r.feature_cursor = min(r.feature_cursor + 62, r.feature_frames)
        assert reallocs / 200 < 0.15, f"{reallocs / 200:.1%} of appends reallocated"

    def test_mismatched_lengths_are_rejected(self):
        dev = torch.device("cuda")
        with pytest.raises((ValueError, RuntimeError)):
            _C_streaming.append_features(
                [None, None],
                [0],
                [0, 0],
                _feats(2, 4, 0, dev),
                [4, 4],
                FEAT_DIM,
                _FEATURE_HEADROOM_APPENDS,
                _FEATURE_HEADROOM_MAX,
            )


# --------------------------------------------------------------------------
# oasr.utils.staging.to_device: the per-step H2D copies
# --------------------------------------------------------------------------


def _congest(device: torch.device, iters: int = 64) -> torch.Tensor:
    """Queue enough GPU work that a synchronising call cannot hide.

    Deliberately congests the stream rather than trusting a timing threshold:
    the assertion below is about pipeline *state* (is work still outstanding),
    which is what the staging discipline actually buys.
    """
    big = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    for _ in range(iters):
        big = torch.mm(big, big)
    return big


@pytest.mark.cuda
class TestHostStaging:
    def test_values_match_the_pageable_build(self, device):
        values = [7, 0, 3, 11]
        staged = to_device(values, dtype=torch.long, device=device)
        assert staged.dtype is torch.long
        assert staged.device.type == "cuda"
        assert staged.tolist() == values

    def test_dtype_is_honoured(self, device):
        staged = to_device([1, 2, 3], dtype=torch.int32, device=device)
        assert staged.dtype is torch.int32
        assert staged.tolist() == [1, 2, 3]

    def test_does_not_drain_the_stream(self, device):
        """The copy is issued, not awaited — queued work is still outstanding."""
        torch.cuda.synchronize(device)
        keep_alive = _congest(device)  # noqa: F841 — must outlive the assertion
        staged = to_device(list(range(32)), dtype=torch.long, device=device)
        outstanding = not torch.cuda.current_stream(device).query()
        torch.cuda.synchronize(device)
        assert staged.tolist() == list(range(32))
        assert outstanding, (
            "to_device drained the stream — the staged copy must be an async DMA "
            "out of pinned memory, not a pageable copy that synchronises first"
        )

    def test_pageable_build_does_drain_the_stream(self, device):
        """The control: what the call sites used to do, and why it cost.

        Without this the test above proves nothing — a GPU fast enough to
        finish the congestion would pass it either way.
        """
        torch.cuda.synchronize(device)
        keep_alive = _congest(device)  # noqa: F841
        torch.tensor(list(range(32)), dtype=torch.long, device=device)
        assert torch.cuda.current_stream(device).query(), (
            "a pageable host->device copy is expected to synchronise the stream; "
            "if this fails the platform changed and the staging discipline needs "
            "re-measuring rather than re-asserting"
        )
        torch.cuda.synchronize(device)

    def test_cpu_device_takes_the_plain_path(self):
        staged = to_device([4, 5], dtype=torch.int32, device="cpu")
        assert staged.device.type == "cpu"
        assert not staged.is_pinned()
        assert staged.tolist() == [4, 5]

    def test_empty_sequence(self, device):
        staged = to_device([], dtype=torch.long, device=device)
        assert staged.numel() == 0
        assert staged.device.type == "cuda"


# --------------------------------------------------------------------------
# One sample rate -- the model's -- and a loud refusal otherwise
# --------------------------------------------------------------------------


def _processor(sample_rate: int = 16000, **fcfg) -> InputProcessor:
    cfg = EngineConfig(
        ckpt_dir="/nonexistent",
        feature_config=FeatureConfig(sample_rate=sample_rate, num_mel_bins=80, **fcfg),
    )
    return InputProcessor(cfg, torch.device("cpu"))


class TestCheckSampleRate:
    def test_matching_rate_is_accepted(self):
        _processor().check_sample_rate(16000)

    def test_unspecified_rate_is_accepted(self):
        """``None`` means "the model's rate" — the engine resolves it before
        building the request, so the check must not treat it as a mismatch."""
        _processor().check_sample_rate(None)

    def test_mismatch_names_both_rates(self):
        with pytest.raises(ValueError, match=r"8000 Hz.*requires 16000 Hz"):
            _processor().check_sample_rate(8000)

    def test_mismatch_message_says_the_engine_does_not_resample(self):
        """The error has to tell the caller what to do; "invalid sample rate"
        alone sends them looking for a config knob that does not exist."""
        with pytest.raises(ValueError, match="does not resample"):
            _processor().check_sample_rate(44100)

    def test_the_accepted_rate_comes_from_the_feature_config(self):
        """Not a hardcoded 16000.  Every checkpoint in tree happens to run at
        16 kHz, which is exactly what would let a hardcoded constant survive."""
        proc = _processor(sample_rate=8000)
        proc.check_sample_rate(8000)
        with pytest.raises(ValueError, match=r"16000 Hz.*requires 8000 Hz"):
            proc.check_sample_rate(16000)


class TestPreparePaths:
    def test_prepare_offline_accepts_the_model_rate(self):
        proc = _processor()
        req = Request(audio=torch.zeros(16000), streaming=False, sample_rate=16000)
        proc.prepare_offline(req)
        assert req.num_frames > 0

    def test_prepare_offline_rejects_a_mismatch(self):
        proc = _processor()
        req = Request(audio=torch.zeros(8000), streaming=False, sample_rate=8000)
        with pytest.raises(ValueError, match="requires 16000 Hz"):
            proc.prepare_offline(req)

    def test_prepare_streaming_rejects_at_open(self):
        """Not on the first chunk: by then the client has been told the stream
        is live, and a mid-stream error is far harder to attribute."""
        proc = _processor()
        req = Request(audio=None, streaming=True, sample_rate=44100)
        with pytest.raises(ValueError, match="requires 16000 Hz"):
            proc.prepare_streaming(req)
        # And nothing was set up for it.
        assert req.audio_chunks is None

    def test_prepare_streaming_accepts_the_model_rate(self):
        proc = _processor()
        req = Request(audio=None, streaming=True, sample_rate=16000)
        proc.prepare_streaming(req)
        assert req.audio_chunks is not None


# --------------------------------------------------------------------------
# Pinned audio, collate buffers, the captured feature graph, and the streaming append
# --------------------------------------------------------------------------


@pytest.mark.cuda
class TestGraphedFeatureExtraction:
    """Bit-exact parity between captured replay and the eager batched path."""

    def _make_cfg(self, *, feature_type: str = "fbank"):
        from oasr.features import FeatureConfig

        return FeatureConfig(dither=0.0, feature_type=feature_type)

    def _eager_feats(
        self,
        wave_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
        bucket: int,
        t_pad: int,
        cfg,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run the unbatched-graph eager path on a (bucket, t_pad) buffer."""
        from oasr.features.batched import batched_fbank, batched_mfcc

        device = torch.device("cuda")
        B_active = wave_cpu.size(0)
        T = wave_cpu.size(1)
        wav_gpu = torch.zeros(bucket, t_pad, dtype=torch.float32, device=device)
        wav_gpu[:B_active, :T] = wave_cpu.to(device)
        len_gpu = torch.zeros(bucket, dtype=torch.int64, device=device)
        len_gpu[:B_active] = lengths_cpu.to(device)
        fn = batched_mfcc if cfg.feature_type == "mfcc" else batched_fbank
        feats_f32, _ = fn(wav_gpu, len_gpu, cfg)
        return feats_f32.to(dtype=out_dtype)

    def test_batched_fbank_cuda_graph_matches_eager(self):
        """Replay output matches an eager call on the same padded inputs."""
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg()
        device = torch.device("cuda")
        chunk_samples = 16 * 4 * cfg.frame_shift_samples  # ASR steady-state stride
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=device,
            feature_config=cfg,
            output_dtype=torch.float16,
            chunk_samples=chunk_samples,
            max_batch_size=8,
        )
        assert gfe.buckets == [1, 2, 4, 8]

        torch.manual_seed(0)
        for B_active in [1, 2, 3, 5, 8]:
            bucket = gfe.pick_bucket(B_active)
            assert bucket is not None
            # Vary T inside [frame_len, t_pad].
            for T in [
                cfg.frame_length_samples,
                chunk_samples,
                chunk_samples + 117,
                gfe.t_pad,
            ]:
                wave = torch.randn(B_active, T, dtype=torch.float32) * 1000.0
                lengths = torch.full((B_active,), T, dtype=torch.int64)

                feats_graph_view = gfe.replay(B_active, wave, lengths)
                assert feats_graph_view is not None
                assert feats_graph_view.dtype == torch.float16
                # Caller slices to the first B_active rows.
                feats_graph = feats_graph_view[:B_active].clone()

                feats_eager = self._eager_feats(
                    wave, lengths, bucket, gfe.t_pad, cfg, torch.float16
                )[:B_active]

                torch.testing.assert_close(
                    feats_graph,
                    feats_eager,
                    rtol=0,
                    atol=0,
                    msg=f"B_active={B_active} T={T}",
                )

    def test_batched_mfcc_cuda_graph_matches_eager(self):
        """Same bit-exact check on the MFCC path."""
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg(feature_type="mfcc")
        device = torch.device("cuda")
        chunk_samples = 16 * 4 * cfg.frame_shift_samples
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=device,
            feature_config=cfg,
            output_dtype=torch.float32,
            chunk_samples=chunk_samples,
            max_batch_size=4,
        )

        torch.manual_seed(1)
        for B_active in [1, 4]:
            bucket = gfe.pick_bucket(B_active)
            T = chunk_samples + 17
            wave = torch.randn(B_active, T, dtype=torch.float32) * 500.0
            lengths = torch.full((B_active,), T, dtype=torch.int64)
            feats_graph = gfe.replay(B_active, wave, lengths)[:B_active].clone()
            feats_eager = self._eager_feats(wave, lengths, bucket, gfe.t_pad, cfg, torch.float32)[
                :B_active
            ]
            torch.testing.assert_close(feats_graph, feats_eager, rtol=0, atol=0)

    def test_pick_bucket_returns_smallest_fit(self):
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg()
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=torch.device("cuda"),
            feature_config=cfg,
            output_dtype=torch.float16,
            chunk_samples=16 * 4 * cfg.frame_shift_samples,
            max_batch_size=16,
        )
        assert gfe.buckets == [1, 2, 4, 8, 16]
        assert gfe.pick_bucket(1) == 1
        assert gfe.pick_bucket(3) == 4
        assert gfe.pick_bucket(8) == 8
        assert gfe.pick_bucket(9) == 16
        assert gfe.pick_bucket(17) is None  # oversize → eager fallback

    def test_custom_batch_buckets_override(self):
        """Explicit ``batch_buckets`` overrides the default power-of-two ladder."""
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg()
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=torch.device("cuda"),
            feature_config=cfg,
            output_dtype=torch.float16,
            chunk_samples=16 * 4 * cfg.frame_shift_samples,
            max_batch_size=64,
            batch_buckets=[8, 32],
        )
        assert gfe.buckets == [8, 32]
        assert gfe.pick_bucket(1) == 8
        assert gfe.pick_bucket(8) == 8
        assert gfe.pick_bucket(9) == 32
        assert gfe.pick_bucket(33) is None

    def test_oversize_returns_none(self):
        """Combined waveform longer than ``t_pad`` triggers the eager fallback."""
        from oasr.engine.graph_cache import GraphedFeatureExtraction

        cfg = self._make_cfg()
        chunk_samples = 16 * 4 * cfg.frame_shift_samples
        gfe = GraphedFeatureExtraction(
            pool=torch.cuda.graph_pool_handle(),
            device=torch.device("cuda"),
            feature_config=cfg,
            output_dtype=torch.float16,
            chunk_samples=chunk_samples,
            max_batch_size=4,
        )
        oversize_T = gfe.t_pad + 1
        wave = torch.zeros(2, oversize_T, dtype=torch.float32)
        lengths = torch.tensor([oversize_T, oversize_T], dtype=torch.int64)
        assert gfe.replay(2, wave, lengths) is None


class TestCollateOutputIsNotReused:
    """``collate`` must return a fresh feature tensor every call.

    ``OfflineExecutor``'s collate prefetch holds *two* micro-batches' features
    alive at once — one being forwarded, one staged for the next tick.  Every
    other buffer on that path (``_wav_flat``, ``_wav_padded``) is deliberately
    reused, so the one that must not be is worth pinning: a reused feature
    buffer would let the staged batch overwrite the one the encoder is reading,
    which corrupts a transcript rather than raising.  The graph-replay buffers
    are the same class of hazard, and they *did* ship.
    """

    def _proc(self):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="x", device="cpu", dtype=torch.float32)
        return InputProcessor(cfg, torch.device("cpu"))

    def _batch(self, proc, n, samples):
        from oasr.engine.request import Request

        out = []
        for i in range(n):
            req = Request(request_id=f"c{i}", streaming=False)
            req.audio = torch.randn(samples)
            req.sample_rate = 16000
            proc.prepare_offline(req)
            out.append(req)
        return out

    def test_two_collates_do_not_alias(self):
        proc = self._proc()
        # Held simultaneously, which is what the prefetch does — otherwise the
        # caching allocator could hand back the same address legitimately.
        f1, l1 = proc.collate(self._batch(proc, 4, 16000))
        f2, l2 = proc.collate(self._batch(proc, 4, 16000))
        assert f1.shape == f2.shape
        assert f1.data_ptr() != f2.data_ptr()
        assert l1.data_ptr() != l2.data_ptr()


class TestStagingBuffers:
    """M4/M5: staging must be reused per step but bounded across the process."""

    def _proc(self, **overrides):

        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="x", device="cpu", **overrides)
        return InputProcessor(cfg, torch.device("cpu"))

    def test_offline_buffer_is_reused_between_calls(self):
        p = self._proc()
        a = p._flat_host(1024)
        b = p._flat_host(1024)
        assert a.data_ptr() == b.data_ptr()

    def test_offline_buffer_grows_geometrically(self):
        p = self._proc()
        p._flat_host(1024)
        first = p._wav_flat.numel()
        p._flat_host(first + 1)
        assert p._wav_flat.numel() >= 2 * first

    def test_an_outlier_batch_is_not_retained(self):
        """One huge request must not pin its peak for the process lifetime.

        Geometric growth sized by the longest utterance ever seen never shrinks,
        and pinned host memory is process-global.
        """
        p = self._proc()
        p._max_staging_elems = 4096
        p._flat_host(1024)
        retained = p._wav_flat.numel()
        big = p._flat_host(1_000_000)
        assert big.numel() == 1_000_000
        assert p._wav_flat.numel() == retained, "the outlier was retained"

    def test_retained_buffer_never_exceeds_the_cap(self):
        p = self._proc()
        p._max_staging_elems = 4096
        p._flat_host(4096)
        assert p._wav_flat.numel() <= 4096

    def test_streaming_staging_is_reused_within_a_slot(self):
        p = self._proc()
        slot = p._next_stream_slot()
        a = p._stream_host(slot, 4, 100)
        b = p._stream_host(slot, 4, 100)
        assert a.data_ptr() == b.data_ptr()
        la = p._stream_lengths_host(slot, 4)
        lb = p._stream_lengths_host(slot, 4)
        assert la.data_ptr() == lb.data_ptr()

    def test_consecutive_streaming_steps_get_different_buffers(self):
        """Double buffering: back-to-back steps must not share staging memory.

        The pinned pair is read by an async H2D, so a step that rewrites the
        buffer the previous step's copy is still reading corrupts it — measured
        as nemotron streaming WER 2.44% -> 2.53% under a co-tenant GPU load, and
        the *reason* the buffers rotate rather than being reused in place
        (`.artifacts/known_issues.md`
        "Lessons that outlived their bugs").  Reuse two steps apart is fine, and
        gated by the slot's completion event on CUDA.
        """
        p = self._proc()
        seen = []
        for _ in range(4):
            slot = p._next_stream_slot()
            seen.append(p._stream_host(slot, 4, 100).data_ptr())
            p._stream_lengths_host(slot, 4)
        assert seen[0] != seen[1], "consecutive steps shared a staging buffer"
        assert seen[0] == seen[2] and seen[1] == seen[3], "slots should cycle, not grow"

    def test_release_drops_everything(self):
        p = self._proc()
        p._flat_host(64)
        slot = p._next_stream_slot()
        p._stream_host(slot, 2, 8)
        p._stream_lengths_host(slot, 2)
        p.release_staging()
        assert p._wav_flat is None
        assert all(s.flat is None and s.lens is None for s in p._stream_slots)
        assert all(s.ready is None for s in p._stream_slots)


class TestPinnedAudioBuffers:
    """``new_audio_buffer`` — the buffer the front-end fills so ``collate``
    can DMA straight from it instead of packing the batch into staging."""

    def _proc(self, device="cpu", **overrides):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="x", device=device, **overrides)
        return InputProcessor(cfg, torch.device(device))

    def test_cpu_engine_declines(self):
        """No CUDA context to page-lock against — the caller uses the heap."""
        assert self._proc().new_audio_buffer(16000) is None

    @pytest.mark.cuda
    def test_offers_pinned_memory(self, device):
        p = self._proc(device="cuda")
        buf = p.new_audio_buffer(16000)
        assert buf is not None
        assert buf.is_pinned() and buf.dtype is torch.float32
        assert buf.numel() == 16000

    @pytest.mark.cuda
    def test_declines_past_the_cap(self, device):
        """Page-locked memory is process-global; one long request must not be
        able to reserve an unbounded amount of it."""
        p = self._proc(device="cuda", max_pinned_audio_seconds=1.0)
        sr = p._feature_config.sample_rate
        assert p.new_audio_buffer(sr) is not None
        assert p.new_audio_buffer(sr + 1) is None

    @pytest.mark.cuda
    def test_zero_cap_declines_everything(self, device):
        p = self._proc(device="cuda", max_pinned_audio_seconds=0.0)
        assert p.new_audio_buffer(16000) is None

    def test_non_positive_size_declines(self):
        p = self._proc()
        assert p.new_audio_buffer(0) is None
        assert p.new_audio_buffer(-1) is None

    @pytest.mark.cuda
    def test_pinned_and_unpinned_batches_agree(self, device):
        """The two collate paths must produce the same device batch.

        The pinned path skips the host pack entirely and DMAs each row into
        place; the unpinned one packs into staging first.  Padding and
        ``audio_scale`` are applied identically (on the GPU, after padding), so
        this is an exact comparison, not a tolerance.
        """
        p = self._proc(device="cuda")
        waves = [torch.randn(n) for n in (16000, 12345, 8000, 1)]
        plain = p._padded_waveform_batch(list(waves))
        pinned = p._padded_waveform_batch([w.pin_memory() for w in waves])
        torch.cuda.synchronize()
        assert torch.equal(plain, pinned)

    @pytest.mark.cuda
    def test_mixed_batch_takes_the_pack_path(self, device):
        """One unpinned row sends the whole micro-batch through the pack — and
        still produces the same batch."""
        p = self._proc(device="cuda")
        waves = [torch.randn(4000), torch.randn(4000)]
        mixed = [waves[0].pin_memory(), waves[1]]
        torch.cuda.synchronize()
        assert torch.equal(p._padded_waveform_batch(list(waves)), p._padded_waveform_batch(mixed))


class TestStreamingFeatureStreamHandoff:
    """The feature stream → default stream hand-off must be ordered.

    ``extract_streaming_batch`` runs the H2D and the frontend on a caller-supplied
    stream so they overlap the previous step's encoder forward, then appends the
    result into each request's ``feature_buffer`` **on the current stream**.  That
    cross-stream read has to be ordered inside the method: the step loop's own
    ``wait_stream`` fires only after it returns, which is after the append has
    already been issued.

    Unordered, the append reads feature memory the frontend has not finished
    writing.  An idle GPU always hides it — the kernels beat the host to it — so
    the reproduction here **congests the feature stream on purpose** rather than
    relying on timing.  Measured cost of the missing wait on real audio: conformer
    streaming 3.70% → 99.32% WER with 195 of 200 transcripts empty, nemotron
    2.44% → 59.71%, in both cases NaN arriving one whole mel frame at a time with
    nothing raised (`.artifacts/known_issues.md` "Lessons that outlived their bugs").
    """

    def _processor_and_requests(self, device, n=3, samples=8000):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor
        from oasr.engine.request import Request
        from oasr.features import FeatureConfig

        cfg = EngineConfig(
            ckpt_dir="x",
            device=str(device),
            dtype=torch.float32,
            max_batch_size=8,
            feature_config=FeatureConfig(feature_type="fbank", num_mel_bins=80, dither=0.0),
            use_cuda_graphs=False,  # exercise the eager path; the graph path
            # stages through its own captured buffers
        )
        proc = InputProcessor(cfg, device)
        torch.manual_seed(1234)  # same audio in both arms
        reqs = []
        for i in range(n):
            req = Request(None, request_id=f"r{i}", streaming=True)
            proc.prepare_streaming(req)
            proc.append_streaming_chunk(req, torch.randn(samples).clamp(-1, 1))
            reqs.append(req)
        return proc, reqs

    @pytest.mark.cuda
    def test_features_survive_a_congested_feature_stream(self, device):
        """Features must match the single-stream result, however backed-up the
        feature stream is when the append is issued."""
        proc_ref, reqs_ref = self._processor_and_requests(device)
        proc_ref.extract_streaming_batch(reqs_ref, cuda_stream=None)
        torch.cuda.synchronize()
        reference = [r.feature_buffer[: r.feature_frames].clone() for r in reqs_ref]

        proc, reqs = self._processor_and_requests(device)
        feat_stream = torch.cuda.Stream(device=device)
        # Queue enough work on the feature stream that the frontend's kernels
        # cannot possibly have completed by the time the append is enqueued.
        with torch.cuda.stream(feat_stream):
            a = torch.randn(2048, 2048, device=device)
            b = torch.randn(2048, 2048, device=device)
            for _ in range(60):
                a = (a @ b).mul_(1e-4)
        proc.extract_streaming_batch(reqs, cuda_stream=feat_stream)
        torch.cuda.synchronize()

        for i, (req, ref) in enumerate(zip(reqs, reference)):
            got = req.feature_buffer[: req.feature_frames]
            assert not torch.isnan(got).any(), f"stream {i}: NaN in features"
            assert got.shape == ref.shape
            torch.testing.assert_close(got, ref, rtol=0, atol=0)


# ===========================================================================
# stft_frame / mel_log kernels (KG23's shared primitives)
# ===========================================================================


class TestStreamingFeatureAppend:
    """``_plan_append_features`` — the batched feature-buffer append.

    Appending each stream's new frames one call at a time was 146 device-to-device
    copies per streaming step carrying 0.09 ms of work: 21% of the wall clock
    spent submitting copies, not making them.  The batched form plans every
    stream's growth on the host and commits the copies as one
    ``torch._foreach_copy_``, whose members are unordered with respect to each
    other — so what has to hold is that no queued pair reads what another writes,
    and that the buffer contents are what the per-stream form produced.

    The compaction path is the one that had to change shape: a standalone
    compaction left a buffer exactly as long as what it kept, so a grow *always*
    followed it, chaining old->keep->new.  Two copies of the same frames, and two
    that could not have shared a batch.

    Compaction *triggers* only when the append would otherwise not fit.  The
    eager rule it replaced (``cursor >= have // 2``) was true on ~91 % of appends
    at steady state, because a stream consumes about as many frames per tick as
    it gains -- see ``tests/test_feature_buffer_growth.py``.  So a test of
    compaction has to fill the buffer first; moving the cursor is no longer
    enough on its own, which is what the two tests below assert around.
    """

    def _proc(self):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        return InputProcessor(EngineConfig(ckpt_dir="x", device="cpu"), torch.device("cpu"))

    @staticmethod
    def _req():
        from oasr.engine.request import Request

        req = Request(request_id="r", streaming=True)
        req.feature_buffer = None
        req.feature_frames = 0
        req.feature_cursor = 0
        return req

    def _append(self, proc, req, frames, feat_dim):
        dsts, srcs = [], []
        proc._plan_append_features(req, frames, feat_dim, dsts, srcs)
        for d, s in zip(dsts, srcs):
            d.copy_(s)
        return req

    def test_appends_are_contiguous_and_ordered(self):
        proc, req, F = self._proc(), self._req(), 4
        expected = []
        for step in range(6):
            frames = torch.full((3, F), float(step))
            expected.append(frames)
            self._append(proc, req, frames, F)
        want = torch.cat(expected)
        assert req.feature_frames == want.size(0)
        torch.testing.assert_close(req.feature_buffer[: req.feature_frames], want)

    def _fill_to_capacity(self, proc, req, F, chunk=4):
        """Append ``chunk`` frames at a time until one more would not fit.

        The trigger is "the append overflows", not "the cursor is past half", so
        reaching the compaction edge means filling the buffer.  The loop stops
        one append short of the capacity, so nothing inside it reallocates.
        """
        self._append(proc, req, torch.zeros(chunk, F), F)
        while req.feature_frames + chunk <= req.feature_buffer.size(0):
            self._append(proc, req, torch.full((chunk, F), float(req.feature_frames)), F)
        assert req.feature_frames == req.feature_buffer.size(0)
        return req.feature_frames

    def test_a_large_cursor_alone_does_not_compact(self):
        """The rule that replaced ``cursor >= have // 2``: room still means no copy."""
        proc, req, F = self._proc(), self._req(), 4
        self._append(proc, req, torch.zeros(4, F), F)
        buf, cap = req.feature_buffer, req.feature_buffer.size(0)
        req.feature_cursor = req.feature_frames  # fully consumed, but there is room

        self._append(proc, req, torch.full((4, F), 99.0), F)
        assert req.feature_buffer is buf, "reallocated with capacity to spare"
        assert req.feature_cursor == 4, "rebased the cursor without compacting"
        assert req.feature_buffer.size(0) == cap

    def test_consumed_prefix_is_dropped_without_losing_live_frames(self):
        """Compaction moves the cursor to 0 and keeps everything after it."""
        proc, req, F = self._proc(), self._req(), 4
        have = self._fill_to_capacity(proc, req, F)

        req.feature_cursor = 12  # the append below no longer fits, so it compacts
        live = req.feature_buffer[12:have].clone()
        base = req.feature_base
        self._append(proc, req, torch.full((4, F), 99.0), F)

        keep = have - 12
        assert req.feature_cursor == 0, "compaction must rebase the cursor"
        assert req.feature_base == base + 12, "the rebased frames must move into the base"
        assert req.feature_frames == keep + 4
        torch.testing.assert_close(req.feature_buffer[:keep], live)
        torch.testing.assert_close(req.feature_buffer[keep : keep + 4], torch.full((4, F), 99.0))

    def test_compaction_relocates_once(self):
        """Not old->keep->new: one allocation, one copy of the retained frames."""
        proc, req, F = self._proc(), self._req(), 4
        self._fill_to_capacity(proc, req, F)
        req.feature_cursor = 12

        dsts, srcs = [], []
        proc._plan_append_features(req, torch.full((4, F), 99.0), F, dsts, srcs)
        assert len(dsts) == 2, "one relocation copy plus the append, and no more"
        # Every destination lives in the new buffer; no source does — which is
        # what makes the pairs safe to run unordered.
        new_storage = req.feature_buffer.untyped_storage().data_ptr()
        assert all(d.untyped_storage().data_ptr() == new_storage for d in dsts)
        assert all(s.untyped_storage().data_ptr() != new_storage for s in srcs)

    def test_growth_preserves_the_unconsumed_prefix(self):
        """A grow with a small cursor must not drop the frames before it."""
        proc, req, F = self._proc(), self._req(), 4
        self._append(proc, req, torch.arange(200 * F, dtype=torch.float32).reshape(200, F), F)
        req.feature_cursor = 1  # well under have // 2: no compaction, only growth
        before = req.feature_buffer[:200].clone()
        self._append(proc, req, torch.full((200, F), 7.0), F)

        assert req.feature_cursor == 1, "growth alone must not rebase the cursor"
        assert req.feature_frames == 400
        torch.testing.assert_close(req.feature_buffer[:200], before)
        torch.testing.assert_close(req.feature_buffer[200:400], torch.full((200, F), 7.0))


class TestStreamingSegmentPack:
    """The streaming pack keeps each stream's pieces apart until one batched
    ``cat`` writes them straight into the staging row.

    Two costs were being paid to build a buffer that is copied again a moment
    later: concatenating the carried-over tail onto every stream's chunk
    (0.44 ms/step at 64 streams), then copying each result into the padded
    staging batch (0.46 ms/step).  Packing the pieces directly is one pass
    instead of two.

    What has to hold is that the packed batch is *exactly* what concatenating
    would have produced — same bytes, same padding, same lengths — for ragged
    cohorts as well as steady-state ones, since a wrong row here is a wrong
    transcript with nothing raised.
    """

    def _proc(self, **overrides):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="x", device="cpu", **overrides)
        return InputProcessor(cfg, torch.device("cpu"))

    @staticmethod
    def _inputs(pieces, flush=False):
        from oasr.engine.input_processor import _StreamInput

        return [
            _StreamInput(
                request=None,  # type: ignore[arg-type]
                segments=segs,
                n_samples=sum(int(s.numel()) for s in segs),
                flush=flush,
            )
            for segs in pieces
        ]

    def _pack(self, proc, inputs):
        """The real packing stage, not a re-implementation of it — a test that
        restates the layout rule cannot catch the layout rule changing."""
        t_max = max(inp.n_samples for inp in inputs)
        slot = proc._next_stream_slot()
        return proc._pack_streaming_waveforms(inputs, slot, t_max)

    @staticmethod
    def _reference(pieces, t_max):
        """What the per-stream concatenate-then-copy form produced."""
        out = torch.zeros(len(pieces), t_max)
        for i, segs in enumerate(pieces):
            cat = torch.cat(segs)
            out[i, : cat.numel()] = cat
        return out

    def test_equal_length_rows_match_the_concatenated_form(self):
        proc = self._proc()
        pieces = [[torch.randn(37), torch.randn(320)] for _ in range(6)]
        packed = self._pack(proc, self._inputs(pieces))
        assert torch.equal(packed, self._reference(pieces, 357))

    def test_ragged_rows_match_and_are_zero_padded(self):
        proc = self._proc()
        pieces = [
            [torch.randn(37), torch.randn(320)],
            [torch.randn(320)],  # freshly admitted stream: no carry-over tail
            [torch.randn(11), torch.randn(200), torch.randn(9)],  # closing flush pad
        ]
        packed = self._pack(proc, self._inputs(pieces))
        assert torch.equal(packed, self._reference(pieces, 357))
        assert torch.equal(packed[1, 320:], torch.zeros(37))

    def test_the_shared_pad_buffer_is_never_written_through(self):
        """Every ragged row borrows the same zero run, so a row that wrote back
        into it would silently corrupt its peers in the same step."""
        proc = self._proc()
        pieces = [[torch.randn(400)], [torch.randn(10)], [torch.randn(10)]]
        self._pack(proc, self._inputs(pieces))
        assert torch.count_nonzero(proc._stream_pad) == 0
        packed = self._pack(proc, self._inputs(pieces))
        assert torch.equal(packed, self._reference(pieces, 400))


class TestStreamingTailSuffix:
    """``_suffix`` — the retained carry-over, without materialising the join.

    ``torch.cat(segments)[start:]`` is the rule; the point is to get there
    without the ``cat``.  In steady state ``start`` lands inside the last
    segment and the result is a view, which is the whole reason the pieces are
    kept apart.
    """

    @staticmethod
    def _suffix(segments, start):
        from oasr.engine.input_processor import _suffix

        return _suffix(segments, start)

    @pytest.mark.parametrize("start", [0, 1, 9, 10, 11, 24, 25, 26, 39, 40])
    def test_matches_the_concatenated_slice(self, start):
        segs = [torch.randn(10), torch.randn(15), torch.randn(15)]
        assert torch.equal(self._suffix(segs, start), torch.cat(segs)[start:])

    def test_a_start_inside_the_last_segment_is_a_view(self):
        segs = [torch.randn(10), torch.randn(15)]
        out = self._suffix(segs, 20)
        assert out.data_ptr() == segs[1][10:].data_ptr(), "steady state must not copy"

    def test_a_start_past_the_end_is_empty(self):
        segs = [torch.randn(10), torch.randn(15)]
        assert self._suffix(segs, 25).numel() == 0

    def test_a_single_segment_behaves_like_a_slice(self):
        segs = [torch.randn(20)]
        assert torch.equal(self._suffix(segs, 7), segs[0][7:])


class TestStreamingAudioScaleSites:
    """``audio_scale`` must be applied exactly once, on whichever copy exists.

    The streaming pack writes **raw** samples so it stays one pass over the
    batch; the multiply then rides on the device copy.  Two paths have no device
    copy to ride on — a CPU engine, and the captured feature graph, which owns
    its own H2D — and scale the pinned host buffer instead.

    The hazard is the seam: a feature-graph *bucket miss* returns ``None`` and
    falls through to the eager path with a buffer that has already been scaled.
    Scaling it again is silent — every sample off by ``audio_scale``, which for
    a WeNet checkpoint is 32768.
    """

    def _processor(self, device, scale):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor
        from oasr.features import FeatureConfig

        cfg = EngineConfig(
            ckpt_dir="x",
            device=str(device),
            dtype=torch.float32,
            max_batch_size=8,
            audio_scale=scale,
            feature_config=FeatureConfig(feature_type="fbank", num_mel_bins=80, dither=0.0),
            use_cuda_graphs=False,
        )
        return InputProcessor(cfg, device)

    @staticmethod
    def _inputs(waves):
        from oasr.engine.input_processor import _StreamInput

        return [
            _StreamInput(request=None, segments=[w], n_samples=int(w.numel()), flush=False)  # type: ignore[arg-type]
            for w in waves
        ]

    @pytest.mark.cuda
    def test_a_feature_graph_miss_does_not_scale_twice(self, device):
        if device.type != "cuda":
            pytest.skip("the fallthrough only exists on the CUDA path")
        from types import SimpleNamespace

        torch.manual_seed(7)
        waves = [torch.randn(4000).clamp(-1, 1) for _ in range(3)]

        proc = self._processor(device, 32768.0)
        expected, _ = proc._run_streaming_features(self._inputs(waves), None)

        # A graph whose bucket never matches: replay returns None, so the eager
        # path runs on a buffer the graph path has already scaled.
        missing = self._processor(device, 32768.0)
        missing._feature_graph = SimpleNamespace(t_pad=1 << 30, replay=lambda *a, **k: None)
        got, _ = missing._run_streaming_features(self._inputs(waves), None)

        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @pytest.mark.cuda
    def test_the_device_scale_matches_scaling_on_the_host(self, device):
        if device.type != "cuda":
            pytest.skip("compares the device site against the host site")
        torch.manual_seed(11)
        waves = [torch.randn(4000).clamp(-1, 1) for _ in range(3)]

        scaled = self._processor(device, 32768.0)
        got, _ = scaled._run_streaming_features(self._inputs(waves), None)

        # The same audio pre-scaled on the host, through an engine that does not
        # scale at all — the two must agree bit for bit.
        plain = self._processor(device, 1.0)
        expected, _ = plain._run_streaming_features(
            self._inputs([w * 32768.0 for w in waves]), None
        )

        torch.testing.assert_close(got, expected, rtol=0, atol=0)
