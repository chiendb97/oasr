# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Silero VAD v5 rebuilt on ``oasr.layers``, against the upstream archive.

The archive is both the weight source and the oracle, which is the useful part:
there is no captured reference to drift, and every test here compares the
rebuilt network against the *same file* the weights came from.

Three properties, in the order they matter:

* **frame parity** — the probability trace must match upstream's, not merely
  correlate with it, because a VAD trace that is close but shifted still
  produces segments and they are wrong in a way nothing downstream can see;
* **the two entry points agree** — a one-shot pass over a whole file and a
  chunked pass over the same file must give the same trace, including when the
  chunks are not whole numbers of frames.  This is where a carried-state model
  goes wrong, and a single-chunk test cannot see it;
* **rows in one batched call do not contaminate each other** — the batch is
  ragged in the streaming stage, and a short row that inherits a long row's
  recurrent state produces a plausible trace for audio it never heard.
"""

from __future__ import annotations

import pytest
import torch

from oasr.vad import VadConfig, build_detector, get_vad_spec
from oasr.vad.segmenter import SpeechSegmenter

SR = 16000
WINDOW = 512


def upstream(model, wav: torch.Tensor) -> torch.Tensor:
    """The archive's own wrapper, driven the way upstream's helper drives it."""
    model.reset_states()
    frames = wav.shape[1] // WINDOW
    return torch.cat([model(wav[:, k * WINDOW : (k + 1) * WINDOW], SR) for k in range(frames)], 1)


@pytest.fixture(scope="module")
def archive(silero_vad_dir):
    import os

    return torch.jit.load(os.path.join(str(silero_vad_dir), "silero_vad.jit"), map_location="cpu")


def detector(silero_vad_dir, device="cpu", **kw):
    cfg = VadConfig(backend="silero", mode="segment", model_dir=str(silero_vad_dir), **kw)
    return build_detector(cfg.resolve("offline"), device=torch.device(device))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_it_declares_what_it_needs_and_what_it_can_do(self):
        spec = get_vad_spec("silero")
        assert spec.consumes == "waveform"
        # It runs on the audio, so it can precede the encoder in either flow,
        # and it carries an LSTM, so it says so.
        assert spec.can("presegment") and spec.can("stream") and spec.can("posthoc")
        assert spec.stateful and spec.needs_weights
        # A frame-level acoustic model needs no floor: unlike a CTC posterior it
        # is not sparse between emissions, because it has no emissions.
        assert spec.min_silence_floor_ms == 0

    def test_missing_weights_name_the_flag(self):
        cfg = VadConfig(backend="silero", mode="segment").resolve("offline")
        with pytest.raises(ValueError, match="--vad-model-dir"):
            build_detector(cfg, device=torch.device("cpu"))

    def test_an_unsupported_rate_is_refused_rather_than_resampled(self):
        from oasr.vad.detectors.silero import silero_framing

        with pytest.raises(ValueError, match="trained constant"):
            silero_framing(VadConfig(backend="silero", sample_rate=22050))

    @pytest.mark.requires_assets("SILERO_VAD_DIR")
    def test_the_frame_grid_is_the_trained_one(self, silero_vad_dir):
        det = detector(silero_vad_dir)
        assert det.framing.span == det.framing.hop == WINDOW
        # The 64 samples of left context are *state*, not framing history: a
        # frame still advances a whole window, which is what keeps the streaming
        # stage's "this call consumed frames * hop samples" rule true.
        assert det.framing.history == 0 and det.framing.prefill == 0
        assert det.seconds_per_frame == pytest.approx(WINDOW / SR)

    @pytest.mark.requires_assets("SILERO_VAD_DIR")
    def test_the_published_parameter_count(self, silero_vad_dir):
        det = detector(silero_vad_dir)
        assert sum(p.numel() for p in det.net.parameters()) == 309_633


# ---------------------------------------------------------------------------
# Parity with the archive the weights came from
# ---------------------------------------------------------------------------


@pytest.mark.requires_assets("SILERO_VAD_DIR")
class TestUpstreamParity:
    @pytest.mark.parametrize(
        "shape,scale",
        [((1, WINDOW * 40), 0.1), ((4, WINDOW * 25), 0.1), ((1, WINDOW * 20), 0.0)],
        ids=["single", "batch", "digital-silence"],
    )
    def test_one_shot_matches_the_archive(self, silero_vad_dir, archive, shape, scale):
        torch.manual_seed(0)
        wav = torch.randn(*shape) * scale
        want = upstream(archive, wav)
        got, lengths = detector(silero_vad_dir).detect(wav, torch.full((shape[0],), shape[1]))
        assert tuple(got.shape) == tuple(want.shape)
        assert lengths.tolist() == [shape[1] // WINDOW] * shape[0]
        assert (got - want).abs().max().item() < 1e-5

    def test_chunked_matches_one_shot_at_a_size_that_is_not_a_frame(self, silero_vad_dir, archive):
        """The carried-state test.

        A 900-sample chunk is not a whole number of 512-sample frames, so every
        call ends mid-frame — which is the normal case for a network chunk and
        the one where dropping the remainder, or restarting the left context,
        stops showing up as an error and starts showing up as a drift.
        """
        torch.manual_seed(1)
        wav = torch.randn(1, WINDOW * 40) * 0.1
        want = upstream(archive, wav)

        det = detector(silero_vad_dir)
        state, pieces, buffer = None, [], torch.zeros(1, 0)
        for start in range(0, wav.shape[1], 900):
            buffer = torch.cat([buffer, wav[:, start : start + 900]], 1)
            probs, frames, state = det.detect_streaming(
                buffer, torch.tensor([buffer.shape[1]]), state
            )
            n = int(frames[0])
            pieces.append(probs[:, :n])
            buffer = buffer[:, n * WINDOW :]
        got = torch.cat(pieces, 1)
        assert tuple(got.shape) == tuple(want.shape)
        assert (got - want).abs().max().item() < 1e-5

    def test_a_ragged_batch_keeps_its_rows_apart(self, silero_vad_dir, archive):
        """Rows of different lengths share one call, never one recurrent state.

        The padded tail of a short row is not audio; a row whose LSTM was stepped
        over it would carry state derived from zeros into its next chunk, and the
        trace it then produces is plausible for audio that never arrived.
        """
        torch.manual_seed(2)
        rows = [torch.randn(1, WINDOW * n) * 0.1 for n in (5, 12, 8)]
        alone = [upstream(archive, row) for row in rows]

        padded = torch.zeros(len(rows), WINDOW * 12)
        for i, row in enumerate(rows):
            padded[i, : row.shape[1]] = row[0]
        lengths = torch.tensor([row.shape[1] for row in rows])
        probs, frames, _ = detector(silero_vad_dir).detect_streaming(padded, lengths, None)

        for i, want in enumerate(alone):
            n = int(frames[i])
            assert n == want.shape[1]
            assert (probs[i, :n] - want[0]).abs().max().item() < 1e-5
            # And nothing leaked into the padding (the longest row has none).
            assert probs[i, n:].abs().sum().item() == 0.0

    def test_the_equal_length_fast_path_matches_the_grouped_one(self, silero_vad_dir):
        """``_recur``'s two routes must not be free to disagree.

        Every row having the same frame count is the steady state, so the fast
        path is what actually runs and the grouped path below it is what almost
        never does -- the arrangement where a divergence hides.  Both are driven
        here with the same input, and the grouped one is reached by asking for a
        frame count that is a row short.
        """
        torch.manual_seed(7)
        net = detector(silero_vad_dir).net
        batch, frames = 4, 6
        sequence = torch.randn(batch, frames, 128)
        hidden = torch.randn(1, batch, 128)
        cell = torch.randn(1, batch, 128)

        with torch.no_grad():
            fast = net._recur(sequence, [frames] * batch, hidden, cell)
            # One row short of uniform: same rows, same order, grouped route.
            grouped = net._recur(sequence, [frames] * (batch - 1) + [frames - 1], hidden, cell)

        for i in range(batch - 1):
            torch.testing.assert_close(fast[0][i], grouped[0][i])
            torch.testing.assert_close(fast[1][:, i], grouped[1][:, i])
            torch.testing.assert_close(fast[2][:, i], grouped[2][:, i])

    @pytest.mark.requires_assets("WAV_DIR")
    def test_segment_boundaries_agree_on_real_speech(self, silero_vad_dir, archive, wav_dir):
        """The property a caller actually sees, on audio rather than noise.

        Frame parity is the mechanism; identical *segments* through the shared
        segmenter is the claim, and it is the one that survives a device whose
        fp32 GEMMs are not bit-exact.
        """
        wav, _total = _speech_corpus(wav_dir)
        want = upstream(archive, wav)
        det = detector(silero_vad_dir)
        got, _ = det.detect(wav, torch.tensor([wav.shape[1]]))
        assert (got - want).abs().max().item() < 1e-4

        cfg = VadConfig(backend="silero", mode="segment").resolve("offline")
        mine = SpeechSegmenter(cfg, det.seconds_per_frame).run(got[0].tolist())
        theirs = SpeechSegmenter(cfg, det.seconds_per_frame).run(want[0].tolist())
        assert len(mine) == len(theirs) == 3, "the corpus has three utterances"
        for a, b in zip(mine, theirs):
            assert a.start == pytest.approx(b.start, abs=det.seconds_per_frame)
            assert a.end == pytest.approx(b.end, abs=det.seconds_per_frame)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.requires_assets("WAV_DIR")
    def test_the_device_does_not_change_the_segments(self, silero_vad_dir, archive, wav_dir):
        """On CUDA the fp32 GEMMs are not bit-exact (TF32), so the claim is the
        segmentation, not the last digit of the trace.  The engine runs this
        detector on the host by default anyway — measured, it is *faster* there,
        because a 128-wide recurrence is launch-bound on a GPU."""
        wav, _total = _speech_corpus(wav_dir)
        want = upstream(archive, wav)
        det = detector(silero_vad_dir, device="cuda")
        got, _ = det.detect(wav.cuda(), torch.tensor([wav.shape[1]], device="cuda"))
        got = got.cpu()

        cfg = VadConfig(backend="silero", mode="segment").resolve("offline")
        mine = SpeechSegmenter(cfg, det.seconds_per_frame).run(got[0].tolist())
        theirs = SpeechSegmenter(cfg, det.seconds_per_frame).run(want[0].tolist())
        assert [(round(s.start, 2), round(s.end, 2)) for s in mine] == [
            (round(s.start, 2), round(s.end, 2)) for s in theirs
        ]


def _speech_corpus(wav_dir, gap_s: float = 3.0, n: int = 3):
    """``n`` utterances separated by digital silence, as one ``(1, T)`` waveform."""
    import pathlib

    import numpy as np
    import soundfile as sf

    paths = sorted(pathlib.Path(wav_dir).glob("*.wav"))[:n]
    if len(paths) < n:
        pytest.skip(f"need {n} wav files")
    parts, total = [], 0.0
    for i, path in enumerate(paths):
        data, rate = sf.read(str(path), dtype="float32")
        assert rate == SR
        if i:
            parts.append(np.zeros(int(gap_s * SR), dtype="float32"))
            total += gap_s
        parts.append(data)
        total += len(data) / SR
    return torch.from_numpy(np.concatenate(parts)).unsqueeze(0), total


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------


@pytest.mark.requires_assets("SILERO_VAD_DIR")
class TestConversion:
    def test_every_upstream_parameter_is_accounted_for(self, silero_vad_dir, archive):
        """A silently dropped tensor would leave that layer at its random init,
        which for a 309 K model still produces probabilities in ``[0, 1]``."""
        from oasr.vad.detectors.silero import convert_silero_state_dict

        state = dict(archive.state_dict())
        converted = convert_silero_state_dict(state, 16000)
        upstream_16k = {k for k in state if k.startswith("_model.")}
        assert len(converted) == len(upstream_16k) == 15

        from oasr.vad.detectors.silero import SileroVadNet

        net = SileroVadNet(16000)
        missing, unexpected = net.load_state_dict(converted, strict=True)
        assert not missing and not unexpected

    def test_the_8k_weights_load_too(self, silero_vad_dir, archive):
        """Both models are in the archive; the rate picks the prefix."""
        from oasr.vad.detectors.silero import SileroVadNet, convert_silero_state_dict

        net = SileroVadNet(8000)
        net.load_state_dict(convert_silero_state_dict(dict(archive.state_dict()), 8000))
        probs, _h, _c = net(torch.zeros(1, 3, 256 + 32), [3], *_zero_state())
        assert probs.shape == (1, 3)

    def test_a_checkpoint_that_is_not_silero_says_so(self):
        from oasr.vad.detectors.silero import convert_silero_state_dict

        with pytest.raises(ValueError, match="does not look like a Silero VAD archive"):
            convert_silero_state_dict({"encoder.weight": torch.zeros(1)}, 16000)

    def test_a_missing_directory_names_what_it_looked_for(self, tmp_path):
        from oasr.vad.detectors.silero import load_silero_weights

        with pytest.raises(FileNotFoundError, match="silero_vad.jit"):
            load_silero_weights(str(tmp_path), 16000)

    def test_converted_weights_round_trip_through_torch_save(
        self, silero_vad_dir, archive, tmp_path
    ):
        """The archive is the convenient form, not the only one — an operator who
        would rather not ship a TorchScript file can save the plain tensors."""
        from oasr.vad.detectors.silero import convert_silero_state_dict, load_silero_weights

        converted = convert_silero_state_dict(dict(archive.state_dict()), 16000)
        path = tmp_path / "silero_vad.pt"
        torch.save(converted, path)
        loaded = load_silero_weights(str(tmp_path), 16000)
        assert set(loaded) == set(converted)


def _zero_state(batch: int = 1, hidden: int = 128):
    return torch.zeros(1, batch, hidden), torch.zeros(1, batch, hidden)


# ---------------------------------------------------------------------------
# Engine level
# ---------------------------------------------------------------------------


@pytest.mark.requires_assets("SILERO_VAD_DIR", "CKPT_DIR")
class TestEngineResolution:
    def config(self, ckpt_dir, **kw):
        from oasr.engine import EngineConfig

        base = {"ckpt_dir": ckpt_dir, "service_mode": "offline", "max_batch_size": 2}
        base.update(kw)
        return EngineConfig(**base)

    def test_offline_segmentation_uses_it(self, ckpt_dir, silero_vad_dir):
        from oasr.engine import ASREngine

        cfg = self.config(
            ckpt_dir,
            vad={"mode": "segment", "backend": "silero", "model_dir": str(silero_vad_dir)},
        )
        engine = ASREngine(cfg)
        try:
            assert engine._vad_splitter is not None
            assert engine._vad_splitter.detector.kind == "silero"
        finally:
            engine.shutdown()

    def test_streaming_segmentation_uses_it_on_the_host(self, ckpt_dir, silero_vad_dir):
        from oasr.engine import ASREngine

        cfg = self.config(
            ckpt_dir,
            service_mode="streaming",
            max_num_blocks=256,
            vad={"mode": "segment", "backend": "silero", "model_dir": str(silero_vad_dir)},
        )
        engine = ASREngine(cfg)
        try:
            stage = engine._vad_stage
            assert stage is not None and stage.needs_audio and stage.gates_encoder
            assert stage.detector.kind == "silero"
            assert stage._device.type == "cpu"
        finally:
            engine.shutdown()

    def test_weights_are_required_at_construction(self, ckpt_dir):
        """Named before anything is built, not discovered inside the factory."""
        from oasr.engine import ASREngine

        cfg = self.config(ckpt_dir, vad={"mode": "segment", "backend": "silero"})
        with pytest.raises(ValueError, match="--vad-model-dir"):
            ASREngine(cfg)

    @pytest.mark.requires_assets("WAV_DIR")
    def test_it_gates_and_resets_a_live_stream(self, ckpt_dir, silero_vad_dir, wav_dir):
        """The neural detector through the streaming gate, end to end.

        Different from the offline path in the way that matters: the per-stream
        state is three tensors (two recurrent, one of carried audio) that have to
        survive being stacked into a batched call and split back out, tick after
        tick.  A state that came apart there would still produce probabilities.
        """
        from oasr.engine import ASREngine

        wav, _total = _speech_corpus(wav_dir, gap_s=6.0)
        wav = wav.reshape(-1)
        texts, turns = {}, {}
        for tag, vad in (
            ("off", None),
            (
                "segment",
                {"mode": "segment", "backend": "silero", "model_dir": str(silero_vad_dir)},
            ),
        ):
            engine = ASREngine(
                self.config(ckpt_dir, service_mode="streaming", max_num_blocks=1024, vad=vad)
            )
            try:
                rid = engine.add_streaming_request()
                pos, final, boundaries = 0, None, 0
                total = int(wav.numel())
                while final is None:
                    if pos < total:
                        end = min(pos + SR // 5, total)
                        engine.feed_chunk(rid, wav[pos:end], is_last=(end >= total))
                        pos = end
                    for out in engine.step():
                        boundaries += out.endpoint_reason == "vad_segment"
                        if out.finished:
                            final = out
            finally:
                engine.shutdown()
            texts[tag] = final.text
            turns[tag] = (boundaries, final.segments or [])

        assert turns["off"] == (0, [])
        assert turns["segment"][0] == 2, "one turn boundary per confirmed gap"
        assert len(turns["segment"][1]) == 3
        import difflib

        ratio = difflib.SequenceMatcher(a=texts["off"].split(), b=texts["segment"].split()).ratio()
        assert ratio > 0.95, f"segmentation changed the transcript: {ratio:.3f}"

    @pytest.mark.requires_assets("WAV_DIR")
    def test_it_segments_a_real_file_at_the_speech(self, ckpt_dir, silero_vad_dir, wav_dir):
        from oasr.engine import ASREngine

        wav, total = _speech_corpus(wav_dir)
        cfg = self.config(
            ckpt_dir,
            vad={"mode": "segment", "backend": "silero", "model_dir": str(silero_vad_dir)},
        )
        engine = ASREngine(cfg)
        try:
            spans = engine._vad_splitter.spans(wav.reshape(-1))
        finally:
            engine.shutdown()
        assert spans is not None and len(spans) == 3
        kept = sum(b - a for a, b in spans) / SR
        assert kept < total - 3.0, "the two silences were not dropped"
