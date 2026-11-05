# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The asset gate for what a *present* asset holds (audit L12).

A ``WAV_DIR`` with too few clips, a checkpoint without the file a test reads, or
an audio file too short to chunk used to be met by a bare ``pytest.skip``. Such
a skip was not counted, did not fail under ``--strict-assets``, and left the
asset reported ``ok``. ``assets.require_contents`` routes those through the gate.

These tests drive the gate directly. Each works on a fresh ``assets.STATE``, so
the running session's own report is untouched.
"""

from __future__ import annotations

import assets
import pytest


def _outcome(call):
    """``("failed" | "skipped" | "passed", message)`` of a gate call.

    Captured rather than asserted with ``pytest.raises``. A skip raised inside
    ``pytest.raises(pytest.fail.Exception)`` escapes it and the test *skips*, so
    a gate that skipped where it should fail would have looked like a skipped
    test here, which is the very failure mode under test.
    """
    try:
        call()
    except pytest.fail.Exception as exc:
        return "failed", str(exc)
    except pytest.skip.Exception as exc:
        return "skipped", str(exc)
    return "passed", ""


@pytest.fixture
def state(monkeypatch):
    """A fresh gate state, restored after the test."""
    fresh = assets._State()
    monkeypatch.setattr(assets, "STATE", fresh)
    return fresh


@pytest.fixture
def one_wav(tmp_path, monkeypatch):
    """A ``WAV_DIR`` holding exactly one clip."""
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("WAV_DIR", str(tmp_path))
    return tmp_path


class TestRequireContents:
    def test_a_satisfied_check_passes_and_is_recorded_as_used(self, state):
        assets.require_contents("WAV_DIR", True, "unused")
        assert "WAV_DIR" in state.requested
        assert state.gated == {} and state.short == {}

    def test_an_unmet_check_skips_and_is_counted(self, state):
        kind, msg = _outcome(lambda: assets.require_contents("WAV_DIR", False, "need 4"))
        assert kind == "skipped" and "need 4" in msg
        assert state.gated == {"WAV_DIR": 1}
        assert state.short == {"WAV_DIR": 1}

    def test_strict_mode_turns_it_into_a_failure(self, state):
        state.strict = True
        kind, msg = _outcome(lambda: assets.require_contents("WAV_DIR", False, "need 4"))
        assert kind == "failed" and msg.startswith("--strict-assets: ") and "need 4" in msg
        assert state.gated == {"WAV_DIR": 1}

    def test_an_allowed_asset_still_only_skips_under_strict(self, state):
        state.strict = True
        state.allowed_missing = frozenset({"WAV_DIR"})
        kind, _ = _outcome(lambda: assets.require_contents("WAV_DIR", False, "need 4"))
        assert kind == "skipped"

    def test_an_undeclared_asset_is_a_bug_in_the_test(self, state):
        with pytest.raises(KeyError, match="declare it in tests/assets.py"):
            assets.require_contents("NO_SUCH_ASSET", True, "")


class TestWavCount:
    def test_require_wavs_gates_on_the_count(self, state, one_wav):
        assert assets.require_wavs(1) == [str(one_wav / "a.wav")]
        kind, msg = _outcome(lambda: assets.require_wavs(2))
        assert kind == "skipped" and "need 2 .wav file" in msg
        assert state.short == {"WAV_DIR": 1}

    def test_the_audio_helpers_gate_instead_of_indexing_past_the_end(self, state, one_wav):
        """``waveforms(wav_dir, 2)`` over one clip was an ``IndexError``;
        ``speech_corpus`` skipped by hand."""
        from helpers.audio import speech_corpus, wav_path, wav_paths, waveforms

        state.strict = True
        assert wav_paths(str(one_wav)) == [str(one_wav / "a.wav")]
        assert wav_path(str(one_wav), 0) == str(one_wav / "a.wav")
        for call in (
            lambda: wav_path(str(one_wav), 1),
            lambda: waveforms(str(one_wav), 2),
            lambda: speech_corpus(str(one_wav), n=2),
        ):
            kind, msg = _outcome(call)
            assert kind == "failed" and "need 2 .wav file" in msg
        assert state.short == {"WAV_DIR": 3}


class TestReport:
    def test_a_present_asset_shows_the_tests_it_could_not_serve(self, state, one_wav):
        """Before, the row read ``ok`` and nothing else, so a run that skipped
        tests on a short directory looked complete."""
        assert _outcome(lambda: assets.require_wavs(3))[0] == "skipped"
        (row,) = [line for line in assets.report_lines() if line.strip().startswith("WAV_DIR")]
        assert " ok " in row and "[1 test(s) skipped: incomplete]" in row

    def test_strict_mode_reports_them_as_failed(self, state, one_wav):
        state.strict = True
        assert _outcome(lambda: assets.require_wavs(3))[0] == "failed"
        (row,) = [line for line in assets.report_lines() if line.strip().startswith("WAV_DIR")]
        assert "[1 test(s) failed: incomplete]" in row


class TestWenetCheckpointDir:
    def test_final_pt_alone_is_not_a_wenet_dir(self, state, tmp_path, monkeypatch):
        """``train.yaml`` is what the converter reads the architecture from. The
        Conformer parity test used to skip by hand when it was missing; the asset
        itself now requires it, so every ``CKPT_DIR`` test is gated alike."""
        (tmp_path / "final.pt").write_bytes(b"x")
        monkeypatch.setenv("CKPT_DIR", str(tmp_path))
        assert not assets.present("CKPT_DIR")
        (tmp_path / "train.yaml").write_text("encoder: conformer\n")
        assert assets.resolve("CKPT_DIR") == str(tmp_path)
