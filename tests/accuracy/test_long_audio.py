# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Long audio is decoded whole: no crash, and no transcript cut short in silence.

The accuracy gate measures 200 short utterances, so nothing in it reaches the
limits only long audio hits -- and each of those failed as a crash or as a
plausible, truncated transcript, never as a slightly worse number:

- the GPU CTC beam copied out at most 200 tokens, so every transcript past about
  a minute (500-unit vocabulary) or three (5000) lost its tail with
  ``finish_reason`` unset;
- WeNet's 5000-row position table failed U2++ outright above 200 s offline, and
  in streaming with bounded history an out-of-bounds gather took the CUDA
  context down with it;
- Nemotron refused anything past 400 s.

So this joins the same manifest's utterances, in order, into one recording past
the first two (``TARGET_S``), decodes it as one request, and requires a whole
transcript: no ``finish_reason`` but ``stop``, and an error rate a cut or an
empty output cannot reach.  The ceiling is loose on purpose --
``test_accuracy.py`` owns accuracy; this owns completeness.

Whisper and the speech-LLM take a fixed 30 s window, so long audio there is
``long_form``: the file is fanned out into windows and stitched back.  Those
windows were cut every 30 s wherever that fell, and Whisper answers a window
that starts or ends mid-word with a word or two (2026-11-09 report, D8): the
same plausible, short transcript, from a different cause.
"""

from __future__ import annotations

import json
from pathlib import Path

import assets
import pytest
import torch
from helpers import REPO_ROOT

from oasr.testing.accuracy import load_audio, load_manifest
from oasr.testing.wer import compute, normalizer

REFERENCE = json.loads((REPO_ROOT / "ci" / "wer-reference.json").read_text())
SPECS = {e["key"]: e for e in REFERENCE["entries"]}

TARGET_S = 300.0
GAP_S = 0.3
#: A cut transcript loses most of a 300 s reference and an empty one all of it; a
#: whole one stays near the short-utterance rate (3.5-5.8% measured, RTX 5090).
MAX_ERROR_PCT = 10.0

#: ``case -> (WER-reference entry, engine overrides)``.
CASES = {
    "conformer": ("conformer", {}),
    # Bounded history: the stream's position keeps growing past the table.
    "conformer_streaming_bounded": (
        "conformer",
        {"service_mode": "streaming", "chunk_size": 16, "num_left_chunks": 16},
    ),
    "zipformer": ("zipformer", {}),
    "transducer": ("transducer", {}),
    "zipformer_streaming": ("zipformer_streaming", {}),
    "nemotron": ("nemotron", {}),
    "nemotron_streaming": ("nemotron_streaming", {}),
    "whisper_long_form": ("whisper", {"long_form": True}),
    "llm_long_form": ("llm", {"long_form": True}),
}

#: Pauses that reproduce D8.  Cut every 30 s, a window of this recording that
#: started or ended mid-word came back with a few words: 15.8 % for whisper-tiny,
#: 57 of 608 words deleted, against 3.5 % cut in pauses.  The speech-LLM keeps the
#: default: given two sentences 2 s of digital silence apart, Qwen2-Audio often
#: transcribes one of them, wherever the window starts.
GAPS = {"whisper_long_form": 2.0}


def _joined(sample_rate: int, gap_s: float = GAP_S):
    """The manifest's utterances end to end until ``TARGET_S``: ``(wave, reference)``."""
    audio_root = Path(assets.resolve(REFERENCE["audio_asset"]))
    entries = load_manifest(REPO_ROOT / REFERENCE["manifest"], audio_root)
    waves, _ = load_audio(entries, sample_rate)
    gap = torch.zeros(int(gap_s * sample_rate))
    pieces, texts, total = [], [], 0
    for entry, wave in zip(entries, waves):
        if total >= TARGET_S * sample_rate:
            break
        pieces += [wave, gap]
        texts.append(entry.text)
        total += wave.numel() + gap.numel()
    assert total >= TARGET_S * sample_rate, "the manifest is shorter than TARGET_S"
    return torch.cat(pieces), " ".join(texts)


@pytest.mark.cuda
@pytest.mark.parametrize("case", list(CASES))
def test_long_audio_is_decoded_whole(case, device):
    key, overrides = CASES[case]
    spec = SPECS[key]
    assets.require(spec["asset"], REFERENCE["audio_asset"])

    from oasr.engine import ASREngine, EngineConfig

    cfg_kwargs = {
        "ckpt_dir": assets.resolve(spec["asset"]),
        "service_mode": spec.get("service_mode", "offline"),
        "dtype": getattr(torch, spec["dtype"]),
    }
    for field, name in (
        ("load_architecture", "architecture"),
        ("checkpoint_name", "checkpoint_name"),
        ("decode_method", "decode_method"),
        ("chunk_size", "chunk_size"),
    ):
        if spec.get(field):
            cfg_kwargs[name] = spec[field]
    if spec.get("decode_options"):
        cfg_kwargs["decode_options"] = dict(spec["decode_options"])
    cfg_kwargs.update(overrides)

    engine = ASREngine(EngineConfig(**cfg_kwargs))
    sample_rate = int(engine.sample_rate)
    try:
        wave, reference = _joined(sample_rate, GAPS.get(case, GAP_S))
        (out,) = engine.transcribe_outputs(
            [wave], streaming=cfg_kwargs["service_mode"] == "streaming"
        )
    finally:
        del engine
        torch.cuda.empty_cache()

    result = compute([reference], [out.text], normalizer=normalizer(spec["normalizer"]))
    seconds = wave.numel() / sample_rate
    # ``stop`` is an end-of-text token: a whole transcript from a family that
    # has one.  ``length`` is a cap that bound, ``error`` a failed window.
    assert out.finish_reason in (None, "stop"), (
        f"{case}: a {seconds:.0f}s request finished with {out.finish_reason!r} "
        f"({out.error_stage or 'no stage'}); {result.summary()}"
    )
    assert result.percent <= MAX_ERROR_PCT, (
        f"{case}: {result.percent:.2f}% over {seconds:.0f}s, {result.summary()} -- "
        "deletions mean a cut transcript"
    )
