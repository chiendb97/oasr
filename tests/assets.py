# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry of the external assets the test suite gates on.

``pytest tests/`` used to report a fully green suite while silently skipping
every real-checkpoint test: the gating env vars were read ad hoc in a dozen
modules, each with its own default path and its own ``pytest.skip`` string, and
pytest's summary does not distinguish "passed" from "skipped because the
checkpoint was not on this box".  A green run was not evidence of anything.

This module makes that visible and, on demand, fatal:

* every gated asset is **declared once** here, with the marker file that proves
  it is really present (an HF snapshot of dangling LFS symlinks is not);
* :func:`require` is the single skip site, so the reason string is uniform and
  every skip is *counted*;
* the counts are printed at the end of every run — see ``conftest.py`` — so a
  local run says how much it did not check;
* ``--strict-assets`` turns those skips into failures, which is what CI runs.
  Assets the runner genuinely cannot have are named one by one with
  ``--allow-missing-asset NAME``, so the gap is in the workflow file rather
  than hidden in a skip.

Add an asset by adding an :class:`Asset` to :data:`ASSETS`; nothing else needs
to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pytest

# Asset kinds.  Only used for grouping in the report.
CHECKPOINT = "checkpoint"  # model weights
AUDIO = "audio"  # wav files to transcribe
REFERENCE = "reference"  # an upstream implementation or a captured oracle
GRAPH = "graph"  # a decoding graph (HLG / lang dir)


@dataclass(frozen=True)
class Asset:
    """One externally-supplied input the suite needs."""

    env: str
    kind: str
    what: str
    default: str
    #: Relative path under the root that must exist for the asset to count as
    #: present.  Empty means "the root itself is enough".
    marker: str = ""
    #: Optional extra predicate over the resolved root, for cases a single
    #: marker file cannot express.
    probe: Optional[Callable[[Path], bool]] = None
    #: One line on how to obtain it, shown when the asset is missing.
    how: str = ""

    def declared(self) -> str:
        """Path the suite *would* use — env var if set, else the default.

        Does not consult :data:`_DERIVED_DEFAULTS`; use the module-level
        :func:`declared` for that.
        """
        return os.environ.get(self.env) or self.default


def _has_nonempty_pt(root: Path) -> bool:
    """An HF snapshot can be present as dangling LFS symlinks with no payload."""
    return root.is_dir() and any(p.is_file() and p.stat().st_size > 0 for p in root.rglob("*.pt"))


def _has_wavs(root: Path) -> bool:
    return root.is_dir() and any(root.glob("*.wav"))


_ASSET_LIST: List[Asset] = [
    Asset(
        env="CKPT_DIR",
        kind=CHECKPOINT,
        what="WeNet U2++ conformer checkpoint dir (train.yaml + final.pt)",
        default="/data01/kilm/users/chiendb/models/asr/am/20210610_u2pp_conformer_exp_librispeech",
        marker="final.pt",
        how="a WeNet release dir; also settable with --ckpt-dir",
    ),
    Asset(
        env="WAV_DIR",
        kind=AUDIO,
        what="directory of 16 kHz .wav files for engine/E2E tests",
        default="/data01/kilm/users/chiendb/data/asr/ljspeech-sr16k-dataset/wavs",
        probe=_has_wavs,
        how="any dir of .wav files; also settable with --wav-dir",
    ),
    Asset(
        env="AUDIO_PATH",
        kind=AUDIO,
        what="a single .wav file for the CPU decoder integration tests",
        default="",
        how="export AUDIO_PATH=$(ls $WAV_DIR/*.wav | head -1); also --audio-path",
    ),
    Asset(
        env="ZIPFORMER_CKPT",
        kind=CHECKPOINT,
        what="icefall Zipformer CTC release",
        default=(
            "/data01/kilm/users/chiendb/models/asr/"
            "icefall-asr-librispeech-zipformer-large-cr-ctc-20241018"
        ),
        probe=_has_nonempty_pt,
        how="huggingface.co/k2-fsa icefall-asr-librispeech-zipformer-*",
    ),
    Asset(
        env="WHISPER_CKPT",
        kind=CHECKPOINT,
        what="HF-format Whisper checkpoint",
        default="/data01/kilm/users/chiendb/models/asr/whisper-tiny",
        marker="model.safetensors",
        how="huggingface.co/openai/whisper-tiny",
    ),
    Asset(
        env="OASR_PARAFORMER_CKPT",
        kind=CHECKPOINT,
        what="FunASR Paraformer checkpoint",
        default="/data01/kilm/users/chiendb/models/asr/paraformer-zh",
        marker="model.pt",
        how="modelscope / huggingface funasr paraformer-zh",
    ),
    Asset(
        env="SPEECH_LLM_TINY",
        kind=CHECKPOINT,
        what="tiny random Qwen2-Audio fixture (fp32 parity oracle)",
        default="/data01/kilm/users/chiendb/models/asr/qwen2-audio-tiny-random",
        marker="model.safetensors",
        how="generated fixture — see tests/test_speech_llm.py",
    ),
    Asset(
        env="SPEECH_LLM_CKPT",
        kind=CHECKPOINT,
        what="real Qwen2-Audio-7B snapshot",
        default="/data01/kilm/users/chiendb/models/asr/qwen2-audio-7b-instruct",
        marker="model.safetensors.index.json",
        how="huggingface.co/Qwen/Qwen2-Audio-7B-Instruct",
    ),
    Asset(
        env="LANG_DIR",
        kind=GRAPH,
        what="prebuilt lang dir (HLG.pt + words.txt) for the CPU WFST decoder",
        default="",
        how="k2 lang_bpe dir; also settable with --lang-dir",
    ),
    Asset(
        env="OASR_TEST_FST",
        kind=GRAPH,
        what="decoding graph for the GPU WFST decoder smoke tests",
        default=(
            "/data01/kilm/users/chiendb/models/asr/lm/"
            "20210610_u2pp_conformer_exp_librispeech/lang_bpe/HLG.pt"
        ),
        how="a k2 HLG.pt, or a prebuilt .img",
    ),
    Asset(
        env="WENET_REF_DIR",
        kind=REFERENCE,
        what="upstream WeNet v2.0.1 decoder sources (transformer decoder oracle)",
        default="/tmp/wenet_ref",
        marker="transformer/decoder.py",
        how="see the docstring at the top of tests/test_transformer_decoder.py",
    ),
    Asset(
        env="ICEFALL_ZIPFORMER_DIR",
        kind=REFERENCE,
        what="upstream icefall Zipformer sources (encoder parity oracle)",
        default="/tmp/icefall_ref",
        marker="zipformer.py",
        how="copy zipformer.py + scaling.py + subsampling.py from icefall",
    ),
    Asset(
        env="SPEECH_LLM_TINY_REF",
        kind=REFERENCE,
        what="captured HF greedy output for the tiny Qwen2-Audio fixture",
        # Lives inside the fixture dir; the env var only exists to override it.
        default="",
        marker="ref.pt",
        how="generated alongside the tiny fixture",
    ),
    Asset(
        env="OASR_PARAFORMER_REF",
        kind=REFERENCE,
        what="captured FunASR output for the Paraformer checkpoint",
        default="",
        marker="ref.pt",
        how="generated alongside the paraformer checkpoint",
    ),
]

ASSETS: Dict[str, Asset] = {a.env: a for a in _ASSET_LIST}

# ``SPEECH_LLM_TINY_REF`` / ``OASR_PARAFORMER_REF`` default to a subdirectory of
# their checkpoint, which is only known once that checkpoint resolves.
_DERIVED_DEFAULTS = {
    "SPEECH_LLM_TINY_REF": ("SPEECH_LLM_TINY", "oasr_ref"),
    "OASR_PARAFORMER_REF": ("OASR_PARAFORMER_CKPT", "oasr_ref"),
}


@dataclass
class _State:
    """Session state: how each asset was used, and whether skips are fatal."""

    strict: bool = False
    allowed_missing: frozenset = frozenset()
    #: env name -> number of tests skipped/failed for want of it.
    gated: Dict[str, int] = field(default_factory=dict)
    #: env names that at least one test actually asked for.
    requested: set = field(default_factory=set)


STATE = _State()


def configure(strict: bool, allow_missing: Iterable[str]) -> None:
    """Called once from ``conftest.pytest_configure``."""
    STATE.strict = strict
    STATE.allowed_missing = frozenset(allow_missing)
    unknown = STATE.allowed_missing - set(ASSETS)
    if unknown:
        raise pytest.UsageError(
            f"--allow-missing-asset names unknown assets: {sorted(unknown)}; "
            f"known: {sorted(ASSETS)}"
        )


def _declared(name: str) -> str:
    asset = ASSETS[name]
    explicit = os.environ.get(asset.env)
    if explicit:
        return explicit
    if name in _DERIVED_DEFAULTS:
        parent, sub = _DERIVED_DEFAULTS[name]
        return str(Path(ASSETS[parent].declared()) / sub)
    return asset.default


def declared(name: str) -> str:
    """Path the suite would use for *name*, present or not.

    Use this for building an error message or a nested path; use
    :func:`resolve` / :func:`require` to decide whether to run.
    """
    return _declared(name)


def resolve(name: str) -> Optional[str]:
    """Resolved path if the asset is really present, else ``None``."""
    asset = ASSETS[name]
    raw = _declared(name)
    if not raw:  # e.g. AUDIO_PATH with no env var and no default
        return None
    root = Path(raw)
    if asset.marker:
        if not (root / asset.marker).exists():
            return None
    elif not root.exists():
        return None
    if asset.probe is not None and not asset.probe(root):
        return None
    return str(root)


def present(name: str) -> bool:
    return resolve(name) is not None


def require(*names: str) -> str | Tuple[str, ...]:
    """Return the resolved path(s), or skip — or, under ``--strict-assets``, fail.

    This is the *only* place the suite skips for a missing asset, which is what
    makes the end-of-run report and the strict gate complete.
    """
    if not names:
        raise ValueError("require() needs at least one asset name")
    paths = []
    for name in names:
        if name not in ASSETS:
            raise KeyError(f"unknown asset {name!r}; declare it in tests/assets.py")
        STATE.requested.add(name)
        path = resolve(name)
        if path is None:
            _gate(name)
        paths.append(path)
    return paths[0] if len(paths) == 1 else tuple(paths)


def require_wavs(n: int = 1, env: str = "WAV_DIR") -> List[str]:
    """Return the first *n* ``.wav`` paths under an audio asset.

    Gates twice through the same mechanism: on the directory being present, and
    on it holding enough files.  A directory with too few clips is the same
    class of problem as a missing one — the test does not run — so
    ``--strict-assets`` should fail on it rather than let the count slide.
    """
    root = require(env)
    assert isinstance(root, str)
    wavs = sorted(Path(root).glob("*.wav"))
    if len(wavs) < n:
        STATE.gated[env] = STATE.gated.get(env, 0) + 1
        detail = f"need {n} .wav file(s) under {root}, found {len(wavs)} (set ${env})"
        if STATE.strict and env not in STATE.allowed_missing:
            pytest.fail(f"--strict-assets: {detail}", pytrace=False)
        pytest.skip(detail)
    return [str(p) for p in wavs[:n]]


def _gate(name: str) -> None:
    """Skip or fail for a missing asset, and record it either way."""
    asset = ASSETS[name]
    STATE.gated[name] = STATE.gated.get(name, 0) + 1
    where = _declared(name) or "<unset>"
    detail = f"{asset.what} not found at {where} (set ${asset.env})"
    if asset.how:
        detail += f" — {asset.how}"
    if STATE.strict and name not in STATE.allowed_missing:
        pytest.fail(
            f"--strict-assets: {detail}.\n"
            f"A skipped real-checkpoint test is the failure mode this flag exists "
            f"to catch. Provide the asset, or allow it explicitly with "
            f"--allow-missing-asset {asset.env}.",
            pytrace=False,
        )
    pytest.skip(detail)


def report_lines() -> List[str]:
    """Human-readable status of every declared asset, for the run report."""
    lines = []
    for name in ASSETS:
        path = resolve(name)
        gated = STATE.gated.get(name, 0)
        if path is not None:
            status = "ok"
            note = path
        else:
            status = "MISSING"
            note = _declared(name) or "<unset>"
            if gated:
                verb = "failed" if STATE.strict and name not in STATE.allowed_missing else "skipped"
                note += f"  [{gated} test(s) {verb}]"
            if name in STATE.allowed_missing:
                note += "  [allowed]"
        lines.append(f"  {name:<24} {status:<8} {note}")
    return lines
