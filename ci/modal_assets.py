#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Where every external test asset comes from, so Modal can fetch its own.

``tests/assets.py`` says *what* each asset is and how to tell a real one from a
dangling LFS symlink.  This file says *where it comes from* — a Hub repo id at a
pinned revision, a URL, a handful of raw files out of a git tree — so a rented
GPU can populate its own cache instead of being fed 29 GiB over the wire from
whichever laptop ran ``seed_assets`` last.

That upload was the thing worth deleting.  It is slow (a home connection, once
per asset refresh), it is not reproducible (whatever happened to be on that
box), and it means the CI that is supposed to check the code also depends on a
particular developer's filesystem.  Everything here is instead named by
``(source, revision)`` and pulled from inside the container, where the network
is a datacentre link and the result lands straight in the Volume it will be
read from next time.

Revisions are pinned on purpose.  An asset that moves under CI turns a WER
regression into an unattributable one, and ``ci/wer-reference.json`` records
rates against *these* weights.  Refresh a pin deliberately, in the same commit
as the re-recorded rate.

Four assets have no public source and are listed in :data:`NO_PUBLIC_SOURCE`:
they are uploaded once with ``modal run ci/modal_app.py::seed_assets``, or
allowed to be missing.  Naming them here rather than leaving them undescribed is
the point — a gap that is written down is a gap somebody can close.

This module deliberately imports nothing heavy at module scope: the local
entrypoint imports it to build the plan, and only the container needs torch or
``huggingface_hub``.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# The source table
# ---------------------------------------------------------------------------

HF = "hf"  # huggingface_hub.snapshot_download
FILES = "files"  # a list of (url, dest-relative-path)
LJSPEECH = "ljspeech"  # tarball -> resample -> flat wavs/


@dataclass(frozen=True)
class Source:
    """Where one asset comes from, and which slice of it is worth having."""

    env: str
    kind: str
    #: Hub repo id, or "" for the kinds that carry their own URLs.
    repo: str = ""
    #: Pinned Hub commit / git sha.  Never a branch name: see the module doc.
    revision: str = ""
    #: ``allow_patterns`` for a Hub snapshot.  The default (empty) means "the
    #: whole repo", which for several of these is 2-4x the bytes that any test
    #: reads — icefall ships its training logs and tensorboard events, Nemotron
    #: ships a 2.3 GiB ``.nemo`` beside the safetensors, and neither is loaded.
    allow: Tuple[str, ...] = ()
    #: (url, relative destination) pairs for ``FILES``.
    urls: Tuple[Tuple[str, str], ...] = ()
    #: Approximate download size, for the plan printout only.
    approx_mib: int = 0
    note: str = ""


#: The Zipformer CTC and transducer releases both ship an ``exp/epoch-*.pt``
#: (2.3 GiB) next to the ``exp/pretrained.pt`` the converter actually opens
#: (``ZipformerConverter.default_checkpoint_name``), plus decoding logs and
#: tensorboard events.  Ask for the three paths that are read.
_ICEFALL_ALLOW = (
    "exp/pretrained.pt",
    "data/lang_bpe_500/tokens.txt",
    "data/lang_bpe_500/bpe.model",
)

_WENET_REF_SHA = "bda6c86ff74d3ad257234a672d8a7bc4d0f32e81"  # v2.0.1
_ICEFALL_SHA = "3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1"
_SILERO_SHA = "867c2aa692646a1f1de3e94a15c9dd9f614c0acb"


def _raw(repo: str, sha: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"


_SOURCE_LIST: List[Source] = [
    Source(
        env="WAV_DIR",
        kind=LJSPEECH,
        # 22.05 kHz upstream; resampled to 16 kHz on arrival, because
        # ``ci/wer-reference.json`` was recorded against 16 kHz clips and the
        # frontend resamples anything else (which is a different measurement,
        # not a broken one -- see docs/features.md).
        repo="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        revision="LJSpeech-1.1",
        approx_mib=2600,
        note="the accuracy manifest names LJ001-* and LJ002-* only",
    ),
    Source(
        env="ZIPFORMER_CKPT",
        kind=HF,
        repo="Zengwei/icefall-asr-librispeech-zipformer-large-cr-ctc-20241018",
        revision="cfde898d2a0f61acb48db0d16337c039bc0eb3ca",
        allow=_ICEFALL_ALLOW,
        approx_mib=562,
    ),
    Source(
        env="TRANSDUCER_CKPT",
        kind=HF,
        repo="Zengwei/icefall-asr-librispeech-zipformer-2023-05-15",
        revision="d8bdbc60b27c21133fd4097222ad5f80bfac9f0d",
        allow=_ICEFALL_ALLOW,
        approx_mib=251,
        note="loads with architecture='transducer'; the dir sniffs as zipformer",
    ),
    Source(
        env="WHISPER_CKPT",
        kind=HF,
        repo="openai/whisper-tiny",
        revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        # No *.bin / flax / tf duplicates of the same weights.
        allow=("*.json", "*.txt", "model.safetensors"),
        approx_mib=148,
    ),
    Source(
        env="OASR_PARAFORMER_CKPT",
        kind=HF,
        repo="funasr/paraformer-zh",
        revision="d7811ee3ac581fbcfdeb37c98c6ba674028433dc",
        # ParaformerConverter reads config.yaml + model.pt + am.mvn +
        # tokens.json, and seg_dict when present.
        allow=(
            "config.yaml",
            "configuration.json",
            "model.pt",
            "am.mvn",
            "tokens.json",
            "seg_dict",
        ),
        approx_mib=848,
    ),
    Source(
        env="NEMOTRON_CKPT",
        kind=HF,
        repo="nvidia/nemotron-3.5-asr-streaming-0.6b",
        revision="1c8deaecc64b91f034d73e08dd8b64625eb3395d",
        # Skips the 2.3 GiB .nemo archive and ~800 KiB of README figures.
        allow=("*.json", "model.safetensors"),
        approx_mib=2434,
    ),
    Source(
        env="SPEECH_LLM_CKPT",
        kind=HF,
        repo="Qwen/Qwen2-Audio-7B-Instruct",
        revision="0a095220c30b7b31434169c3086508ef3ea5bf0a",
        allow=("*.json", "*.txt", "*.safetensors"),
        approx_mib=16000,
        note="the one big asset; --skip SPEECH_LLM_CKPT to leave it out",
    ),
    Source(
        env="SILERO_VAD_DIR",
        kind=FILES,
        urls=(
            (
                _raw("snakers4/silero-vad", _SILERO_SHA, "src/silero_vad/data/silero_vad.jit"),
                "silero_vad.jit",
            ),
        ),
        approx_mib=3,
    ),
    # The two upstream parity oracles.  Both were "allowed missing" on Modal
    # because uploading a source tree from a laptop is absurd; fetching seven
    # files over HTTPS is not, so they now run there too.
    Source(
        env="WENET_REF_DIR",
        kind=FILES,
        urls=tuple(
            (_raw("wenet-e2e/wenet", _WENET_REF_SHA, f"wenet/{p}"), p)
            for p in (
                "transformer/decoder.py",
                "transformer/decoder_layer.py",
                "transformer/attention.py",
                "transformer/embedding.py",
                "transformer/positionwise_feed_forward.py",
                "utils/mask.py",
                "utils/common.py",
            )
        ),
        approx_mib=1,
        note="the file list is the one in tests/test_transformer_decoder.py",
    ),
    Source(
        env="ICEFALL_ZIPFORMER_DIR",
        kind=FILES,
        urls=tuple(
            (_raw("k2-fsa/icefall", _ICEFALL_SHA, f"egs/librispeech/ASR/zipformer/{p}"), p)
            for p in ("zipformer.py", "scaling.py", "subsampling.py")
        ),
        approx_mib=1,
    ),
]

SOURCES: Dict[str, Source] = {s.env: s for s in _SOURCE_LIST}

#: Assets with no public source, and why.  ``seed_assets`` uploads these from a
#: box that has them — once, into the persistent Volume — or they stay missing
#: and ``--allow-missing-asset`` names them in the run report.
NO_PUBLIC_SOURCE: Dict[str, str] = {
    "CKPT_DIR": (
        "WeNet's librispeech u2pp_conformer release. wenet.org.cn/downloads now "
        "404s for it and no Hub mirror ships the raw final.pt (the one that "
        "exists is an exported TorchScript final.zip, which the converter "
        "cannot read). Upload it once, or mirror it to a Hub repo of your own "
        "and add it to SOURCES."
    ),
    "SPEECH_LLM_TINY": (
        "A generated fixture: real Qwen2-Audio tokenizer, random 2+2-layer "
        "weights. Reproducible only from the Phase 4 fixture script's seed, "
        "which is not in the tree, and its oasr_ref/ref.pt was captured against "
        "these exact weights. 94 MiB; upload it once."
    ),
    "LANG_DIR": "A locally built k2 lang dir (HLG.pt is 2.3 GiB).",
    "OASR_TEST_FST": "The same HLG.pt as LANG_DIR.",
}

#: Derived assets that come along with the checkpoint they live inside.
#: ``AUDIO_PATH`` is picked out of ``WAV_DIR`` at run time; the two ``*_REF``
#: oracles sit in ``<ckpt>/oasr_ref/`` and arrive with their parent (uploaded)
#: or not at all.
DERIVED = ("AUDIO_PATH", "SPEECH_LLM_TINY_REF", "OASR_PARAFORMER_REF")


def plan(names: Sequence[str] = ()) -> List[Source]:
    """The sources to fetch, in table order.  Empty *names* means all of them."""
    if not names:
        return list(_SOURCE_LIST)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(
            f"no public source for {unknown}; known: {sorted(SOURCES)}"
            + (
                f"\n(and these are known to have none: {sorted(NO_PUBLIC_SOURCE)})"
                if any(n in NO_PUBLIC_SOURCE for n in unknown)
                else ""
            )
        )
    return [SOURCES[n] for n in names]


# ---------------------------------------------------------------------------
# Fetching (runs in the container)
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[assets] {msg}", flush=True)


def _download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dest)
            return
        except Exception as exc:  # network flake, not a bug in the plan
            last = exc
            _log(f"  attempt {attempt}/{retries} failed for {url}: {exc}")
            time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url}") from last


def _fetch_hf(src: Source, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    _log(f"  snapshot_download({src.repo}@{src.revision[:8]}) -> {dest}")
    snapshot_download(
        repo_id=src.repo,
        revision=src.revision,
        allow_patterns=list(src.allow) or None,
        local_dir=str(dest),
        max_workers=8,
    )
    # snapshot_download leaves a .cache/huggingface bookkeeping tree next to the
    # payload.  On a Volume that is thousands of tiny files nothing reads again.
    shutil.rmtree(dest / ".cache", ignore_errors=True)


def _fetch_files(src: Source, dest: Path) -> None:
    for url, rel in src.urls:
        target = dest / rel
        _log(f"  {url} -> {rel}")
        _download(url, target)


#: The accuracy manifest, whose clips are not optional: ``test_accuracy.py``
#: measures exactly these and ``ci/wer-reference.json`` records the rate against
#: them.  Read rather than hardcoded, so re-building the manifest cannot leave
#: the fetcher quietly one clip short.
_MANIFEST = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "manifests" / "ljspeech_200.jsonl"
)


def _manifest_wavs() -> set:
    try:
        with open(_MANIFEST) as fh:
            return {json.loads(line)["audio"] for line in fh if line.strip()}
    except Exception as exc:  # a fetch is not the place to fail over a manifest
        _log(f"  could not read {_MANIFEST} ({exc}); falling back to the limit alone")
        return set()


def ljspeech_is_complete(wavs: Path, wav_limit: int) -> bool:
    """Every manifest clip present, and the quota met.

    ``tests/assets.py`` asks only "does this directory hold a .wav", which is the
    right question for a suite and the wrong one for a resumable download: a run
    that died a third of the way through leaves a directory that looks finished
    and an accuracy gate measuring 70 utterances against a 200-utterance
    reference.
    """
    if not wavs.is_dir():
        return False
    have = {p.name for p in wavs.glob("*.wav")}
    return _manifest_wavs() <= have and len(have) >= wav_limit


def _fetch_ljspeech(src: Source, dest: Path, wav_limit: int) -> None:
    """LJSpeech-1.1 -> ``wavs/LJxxx-nnnn.wav`` at 16 kHz.

    Two things about this archive drive the shape of the loop.

    **Its members are not in clip order.**  The first wav in the tar is
    ``LJ007-0048``, not ``LJ001-0001`` -- it was packed in filesystem order.  So
    "take the first 600" is not "take the 200 the manifest names", and a fetcher
    that assumed otherwise would hand ``test_accuracy.py`` a corpus with holes in
    it and a WER computed over whatever happened to survive.  The manifest's
    clips are therefore *required* by name, and ``wav_limit`` only governs how
    many extras come along for the engine tests (the largest ``require_wavs``
    asks for 12).  The whole archive is walked, because there is no other way to
    find a named member in a stream.

    **libsndfile seeks.**  It asks for the file length before reading a frame,
    and a member of a ``r|bz2`` stream cannot seek -- the symptom is
    ``'_Stream' object has no attribute 'seekable'`` followed by "No 'data' chunk
    marker", which reads like a corrupt archive and is not.  Each member is read
    into memory first; they are a few hundred KiB each.
    """
    import soundfile as sf
    import torch
    import torchaudio

    wavs = dest / "wavs" if dest.name != "wavs" else dest
    wavs.mkdir(parents=True, exist_ok=True)

    required = _manifest_wavs() - {p.name for p in wavs.glob("*.wav")}
    kept = len(list(wavs.glob("*.wav")))
    _log(f"  streaming {src.repo} ({len(required)} required clip(s) still missing)")
    with urllib.request.urlopen(src.repo, timeout=300) as response:
        # "r|bz2": stream mode -- members arrive once, in archive order, and
        # cannot be revisited.  Fine here: every member is inspected anyway.
        with tarfile.open(fileobj=response, mode="r|bz2") as tf:
            for member in tf:
                name = Path(member.name).name
                if not (member.isfile() and name.startswith("LJ") and name.endswith(".wav")):
                    continue
                out = wavs / name
                if out.is_file():
                    required.discard(name)
                    continue
                if name not in required and kept >= wav_limit:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                audio, sr = sf.read(io.BytesIO(fh.read()), dtype="float32", always_2d=False)
                x = torch.from_numpy(audio)
                if sr != 16000:
                    x = torchaudio.functional.resample(x, sr, 16000)
                sf.write(str(out), x.numpy(), 16000, subtype="PCM_16")
                required.discard(name)
                kept += 1
                if kept % 100 == 0:
                    _log(f"  {kept} clips ({len(required)} required still missing)")
                if not required and kept >= wav_limit:
                    break
    _log(f"  {kept} clips at 16 kHz in {wavs}; {len(required)} required clip(s) not found")


def _is_present(env: str, root: Path, relpath: str) -> bool:
    """Reuse the suite's own definition of "really there"."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    import assets as test_assets  # noqa: E402

    asset = test_assets.ASSETS[env]
    target = root / relpath
    if asset.marker:
        return (target / asset.marker).exists()
    if asset.probe is not None:
        return asset.probe(target)
    return target.exists()


def _complete(src: Source, root: Path, rel: str, wav_limit: int) -> bool:
    """Is this asset finished, not merely started?

    For most sources the suite's own marker check is the right question.  For
    LJSpeech it is not: see :func:`ljspeech_is_complete`.
    """
    if src.kind == LJSPEECH:
        dest = root / rel
        return ljspeech_is_complete(dest / "wavs" if dest.name != "wavs" else dest, wav_limit)
    return _is_present(src.env, root, rel)


def fetch(
    root: Path,
    names: Sequence[str] = (),
    force: bool = False,
    wav_limit: int = 600,
    relpaths: Optional[Dict[str, str]] = None,
    commit: Optional[Callable[[], None]] = None,
) -> List[str]:
    """Populate *root* with every requested asset.  Returns report lines.

    Idempotent and resumable.  An asset that is already complete costs one
    marker check, so re-running after adding a source downloads that source
    alone; ``force`` re-fetches regardless.  *commit* is called after each asset
    and after each failure — on Modal it flushes the Volume, which is what makes
    a run that dies on asset five keep assets one through four instead of
    starting over.  A failure is reported and the remaining assets still run:
    one unreachable host should not cost the other nine downloads.
    """
    relpaths = relpaths or {}
    lines: List[str] = []
    failures = 0
    for src in plan(names):
        rel = relpaths.get(src.env) or src.env.lower()
        dest = Path(root) / rel
        if not force and _complete(src, Path(root), rel, wav_limit):
            _log(f"{src.env}: already present at {dest}")
            lines.append(f"  {src.env:<24} cached    {dest}")
            continue
        _log(f"{src.env}: fetching (~{src.approx_mib} MiB)")
        started = time.time()
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if src.kind == HF:
                _fetch_hf(src, dest)
            elif src.kind == FILES:
                _fetch_files(src, dest)
            elif src.kind == LJSPEECH:
                _fetch_ljspeech(src, dest, wav_limit)
            else:  # pragma: no cover — the table is closed
                raise ValueError(f"unknown source kind {src.kind!r} for {src.env}")
        except Exception as exc:
            failures += 1
            _log(f"{src.env}: FAILED — {type(exc).__name__}: {exc}")
            lines.append(f"  {src.env:<24} FAILED    {type(exc).__name__}: {exc}")
            if commit is not None:
                commit()
            continue
        took = time.time() - started
        ok = _complete(src, Path(root), rel, wav_limit)
        if not ok:
            failures += 1
            _log(f"{src.env}: fetched but still incomplete — check `allow` / the marker")
        lines.append(f"  {src.env:<24} {'ok' if ok else 'INCOMPLETE':<9} {dest}  ({took:.0f}s)")
        if commit is not None:
            commit()
    if failures:
        lines.append(f"  {failures} asset(s) did not complete")
    return lines
