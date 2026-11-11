# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The tuning database: measured kernel selections as data, per architecture.

A *tuning file* holds every measured selection for one ``(arch family, kernel
family)`` pair -- ``sm120/gemm.json``, ``sm120/conv1d.json`` -- and lives in one
of two tiers:

* the **system** tier ships with the package (``oasr/tune/db/``) and changes
  only by review, like ``ci/wer-reference.json``;
* the **user** tier lives in the cache directory (``~/.cache/oasr/tune/v2``),
  is written by tuning runs on the machine that measured it, and wins over the
  system tier on conflict -- the MIOpen User-DB rule.

A file is keyed by the *static* signature of an operation (``gemm|op=gemm|
dt=half|N=512|K=256``).  The dynamic dimension -- M for a GEMM -- indexes the
entry's **regions**, ascending ``(m_hi, config_id)`` pairs looked up by rounding
**up**: ``M`` takes the first region whose ``m_hi >= M``, and a ``None`` bound
is the catch-all.  Rounding up is a validity contract, not a style: a config
measured at points ``<= m_hi`` is only ever served inside what it was measured
on (the FlashInfer bucket-direction defect, issue #5449, is what rounding down
does).  Entries whose dynamic shape has no natural total order use exact
``points`` instead.

This module knows nothing about any kernel family.  Each family module
(``oasr.jit.gemm``, ``oasr.jit.conv``, ``oasr.jit.scaled_mm``) owns the codec between a config's
parameters and its config object, and builds its lookup views from
:func:`tiers`.  That keeps the import graph acyclic: the JIT modules import
this one, never the reverse.

Validators
----------
*Hard* validators decide whether an entry means anything at all -- the schema,
the kernel family, the architecture family and its MMA lane.  A mismatch makes
the file unusable and it is skipped, loudly.

*Soft* validators decide whether it is *current* -- the kernel implementation
hash, the SKU, the nvcc and CUTLASS versions, and the library versions of each
runner (:mod:`oasr.tune.runners`; cuBLAS, say) -- that last one counted only
against the entries that select that runner.  A mismatch keeps the file in
service and marks it stale.  That is deliberately gentler than TensorRT or XLA,
which drop on any mismatch: their entries are opaque tactic ids that may no
longer exist, whereas an entry here is an explicit parameter set that the
resolver checks against the compiled set and the static feasibility predicates
every time it resolves.  A stale entry can therefore only be *slower* than a
fresh tune, never invalid -- and dropping it would trade a measured choice for
the untuned default.  Staleness is reported (:func:`tuning_report`) so it gets
re-measured rather than trusted forever.

The soft validators describe the *entries*, not the last writer.  A file is
stamped with the validators of the process that writes it, and an entry that
process did not measure keeps the validators it was measured under as its own
(:attr:`Entry.soft`, set by :func:`stamp_current`) -- re-stamping a file used to
vouch for every entry in it, measured under another kernel version or not.  An
entry that mixes measurements from two validator sets keeps the older one, so it
stays reported stale until it is rebuilt.

Writers
-------
Every write to a file another process may also write goes through
:func:`update_file` or :func:`save_merged`: under the file's lock, re-read what
is on disk, apply this writer's own changes to it, and write it back atomically.
A build reads its base minutes before it writes, and the JIT tier accumulates
points for a whole run, so the plain read-modify-write they used lost every
entry another process wrote in between.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import filelock

logger = logging.getLogger("oasr.tune")

__all__ = [
    "SCHEMA",
    "Entry",
    "TuningFile",
    "Tiers",
    "sig_key",
    "parse_sig_key",
    "region_lookup",
    "system_dir",
    "system_path",
    "user_root",
    "user_path",
    "load_file",
    "save_file",
    "save_merged",
    "update_file",
    "stamp_current",
    "tiers",
    "reload",
    "epoch",
    "register_reload_hook",
    "register_soft_validator",
    "current_soft_validators",
    "record_tier",
    "record_runner",
    "tier_counts",
    "runner_counts",
    "reset_tier_counts",
    "tuning_report",
    "hash_paths",
    "record_point",
    "flush_user",
    "regions_from_points",
    "new_file",
]

#: Schema of the files this module reads and writes.  A bump is a hard
#: invalidation of every file written under the previous value.
SCHEMA = 2

#: The kernel families that keep tuning files.  A file for any other family is
#: rejected on load: an unknown family means a reader older than the writer.
#: ``scaled_mm`` is the int8 W8A8 GEMM (``oasr.jit.scaled_mm``), whose regions
#: may name the ``"heuristic"`` sentinel -- "measured; the shape heuristic's
#: own pick is the best here".  ``fmha`` is the choice between the attention
#: lanes and their tiles (``oasr.jit.attention_select``), whose sentinel is
#: ``"auto"`` -- "measured; today's routing is the best here".
KERNEL_FAMILIES = ("gemm", "conv1d", "scaled_mm", "scaled_mm_fp8", "fmha")

_USER_ENV = "OASR_TUNE_USER_DB"
_OFF_VALUES = ("0", "off", "none", "false", "no")


# =============================================================================
# Signature keys
# =============================================================================


def sig_key(family: str, **fields: Any) -> str:
    """The text key of a static signature, fields in the order given.

    ``sig_key("gemm", op="gemm", dt="half", N=512, K=256)`` is
    ``"gemm|op=gemm|dt=half|N=512|K=256"``.  Stable, greppable, and diffable in
    review -- which is the property a key in a reviewed file needs most.
    """
    parts = [family]
    for k, v in fields.items():
        if "|" in str(v) or "=" in str(v):
            raise ValueError(f"signature field {k}={v!r} contains a reserved character")
        parts.append(f"{k}={v}")
    return "|".join(parts)


def parse_sig_key(key: str) -> Tuple[str, Dict[str, str]]:
    """Inverse of :func:`sig_key`; values come back as strings."""
    family, *rest = key.split("|")
    fields = {}
    for part in rest:
        k, _, v = part.partition("=")
        fields[k] = v
    return family, fields


# =============================================================================
# Data model
# =============================================================================


def region_lookup(regions: Sequence[Tuple[Optional[int], str]], m: int) -> Optional[str]:
    """The config id for dynamic size ``m``: the first region with ``m_hi >= m``.

    ``None`` as ``m_hi`` is the catch-all.  Returns ``None`` when ``m`` lies
    above every bounded region and there is no catch-all.
    """
    for m_hi, cfg in regions:
        if m_hi is None or m <= m_hi:
            return cfg
    return None


@dataclass
class Entry:
    """Every measured selection for one static signature."""

    #: Ascending ``(m_hi | None, config_id)``; see :func:`region_lookup`.
    regions: List[Tuple[Optional[int], str]] = field(default_factory=list)
    #: Exact dynamic-shape matches, ``"1x3000" -> config_id``, for signatures
    #: whose dynamic shape is not a single ordered size.
    points: Dict[str, str] = field(default_factory=dict)
    #: ``config_id -> [lo, hi]``: the dynamic range a config was validated for,
    #: when its validity depends on it.  Absent means valid for any size.
    valid_m: Dict[str, List[int]] = field(default_factory=dict)
    #: Measured timings, ``{"M=416": {"cfg": [median_ms, sigma_ms, n]}}``.
    evidence: Dict[str, Any] = field(default_factory=dict)
    #: Where the entry came from: ``"converted"``, ``"aot"``, ``"jit"``, ``"prior"``.
    source: str = "aot"
    #: Free-text provenance carried with the entry (why a boundary is where it is).
    notes: List[str] = field(default_factory=list)
    #: The soft validators this entry was measured under, when they are not the
    #: file's own: ``None`` means the file's.  Set by :func:`stamp_current` when
    #: a writer re-stamps a file whose entries it did not measure.
    soft: Optional[Dict[str, Any]] = None

    def lookup(self, m: int) -> Optional[str]:
        return region_lookup(self.regions, m)

    def valid(self, cfg: str, m: int) -> bool:
        env = self.valid_m.get(cfg)
        return env is None or env[0] <= m <= env[1]

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.regions:
            out["regions"] = [[hi, cfg] for hi, cfg in self.regions]
        if self.points:
            out["points"] = dict(self.points)
        if self.valid_m:
            out["valid_m"] = dict(self.valid_m)
        out["source"] = self.source
        if self.notes:
            out["notes"] = list(self.notes)
        if self.evidence:
            out["evidence"] = self.evidence
        if self.soft is not None:
            out["soft"] = dict(self.soft)
        return out

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Entry":
        regions = []
        for hi, cfg in d.get("regions", []):
            regions.append((None if hi is None else int(hi), str(cfg)))
        bounded = [hi for hi, _ in regions if hi is not None]
        if bounded != sorted(bounded) or any(hi is None for hi, _ in regions[:-1]):
            raise ValueError(f"regions must ascend and end with at most one catch-all: {regions}")
        return cls(
            regions=regions,
            points={str(k): str(v) for k, v in d.get("points", {}).items()},
            valid_m={str(k): [int(v[0]), int(v[1])] for k, v in d.get("valid_m", {}).items()},
            evidence=d.get("evidence", {}),
            source=str(d.get("source", "aot")),
            notes=[str(n) for n in d.get("notes", [])],
            soft=dict(d["soft"]) if isinstance(d.get("soft"), dict) else None,
        )


@dataclass
class TuningFile:
    """One ``(arch family, kernel family)`` tuning file."""

    family: str
    arch: Dict[str, Any]
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    entries: Dict[str, Entry] = field(default_factory=dict)
    validators: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {"hard": {}, "soft": {}})
    provenance: Dict[str, Any] = field(default_factory=dict)
    coverage_basis: List[str] = field(default_factory=list)
    #: The device model the file was measured with (``oasr.tune.cost_model``):
    #: SM count, bandwidth, throughput, launch cost, and per-config calibration.
    #: What the resolver's ``model`` tier ranks with for a signature no entry covers.
    model: Dict[str, Any] = field(default_factory=dict)
    schema: int = SCHEMA
    #: Where it was loaded from (not serialised).
    path: Optional[Path] = None
    #: Soft-validator mismatches, computed on first use (not serialised).
    _stale: Optional[List[str]] = field(default=None, repr=False)

    @property
    def arch_family(self) -> int:
        return int(self.arch["family"])

    @property
    def stale(self) -> List[str]:
        """Soft-validator mismatches against this process, as readable reasons.

        Lazy on purpose: it hashes the family's kernel sources and queries the
        device, which an ``import oasr`` should not pay for.
        """
        if self._stale is None:
            self._stale = _stale_reasons(self)
        return self._stale

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "family": self.family,
            "arch": self.arch,
            "validators": self.validators,
            "provenance": self.provenance,
            "configs": {k: self.configs[k] for k in sorted(self.configs)},
            "coverage_basis": list(self.coverage_basis),
            **({"model": self.model} if self.model else {}),
            "entries": {k: self.entries[k].to_json() for k in sorted(self.entries)},
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any], path: Optional[Path] = None) -> "TuningFile":
        tf = cls(
            family=str(d["family"]),
            arch=dict(d["arch"]),
            configs={str(k): dict(v) for k, v in d.get("configs", {}).items()},
            entries={str(k): Entry.from_json(v) for k, v in d.get("entries", {}).items()},
            validators={
                "hard": dict(d.get("validators", {}).get("hard", {})),
                "soft": dict(d.get("validators", {}).get("soft", {})),
            },
            provenance=dict(d.get("provenance", {})),
            coverage_basis=[str(c) for c in d.get("coverage_basis", [])],
            model=dict(d.get("model", {})),
            schema=int(d.get("schema", 0)),
            path=path,
        )
        missing = sorted(
            {cfg for e in tf.entries.values() for _, cfg in e.regions}
            | {cfg for e in tf.entries.values() for cfg in e.points.values()}
            | set(tf.coverage_basis)
        )
        missing = [c for c in missing if c not in tf.configs]
        if missing:
            raise ValueError(f"entries reference undefined configs: {missing}")
        return tf

    def referenced_configs(self) -> List[str]:
        """Every config id an entry or the coverage basis names, sorted."""
        ids = set(self.coverage_basis)
        for e in self.entries.values():
            ids.update(cfg for _, cfg in e.regions)
            ids.update(e.points.values())
        return sorted(ids)


# =============================================================================
# Paths
# =============================================================================


def system_dir() -> Path:
    """The shipped tier's root, ``oasr/tune/db``."""
    return Path(__file__).resolve().parent / "db"


def system_path(sm: int, family: str) -> Path:
    return system_dir() / f"sm{int(sm)}" / f"{family}.json"


def user_root() -> Optional[Path]:
    """The user tier's root, or ``None`` when it is disabled.

    ``OASR_TUNE_USER_DB`` overrides it: a path moves it, and ``off`` disables it
    (the test suite does, so a developer's own tuning never changes what a test
    observes).
    """
    raw = os.environ.get(_USER_ENV)
    if raw is not None:
        if raw.strip().lower() in _OFF_VALUES or not raw.strip():
            return None
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "oasr" / "tune" / f"v{SCHEMA}"


def user_path(sm: int, family: str) -> Optional[Path]:
    root = user_root()
    return None if root is None else root / f"sm{int(sm)}" / f"{family}.json"


# =============================================================================
# Load / save
# =============================================================================

#: ``family -> callable returning the current soft validators for that family``.
#: Registered by the family module (it knows its own implementation hash).
_SOFT_VALIDATORS: Dict[str, Callable[[], Dict[str, Any]]] = {}


def register_soft_validator(family: str, fn: Callable[[], Dict[str, Any]]) -> None:
    """Register how *family* computes its current soft validators."""
    _SOFT_VALIDATORS[family] = fn


def current_soft_validators(family: str) -> Dict[str, Any]:
    """The soft validators a file for *family* written now would carry.

    ``"runners"`` holds the library versions of every runner of the family that
    can run here (:meth:`oasr.tune.runners.Runner.versions`).  A mismatch there
    makes stale only the entries that select that runner.
    """
    out = dict(_common_soft_validators())
    fn = _SOFT_VALIDATORS.get(family)
    if fn is not None:
        with contextlib.suppress(Exception):
            out.update(fn())
    with contextlib.suppress(Exception):
        from oasr.tune.runners import runner_versions

        versions = runner_versions(family)
        if versions:
            out["runners"] = versions
    return out


@functools.lru_cache(maxsize=1)
def _common_soft_validators() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with contextlib.suppress(Exception):
        from oasr.jit.cpp_ext import get_nvcc_build

        out["nvcc"] = get_nvcc_build()
    with contextlib.suppress(Exception):
        from oasr.jit import env

        out["cutlass"] = hash_bytes(v for _, v in env.cutlass_version_stamp())
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["sku"] = {
                "name": p.name,
                "sms": int(p.multi_processor_count),
                "l2": int(getattr(p, "L2_cache_size", 0) or 0),
            }
    return out


def _hard_problems(tf: TuningFile, family: str, arch_family: int) -> List[str]:
    problems = []
    if tf.schema != SCHEMA:
        problems.append(f"schema {tf.schema} != {SCHEMA}")
    if tf.family != family:
        problems.append(f"family {tf.family!r} != {family!r}")
    if tf.family not in KERNEL_FAMILIES:
        problems.append(f"unknown kernel family {tf.family!r}")
    if tf.arch_family != int(arch_family):
        problems.append(f"arch family {tf.arch_family} != {arch_family}")
    return problems


def _stale_reasons(tf: TuningFile) -> List[str]:
    """Soft-validator mismatches, as human-readable reasons.

    The file's own stamp covers the entries that carry none; an entry measured
    under other validators (:attr:`Entry.soft`) is reported against its own.
    """
    current = current_soft_validators(tf.family)
    stamped = [e for e in tf.entries.values() if e.soft is None]
    reasons = _soft_mismatches(tf.validators.get("soft", {}), current, _runners_used(tf, stamped))
    own: Dict[str, Tuple[Dict[str, Any], List[Entry]]] = {}
    for entry in tf.entries.values():
        if entry.soft is not None:
            tag = json.dumps(entry.soft, sort_keys=True)
            own.setdefault(tag, (entry.soft, []))[1].append(entry)
    for saved, group in own.values():
        n = len(group)
        for reason in _soft_mismatches(saved, current, _runners_used(tf, group)):
            reasons.append(f"{n} entr{'y' if n == 1 else 'ies'} measured under {reason}")
    return reasons


def _runners_used(tf: TuningFile, entries: Iterable[Entry]) -> Dict[str, int]:
    """``{runner name: entries selecting it}`` over *entries*, by their configs' kinds."""
    from oasr.tune.runners import runner_for_kind

    used: Counter = Counter()
    for entry in entries:
        names = set()
        for cid in [c for _, c in entry.regions] + list(entry.points.values()):
            kind = tf.configs.get(cid, {}).get("kind")
            runner = None if kind is None else runner_for_kind(tf.family, str(kind))
            if runner is not None:
                names.add(runner.name)
        used.update(names)
    return dict(used)


def _soft_mismatches(
    saved_validators: Dict[str, Any],
    current: Dict[str, Any],
    used: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Readable mismatches of *saved_validators* against *current*.

    ``"runners"`` is compared runner by runner, and only for the runners in
    *used* (``None``: every runner): a library nobody's entry selects cannot
    make an entry stale.
    """
    reasons = []
    for key, saved in sorted(saved_validators.items()):
        now = current.get(key)
        if now is None or saved in (None, "*"):
            continue
        if key == "runners" and isinstance(saved, dict) and isinstance(now, dict):
            for name, was in sorted(saved.items()):
                is_now = now.get(name)
                if is_now is None or was in (None, "*") or (used is not None and name not in used):
                    continue
                if was != is_now:
                    reasons.append(f"runner {name}: saved {was!r}, now {is_now!r}")
            continue
        if key == "sku" and isinstance(saved, dict) and isinstance(now, dict):
            # The identity that moves a crossover is (SM count, L2); a name
            # alone differs across cloud SKUs of the same silicon.
            if (saved.get("sms"), saved.get("l2")) != (now.get("sms"), now.get("l2")):
                reasons.append(
                    f"sku: measured on {saved.get('name')} ({saved.get('sms')} SMs, "
                    f"L2 {saved.get('l2')}), running on {now.get('name')} "
                    f"({now.get('sms')} SMs, L2 {now.get('l2')})"
                )
            continue
        if saved != now:
            reasons.append(f"{key}: saved {saved!r}, now {now!r}")
    return reasons


def load_file(path: Path, family: str, arch_family: int) -> Optional[TuningFile]:
    """Load one tuning file, or ``None`` (logged once) when it cannot be used.

    Hard-validator failures and malformed files return ``None``.  Soft-validator
    mismatches do not stop a load; :attr:`TuningFile.stale` reports them.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            tf = TuningFile.from_json(json.load(f), path=path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("[tune.db] ignoring unreadable tuning file %s: %s", path, exc)
        return None
    problems = _hard_problems(tf, family, arch_family)
    if problems:
        logger.warning(
            "[tune.db] ignoring %s: %s (a file written for another schema, family or "
            "architecture has no meaning here)",
            path,
            "; ".join(problems),
        )
        return None
    return tf


def save_file(tf: TuningFile, path: Path) -> None:
    """Write *tf* atomically (temp file + ``os.replace``), creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tune_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(tf.to_json(), f, indent=1, sort_keys=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def stamp_current(tf: TuningFile, validators: Optional[Dict[str, Any]] = None) -> None:
    """Stamp *tf* with this process's soft validators, without vouching for its entries.

    Every entry that carries no validators of its own was measured under the
    file's old stamp, so it is given that stamp before the file's changes: the
    entries this writer then measures are the ones the new stamp describes.
    An entry whose own validators equal the new stamp drops them again.
    """
    now = dict(current_soft_validators(tf.family) if validators is None else validators)
    old = dict(tf.validators.get("soft", {}))
    if old != now:
        for entry in tf.entries.values():
            if entry.soft is None:
                entry.soft = old
    for entry in tf.entries.values():
        if entry.soft == now:
            entry.soft = None
    tf.validators["soft"] = now
    tf._stale = None


@contextlib.contextmanager
def _file_lock(path: Path, timeout: float = 60.0):
    """The lock every writer of *path* takes around its read-merge-write.

    Held for the merge and the write only, never for a build, so a waiter waits
    milliseconds.  ``filelock.Timeout`` is an ``OSError``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(f"{path}.lock", timeout=timeout):
        yield


def update_file(
    path: Path,
    family: str,
    arch_family: int,
    lane: str,
    apply: Callable[[TuningFile], None],
) -> TuningFile:
    """Apply *apply* to the file at *path* as it is now, under its lock, and write it.

    The file is re-read under the lock, so whatever another writer put there
    since this one last looked survives: *apply* adds this writer's changes to
    it rather than replacing it.  It is stamped (:func:`stamp_current`) before
    *apply* runs, so what *apply* adds is what the new stamp describes.
    """
    path = Path(path)
    with _file_lock(path):
        tf = load_file(path, family, arch_family) if path.is_file() else None
        if tf is None:
            tf = new_file(family, arch_family, lane)
        stamp_current(tf)
        apply(tf)
        save_file(tf, path)
    return tf


def _measured_content(entry_json: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in entry_json.items() if k != "soft"}


def save_merged(
    tf: TuningFile, path: Path, *, since: Optional[Dict[str, Any]] = None
) -> TuningFile:
    """Write the entries *tf* changed since *since* into the file at *path*.

    For a build: *since* is its base's ``to_json()`` taken when it was loaded,
    before the build changed it, and *tf* is the build's result.  An entry the
    build added or re-measured replaces the one on disk; every other entry on
    disk -- including one another process wrote while this build ran -- is kept
    as the disk has it, with the validators it was measured under.  An entry the
    build dropped is dropped from disk too, unless it changed there meanwhile.
    Configs and the coverage basis are unions, and the build's device model
    wins.  Under the file's lock, like :func:`update_file`.
    """
    path = Path(path)
    before = (since or {}).get("entries", {})
    changed = [
        key
        for key, entry in tf.entries.items()
        if key not in before or _measured_content(entry.to_json()) != _measured_content(before[key])
    ]
    dropped = [key for key in before if key not in tf.entries]
    with _file_lock(path):
        disk = load_file(path, tf.family, tf.arch_family) if path.is_file() else None
        if disk is None:
            save_file(tf, path)
            return tf
        stamp_current(disk, tf.validators.get("soft", {}))
        for key in changed:
            disk.entries[key] = tf.entries[key]
        for key in dropped:
            # Only the entry the build started from: one another process
            # re-measured meanwhile is newer than this build's decision.
            on_disk = disk.entries.get(key)
            if on_disk is not None and _measured_content(on_disk.to_json()) == (
                _measured_content(before[key])
            ):
                del disk.entries[key]
        disk.configs.update(tf.configs)
        disk.coverage_basis = list(dict.fromkeys(disk.coverage_basis + tf.coverage_basis))
        if tf.model:
            disk.model = tf.model
        disk.provenance.update(tf.provenance)
        save_file(disk, path)
    return disk


def new_file(family: str, arch_family: int, lane: str) -> TuningFile:
    """An empty file carrying this machine's validators."""
    return TuningFile(
        family=family,
        arch={"family": int(arch_family), "lane": lane},
        validators={
            "hard": {
                "schema": SCHEMA,
                "family": family,
                "arch_family": int(arch_family),
                "lane": lane,
            },
            "soft": current_soft_validators(family),
        },
        provenance={"created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    )


# =============================================================================
# Tiers and the snapshot epoch
# =============================================================================


@dataclass(frozen=True)
class Tiers:
    """The loaded tiers for one ``(kernel family, arch family)``, highest first."""

    family: str
    arch_family: int
    user: Optional[TuningFile]
    system: Optional[TuningFile]
    #: Content identity of this snapshot -- what a captured graph was built under.
    snapshot_id: str

    def ordered(self) -> List[Tuple[str, TuningFile]]:
        out = []
        if self.user is not None:
            out.append(("user", self.user))
        if self.system is not None:
            out.append(("system", self.system))
        return out


_lock = threading.RLock()
_TIERS: Dict[Tuple[str, int], Tiers] = {}
_EPOCH = 0
_RELOAD_HOOKS: List[Callable[[], None]] = []


def tiers(family: str, arch_family: int) -> Tiers:
    """The tiers for ``(family, arch_family)``, loaded once per epoch.

    Immutable for the epoch: every selection made between two :func:`reload`
    calls reads the same bytes (design invariant I2), which is what keeps a
    captured graph and an eager call on the same kernel.
    """
    key = (family, int(arch_family))
    t = _TIERS.get(key)
    if t is not None:
        return t
    with _lock:
        t = _TIERS.get(key)
        if t is not None:
            return t
        system = load_file(system_path(arch_family, family), family, arch_family)
        upath = user_path(arch_family, family)
        user = load_file(upath, family, arch_family) if upath is not None else None
        h = hashlib.sha256()
        for tf in (user, system):
            if tf is not None and tf.path is not None:
                with contextlib.suppress(OSError):
                    h.update(tf.path.read_bytes())
        t = Tiers(family, int(arch_family), user, system, h.hexdigest()[:16])
        _TIERS[key] = t
        return t


def reload() -> int:
    """Start a new epoch: drop every loaded tier and notify the family views.

    The only point at which a selection can change inside a process.  Callers
    that hold captured graphs must re-capture after this (``ASREngine``'s
    ``reload_tuning`` does).  Returns the new epoch number.
    """
    global _EPOCH
    with _lock:
        _TIERS.clear()
        _EPOCH += 1
        hooks = list(_RELOAD_HOOKS)
    _common_soft_validators.cache_clear()
    for hook in hooks:
        hook()
    return _EPOCH


def epoch() -> int:
    return _EPOCH


def register_reload_hook(fn: Callable[[], None]) -> None:
    """Call *fn* whenever :func:`reload` starts a new epoch."""
    with _lock:
        _RELOAD_HOOKS.append(fn)


# =============================================================================
# The user-tier writer (JIT tuning results)
# =============================================================================

#: ``(family, arch family) -> (lane, the points recorded since the last flush)``,
#: flushed by :func:`flush_user`.  Only this process's own points: the file they
#: go into is read at the flush, under its lock, so the points another process
#: flushed in between are kept.  Writes never touch the snapshot a running
#: selection reads: they take effect at the next :func:`reload`.
_PENDING: Dict[Tuple[str, int], Tuple[str, List[Tuple[Any, ...]]]] = {}


def regions_from_points(points: Dict[str, str]) -> List[Tuple[Optional[int], str]]:
    """Ascending regions from measured ``{str(m): config_id}`` points.

    Each point covers ``(previous point, m]``; runs of one config merge.  There
    is **no catch-all**: a size above the largest measured point was never
    measured, so it falls through to the next tier instead of borrowing a
    small-M winner (the shipped tables have catch-alls because a sweep measured
    the top bucket on purpose).
    """
    regions: List[Tuple[Optional[int], str]] = []
    for m, cid in sorted((int(k), v) for k, v in points.items()):
        if regions and regions[-1][1] == cid:
            regions[-1] = (m, cid)
        else:
            regions.append((m, cid))
    return regions


def record_point(
    family: str,
    arch_family: int,
    lane: str,
    sig: str,
    m: int,
    config_id: str,
    params: Dict[str, Any],
    timings: Optional[Dict[str, Any]] = None,
    source: str = "jit",
) -> bool:
    """Record one measured winner in the user tier (pending until :func:`flush_user`).

    Returns ``False`` when the user tier is disabled.
    """
    path = user_path(arch_family, family)
    if path is None:
        return False
    key = (family, int(arch_family))
    with _lock:
        _lane, points = _PENDING.setdefault(key, (lane, []))
        points.append((sig, int(m), config_id, dict(params), timings, source))
    return True


def _apply_points(tf: TuningFile, points: Sequence[Tuple[Any, ...]]) -> None:
    for sig, m, config_id, params, timings, source in points:
        tf.configs[config_id] = params
        entry = tf.entries.setdefault(sig, Entry(source=source))
        measured = entry.evidence.setdefault("points", {})
        measured[str(m)] = config_id
        if timings:
            entry.evidence.setdefault("timings", {})[f"M={m}"] = timings
        entry.regions = regions_from_points(measured)
        entry.source = source


def flush_user(reload_after: bool = True) -> List[Path]:
    """Write every pending point into its user-tier file; start a new epoch if any.

    Each file is read and written under its lock (:func:`update_file`), so a
    point another process flushed meanwhile survives this one's flush.  A point
    added to an entry measured under other validators leaves that entry stale.
    """
    with _lock:
        pending = dict(_PENDING)
        _PENDING.clear()
    written = []
    for (family, arch_family), (lane, points) in pending.items():
        path = user_path(arch_family, family)
        if path is None:
            continue
        update_file(
            path, family, arch_family, lane, functools.partial(_apply_points, points=points)
        )
        written.append(path)
    if written and reload_after:
        reload()
    return written


# =============================================================================
# Selection statistics
# =============================================================================

#: ``(kernel family, op, tier) -> resolutions``.  Counted per distinct shape
#: resolution, not per call: the per-call path is a memoised plan.
_TIER_COUNTS: "Counter[Tuple[str, str, str]]" = Counter()

#: ``(kernel family, op, runner name) -> dispatch plans``: which runner each
#: distinct shape was *dispatched* to.  Recorded by the dispatcher, not the
#: resolver, because a caller may resolve and then not use the answer (the
#: batch-invariant selector reads the tuned choice and keeps it only if it is
#: in its class).
_RUNNER_COUNTS: "Counter[Tuple[str, str, str]]" = Counter()


def record_tier(family: str, op: str, tier: str) -> None:
    _TIER_COUNTS[(family, op, tier)] += 1


def record_runner(family: str, op: str, runner: str) -> None:
    """Count one dispatch plan by the runner it runs on.

    A plan on an *external* runner (one whose kernels are not OASR's) is also a
    policy hit (``oasr.layers._backend.policy_hits``): a measured decision that
    a library is faster here today, so a standing argument for kernel work
    rather than an invisible bypass.
    """
    _RUNNER_COUNTS[(family, op, runner)] += 1
    from oasr.tune.runners import get_runner

    r = get_runner(family, runner)
    if r is None or not r.traits.external:
        return
    try:
        from oasr.layers._backend import take_policy
    except Exception:  # noqa: BLE001 -- accounting must never break a dispatch
        return
    take_policy(f"tuned-{runner}:{family}.{op}")


def tier_counts() -> Dict[Tuple[str, str, str], int]:
    return dict(_TIER_COUNTS)


def runner_counts() -> Dict[Tuple[str, str, str], int]:
    """``(family, op, runner) -> distinct shapes dispatched``, since the last reset."""
    return dict(_RUNNER_COUNTS)


def reset_tier_counts() -> None:
    _TIER_COUNTS.clear()
    _RUNNER_COUNTS.clear()


#: The order :func:`tuning_report` lists tiers in: measured first, then each
#: family's untuned fallback (``default`` for the GEMM family, ``heuristic`` for
#: ``scaled_mm``, ``auto`` for ``fmha``).
_REPORT_TIERS = ("user", "system", "prior", "model", "default", "heuristic", "auto")


def tuning_report(families: Optional[Iterable[str]] = None) -> str:
    """Which tier served each family's selections, and which files are stale.

    *families* restricts the report to those kernel families (default: every
    family).  The counters are process-wide, so a caller reporting on one model
    -- ``ASREngine.tuning_report`` for a full-precision model in a process that
    also served a quantized one -- names the families that model can issue.
    """
    wanted = None if families is None else set(families)
    lines = []
    by_family: Dict[Tuple[str, str], Counter] = {}
    for (family, op, tier), n in _TIER_COUNTS.items():
        if wanted is None or family in wanted:
            by_family.setdefault((family, op), Counter())[tier] += n
    if by_family:
        lines.append("kernel selection by tier (distinct shapes resolved):")
        for (family, op), c in sorted(by_family.items()):
            parts = ", ".join(f"{t}={c[t]}" for t in _REPORT_TIERS if c.get(t))
            lines.append(f"    {family}.{op:<18} {parts}")
    by_runner: Dict[Tuple[str, str], Counter] = {}
    for (family, op, runner), n in _RUNNER_COUNTS.items():
        if wanted is None or family in wanted:
            by_runner.setdefault((family, op), Counter())[runner] += n
    if by_runner:
        from oasr.tune.runners import get_runner

        lines.append("kernel dispatch by runner (distinct shapes; * = not an OASR kernel):")
        for (family, op), c in sorted(by_runner.items()):
            labels = []
            for name, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                r = get_runner(family, name)
                star = "*" if r is not None and r.traits.external else ""
                labels.append(f"{name}{star}={n}")
            lines.append(f"    {family}.{op:<18} {', '.join(labels)}")
    for t in list(_TIERS.values()):
        if wanted is not None and t.family not in wanted:
            continue
        for tier, tf in t.ordered():
            if tf.stale:
                lines.append(f"stale {tier} tuning file {tf.path}: " + "; ".join(tf.stale))
    return "\n".join(lines) if lines else "no tuned kernel selections resolved yet"


# =============================================================================
# Hash helpers (implementation identity for the soft validators)
# =============================================================================


def hash_bytes(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()[:16]


def hash_paths(paths: Iterable[Path]) -> str:
    """Content hash of *paths* (relative names + bytes), order-independent."""
    h = hashlib.sha256()
    for p in sorted({Path(p) for p in paths}, key=str):
        h.update(p.name.encode())
        with contextlib.suppress(OSError):
            h.update(p.read_bytes())
    return h.hexdigest()[:16]
