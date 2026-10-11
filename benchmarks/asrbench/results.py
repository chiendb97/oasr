# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Run directories: raw records plus the summaries derived from them.

    <results_root>/<YYYYmmdd-HHMMSS>_<profile>_<backend>_<run_id>/
        run.json          environment, profile, backend, workloads, result summaries
        requests.jsonl    one line per request (offline, batch, load, long-form)
        sessions.jsonl    one line per streaming session, its events inline
        resources.jsonl   resource monitor samples (monotonic time)
        workloads/<name>.json   each resolved workload, every item listed

Raw records are the source of truth.  Every number in ``run.json`` is
:mod:`benchmarks.asrbench.analysis` applied to them, and ``recompute`` re-derives
the summaries and reports any disagreement -- so a summary can be audited, and
a metric added later can be computed for an old run.

Times in records are seconds on the run clock: ``perf_counter`` minus the run's
origin.  ``run.json`` stores the origin on both ``perf_counter`` and
``monotonic`` so resource samples taken in another process line up.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from benchmarks.asrbench import RESULT_SCHEMA_VERSION, SUITE_VERSION

SCHEMA = "oasr.asrbench.run"
RESULTS_DIR_ENV = "ASRBENCH_RESULTS_DIR"

RUN_JSON = "run.json"
REQUESTS = "requests.jsonl"
SESSIONS = "sessions.jsonl"
RESOURCES = "resources.jsonl"

# Request / session outcomes.
OK = "ok"
FAILED = "failed"
TIMEOUT = "timeout"
REJECTED = "rejected"

# Benchmark-result statuses.
STATUS_OK = "ok"
STATUS_UNSUPPORTED = "unsupported"
STATUS_FAILED = "failed"


def default_results_root() -> Path:
    """``$ASRBENCH_RESULTS_DIR``, else ``$XDG_DATA_HOME/oasr/asrbench/results``.

    The XDG default (``~/.local/share/...``) is persistent and outside any
    checkout: runs survive a fresh clone and never appear in ``git status``.
    Prepared datasets, which can be rebuilt, live under ``~/.cache`` instead.
    """
    env = os.environ.get(RESULTS_DIR_ENV)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if os.path.isabs(xdg) else Path.home() / ".local" / "share"
    return base / "oasr" / "asrbench" / "results"


def resolve_run(ref: str, root: Optional[Path] = None) -> Path:
    """A run or validation directory from a path, a directory name under the
    results root, a baseline name, or a run id (the directory's last field).

    An unresolvable reference comes back as the path it names, so the caller
    reports it missing in its usual way.
    """
    path = Path(ref).expanduser()
    if path.exists() or os.sep in ref:
        return path
    root = Path(root) if root else default_results_root()
    for cand in (root / ref, root / "baselines" / ref, root / "validation" / ref):
        if cand.exists():
            return cand
    matches = [
        d
        for parent in (root, root / "validation")
        if parent.is_dir()
        for d in parent.glob(f"*_{ref}")
        if d.is_dir()
    ]
    if len(matches) > 1:
        names = ", ".join(sorted(m.name for m in matches))
        raise ValueError(f"run id {ref!r} matches more than one directory: {names}")
    return matches[0] if matches else path


def list_runs(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Runs under the results root, newest first, read from their ``run.json``."""
    root = Path(root) if root else default_results_root()
    out = []
    for d in root.glob(f"*/{RUN_JSON}"):
        try:
            meta = json.loads(d.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        results = meta.get("results") or []
        out.append(
            {
                "run_id": meta.get("run_id", ""),
                "started_utc": meta.get("started_utc", ""),
                "profile": (meta.get("profile") or {}).get("name", ""),
                "backend": (meta.get("backend") or {}).get("name", ""),
                "tag": meta.get("tag") or "",
                "results": len(results),
                "ok": sum(1 for r in results if r.get("status") == STATUS_OK),
                "finished": bool(meta.get("finished_utc")),
                "path": str(d.parent),
            }
        )
    # Directory names stamp whole seconds; run.json has milliseconds.
    return sorted(out, key=lambda r: (r["started_utc"], r["path"]), reverse=True)


class Clock:
    """The run clock: seconds since the run started, on ``perf_counter``."""

    def __init__(self) -> None:
        self.origin_perf = time.perf_counter()
        self.origin_mono = time.monotonic()
        self.origin_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def now(self) -> float:
        return time.perf_counter() - self.origin_perf

    def rel(self, perf_t: Optional[float]) -> Optional[float]:
        return None if perf_t is None else perf_t - self.origin_perf

    def from_mono(self, mono_t: float) -> float:
        return mono_t - self.origin_mono


@dataclass
class RequestRecord:
    result_id: str
    rep: int
    phase: str  # "warmup" | "measure"
    seq: int
    item: str
    duration_s: float
    bucket: str = ""
    batch: Optional[int] = None
    worker: Optional[int] = None
    t_sched: float = 0.0
    t_submit: float = 0.0
    t_first: Optional[float] = None
    t_done: Optional[float] = None
    status: str = OK
    error: Optional[str] = None
    text: Optional[str] = None

    @property
    def latency_ms(self) -> Optional[float]:
        return None if self.t_done is None else (self.t_done - self.t_sched) * 1e3

    @property
    def service_ms(self) -> Optional[float]:
        return None if self.t_done is None else (self.t_done - self.t_submit) * 1e3

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SessionRecord:
    result_id: str
    rep: int
    phase: str
    seq: int
    item: str
    duration_s: float
    chunk_ms: float
    paced: bool
    worker: Optional[int] = None
    #: When the session's audio "started": chunk k covers audio
    #: ``[k*c, (k+1)*c)`` and is due at ``t_audio0 + (k+1)*c`` when paced.
    t_audio0: float = 0.0
    chunk_due: List[float] = field(default_factory=list)
    chunk_sent: List[float] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = OK
    error: Optional[str] = None
    text: Optional[str] = None
    words: Optional[List[Dict[str, Any]]] = None

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class BenchmarkResult:
    result_id: str
    benchmark: str
    kind: str
    mode: str
    point: Dict[str, Any]
    workload: str
    workload_id: str
    params: Dict[str, Any]
    status: str = STATUS_OK
    reason: Optional[str] = None
    #: Measured window on the run clock -- resources are attributed to it.
    window: List[float] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)
    server: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def key(self) -> str:
        """Identity for comparisons: what was measured, independent of the run."""
        point = ",".join(f"{k}={self.point[k]}" for k in sorted(self.point))
        return f"{self.benchmark}[{point}]@{self.workload_id}"


@dataclass
class Run:
    """An opened run directory."""

    path: Path
    meta: Dict[str, Any]
    results: List[BenchmarkResult]

    @property
    def run_id(self) -> str:
        return str(self.meta.get("run_id", ""))

    @property
    def label(self) -> str:
        """``--tag`` if given, else ``backend@commit#run_id`` -- unique per run."""
        tag = self.meta.get("tag") or ""
        backend = self.meta.get("backend", {}).get("name", "")
        git = self.meta.get("environment", {}).get("git", {})
        commit = git.get("short") or str(git.get("commit", ""))[:7]
        return tag or f"{backend}@{commit}#{self.run_id}"

    def requests(self) -> List[RequestRecord]:
        return [RequestRecord(**r) for r in _read_jsonl(self.path / REQUESTS)]

    def sessions(self) -> List[SessionRecord]:
        return [SessionRecord(**r) for r in _read_jsonl(self.path / SESSIONS)]

    def resource_samples(self) -> List[Dict[str, Any]]:
        return list(_read_jsonl(self.path / RESOURCES))

    def workload(self, name: str) -> Dict[str, Any]:
        return json.loads((self.path / "workloads" / f"{name}.json").read_text(encoding="utf-8"))


def new_run_dir(root: Path, profile: str, backend: str, run_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{stamp}_{_slug(profile)}_{_slug(backend)}_{run_id}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "workloads").mkdir()
    return path


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in s)[:40] or "x"


def base_meta(run_id: str, clock: Clock) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "run_id": run_id,
        "started_utc": clock.origin_utc,
        "clock": {"origin_perf_counter": clock.origin_perf, "origin_monotonic": clock.origin_mono},
    }


def write_run_json(path: Path, meta: Dict[str, Any], results: Iterable[BenchmarkResult]) -> None:
    doc = dict(meta)
    doc["results"] = [r.to_json() for r in results]
    tmp = path / (RUN_JSON + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path / RUN_JSON)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and o != o:
        return None
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_run(path: Path) -> Run:
    """Open a run directory (or its ``run.json``)."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    doc_path = path / RUN_JSON
    if not doc_path.is_file():
        raise FileNotFoundError(f"{path} is not an asrbench run (no {RUN_JSON})")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{doc_path}: not an asrbench run (schema={doc.get('schema')!r})")
    if int(doc.get("schema_version", 0)) > RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"{doc_path}: schema_version {doc['schema_version']} is newer than this "
            f"suite reads ({RESULT_SCHEMA_VERSION}); update the checkout"
        )
    results = [BenchmarkResult.from_json(r) for r in doc.pop("results", [])]
    return Run(path=path, meta=doc, results=results)


def copy_run(src: Path, dest: Path) -> Path:
    src = load_run(src).path
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    shutil.copytree(src, dest)
    return dest
