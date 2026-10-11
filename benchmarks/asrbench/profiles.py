# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Profiles: named, versioned benchmark plans (YAML), plus command-line overrides.

A profile declares workloads (what audio) and benchmarks (what protocol, which
sweep).  Built-ins live in ``benchmarks/asrbench/profiles/``; any YAML path works
too.  The *resolved* profile -- after ``${primary}`` substitution and every
override -- is what a run records, so a run is reproducible from its own
``run.json`` regardless of the flags that produced it.

Sizing deliberately does **not** come from ``.env`` (``$NUM_UTTERANCES``,
``$CONCURRENCY``...).  The older harnesses take argparse defaults from there,
and AGENTS.md documents the hazard that creates: an unset variable measures a
different working point without saying so.  Here the profile is the single
source of sizing and it is recorded; ``.env`` supplies only paths (checkpoints,
data directories).

``${primary}`` / ``${primary_slug}`` in a dataset reference expand to the
profile's primary dataset, so ``--dataset ljspeech/all`` retargets every
utterance workload *and* the long-form set derived from it in one flag.
"""

from __future__ import annotations

import copy
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from benchmarks.asrbench.workloads import WorkloadSpec

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"

#: Convenience flags -> (benchmark kinds they apply to, parameter name).
SWEEP_FLAGS: Dict[str, tuple] = {
    "batch_sizes": (("batch",), "batch_sizes"),
    "compositions": (("batch",), "compositions"),
    "concurrency": (("load",), "concurrency"),
    "rates": (("load",), "rates"),
    "chunk_ms": (("streaming",), "chunk_ms"),
    "sessions": (("streaming",), "sessions"),
    "repetitions": (None, "repetitions"),
    "warmup": (None, "warmup"),
    "timeout_s": (None, "timeout_s"),
}


@dataclass
class Profile:
    name: str
    description: str = ""
    defaults: Dict[str, Any] = field(default_factory=dict)
    workloads: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    benchmarks: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""

    # -- loading -------------------------------------------------------------

    @classmethod
    def load(cls, name_or_path: str) -> "Profile":
        path = Path(name_or_path)
        if not path.is_file():
            path = PROFILE_DIR / f"{name_or_path}.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"no profile {name_or_path!r}; built-ins: {', '.join(builtin_profiles())}"
            )
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(doc, source=str(path))

    @classmethod
    def from_dict(cls, doc: Dict[str, Any], source: str = "") -> "Profile":
        unknown = set(doc) - {"name", "description", "defaults", "workloads", "benchmarks"}
        if unknown:
            raise ValueError(f"profile: unknown top-level key(s) {sorted(unknown)}")
        prof = cls(
            name=str(doc.get("name") or Path(source).stem or "profile"),
            description=str(doc.get("description") or ""),
            defaults=dict(doc.get("defaults") or {}),
            workloads={k: dict(v or {}) for k, v in (doc.get("workloads") or {}).items()},
            benchmarks=[dict(b) for b in (doc.get("benchmarks") or [])],
            source=source,
        )
        names = set()
        for k, b in enumerate(prof.benchmarks):
            if "kind" not in b:
                raise ValueError(f"profile {prof.name}: benchmark #{k} has no kind")
            b.setdefault("name", b["kind"])
            if b["name"] in names:
                raise ValueError(f"profile {prof.name}: duplicate benchmark name {b['name']!r}")
            names.add(b["name"])
            if b.get("workload") not in prof.workloads:
                raise ValueError(
                    f"profile {prof.name}: benchmark {b['name']!r} names unknown workload "
                    f"{b.get('workload')!r}"
                )
        return prof

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "defaults": self.defaults,
            "workloads": self.workloads,
            "benchmarks": self.benchmarks,
        }

    # -- overrides ---------------------------------------------------------------

    def resolved(
        self,
        *,
        primary: Optional[str] = None,
        only: Sequence[str] = (),
        skip: Sequence[str] = (),
        sweeps: Optional[Dict[str, Any]] = None,
        workload_overrides: Optional[Dict[str, Any]] = None,
        sets: Sequence[str] = (),
    ) -> "Profile":
        """A copy with every override applied, in a fixed order."""
        p = copy.deepcopy(self)
        if primary:
            p.defaults["primary"] = primary
        subst = {
            "primary": p.defaults.get("primary", ""),
            "primary_slug": str(p.defaults.get("primary", "")).replace("/", "-"),
        }
        for spec in p.workloads.values():
            if isinstance(spec.get("dataset"), str):
                spec["dataset"] = string.Template(spec["dataset"]).safe_substitute(subst)
        for key, value in (workload_overrides or {}).items():
            if value is None:
                continue
            for spec in p.workloads.values():
                spec[key] = value
        for flag, value in (sweeps or {}).items():
            if value is None:
                continue
            kinds, param = SWEEP_FLAGS[flag]
            for b in p.benchmarks:
                if kinds is None or b["kind"] in kinds:
                    if flag == "rates" and b["kind"] == "load":
                        b["mode"] = "open"
                    if flag == "concurrency" and b["kind"] == "load" and b.get("mode") == "open":
                        continue
                    b[param] = value
        for assignment in sets:
            _apply_set(p, assignment)
        if only:
            keep = set(only)
            p.benchmarks = [b for b in p.benchmarks if b["name"] in keep or b["kind"] in keep]
        if skip:
            drop = set(skip)
            p.benchmarks = [
                b for b in p.benchmarks if b["name"] not in drop and b["kind"] not in drop
            ]
        if not p.benchmarks:
            raise ValueError("no benchmarks left after --only/--skip")
        return p

    def workload_spec(self, name: str) -> WorkloadSpec:
        spec = dict(self.workloads[name])
        spec.setdefault("seed", int(self.defaults.get("seed", 0)))
        return WorkloadSpec.from_dict(name, spec)


def _apply_set(p: Profile, assignment: str) -> None:
    """``defaults.seed=3`` / ``workloads.utts.num_items=50`` / ``batch.batch_sizes=[1,2]``."""
    if "=" not in assignment:
        raise ValueError(f"--set expects path=value, got {assignment!r}")
    path, _, raw = assignment.partition("=")
    value = yaml.safe_load(raw)
    head, _, rest = path.partition(".")
    if not rest:
        raise ValueError(f"--set path needs a section: {path!r}")
    if head == "defaults":
        p.defaults[rest] = value
        return
    if head == "workloads":
        wname, _, key = rest.partition(".")
        if wname not in p.workloads or not key:
            raise ValueError(f"--set: no workload {wname!r}")
        p.workloads[wname][key] = value
        return
    for b in p.benchmarks:
        if b["name"] == head:
            b[rest] = value
            return
    raise ValueError(f"--set: no benchmark named {head!r} (or section defaults/workloads)")


def builtin_profiles() -> List[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
