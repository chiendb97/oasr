# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""What the machine was, captured once per run.

Two reasons this is not a set of CSV columns.  It does not vary within a run, so
repeating the driver version on every row is noise; and the interesting parts
(the full argv, every ``OASR_*`` switch, the JIT cache hashes) do not fit a cell.

The JIT hashes are the point of the sidecar.  ``AGENTS.md`` makes "confirm a
fresh JIT hash directory before trusting a kernel benchmark" a rule, and a rule
you have to remember is one you eventually forget; recorded beside the numbers,
it is checkable after the fact.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_CACHED: Optional[Dict[str, Any]] = None
_RUN_ID: Optional[str] = None
_JIT_CACHE = Path.home() / ".cache" / "oasr" / "jit"


def run_id() -> str:
    """Short identifier shared by every row this process writes."""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = uuid.uuid4().hex[:8]
    return _RUN_ID


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def git_info() -> Dict[str, Any]:
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {
        "commit": commit,
        "dirty": dirty,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def device_info() -> Dict[str, Any]:
    """GPU identity.  Queried once -- the previous writer re-queried per row."""
    if not torch.cuda.is_available():
        return {"name": "N/A", "sm": "N/A", "memory_gb": 0.0, "index": -1, "count": 0}
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "name": props.name,
        "sm": f"{props.major}.{props.minor}",
        "memory_gb": round(props.total_memory / 1e9, 1),
        "index": index,
        "count": torch.cuda.device_count(),
    }


def _version(module: str, attr: str = "__version__") -> str:
    try:
        mod = __import__(module)
        return str(getattr(mod, attr, ""))
    except Exception:
        return ""


def versions() -> Dict[str, str]:
    cutlass = ""
    header = (
        Path(__file__).resolve().parents[2]
        / "3rdparty"
        / "cutlass"
        / "include"
        / "cutlass"
        / "version.h"
    )
    if header.exists():
        parts = {}
        for line in header.read_text().splitlines():
            for key in ("MAJOR", "MINOR", "PATCH"):
                token = f"#define CUTLASS_{key} "
                if line.startswith(token):
                    parts[key] = line[len(token) :].strip()
        if len(parts) == 3:
            cutlass = f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"
    return {
        "python": sys.version.split()[0],
        "oasr": _version("oasr"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "driver": _driver_version(),
        "cutlass": cutlass,
        "triton": _version("triton"),
    }


def _driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return ""


def oasr_env() -> Dict[str, str]:
    """Every ``OASR_*`` switch in the environment.

    These select kernels and disable heuristics, so a number measured under one
    setting is not comparable with one measured under another.
    """
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("OASR_")}


def jit_cache_state() -> Dict[str, Any]:
    if not _JIT_CACHE.is_dir():
        return {"path": str(_JIT_CACHE), "present": False, "hashes": []}
    hashes = sorted(p.name for p in _JIT_CACHE.iterdir() if p.is_dir())
    return {"path": str(_JIT_CACHE), "present": True, "hashes": hashes}


def snapshot() -> Dict[str, Any]:
    """The whole run manifest.  Cached: nothing in it changes mid-run."""
    global _CACHED
    if _CACHED is None:
        _CACHED = {
            "run_id": run_id(),
            "timestamp": timestamp(),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "git": git_info(),
            "device": device_info(),
            "versions": versions(),
            "oasr_env": oasr_env(),
            "jit_cache": jit_cache_state(),
        }
    return _CACHED


def envelope(category: str, case_tag: str = "") -> Dict[str, Any]:
    """The columns both schemas open with."""
    from benchmarks.core.schema import SCHEMA_VERSION

    snap = snapshot()
    return {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "run_id": snap["run_id"],
        "timestamp": snap["timestamp"],
        "case_tag": case_tag or "",
        "git_commit": snap["git"]["commit"] + ("-dirty" if snap["git"]["dirty"] else ""),
        "device": snap["device"]["name"],
        "sm": snap["device"]["sm"],
    }
