# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""In-memory and JSON-persistent cache for autotuning results."""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ._types import ProfileKey, Tactic, TuneResult

logger = logging.getLogger("oasr.tune")


def _get_env_metadata() -> Dict[str, Any]:
    """Collect current environment metadata for cache validation."""
    import oasr

    meta: Dict[str, Any] = {"oasr_version": oasr.__version__}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            meta["gpu_name"] = props.name
            meta["sm"] = props.major * 10 + props.minor
    except Exception:
        pass
    try:
        import torch

        meta["cuda_version"] = torch.version.cuda or "unknown"
    except Exception:
        pass
    return meta


class TuneCache:
    """Three-level cache: in-memory → persisted → bundled defaults.

    Thread-safe for concurrent reads/writes within a single process.
    """

    def __init__(self) -> None:
        self._memory: Dict[str, Tactic] = {}
        self._persisted: Dict[str, Tactic] = {}
        self._bundled: Dict[str, Tactic] = {}
        self._timings: Dict[str, float] = {}  # key → median_ms
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, key: ProfileKey) -> Optional[Tactic]:
        """Look up the best tactic for *key*.

        Priority: in-memory → persisted → bundled → ``None``.
        """
        key_str = key.to_str()
        tactic = self._memory.get(key_str)
        if tactic is not None:
            return tactic
        tactic = self._persisted.get(key_str)
        if tactic is not None:
            return tactic
        return self._bundled.get(key_str)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, key: ProfileKey, result: TuneResult) -> None:
        """Store a profiling result in the in-memory cache."""
        if result.status != "ok":
            return
        key_str = key.to_str()
        with self._lock:
            self._memory[key_str] = result.tactic
            self._timings[key_str] = result.median_ms

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, path: Path) -> None:
        """Load a persisted JSON cache file.

        Merges entries into the persisted layer without overwriting
        in-memory results. Warns if environment metadata differs.
        """
        path = Path(path)
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load tune cache %s: %s", path, exc)
            return

        # Validate environment metadata
        file_env = data.get("env", {})
        current_env = _get_env_metadata()
        for field in ("sm", "cuda_version"):
            if field in file_env and field in current_env:
                if file_env[field] != current_env[field]:
                    logger.warning(
                        "Tune cache %s was created with %s=%s but current is %s=%s. "
                        "Cached configs may not be optimal.",
                        path,
                        field,
                        file_env[field],
                        field,
                        current_env[field],
                    )

        entries = data.get("entries", {})
        with self._lock:
            for key_str, entry_data in entries.items():
                if key_str not in self._persisted:
                    self._persisted[key_str] = Tactic.from_dict(entry_data)

    def save(self, path: Path) -> None:
        """Atomically write the combined cache (memory + persisted) to JSON.

        Uses ``tempfile`` + ``os.replace()`` for atomic writes.
        """
        path = Path(path)

        # Merge: start from persisted, overlay with in-memory
        with self._lock:
            merged: Dict[str, Dict[str, Any]] = {}
            for key_str, tactic in self._persisted.items():
                merged[key_str] = tactic.to_dict()
            for key_str, tactic in self._memory.items():
                entry = tactic.to_dict()
                if key_str in self._timings:
                    entry["median_ms"] = round(self._timings[key_str], 6)
                entry["profiled_at"] = datetime.now(timezone.utc).isoformat()
                merged[key_str] = entry

        # Also merge with existing file to support incremental tuning
        if path.exists():
            try:
                with open(path) as f:
                    existing = json.load(f)
                existing_entries = existing.get("entries", {})
                for key_str, entry_data in existing_entries.items():
                    if key_str not in merged:
                        merged[key_str] = entry_data
            except (json.JSONDecodeError, OSError):
                pass  # Overwrite corrupt file

        data = {
            "version": 1,
            "env": _get_env_metadata(),
            "entries": dict(sorted(merged.items())),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".tune_cache_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=False)
                f.write("\n")
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_bundled(self, path: Path) -> None:
        """Load bundled default configs (shipped with the package)."""
        path = Path(path)
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load bundled configs %s: %s", path, exc)
            return
        entries = data.get("entries", {})
        with self._lock:
            for key_str, entry_data in entries.items():
                if key_str not in self._bundled:
                    self._bundled[key_str] = Tactic.from_dict(entry_data)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all in-memory cached entries."""
        with self._lock:
            self._memory.clear()
            self._timings.clear()

    def size(self) -> int:
        """Return the total number of cached entries across all levels."""
        return len(self._memory) + len(self._persisted) + len(self._bundled)
