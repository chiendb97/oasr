# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Where a measurement goes: the terminal, a typed CSV, and two sidecars.

``<out>.csv``       one row per measurement, columns from :mod:`benchmarks.core.schema`
``<out>.meta.json`` the run manifest -- argv, git, device, versions, OASR_* switches, JIT hashes
``<out>.raw.json``  per-sample arrays that do not belong in a cell

The sidecars exist because the previous suite put raw latency arrays *in* the
JSON result and left the percentiles out, so every consumer had to recompute
them, while the rejection-cause histogram it printed was dropped entirely.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from benchmarks.core import env
from benchmarks.core.schema import (
    CATEGORY_KERNEL,
    CATEGORY_WORKLOAD,
    COLUMNS_FOR_CATEGORY,
    KernelRow,
    WorkloadRow,
    format_cell,
)


class Reporter:
    """Terminal + CSV + sidecars for one benchmark run."""

    def __init__(
        self,
        output: Optional[str] = None,
        category: str = CATEGORY_KERNEL,
        case_tag: str = "",
        verbosity: int = 0,
    ) -> None:
        self.category = category
        self.case_tag = case_tag or ""
        self.verbosity = verbosity
        self.n_rows = 0
        self._raw: Dict[str, Any] = {}
        self._fh: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None
        self._path: Optional[Path] = None

        if output:
            self._path = Path(output)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS_FOR_CATEGORY[category])
            self._writer.writeheader()

    # -- terminal ----------------------------------------------------------

    def header(self, title: str) -> None:
        print(f"\n[INFO] {title}")

    def verbose(self, msg: str, level: int = 1) -> None:
        if self.verbosity >= level:
            print(f"{'[VERBOSE]' if level == 1 else '[VVERBOSE]'} {msg}")

    def warn(self, msg: str) -> None:
        print(f"[WARNING] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}")

    def repro(self, cmd: str) -> None:
        if self.verbosity >= 1:
            print(f"[REPRO] {cmd}")

    # -- rows --------------------------------------------------------------

    def kernel_row(self, row: KernelRow) -> None:
        parts = [f"median time {row.median_ms:.3f} ms", f"std {row.std_ms:.3f} ms"]
        if row.tflops:
            parts.append(f"achieved tflops {row.tflops:.1f} TFLOPs/sec")
        if row.bandwidth_tb_s:
            parts.append(f"achieved tb_per_sec {row.bandwidth_tb_s:.2f} TB/sec")
        if row.speedup_vs_ref is not None and row.backend != row.ref_backend:
            parts.append(f"{row.speedup_vs_ref:.2f}x vs {row.ref_backend}")
        if row.refcheck == "fail":
            parts.append(f"REFCHECK FAIL (max_abs_diff {row.max_abs_diff:.3g})")
        print(f"[PERF] {row.backend:<12} :: {'; '.join(parts)}")
        self._write(row.as_dict())

    def workload_row(self, row: WorkloadRow) -> None:
        label = f"{row.subroutine}/{row.backend}" if row.backend else row.subroutine
        head = f"[PERF] {label:<28} :: "
        if row.median_ms is not None:
            head += f"median time {row.median_ms:.1f} ms; std {row.std_ms or 0.0:.1f} ms"
        else:
            head += f"wall {row.wall_s or 0.0:.2f} s"
        print(head)
        detail: List[str] = []
        if row.rtfx is not None:
            detail.append(f"RTFx={row.rtfx:.2f}")
        if row.throughput_utts_per_s is not None:
            detail.append(f"{row.throughput_utts_per_s:.2f} utts/s")
        if row.throughput_req_per_s is not None:
            detail.append(f"{row.throughput_req_per_s:.2f} req/s")
        if row.tokens_per_s is not None:
            detail.append(f"{row.tokens_per_s:.2f} tok/s")
        if row.throughput_frames_per_s is not None:
            detail.append(f"{row.throughput_frames_per_s:,.0f} frames/s")
        if row.audio_s is not None:
            detail.append(f"audio={row.audio_s:.1f}s")
        if row.error_rate_pct is not None:
            detail.append(f"{row.metric.upper()}={row.error_rate_pct:.2f}%")
        if row.speedup_vs_ref is not None and row.backend != row.ref_backend:
            detail.append(f"{row.speedup_vs_ref:.2f}x vs {row.ref_backend}")
        if detail:
            print("         " + "  ".join(detail))
        lat = [
            (n, v)
            for n, v in (
                ("p50", row.latency_p50_ms),
                ("p90", row.latency_p90_ms),
                ("p99", row.latency_p99_ms),
                ("max", row.latency_max_ms),
            )
            if v is not None
        ]
        if lat:
            print("         latency  " + "  ".join(f"{n}={v:.0f}ms" for n, v in lat))
        self._write(row.as_dict())

    def _write(self, values: Dict[str, Any]) -> None:
        self.n_rows += 1
        if self._writer is None:
            return
        merged = dict(env.envelope(self.category, self.case_tag))
        merged.update(values)
        columns = COLUMNS_FOR_CATEGORY[self.category]
        self._writer.writerow({c: format_cell(c, merged.get(c)) for c in columns})

    # -- sidecars ----------------------------------------------------------

    def raw(self, key: str, payload: Any) -> None:
        """Stash a per-sample array under *key* for ``<out>.raw.json``."""
        self._raw[key] = payload

    def finalize(self) -> None:
        print(f"\n[INFO] Benchmarks complete! ({self.n_rows} measurements)")
        if self._fh is not None and self._path is not None:
            self._fh.close()
            print(f"[INFO] Results saved to: {self._path}")
            meta = self._path.with_suffix(".meta.json")
            meta.write_text(json.dumps(env.snapshot(), indent=2) + "\n")
            print(f"[INFO] Run manifest:     {meta}")
            if self._raw:
                raw = self._path.with_suffix(".raw.json")
                raw.write_text(json.dumps(self._raw, indent=2) + "\n")
                print(f"[INFO] Raw samples:      {raw}")


def print_schema() -> None:
    """Emit both column lists, so the docs can cite one source."""
    for category in (CATEGORY_KERNEL, CATEGORY_WORKLOAD):
        print(f"\n{category} columns ({len(COLUMNS_FOR_CATEGORY[category])}):")
        for col in COLUMNS_FOR_CATEGORY[category]:
            print(f"  {col}")
