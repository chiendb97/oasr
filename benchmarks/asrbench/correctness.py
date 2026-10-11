# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``validate`` -- the correctness guardrail.  Untimed, and separate from ``run``.

A performance number is only worth having if the backend still transcribes.
This runs a fixed labelled subset through a backend and checks:

* **WER** against the references (corpus WER, Whisper's English normaliser by
  default -- ``oasr.testing.wer``, the same code the CI accuracy gate uses);
* **empty transcripts** -- the failure mode several real kernel bugs produced
  (a wrong-but-fast kernel returning ``""`` for every request), which a WER
  threshold set loosely enough for an out-of-domain model can miss;
* **drift against a known-good run** (``--reference``) -- the share of
  utterances whose normalised hypothesis changed, and the WER of the new
  hypotheses scored against the old ones.  This is how an optimisation is
  checked to be output-preserving, or one implementation of a model checked
  against another (e.g. ``oasr_engine`` vs ``transformers`` on one Whisper
  checkpoint).

It is deliberately *not* a quality benchmark: one subset, one normaliser, no
model ranking.  Thresholds are inputs; none are universal.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.asrbench import SUITE_VERSION, backends, environment
from benchmarks.asrbench.results import default_results_root
from benchmarks.asrbench.workloads import WorkItem, Workload, WorkloadSpec, load_audio, resolve

VALIDATION_JSON = "validation.json"
PREDICTIONS = "predictions.jsonl"


@dataclass
class Thresholds:
    max_wer_pct: Optional[float] = None
    max_empty_rate: Optional[float] = 0.0
    max_changed_pct: Optional[float] = None
    max_drift_wer_pct: Optional[float] = None
    max_wer_delta_pct: Optional[float] = None


def manifest_workload(path: Path, audio_root: Optional[Path], num_items: Optional[int]) -> Workload:
    """A workload from an ``{id, audio, text}`` manifest (e.g. ``ljspeech_200.jsonl``)."""
    import hashlib

    from benchmarks.asrbench.audio import read_info

    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(json.loads(line))
    items: List[WorkItem] = []
    root = audio_root or Path(path).parent
    for e in entries[: num_items or None]:
        audio = Path(e["audio"])
        audio = audio if audio.is_absolute() else root / audio
        frames, sr = read_info(audio)
        items.append(
            WorkItem(
                uid=str(e.get("id", audio.stem)),
                source_id=str(e.get("id", audio.stem)),
                audio_path=str(audio),
                duration_s=frames / sr,
                text=e.get("text"),
            )
        )
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Workload(
        spec=WorkloadSpec(name="validation", dataset=f"manifest:{Path(path).name}"),
        items=items,
        dataset={"name": "manifest", "subset": Path(path).name, "manifest_sha256": sha},
    )


async def _transcribe_all(
    backend: backends.Backend, payloads: List[Any], batch_size: int, concurrency: int
) -> List[Tuple[Optional[str], Optional[str]]]:
    """``[(text, error)]`` in input order, by the most efficient path the backend has."""
    caps = backend.capabilities()
    out: List[Tuple[Optional[str], Optional[str]]] = [(None, None)] * len(payloads)
    if caps.batch:
        for start in range(0, len(payloads), batch_size):
            chunk = payloads[start : start + batch_size]
            try:
                res = await backend.transcribe_batch(chunk)
                for k, tr in enumerate(res):
                    out[start + k] = (tr.text, None)
            except Exception as exc:  # noqa: BLE001
                for k in range(len(chunk)):
                    out[start + k] = (None, f"{type(exc).__name__}: {exc}")
        return out
    sem = asyncio.Semaphore(concurrency if caps.concurrent else 1)

    async def one(k: int) -> None:
        async with sem:
            try:
                out[k] = ((await backend.transcribe(payloads[k])).text, None)
            except Exception as exc:  # noqa: BLE001
                out[k] = (None, f"{type(exc).__name__}: {exc}")

    await asyncio.gather(*(one(k) for k in range(len(payloads))))
    return out


def load_reference(path: Path) -> Dict[str, str]:
    """``{id: hypothesis}`` from a predictions.jsonl or a validation directory."""
    path = Path(path)
    if path.is_dir():
        path = path / PREDICTIONS
    refs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("hyp") is not None:
                refs[str(row["id"])] = row["hyp"]
    return refs


def evaluate(
    items: List[WorkItem],
    hyps: List[Optional[str]],
    errors: List[Optional[str]],
    *,
    normalizer: str,
    reference: Optional[Dict[str, str]],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    """Score hypotheses and apply the thresholds.  Pure: no backend, no I/O."""
    from oasr.testing.wer import compute, normalizer as make_normalizer

    norm = make_normalizer(normalizer)
    ok = [k for k, h in enumerate(hyps) if h is not None]
    result = compute(
        [items[k].text or "" for k in ok],
        [hyps[k] or "" for k in ok],
        unit="word",
        normalizer=norm,
        uids=[items[k].uid for k in ok],
    )
    empty = sum(1 for k in ok if not (hyps[k] or "").strip())
    report: Dict[str, Any] = {
        "num_items": len(items),
        "failed": len(items) - len(ok),
        "errors": [e for e in errors if e][:10],
        "normalizer": normalizer,
        "wer_pct": result.percent,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "ref_words": result.counts.ref_len,
        "empty_rate": empty / len(ok) if ok else None,
        "worst": result.worst(5),
    }
    if reference:
        common = [k for k in ok if items[k].uid in reference]
        changed = sum(1 for k in common if norm(reference[items[k].uid]) != norm(hyps[k] or ""))
        drift = compute(
            [reference[items[k].uid] for k in common],
            [hyps[k] or "" for k in common],
            unit="word",
            normalizer=norm,
        )
        base = compute(
            [items[k].text or "" for k in common],
            [reference[items[k].uid] for k in common],
            unit="word",
            normalizer=norm,
        )
        report["reference"] = {
            "matched": len(common),
            "changed_pct": 100.0 * changed / len(common) if common else None,
            "drift_wer_pct": drift.percent if common else None,
            "reference_wer_pct": base.percent if common else None,
            "wer_delta_pct": (
                compute(
                    [items[k].text or "" for k in common],
                    [hyps[k] or "" for k in common],
                    unit="word",
                    normalizer=norm,
                ).percent
                - base.percent
                if common
                else None
            ),
        }
    checks = []

    def check(name: str, value: Optional[float], limit: Optional[float]) -> None:
        if limit is None:
            return
        checks.append(
            {
                "check": name,
                "value": value,
                "limit": limit,
                "ok": value is not None and value <= limit,
            }
        )

    check("failed_requests", float(report["failed"]), 0.0)
    check("wer_pct", report["wer_pct"], thresholds.max_wer_pct)
    check("empty_rate", report["empty_rate"], thresholds.max_empty_rate)
    ref = report.get("reference") or {}
    if reference is not None:
        check("changed_pct", ref.get("changed_pct"), thresholds.max_changed_pct)
        check("drift_wer_pct", ref.get("drift_wer_pct"), thresholds.max_drift_wer_pct)
        check("wer_delta_pct", ref.get("wer_delta_pct"), thresholds.max_wer_delta_pct)
    report["checks"] = checks
    report["status"] = "pass" if all(c["ok"] for c in checks) else "fail"
    return report


def run_validation(
    backend_name: str,
    options: Dict[str, Any],
    *,
    dataset: str = "librispeech/test-clean",
    manifest: Optional[str] = None,
    audio_root: Optional[Path] = None,
    num_items: Optional[int] = 200,
    seed: int = 0,
    normalizer: str = "english",
    batch_size: int = 16,
    concurrency: int = 8,
    reference: Optional[Path] = None,
    thresholds: Optional[Thresholds] = None,
    data_root: Optional[Path] = None,
    out_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], int]:
    thresholds = thresholds or Thresholds()
    if manifest:
        workload = manifest_workload(Path(manifest), audio_root, num_items)
    else:
        spec = WorkloadSpec(
            name="validation",
            dataset=dataset,
            num_items=num_items,
            sampling="random",
            order="manifest",
            seed=seed,
            require_text=True,
        )
        workload = resolve(spec, data_root)
    ref_hyps = load_reference(reference) if reference else None

    options = dict(options)
    options.setdefault("sample_rate", 16000)

    async def go() -> Tuple[List[Tuple[Optional[str], Optional[str]]], Dict[str, Any], float]:
        backend = backends.create(backend_name, options, "offline")
        await backend.load()
        try:
            audio = load_audio(workload.items, backend.sample_rate)
            payloads = [backend.prepare(audio[it.uid], it) for it in workload.items]
            t0 = time.perf_counter()
            outs = await _transcribe_all(backend, payloads, batch_size, concurrency)
            return outs, backend.info().to_json(), time.perf_counter() - t0
        finally:
            await backend.close()

    outs, info, wall = asyncio.run(go())
    hyps = [o[0] for o in outs]
    report = evaluate(
        workload.items,
        hyps,
        [o[1] for o in outs],
        normalizer=normalizer,
        reference=ref_hyps,
        thresholds=thresholds,
    )
    report.update(
        {
            "schema": "oasr.asrbench.validation",
            "suite_version": SUITE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": {"name": backend_name, "options": options, "info": info},
            "workload": {
                "id": workload.id,
                "dataset": f"{workload.dataset.get('name')}/{workload.dataset.get('subset')}",
                "manifest_sha256": workload.dataset.get("manifest_sha256"),
                "num_items": len(workload.items),
                "seed": seed,
            },
            "reference_path": str(reference) if reference else None,
            "thresholds": asdict(thresholds),
            # Wall time is informational only: batching, normalisation and the
            # first-call warm-up all sit inside it.  `run` measures speed.
            "wall_s_untimed": wall,
            "environment": environment.collect(),
        }
    )
    out_dir = (out_root or default_results_root() / "validation") / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{backend_name}_{uuid.uuid4().hex[:6]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / VALIDATION_JSON).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    with open(out_dir / PREDICTIONS, "w", encoding="utf-8") as f:
        for it, (hyp, err) in zip(workload.items, outs):
            f.write(
                json.dumps(
                    {"id": it.uid, "ref": it.text, "hyp": hyp, "error": err}, ensure_ascii=False
                )
                + "\n"
            )
    report["output_dir"] = str(out_dir)
    print(render(report))
    return report, 0 if report["status"] == "pass" else 1


def render(report: Dict[str, Any]) -> str:
    empty = report["empty_rate"]
    lines = [
        f"[asrbench] validate {report['backend']['name']} on {report['workload']['dataset']} "
        f"({report['num_items']} utt, normalizer={report['normalizer']})",
        f"  WER {report['wer_pct']:.2f}%  ({report['substitutions']}S {report['deletions']}D "
        f"{report['insertions']}I / {report['ref_words']} words)  "
        f"empty {'n/a' if empty is None else f'{empty:.1%}'}  failed {report['failed']}",
    ]
    ref = report.get("reference")
    if ref:
        lines.append(
            f"  vs reference ({ref['matched']} matched): changed {ref['changed_pct']:.1f}%  "
            f"drift WER {ref['drift_wer_pct']:.2f}%  WER delta {ref['wer_delta_pct']:+.2f} pts"
        )
    for c in report["checks"]:
        lines.append(
            f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']} = {c['value']} (limit {c['limit']})"
        )
    lines.append(f"  status: {report['status'].upper()}  -> {report.get('output_dir', '')}")
    return "\n".join(lines)
