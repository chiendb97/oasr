# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""OASR's ``ASREngine``, in process -- the GPU + Python ceiling, no transport.

The engine is a step loop: requests are admitted, and each ``step()`` advances
every in-flight request and returns the outputs it finished (or, streaming, the
partials it produced).  :class:`_Pump` drives that loop from the benchmark's
event loop exactly the way ``oasr-server``'s dispatcher does
(``rust/crates/oasr-engine-client/src/dispatcher.rs``): step while there is work,
block on new work when idle, and after a sub-millisecond tick that produced
nothing, wait up to 2 ms for new work instead of spinning.  Requests that arrive
during a step are admitted after it -- the same quantisation a served request
sees -- and every completion is timestamped the moment ``step()`` returns it.

Waveforms are handed over in page-locked buffers from
``ASREngine.new_audio_buffer`` (``pinned_audio=true``, the default), which is
the path a served request takes after the codec; heap tensors measure the
packing path no front-end uses.

Engines are single-mode: the suite builds one instance per service mode.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.asrbench.backends.base import (
    Backend,
    BackendInfo,
    Capabilities,
    RequestFailed,
    StreamEvent,
    StreamSession,
    Transcript,
    Unsupported,
)
from benchmarks.asrbench.workloads import WorkItem

#: Mirrors the dispatcher's NO_WORK_BACKOFF / NO_WORK_TICK_MAX.
NO_WORK_BACKOFF_S = 0.002
NO_WORK_TICK_MAX_S = 0.001

_DTYPES = ("float16", "bfloat16", "float32")


def checkpoint_fingerprint(path: str) -> Dict[str, Any]:
    """Identify a checkpoint without hashing gigabytes: names, sizes, mtimes.

    An HF snapshot path carries its commit (``snapshots/<sha>``), which is
    recorded as the revision when present.
    """
    p = Path(path)
    out: Dict[str, Any] = {"path": str(p)}
    if not p.exists():
        return out
    files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    h = hashlib.sha256()
    total = 0
    for f in files[:2000]:
        st = f.stat()
        total += st.st_size
        h.update(
            f"{f.relative_to(p) if p.is_dir() else f.name}:{st.st_size}:{int(st.st_mtime)}".encode()
        )
    out["fingerprint"] = h.hexdigest()[:16]
    out["bytes"] = total
    parts = p.resolve().parts
    if "snapshots" in parts:
        out["revision"] = parts[parts.index("snapshots") + 1]
    return out


def _quantization_config(cfg: Any) -> Optional[Dict[str, Any]]:
    """``EngineConfig.quantization`` as JSON: ``{scheme, ignore}`` or ``None``."""
    q = getattr(cfg, "quantization", None)
    if q is None:
        return None
    return {"scheme": q.scheme, "ignore": list(q.ignore)}


class _Pump:
    """Steps the engine; resolves offline futures and feeds stream sessions."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.waiters: Dict[str, "asyncio.Future[Tuple[Any, float]]"] = {}
        self.streams: Dict[str, "_EngineStream"] = {}
        self.work = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.closed = False
        self.error: Optional[BaseException] = None

    def start(self) -> None:
        self.task = asyncio.get_running_loop().create_task(self._run())

    def kick(self) -> None:
        self.work.set()

    def _busy(self) -> bool:
        if self.waiters or self.streams:
            return True
        return self.engine.num_running + self.engine.num_waiting > 0

    async def _run(self) -> None:
        try:
            while not self.closed:
                if not self._busy():
                    self.work.clear()
                    await self.work.wait()
                    continue
                self.work.clear()
                t0 = time.perf_counter()
                outputs = self.engine.step()
                t1 = time.perf_counter()
                for out in outputs:
                    self._dispatch(out, t1)
                if not outputs and (t1 - t0) < NO_WORK_TICK_MAX_S and not self.work.is_set():
                    try:
                        await asyncio.wait_for(self.work.wait(), NO_WORK_BACKOFF_S)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0)
        except BaseException as exc:  # noqa: BLE001 -- surface to every waiter
            self.error = exc
            for fut in self.waiters.values():
                if not fut.done():
                    fut.set_exception(RequestFailed(f"engine step failed: {exc!r}"))
            for s in self.streams.values():
                s.push(StreamEvent("error", time.perf_counter(), f"engine step failed: {exc!r}"))
            if not isinstance(exc, (asyncio.CancelledError, Exception)):
                raise

    def _dispatch(self, out: Any, t: float) -> None:
        rid = out.request_id
        failed = getattr(out, "finish_reason", None) == "error"
        stream = self.streams.get(rid)
        if stream is not None:
            if failed:
                stream.push(
                    StreamEvent("error", t, f"engine error ({getattr(out, 'error_stage', '')})")
                )
                self.streams.pop(rid, None)
            elif out.finished:
                stream.push(StreamEvent("final", t, out.text or "", _words(out)))
                self.streams.pop(rid, None)
            else:
                stream.push(StreamEvent("partial", t, out.text or ""))
            return
        fut = self.waiters.get(rid)
        if fut is None or not out.finished:
            return
        self.waiters.pop(rid, None)
        if fut.done():
            return
        if failed:
            fut.set_exception(
                RequestFailed(f"engine error at stage {getattr(out, 'error_stage', '?')}")
            )
        else:
            fut.set_result((out, t))

    async def close(self) -> None:
        self.closed = True
        self.kick()
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


def _words(out: Any) -> Optional[List[Dict[str, Any]]]:
    words = getattr(out, "words", None)
    if not words:
        return None
    return [
        {
            "word": getattr(w, "word", ""),
            "start": float(getattr(w, "start", 0.0)),
            "end": float(getattr(w, "end", 0.0)),
        }
        for w in words
    ]


class _EngineStream(StreamSession):
    def __init__(self, backend: "OasrEngineBackend", rid: str) -> None:
        super().__init__()
        self.backend = backend
        self.rid = rid

    async def send(self, chunk: Any, is_last: bool) -> None:
        self.backend.engine.feed_chunk(self.rid, chunk, is_last=is_last)
        self.backend.pump.kick()

    async def abort(self) -> None:
        self.backend.pump.streams.pop(self.rid, None)
        try:
            self.backend.engine.abort_request(self.rid)
        except Exception:  # noqa: BLE001 -- already finished
            pass


class OasrEngineBackend(Backend):
    name = "oasr_engine"
    description = "OASR ASREngine in process (no transport): the engine ceiling"
    OPTIONS = {
        "ckpt_dir": "checkpoint directory (default: $CKPT_DIR)",
        "architecture": "force a registered architecture (e.g. transducer for icefall RNN-T)",
        "dtype": "float16 | bfloat16 | float32 (bfloat16)",
        "device": "cuda device (cuda)",
        "max_batch_size": "engine micro-batch width / streaming pool (engine default)",
        "chunk_size": "streaming encoder chunk, frames (model default)",
        "num_left_chunks": "streaming left context, chunks (-1 = unlimited)",
        "decode_method": "decode family, e.g. ctc_aed_rescoring (model default)",
        "decode_options": "per-family decode options, YAML mapping",
        "use_cuda_graphs": "true | false (engine default)",
        "long_form": "window over-length audio for fixed-window frontends (false)",
        "vad": "VadConfig mapping, e.g. {mode: segment, backend: silero}",
        "quantization": "online GEMM quantization: a scheme (int8_w8a8) or "
        "{scheme: int8_w8a8, ignore: 'glob,glob'} (none; a quantized checkpoint "
        "is served quantized regardless)",
        "word_timestamps": "request word timings on offline requests (false)",
        "pinned_audio": "hand the engine page-locked waveforms, as the server does (true)",
        "engine": "any other EngineConfig fields, YAML mapping",
    }

    def __init__(self, options: Optional[Dict[str, Any]] = None, mode: str = "offline") -> None:
        super().__init__(options, mode)
        self.engine: Any = None
        self.pump: _Pump = None  # type: ignore[assignment]
        self.config: Any = None

    @classmethod
    def available(cls) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401

            from oasr.engine import ASREngine  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot import oasr.engine ({exc})"
        return True, ""

    def _engine_kwargs(self) -> Dict[str, Any]:
        import torch

        o = self.options
        ckpt = o.get("ckpt_dir") or os.environ.get("CKPT_DIR")
        if not ckpt:
            raise ValueError("oasr_engine needs -o ckpt_dir=... (or $CKPT_DIR)")
        dtype = str(o.get("dtype", "bfloat16"))
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}")
        kw: Dict[str, Any] = {
            "ckpt_dir": str(ckpt),
            "device": str(o.get("device", "cuda")),
            "dtype": getattr(torch, dtype),
            "service_mode": self.mode,
        }
        for key in (
            "architecture",
            "max_batch_size",
            "chunk_size",
            "num_left_chunks",
            "decode_method",
            "use_cuda_graphs",
            "long_form",
            "vad",
            "quantization",
        ):
            if o.get(key) is not None:
                kw[key] = o[key]
        if o.get("decode_options"):
            kw["decode_options"] = dict(o["decode_options"])
        kw.update(dict(o.get("engine") or {}))
        if self.mode == "streaming" and "chunk_size" in kw:
            from oasr.engine import EngineConfig

            # One paged block holds one encoder chunk (see the engine harness).
            kw.setdefault(
                "block_size_frames", max(EngineConfig.block_size_frames, int(kw["chunk_size"]))
            )
        return kw

    async def load(self) -> None:
        from oasr.engine import ASREngine, EngineConfig

        self.config = EngineConfig(**self._engine_kwargs())
        self.engine = ASREngine(self.config)
        if int(self.engine.sample_rate) != self.sample_rate:
            raise ValueError(
                f"model expects {self.engine.sample_rate} Hz; run the profile with "
                f"defaults.sample_rate={self.engine.sample_rate}"
            )
        self.pump = _Pump(self.engine)
        self.pump.start()

    async def close(self) -> None:
        if self.pump is not None:
            await self.pump.close()
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self.engine = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def info(self) -> BackendInfo:
        import torch

        from benchmarks.asrbench.environment import packages

        cfg = self.config
        model = checkpoint_fingerprint(cfg.ckpt_dir)
        model_obj = getattr(self.engine, "_model", None)
        model.update(
            {
                "id": Path(cfg.ckpt_dir).name,
                "architecture": cfg.architecture or type(model_obj).__name__,
                "decode_method": self.engine.decode_method,
                "capabilities": self.engine.capabilities,
            }
        )
        dev = {}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            dev = {"name": props.name, "sm": f"{props.major}.{props.minor}"}
        return BackendInfo(
            name=self.name,
            version=packages().get("oasr", ""),
            mode=self.mode,
            model=model,
            config={
                "service_mode": cfg.service_mode,
                "max_batch_size": cfg.max_batch_size,
                "chunk_size": cfg.chunk_size if self.mode == "streaming" else None,
                "use_cuda_graphs": cfg.use_cuda_graphs,
                "long_form": cfg.long_form,
                "schedule_policy": cfg.schedule_policy,
                "decode_options": dict(cfg.decode_options or {}),
                "vad": str(cfg.vad) if cfg.vad else None,
                # Asked for, and served: a checkpoint written by `oasr quantize`
                # is quantized with no option, so only the second tells such a
                # run apart from a full-precision one.
                "quantization": _quantization_config(cfg),
                "quantization_info": self.engine.quantization_info(),
                "pinned_audio": bool(self.options.get("pinned_audio", True)),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": dev,
            },
            device=str(cfg.device),
            dtype=str(cfg.dtype).replace("torch.", ""),
        )

    def capabilities(self) -> Capabilities:
        strategy = self.engine._output_processor.strategy
        timing_modes = tuple(getattr(strategy, "word_timing_modes", ()) or ())
        window = self.config.feature_config.fixed_window_seconds
        if window is None:
            long_form = "whole"
        else:
            long_form = "windowed" if (self.config.long_form or self.config.vad) else None
        streaming = self.mode == "streaming"
        return Capabilities(
            offline=not streaming,
            batch=not streaming,
            max_batch_size=int(self.config.max_batch_size),
            concurrent=True,
            streaming="native" if streaming else None,
            partials=streaming,
            word_timestamps=self.mode in timing_modes,
            long_form=long_form if not streaming else None,
            in_process=True,
            server_metrics=True,
            notes=(
                "batch = one engine admission of all inputs; the scheduler may split it "
                "(see batch_size_observed). long_form=whole means no fixed-window "
                "frontend, not an unbounded input: full attention's memory grows with "
                "the square of the length, so a long enough request is rejected for "
                "memory, which the longform benchmark reports per request"
            ),
        )

    # -- inputs ---------------------------------------------------------------------

    def prepare(self, audio: np.ndarray, item: WorkItem) -> Any:
        import torch

        if self.options.get("pinned_audio", True):
            buf = self.engine.new_audio_buffer(int(audio.size))
            if buf is not None:
                buf.numpy()[:] = audio
                return buf
        return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))

    def prepare_chunk(self, chunk: np.ndarray) -> Any:
        import torch

        return torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))

    # -- inference ------------------------------------------------------------------

    def _decoding(self) -> Optional[Dict[str, Any]]:
        return {"word_timestamps": True} if self.options.get("word_timestamps") else None

    def _admit(self, payload: Any) -> Tuple[str, "asyncio.Future[Tuple[Any, float]]"]:
        if self.pump.error is not None:
            raise RequestFailed(f"engine pump died: {self.pump.error!r}")
        rid = self.engine.add_request(payload, streaming=False, decoding=self._decoding())
        fut: "asyncio.Future[Tuple[Any, float]]" = asyncio.get_running_loop().create_future()
        self.pump.waiters[rid] = fut
        self.pump.kick()
        return rid, fut

    async def transcribe(self, payload: Any) -> Transcript:
        if self.mode != "offline":
            raise Unsupported("this engine instance is streaming-mode")
        t_submit = time.perf_counter()
        try:
            rid, fut = self._admit(payload)
        except ValueError as exc:  # admission refusal (too long for a fixed window, ...)
            raise RequestFailed(str(exc)) from None
        try:
            out, t_done = await fut
        except asyncio.CancelledError:
            self.pump.waiters.pop(rid, None)
            try:
                self.engine.abort_request(rid)
            except Exception:  # noqa: BLE001
                pass
            raise
        return Transcript(text=out.text or "", t_submit=t_submit, t_done=t_done, words=_words(out))

    async def transcribe_batch(self, payloads: Sequence[Any]) -> List[Transcript]:
        if self.mode != "offline":
            raise Unsupported("this engine instance is streaming-mode")
        t_submit = time.perf_counter()
        futs = []
        for p in payloads:
            try:
                futs.append(self._admit(p)[1])
            except ValueError as exc:
                raise RequestFailed(str(exc)) from None
        outs = await asyncio.gather(*futs)
        return [
            Transcript(text=o.text or "", t_submit=t_submit, t_done=t, words=_words(o))
            for o, t in outs
        ]

    async def open_stream(self, *, word_timestamps: bool = False) -> StreamSession:
        if self.mode != "streaming":
            raise Unsupported("this engine instance is offline-mode")
        decoding = {"word_timestamps": True} if word_timestamps else None
        rid = self.engine.add_streaming_request(decoding=decoding)
        session = _EngineStream(self, rid)
        self.pump.streams[rid] = session
        self.pump.kick()
        return session

    async def cancel(self, payload: Any) -> None:
        return None  # transcribe() aborts on cancellation itself

    # -- observability ---------------------------------------------------------------

    async def synchronize(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_peak_memory(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def peak_memory(self) -> Dict[str, float]:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_alloc_mb.max": torch.cuda.max_memory_allocated() / 2**20,
            "gpu_reserved_mb.max": torch.cuda.max_memory_reserved() / 2**20,
        }

    def server_metrics_begin(self) -> None:
        if self.engine is not None:
            # Draining restarts the histograms at the window; counters are
            # absolute, so remember the drop counter to diff it at the end.
            snap = self.engine.metrics_snapshot()
            self._dropped0 = float(
                (snap.get("counters") or {}).get("oasr_engine_metric_samples_dropped_total", 0.0)
            )

    def server_metrics_end(self) -> Dict[str, Any]:
        from benchmarks.asrbench.metrics import summarize

        snap = self.engine.metrics_snapshot()
        out: Dict[str, Any] = {}
        hist = snap.get("hist", {}) or {}
        keyed = snap.get("keyed_hist", {}) or {}

        def samples(name: str) -> List[float]:
            vals: List[float] = list(hist.get(name, []) or [])
            for series in (keyed.get(name) or {}).values():
                vals.extend(series or [])
            return vals

        q = samples("oasr_engine_queue_wait_seconds")
        if q:
            out.update(summarize((v * 1e3 for v in q), "queue_wait_ms"))
        b = samples("oasr_engine_batch_size")
        if b:
            stats = summarize(b, "batch_size_observed")
            out.update(
                {
                    k: v
                    for k, v in stats.items()
                    if k.endswith((".n", ".mean", ".min", ".max", ".p50"))
                }
            )
        pad = samples("oasr_engine_batch_padding_ratio")
        if pad:
            # The engine's definition: padded fraction of each padded batch.
            out["engine_pad_fraction.mean"] = sum(pad) / len(pad)
        dropped = float(
            (snap.get("counters") or {}).get("oasr_engine_metric_samples_dropped_total", 0.0)
        ) - getattr(self, "_dropped0", 0.0)
        if dropped > 0:
            out["server_samples_dropped"] = dropped
        stages = keyed.get("oasr_engine_stage_host_seconds") or {}
        if stages:
            out["stage_host_ms"] = {
                str(k): round(1e3 * sum(v) / max(1, len(v)), 4) for k, v in stages.items() if v
            }
        return out
