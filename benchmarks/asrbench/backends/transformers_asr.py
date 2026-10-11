# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face ``transformers`` ASR pipeline -- the reference implementation.

Most checkpoints the repo serves (Whisper, Qwen2-Audio, Parakeet/Nemotron-style
CTC/RNN-T, wav2vec2) have a ``transformers`` implementation, which makes this the
natural *known-good* side of a correctness comparison
(``validate --reference``) and a baseline for batch throughput.

It is a library, not a server: one call at a time, with a native batch
argument.  So it declares ``batch`` and refuses ``concurrent`` -- a load test
against it would measure a Python queue in front of a sequential model, not a
scheduler.  Long inputs are decoded only when ``chunk_length_s`` turns on the
pipeline's chunked mode (declared ``windowed``); otherwise ``long_form`` is
``None`` and the long-form benchmark reports ``unsupported``.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.asrbench.backends.base import (
    Backend,
    BackendInfo,
    Capabilities,
    RequestFailed,
    Transcript,
)
from benchmarks.asrbench.workloads import WorkItem


class TransformersBackend(Backend):
    name = "transformers"
    description = "Hugging Face transformers ASR pipeline (reference implementation)"
    OPTIONS = {
        "model": "HF model id or local directory (default: $WHISPER_CKPT)",
        "dtype": "float16 | bfloat16 | float32 (float16)",
        "device": "torch device (cuda:0)",
        "chunk_length_s": "enable the pipeline's chunked long-form decoding (seconds)",
        "generate_kwargs": "YAML mapping for generate(), e.g. {language: en, task: transcribe}",
        "max_batch_size": "widest batch accepted (64)",
        "attn_implementation": "sdpa | eager | flash_attention_2 (library default)",
    }

    def __init__(self, options: Optional[Dict[str, Any]] = None, mode: str = "offline") -> None:
        super().__init__(options, mode)
        self.pipe: Any = None
        self.model_path = str(self.options.get("model") or os.environ.get("WHISPER_CKPT") or "")

    @classmethod
    def available(cls) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return False, f"needs torch + transformers ({exc})"
        return True, ""

    @classmethod
    def mode_unsupported(cls, mode: str, options: Dict[str, Any]) -> Optional[str]:
        return (
            "the transformers pipeline has no streaming interface" if mode == "streaming" else None
        )

    async def load(self) -> None:
        import torch
        from transformers import pipeline

        if not self.model_path:
            raise ValueError("transformers needs -o model=<id or path> (or $WHISPER_CKPT)")
        dtype = getattr(torch, str(self.options.get("dtype", "float16")))
        model_kwargs = {}
        if self.options.get("attn_implementation"):
            model_kwargs["attn_implementation"] = self.options["attn_implementation"]
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_path,
            dtype=dtype,
            device=str(self.options.get("device", "cuda:0")),
            model_kwargs=model_kwargs or None,
        )

    async def close(self) -> None:
        self.pipe = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> BackendInfo:
        import torch
        import transformers

        from benchmarks.asrbench.backends.oasr_engine import checkpoint_fingerprint

        cfg = self.pipe.model.config
        model = (
            checkpoint_fingerprint(self.model_path)
            if os.path.exists(self.model_path)
            else {"id": self.model_path}
        )
        model.update(
            {
                "id": os.path.basename(self.model_path.rstrip("/")) or self.model_path,
                "architecture": (cfg.architectures or [type(self.pipe.model).__name__])[0],
                "revision": model.get("revision") or getattr(cfg, "_commit_hash", None),
            }
        )
        return BackendInfo(
            name=self.name,
            version=transformers.__version__,
            mode=self.mode,
            model=model,
            config={
                "chunk_length_s": self.options.get("chunk_length_s"),
                "generate_kwargs": dict(self.options.get("generate_kwargs") or {}),
                "attn_implementation": getattr(cfg, "_attn_implementation", None),
                "torch": torch.__version__,
            },
            device=str(self.pipe.device),
            dtype=str(self.pipe.model.dtype).replace("torch.", ""),
        )

    def capabilities(self) -> Capabilities:
        return Capabilities(
            offline=True,
            batch=True,
            max_batch_size=int(self.options.get("max_batch_size", 64)),
            concurrent=False,
            streaming=None,
            word_timestamps=False,
            long_form="windowed" if self.options.get("chunk_length_s") else None,
            in_process=True,
            notes="library call, one at a time; no request scheduler",
        )

    def prepare(self, audio: np.ndarray, item: WorkItem) -> Dict[str, Any]:
        return {
            "raw": np.ascontiguousarray(audio, dtype=np.float32),
            "sampling_rate": self.sample_rate,
        }

    def _call_kwargs(self, batch_size: int) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"batch_size": batch_size}
        if self.options.get("chunk_length_s"):
            kw["chunk_length_s"] = float(self.options["chunk_length_s"])
        if self.options.get("generate_kwargs"):
            kw["generate_kwargs"] = dict(self.options["generate_kwargs"])
        return kw

    def _run(self, inputs: List[Dict[str, Any]]) -> Tuple[List[str], float]:
        import torch

        # The pipeline mutates its input dicts; hand it copies.
        out = self.pipe([dict(x) for x in inputs], **self._call_kwargs(len(inputs)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_done = time.perf_counter()
        return [o.get("text", "") for o in out], t_done

    async def transcribe(self, payload: Any) -> Transcript:
        t_submit = time.perf_counter()
        try:
            texts, t_done = await asyncio.to_thread(self._run, [payload])
        except Exception as exc:  # noqa: BLE001 -- e.g. input longer than the window
            raise RequestFailed(f"{type(exc).__name__}: {exc}") from None
        return Transcript(text=texts[0].strip(), t_submit=t_submit, t_done=t_done)

    async def transcribe_batch(self, payloads: Sequence[Any]) -> List[Transcript]:
        t_submit = time.perf_counter()
        try:
            texts, t_done = await asyncio.to_thread(self._run, list(payloads))
        except Exception as exc:  # noqa: BLE001
            raise RequestFailed(f"{type(exc).__name__}: {exc}") from None
        return [Transcript(text=t.strip(), t_submit=t_submit, t_done=t_done) for t in texts]

    async def synchronize(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_peak_memory(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_memory(self) -> Dict[str, float]:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_alloc_mb.max": torch.cuda.max_memory_allocated() / 2**20,
            "gpu_reserved_mb.max": torch.cuda.max_memory_reserved() / 2**20,
        }
