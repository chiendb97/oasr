# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR inference and serving engine."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from oasr.models.conformer.convert import load_wenet_checkpoint

from .config import EngineConfig
from .input_processor import InputProcessor
from .model_runner import ModelRunner
from .output_processor import OutputProcessor
from .request import Request, RequestOutput, RequestState
from .scheduler import Scheduler, SchedulerOutput

logger = logging.getLogger(__name__)


class ASREngine:
    """ASR inference engine with streaming chunk-by-chunk processing.

    Follows a vLLM-inspired three-stage step loop:

    1. **Schedule** — admit waiting requests, collect running ones.
    2. **Forward** — run encoder chunks for all scheduled requests.
    3. **Postprocess** — decode log-probs, finalize completed requests.

    Supports both streaming (chunk-by-chunk with paged KV cache) and
    offline (single-pass) operation through a unified :meth:`transcribe`
    convenience method.

    Parameters
    ----------
    config : EngineConfig
        Fully configured engine settings.  ``ckpt_dir`` must be set.

    Examples
    --------
    Offline convenience (internally uses streaming path)::

        engine = ASREngine(EngineConfig(ckpt_dir="/path/to/ckpt"))
        text = engine.transcribe("audio.wav")

    Explicit streaming loop::

        engine = ASREngine(config)
        rid = engine.add_request("audio.wav")
        results = engine.run()
        text = {r.request_id: r.text for r in results}[rid]
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        device_str = config.device
        dtype = config.dtype

        logger.info("Loading model from %s ...", config.ckpt_dir)
        model, model_config = load_wenet_checkpoint(
            config.ckpt_dir,
            config.checkpoint_name,
            device=device_str,
            dtype=dtype,
        )
        self._model = model
        config._model_config = model_config

        self._device = torch.device(device_str)
        cache_config = config.build_cache_config(model_config)

        self._input_processor = InputProcessor(config, self._device)
        self._scheduler = Scheduler(config)
        self._model_runner = ModelRunner(model, config, cache_config)
        self._output_processor = OutputProcessor(config)

    # ------------------------------------------------------------------
    # Request management
    # ------------------------------------------------------------------

    def add_request(
        self,
        audio: Union[str, torch.Tensor, "np.ndarray"],
        request_id: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> str:
        """Add a new request to the engine.

        Audio features are extracted and pre-chunked immediately so the step
        loop sees only already-computed tensors.

        Parameters
        ----------
        audio : str, Tensor, or ndarray
            Audio input (file path, waveform tensor, or NumPy array).
        request_id : str, optional
            Unique identifier.  Auto-generated if omitted.
        sample_rate : int
            Audio sample rate in Hz.

        Returns
        -------
        str
            The assigned ``request_id``.
        """
        req = Request(audio, request_id=request_id, streaming=True, sample_rate=sample_rate)
        # Pre-process: load audio, extract features, chunk for streaming
        self._input_processor.process_for_streaming(req)
        self._scheduler.add_request(req)
        return req.request_id

    def abort_request(self, request_id: str) -> None:
        """Remove a request from the engine (clean up cache if running).

        Parameters
        ----------
        request_id : str
            The request to abort.
        """
        req = self._scheduler.abort_request(request_id)
        if req is not None and req.stream_context is not None:
            self._model_runner.free_stream(req)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    def step(self) -> List[RequestOutput]:
        """Execute one engine step.

        Each call performs three stages:

        1. **Schedule** — admit new requests, find runnable ones, identify
           completed ones.
        2. **Forward** — run one encoder chunk per scheduled request.
        3. **Postprocess** — feed log-probs to the CTC decoder; finalize and
           clean up requests that have no remaining chunks.

        Returns
        -------
        List[RequestOutput]
            Partial and final outputs produced this step.
        """
        sched: SchedulerOutput = self._scheduler.schedule()
        outputs: List[RequestOutput] = []

        # Admit: allocate cache for newly admitted requests
        for req in sched.newly_admitted:
            self._model_runner.allocate_stream(req)

        # Forward: run encoder chunk for requests that have work to do
        if sched.scheduled_requests:
            log_probs_map: Dict[str, torch.Tensor] = (
                self._model_runner.forward_streaming_step(sched.scheduled_requests)
            )

            # Decode each chunk's log-probs
            for req in sched.scheduled_requests:
                lp = log_probs_map.get(req.request_id)
                if lp is not None:
                    partial = self._output_processor.decode_streaming_chunk(req, lp)
                    outputs.append(partial)

        # Finalize: requests whose chunk queue is now empty
        for req in sched.to_finalize:
            final = self._output_processor.finalize_streaming(req)
            req.output = final
            outputs.append(final)
            self._model_runner.free_stream(req)
            self._scheduler.finish_request(req.request_id)

        return outputs

    def run(self) -> List[RequestOutput]:
        """Run the engine until all pending requests are complete.

        Returns
        -------
        List[RequestOutput]
            All **final** outputs (``finished=True``) in completion order.
        """
        final_outputs: List[RequestOutput] = []
        while self._scheduler.has_pending():
            step_outputs = self.step()
            final_outputs.extend(o for o in step_outputs if o.finished)
        return final_outputs

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, List[str], torch.Tensor, "np.ndarray"],
        sample_rate: int = 16000,
    ) -> Union[str, List[str]]:
        """Transcribe one or more audio inputs using the streaming engine.

        Parameters
        ----------
        audio : str, list, Tensor, or ndarray
            Single or multiple audio inputs.
        sample_rate : int
            Sample rate of the audio (Hz).

        Returns
        -------
        str or List[str]
            Transcribed text(s).
        """
        is_single = not isinstance(audio, list)
        audio_list: List = [audio] if is_single else audio  # type: ignore[list-item]

        request_ids = [
            self.add_request(a, sample_rate=sample_rate) for a in audio_list
        ]
        final = self.run()

        id_to_text: Dict[str, str] = {o.request_id: o.text for o in final}
        texts = [id_to_text.get(rid, "") for rid in request_ids]
        return texts[0] if is_single else texts

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def num_running(self) -> int:
        """Number of currently active streaming requests."""
        return self._scheduler.num_running

    @property
    def num_waiting(self) -> int:
        """Number of requests waiting to be admitted."""
        return self._scheduler.num_waiting
