# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Simple offline (non-streaming) ASR transcription engine."""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import torch

from oasr.models.conformer.convert import load_wenet_checkpoint

from .config import EngineConfig
from .input_processor import InputProcessor
from .output_processor import OutputProcessor
from .request import Request, RequestOutput

logger = logging.getLogger(__name__)


class OfflineEngine:
    """Offline batch transcription engine.

    Loads a Conformer-CTC model and provides a simple
    :meth:`transcribe` method for single files or batches. No scheduler or
    paged cache is needed for offline decoding — the model processes all
    frames in a single forward pass.

    Parameters
    ----------
    config : EngineConfig
        Engine configuration.  ``ckpt_dir`` must be set.

    Examples
    --------
    Single file::

        engine = OfflineEngine(EngineConfig(ckpt_dir="/path/to/ckpt"))
        text = engine.transcribe("audio.wav")

    Batch::

        texts = engine.transcribe(["a.wav", "b.wav", "c.wav"])

    Streaming simulation (chunk-by-chunk with no real-time constraints)::

        texts = engine.transcribe_streaming("long_audio.wav", chunk_size=16)
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
        self._model_config = model_config
        config._model_config = model_config

        self._device = torch.device(device_str)
        self._input_processor = InputProcessor(config, self._device)
        self._output_processor = OutputProcessor(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, List[str], torch.Tensor, "np.ndarray", List[torch.Tensor]],
    ) -> Union[str, List[str]]:
        """Transcribe one or more audio inputs.

        Parameters
        ----------
        audio : str, list, Tensor, or ndarray
            * A single file path (``str``) — returns a single ``str``.
            * A list of file paths or waveform tensors — returns a list.
            * A single waveform tensor ``(T,)`` or ``(1, T)`` — returns ``str``.
            * A NumPy array — returns ``str``.

        Returns
        -------
        str or List[str]
            Transcribed text.  Returns a list when the input is a list,
            a plain string otherwise.
        """
        is_single = not isinstance(audio, list)
        audio_list: List = [audio] if is_single else audio  # type: ignore[list-item]

        requests = [Request(a) for a in audio_list]
        outputs = self._run_batch(requests)
        texts = [o.text for o in outputs]
        return texts[0] if is_single else texts

    def transcribe_streaming(
        self,
        audio: Union[str, torch.Tensor, "np.ndarray"],
        chunk_size: int = None,  # type: ignore[assignment]
    ) -> str:
        """Simulate streaming by processing audio chunk-by-chunk.

        Uses :meth:`~oasr.models.conformer.ConformerModel.forward_chunk_by_chunk`
        internally — no paged cache.  Useful for measuring streaming accuracy
        without real-time latency constraints.

        Parameters
        ----------
        audio : str, Tensor, or ndarray
            Audio to transcribe.
        chunk_size : int, optional
            Encoder output frames per chunk.  Defaults to
            ``config.chunk_size``.

        Returns
        -------
        str
            Transcribed text.
        """
        chunk_size = chunk_size or self._config.chunk_size

        req = Request(audio)
        features, _ = self._input_processor.process_offline([req])

        with torch.no_grad():
            log_probs = self._model.forward_chunk_by_chunk(
                features,
                decoding_chunk_size=chunk_size,
                num_decoding_left_chunks=self._config.num_left_chunks,
            )  # (1, T_out, V)

        # Compute output length (encoder output T) for the decoder
        T_out = log_probs.size(1)
        lengths = torch.tensor([T_out], dtype=torch.int32, device=self._device)

        outputs = self._output_processor.decode_offline(log_probs, lengths)
        return outputs[0].text if outputs else ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_batch(self, requests: List[Request]) -> List[RequestOutput]:
        """Load features and run a full forward pass for a batch of requests."""
        features, feat_lengths = self._input_processor.process_offline(requests)

        with torch.no_grad():
            # Get encoder hidden states and masks (True=valid)
            hidden, masks = self._model.encoder(features, feat_lengths)
            log_probs = self._model.ctc(hidden)  # (B, T_out, V)

        # Extract valid output lengths from masks: (B, 1, T_out) True=valid
        output_lengths = masks.squeeze(1).sum(dim=-1).to(torch.int32)

        outputs = self._output_processor.decode_offline(log_probs, output_lengths)

        # Stamp request IDs onto outputs
        for req, out in zip(requests, outputs):
            out.request_id = req.request_id
            req.output = out

        return outputs

