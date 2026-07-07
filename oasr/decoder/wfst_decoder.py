# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GPU WFST beam search backed by the standalone ``wfst`` decoder.

Drop-in replacement for the k2-based WFST path in :mod:`oasr.decode`: exposes the same
duck-typed searcher protocol (``reset`` / ``search`` / ``finalize_search`` +
``outputs`` / ``likelihood`` / ``times``) plus a batched-exact ``decode_offline`` fast
path. Log-probs stay on the GPU end to end (``wants_device_tensor = True``).

The heavyweight state — graph image and GPU decoder instances — is shared process-wide
through a cache keyed by (graph, options, device): offline strategies construct
:class:`oasr.decode.Decoder` per call and streaming holds one per request, so per-object
construction must stay cheap. Streaming searchers borrow a channel from one shared
multi-channel decoder and release it on ``finalize_search`` (or GC).

Graphs: pass either a prebuilt ``.img`` (built by
:mod:`oasr.decoder.wfst.graph_export`) or a k2 ``HLG.pt`` — the latter is exported once
and cached next to the source file. The CUDA decoder itself is the in-tree
``oasr._C.decoder`` module (built with ``OASR_USE_WFST_DECODER=1``); no external ``wfst``
checkout is required.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import List, Optional

import torch

_lock = threading.Lock()
_lib = None
_graphs: dict = {}
_offline: dict = {}
_streaming: dict = {}

_OFFLINE_MAX_LANES = 8
_STREAM_CHUNK_FRAMES = 128


def _get_lib():
    """Return the in-tree GPU WFST decoder module (``oasr._C.decoder``).

    Raises if oasr was not built with the decoder (``OASR_USE_WFST_DECODER=1``).
    """
    global _lib
    if _lib is None:
        import oasr._C as _C

        decoder = _C.decoder
        if not getattr(decoder, "wfst_decoder_available", False):
            raise RuntimeError(
                "the in-tree GPU WFST decoder is not available; reinstall oasr with "
                "OASR_USE_WFST_DECODER=1 to enable it"
            )
        _lib = decoder
    return _lib


def _resolve_graph_image(fst: str) -> str:
    """Return a .img path for *fst*, exporting a k2 ``.pt`` once if needed."""
    path = Path(fst)
    if path.suffix == ".img":
        return str(path)
    cache = path.with_suffix(path.suffix + ".wfst.img")
    if not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
        from oasr.decoder.wfst.graph_export import export_hlg

        export_hlg(str(path), str(cache))
    return str(cache)


def _get_graph(fst: str):
    with _lock:
        if fst not in _graphs:
            lib = _get_lib()
            _graphs[fst] = lib.load_graph(_resolve_graph_image(fst))
        return _graphs[fst]


def _decoder_key(fst: str, opts: "WfstDecoderOptions", device: int) -> tuple:
    return (
        fst,
        opts.search_beam,
        opts.output_beam,
        opts.min_active_states,
        opts.max_active_states,
        device,
    )


class WfstDecoderOptions:
    """Options mirroring the k2 path's ``CtcWfstBeamSearchOptions`` names."""

    def __init__(
        self,
        blank: int = 0,
        search_beam: float = 20.0,
        output_beam: float = 8.0,
        min_active_states: int = 30,
        max_active_states: int = 10000,
        blank_skip_thresh: float = 0.98,
        max_frames: int = 4096,
        max_streams: int = 32,
        max_offline_lanes: int = _OFFLINE_MAX_LANES,
    ) -> None:
        self.blank = blank
        self.search_beam = search_beam
        self.output_beam = output_beam
        self.min_active_states = min_active_states
        self.max_active_states = max_active_states
        self.blank_skip_thresh = blank_skip_thresh
        self.max_frames = max_frames
        self.max_streams = max_streams
        # Lane pool of the shared offline decoder; a batched decode of B utterances
        # fills up to this many lanes per GPU launch (larger B is sub-batched).
        self.max_offline_lanes = max_offline_lanes


def _get_offline_decoder(fst: str, opts: WfstDecoderOptions, device: int):
    key = _decoder_key(fst, opts, device) + (opts.max_frames, opts.max_offline_lanes)
    graph = _get_graph(fst)  # takes _lock itself; resolve before re-acquiring
    with _lock:
        if key not in _offline:
            lib = _get_lib()
            _offline[key] = lib.GpuDecoder(
                graph,
                search_beam=opts.search_beam,
                output_beam=opts.output_beam,
                min_active=opts.min_active_states,
                max_active=opts.max_active_states,
                allow_partial=True,
                max_lanes=opts.max_offline_lanes,
                max_frames=opts.max_frames,
                device=device,
                main_q_factor=32,
                cand_factor=3,
            )
        return _offline[key]


def _get_streaming_decoder(fst: str, opts: WfstDecoderOptions, device: int):
    key = _decoder_key(fst, opts, device) + (opts.max_streams,)
    graph = _get_graph(fst)  # takes _lock itself; resolve before re-acquiring
    with _lock:
        if key not in _streaming:
            lib = _get_lib()
            _streaming[key] = lib.GpuDecoder(
                graph,
                search_beam=opts.search_beam,
                output_beam=opts.output_beam,
                min_active=opts.min_active_states,
                max_active=opts.max_active_states,
                allow_partial=True,
                max_lanes=opts.max_streams,
                max_frames=_STREAM_CHUNK_FRAMES,
                device=device,
                main_q_factor=16,
                cand_factor=4,
                streaming=True,
            )
        return _streaming[key]


class WfstDecoderSearch:
    """Searcher-protocol adapter over the wfst GPU decoder.

    Offline (``decode_offline``) runs the exact offline beam semantics in one batched
    call; the streaming protocol (``reset`` / ``search`` / ``finalize_search``) maps to
    a channel of the shared streaming decoder (k2-online beam semantics, identical to
    the previous OnlineDenseIntersecter flavor).
    """

    wants_device_tensor = True

    def __init__(self, fst: str, opts: WfstDecoderOptions) -> None:
        self._fst = fst
        self._opts = opts
        self._channel: Optional[int] = None
        self._stream_dec = None
        self._device: Optional[int] = None
        self._outputs: List[List[int]] = []
        self._likelihood: List[float] = []

    # -- searcher protocol -------------------------------------------------

    @property
    def outputs(self) -> List[List[int]]:
        return self._outputs

    @property
    def likelihood(self) -> List[float]:
        return self._likelihood

    @property
    def times(self) -> List[List[int]]:
        return []

    def reset(self) -> None:
        self._release()
        self._outputs = []
        self._likelihood = []

    def search(self, logp: torch.Tensor) -> None:
        """Streaming: advance this request's channel by one chunk of frames."""
        logp = self._prepare(logp)
        if logp.size(0) == 0:
            return
        if self._channel is None:
            self._device = logp.device.index or 0
            self._stream_dec = _get_streaming_decoder(self._fst, self._opts, self._device)
            self._channel = self._stream_dec.create_stream()
            if self._channel < 0:
                raise RuntimeError(
                    f"no free WFST stream channels (max_streams={self._opts.max_streams})"
                )
        partial = None
        for start in range(0, logp.size(0), _STREAM_CHUNK_FRAMES):
            piece = logp[start : start + _STREAM_CHUNK_FRAMES].unsqueeze(0).contiguous()
            out = self._stream_dec.advance_chunk(
                [self._channel], piece, torch.tensor([piece.size(1)]), partial=True
            )
            partial = out[0]
        if partial is not None:
            self._outputs = [list(partial["words"])]
            self._likelihood = [0.0]

    def finalize_search(self) -> None:
        if self._channel is None:
            return
        result = self._stream_dec.finalize_stream(self._channel)
        self._release()
        self._outputs = [list(result["words"])] if result["ok"] else [[]]
        self._likelihood = [float(result["score"])] if result["ok"] else [0.0]

    # -- offline fast path ---------------------------------------------------

    def decode_offline(self, logp: torch.Tensor):
        """Exact offline decode of one utterance; returns (tokens, scores)."""
        logp = self._prepare(logp)
        device = logp.device.index or 0
        dec = _get_offline_decoder(self._fst, self._opts, device)
        out = dec.decode_batch(logp.unsqueeze(0).contiguous(), torch.tensor([logp.size(0)]))[0]
        if out["ok"]:
            return [list(out["words"])], [float(out["score"])]
        return [[]], [0.0]

    def decode_offline_batch(self, enc_out: torch.Tensor, enc_lengths):
        """Exact offline decode of a padded ``[B, T, V]`` batch; returns (tokens, scores).

        The batched equivalent of calling :meth:`decode_offline` on each row: every row
        is length-clipped and blank-skipped exactly as in the single-utterance path, then
        rows are grouped into ``<= max_offline_lanes`` padded sub-batches and decoded with
        one ``decode_batch`` GPU launch per group.  This preserves the original decoder's
        batched throughput (the headline perf lever) while producing per-row results
        identical to the one-at-a-time path.  ``tokens[b]`` is the best-path word list;
        ``scores[b]`` its total score (0.0 for a lane that produced nothing).
        """
        if enc_out.dim() != 3:
            raise ValueError(f"enc_out must be [B, T, V], got {enc_out.dim()}-D")
        batch = enc_out.size(0)
        if torch.is_tensor(enc_lengths):
            lengths = [int(x) for x in enc_lengths.detach().cpu().tolist()]
        else:
            lengths = [int(x) for x in enc_lengths]
        tokens: List[List[int]] = [[] for _ in range(batch)]
        scores: List[float] = [0.0 for _ in range(batch)]
        if batch == 0:
            return tokens, scores
        device = enc_out.device.index or 0
        dec = _get_offline_decoder(self._fst, self._opts, device)
        lanes = max(1, self._opts.max_offline_lanes)
        for start in range(0, batch, lanes):
            stop = min(start + lanes, batch)
            # _prepare handles device/dtype/contiguity + WeNet blank-skip, per row.
            rows = [self._prepare(enc_out[b, : lengths[b], :]) for b in range(start, stop)]
            max_t = max((r.size(0) for r in rows), default=0)
            if max_t == 0:
                continue  # every row emptied by blank-skip / zero length -> defaults
            # Left-aligned, log-space-padded batch; the padded tail is ignored because
            # each lane's frame count bounds its decode.
            packed = rows[0].new_full((len(rows), max_t, rows[0].size(1)), -30.0)
            lens = []
            for i, r in enumerate(rows):
                t = r.size(0)
                if t > 0:
                    packed[i, :t] = r
                lens.append(t)
            outs = dec.decode_batch(packed.contiguous(), torch.tensor(lens, dtype=torch.int32))
            for i, out in enumerate(outs):
                if out["ok"]:
                    tokens[start + i] = list(out["words"])
                    scores[start + i] = float(out["score"])
        return tokens, scores

    # -- helpers ---------------------------------------------------------------

    def _prepare(self, logp: torch.Tensor) -> torch.Tensor:
        if logp.dim() != 2:
            raise ValueError(f"logp must be [T, V], got {logp.dim()}-D")
        if not logp.is_cuda:
            logp = logp.cuda()
        if logp.dtype != torch.float32:
            logp = logp.float()
        # WeNet-style blank skip: drop frames dominated by blank BEFORE the search
        # (matches the previous k2 path's frame pre-slicing exactly).
        thresh = self._opts.blank_skip_thresh
        if 0.0 < thresh < 1.0 and logp.size(0) > 0:
            keep = logp[:, self._opts.blank] <= math.log(thresh)
            if not bool(keep.all()):
                logp = logp[keep]
        return logp.contiguous()

    def _release(self) -> None:
        if self._channel is not None and self._stream_dec is not None:
            self._stream_dec.release_stream(self._channel)
        self._channel = None

    def __del__(self) -> None:  # channel safety net for aborted requests
        try:
            self._release()
        except Exception:
            pass
