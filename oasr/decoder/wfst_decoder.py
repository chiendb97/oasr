# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GPU WFST beam search over the in-tree CUDA decoder (JIT-compiled).

Drop-in replacement for the k2-based WFST path in :mod:`oasr.decode`: exposes the same
duck-typed searcher protocol (``reset`` / ``search`` / ``finalize_search`` +
``outputs`` / ``likelihood`` / ``times``) plus a batched-exact ``decode_offline`` fast
path. Log-probs stay on the GPU end to end (``wants_device_tensor = True``).

The CUDA decoder is compiled on first use via TVM-FFI JIT (``oasr.jit.wfst_decoder``),
mirroring the GPU CTC decoder. The stateful ``GpuDecoder`` C++ object lives behind an
opaque ``int64`` handle; decode results (host-side after backtrack) are returned through
caller-allocated CPU output tensors. The heavyweight state — graph image and decoder
instances — is shared process-wide through caches keyed by (graph, options, device), so
per-``WfstDecoderSearch`` construction stays cheap. Streaming searchers borrow a channel
from one shared multi-channel decoder and release it on ``finalize_search`` (or GC).

Graphs: pass either a prebuilt ``.img`` (built by :mod:`oasr.decoder.wfst.graph_export`)
or a k2 ``HLG.pt`` — the latter is exported once and cached next to the source file.
"""

from __future__ import annotations

import functools
import math
import threading
from pathlib import Path
from typing import List, Optional

import torch

_lock = threading.Lock()
_graphs: dict = {}
_offline: dict = {}
_streaming: dict = {}

_OFFLINE_MAX_LANES = 8
_STREAM_CHUNK_FRAMES = 128


@functools.cache
def _get_module():
    """Return the JIT-compiled WFST decoder module (compiled on first use)."""
    from oasr.jit.wfst_decoder import gen_wfst_decoder_module

    return gen_wfst_decoder_module().build_and_load()


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


def _get_graph(fst: str) -> int:
    """Return the process-wide graph handle for *fst* (loading it once)."""
    with _lock:
        if fst not in _graphs:
            mod = _get_module()
            _graphs[fst] = int(mod.wfst_load_graph(_resolve_graph_image(fst)))
        return _graphs[fst]


def _decoder_key(fst: str, opts: "WfstDecoderOptions", device: int) -> tuple:
    return (
        fst,
        opts.search_beam,
        opts.output_beam,
        opts.min_active_states,
        opts.max_active_states,
        opts.arena_budget_entries,
        opts.stream_log_entries,
        opts.gc_interval,
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
        arena_budget_entries: int = 0,
        stream_log_entries: int = 0,
        gc_interval: int = 0,
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
        # Winners-log budgets (8-byte entries); 0 keeps the decoder's built-in formulas.
        # `arena_budget_entries` caps the shared offline arena; `stream_log_entries`
        # sizes each streaming channel's region (committed per active channel).
        self.arena_budget_entries = arena_budget_entries
        self.stream_log_entries = stream_log_entries
        # >0 (even): offline winners-log GC cadence in steps — long audio decodes in
        # O(live window) winners memory instead of O(T). Off by default.
        self.gc_interval = gc_interval


def _get_offline_decoder(fst: str, opts: WfstDecoderOptions, device: int) -> int:
    key = _decoder_key(fst, opts, device) + (opts.max_frames, opts.max_offline_lanes)
    graph = _get_graph(fst)  # takes _lock itself; resolve before re-acquiring
    with _lock:
        if key not in _offline:
            mod = _get_module()
            _offline[key] = int(
                mod.wfst_create_decoder(
                    graph,
                    opts.search_beam,
                    opts.output_beam,
                    opts.min_active_states,
                    opts.max_active_states,
                    1,  # allow_partial
                    opts.max_offline_lanes,
                    opts.max_frames,
                    device,
                    32,
                    3,  # main_q_factor, cand_factor (offline: bench-proven)
                    1,  # use_cuda_graphs
                    0,
                    0,
                    0,  # lattice, fp16_logprobs, streaming
                    0,
                    3,  # lat_prune_interval, eps_iterations
                    opts.arena_budget_entries,
                    opts.stream_log_entries,
                    opts.gc_interval,
                )
            )
        return _offline[key]


def _get_streaming_decoder(fst: str, opts: WfstDecoderOptions, device: int) -> int:
    key = _decoder_key(fst, opts, device) + (opts.max_streams,)
    graph = _get_graph(fst)  # takes _lock itself; resolve before re-acquiring
    with _lock:
        if key not in _streaming:
            mod = _get_module()
            _streaming[key] = int(
                mod.wfst_create_decoder(
                    graph,
                    opts.search_beam,
                    opts.output_beam,
                    opts.min_active_states,
                    opts.max_active_states,
                    1,  # allow_partial
                    opts.max_streams,
                    _STREAM_CHUNK_FRAMES,
                    device,
                    16,
                    4,  # main_q_factor, cand_factor (streaming)
                    1,  # use_cuda_graphs
                    0,
                    0,
                    1,  # lattice, fp16_logprobs, streaming=1
                    0,
                    3,  # lat_prune_interval, eps_iterations
                    opts.arena_budget_entries,
                    opts.stream_log_entries,
                    # Streaming GC always on: it runs once per chunk (the value is just the
                    # enable switch here), drains finalized arcs to the host, and makes the
                    # per-channel winners region a ring — streams are no longer length-capped
                    # and long-stream results no longer truncate to the last path_cap arcs.
                    opts.gc_interval or 2,
                )
            )
        return _streaming[key]


def _run_decode_batch(mod, dec_handle: int, packed: torch.Tensor, lens: List[int]):
    """Decode a padded ``[g, T, V]`` CUDA batch; return (words_per_row, scores, oks).

    Results marshal through caller-allocated CPU tensors. ``words`` per lane <= frames
    for HLG, so ``cap = T`` almost never truncates; if a lane's true word count exceeds
    ``cap`` (multi-word arcs), we re-run with a large-enough buffer — ``decode_batch`` is
    idempotent (deterministic, same CUDA-graph bucket), so the re-run reproduces it.
    """
    g = int(packed.size(0))
    lens_t = torch.tensor(lens, dtype=torch.int32)
    cap = max(1, int(packed.size(1)))
    out_words = out_wlen = out_scores = out_meta = None
    for _ in range(2):
        out_words = torch.empty((g, cap), dtype=torch.int32)
        out_wlen = torch.empty((g,), dtype=torch.int32)
        out_scores = torch.empty((g,), dtype=torch.float64)
        out_meta = torch.empty((g, 3), dtype=torch.int32)
        mod.wfst_decode_batch(dec_handle, packed, lens_t, out_words, out_wlen, out_scores, out_meta)
        need = int(out_wlen.max().item()) if g else 0
        if need <= cap:
            break
        cap = need  # widen and re-run (idempotent)
    words: List[List[int]] = []
    scores: List[float] = []
    oks: List[bool] = []
    wl = out_wlen.tolist()
    ok = out_meta[:, 0].tolist()
    sc = out_scores.tolist()
    for b in range(g):
        if ok[b]:
            length = min(int(wl[b]), cap)
            words.append(out_words[b, :length].tolist())
            scores.append(float(sc[b]))
        else:
            words.append([])
            scores.append(0.0)
        oks.append(bool(ok[b]))
    return words, scores, oks


class WfstDecoderSearch:
    """Searcher-protocol adapter over the in-tree GPU WFST decoder.

    Offline (``decode_offline``) runs the exact offline beam semantics in one batched
    call; the streaming protocol (``reset`` / ``search`` / ``finalize_search``) maps to
    a channel of the shared streaming decoder (k2-online beam semantics).
    """

    wants_device_tensor = True

    def __init__(self, fst: str, opts: WfstDecoderOptions) -> None:
        self._fst = fst
        self._opts = opts
        self._channel: Optional[int] = None
        self._stream_dec: Optional[int] = None
        self._mod = None
        self._device: Optional[int] = None
        self._fed_frames = 0
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
        self._fed_frames = 0
        self._outputs = []
        self._likelihood = []

    def search(self, logp: torch.Tensor) -> None:
        """Streaming: advance this request's channel by one chunk of frames."""
        logp = self._prepare(logp)
        if logp.size(0) == 0:
            return
        mod = _get_module()
        if self._channel is None:
            self._device = logp.device.index or 0
            self._mod = mod
            self._stream_dec = _get_streaming_decoder(self._fst, self._opts, self._device)
            self._channel = int(mod.wfst_create_stream(self._stream_dec))
            if self._channel < 0:
                raise RuntimeError(
                    f"no free WFST stream channels (max_streams={self._opts.max_streams})"
                )
        partial_words: Optional[List[int]] = None
        for start in range(0, logp.size(0), _STREAM_CHUNK_FRAMES):
            piece = logp[start : start + _STREAM_CHUNK_FRAMES].unsqueeze(0).contiguous()
            tc = int(piece.size(1))
            self._fed_frames += tc
            cap = max(1, self._fed_frames)  # cumulative best-path words <= fed frames (HLG)
            out_words = torch.empty((1, cap), dtype=torch.int32)
            out_wlen = torch.empty((1,), dtype=torch.int32)
            out_channels = torch.empty((1,), dtype=torch.int32)
            out_overflow = torch.empty((1,), dtype=torch.int32)
            mod.wfst_advance_chunk(
                self._stream_dec,
                torch.tensor([self._channel], dtype=torch.int32),
                piece,
                torch.tensor([tc], dtype=torch.int32),
                1,  # want_partial
                out_words,
                out_wlen,
                out_channels,
                out_overflow,
            )
            partial_words = out_words[0, : min(int(out_wlen[0]), cap)].tolist()
        if partial_words is not None:
            self._outputs = [partial_words]
            self._likelihood = [0.0]

    def finalize_search(self) -> None:
        if self._channel is None:
            return
        mod = self._mod or _get_module()
        cap = max(1, self._fed_frames)
        out_words = torch.empty((cap,), dtype=torch.int32)
        out_wlen = torch.empty((1,), dtype=torch.int32)
        out_score = torch.empty((1,), dtype=torch.float64)
        out_meta = torch.empty((3,), dtype=torch.int32)
        mod.wfst_finalize_stream(
            self._stream_dec, self._channel, out_words, out_wlen, out_score, out_meta
        )
        self._release()
        if int(out_meta[0]):
            self._outputs = [out_words[: min(int(out_wlen[0]), cap)].tolist()]
            self._likelihood = [float(out_score[0])]
        else:
            self._outputs = [[]]
            self._likelihood = [0.0]

    # -- offline fast path ---------------------------------------------------

    def decode_offline(self, logp: torch.Tensor):
        """Exact offline decode of one utterance; returns (tokens, scores)."""
        logp = self._prepare(logp)
        device = logp.device.index or 0
        dec = _get_offline_decoder(self._fst, self._opts, device)
        words, scores, _ = _run_decode_batch(
            _get_module(), dec, logp.unsqueeze(0).contiguous(), [int(logp.size(0))]
        )
        return [words[0]], [scores[0]]

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
        mod = _get_module()
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
            w, s, _ = _run_decode_batch(mod, dec, packed.contiguous(), lens)
            for i in range(len(w)):
                tokens[start + i] = w[i]
                scores[start + i] = s[i]
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
        if self._channel is not None and self._stream_dec is not None and self._mod is not None:
            try:
                self._mod.wfst_release_stream(self._stream_dec, self._channel)
            except Exception:
                pass
        self._channel = None

    def __del__(self) -> None:  # channel safety net for aborted requests
        try:
            self._release()
        except Exception:
            pass
