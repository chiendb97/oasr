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
instances — is cached by (graph, options, device), so per-``WfstDecoderSearch``
construction stays cheap. Streaming searchers borrow a channel from a multi-channel
decoder and release it on ``finalize_search`` (or GC).

Decoders are cached process-wide unless a :class:`WfstDecoderScope` is passed, in which
case they are private to it.  A decoder serializes its callers, so the engine gives each
``ASREngine`` its own scope: engine workers sharing one decoder would queue on each other.
The graph's device copy is shared either way (one per graph per device).

Graphs: pass either a prebuilt ``.img`` (built by :mod:`oasr.decoder.wfst.graph_export`)
or a k2 ``HLG.pt`` — the latter is exported once and cached next to the source file.
"""

from __future__ import annotations

import functools
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_lock = threading.Lock()
_graphs: dict = {}
_offline: Dict[tuple, "_DecoderRef"] = {}
_streaming: Dict[tuple, "_DecoderRef"] = {}

_OFFLINE_MAX_LANES = 8
_STREAM_CHUNK_FRAMES = 128

#: Per-lane capacity is ``main_q_factor * max_active_states`` (the surviving frontier),
#: and the claims, candidate and hash tables scale with it.  Offline: bench-proven.
_OFFLINE_MAIN_Q_FACTOR = 32
_OFFLINE_CAND_FACTOR = 3
#: The rescue instance a flagged lane is re-decoded on: twice the frontier capacity
#: (so twice every table derived from it), and only two lanes — overflow is rare
#: (15 of 2000 LJSpeech utterances at engine settings), so it stays small.
_RESCUE_MAIN_Q_FACTOR = 64
_RESCUE_LANES = 2

#: ``DecodeResult::overflow`` bits (``include/oasr/wfst/common.cuh``).  Any of them
#: means the search hit a capacity limit and its result may be missing words.
OVERFLOW_CAND = 1  # candidate buffer full
OVERFLOW_HASH = 2  # hash probe bound exceeded
OVERFLOW_ARENA = 4  # winners arena (offline) or ring (streaming) full
OVERFLOW_CLAIMS = 8  # distinct-state claims list full
OVERFLOW_KEPT = 16  # surviving frontier > main_q


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


class _DecoderRef:
    """One ``GpuDecoder`` handle, freed when the last reference to it goes.

    Whatever is about to call into a decoder holds one of these for the length of the
    call, and a stream holds one for the life of its channel, so a decoder dropped by
    its scope is freed only once nothing can still be using it.
    """

    __slots__ = ("handle", "__weakref__")

    def __init__(self, handle: int) -> None:
        self.handle = handle

    def __del__(self) -> None:
        try:
            _get_module().wfst_free_decoder(self.handle)
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass


class WfstDecoderScope:
    """The GPU decoders private to one owner — in the engine, one ``ASREngine``.

    A ``GpuDecoder`` runs one call at a time: it has one stream and one set of staging
    buffers, and holds a mutex across every call.  Two engine workers sharing one
    therefore queue on each other; measured on per-chunk streaming calls, two threads
    on one decoder managed 0.55x the throughput of two threads on a decoder each, at
    1.8x the median latency.  A scope also sizes the streaming decoder's channels to
    its own owner, where a shared one gave N workers ``max_streams`` channels between
    them, and failed whole cohorts once they ran out.

    Decoders are created on first use and reused within the scope.  :meth:`close`
    drops the scope's references; a stream still holding a channel keeps its decoder
    until the channel is released.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._offline: Dict[tuple, _DecoderRef] = {}
        self._streaming: Dict[tuple, _DecoderRef] = {}

    def close(self) -> None:
        with self._lock:
            self._offline.clear()
            self._streaming.clear()


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


def _caches(
    scope: Optional[WfstDecoderScope],
) -> Tuple[threading.Lock, Dict[tuple, _DecoderRef], Dict[tuple, _DecoderRef]]:
    if scope is None:
        return _lock, _offline, _streaming
    return scope._lock, scope._offline, scope._streaming


def _get_offline_decoder(
    fst: str,
    opts: WfstDecoderOptions,
    device: int,
    scope: Optional[WfstDecoderScope] = None,
    rescue: bool = False,
) -> _DecoderRef:
    """The offline decoder for these options, or with ``rescue`` its rescue instance."""
    key = _decoder_key(fst, opts, device) + (opts.max_frames, opts.max_offline_lanes, rescue)
    lanes = _RESCUE_LANES if rescue else opts.max_offline_lanes
    main_q_factor = _RESCUE_MAIN_Q_FACTOR if rescue else _OFFLINE_MAIN_Q_FACTOR
    # The rescue gets the main instance's whole arena budget rather than a two-lane
    # share of it: an utterance that filled the shared arena needs at least that much.
    # Reserved, not committed — the arena commits on demand.
    arena_budget = opts.arena_budget_entries
    if rescue and arena_budget == 0:
        arena_budget = min(512 << 20, max(64 << 20, (16 << 20) * opts.max_offline_lanes))
    graph = _get_graph(fst)  # takes _lock itself; resolve before re-acquiring
    lock, offline, _ = _caches(scope)
    with lock:
        if key not in offline:
            mod = _get_module()
            offline[key] = _DecoderRef(
                mod.wfst_create_decoder(
                    graph,
                    opts.search_beam,
                    opts.output_beam,
                    opts.min_active_states,
                    opts.max_active_states,
                    1,  # allow_partial
                    lanes,
                    opts.max_frames,
                    device,
                    main_q_factor,
                    _OFFLINE_CAND_FACTOR,
                    1,  # use_cuda_graphs
                    0,
                    0,
                    0,  # lattice, fp16_logprobs, streaming
                    0,
                    3,  # lat_prune_interval, eps_iterations
                    arena_budget,
                    opts.stream_log_entries,
                    opts.gc_interval,
                )
            )
        return offline[key]


def _get_streaming_decoder(
    fst: str, opts: WfstDecoderOptions, device: int, scope: Optional[WfstDecoderScope] = None
) -> _DecoderRef:
    key = _decoder_key(fst, opts, device) + (opts.max_streams,)
    graph = _get_graph(fst)  # takes _lock itself; resolve before re-acquiring
    lock, _, streaming = _caches(scope)
    with lock:
        if key not in streaming:
            mod = _get_module()
            streaming[key] = _DecoderRef(
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
                    # Per-chunk collection drains finalized arcs and keeps the
                    # per-channel winners region bounded.
                    opts.gc_interval or 2,
                )
            )
        return streaming[key]


def _run_decode_batch(mod, dec_handle: int, packed: torch.Tensor, lens: List[int]):
    """Decode a padded ``[g, T, V]`` CUDA batch; return (words, scores, oks, overflow).

    ``overflow`` is each row's ``OVERFLOW_*`` bits (0 when the search stayed in bounds).

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
    overflow = [int(v) for v in out_meta[:, 2].tolist()]
    for b in range(g):
        if ok[b]:
            length = min(int(wl[b]), cap)
            words.append(out_words[b, :length].tolist())
            scores.append(float(sc[b]))
        else:
            words.append([])
            scores.append(0.0)
        oks.append(bool(ok[b]))
    return words, scores, oks, overflow


class WfstDecoderSearch:
    """Searcher-protocol adapter over the in-tree GPU WFST decoder.

    Offline (``decode_offline``) runs the exact offline beam semantics in one batched
    call; the streaming protocol (``reset`` / ``search`` / ``finalize_search``) maps to
    a channel of the shared streaming decoder (k2-online beam semantics).  ``scope``
    selects whose decoders it uses: a :class:`WfstDecoderScope`, or ``None`` for the
    process-wide ones.

    A lane that hits a capacity limit is recovered here rather than returned as if it
    were complete.  Offline, it is decoded again on a rescue instance with twice the
    frontier capacity.  Streaming, a winners-ring overrun is the one limit the decoder
    cannot outlive in place — the lane's collector stops for good and its transcript
    freezes — so the chunk is decoded again on a fresh channel, after the words already
    decoded.  Whatever was not recovered is reported: :attr:`overflow` (the last search
    or single decode) and :attr:`batch_overflow` (the last batch) hold the
    ``OVERFLOW_*`` bits, 0 when the result is complete.
    """

    wants_device_tensor = True

    def __init__(
        self, fst: str, opts: WfstDecoderOptions, scope: Optional[WfstDecoderScope] = None
    ) -> None:
        self._fst = fst
        self._opts = opts
        self._scope = scope
        self._channel: Optional[int] = None
        self._stream_dec: Optional[_DecoderRef] = None
        self._mod = None
        self._device: Optional[int] = None
        self._fed_frames = 0
        self._outputs: List[List[int]] = []
        self._likelihood: List[float] = []
        self._overflow = 0
        self._batch_overflow: List[int] = []
        # Streaming: words decoded on channels abandoned after a ring overrun.
        self._carry: List[int] = []

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

    @property
    def overflow(self) -> int:
        """``OVERFLOW_*`` bits the last stream or single decode could not recover from."""
        return self._overflow

    @property
    def batch_overflow(self) -> List[int]:
        """Per-row ``OVERFLOW_*`` bits of the last :meth:`decode_offline_batch`."""
        return self._batch_overflow

    def reset(self) -> None:
        self._release()
        self._fed_frames = 0
        self._outputs = []
        self._likelihood = []
        self._overflow = 0
        self._carry = []

    def search(self, logp: torch.Tensor) -> None:
        """Streaming: advance this request's channel by one chunk of frames."""
        logp = self._prepare(logp)
        if logp.size(0) == 0:
            return
        for start in range(0, logp.size(0), _STREAM_CHUNK_FRAMES):
            piece = logp[start : start + _STREAM_CHUNK_FRAMES].unsqueeze(0).contiguous()
            words, bits = self._advance(piece)
            if bits & OVERFLOW_ARENA:
                # The ring overran: the decoder dropped this round's tokens and the
                # lane's collector skips it from now on, so it would never free ring
                # space again.  Keep the last complete hypothesis and decode the piece
                # again on a fresh channel — an empty ring, so this rarely recurs.
                self._restart(bits)
                words, bits = self._advance(piece)
                if bits & OVERFLOW_ARENA:  # too much even for an empty ring: skip it
                    self._restart(bits)
                    self._outputs = [list(self._carry)]
                    self._likelihood = [0.0]
                    continue
            self._overflow |= bits
            self._outputs = [self._carry + words]
            self._likelihood = [0.0]

    def _advance(self, piece: torch.Tensor) -> Tuple[List[int], int]:
        """Feed one ``[1, Tc, V]`` piece to the channel; return (best words, overflow bits).

        The words are this channel's hypothesis so far, without :attr:`_carry`; the
        bits are the lane's, cumulative since the channel opened.
        """
        mod = _get_module()
        if self._channel is None:
            self._device = piece.device.index or 0
            self._mod = mod
            stream_dec = _get_streaming_decoder(self._fst, self._opts, self._device, self._scope)
            channel = int(mod.wfst_create_stream(stream_dec.handle))
            if channel < 0:
                raise RuntimeError(
                    f"no free WFST stream channels (max_streams={self._opts.max_streams})"
                )
            self._channel, self._stream_dec = channel, stream_dec
        stream_dec = self._stream_dec
        assert stream_dec is not None  # held for as long as the channel is
        tc = int(piece.size(1))
        self._fed_frames += tc
        cap = max(1, self._fed_frames)  # cumulative best-path words <= fed frames (HLG)
        out_words = torch.empty((1, cap), dtype=torch.int32)
        out_wlen = torch.empty((1,), dtype=torch.int32)
        out_channels = torch.empty((1,), dtype=torch.int32)
        out_overflow = torch.empty((1,), dtype=torch.int32)
        mod.wfst_advance_chunk(
            stream_dec.handle,
            torch.tensor([self._channel], dtype=torch.int32),
            piece,
            torch.tensor([tc], dtype=torch.int32),
            1,  # want_partial
            out_words,
            out_wlen,
            out_channels,
            out_overflow,
        )
        return out_words[0, : min(int(out_wlen[0]), cap)].tolist(), int(out_overflow[0])

    def _restart(self, bits: int) -> None:
        """Abandon the channel after a ring overrun, keeping the last complete hypothesis."""
        self._overflow |= bits
        if self._outputs:
            self._carry = list(self._outputs[0])
        self._release()

    def finalize_search(self) -> None:
        stream_dec = self._stream_dec
        if self._channel is None or stream_dec is None:
            return
        mod = self._mod or _get_module()
        cap = max(1, self._fed_frames)
        out_words = torch.empty((cap,), dtype=torch.int32)
        out_wlen = torch.empty((1,), dtype=torch.int32)
        out_score = torch.empty((1,), dtype=torch.float64)
        out_meta = torch.empty((3,), dtype=torch.int32)
        mod.wfst_finalize_stream(
            stream_dec.handle, self._channel, out_words, out_wlen, out_score, out_meta
        )
        self._release()
        self._overflow |= int(out_meta[2])
        if int(out_meta[0]):
            self._outputs = [self._carry + out_words[: min(int(out_wlen[0]), cap)].tolist()]
            self._likelihood = [float(out_score[0])]
        else:
            self._outputs = [list(self._carry)]
            self._likelihood = [0.0]

    # -- offline fast path ---------------------------------------------------

    def decode_offline(self, logp: torch.Tensor):
        """Exact offline decode of one utterance; returns (tokens, scores)."""
        logp = self._prepare(logp)
        device = logp.device.index or 0
        dec = _get_offline_decoder(self._fst, self._opts, device, self._scope)
        words, scores, _, overflow = _run_decode_batch(
            _get_module(), dec.handle, logp.unsqueeze(0).contiguous(), [int(logp.size(0))]
        )
        if overflow[0]:
            words, scores, overflow = self._rescue([logp], device)
        self._overflow = overflow[0]
        return [words[0]], [scores[0]]

    def decode_offline_batch(self, enc_out: torch.Tensor, enc_lengths):
        """Exact offline decode of a padded ``[B, T, V]`` batch; returns (tokens, scores).

        The batched equivalent of calling :meth:`decode_offline` on each row: every row
        is length-clipped and blank-skipped exactly as in the single-utterance path, then
        rows are grouped into ``<= max_offline_lanes`` padded sub-batches and decoded with
        one ``decode_batch`` GPU launch per group.  This preserves the original decoder's
        batched throughput (the headline perf lever) while producing per-row results
        identical to the one-at-a-time path.  ``tokens[b]`` is the best-path word list;
        ``scores[b]`` its total score (0.0 for a lane that produced nothing).  Rows that
        overflowed are decoded again on the rescue instance afterwards, and
        :attr:`batch_overflow` reports any it could not recover.
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
        self._batch_overflow = [0] * batch
        if batch == 0:
            return tokens, scores
        device = enc_out.device.index or 0
        dec = _get_offline_decoder(self._fst, self._opts, device, self._scope)
        mod = _get_module()
        lanes = max(1, self._opts.max_offline_lanes)
        flagged: List[Tuple[int, torch.Tensor]] = []  # (row, prepared log-probs)
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
            w, s, _, o = _run_decode_batch(mod, dec.handle, packed.contiguous(), lens)
            for i in range(len(w)):
                tokens[start + i] = w[i]
                scores[start + i] = s[i]
                if o[i]:
                    flagged.append((start + i, rows[i]))
        if flagged:
            w, s, o = self._rescue([row for _, row in flagged], device)
            for (b, _), words, score, bits in zip(flagged, w, s, o):
                tokens[b], scores[b], self._batch_overflow[b] = words, score, bits
        return tokens, scores

    def _rescue(self, rows: List[torch.Tensor], device: int):
        """Decode prepared ``[T, V]`` rows on the rescue instance; (words, scores, overflow).

        Its bits are what is reported: a row the rescue also overflowed on is degraded,
        and still at least as good as the first attempt, which had half the capacity.
        """
        dec = _get_offline_decoder(self._fst, self._opts, device, self._scope, rescue=True)
        mod = _get_module()
        words: List[List[int]] = []
        scores: List[float] = []
        overflow: List[int] = []
        for start in range(0, len(rows), _RESCUE_LANES):
            group = rows[start : start + _RESCUE_LANES]
            lens = [r.size(0) for r in group]
            packed = group[0].new_full((len(group), max(lens), group[0].size(1)), -30.0)
            for i, r in enumerate(group):
                packed[i, : r.size(0)] = r
            w, s, _, o = _run_decode_batch(mod, dec.handle, packed.contiguous(), lens)
            words.extend(w)
            scores.extend(s)
            overflow.extend(o)
        return words, scores, overflow

    # -- helpers ---------------------------------------------------------------

    def _prepare(self, logp: torch.Tensor) -> torch.Tensor:
        if logp.dim() != 2:
            raise ValueError(f"logp must be [T, V], got {logp.dim()}-D")
        if not logp.is_cuda:
            logp = logp.cuda()
        if logp.dtype != torch.float32:
            logp = logp.float()
        # Drop blank-dominated frames before search.
        thresh = self._opts.blank_skip_thresh
        if 0.0 < thresh < 1.0 and logp.size(0) > 0:
            keep = logp[:, self._opts.blank] <= math.log(thresh)
            if not bool(keep.all()):
                logp = logp[keep]
        return logp.contiguous()

    def _release(self) -> None:
        if self._channel is not None and self._stream_dec is not None and self._mod is not None:
            try:
                self._mod.wfst_release_stream(self._stream_dec.handle, self._channel)
            except Exception:
                pass
        self._channel = None
        self._stream_dec = None  # the decoder may now be freed once its scope lets go

    def __del__(self) -> None:  # channel safety net for aborted requests
        try:
            self._release()
        except Exception:
            pass
