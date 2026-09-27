# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Slot streaming backend: engine-owned, fixed-extent streaming state.

For encoders whose whole streaming cache is **fixed-extent per stream** — a
left-context window of keys and values that rolls rather than grows, conv tails,
counters (Zipformer: icefall's six caches per layer, the embed's cached frames and
a processed-length counter) — and that declare it
(``encoder.slot_state_specs``, one :class:`~oasr.cache.StreamStateSpec`
per tensor, in the order of the encoder's list API).

The state lives in one :class:`~oasr.cache.SlotStateCache`: a persistent buffer
per declared tensor with the stream slot on the tensor's own batch axis
(``slot_axis``), so each tensor keeps its native layout.  A tick gathers the
active slots' rows, runs one batched chunk forward over the encoder's list API,
and scatters the new state back — against the per-request lists the
``"stateful"`` runtime threads, which it replaces for such an encoder:

* **no per-tick stacking**: the ``stateful`` runtime ``cat``s every stream's
  state into a batch and splits it back each tick; here a stream's state never
  leaves its rows;
* **stable addresses**, so the whole step — gather, chunk forward, scatter —
  is a CUDA graph per batch width.  Every window is full (a short final one is
  padded with the encoder's ``streaming_pad_value``), so the width is the only
  shape axis there is, and with the padding lane (``streaming_graph_pad_batch``,
  on whenever steps are graphed) a step runs at the next power-of-two width.

Capture writes: a graph's warm-up and capture *run* the step, scatter included.
They run against a reserved scratch slot (``max_batch_size``, never handed to a
stream) — capturing against live slots would advance those streams' state three
times for one chunk.  The replay then reads the real slot ids from its static
buffer, like every other input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import torch

from oasr.cache.slot_pool import StreamSlotPool
from oasr.cache.state import SlotStateCache
from oasr.utils.staging import to_device

from ..capture_recovery import capture_warmup_stream, recover_from_failed_capture
from ..request import Request
from .base import StreamingEncoderBackend, register_streaming_backend

if TYPE_CHECKING:
    from oasr.cache.types import CacheConfig
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig

logger = logging.getLogger(__name__)

__all__ = ["SlotStreamingBackend"]


@dataclass
class _Captured:
    """One captured step and the static buffers it reads and writes."""

    graph: "torch.cuda.CUDAGraph"
    xs: torch.Tensor  # (B, window, F)
    lens: torch.Tensor  # (B,) int32, the window length
    slots: torch.Tensor  # (B,) int64 slot ids
    out: torch.Tensor  # (B, T_out, D) — live until this key's next replay


@register_streaming_backend("slot")
class SlotStreamingBackend(StreamingEncoderBackend):
    """Fixed-extent streaming state in a slot cache, graph-captured per batch width."""

    streaming_kind: ClassVar[str] = "slot"

    def __init__(
        self,
        model: "BaseAsrModel",
        config: "EngineConfig",
        cache_config: "Optional[CacheConfig]" = None,
        *,
        graph_pool: Optional[Tuple[int, int]] = None,
        consumes: str = "log_probs",
    ) -> None:
        del cache_config  # no paged pool: the slot cache is this runtime's own
        # Duck-typed: the slot contract is a declaration on the encoder
        # (``slot_state_specs`` + the list API), checked here, not a base class.
        enc: Any = model.encoder
        surface: Any = model
        self._device = torch.device(config.device)
        self._dtype = config.dtype
        # "log_probs": encoder + head (``model.streaming_forward``); "hidden": the
        # encoder alone, for the families that decode hidden states.
        self._chunk_forward: Callable[..., Any] = (
            enc.streaming_forward if consumes == "hidden" else surface.streaming_forward
        )
        self._specs = tuple(enc.slot_state_specs)
        self._names = [s.name for s in self._specs]
        self._cap = max(1, int(config.max_batch_size))
        #: One slot past the streams': the capture scratch row (module docstring).
        self._scratch = self._cap
        self._state = SlotStateCache(
            self._specs, max_batch_size=self._cap + 1, device=self._device, dtype=self._dtype
        )
        self._slots = StreamSlotPool(self._cap)
        self._slot_of: Dict[int, int] = {}

        self._stride = int(enc.streaming_chunk_frames)
        self._window = int(getattr(enc, "streaming_window_frames", None) or self._stride)
        pad = getattr(enc, "streaming_pad_value", None)
        if pad is None:
            raise ValueError(
                f"{type(enc).__name__} declares streaming_kind='slot' but no "
                "streaming_pad_value: the slot runtime forwards whole windows only "
                "(its graphs have one shape per batch width), so a short final "
                "window has to be padded with something the encoder expects"
            )
        self._pad_value = float(pad)
        fcfg = getattr(config, "feature_config", None)
        dim = getattr(fcfg, "output_dim", None)
        self._feat_dim = int(dim) if dim else int(getattr(enc.config, "feature_dim", 80))

        self._use_graphs = bool(getattr(config, "use_cuda_graphs", False)) and (
            self._device.type == "cuda"
        )
        self._pool: Optional[Tuple[int, int]] = graph_pool
        if self._use_graphs and self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        self._graphs: Dict[int, _Captured] = {}
        self._failed: Set[int] = set()
        self._graphs_disabled = False
        self._max_captures = int(getattr(config, "streaming_graph_max_shapes", 512))
        # The padding lane: a step runs at the smallest ladder width holding its
        # streams, the extra rows reading pad-valued windows and writing the
        # scratch slot.  Auto pads whenever steps are graphed — every captured
        # width costs graph memory that grows with the width (measured ~2.3 MiB
        # per stream for Zipformer, 1.7 GiB for every width of a 32-stream pool),
        # and a power-of-two ladder is 6 captures there instead of 32.
        pad = getattr(config, "streaming_graph_pad_batch", None)
        self._pad_batch = self._use_graphs if pad is None else bool(pad)
        self._widths = self._resolve_widths(getattr(config, "streaming_graph_batch_ladder", None))
        #: Accounting: steps served by a replay vs run eager.
        self.replays = 0
        self.eager_steps = 0

    # ------------------------------------------------------------------
    # Window geometry
    # ------------------------------------------------------------------

    @property
    def decoding_window(self) -> int:
        return self._window

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def state_cache(self) -> SlotStateCache:
        return self._state

    @property
    def graph_batch_widths(self) -> Sequence[int]:
        """Every width a step can run at — each a graph key."""
        return tuple(self._widths)

    def _resolve_widths(self, ladder: Optional[Sequence[int]]) -> List[int]:
        """Ascending run widths, ``max_batch_size`` always the last."""
        cap = self._cap
        if ladder:
            return sorted({int(b) for b in ladder if 1 <= int(b) <= cap} | {cap})
        if not self._pad_batch:
            return list(range(1, cap + 1))
        widths, b = [], 1
        while b < cap:
            widths.append(b)
            b *= 2
        return widths + [cap]

    def _run_width(self, active: int) -> int:
        if not self._pad_batch:
            return active
        return next((w for w in self._widths if w >= active), active)

    # ------------------------------------------------------------------
    # Per-request lifecycle
    # ------------------------------------------------------------------

    def allocate(self, request: Request) -> None:
        sid = request.stream_id
        assert sid is not None, "stream_id must be assigned before allocate"
        slot = self._slots.allocate()
        # ``allocate_stream`` zeroes the slot in every state, and zero is every
        # declared state's initial value.
        self._state.allocate_stream(sid, slot)
        self._slot_of[sid] = slot
        request.slot_id = slot
        request.stream_context = None

    def free(self, request: Request) -> None:
        sid = request.stream_id
        if sid is None or sid not in self._slot_of:
            return
        self._state.free_stream(sid)
        self._slots.free(self._slot_of.pop(sid))
        request.slot_id = None

    def reset(self, request: Request) -> None:
        """Rule 13, in place: the slot's state back to zeros, the position to 0.

        Keeps the slot — nothing is gained by moving the stream, and the state
        buffers a graph addressed stay where they are either way.
        """
        sid = request.stream_id
        assert sid is not None and sid in self._slot_of, "reset of an unallocated stream"
        self._state.reset_stream(sid)
        request.offset = 0

    # ------------------------------------------------------------------
    # Per-tick forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_step(self, requests: List[Request]) -> Dict[str, torch.Tensor]:
        """Advance every ready stream one chunk, in one batched step."""
        window = self._window
        ready = [
            r
            for r in requests
            if r.feature_buffer is not None and r.has_ready_encoder_chunk(window)
        ]
        results: Dict[str, torch.Tensor] = {}
        if not ready:
            return results
        slots: List[int] = []
        for r in ready:
            sid = r.stream_id
            assert sid is not None and sid in self._slot_of, "slot stream must be allocated"
            slots.append(self._slot_of[sid])

        xs = torch.stack([self._window_of(r) for r in ready])  # (B, window, F)
        slot_ids = to_device(slots, dtype=torch.long, device=self._device)
        out = self._step(xs, slot_ids)
        for i, r in enumerate(ready):
            r.feature_cursor += self._stride
            r.offset += int(out.size(1))
            results[r.request_id] = out[i : i + 1]
        return results

    def _window_of(self, req: Request) -> torch.Tensor:
        """``(window, F)`` at the cursor, a short final window padded to full."""
        buf = req.feature_buffer
        assert buf is not None
        chunk = buf[req.feature_cursor : req.feature_cursor + self._window]
        short = self._window - chunk.size(0)
        if short > 0:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, short), value=self._pad_value)
        return chunk

    def _step(self, xs: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        active = int(xs.size(0))
        # Padded *before* the graph/eager branch, so a width that falls back to
        # eager computes exactly what its replay would: pad rows are pad-valued
        # windows at the scratch slot, and every op is row-local, so they touch
        # nothing but themselves and the scratch rows.
        batch = self._run_width(active)
        if batch > active:
            xs = torch.cat([xs, xs.new_full((batch - active, *xs.shape[1:]), self._pad_value)])
            slot_ids = torch.cat([slot_ids, slot_ids.new_full((batch - active,), self._scratch)])
        captured = self._graph(batch) if self._use_graphs else None
        if captured is None:
            self.eager_steps += 1
            lens = torch.full((batch,), self._window, dtype=torch.int32, device=self._device)
            out = self._run(xs.to(device=self._device, dtype=self._dtype), lens, slot_ids)
            return out[:active]
        captured.xs.copy_(xs)
        captured.slots.copy_(slot_ids)
        captured.graph.replay()
        self.replays += 1
        return captured.out[:active]

    def _run(self, xs: torch.Tensor, lens: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        """Gather → chunk forward → scatter: the step, eager or being captured."""
        views = self._state.views(slot_ids)
        states = [views[n].gather() for n in self._names]
        out, _out_lens, new_states = self._chunk_forward(xs, lens, states)
        for name, new in zip(self._names, new_states):
            views[name].scatter(new)
        return cast(torch.Tensor, out)

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------

    def _graph(self, batch: int) -> Optional[_Captured]:
        captured = self._graphs.get(batch)
        if captured is not None:
            return captured
        if (
            self._graphs_disabled
            or batch in self._failed
            or len(self._graphs) >= self._max_captures
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        return self._capture(batch)

    def _capture(self, batch: int) -> Optional[_Captured]:
        import tvm_ffi

        device = self._device
        c = _Captured(
            graph=torch.cuda.CUDAGraph(),
            xs=torch.full(
                (batch, self._window, self._feat_dim),
                self._pad_value,
                dtype=self._dtype,
                device=device,
            ),
            lens=torch.full((batch,), self._window, dtype=torch.int32, device=device),
            # Every row at the scratch slot while the step is run to warm up and
            # capture; the replay reads the real ids from this buffer.
            slots=torch.full((batch,), self._scratch, dtype=torch.long, device=device),
            out=torch.empty(0, device=device),
        )
        try:
            side = capture_warmup_stream(device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._run(c.xs, c.lens, c.slots)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize(device)
            ctx = torch.cuda.graph(c.graph, pool=self._pool)  # type: ignore[arg-type]
            with tvm_ffi.use_torch_stream(ctx):
                c.out = self._run(c.xs, c.lens, c.slots)
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("slot streaming graph capture ran out of memory; steps run eager")
            self._graphs_disabled = True
            recover_from_failed_capture(device, self._pool)
            self._pool = None
            return None
        except Exception as exc:  # noqa: BLE001 - capture is best-effort
            logger.warning("slot streaming graph capture failed at B=%d: %s", batch, exc)
            self._failed.add(batch)
            recover_from_failed_capture(device, self._pool)
            self._pool = torch.cuda.graph_pool_handle()
            return None
        self._state.reset_slot(self._scratch)
        self._graphs[batch] = c
        logger.info("captured slot streaming step: B=%d (%d live)", batch, len(self._graphs))
        return c

    def prewarm(
        self, batch_sizes: Sequence[int], cache_t1_buckets: Optional[Sequence[int]] = None
    ) -> None:
        """Capture every width a live tick can run at, before traffic.

        **Widest first.**  The captures share one graph pool, and a capture can
        reuse the blocks an earlier one freed only if they are big enough:
        captured narrowest-first, every width needs slightly more than any
        predecessor left behind, so the pool grows by the sum of all of them —
        3.1 GiB for 32 widths of a 16-layer Zipformer, against about one
        capture's worth widest-first.
        """
        del cache_t1_buckets  # the slot state has no growing axis to key on
        if not self._use_graphs:
            return
        widths = {int(b) for b in batch_sizes if 1 <= int(b) <= self._cap}
        for b in sorted(widths, reverse=True):
            self._graph(b)

    def recover_capture_state(self) -> None:
        recover_from_failed_capture(self._device, self._pool)

    def release_graphs(self) -> None:
        for c in self._graphs.values():
            try:
                c.graph.reset()
            except Exception:  # pragma: no cover - teardown must not raise
                pass
        self._graphs.clear()
        self._failed.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "captured": len(self._graphs),
            "replays": self.replays,
            "eager_steps": self.eager_steps,
            "state_bytes_per_stream": self._state.nbytes_per_stream(),
        }
