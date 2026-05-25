# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""MessagePack wire codec mirroring the Rust ``oasr-wire`` crate.

Each logical message is one or two ZMQ frames:

- Frame 1: msgpack-encoded header — a tagged dict ``{"type": ..., ...}``.
- Frame 2 (optional): raw little-endian f32 mono PCM samples for audio carriers
  (``CreateOffline``, ``FeedChunk``).  Decoded zero-copy via
  ``np.frombuffer(payload, dtype=np.float32)``.

The schema is shared verbatim with ``rust/crates/oasr-wire`` so the two
sides interoperate without per-message translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import msgpack

# ---------------------------------------------------------------------------
# Command tags (Rust = enum Cmd)
# ---------------------------------------------------------------------------

CMD_CREATE_OFFLINE = "CreateOffline"
CMD_CREATE_STREAMING = "CreateStreaming"
CMD_FEED_CHUNK = "FeedChunk"
CMD_CANCEL = "Cancel"
CMD_PING = "Ping"

CMDS_WITH_AUDIO = frozenset({CMD_CREATE_OFFLINE, CMD_FEED_CHUNK})

# ---------------------------------------------------------------------------
# Event tags (Rust = enum Event)
# ---------------------------------------------------------------------------

EVT_ACCEPTED = "Accepted"
EVT_PARTIAL = "Partial"
EVT_FINAL = "Final"
EVT_ERROR = "Error"
EVT_PONG = "Pong"
EVT_OVERLOADED = "Overloaded"

# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

ERR_BUSY = "BUSY"
ERR_UNKNOWN_REQUEST = "UNKNOWN_REQUEST"
ERR_INVALID_CMD = "INVALID_CMD"
ERR_INTERNAL = "INTERNAL"
ERR_SHUTDOWN = "SHUTDOWN"


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def encode_header(header: Dict[str, Any]) -> bytes:
    """MessagePack-encode a command/event header dict."""
    return msgpack.packb(header, use_bin_type=True)


def decode_header(blob: bytes) -> Dict[str, Any]:
    """MessagePack-decode a header frame.  ``raw=False`` returns ``str`` keys."""
    return msgpack.unpackb(blob, raw=False)


# ---------------------------------------------------------------------------
# Event builders (compact helpers used by EngineWorker)
# ---------------------------------------------------------------------------


def make_accepted(request_id: str) -> Dict[str, Any]:
    return {"type": EVT_ACCEPTED, "request_id": request_id}


def make_partial(
    request_id: str,
    text: str,
    tokens: List[List[int]],
    scores: Optional[List[float]],
) -> Dict[str, Any]:
    return {
        "type": EVT_PARTIAL,
        "request_id": request_id,
        "text": text,
        "tokens": tokens,
        "scores": scores,
    }


def make_final(
    request_id: str,
    text: str,
    tokens: List[List[int]],
    scores: Optional[List[float]],
) -> Dict[str, Any]:
    return {
        "type": EVT_FINAL,
        "request_id": request_id,
        "text": text,
        "tokens": tokens,
        "scores": scores,
    }


def make_error(request_id: str, code: str, message: str) -> Dict[str, Any]:
    return {
        "type": EVT_ERROR,
        "request_id": request_id,
        "code": code,
        "message": message,
    }


def make_pong(
    seq: int,
    model_info: Optional[Dict[str, Any]],
    num_running: int,
    num_waiting: int,
) -> Dict[str, Any]:
    return {
        "type": EVT_PONG,
        "seq": seq,
        "model_info": model_info,
        "num_running": num_running,
        "num_waiting": num_waiting,
    }


def make_overloaded(reason: str, queue_depth: int) -> Dict[str, Any]:
    return {
        "type": EVT_OVERLOADED,
        "reason": reason,
        "queue_depth": queue_depth,
    }


# ---------------------------------------------------------------------------
# Multipart helpers (ROUTER socket frames)
# ---------------------------------------------------------------------------


@dataclass
class IncomingMessage:
    """One decoded inbound multipart message from a DEALER.

    ROUTER prepends the DEALER's identity frame; we keep it so the worker can
    address the reply back to the same peer.
    """

    identity: bytes
    header: Dict[str, Any]
    payload: Optional[bytes]  # None when the cmd does not carry audio


def parse_incoming(frames: List[bytes]) -> IncomingMessage:
    """Parse a ROUTER recv_multipart result into a typed ``IncomingMessage``.

    Layout: ``[identity, header_msgpack, optional_payload]``.
    """
    if len(frames) < 2:
        raise ValueError(
            f"expected >=2 frames (identity, header[, payload]); got {len(frames)}"
        )
    identity = frames[0]
    header = decode_header(frames[1])
    payload = frames[2] if len(frames) >= 3 else None
    return IncomingMessage(identity=identity, header=header, payload=payload)


def build_outgoing(identity: bytes, event: Dict[str, Any]) -> List[bytes]:
    """Build a ROUTER send_multipart frame list addressed to ``identity``."""
    return [identity, encode_header(event)]


def build_outgoing_with_payload(
    identity: bytes, event: Dict[str, Any], payload: bytes
) -> List[bytes]:
    """Same as :func:`build_outgoing` plus a binary payload frame."""
    return [identity, encode_header(event), payload]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def require_fields(header: Dict[str, Any], *fields: str) -> Tuple[Any, ...]:
    """Validate that ``header`` contains all of ``fields``; return the values."""
    out = []
    for f in fields:
        if f not in header:
            raise KeyError(f"missing field {f!r} in {header.get('type')!r}")
        out.append(header[f])
    return tuple(out)
