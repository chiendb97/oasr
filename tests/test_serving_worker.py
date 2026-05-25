# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``oasr/serving/`` — IPC codec + EngineWorker dispatch.

These tests use a mocked ``ASREngine`` so they don't need a real checkpoint
or a GPU.  The end-to-end serving test (engine + Rust frontend + WebSocket)
lives in ``rust/crates/oasr-server/tests/e2e_smoke.rs``.
"""

from __future__ import annotations

import threading
import time
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import zmq


# ---------------------------------------------------------------------------
# IPC codec
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip():
    from oasr.serving import ipc

    cmd = {
        "type": ipc.CMD_FEED_CHUNK,
        "request_id": "abc-123",
        "is_last": True,
    }
    blob = ipc.encode_header(cmd)
    assert isinstance(blob, (bytes, bytearray))
    decoded = ipc.decode_header(bytes(blob))
    assert decoded == cmd


def test_parse_incoming_with_payload():
    from oasr.serving import ipc

    audio = np.linspace(-1.0, 1.0, 128, dtype=np.float32).tobytes()
    frames = [b"\x01dealer", ipc.encode_header({
        "type": ipc.CMD_FEED_CHUNK, "request_id": "r1", "is_last": False,
    }), audio]
    msg = ipc.parse_incoming(frames)
    assert msg.identity == b"\x01dealer"
    assert msg.header["type"] == ipc.CMD_FEED_CHUNK
    assert msg.payload == audio
    samples = np.frombuffer(msg.payload, dtype=np.float32)
    assert samples.shape == (128,)


def test_parse_incoming_without_payload():
    from oasr.serving import ipc

    frames = [b"id", ipc.encode_header({"type": ipc.CMD_PING, "seq": 7})]
    msg = ipc.parse_incoming(frames)
    assert msg.payload is None
    assert msg.header["seq"] == 7


def test_require_fields_raises():
    from oasr.serving import ipc

    with pytest.raises(KeyError):
        ipc.require_fields({"type": ipc.CMD_FEED_CHUNK, "request_id": "x"}, "request_id", "is_last")


# ---------------------------------------------------------------------------
# Worker dispatch — mocked engine, real ZMQ socket
# ---------------------------------------------------------------------------


def _make_worker(tmp_path, threads: int = 1, engine: MagicMock = None):
    from oasr.serving.engine_worker import EngineWorker, WorkerOptions

    if engine is None:
        engine = MagicMock()
        engine.num_running = 0
        engine.num_waiting = 0
        engine._config = MagicMock(
            ckpt_dir="/tmp/none", device="cpu", dtype="float32",
            chunk_size=16, max_batch_size=8, decoder_type="ctc_prefix_beam",
            _model_config=MagicMock(vocab_size=32),
        )
        engine.add_streaming_request.return_value = "rid-from-engine"
        engine.add_request.return_value = "rid-from-engine"
        engine.step.return_value = []

    endpoint = f"ipc://{tmp_path}/sock"
    opts = WorkerOptions(
        zmq_endpoint=endpoint,
        engine_config_json="{}",
        worker_threads=threads,
        max_concurrent_requests=4,
        overload_emit_interval_s=0.05,
        poll_block_ms=2,
    )
    worker = EngineWorker(opts, engine=engine)
    return worker, endpoint, engine


def _make_client(endpoint: str) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.IDENTITY, b"test-client")
    sock.connect(endpoint)
    return sock


def _drain(worker, deadline_s: float = 0.5) -> List[List[bytes]]:
    """Spin the worker's single-thread loop briefly so tests don't need to
    start the full run() loop."""
    import zmq
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        events = worker._poller.poll(timeout=10)
        if not events:
            break
        try:
            frames = worker._sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        worker._dispatch_frames(frames)
    return []


def _recv_one(client: zmq.Socket, timeout_ms: int = 500):
    from oasr.serving import ipc

    if client.poll(timeout_ms) == 0:
        raise AssertionError("no event received within timeout")
    frames = client.recv_multipart()
    return ipc.decode_header(frames[0])


def test_ping_returns_pong(tmp_path):
    from oasr.serving import ipc

    worker, endpoint, engine = _make_worker(tmp_path)
    client = _make_client(endpoint)
    try:
        client.send_multipart([ipc.encode_header({"type": ipc.CMD_PING, "seq": 42})])
        _drain(worker)
        ev = _recv_one(client)
        assert ev["type"] == ipc.EVT_PONG
        assert ev["seq"] == 42
        assert ev["model_info"]["ckpt_dir"] == "/tmp/none"
        assert ev["num_running"] == 0
    finally:
        client.close()
        worker._cleanup()


def test_create_streaming_dispatches_to_engine(tmp_path):
    from oasr.serving import ipc

    worker, endpoint, engine = _make_worker(tmp_path)
    client = _make_client(endpoint)
    try:
        client.send_multipart([ipc.encode_header({
            "type": ipc.CMD_CREATE_STREAMING,
            "request_id": "rid-1",
            "sample_rate": 16000,
            "priority": 0,
        })])
        _drain(worker)
        engine.add_streaming_request.assert_called_once_with(
            request_id="rid-1", sample_rate=16000, priority=0,
        )
        ev = _recv_one(client)
        assert ev["type"] == ipc.EVT_ACCEPTED
        assert ev["request_id"] == "rid-1"
    finally:
        client.close()
        worker._cleanup()


def test_feed_chunk_dispatches_with_payload(tmp_path):
    from oasr.serving import ipc

    worker, endpoint, engine = _make_worker(tmp_path)
    # Pretend rid-1 was already registered.
    worker._rid_to_identity["rid-1"] = b"test-client"
    client = _make_client(endpoint)
    try:
        audio = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32).tobytes()
        client.send_multipart([
            ipc.encode_header({
                "type": ipc.CMD_FEED_CHUNK, "request_id": "rid-1", "is_last": True,
            }),
            audio,
        ])
        _drain(worker)
        engine.feed_chunk.assert_called_once()
        rid_arg, chunk_arg = engine.feed_chunk.call_args.args
        assert rid_arg == "rid-1"
        assert isinstance(chunk_arg, np.ndarray)
        assert chunk_arg.dtype == np.float32
        assert chunk_arg.tolist() == pytest.approx([0.1, -0.2, 0.3, -0.4])
        assert engine.feed_chunk.call_args.kwargs == {"is_last": True}
    finally:
        client.close()
        worker._cleanup()


def test_create_offline_with_audio_payload(tmp_path):
    from oasr.serving import ipc

    worker, endpoint, engine = _make_worker(tmp_path)
    client = _make_client(endpoint)
    try:
        audio = np.zeros(256, dtype=np.float32).tobytes()
        client.send_multipart([
            ipc.encode_header({
                "type": ipc.CMD_CREATE_OFFLINE, "request_id": "off-1",
                "sample_rate": 16000, "priority": 0,
            }),
            audio,
        ])
        _drain(worker)
        engine.add_request.assert_called_once()
        kwargs = engine.add_request.call_args.kwargs
        assert kwargs["request_id"] == "off-1"
        assert kwargs["streaming"] is False
        ev = _recv_one(client)
        assert ev["type"] == ipc.EVT_ACCEPTED
    finally:
        client.close()
        worker._cleanup()


def test_cancel_calls_abort_request(tmp_path):
    from oasr.serving import ipc

    worker, endpoint, engine = _make_worker(tmp_path)
    worker._rid_to_identity["rid-1"] = b"test-client"
    client = _make_client(endpoint)
    try:
        client.send_multipart([ipc.encode_header({
            "type": ipc.CMD_CANCEL, "request_id": "rid-1",
        })])
        _drain(worker)
        engine.abort_request.assert_called_once_with("rid-1")
        assert "rid-1" not in worker._rid_to_identity
    finally:
        client.close()
        worker._cleanup()


def test_admission_blocked_when_over_cap(tmp_path):
    from oasr.serving import ipc

    engine = MagicMock()
    engine.num_running = 4  # == max_concurrent_requests
    engine.num_waiting = 0
    engine._config = MagicMock(ckpt_dir="/tmp/x", device="cpu", dtype="float32",
                                chunk_size=16, max_batch_size=8,
                                decoder_type="ctc_prefix_beam",
                                _model_config=MagicMock(vocab_size=32))
    worker, endpoint, _ = _make_worker(tmp_path, engine=engine)
    client = _make_client(endpoint)
    try:
        client.send_multipart([ipc.encode_header({
            "type": ipc.CMD_CREATE_STREAMING,
            "request_id": "rid-busy", "sample_rate": 16000,
        })])
        _drain(worker)
        ev = _recv_one(client)
        assert ev["type"] == ipc.EVT_ERROR
        assert ev["code"] == ipc.ERR_BUSY
        engine.add_streaming_request.assert_not_called()
    finally:
        client.close()
        worker._cleanup()


def test_unknown_request_feed_returns_error(tmp_path):
    from oasr.serving import ipc

    engine = MagicMock()
    engine.num_running = 0
    engine.num_waiting = 0
    engine._config = MagicMock(ckpt_dir="/tmp/x", device="cpu", dtype="float32",
                                chunk_size=16, max_batch_size=8,
                                decoder_type="ctc_prefix_beam",
                                _model_config=MagicMock(vocab_size=32))
    engine.feed_chunk.side_effect = KeyError("unknown rid")
    worker, endpoint, _ = _make_worker(tmp_path, engine=engine)
    client = _make_client(endpoint)
    try:
        audio = np.zeros(16, dtype=np.float32).tobytes()
        client.send_multipart([
            ipc.encode_header({
                "type": ipc.CMD_FEED_CHUNK, "request_id": "ghost", "is_last": False,
            }),
            audio,
        ])
        _drain(worker)
        ev = _recv_one(client)
        assert ev["type"] == ipc.EVT_ERROR
        assert ev["code"] == ipc.ERR_UNKNOWN_REQUEST
    finally:
        client.close()
        worker._cleanup()


def test_publish_outputs_routes_partial_and_final(tmp_path):
    from oasr.serving import ipc

    worker, endpoint, engine = _make_worker(tmp_path)
    worker._rid_to_identity["rid-1"] = b"test-client"
    client = _make_client(endpoint)
    try:
        partial = MagicMock(request_id="rid-1", text="hello",
                            tokens=[[1, 2]], scores=[-0.1], finished=False)
        final = MagicMock(request_id="rid-1", text="hello world",
                          tokens=[[1, 2, 3]], scores=[-0.05], finished=True)
        worker._publish_outputs([partial, final])

        ev_partial = _recv_one(client)
        assert ev_partial["type"] == ipc.EVT_PARTIAL
        assert ev_partial["text"] == "hello"

        ev_final = _recv_one(client)
        assert ev_final["type"] == ipc.EVT_FINAL
        assert ev_final["text"] == "hello world"

        # After Final, the routing entry must be removed.
        assert "rid-1" not in worker._rid_to_identity
    finally:
        client.close()
        worker._cleanup()


def test_two_thread_mode_smoke(tmp_path):
    """Sanity check that the two-thread loop wires inbound + outbound."""
    from oasr.serving import ipc

    worker, endpoint, engine = _make_worker(tmp_path, threads=2)
    client = _make_client(endpoint)
    runner = threading.Thread(target=worker.run, daemon=True)
    runner.start()
    try:
        # Give the step thread a moment to spin up.
        time.sleep(0.05)
        client.send_multipart([ipc.encode_header({
            "type": ipc.CMD_PING, "seq": 1,
        })])
        ev = _recv_one(client, timeout_ms=1000)
        assert ev["type"] == ipc.EVT_PONG
        assert ev["seq"] == 1
    finally:
        worker._shutdown.set()
        client.close()
        runner.join(timeout=2.0)
