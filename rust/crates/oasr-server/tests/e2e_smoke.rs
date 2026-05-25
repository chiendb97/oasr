// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end smoke test using `scripts/fake_engine_worker.py` (a Python
// EngineWorker driven by a mock ASREngine, so no torch/checkpoint required).
//
// What it covers:
// - Spawn the Python worker, wait for READY, build an EngineClient.
// - Submit an offline request; assert the Final event arrives with the
//   expected echo'd text.
// - Open a streaming request; push N chunks; receive ≥1 Partial then 1 Final.
//
// HTTP and gRPC paths are exercised by `scripts/ws_stream.py` and
// `scripts/grpc_stream.py` via the verification checklist in
// `docs/serving.md`.  The mock worker plus a real `oasr-server` binary is a
// follow-up TODO (would need an `--engine-worker-cmd` override).

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use bytes::Bytes;
use oasr_engine_client::client::EngineClientConfig;
use oasr_engine_client::EngineClient;
use oasr_wire::Event;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;
use futures::StreamExt;

fn project_root() -> PathBuf {
    // CARGO_MANIFEST_DIR for tests in oasr-server points at
    // `<repo>/rust/crates/oasr-server`.  Walk up to the repo root.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

async fn spawn_fake_worker(endpoint: &str) -> (tokio::process::Child, EngineClient) {
    let root = project_root();
    let script = root.join("scripts/fake_engine_worker.py");
    assert!(script.exists(), "fake worker script missing: {script:?}");

    let mut cmd = Command::new("python");
    cmd.arg(&script)
        .arg("--zmq-endpoint")
        .arg(endpoint)
        .arg("--engine-config-json")
        .arg("{}")
        .arg("--worker-threads")
        .arg("1")
        .arg("--max-concurrent-requests")
        .arg("8")
        .current_dir(&root)
        .env("PYTHONPATH", &root)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    let mut child = cmd.spawn().expect("spawn python fake worker");

    // Wait for READY.
    let stdout = child.stdout.take().expect("stdout piped");
    let mut lines = BufReader::new(stdout).lines();
    let ok = timeout(Duration::from_secs(15), async {
        while let Some(line) = lines.next_line().await.expect("read line") {
            eprintln!("[worker] {line}");
            if line.trim() == "READY" {
                return true;
            }
        }
        false
    })
    .await
    .expect("READY timeout");
    assert!(ok, "fake worker exited before READY");

    // Connect a client.
    let cfg = EngineClientConfig::new(endpoint.to_owned(), "test".to_owned());
    let client = EngineClient::connect(cfg).await.expect("connect");

    // Sanity ping.
    client
        .ping(Duration::from_secs(5))
        .await
        .expect("initial ping");

    (child, client)
}

#[tokio::test]
async fn offline_round_trip() {
    // tcp:// works reliably between Rust `zeromq` and pyzmq; `ipc://` is
    // unstable with the pure-Rust crate (handshake hangs).  Use a per-PID
    // port to allow parallel test runs.
    let port = 55500 + (std::process::id() % 100) as u16;
    let endpoint = format!("tcp://127.0.0.1:{port}");
    let (mut child, client) = spawn_fake_worker(&endpoint).await;

    let audio = Bytes::from_static(&[0u8; 64]);
    let handle = client
        .submit_offline(audio, 16000, 0)
        .await
        .expect("submit_offline");
    let ev = timeout(Duration::from_secs(5), handle.finish())
        .await
        .expect("finish timeout")
        .expect("finish");
    match ev {
        Event::Final { text, .. } => {
            assert!(
                text.starts_with("offline-echo:"),
                "unexpected text: {text}"
            );
        }
        other => panic!("expected Final, got {other:?}"),
    }

    let _ = child.start_kill();
    let _ = child.wait().await;
}

#[tokio::test]
async fn streaming_round_trip() {
    let port = 55600 + (std::process::id() % 100) as u16;
    let endpoint = format!("tcp://127.0.0.1:{port}");
    let (mut child, client) = spawn_fake_worker(&endpoint).await;

    let mut handle = client.open_streaming(16000, 0).await.expect("open_streaming");

    // Push 3 chunks, then flush_last with an empty chunk.
    let pad = Bytes::from(vec![0u8; 256 * 4]); // 256 f32 samples
    for _ in 0..3 {
        handle.push_chunk(pad.clone()).await.expect("push");
    }
    handle.flush_last(Bytes::new()).await.expect("flush_last");

    let mut partials = 0usize;
    let mut got_final = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        let Some(ev) = timeout(Duration::from_millis(500), handle.events.next())
            .await
            .ok()
            .flatten()
        else {
            continue;
        };
        match ev {
            Event::Accepted { .. } => {}
            Event::Partial { .. } => partials += 1,
            Event::Final { text, .. } => {
                assert!(
                    text.starts_with("streaming-echo:"),
                    "unexpected text: {text}"
                );
                got_final = true;
                break;
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert!(got_final, "did not see Final");
    assert!(partials >= 1, "expected ≥1 Partial, got {partials}");

    let _ = child.start_kill();
    let _ = child.wait().await;
}
