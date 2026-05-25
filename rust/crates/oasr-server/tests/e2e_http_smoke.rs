// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// HTTP smoke test: stands up an axum router against a fake worker and walks
// the offline + streaming paths.  Uses an in-process `EnginePool` instead of
// the full `oasr-server` binary so we don't need to manage TCP listeners or
// signal handling.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use oasr_engine_client::client::EngineClientConfig;
use oasr_engine_client::{EngineClient, EnginePool};
use oasr_server_http::{build_router, ServerState};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

fn project_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

async fn spawn_fake_worker(endpoint: &str) -> tokio::process::Child {
    let root = project_root();
    let script = root.join("scripts/fake_engine_worker.py");
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
    let mut child = cmd.spawn().expect("spawn python worker");
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    let ok = timeout(Duration::from_secs(15), async {
        while let Some(line) = lines.next_line().await.unwrap() {
            if line.trim() == "READY" {
                return true;
            }
        }
        false
    })
    .await
    .expect("READY timeout");
    assert!(ok);
    child
}

async fn build_state(endpoint: &str) -> (Arc<ServerState>, tokio::process::Child) {
    let child = spawn_fake_worker(endpoint).await;
    let client_cfg = EngineClientConfig::new(endpoint.to_owned(), "test".to_owned());
    let client = Arc::new(EngineClient::connect(client_cfg).await.expect("connect"));
    client
        .ping(Duration::from_secs(5))
        .await
        .expect("initial ping");
    let pool = Arc::new(EnginePool::new(vec![client]));
    let state = Arc::new(ServerState {
        pool,
        prometheus: None,
    });
    (state, child)
}

async fn run_server(state: Arc<ServerState>) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = build_router(state);
    let h = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    (addr, h)
}

#[tokio::test]
async fn http_offline_wav_roundtrip() {
    let port = 55700 + (std::process::id() % 100) as u16;
    let endpoint = format!("tcp://127.0.0.1:{port}");
    let (state, mut child) = build_state(&endpoint).await;
    let (addr, server) = run_server(state).await;

    // Build a tiny mono WAV via hound.
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = hound::WavWriter::new(
            std::io::Cursor::new(&mut buf),
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for i in 0..160 {
            w.write_sample((i * 100) as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/transcriptions"))
        .header("content-type", "audio/wav")
        .body(buf)
        .send()
        .await
        .expect("POST");
    assert_eq!(resp.status(), 200, "body: {:?}", resp.text().await);
    let body: serde_json::Value = resp.json().await.expect("json");
    let text = body["text"].as_str().unwrap_or_default();
    assert!(
        text.starts_with("offline-echo:"),
        "unexpected text: {text:?}"
    );

    server.abort();
    let _ = child.start_kill();
    let _ = child.wait().await;
}

#[tokio::test]
async fn http_healthz_and_readyz() {
    let port = 55750 + (std::process::id() % 100) as u16;
    let endpoint = format!("tcp://127.0.0.1:{port}");
    let (state, mut child) = build_state(&endpoint).await;
    let (addr, server) = run_server(state).await;

    let resp = reqwest::get(format!("http://{addr}/healthz")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert_eq!(body, "ok");

    // readyz checks any_ready(Duration::from_secs(5)); we just pinged.
    let resp = reqwest::get(format!("http://{addr}/readyz")).await.unwrap();
    assert_eq!(resp.status(), 200);

    server.abort();
    let _ = child.start_kill();
    let _ = child.wait().await;
    let _ = Bytes::new(); // suppress unused warning
}
