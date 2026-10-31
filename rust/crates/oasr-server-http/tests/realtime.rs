// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `/v1/realtime` over a real socket.
//!
//! A WebSocket session outlives the upgrade request, so the route's HTTP layers
//! (the concurrency limit, the request timeout) never saw it: a client that
//! configured and went quiet held an engine stream forever.  These drive the
//! router over TCP, because the bounds they check live on the socket.
//!
//! The pool has no workers, as in `routes.rs`, so opening an engine stream
//! fails at once — which is how these tell *when* the server tried.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use oasr_asr::DecodePool;
use oasr_engine_client::EnginePool;
use oasr_server_http::{build_router, RouterLimits, ServerState, ServiceMode, SessionTracker};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Client = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Long enough for any reply on loopback, short against the test timeout.
const REPLY: Duration = Duration::from_secs(5);

async fn serve(mode: ServiceMode, limits: RouterLimits) -> SocketAddr {
    let state = Arc::new(ServerState {
        pool: Arc::new(EnginePool::new(Vec::new())),
        prometheus: None,
        service_mode: mode,
        sample_rate: 16_000,
        max_body_bytes: 1024 * 1024,
        max_audio_samples: Some(16_000 * 60),
        decode_pool: DecodePool::new(2),
        served_model_names: Vec::new(),
        model_id: "/ckpt/test".into(),
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = build_router(state, limits);
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    addr
}

/// Boxed: tungstenite's error is 136 bytes, past clippy's `result_large_err`.
async fn connect(addr: SocketAddr) -> Result<Client, Box<tungstenite::Error>> {
    tokio_tungstenite::connect_async(format!("ws://{addr}/v1/realtime"))
        .await
        .map(|(ws, _)| ws)
        .map_err(Box::new)
}

/// Open a session and complete the handshake.
async fn configured(addr: SocketAddr) -> Client {
    let mut ws = connect(addr).await.expect("the upgrade was refused");
    ws.send(Message::text(r#"{"type":"session.update","session":{}}"#))
        .await
        .unwrap();
    let created = next_event(&mut ws).await.expect("no handshake reply");
    assert_eq!(
        created["type"], "transcription_session.created",
        "{created}"
    );
    ws
}

/// The next JSON event, or `None` once the server has closed the session.
async fn next_event(ws: &mut Client) -> Option<Value> {
    loop {
        match tokio::time::timeout(REPLY, ws.next()).await {
            Err(_) => panic!("the server neither replied nor closed within {REPLY:?}"),
            Ok(Some(Ok(Message::Text(t)))) => return Some(serde_json::from_str(&t).unwrap()),
            Ok(Some(Ok(Message::Close(_)))) | Ok(Some(Err(_))) | Ok(None) => return None,
            Ok(Some(Ok(_))) => continue,
        }
    }
}

/// Assert the server sends nothing and keeps the socket open for `quiet`.
async fn assert_quiet(ws: &mut Client, quiet: Duration) {
    if let Ok(msg) = tokio::time::timeout(quiet, ws.next()).await {
        panic!("expected the session to wait, got {msg:?}");
    }
}

fn idle_limits(idle: Duration) -> RouterLimits {
    RouterLimits {
        stream_idle_timeout: Some(idle),
        ..Default::default()
    }
}

#[tokio::test]
async fn a_configured_session_that_goes_quiet_is_ended() {
    for mode in [ServiceMode::Streaming, ServiceMode::Offline] {
        let addr = serve(mode, idle_limits(Duration::from_millis(200))).await;
        let mut ws = configured(addr).await;
        let err = next_event(&mut ws).await.expect("ended without saying why");
        assert_eq!(err["type"], "error", "{mode:?}: {err}");
        assert!(
            err["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no client message"),
            "{mode:?}: {err}"
        );
        assert!(next_event(&mut ws).await.is_none(), "{mode:?}: still open");
    }
}

#[tokio::test]
async fn a_socket_that_never_speaks_is_ended() {
    let addr = serve(
        ServiceMode::Streaming,
        idle_limits(Duration::from_millis(200)),
    )
    .await;
    let mut ws = connect(addr).await.unwrap();
    let err = next_event(&mut ws).await.expect("ended without saying why");
    assert_eq!(err["type"], "error", "{err}");
}

/// A handshake alone must not take an engine stream; the first audio does.
#[tokio::test]
async fn the_engine_stream_opens_at_the_first_audio() {
    let addr = serve(ServiceMode::Streaming, RouterLimits::default()).await;
    let mut ws = configured(addr).await;
    // The pool refuses every open, so a session that opened at the handshake
    // would already be over.
    assert_quiet(&mut ws, Duration::from_millis(300)).await;

    ws.send(Message::binary(vec![0u8; 3200])).await.unwrap();
    let err = next_event(&mut ws)
        .await
        .expect("refused without saying why");
    assert_eq!(err["type"], "error", "{err}");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("submit failed"),
        "{err}"
    );
}

/// The upgrade returns at once, so the session itself has to hold the permit.
#[tokio::test]
async fn a_session_holds_an_inflight_permit_for_its_life() {
    let addr = serve(
        ServiceMode::Streaming,
        RouterLimits {
            inflight_permits: Some(Arc::new(Semaphore::new(1))),
            ..Default::default()
        },
    )
    .await;
    let mut first = configured(addr).await;

    match connect(addr).await.map_err(|e| *e) {
        Err(tungstenite::Error::Http(resp)) => assert_eq!(resp.status(), 503),
        other => panic!("a second session got past a budget of one: {other:?}"),
    }

    first.close(None).await.unwrap();
    while next_event(&mut first).await.is_some() {}
    // The permit returns when the server's session task ends, a moment after
    // the close is acknowledged.
    let deadline = tokio::time::Instant::now() + REPLY;
    loop {
        match connect(addr).await {
            Ok(_) => break,
            Err(_) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(20)).await
            }
            Err(e) => panic!("the permit was never returned: {e:?}"),
        }
    }
}

/// Limits whose session tracker the test keeps, to drive a shutdown drain.
fn tracked() -> (RouterLimits, SessionTracker) {
    let sessions = SessionTracker::default();
    let limits = RouterLimits {
        sessions: sessions.clone(),
        ..Default::default()
    };
    (limits, sessions)
}

/// The close frame that ends the session, skipping any events before it.
async fn close_code(ws: &mut Client) -> Option<u16> {
    loop {
        match tokio::time::timeout(REPLY, ws.next()).await {
            Err(_) => panic!("the server neither closed nor hung up within {REPLY:?}"),
            Ok(Some(Ok(Message::Close(frame)))) => return frame.map(|f| f.code.into()),
            Ok(Some(Ok(_))) => continue,
            Ok(Some(Err(_))) | Ok(None) => return None,
        }
    }
}

/// An upgraded socket is not an HTTP request any more, so the listener's
/// graceful shutdown never waited for one: the process exited under every open
/// session, mid-utterance.  The drain waits for it, and lets it finish.
#[tokio::test]
async fn a_drain_waits_for_an_open_session_and_refuses_new_ones() {
    let (limits, sessions) = tracked();
    let addr = serve(ServiceMode::Offline, limits).await;
    let mut ws = configured(addr).await;
    ws.send(Message::binary(vec![0u8; 3200])).await.unwrap();
    assert_quiet(&mut ws, Duration::from_millis(100)).await;
    assert_eq!(sessions.len(), 1);

    sessions.drain();
    assert!(
        tokio::time::timeout(Duration::from_millis(300), sessions.wait())
            .await
            .is_err(),
        "the drain finished with a session still open"
    );
    match connect(addr).await.map_err(|e| *e) {
        Err(tungstenite::Error::Http(resp)) => assert_eq!(resp.status(), 503),
        other => panic!("a new session was opened during the drain: {other:?}"),
    }

    // The open session runs to its own end: more audio, then the commit,
    // which a pool with no workers refuses.
    ws.send(Message::binary(vec![0u8; 3200])).await.unwrap();
    ws.send(Message::text(r#"{"type":"input_audio_buffer.commit"}"#))
        .await
        .unwrap();
    let err = next_event(&mut ws).await.expect("no reply to the commit");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("submit failed"),
        "{err}"
    );
    tokio::time::timeout(REPLY, sessions.wait())
        .await
        .expect("the drain never finished after the session ended");
}

/// At the drain's deadline what is left is ended, and the client is told:
/// `1001 going away` is the close a client takes as "reconnect elsewhere".
#[tokio::test]
async fn a_cut_session_is_told_the_server_is_going_away() {
    let (limits, sessions) = tracked();
    let addr = serve(ServiceMode::Offline, limits).await;
    let mut ws = configured(addr).await;
    ws.send(Message::binary(vec![0u8; 3200])).await.unwrap();
    assert_quiet(&mut ws, Duration::from_millis(100)).await;

    sessions.drain();
    sessions.cut();
    let err = next_event(&mut ws).await.expect("cut without saying why");
    assert_eq!(err["type"], "error", "{err}");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("shutting down"),
        "{err}"
    );
    assert_eq!(close_code(&mut ws).await, Some(1001));
    tokio::time::timeout(REPLY, sessions.wait())
        .await
        .expect("the cut session never ended");
}
