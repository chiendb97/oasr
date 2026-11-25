// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! The launcher end to end, with stand-ins for the replicas: axum and raw h2
//! servers for the router, and a small Python HTTP server for the supervisor
//! (a real replica loads a model; what the supervisor needs from one is a
//! process that reports its ports and answers `/readyz`).

use std::collections::HashSet;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message as WsMessage, WebSocketUpgrade};
use axum::routing::{get, post};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use http::{header, HeaderMap, HeaderValue, Request, Response, StatusCode};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch, Semaphore};

use super::plan::ReplicaSpec;
use super::replica::{BoundAddrs, Replica};
use super::router::{self, full, get_text, GrpcConnection, ProxyClient, Router};
use super::supervisor::{self, FirstStarts, LaunchContext, LogSinks};

async fn wait_until(what: &str, within: Duration, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + within;
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn ready_replica(index: usize, http: SocketAddr, grpc: SocketAddr) -> Arc<Replica> {
    let r = Replica::new(ReplicaSpec {
        index,
        device: index.to_string(),
    });
    r.set_addrs(Some(BoundAddrs {
        pid: 1000 + index as u32,
        http,
        grpc,
    }));
    r.set_ready(true);
    r
}

/// An address nothing listens on.
async fn closed_port() -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    l.local_addr().unwrap()
}

// ---- stand-in replicas ------------------------------------------------------

/// An HTTP replica: `/who` names it, `/hold` answers once `gate` gives a
/// permit, `/drip` sends its head and a first chunk at once and the rest
/// after a permit (an SSE transcription's shape), `/echo` returns its body,
/// `/ws` echoes WebSocket text.
async fn http_replica(id: usize, gate: Arc<Semaphore>) -> SocketAddr {
    let drip_gate = Arc::clone(&gate);
    let app = axum::Router::new()
        .route("/readyz", get(|| async { "ready" }))
        .route("/who", get(move || async move { id.to_string() }))
        .route(
            "/hold",
            get(move || async move {
                gate.acquire().await.unwrap().forget();
                id.to_string()
            }),
        )
        .route(
            "/drip",
            get(move || async move {
                let gate = Arc::clone(&drip_gate);
                let rest = async move {
                    gate.acquire().await.unwrap().forget();
                    Ok::<_, Infallible>(Bytes::from(format!("{id}\n")))
                };
                let chunks = futures::stream::once(async { Ok(Bytes::from("first\n")) })
                    .chain(futures::stream::once(rest));
                axum::body::Body::from_stream(chunks)
            }),
        )
        .route(
            "/echo",
            post(|body: Bytes| async move { body })
                .layer(axum::extract::DefaultBodyLimit::disable()),
        )
        .route(
            "/metrics",
            get(move || async move {
                format!("# HELP fake_up Up\n# TYPE fake_up gauge\nfake_up{{engine=\"e\"}} {id}\n")
            }),
        )
        .route(
            "/ws",
            get(|ws: WebSocketUpgrade| async move {
                ws.on_upgrade(|mut socket| async move {
                    while let Some(Ok(msg)) = socket.recv().await {
                        if let WsMessage::Text(t) = msg {
                            let _ = socket.send(WsMessage::Text(format!("echo {t}"))).await;
                        }
                    }
                })
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    addr
}

/// A gRPC-shaped replica: echoes the request body, then trailers naming it.
async fn grpc_replica(id: usize) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                let svc = hyper::service::service_fn(move |req: Request<Incoming>| async move {
                    let body = req.into_body().collect().await.unwrap().to_bytes();
                    let mut trailers = HeaderMap::new();
                    trailers.insert("grpc-status", HeaderValue::from_static("0"));
                    trailers.insert("x-replica", HeaderValue::from(id));
                    let frames = futures::stream::iter([
                        Ok::<_, Infallible>(Frame::data(body)),
                        Ok(Frame::trailers(trailers)),
                    ]);
                    let mut resp = Response::new(StreamBody::new(frames));
                    resp.headers_mut().insert(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("application/grpc"),
                    );
                    Ok::<_, Infallible>(resp)
                });
                let _ = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                    .serve_connection(TokioIo::new(stream), svc)
                    .await;
            });
        }
    });
    addr
}

// ---- the router's two ports -------------------------------------------------

async fn serve_http_port(router: Arc<Router>) -> (SocketAddr, watch::Sender<bool>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = watch::channel(false);
    tokio::spawn(router::serve_listener(
        listener,
        rx,
        auto::Builder::new(TokioExecutor::new()),
        move || {
            let router = Arc::clone(&router);
            move |req| Arc::clone(&router).serve_http(req)
        },
    ));
    (addr, tx)
}

async fn serve_grpc_port(
    router: Arc<Router>,
) -> (SocketAddr, tonic_health::server::HealthReporter) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (reporter, server) = tonic_health::server::health_reporter();
    let health = router::health_service(server);
    let (_tx, rx) = watch::channel(false);
    tokio::spawn(async move {
        let _tx = _tx;
        router::serve_listener(
            listener,
            rx,
            auto::Builder::new(TokioExecutor::new()).http2_only(),
            move || {
                let conn = GrpcConnection::new(Arc::clone(&router), health.clone());
                move |req| Arc::clone(&conn).serve(req)
            },
        )
        .await
    });
    (addr, reporter)
}

async fn fetch(
    client: &ProxyClient,
    addr: SocketAddr,
    path: &str,
) -> (StatusCode, HeaderMap, String) {
    let req = Request::get(format!("http://{addr}{path}"))
        .body(router::empty())
        .unwrap();
    let resp = client.request(req).await.unwrap();
    let (parts, body) = resp.into_parts();
    let body = body.collect().await.unwrap().to_bytes();
    (
        parts.status,
        parts.headers,
        String::from_utf8_lossy(&body).into(),
    )
}

fn total_inflight(replicas: &[Arc<Replica>]) -> usize {
    replicas.iter().map(|r| r.inflight()).sum()
}

// ---- HTTP -------------------------------------------------------------------

#[tokio::test]
async fn requests_on_one_connection_are_balanced_per_request() {
    let gate = Arc::new(Semaphore::new(0));
    let grpc = closed_port().await;
    let replicas = vec![
        ready_replica(0, http_replica(0, Arc::clone(&gate)).await, grpc),
        ready_replica(1, http_replica(1, Arc::clone(&gate)).await, grpc),
    ];
    let router = Arc::new(Router::new(replicas.clone(), router::proxy_client(), None));
    let (addr, _stop) = serve_http_port(Arc::clone(&router)).await;
    // One pooled client, used sequentially: the same keep-alive connection.
    let client = router::proxy_client();

    // Idle: equal loads take turns.
    let mut seen = HashSet::new();
    for _ in 0..4 {
        seen.insert(get_text(&client, addr, "/who").await.unwrap());
    }
    assert_eq!(seen, HashSet::from(["0".to_string(), "1".to_string()]));

    // One request held open on a replica: everything else goes to the other.
    let held = {
        let client = client.clone();
        tokio::spawn(async move { get_text(&client, addr, "/hold").await.unwrap() })
    };
    wait_until("the held request", Duration::from_secs(5), || {
        total_inflight(&replicas) == 1
    })
    .await;
    let busy = replicas.iter().position(|r| r.inflight() == 1).unwrap();
    for _ in 0..6 {
        let who = get_text(&client, addr, "/who").await.unwrap();
        assert_eq!(
            who,
            (1 - busy).to_string(),
            "a request went to the busy replica"
        );
    }
    gate.add_permits(1);
    assert_eq!(held.await.unwrap(), busy.to_string());
    // The load is released once the response body is done.
    wait_until("the load to drain", Duration::from_secs(5), || {
        total_inflight(&replicas) == 0
    })
    .await;
}

#[tokio::test]
async fn a_streaming_response_is_load_until_its_last_byte() {
    let gate = Arc::new(Semaphore::new(0));
    let grpc = closed_port().await;
    let replicas = vec![
        ready_replica(0, http_replica(0, Arc::clone(&gate)).await, grpc),
        ready_replica(1, http_replica(1, Arc::clone(&gate)).await, grpc),
    ];
    let router = Arc::new(Router::new(replicas.clone(), router::proxy_client(), None));
    let (addr, _stop) = serve_http_port(Arc::clone(&router)).await;
    let client = router::proxy_client();

    let req = Request::get(format!("http://{addr}/drip"))
        .body(router::empty())
        .unwrap();
    let mut body = client.request(req).await.unwrap().into_body();
    // The head and the first chunk are here; the stream is still open.
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert_eq!(&first[..], b"first\n");
    assert_eq!(
        total_inflight(&replicas),
        1,
        "an open response stream is load"
    );
    let busy = replicas.iter().position(|r| r.inflight() == 1).unwrap();
    for _ in 0..4 {
        let who = get_text(&client, addr, "/who").await.unwrap();
        assert_eq!(
            who,
            (1 - busy).to_string(),
            "a request went to the streaming replica"
        );
    }
    gate.add_permits(1);
    let rest = body.collect().await.unwrap().to_bytes();
    assert_eq!(rest, format!("{busy}\n"));
    wait_until("the stream's load to drain", Duration::from_secs(5), || {
        total_inflight(&replicas) == 0
    })
    .await;
}

#[tokio::test]
async fn a_large_body_streams_through_intact() {
    let gate = Arc::new(Semaphore::new(0));
    let replicas = vec![ready_replica(
        0,
        http_replica(0, gate).await,
        closed_port().await,
    )];
    let router = Arc::new(Router::new(replicas, router::proxy_client(), None));
    let (addr, _stop) = serve_http_port(router).await;
    let payload: Vec<u8> = (0..5_000_000u32).map(|i| (i % 251) as u8).collect();
    // Chunked, with no length: the router has to stream it, not size it.
    let chunks: Vec<Result<Frame<Bytes>, Infallible>> = payload
        .chunks(64 * 1024)
        .map(|c| Ok(Frame::data(Bytes::copy_from_slice(c))))
        .collect();
    let body = StreamBody::new(futures::stream::iter(chunks))
        .map_err(|e| match e {})
        .boxed_unsync();
    let req = Request::post(format!("http://{addr}/echo"))
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(body)
        .unwrap();
    let resp = router::proxy_client().request(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let got = resp.into_body().collect().await.unwrap().to_bytes();
    assert!(got == payload, "echo differs ({} bytes back)", got.len());
}

#[tokio::test]
async fn a_websocket_session_is_spliced_and_counted_for_its_life() {
    let gate = Arc::new(Semaphore::new(0));
    let replicas = vec![ready_replica(
        0,
        http_replica(0, gate).await,
        closed_port().await,
    )];
    let router = Arc::new(Router::new(replicas.clone(), router::proxy_client(), None));
    let (addr, _stop) = serve_http_port(Arc::clone(&router)).await;

    let (mut ws, resp) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .expect("upgrade through the router");
    assert_eq!(resp.status(), StatusCode::SWITCHING_PROTOCOLS);
    for word in ["one", "two"] {
        ws.send(tokio_tungstenite::tungstenite::Message::Text(word.into()))
            .await
            .unwrap();
        let back = ws.next().await.unwrap().unwrap();
        assert_eq!(back.into_text().unwrap(), format!("echo {word}"));
    }
    assert_eq!(replicas[0].inflight(), 1, "an open session is load");
    assert_eq!(router.tunnels.open(), 1);
    ws.close(None).await.unwrap();
    drop(ws);
    wait_until("the session to close", Duration::from_secs(5), || {
        router.tunnels.open() == 0 && replicas[0].inflight() == 0
    })
    .await;
}

#[tokio::test]
async fn metrics_merge_every_replica_under_one_type_line() {
    let gate = Arc::new(Semaphore::new(0));
    let grpc = closed_port().await;
    let replicas = vec![
        ready_replica(0, http_replica(0, Arc::clone(&gate)).await, grpc),
        ready_replica(1, http_replica(1, gate).await, grpc),
    ];
    let router = Arc::new(Router::new(replicas, router::proxy_client(), None));
    let (addr, _stop) = serve_http_port(router).await;
    let (status, _, body) = fetch(&router::proxy_client(), addr, "/metrics").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.matches("# TYPE fake_up gauge").count(), 1, "{body}");
    assert!(
        body.contains("fake_up{replica=\"0\",gpu=\"0\",engine=\"e\"} 0"),
        "{body}"
    );
    assert!(
        body.contains("fake_up{replica=\"1\",gpu=\"1\",engine=\"e\"} 1"),
        "{body}"
    );
}

#[tokio::test]
async fn without_a_ready_replica_the_router_answers_503_and_is_not_ready() {
    let gate = Arc::new(Semaphore::new(0));
    let replicas = vec![ready_replica(
        0,
        http_replica(0, gate).await,
        closed_port().await,
    )];
    let router = Arc::new(Router::new(replicas.clone(), router::proxy_client(), None));
    let (addr, _stop) = serve_http_port(Arc::clone(&router)).await;
    let client = router::proxy_client();
    assert_eq!(fetch(&client, addr, "/readyz").await.0, StatusCode::OK);

    replicas[0].set_ready(false);
    let (status, headers, body) = fetch(&client, addr, "/who").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(headers.get(header::RETRY_AFTER).unwrap(), "1");
    assert!(body.contains("\"UNAVAILABLE\""), "{body}");
    assert_eq!(
        fetch(&client, addr, "/readyz").await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(fetch(&client, addr, "/healthz").await.0, StatusCode::OK);

    // A shutdown stops readiness even with replicas up.
    replicas[0].set_ready(true);
    router.drain();
    assert_eq!(
        fetch(&client, addr, "/readyz").await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn a_replica_that_refuses_connections_is_taken_out_of_rotation() {
    let dead = ready_replica(0, closed_port().await, closed_port().await);
    let router = Arc::new(Router::new(
        vec![Arc::clone(&dead)],
        router::proxy_client(),
        None,
    ));
    let (addr, _stop) = serve_http_port(router).await;
    let (status, _, _) = fetch(&router::proxy_client(), addr, "/who").await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(!dead.is_ready());
    assert_eq!(dead.inflight(), 0);
}

// ---- gRPC -------------------------------------------------------------------

async fn rpc(
    send: &mut hyper::client::conn::http2::SendRequest<router::ProxyBody>,
    addr: SocketAddr,
    payload: &'static [u8],
) -> (HeaderMap, Bytes, Option<HeaderMap>) {
    let req = Request::post(format!("http://{addr}/oasr.speech.v1.Speech/Recognize"))
        .header(header::CONTENT_TYPE, "application/grpc")
        .header(header::TE, "trailers")
        .body(full(payload))
        .unwrap();
    let resp = send.send_request(req).await.unwrap();
    let (parts, body) = resp.into_parts();
    let collected = body.collect().await.unwrap();
    let trailers = collected.trailers().cloned();
    (parts.headers, collected.to_bytes(), trailers)
}

async fn h2_client(addr: SocketAddr) -> hyper::client::conn::http2::SendRequest<router::ProxyBody> {
    let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (send, conn) = hyper::client::conn::http2::Builder::new(TokioExecutor::new())
        .handshake(TokioIo::new(tcp))
        .await
        .unwrap();
    tokio::spawn(conn);
    send
}

#[tokio::test]
async fn rpcs_on_one_connection_spread_and_keep_their_trailers() {
    let http = closed_port().await;
    let replicas = vec![
        ready_replica(0, http, grpc_replica(0).await),
        ready_replica(1, http, grpc_replica(1).await),
    ];
    let router = Arc::new(Router::new(replicas.clone(), router::proxy_client(), None));
    let (addr, _reporter) = serve_grpc_port(router).await;
    let mut send = h2_client(addr).await;
    let mut seen = HashSet::new();
    for _ in 0..4 {
        let (headers, body, trailers) = rpc(&mut send, addr, b"\0\0\0\0\x03abc").await;
        assert_eq!(
            headers.get(header::CONTENT_TYPE).unwrap(),
            "application/grpc"
        );
        assert_eq!(&body[..], b"\0\0\0\0\x03abc");
        let trailers = trailers.expect("trailers came through");
        assert_eq!(trailers.get("grpc-status").unwrap(), "0");
        seen.insert(
            trailers
                .get("x-replica")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
        );
    }
    assert_eq!(
        seen.len(),
        2,
        "one client connection reached one replica only"
    );
    assert_eq!(total_inflight(&replicas), 0);
}

#[tokio::test]
async fn an_unreachable_replica_is_skipped_before_the_rpc_is_sent() {
    let http = closed_port().await;
    let dead = ready_replica(0, http, closed_port().await);
    let live = ready_replica(1, http, grpc_replica(1).await);
    // Load on the live one, so the dead one is picked first.
    let _load = live.track();
    let router = Arc::new(Router::new(
        vec![Arc::clone(&dead), Arc::clone(&live)],
        router::proxy_client(),
        None,
    ));
    let (addr, _reporter) = serve_grpc_port(router).await;
    let mut send = h2_client(addr).await;
    let (_, body, trailers) = rpc(&mut send, addr, b"x").await;
    assert_eq!(&body[..], b"x");
    assert_eq!(trailers.unwrap().get("x-replica").unwrap(), "1");
    assert!(!dead.is_ready());
}

#[tokio::test]
async fn without_a_replica_an_rpc_is_unavailable_and_health_is_answered_here() {
    let dead = Replica::new(ReplicaSpec {
        index: 0,
        device: "0".into(),
    });
    let router = Arc::new(Router::new(vec![dead], router::proxy_client(), None));
    let (addr, mut reporter) = serve_grpc_port(router).await;
    let mut send = h2_client(addr).await;
    let (headers, body, _) = rpc(&mut send, addr, b"x").await;
    assert_eq!(headers.get("grpc-status").unwrap(), "14");
    assert!(body.is_empty());

    reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut health = tonic_health::pb::health_client::HealthClient::new(channel);
    let status = health
        .check(tonic_health::pb::HealthCheckRequest {
            service: String::new(),
        })
        .await
        .unwrap()
        .into_inner()
        .status;
    assert_eq!(
        status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32
    );
}

// ---- the supervisor, with a stand-in replica process -------------------------

/// Reports its ports the way `oasr-server --report-bound-addrs` does, answers
/// `/readyz`, exits on `/exit`, stops being ready on `/wedge`, exits 0 on
/// SIGTERM, and logs its start and report times with its device to `@LOG@`
/// (when not empty), reporting `@DELAY@` seconds after it starts.
const FAKE_REPLICA: &str = r#"
import http.server, json, os, signal, socket, sys, threading, time
log = "@LOG@"
def note(what):
    if log:
        with open(log, "a") as f:
            f.write("%s %d %s %f\n" % (what, os.getpid(), os.environ.get("CUDA_VISIBLE_DEVICES"), time.time()))
note("start")
args = sys.argv[1:]
report = args[args.index("--report-bound-addrs") + 1]
time.sleep(@DELAY@)
state = {"ready": True}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        code = 200
        if self.path == "/readyz":
            code = 200 if state["ready"] else 503
        elif self.path == "/exit":
            threading.Timer(0.05, lambda: os._exit(7)).start()
        elif self.path == "/wedge":
            state["ready"] = False
        body = str(os.getpid()).encode()
        self.send_response(code)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
g = socket.socket()
g.bind(("127.0.0.1", 0))
g.listen()
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
print("fake replica up", flush=True)
tmp = report + ".tmp"
with open(tmp, "w") as f:
    json.dump({"pid": os.getpid(), "http": "127.0.0.1:%d" % srv.server_port,
               "grpc": "127.0.0.1:%d" % g.getsockname()[1]}, f)
os.replace(tmp, report)
note("report")
srv.serve_forever()
"#;

struct Harness {
    ctx: Arc<LaunchContext>,
    fatal: mpsc::UnboundedReceiver<anyhow::Error>,
    _dir: super::RunDir,
}

fn fake_replica(log: &str, delay_s: f64) -> String {
    FAKE_REPLICA
        .replace("@LOG@", log)
        .replace("@DELAY@", &delay_s.to_string())
}

fn harness(replicas: usize, script: &str, wedged_after: Duration) -> Harness {
    let (fatal_tx, fatal) = mpsc::unbounded_channel();
    let dir = super::RunDir::create_in(
        std::env::temp_dir().join(format!("oasr-launcher-test-{}", uuid_like())),
    )
    .unwrap();
    let mut ctx = LaunchContext::new(
        "python3".into(),
        vec!["-c".into(), script.into()],
        vec!["oasr-server".into(), "--gpus".into(), "0".into()],
        dir.0.clone(),
        router::proxy_client(),
        LogSinks::start(false),
        Duration::from_secs(5),
        FirstStarts::new(replicas, fatal_tx),
    );
    ctx.wedged_after = wedged_after;
    Harness {
        ctx: Arc::new(ctx),
        fatal,
        _dir: dir,
    }
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    )
}

fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 checks for existence only.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[tokio::test]
async fn a_replica_that_exits_or_wedges_is_restarted_and_stopped_on_request() {
    let h = harness(1, &fake_replica("", 0.0), Duration::from_millis(600));
    let replica = Replica::new(ReplicaSpec {
        index: 0,
        device: "0".into(),
    });
    let (stop_tx, stop_rx) = watch::channel(false);
    let task = tokio::spawn(supervisor::supervise(
        Arc::clone(&replica),
        Arc::clone(&h.ctx),
        stop_rx,
    ));
    let client = router::proxy_client();
    let up = Duration::from_secs(20);
    wait_until("the first start", up, || replica.routable().is_some()).await;
    let first = replica.addrs().unwrap();

    // It exits: the supervisor starts another process.
    let _ = get_text(&client, first.http, "/exit").await;
    wait_until("a restart after an exit", up, || {
        replica.routable().is_some_and(|a| a.pid != first.pid)
    })
    .await;
    assert!(!alive(first.pid), "the exited process was reaped");
    let second = replica.addrs().unwrap();

    // It stops answering /readyz while running: stopped, and started again.
    let _ = get_text(&client, second.http, "/wedge").await;
    wait_until("the wedged replica to leave the rotation", up, || {
        replica.routable().is_none()
    })
    .await;
    wait_until("a restart after a wedge", up, || {
        replica.routable().is_some_and(|a| a.pid != second.pid)
    })
    .await;
    let third = replica.addrs().unwrap();

    // Stopping the launcher stops the process.
    stop_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .expect("the supervisor returns")
        .unwrap();
    wait_until("the replica to be reaped", Duration::from_secs(5), || {
        !alive(third.pid)
    })
    .await;
    assert!(replica.routable().is_none());
    let mut fatal = h.fatal;
    assert!(
        fatal.try_recv().is_err(),
        "a replica that started once is not fatal"
    );
}

#[tokio::test]
async fn when_every_first_start_fails_the_launcher_is_told_to_stop() {
    let h = harness(
        2,
        "import sys; print('bad checkpoint', file=sys.stderr); sys.exit(3)",
        supervisor::WEDGED_AFTER,
    );
    let (stop_tx, stop_rx) = watch::channel(false);
    let tasks: Vec<_> = (0..2)
        .map(|i| {
            tokio::spawn(supervisor::supervise(
                Replica::new(ReplicaSpec {
                    index: i,
                    device: i.to_string(),
                }),
                Arc::clone(&h.ctx),
                stop_rx.clone(),
            ))
        })
        .collect();
    let mut fatal = h.fatal;
    let err = tokio::time::timeout(Duration::from_secs(20), fatal.recv())
        .await
        .expect("a fatal error is reported")
        .unwrap();
    assert!(err.to_string().contains("exited with status 3"), "{err}");
    stop_tx.send(true).unwrap();
    for t in tasks {
        tokio::time::timeout(Duration::from_secs(10), t)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn replicas_sharing_a_gpu_start_one_after_another() {
    let log = std::env::temp_dir().join(format!("oasr-launcher-order-{}.log", uuid_like()));
    let _ = std::fs::remove_file(&log);
    let script = fake_replica(log.to_str().unwrap(), 0.3);
    let h = harness(3, &script, supervisor::WEDGED_AFTER);
    let (stop_tx, stop_rx) = watch::channel(false);
    // Two on device 7, one on device 8.
    let replicas: Vec<_> = [(0, "7"), (1, "7"), (2, "8")]
        .into_iter()
        .map(|(index, device)| {
            Replica::new(ReplicaSpec {
                index,
                device: device.into(),
            })
        })
        .collect();
    let tasks: Vec<_> = replicas
        .iter()
        .map(|r| {
            tokio::spawn(supervisor::supervise(
                Arc::clone(r),
                Arc::clone(&h.ctx),
                stop_rx.clone(),
            ))
        })
        .collect();
    wait_until("all three ready", Duration::from_secs(30), || {
        replicas.iter().all(|r| r.routable().is_some())
    })
    .await;
    stop_tx.send(true).unwrap();
    for t in tasks {
        tokio::time::timeout(Duration::from_secs(10), t)
            .await
            .unwrap()
            .unwrap();
    }

    // (event, device, time) per line.
    let events: Vec<(String, String, f64)> = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f[0].to_string(), f[2].to_string(), f[3].parse().unwrap())
        })
        .collect();
    let _ = std::fs::remove_file(&log);
    let on = |dev: &str, what: &str| -> Vec<f64> {
        let mut t: Vec<f64> = events
            .iter()
            .filter(|(e, d, _)| e == what && d == dev)
            .map(|(_, _, t)| *t)
            .collect();
        t.sort_by(f64::total_cmp);
        t
    };
    let (starts7, reports7) = (on("7", "start"), on("7", "report"));
    assert_eq!(starts7.len(), 2, "{events:?}");
    assert!(
        starts7[1] >= reports7[0],
        "the second replica on GPU 7 started before the first was up: {events:?}"
    );
    // A different GPU does not wait: device 8 started before 7's second.
    assert!(on("8", "start")[0] < starts7[1], "{events:?}");
}
