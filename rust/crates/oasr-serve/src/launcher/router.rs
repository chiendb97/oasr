// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! The launcher's router: the server's two ports, forwarded to the replicas.
//!
//! It forwards at the request, not the connection, which is why it is an HTTP
//! proxy rather than `SO_REUSEPORT` or a TCP splice.  A gRPC client multiplexes
//! every RPC over one connection, and an HTTP client keeps its connections
//! alive: balanced per connection, either lands all its work on one GPU.  Here
//! each request, gRPC stream and realtime session goes to the ready replica
//! with the fewest open, ties rotated.
//!
//! * HTTP port: HTTP/1.1 (and h2c) downstream, pooled HTTP/1.1 to the replica.
//!   A WebSocket upgrade is forwarded, and once both sides have switched the
//!   two connections are spliced for the life of the session.
//! * gRPC port: h2c downstream; each downstream connection keeps its own h2
//!   connection per replica, so the stream limits and the framing work stay
//!   per client connection, as they were with the client on the replica.
//!   `grpc.health.v1.Health` is answered here, for the router as a whole.
//! * `/healthz`, `/readyz` and `/metrics` are answered here too: ready while
//!   any replica is, and one scrape merges every replica's series, labelled.
//!
//! Bodies stream through in both directions (an upload is never buffered), and
//! the replicas keep every limit and deadline: the router adds none.

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http::header::{self, HeaderMap, HeaderName, HeaderValue};
use http::{Method, Request, Response, StatusCode, Uri, Version};
use http_body_util::combinators::UnsyncBoxBody;
use http_body_util::{BodyExt, Empty, Full};
use hyper::body::{Body, Frame, Incoming, SizeHint};
use hyper::client::conn::http2;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use hyper_util::server::graceful::GracefulShutdown;
use metrics::Counter;
use metrics_exporter_prometheus::PrometheusHandle;
use oasr_metrics::{label, ROUTER_UNROUTED};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Notify};
use tracing::{debug, warn};

use super::exposition;
use super::replica::{BoundAddrs, InFlight, Replica};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
pub type ProxyBody = UnsyncBoxBody<Bytes, BoxError>;
pub type ProxyClient = Client<HttpConnector, ProxyBody>;

/// How long a replica gets to answer the launcher's own `/metrics` fetch.
const METRICS_FETCH_TIMEOUT: Duration = Duration::from_secs(2);
/// Connecting to a replica is a loopback connect: it is refused at once or
/// answered at once, and anything else means the replica is not serving.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// The pooled HTTP/1.1 client the router forwards the HTTP port with, and the
/// supervisor probes `/readyz` with.
pub fn proxy_client() -> ProxyClient {
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true);
    connector.set_connect_timeout(Some(CONNECT_TIMEOUT));
    Client::builder(TokioExecutor::new()).build(connector)
}

pub fn empty() -> ProxyBody {
    Empty::new().map_err(|e| match e {}).boxed_unsync()
}

pub(super) fn full(body: impl Into<Bytes>) -> ProxyBody {
    Full::new(body.into())
        .map_err(|e| match e {})
        .boxed_unsync()
}

fn boxed<B>(body: B) -> ProxyBody
where
    B: Body<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError>,
{
    body.map_err(Into::into).boxed_unsync()
}

fn text(status: StatusCode, body: &'static str) -> Response<ProxyBody> {
    let mut resp = Response::new(full(body));
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    resp
}

/// The replicas' JSON error envelope (`{"error": {code, status, message}}`),
/// which both the Google and the OpenAI clients in tree read.
fn http_error(status: StatusCode, message: String) -> Response<ProxyBody> {
    let body = serde_json::json!({
        "error": {"code": status.as_u16(), "status": "UNAVAILABLE", "message": message}
    });
    let mut resp = Response::new(full(body.to_string()));
    *resp.status_mut() = status;
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    if status == StatusCode::SERVICE_UNAVAILABLE {
        h.insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    }
    resp
}

/// A trailers-only gRPC response: status `UNAVAILABLE` (14), so a client's
/// retry policy treats it as the transient failure it is.
fn grpc_unavailable(message: &str) -> Response<ProxyBody> {
    let mut resp = Response::new(empty());
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/grpc"),
    );
    h.insert("grpc-status", HeaderValue::from_static("14"));
    if let Ok(v) = HeaderValue::from_str(&grpc_percent_encode(message)) {
        h.insert("grpc-message", v);
    }
    resp
}

/// `grpc-message` is percent-encoded: printable ASCII except `%` as is.
fn grpc_percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if (0x20..=0x7e).contains(&b) && b != b'%' {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// A response body that holds its request's [`InFlight`] until hyper is done
/// sending it -- the end of the body, or the trailers of a gRPC stream -- or
/// drops it.  That is when a request stops being load on its replica.
struct Tracked {
    inner: ProxyBody,
    _load: InFlight,
}

impl Body for Tracked {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

fn tracked(body: ProxyBody, load: InFlight) -> ProxyBody {
    Tracked {
        inner: body,
        _load: load,
    }
    .boxed_unsync()
}

/// The spliced WebSocket sessions still open, which a shutdown waits for:
/// once upgraded they belong to no HTTP connection hyper could drain.
#[derive(Default)]
pub struct Tunnels {
    open: AtomicUsize,
    closed: Notify,
}

impl Tunnels {
    pub fn open(&self) -> usize {
        self.open.load(Ordering::Acquire)
    }

    pub async fn wait_closed(&self) {
        loop {
            let closed = self.closed.notified();
            tokio::pin!(closed);
            closed.as_mut().enable();
            if self.open() == 0 {
                return;
            }
            closed.await;
        }
    }
}

struct TunnelGuard(Arc<Tunnels>);

impl Drop for TunnelGuard {
    fn drop(&mut self) {
        if self.0.open.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.closed.notify_waiters();
        }
    }
}

pub struct Router {
    replicas: Vec<Arc<Replica>>,
    client: ProxyClient,
    /// Where the next pick starts looking, so equal loads take turns.
    next: AtomicUsize,
    draining: AtomicBool,
    pub tunnels: Arc<Tunnels>,
    prometheus: Option<PrometheusHandle>,
    no_replica: Counter,
    replica_error: Counter,
}

impl Router {
    pub fn new(
        replicas: Vec<Arc<Replica>>,
        client: ProxyClient,
        prometheus: Option<PrometheusHandle>,
    ) -> Self {
        Self {
            replicas,
            client,
            next: AtomicUsize::new(0),
            draining: AtomicBool::new(false),
            tunnels: Arc::default(),
            prometheus,
            no_replica: metrics::counter!(ROUTER_UNROUTED, label::REASON => "no_replica"),
            replica_error: metrics::counter!(ROUTER_UNROUTED, label::REASON => "replica_error"),
        }
    }

    /// Stop reporting ready: a shutdown has begun.
    pub fn drain(&self) {
        self.draining.store(true, Ordering::Release);
    }

    /// Ready while any replica is and no shutdown has begun -- the same "any
    /// worker" rule an in-process `EnginePool` applies.
    pub fn is_ready(&self) -> bool {
        !self.draining.load(Ordering::Acquire) && self.replicas.iter().any(|r| r.is_ready())
    }

    /// The ready replica with the fewest in flight, skipping `exclude`.
    pub fn pick(&self, exclude: &[usize]) -> Option<(Arc<Replica>, BoundAddrs)> {
        let n = self.replicas.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed) % n.max(1);
        let mut best: Option<(usize, usize, BoundAddrs)> = None;
        for k in 0..n {
            let i = (start + k) % n;
            let r = &self.replicas[i];
            if exclude.contains(&r.index()) {
                continue;
            }
            let Some(addrs) = r.routable() else { continue };
            let load = r.inflight();
            if best.as_ref().is_none_or(|(b, _, _)| load < *b) {
                best = Some((load, i, addrs));
            }
        }
        best.map(|(_, i, a)| (Arc::clone(&self.replicas[i]), a))
    }

    // ---- HTTP port ---------------------------------------------------------

    pub async fn serve_http(
        self: Arc<Self>,
        req: Request<Incoming>,
    ) -> Result<Response<ProxyBody>, Infallible> {
        Ok(match (req.method(), req.uri().path()) {
            (&Method::GET, "/healthz") => text(StatusCode::OK, "ok"),
            (&Method::GET, "/readyz") => {
                if self.is_ready() {
                    text(StatusCode::OK, "ready")
                } else {
                    text(StatusCode::SERVICE_UNAVAILABLE, "not ready")
                }
            }
            (&Method::GET, "/metrics") => self.render_metrics().await,
            _ => self.forward_http(req).await,
        })
    }

    async fn forward_http(&self, mut req: Request<Incoming>) -> Response<ProxyBody> {
        let Some((replica, addrs)) = self.pick(&[]) else {
            self.no_replica.increment(1);
            return http_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no replica is ready".into(),
            );
        };
        let load = replica.track();
        let upgrade = wants_upgrade(req.headers());
        // Taken before the request is forwarded: it is the downstream
        // connection's half of the switch.
        let downstream = upgrade.then(|| hyper::upgrade::on(&mut req));
        let version = req.version();
        let (mut parts, body) = req.into_parts();
        if parts.headers.get(header::HOST).is_none() {
            // An h2c request names its host in `:authority`, which HTTP/1.1
            // carries as `Host`.
            if let Some(v) = parts
                .uri
                .authority()
                .and_then(|a| HeaderValue::from_str(a.as_str()).ok())
            {
                parts.headers.insert(header::HOST, v);
            }
        }
        parts.uri = upstream_uri(addrs.http, &parts.uri);
        parts.version = Version::HTTP_11;
        strip_hop_by_hop(&mut parts.headers, upgrade);
        let forwarded = Request::from_parts(parts, boxed(body));

        let mut resp = match self.client.request(forwarded).await {
            Ok(resp) => resp,
            Err(e) => {
                if e.is_connect() {
                    // Off the rotation until its next /readyz says otherwise.
                    replica.set_ready(false);
                }
                self.replica_error.increment(1);
                warn!(replica = replica.index(), error = %e, "forwarding to a replica failed");
                return http_error(
                    StatusCode::BAD_GATEWAY,
                    format!("replica {} did not answer: {e}", replica.index()),
                );
            }
        };
        if resp.status() == StatusCode::SWITCHING_PROTOCOLS {
            if let Some(downstream) = downstream {
                let upstream = hyper::upgrade::on(&mut resp);
                self.splice(downstream, upstream, load);
            }
            let (mut parts, _) = resp.into_parts();
            parts.version = version;
            return Response::from_parts(parts, empty());
        }
        let (mut parts, body) = resp.into_parts();
        strip_hop_by_hop(&mut parts.headers, false);
        parts.version = version;
        Response::from_parts(parts, tracked(boxed(body), load))
    }

    /// Join an upgraded client connection to its replica's for the session's
    /// life.  The session stays counted against the replica until it ends.
    fn splice(
        &self,
        downstream: hyper::upgrade::OnUpgrade,
        upstream: hyper::upgrade::OnUpgrade,
        load: InFlight,
    ) {
        self.tunnels.open.fetch_add(1, Ordering::AcqRel);
        let guard = TunnelGuard(Arc::clone(&self.tunnels));
        tokio::spawn(async move {
            let _guard = guard;
            let _load = load;
            match tokio::try_join!(downstream, upstream) {
                Ok((down, up)) => {
                    let mut down = TokioIo::new(down);
                    let mut up = TokioIo::new(up);
                    if let Err(e) = tokio::io::copy_bidirectional(&mut down, &mut up).await {
                        debug!("upgraded session ended: {e}");
                    }
                }
                Err(e) => debug!("upgrade did not complete: {e}"),
            }
        });
    }

    // ---- /metrics ----------------------------------------------------------

    async fn render_metrics(&self) -> Response<ProxyBody> {
        let own = self
            .prometheus
            .as_ref()
            .map(|h| h.render())
            .unwrap_or_default();
        let fetches = self.replicas.iter().filter_map(|r| {
            let addrs = r.addrs()?;
            let client = self.client.clone();
            Some(async move {
                let got = tokio::time::timeout(
                    METRICS_FETCH_TIMEOUT,
                    get_text(&client, addrs.http, "/metrics"),
                )
                .await
                .unwrap_or_else(|_| Err("timed out".into()));
                (Arc::clone(r), got)
            })
        });
        let fetched = futures::future::join_all(fetches).await;

        let mut parts = vec![exposition::Part {
            labels: Vec::new(),
            text: &own,
        }];
        let mut missing = String::new();
        for (r, got) in &fetched {
            match got {
                Ok(text) => parts.push(exposition::Part {
                    labels: vec![
                        (label::REPLICA, r.index().to_string()),
                        (label::GPU, r.spec.device.clone()),
                    ],
                    text,
                }),
                Err(e) => missing.push_str(&format!(
                    "# replica {} (gpu {}): metrics unavailable: {}\n",
                    r.index(),
                    r.spec.device,
                    e.to_string().replace('\n', " ")
                )),
            }
        }
        let mut body = exposition::merge(&parts);
        body.push_str(&missing);
        let mut resp = Response::new(full(body));
        resp.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        );
        resp
    }
}

/// `GET http://{addr}{path}` with a 200 required, as text.
pub async fn get_text(
    client: &ProxyClient,
    addr: SocketAddr,
    path: &str,
) -> Result<String, BoxError> {
    let uri: Uri = format!("http://{addr}{path}").parse()?;
    let req = Request::get(uri).body(empty())?;
    let resp = client.request(req).await?;
    let status = resp.status();
    let body = resp.into_body().collect().await?.to_bytes();
    if status != StatusCode::OK {
        return Err(format!("{path} answered {status}").into());
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// Whether `GET /readyz` on `addr` answers 200 within `within`.
pub async fn probe_ready(client: &ProxyClient, addr: SocketAddr, within: Duration) -> bool {
    let uri: Uri = match format!("http://{addr}/readyz").parse() {
        Ok(u) => u,
        Err(_) => return false,
    };
    let Ok(req) = Request::get(uri).body(empty()) else {
        return false;
    };
    match tokio::time::timeout(within, client.request(req)).await {
        Ok(Ok(resp)) => {
            let ok = resp.status() == StatusCode::OK;
            // Read it out so the connection goes back to the pool.
            let _ = tokio::time::timeout(within, resp.into_body().collect()).await;
            ok
        }
        _ => false,
    }
}

fn upstream_uri(addr: SocketAddr, original: &Uri) -> Uri {
    let pq = original.path_and_query().map(|p| p.as_str()).unwrap_or("/");
    Uri::builder()
        .scheme("http")
        .authority(addr.to_string())
        .path_and_query(pq)
        .build()
        .unwrap_or_else(|_| Uri::from_static("http://127.0.0.1/"))
}

fn wants_upgrade(h: &HeaderMap) -> bool {
    h.contains_key(header::UPGRADE)
        && h.get_all(header::CONNECTION)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .flat_map(|v| v.split(','))
            .any(|t| t.trim().eq_ignore_ascii_case("upgrade"))
}

/// Remove the headers that describe one hop's connection rather than the
/// message.  An upgrade keeps `Connection: upgrade` and `Upgrade`: they are the
/// request to switch, and the 101 that grants it.
fn strip_hop_by_hop(h: &mut HeaderMap, upgrade: bool) {
    let named: Vec<HeaderName> = h
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|t| HeaderName::from_bytes(t.trim().as_bytes()).ok())
        .collect();
    for name in named {
        if !(upgrade && name == header::UPGRADE) {
            h.remove(name);
        }
    }
    for name in [
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        HeaderName::from_static("keep-alive"),
        HeaderName::from_static("proxy-connection"),
    ] {
        h.remove(name);
    }
    if !upgrade {
        h.remove(header::CONNECTION);
        h.remove(header::UPGRADE);
    }
}

// ---- gRPC port --------------------------------------------------------------

/// The standard health service, answered by the launcher for itself.  Boxed:
/// `tonic_health::server::health_reporter` returns it as an opaque type.
pub type HealthService =
    tower::util::BoxCloneSyncService<Request<Incoming>, Response<ProxyBody>, Infallible>;

/// Box the server half of `tonic_health::server::health_reporter()`.
pub fn health_service<S>(server: S) -> HealthService
where
    S: tower::Service<
            Request<Incoming>,
            Response = Response<tonic::body::BoxBody>,
            Error = Infallible,
        > + Clone
        + Send
        + Sync
        + 'static,
    S::Future: Send + 'static,
{
    use tower::ServiceExt;
    tower::util::BoxCloneSyncService::new(server.map_response(|resp| resp.map(boxed)))
}

const HEALTH_PREFIX: &str = "/grpc.health.v1.Health/";

/// One downstream gRPC connection: its h2 connection to each replica it has
/// sent to, opened on first use.
pub struct GrpcConnection {
    router: Arc<Router>,
    health: HealthService,
    upstreams: Mutex<HashMap<usize, (u32, http2::SendRequest<ProxyBody>)>>,
}

impl GrpcConnection {
    pub fn new(router: Arc<Router>, health: HealthService) -> Arc<Self> {
        Arc::new(Self {
            router,
            health,
            upstreams: Mutex::new(HashMap::new()),
        })
    }

    pub async fn serve(
        self: Arc<Self>,
        req: Request<Incoming>,
    ) -> Result<Response<ProxyBody>, Infallible> {
        if req.uri().path().starts_with(HEALTH_PREFIX) {
            use tower::ServiceExt;
            return self.health.clone().oneshot(req).await;
        }
        Ok(self.forward(req).await)
    }

    async fn forward(&self, req: Request<Incoming>) -> Response<ProxyBody> {
        let mut refused: Vec<usize> = Vec::new();
        loop {
            let Some((replica, addrs)) = self.router.pick(&refused) else {
                if refused.is_empty() {
                    self.router.no_replica.increment(1);
                    return grpc_unavailable("no replica is ready");
                }
                self.router.replica_error.increment(1);
                return grpc_unavailable("no replica accepted a connection");
            };
            // The body is not sent until a connection exists, so a replica
            // that refuses one costs nothing but a retry on the next.
            let mut send = match self.sender(&replica, addrs).await {
                Ok(send) => send,
                Err(e) => {
                    warn!(replica = replica.index(), error = %e, "connecting to a replica failed");
                    replica.set_ready(false);
                    refused.push(replica.index());
                    continue;
                }
            };
            let load = replica.track();
            let (mut parts, body) = req.into_parts();
            parts.uri = upstream_uri(addrs.grpc, &parts.uri);
            parts.version = Version::HTTP_2;
            return match send
                .send_request(Request::from_parts(parts, boxed(body)))
                .await
            {
                Ok(resp) => resp.map(|b| tracked(boxed(b), load)),
                Err(e) => {
                    self.router.replica_error.increment(1);
                    warn!(replica = replica.index(), error = %e, "forwarding an RPC failed");
                    grpc_unavailable(&format!("replica {} did not answer: {e}", replica.index()))
                }
            };
        }
    }

    /// This connection's h2 sender to `replica`, opening one if it has none
    /// or the one it has belongs to a process that is gone.
    async fn sender(
        &self,
        replica: &Replica,
        addrs: BoundAddrs,
    ) -> Result<http2::SendRequest<ProxyBody>, BoxError> {
        let cached = {
            let map = self.upstreams.lock().unwrap_or_else(|e| e.into_inner());
            map.get(&replica.index())
                .filter(|(pid, s)| *pid == addrs.pid && !s.is_closed())
                .map(|(_, s)| s.clone())
        };
        if let Some(mut send) = cached {
            if send.ready().await.is_ok() {
                return Ok(send);
            }
        }
        let tcp = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addrs.grpc))
            .await
            .map_err(|_| "connect timed out")??;
        tcp.set_nodelay(true)?;
        let (mut send, conn) = http2::Builder::new(TokioExecutor::new())
            .handshake(TokioIo::new(tcp))
            .await?;
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                debug!("replica h2 connection ended: {e}");
            }
        });
        send.ready().await?;
        self.upstreams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(replica.index(), (addrs.pid, send.clone()));
        Ok(send)
    }
}

// ---- listeners --------------------------------------------------------------

/// Accept on `listener` until `stop`, serving each connection with the
/// service `make` builds for it; then let the open connections finish.
pub async fn serve_listener<M, S, F>(
    listener: TcpListener,
    mut stop: watch::Receiver<bool>,
    builder: auto::Builder<TokioExecutor>,
    make: M,
) where
    M: Fn() -> S,
    S: Fn(Request<Incoming>) -> F + Send + Clone + 'static,
    F: std::future::Future<Output = Result<Response<ProxyBody>, Infallible>> + Send + 'static,
{
    let graceful = GracefulShutdown::new();
    loop {
        let stream = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(e) => {
                    // Out of descriptors, mostly: back off rather than spin.
                    warn!("accept: {e}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            },
            _ = super::stopped(&mut stop) => break,
        };
        let _ = stream.set_nodelay(true);
        let service = hyper::service::service_fn(make());
        let conn = builder
            .serve_connection_with_upgrades(TokioIo::new(stream), service)
            .into_owned();
        let conn = graceful.watch(conn);
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                debug!("connection ended: {e}");
            }
        });
    }
    drop(listener);
    graceful.shutdown().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grpc_messages_are_percent_encoded() {
        assert_eq!(
            grpc_percent_encode("no replica: 100%"),
            "no replica: 100%25"
        );
        assert_eq!(grpc_percent_encode("é\n"), "%C3%A9%0A");
    }

    #[test]
    fn hop_by_hop_headers_go_and_an_upgrade_keeps_its_own() {
        let mut h = HeaderMap::new();
        h.insert(
            header::CONNECTION,
            HeaderValue::from_static("keep-alive, x-hop"),
        );
        h.insert("x-hop", HeaderValue::from_static("1"));
        h.insert("keep-alive", HeaderValue::from_static("timeout=5"));
        h.insert(
            header::TRANSFER_ENCODING,
            HeaderValue::from_static("chunked"),
        );
        h.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
        let mut plain = h.clone();
        strip_hop_by_hop(&mut plain, false);
        assert_eq!(plain.len(), 1);
        assert!(plain.contains_key(header::CONTENT_TYPE));

        let mut up = HeaderMap::new();
        up.insert(header::CONNECTION, HeaderValue::from_static("Upgrade"));
        up.insert(header::UPGRADE, HeaderValue::from_static("websocket"));
        up.insert("sec-websocket-key", HeaderValue::from_static("k"));
        assert!(wants_upgrade(&up));
        strip_hop_by_hop(&mut up, true);
        assert_eq!(up.len(), 3);
    }

    #[test]
    fn upstream_uris_keep_the_path_and_query() {
        let u = upstream_uri(
            "127.0.0.1:9".parse().unwrap(),
            &"/v1/speech:recognize?encoding=LINEAR16".parse().unwrap(),
        );
        assert_eq!(
            u,
            "http://127.0.0.1:9/v1/speech:recognize?encoding=LINEAR16"
        );
        let u = upstream_uri(
            "127.0.0.1:9".parse().unwrap(),
            &"http://example/oasr.speech.v1.Speech/Recognize"
                .parse()
                .unwrap(),
        );
        assert_eq!(u, "http://127.0.0.1:9/oasr.speech.v1.Speech/Recognize");
    }
}
