// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! A dispatcher thread's end, driven with a stand-in engine.
//!
//! The server never joined its dispatcher threads and never called
//! `ASREngine.shutdown()`: at exit a request still in flight heard nothing, and
//! the engine's graph pools and pinned buffers were released only if the
//! process went with them.  These run a real dispatcher thread against a
//! pure-Python engine (no `oasr`, no numpy, no GPU), streaming requests only.

use std::time::Duration;

use oasr_engine_client::{EngineClient, EngineClientConfig, EngineClientError, PyEngine};
use oasr_wire::{ErrorCode, Event, ModelInfo};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tokio_stream::StreamExt;

/// The engine surface a dispatcher uses for streaming admissions.
const FAKE_ENGINE: &str = r#"
class Engine:
    num_waiting = 0

    def __init__(self):
        self.live = set()
        self.aborted = []
        self.shut_down = False

    @property
    def num_running(self):
        return len(self.live)

    def add_requests_batch_checked(self, specs):
        for spec in specs:
            self.live.add(spec["request_id"])
        return [{"request_id": spec["request_id"]} for spec in specs]

    def step(self):
        return []

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.live.discard(request_id)

    def shutdown(self):
        self.shut_down = True
"#;

/// A dispatcher-ready engine, and the Python object behind it to inspect.
fn fake_engine() -> (PyEngine, Py<PyAny>) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = PyModule::from_code_bound(py, FAKE_ENGINE, "fake_engine.py", "fake_engine")
            .expect("the stand-in engine does not compile");
        let engine = module.getattr("Engine").unwrap().call0().unwrap().unbind();
        let wrapped = PyEngine::from_object(engine.clone_ref(py), ModelInfo::default());
        (wrapped, engine)
    })
}

async fn next_event<S: tokio_stream::Stream<Item = Event> + Unpin>(
    events: &mut S,
) -> Option<Event> {
    tokio::time::timeout(Duration::from_secs(5), events.next())
        .await
        .expect("no event and no end of stream within 5 s")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stopped_dispatcher_fails_what_it_holds_releases_its_engine_and_exits() {
    let (engine, fake) = fake_engine();
    let client = EngineClient::start(engine, EngineClientConfig::new("t"));
    let mut handle = client.open_streaming(16_000, 0, None).await.unwrap();
    let rid = handle.request_id.clone();
    assert!(
        matches!(
            next_event(&mut handle.events).await,
            Some(Event::Accepted { .. })
        ),
        "the stream was not admitted"
    );

    assert!(
        client.shutdown(Duration::from_secs(10)).await,
        "the dispatcher thread did not exit"
    );

    match next_event(&mut handle.events).await {
        Some(Event::Error {
            code: ErrorCode::Shutdown,
            ..
        }) => {}
        other => panic!("the open stream was not told the engine stopped: {other:?}"),
    }
    assert!(
        next_event(&mut handle.events).await.is_none(),
        "the stream did not end after its terminal event"
    );
    assert!(client.router().is_empty());
    Python::with_gil(|py| {
        let fake = fake.bind(py);
        assert!(
            fake.getattr("shut_down")
                .unwrap()
                .extract::<bool>()
                .unwrap(),
            "ASREngine.shutdown() was not called"
        );
        let aborted: Vec<String> = fake.getattr("aborted").unwrap().extract().unwrap();
        assert_eq!(aborted, vec![rid], "the open stream was not aborted");
    });

    // Nothing is admitted once it has stopped, and trying leaves nothing behind.
    match client.open_streaming(16_000, 0, None).await {
        Err(EngineClientError::WorkerDown(_)) => {}
        Err(e) => panic!("expected WorkerDown, got {e}"),
        Ok(_) => panic!("a stopped dispatcher admitted a stream"),
    }
    assert!(client.router().is_empty());
    // Already joined: nothing left to wait for.
    assert!(client.shutdown(Duration::from_millis(1)).await);
}

/// An idle dispatcher blocks on its command channel between heartbeats; a stop
/// has to reach it there rather than wait for that receive to time out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_idle_dispatcher_stops_and_still_releases_its_engine() {
    let (engine, fake) = fake_engine();
    let client = EngineClient::start(engine, EngineClientConfig::new("t"));
    client
        .ping(Duration::from_secs(5))
        .await
        .expect("the dispatcher never ticked");
    assert!(client.shutdown(Duration::from_secs(10)).await);
    Python::with_gil(|py| {
        assert!(fake
            .bind(py)
            .getattr("shut_down")
            .unwrap()
            .extract::<bool>()
            .unwrap());
    });
}
