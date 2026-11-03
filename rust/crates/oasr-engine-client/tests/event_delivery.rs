// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Events reaching a client that reads slower than the engine emits.
//!
//! A real dispatcher thread against a pure-Python engine (no `oasr`, no numpy,
//! no GPU) that emits one partial per tick, with a speech transition on some
//! of them, for a stream whose client is not reading.  Past the channel's
//! capacity, partials are dropped, which is harmless for the transcript (each
//! one carries all of it) but not for the transitions riding on them: a lost
//! `speech_stopped` left a `/v1/realtime` client in "speaking".

use std::time::{Duration, Instant};

use oasr_engine_client::{EngineClient, EngineClientConfig, PyEngine};
use oasr_wire::{Event, ModelInfo};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tokio_stream::StreamExt;

/// A streaming engine that needs no audio: each tick emits the next partial.
const SCRIPTED_ENGINE: &str = r#"
class Transition:
    def __init__(self, kind, time):
        self.kind = kind
        self.time = time


class Output:
    def __init__(self, request_id, text, finished, speech_events=None):
        self.request_id = request_id
        self.text = text
        self.finished = finished
        self.tokens = []
        self.speech_events = speech_events


class Engine:
    num_waiting = 0

    def __init__(self, partials, every):
        self.partials = partials
        self.every = every
        self.live = {}
        self.emitted_final = False

    @property
    def num_running(self):
        return len(self.live)

    def add_requests_batch_checked(self, specs):
        for spec in specs:
            self.live[spec["request_id"]] = 0
        return [{"request_id": spec["request_id"]} for spec in specs]

    @staticmethod
    def text(n):
        return " ".join(f"w{i}" for i in range(n))

    def step(self):
        out = []
        for rid, i in list(self.live.items()):
            if i == self.partials:
                out.append(Output(rid, self.text(i), True))
                del self.live[rid]
                self.emitted_final = True
                continue
            events = None
            if i % self.every == 0:
                kind = "speech_started" if (i // self.every) % 2 == 0 else "speech_stopped"
                events = [Transition(kind, float(i))]
            out.append(Output(rid, self.text(i + 1), False, events))
            self.live[rid] = i + 1
        return out

    def abort_request(self, request_id):
        self.live.pop(request_id, None)

    def shutdown(self):
        pass
"#;

fn scripted_engine(partials: usize, every: usize) -> (PyEngine, Py<PyAny>) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module =
            PyModule::from_code_bound(py, SCRIPTED_ENGINE, "scripted_engine.py", "scripted_engine")
                .expect("the stand-in engine does not compile");
        let engine = module
            .getattr("Engine")
            .unwrap()
            .call1((partials, every))
            .unwrap()
            .unbind();
        let wrapped = PyEngine::from_object(engine.clone_ref(py), ModelInfo::default());
        (wrapped, engine)
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_speech_transition_reaches_a_client_that_fell_behind() {
    let (partials, every) = (200, 10);
    let config = EngineClientConfig::new("t");
    assert!(
        partials > 2 * config.event_channel_cap,
        "the client would not fall behind; nothing under test"
    );
    let (engine, fake) = scripted_engine(partials, every);
    let client = EngineClient::start(engine, config);
    let mut handle = client.open_streaming(16_000, 0, None).await.unwrap();

    // Read nothing until the engine has emitted the whole stream.
    let deadline = Instant::now() + Duration::from_secs(10);
    while !Python::with_gil(|py| {
        fake.bind(py)
            .getattr("emitted_final")
            .unwrap()
            .extract::<bool>()
            .unwrap()
    }) {
        assert!(Instant::now() < deadline, "the engine never finished");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let mut seen = Vec::new();
    let mut last_text = String::new();
    let final_text = loop {
        let event = tokio::time::timeout(Duration::from_secs(5), handle.events.next())
            .await
            .expect("no event within 5 s")
            .expect("the stream ended without a final");
        match event {
            Event::Accepted { .. } => {}
            Event::Partial {
                text,
                speech_events,
                ..
            } => {
                seen.extend(speech_events.unwrap_or_default());
                last_text = text;
            }
            Event::Final { text, .. } => break text,
            other => panic!("unexpected {other:?}"),
        }
    };

    let expected: Vec<(String, f32)> = (0..partials)
        .step_by(every)
        .map(|i| {
            let kind = if (i / every) % 2 == 0 {
                "speech_started"
            } else {
                "speech_stopped"
            };
            (kind.to_string(), i as f32)
        })
        .collect();
    let seen: Vec<(String, f32)> = seen.into_iter().map(|t| (t.kind, t.time)).collect();
    assert_eq!(seen, expected, "speech transitions lost or reordered");
    let all: Vec<String> = (0..partials).map(|i| format!("w{i}")).collect();
    assert_eq!(final_text, all.join(" "));
    assert_eq!(
        last_text, final_text,
        "the newest partial was not delivered"
    );

    assert!(client.shutdown(Duration::from_secs(10)).await);
}
