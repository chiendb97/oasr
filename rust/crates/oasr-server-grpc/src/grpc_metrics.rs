// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Transport and request metrics for the gRPC surface.
//!
//! Recorded **inside the handlers**, not by a tower layer, for one reason that
//! is specific to gRPC: a server-streaming RPC's status arrives in the HTTP/2
//! *trailers*, long after the response head. A generic middleware sees
//! `streaming_recognize` return `Ok` the moment the stream is handed back —
//! typically a few hundred microseconds in, before a single byte of audio has
//! been transcribed — so it would report a millisecond-scale p99 for an RPC
//! that ran for a minute, and would label every failed stream `OK`. The only
//! place that knows when the RPC really ended, and how, is the task draining
//! it.

use std::sync::OnceLock;
use std::time::Duration;

use oasr_metrics as om;
use oasr_metrics::RequestRecorder;
use tonic::{Code, Status};

/// gRPC method names, as they appear on the wire.
///
/// Literals rather than something built from [`crate::SPEECH_SERVICE_NAME`],
/// because `concat!` only takes literals — so
/// `method_paths_match_the_service_name` checks the relationship instead. A
/// proto package rename would otherwise leave these labelling a method path
/// that no longer exists, and nothing would fail.
pub mod method {
    pub const RECOGNIZE: &str = "/oasr.speech.v1.Speech/Recognize";
    pub const STREAMING_RECOGNIZE: &str = "/oasr.speech.v1.Speech/StreamingRecognize";
}

/// `Recognize` — unary, offline.
pub fn unary() -> &'static RequestRecorder {
    static R: OnceLock<RequestRecorder> = OnceLock::new();
    R.get_or_init(|| RequestRecorder::new(om::api::GRPC, om::mode::OFFLINE))
}

/// `StreamingRecognize`.
pub fn streaming() -> &'static RequestRecorder {
    static R: OnceLock<RequestRecorder> = OnceLock::new();
    R.get_or_init(|| RequestRecorder::new(om::api::GRPC_STREAMING, om::mode::STREAMING))
}

/// Record one RPC's terminal status and wall time.
///
/// The `code` label is the canonical gRPC status name, which is a closed set —
/// safe to use as a label without any of the cardinality care an HTTP path
/// needs.
pub fn record_rpc(method: &'static str, code: Code, elapsed: Duration) {
    metrics::counter!(
        om::GRPC_REQUESTS,
        om::label::METHOD => method,
        om::label::CODE => code_name(code),
    )
    .increment(1);
    metrics::histogram!(om::GRPC_DURATION, om::label::METHOD => method)
        .record(elapsed.as_secs_f64());
}

/// Map a gRPC status to the request-scope `outcome`.
///
/// `CANCELLED` and `DEADLINE_EXCEEDED` are the two a client causes — a hang-up
/// and a timeout — and folding them into `error` would put every disconnected
/// stream in the error-rate panel. `UNAVAILABLE` is deliberately *not* in that
/// set: it is what a draining or overloaded server returns, which is the
/// server's problem to see.
pub fn outcome_for(code: Code) -> om::Outcome {
    match code {
        Code::Ok => om::Outcome::Ok,
        Code::Cancelled | Code::DeadlineExceeded => om::Outcome::Cancelled,
        _ => om::Outcome::Error,
    }
}

/// The terminal code of a handler result.
pub fn code_of<T>(result: &Result<T, Status>) -> Code {
    match result {
        Ok(_) => Code::Ok,
        Err(s) => s.code(),
    }
}

/// A `'static` name for a status code, so it can be a label without allocating.
fn code_name(code: Code) -> &'static str {
    match code {
        Code::Ok => "OK",
        Code::Cancelled => "CANCELLED",
        Code::Unknown => "UNKNOWN",
        Code::InvalidArgument => "INVALID_ARGUMENT",
        Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
        Code::NotFound => "NOT_FOUND",
        Code::AlreadyExists => "ALREADY_EXISTS",
        Code::PermissionDenied => "PERMISSION_DENIED",
        Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
        Code::FailedPrecondition => "FAILED_PRECONDITION",
        Code::Aborted => "ABORTED",
        Code::OutOfRange => "OUT_OF_RANGE",
        Code::Unimplemented => "UNIMPLEMENTED",
        Code::Internal => "INTERNAL",
        Code::Unavailable => "UNAVAILABLE",
        Code::DataLoss => "DATA_LOSS",
        Code::Unauthenticated => "UNAUTHENTICATED",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A client hanging up is not a server error, and a draining server is not
    /// a client hang-up.  Both directions matter: the first keeps normal
    /// disconnects out of the error panel, the second keeps a real outage in it.
    #[test]
    fn outcome_separates_client_hangups_from_server_faults() {
        assert_eq!(outcome_for(Code::Ok), om::Outcome::Ok);
        assert_eq!(outcome_for(Code::Cancelled), om::Outcome::Cancelled);
        assert_eq!(outcome_for(Code::DeadlineExceeded), om::Outcome::Cancelled);
        assert_eq!(outcome_for(Code::Unavailable), om::Outcome::Error);
        assert_eq!(outcome_for(Code::ResourceExhausted), om::Outcome::Error);
        assert_eq!(outcome_for(Code::Internal), om::Outcome::Error);
    }

    #[test]
    fn method_paths_match_the_service_name() {
        let prefix = format!("/{}/", crate::SPEECH_SERVICE_NAME);
        assert_eq!(method::RECOGNIZE, format!("{prefix}Recognize"));
        assert_eq!(
            method::STREAMING_RECOGNIZE,
            format!("{prefix}StreamingRecognize")
        );
    }

    #[test]
    fn code_of_reads_the_status() {
        let ok: Result<(), Status> = Ok(());
        assert_eq!(code_of(&ok), Code::Ok);
        let err: Result<(), Status> = Err(Status::resource_exhausted("full"));
        assert_eq!(code_of(&err), Code::ResourceExhausted);
    }
}
