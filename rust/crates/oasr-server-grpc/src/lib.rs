// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! tonic gRPC service for the OASR serving frontend.
//!
//! Exposes `oasr.speech.v1.Speech` plus the standard
//! `grpc.health.v1.Health` health-checking service.

// Every fallible helper in this crate returns `tonic::Status`, which is 176
// bytes — that is tonic's error type, not ours, and boxing it at each call
// site would only move the cost while breaking the `?` ergonomics the
// generated service traits are built around.
#![allow(clippy::result_large_err)]

pub mod speech;

pub mod pb {
    tonic::include_proto!("oasr.speech.v1");
}

pub use speech::{ServiceMode, SpeechService};

/// Service name advertised by the gRPC health-checking service.
///
/// Use this string when calling `HealthReporter::set_service_status` so
/// load-balancers / sidecars can probe a specific service rather than the
/// whole process.
pub const SPEECH_SERVICE_NAME: &str = "oasr.speech.v1.Speech";
