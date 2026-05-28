// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! tonic gRPC service for the OASR serving frontend.
//!
//! Exposes `oasr.speech.v1.Speech` plus the standard
//! `grpc.health.v1.Health` health-checking service.

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
