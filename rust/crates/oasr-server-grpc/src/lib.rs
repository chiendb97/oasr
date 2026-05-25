// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! tonic gRPC service for the OASR serving frontend.

pub mod speech;

pub mod pb {
    tonic::include_proto!("oasr.asr.v1");
}

pub use speech::SpeechService;
