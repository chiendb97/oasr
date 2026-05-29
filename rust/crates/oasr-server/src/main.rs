// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Entry point for `oasr-server`.
//!
//! Hosts one in-process Python `ASREngine` (via PyO3) and serves it over
//! HTTP + gRPC.  Multi-GPU scaling = launch one `oasr-server` per GPU
//! behind a process manager and set `CUDA_VISIBLE_DEVICES` per launch.
//!
//! All serving logic lives in the `oasr-serve` crate, shared with the
//! `oasr._core` extension module; this binary just parses the CLI and embeds
//! Python (`pyo3/auto-initialize`, pulled in via the `oasr-engine-client`
//! dependency below).

use anyhow::Result;
use clap::Parser;
use oasr_serve::Cli;

fn main() -> Result<()> {
    let cli = Cli::parse();
    oasr_serve::run(cli)
}
