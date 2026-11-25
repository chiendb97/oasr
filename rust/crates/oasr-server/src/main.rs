// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Entry point for `oasr-server`.
//!
//! Hosts one in-process Python `ASREngine` (via PyO3) and serves it over
//! HTTP + gRPC.  With `--gpus` it starts one replica of itself per GPU instead
//! and routes to them from its own ports.
//!
//! All serving logic lives in the `oasr-serve` crate, shared with the
//! `oasr._core` extension module; this binary just parses the CLI and embeds
//! Python (`pyo3/auto-initialize`, pulled in via the `oasr-engine-client`
//! dependency below).

use anyhow::Result;
use clap::Parser;
use oasr_serve::{Cli, SelfCommand};

fn main() -> Result<()> {
    let argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let cli = Cli::parse_from(&argv);
    oasr_serve::run(cli, argv, SelfCommand::current_exe()?)
}
