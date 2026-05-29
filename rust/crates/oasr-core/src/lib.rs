// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `oasr._core` — PyO3 extension module exposing the OASR serving core.
//!
//! Built by setuptools-rust during `pip install` (see pyproject.toml
//! `[[tool.setuptools-rust.ext-modules]]`).  The `oasr-server` console script
//! (`oasr/_server_cli.py`) imports this module and calls [`serve`], so the
//! Rust front-end ships with the wheel instead of a separately-built binary.

use clap::Parser;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Run the OASR HTTP + gRPC server to completion.
///
/// `args` is a full command line *including* the program name at index 0
/// (i.e. pass `sys.argv`), parsed with the same clap CLI as the standalone
/// binary.  Blocks until SIGINT / SIGTERM.  The GIL is released for the
/// server's lifetime via `allow_threads` so the engine dispatcher thread can
/// re-acquire it through `Python::with_gil`.
#[pyfunction]
fn serve(py: Python<'_>, args: Vec<String>) -> PyResult<()> {
    // `--help` / `--version` / parse errors print and exit the process, matching
    // the standalone binary's behaviour rather than raising a Python exception.
    let cli = match oasr_serve::Cli::try_parse_from(&args) {
        Ok(cli) => cli,
        Err(e) => e.exit(),
    };
    py.allow_threads(|| oasr_serve::run(cli))
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serve, m)?)?;
    Ok(())
}
