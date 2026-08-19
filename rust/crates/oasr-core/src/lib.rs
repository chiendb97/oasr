// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! `oasr._core` — PyO3 extension module exposing the OASR serving core.
//!
//! Built into the wheel; the console script imports this module and calls
//! [`serve`].

// The pyfunction expansion triggers this lint outside function-level scope.
#![allow(clippy::useless_conversion)]

use clap::Parser;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Run the OASR HTTP + gRPC server to completion.
///
/// `args` includes the program name. Blocks until shutdown while releasing the
/// GIL so the dispatcher may acquire it.
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
