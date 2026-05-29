#!/usr/bin/env bash
set -Eeo pipefail
trap 'echo "[install_rust.sh] Error on line $LINENO" >&2' ERR

# Install the Rust toolchain via rustup (needed to build the oasr-server binary).
# Override RUST_TOOLCHAIN to pin a channel; defaults to the workspace
# rust-toolchain.toml (stable) when unset.
RUSTUP_HOME="${RUSTUP_HOME:-/opt/rust/rustup}"
CARGO_HOME="${CARGO_HOME:-/opt/rust/cargo}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-stable}"

export RUSTUP_HOME CARGO_HOME

echo "Installing Rust toolchain '${RUST_TOOLCHAIN}' (RUSTUP_HOME=${RUSTUP_HOME}, CARGO_HOME=${CARGO_HOME})"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path \
        --default-toolchain "${RUST_TOOLCHAIN}" \
        --profile minimal \
        --component rustfmt clippy

# Make cargo/rustc available on PATH for subsequent build steps and at runtime.
"${CARGO_HOME}/bin/rustc" --version
"${CARGO_HOME}/bin/cargo" --version
