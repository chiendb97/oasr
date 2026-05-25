// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = "../../proto/oasr_asr.proto";
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&[proto], &["../../proto"])?;
    println!("cargo:rerun-if-changed={proto}");
    Ok(())
}
