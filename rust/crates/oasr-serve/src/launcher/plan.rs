// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! What the launcher starts: the replicas, their devices, and their argv.

use std::collections::HashSet;
use std::ffi::OsString;

use anyhow::{bail, Context, Result};

/// One replica process the launcher keeps running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaSpec {
    /// Position in the launcher's replica list: the `replica` label.
    pub index: usize,
    /// What the replica's `CUDA_VISIBLE_DEVICES` is set to: the physical
    /// device (an ordinal or a UUID), and its `gpu` label.
    pub device: String,
}

/// The replicas for `--gpus` and `--processes-per-gpu`, GPU-major.
///
/// `gpus` is read the way CUDA would read a device ordinal in this process:
/// through `visible` (this process's `CUDA_VISIBLE_DEVICES`) when that is set,
/// so `--gpus 0` under `CUDA_VISIBLE_DEVICES=2,3` is device `2`.  `count_gpus`
/// says how many devices there are when nothing is visible-restricted; it is
/// asked only for `all`, and for range checking when it can answer.
pub fn plan_replicas(
    gpus: Option<&str>,
    per_gpu: u32,
    visible: Option<&str>,
    count_gpus: impl FnOnce() -> Result<usize>,
) -> Result<Vec<ReplicaSpec>> {
    // CUDA stops reading the list at its first empty entry; so do we.
    let visible: Option<Vec<String>> = visible.map(|v| {
        v.split(',')
            .map(str::trim)
            .take_while(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    });
    if let Some(v) = &visible {
        if v.is_empty() {
            bail!("CUDA_VISIBLE_DEVICES is set but names no device; there is no GPU to launch on");
        }
    }
    let available = || -> Result<usize> {
        match &visible {
            Some(v) => Ok(v.len()),
            None => count_gpus(),
        }
    };

    let spec = gpus.unwrap_or("0").trim();
    let ordinals: Vec<usize> = if spec.eq_ignore_ascii_case("all") {
        let n = available().context("count the GPUs for --gpus all")?;
        if n == 0 {
            bail!("--gpus all: no GPU found");
        }
        (0..n).collect()
    } else {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for tok in spec.split(',').map(str::trim) {
            let i: usize = tok.parse().with_context(|| {
                format!("--gpus: {tok:?} is not a device ordinal (expected e.g. `0,1` or `all`)")
            })?;
            if !seen.insert(i) {
                bail!("--gpus lists device {i} twice; use --processes-per-gpu for several replicas on one GPU");
            }
            out.push(i);
        }
        // Checked when the count is known; otherwise a replica on a missing
        // device fails its start, and says so.
        if let Ok(n) = available() {
            if let Some(&bad) = out.iter().find(|&&i| i >= n) {
                bail!("--gpus: device {bad} does not exist; {n} visible");
            }
        }
        out
    };

    let mut replicas = Vec::with_capacity(ordinals.len() * per_gpu as usize);
    for &ordinal in &ordinals {
        let device = match &visible {
            Some(v) => v[ordinal].clone(),
            None => ordinal.to_string(),
        };
        for _ in 0..per_gpu {
            replicas.push(ReplicaSpec {
                index: replicas.len(),
                device: device.clone(),
            });
        }
    }
    Ok(replicas)
}

/// The number of GPUs `nvidia-smi` lists (MIG instances are listed indented
/// under their GPU and are not counted).
pub fn count_gpus_with_nvidia_smi() -> Result<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .arg("--list-gpus")
        .output()
        .context("run `nvidia-smi --list-gpus` (pass --gpus 0,1,... to skip it)")?;
    if !out.status.success() {
        bail!(
            "`nvidia-smi --list-gpus` failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.starts_with("GPU "))
        .count())
}

/// Flags that belong to the launcher: a replica must not see them, or gets its
/// own value instead.
const LAUNCHER_FLAGS: &[&str] = &[
    "--gpus",
    "--processes-per-gpu",
    "--http-bind",
    "--grpc-bind",
    "--report-bound-addrs",
];

/// A replica's arguments: the launcher's own (`argv` less the program name)
/// without the launcher's flags, then a loopback bind on port 0 for both
/// listeners and the file to report the ports in.
///
/// Filtering the command line rather than re-rendering [`crate::Cli`] keeps
/// every other flag exactly as the operator wrote it, including the ones a
/// future change adds.
pub fn replica_args(argv: &[OsString], report_to: &std::path::Path) -> Vec<OsString> {
    let mut out = Vec::with_capacity(argv.len() + 6);
    // After `--` nothing is a flag, ours included, and ours must go before it.
    let mut tail: Vec<OsString> = Vec::new();
    let mut it = argv.iter().skip(1);
    while let Some(arg) = it.next() {
        if let Some(s) = arg.to_str() {
            if s == "--" {
                tail.push(arg.clone());
                tail.extend(it.cloned());
                break;
            }
            if LAUNCHER_FLAGS.contains(&s) {
                it.next(); // its value
                continue;
            }
            if LAUNCHER_FLAGS
                .iter()
                .any(|f| s.strip_prefix(f).is_some_and(|rest| rest.starts_with('=')))
            {
                continue;
            }
        }
        out.push(arg.clone());
    }
    for (flag, value) in [
        ("--http-bind", OsString::from("127.0.0.1:0")),
        ("--grpc-bind", OsString::from("127.0.0.1:0")),
        ("--report-bound-addrs", report_to.as_os_str().to_owned()),
    ] {
        out.push(flag.into());
        out.push(value);
    }
    out.extend(tail);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cli;
    use clap::Parser;

    fn no_count() -> Result<usize> {
        bail!("nvidia-smi absent")
    }

    fn devices(r: &[ReplicaSpec]) -> Vec<&str> {
        r.iter().map(|s| s.device.as_str()).collect()
    }

    #[test]
    fn ordinals_map_through_the_inherited_visible_list() {
        let r = plan_replicas(Some("1,0"), 1, Some("4,GPU-abc"), no_count).unwrap();
        assert_eq!(devices(&r), ["GPU-abc", "4"]);
        assert_eq!(r.iter().map(|s| s.index).collect::<Vec<_>>(), [0, 1]);
    }

    #[test]
    fn all_is_every_visible_device_or_every_counted_one() {
        let r = plan_replicas(Some("all"), 1, Some("2,3"), no_count).unwrap();
        assert_eq!(devices(&r), ["2", "3"]);
        let r = plan_replicas(Some("ALL"), 1, None, || Ok(3)).unwrap();
        assert_eq!(devices(&r), ["0", "1", "2"]);
        assert!(plan_replicas(Some("all"), 1, None, no_count).is_err());
        assert!(plan_replicas(Some("all"), 1, None, || Ok(0)).is_err());
    }

    #[test]
    fn processes_per_gpu_repeats_each_device_gpu_major() {
        let r = plan_replicas(Some("0,1"), 2, None, || Ok(2)).unwrap();
        assert_eq!(devices(&r), ["0", "0", "1", "1"]);
        // Without --gpus: the first visible device, as the plain server uses.
        let r = plan_replicas(None, 3, Some("5"), no_count).unwrap();
        assert_eq!(devices(&r), ["5", "5", "5"]);
    }

    #[test]
    fn a_bad_list_is_refused_before_anything_starts() {
        assert!(plan_replicas(Some("0,0"), 1, None, || Ok(2)).is_err());
        assert!(plan_replicas(Some("0,x"), 1, None, || Ok(2)).is_err());
        assert!(plan_replicas(Some("2"), 1, None, || Ok(2)).is_err());
        assert!(plan_replicas(Some("1"), 1, Some("7"), no_count).is_err());
        assert!(plan_replicas(Some("0"), 1, Some(""), no_count).is_err());
        // An unknown count does not block an explicit list.
        assert_eq!(
            devices(&plan_replicas(Some("3"), 1, None, no_count).unwrap()),
            ["3"]
        );
        // CUDA reads up to the first empty entry.
        assert!(plan_replicas(Some("1"), 1, Some("0,,1"), no_count).is_err());
    }

    fn os(v: &[&str]) -> Vec<OsString> {
        v.iter().map(OsString::from).collect()
    }

    #[test]
    fn replica_args_drop_the_launcher_flags_in_both_spellings() {
        let argv = os(&[
            "oasr-server",
            "--gpus",
            "0,1",
            "--ckpt-dir",
            "/m",
            "--processes-per-gpu=2",
            "--http-bind=0.0.0.0:80",
            "--grpc-bind",
            "0.0.0.0:9",
            "--max-batch-size",
            "64",
        ]);
        let got = replica_args(&argv, std::path::Path::new("/tmp/r0.json"));
        assert_eq!(
            got,
            os(&[
                "--ckpt-dir",
                "/m",
                "--max-batch-size",
                "64",
                "--http-bind",
                "127.0.0.1:0",
                "--grpc-bind",
                "127.0.0.1:0",
                "--report-bound-addrs",
                "/tmp/r0.json",
            ])
        );
        // What a replica receives parses, binds the loopback, and is not a
        // launcher -- the recursion guard in `run` depends on that.
        let mut child = os(&["oasr-server"]);
        child.extend(got);
        let cli = Cli::try_parse_from(&child).expect("replica argv parses");
        assert!(!cli.launches_replicas());
        assert_eq!(cli.http_bind, "127.0.0.1:0".parse().unwrap());
        assert_eq!(cli.max_batch_size, Some(64));
    }

    #[test]
    fn a_flag_that_only_starts_like_ours_is_kept() {
        let argv = os(&[
            "oasr-server",
            "--gpus",
            "0",
            "--grpc-bind-extra=1",
            "--",
            "--gpus",
        ]);
        let got = replica_args(&argv, std::path::Path::new("r"));
        assert_eq!(got[0], "--grpc-bind-extra=1");
        // Ours before the `--`; what followed it, untouched, after.
        assert_eq!(&got[1..3], &os(&["--http-bind", "127.0.0.1:0"])[..]);
        assert_eq!(&got[got.len() - 2..], &os(&["--", "--gpus"])[..]);
    }
}
