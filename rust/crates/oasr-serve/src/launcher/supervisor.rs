// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Keeping one replica process running.
//!
//! A replica's life: start (under its GPU's start lock), report its ports,
//! answer `/readyz`, serve; and if it exits -- or stops answering `/readyz`
//! for [`WEDGED_AFTER`] once it has -- start again after a backoff.  A
//! replica's `/readyz` fails once its dispatcher has given up on a failing
//! engine, which no amount of waiting repairs: restarting it is what a
//! Kubernetes liveness probe would have done to the single-engine server.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::anyhow;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

use super::plan;
use super::replica::{BoundAddrs, Replica};
use super::router::{probe_ready, ProxyClient};

/// How often a starting replica is asked whether it is up yet.
const START_POLL: Duration = Duration::from_millis(200);
/// How often a running replica's `/readyz` is polled.
const READY_POLL: Duration = Duration::from_secs(1);
/// A replica that was ready and has not answered `/readyz` for this long is
/// restarted.  Generous: its own heartbeat goes stale after 5 s, and only a
/// dispatcher that gave up stays stale.
pub const WEDGED_AFTER: Duration = Duration::from_secs(60);
const RESTART_BACKOFF_MIN: Duration = Duration::from_secs(1);
const RESTART_BACKOFF_MAX: Duration = Duration::from_secs(30);

/// What every supervisor shares.
pub struct LaunchContext {
    pub program: OsString,
    pub prefix_args: Vec<OsString>,
    /// The launcher's own command line, program name first.
    pub argv: Vec<OsString>,
    /// Where replicas report their ports; private to this launcher.
    pub run_dir: PathBuf,
    pub client: ProxyClient,
    pub logs: LogSinks,
    /// A replica's drain after SIGTERM: its `--shutdown-grace-secs`, the engine
    /// stop and the cut grace on either side.  SIGKILL after this.
    pub stop_timeout: Duration,
    /// [`WEDGED_AFTER`], but for tests.
    pub wedged_after: Duration,
    pub first_starts: FirstStarts,
    /// One start at a time per device: a replica sizes its pools from the
    /// memory its GPU has free when it starts, so two starting together would
    /// each count what the other is about to take.
    gpu_locks: Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl LaunchContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        program: OsString,
        prefix_args: Vec<OsString>,
        argv: Vec<OsString>,
        run_dir: PathBuf,
        client: ProxyClient,
        logs: LogSinks,
        stop_timeout: Duration,
        first_starts: FirstStarts,
    ) -> Self {
        Self {
            program,
            prefix_args,
            argv,
            run_dir,
            client,
            logs,
            stop_timeout,
            wedged_after: WEDGED_AFTER,
            first_starts,
            gpu_locks: Mutex::default(),
        }
    }

    fn gpu_lock(&self, device: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.gpu_locks.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(locks.entry(device.to_string()).or_default())
    }
}

/// Whether the first start of every replica failed: then the configuration
/// is what is wrong (a bad checkpoint path fails them all alike), and the
/// launcher stops rather than restarting them forever.  Once any replica has
/// been ready, a failing one is restarted instead.
pub struct FirstStarts {
    pending: AtomicUsize,
    any_ready: AtomicBool,
    fatal: mpsc::UnboundedSender<anyhow::Error>,
}

impl FirstStarts {
    pub fn new(replicas: usize, fatal: mpsc::UnboundedSender<anyhow::Error>) -> Self {
        Self {
            pending: AtomicUsize::new(replicas),
            any_ready: AtomicBool::new(false),
            fatal,
        }
    }

    fn ready(&self) {
        self.any_ready.store(true, Ordering::Release);
        self.pending.fetch_sub(1, Ordering::AcqRel);
    }

    fn failed(&self, replica: usize, why: &str) {
        let left = self.pending.fetch_sub(1, Ordering::AcqRel) - 1;
        if left == 0 && !self.any_ready.load(Ordering::Acquire) {
            let _ = self.fatal.send(anyhow!(
                "no replica started: every first start failed (the last, replica {replica}: \
                 {why}); see the replicas' logs above"
            ));
        }
    }
}

enum Ended {
    /// The process exited (or could not be started).
    Exited(String),
    /// The launcher is stopping.
    Stopped,
}

/// Keep `replica` running until `stop`.
pub async fn supervise(
    replica: Arc<Replica>,
    ctx: Arc<LaunchContext>,
    mut stop: watch::Receiver<bool>,
) {
    let i = replica.index();
    let device = replica.spec.device.clone();
    let report = ctx.run_dir.join(format!("replica-{i}.json"));
    let mut backoff = RESTART_BACKOFF_MIN;
    // Until the first start's outcome has been told to `FirstStarts`.
    let mut first_start = true;
    loop {
        if *stop.borrow() {
            return;
        }
        let ended = run_once(
            &replica,
            &ctx,
            &report,
            &mut stop,
            &mut backoff,
            &mut first_start,
        )
        .await;
        replica.set_addrs(None);
        let why = match ended {
            Ended::Stopped => return,
            Ended::Exited(why) => why,
        };
        if std::mem::take(&mut first_start) {
            ctx.first_starts.failed(i, &why);
        }
        if *stop.borrow() {
            return;
        }
        replica.restarts.increment(1);
        error!(replica = i, gpu = %device, "replica {why}; restarting in {backoff:?}");
        tokio::select! {
            _ = tokio::time::sleep(backoff) => {}
            _ = super::stopped(&mut stop) => return,
        }
        backoff = (backoff * 2).min(RESTART_BACKOFF_MAX);
    }
}

/// One process, start to exit.
async fn run_once(
    replica: &Arc<Replica>,
    ctx: &LaunchContext,
    report: &Path,
    stop: &mut watch::Receiver<bool>,
    backoff: &mut Duration,
    first_start: &mut bool,
) -> Ended {
    let i = replica.index();
    let device = replica.spec.device.clone();
    let _ = std::fs::remove_file(report);

    // Held from spawn until ready, or until the process is gone.
    let lock = ctx.gpu_lock(&device);
    let mut start_lock = tokio::select! {
        g = lock.lock_owned() => Some(g),
        _ = super::stopped(stop) => return Ended::Stopped,
    };

    let mut child = match spawn(ctx, &device, report) {
        Ok(child) => child,
        Err(e) => return Ended::Exited(format!("could not be started: {e}")),
    };
    let pid = child.id().unwrap_or(0);
    info!(replica = i, gpu = %device, pid, "replica starting");
    let tag = LogTag::new(i, &device, ctx.logs.json);
    if let Some(out) = child.stdout.take() {
        ctx.logs.pump(out, tag.clone(), false);
    }
    if let Some(err) = child.stderr.take() {
        ctx.logs.pump(err, tag, true);
    }

    let mut addrs: Option<BoundAddrs> = None;
    let mut was_ready = false;
    let mut unready_since: Option<Instant> = None;
    let mut terminating: Option<Instant> = None;
    loop {
        let poll = if was_ready { READY_POLL } else { START_POLL };
        tokio::select! {
            status = child.wait() => {
                let why = match status {
                    Ok(s) => describe_exit(s),
                    Err(e) => format!("could not be waited on: {e}"),
                };
                return Ended::Exited(why);
            }
            _ = super::stopped(stop) => {
                terminate(&mut child, ctx.stop_timeout, i).await;
                return Ended::Stopped;
            }
            _ = tokio::time::sleep(poll) => {}
        }

        if let Some(since) = terminating {
            if since.elapsed() > ctx.stop_timeout {
                warn!(
                    replica = i,
                    "replica did not stop in {:?}; killing it", ctx.stop_timeout
                );
                kill_group(&mut child);
            }
            continue;
        }
        if addrs.is_none() {
            addrs = read_report(report, pid);
            if let Some(a) = addrs {
                replica.set_addrs(Some(a));
                info!(replica = i, gpu = %device, http = %a.http, grpc = %a.grpc, "replica bound");
            }
        }
        let Some(a) = addrs else { continue };
        let ready = probe_ready(&ctx.client, a.http, READY_POLL).await;
        if replica.set_ready(ready) {
            if ready {
                info!(replica = i, gpu = %device, "replica ready");
            } else {
                warn!(replica = i, gpu = %device, "replica not ready; routing around it");
            }
        }
        if ready {
            if !was_ready {
                was_ready = true;
                // The GPU's next replica may start now.
                drop(start_lock.take());
                *backoff = RESTART_BACKOFF_MIN;
                if std::mem::take(first_start) {
                    ctx.first_starts.ready();
                }
            }
            unready_since = None;
        } else if was_ready {
            let since = *unready_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= ctx.wedged_after {
                error!(
                    replica = i,
                    gpu = %device,
                    "replica has not answered /readyz for {:?}; stopping it to restart",
                    ctx.wedged_after
                );
                send_sigterm(&child);
                terminating = Some(Instant::now());
            }
        }
    }
}

fn spawn(ctx: &LaunchContext, device: &str, report: &Path) -> std::io::Result<Child> {
    let mut cmd = Command::new(&ctx.program);
    cmd.args(&ctx.prefix_args)
        .args(plan::replica_args(&ctx.argv, report))
        .env("CUDA_VISIBLE_DEVICES", device)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // A supervisor task that is dropped -- the runtime ending -- takes
        // its process with it.
        .kill_on_drop(true);
    if std::env::var_os("PYTHONUNBUFFERED").is_none() {
        // Its stdout is a pipe now, which Python block-buffers: a `print` from
        // the engine would reach the log minutes after the lines around it.
        cmd.env("PYTHONUNBUFFERED", "1");
    }
    #[cfg(unix)]
    {
        // Its own process group: a terminal's Ctrl-C reaches the launcher
        // alone, which then stops the replicas in order instead of every
        // process draining on its own.
        cmd.process_group(0);
    }
    #[cfg(target_os = "linux")]
    {
        let launcher = std::process::id();
        // SAFETY: only async-signal-safe calls between fork and exec.
        unsafe {
            cmd.pre_exec(move || {
                // A launcher that dies without stopping its replicas (SIGKILL,
                // a crash) takes them with it instead of leaving GPUs and ports
                // held.  The signal follows the *thread* that forked: this runs
                // on a tokio worker, which lives as long as the launcher's
                // runtime.
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // The launcher died before the prctl took effect.
                if libc::getppid() as u32 != launcher {
                    libc::_exit(1);
                }
                Ok(())
            });
        }
    }
    cmd.spawn()
}

fn read_report(path: &Path, pid: u32) -> Option<BoundAddrs> {
    let bytes = std::fs::read(path).ok()?;
    let a: BoundAddrs = serde_json::from_slice(&bytes).ok()?;
    (a.pid == pid).then_some(a)
}

fn describe_exit(status: ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return format!("was killed by signal {sig}");
        }
    }
    match status.code() {
        Some(code) => format!("exited with status {code}"),
        None => format!("exited ({status})"),
    }
}

fn send_sigterm(child: &Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // SAFETY: plain kill(2) on a child we have not reaped.
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    let _ = child;
}

/// SIGKILL the replica's process group, so whatever it started goes too.
fn kill_group(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // SAFETY: as above; the group is the child's own (`process_group(0)`).
        unsafe {
            libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
        }
    }
    let _ = child.start_kill();
}

/// SIGTERM, wait out the replica's own drain, then SIGKILL.
async fn terminate(child: &mut Child, within: Duration, replica: usize) {
    send_sigterm(child);
    match tokio::time::timeout(within, child.wait()).await {
        Ok(status) => {
            let how = status.map_or_else(|e| e.to_string(), describe_exit);
            info!(replica, "replica stopped: {how}");
        }
        Err(_) => {
            warn!(replica, "replica did not stop in {within:?}; killing it");
            kill_group(child);
            let _ = child.wait().await;
        }
    }
}

// ---- logs -------------------------------------------------------------------

/// How a replica's log lines are marked when the launcher re-emits them.
#[derive(Clone)]
struct LogTag {
    text: Arc<[u8]>,
    json: Option<Arc<[u8]>>,
}

impl LogTag {
    fn new(replica: usize, device: &str, json: bool) -> Self {
        let device_json = serde_json::to_string(device).unwrap_or_else(|_| "\"?\"".into());
        Self {
            text: format!("[replica {replica} gpu {device}] ")
                .into_bytes()
                .into(),
            json: json.then(|| {
                format!("\"replica\":{replica},\"gpu\":{device_json}")
                    .into_bytes()
                    .into()
            }),
        }
    }

    /// `line` (its newline included) marked: a JSON object gets two fields
    /// up front, so it stays one object a log pipeline can parse; anything
    /// else -- text logs, a Python traceback -- gets a prefix.
    fn apply(&self, line: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(line.len() + self.text.len() + 2);
        if let (Some(fields), Some(rest)) = (&self.json, line.strip_prefix(b"{")) {
            out.push(b'{');
            out.extend_from_slice(fields);
            if !rest.trim_ascii_start().starts_with(b"}") {
                out.push(b',');
            }
            out.extend_from_slice(rest);
        } else {
            out.extend_from_slice(&self.text);
            out.extend_from_slice(line);
        }
        if !out.ends_with(b"\n") {
            out.push(b'\n');
        }
        out
    }
}

/// The launcher's stdout and stderr, written whole lines at a time from one
/// thread each, so replicas' lines never interleave mid-line.
#[derive(Clone)]
pub struct LogSinks {
    stdout: mpsc::Sender<Vec<u8>>,
    stderr: mpsc::Sender<Vec<u8>>,
    pub json: bool,
}

/// Lines buffered per stream before a replica's log writes wait.
const LOG_QUEUE: usize = 1024;

impl LogSinks {
    pub fn start(json: bool) -> Self {
        fn writer(mut rx: mpsc::Receiver<Vec<u8>>, err: bool) {
            use std::io::Write;
            while let Some(line) = rx.blocking_recv() {
                let _ = if err {
                    std::io::stderr().lock().write_all(&line)
                } else {
                    std::io::stdout().lock().write_all(&line)
                };
            }
        }
        let (stdout, out_rx) = mpsc::channel(LOG_QUEUE);
        let (stderr, err_rx) = mpsc::channel(LOG_QUEUE);
        let _ = std::thread::Builder::new()
            .name("oasr-replica-stdout".into())
            .spawn(move || writer(out_rx, false));
        let _ = std::thread::Builder::new()
            .name("oasr-replica-stderr".into())
            .spawn(move || writer(err_rx, true));
        Self {
            stdout,
            stderr,
            json,
        }
    }

    fn pump(&self, from: impl AsyncRead + Unpin + Send + 'static, tag: LogTag, err: bool) {
        let to = if err {
            self.stderr.clone()
        } else {
            self.stdout.clone()
        };
        tokio::spawn(async move {
            let mut reader = BufReader::new(from);
            let mut line = Vec::new();
            loop {
                line.clear();
                match reader.read_until(b'\n', &mut line).await {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {
                        if to.send(tag.apply(&line)).await.is_err() {
                            return;
                        }
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_lines_are_prefixed_and_json_lines_gain_fields() {
        let t = LogTag::new(1, "GPU-a", true);
        assert_eq!(
            t.apply(b"{\"level\":\"INFO\"}\n"),
            b"{\"replica\":1,\"gpu\":\"GPU-a\",\"level\":\"INFO\"}\n"
        );
        assert_eq!(t.apply(b"{}"), b"{\"replica\":1,\"gpu\":\"GPU-a\"}\n");
        assert_eq!(
            t.apply(b"Traceback (most recent call last):\n"),
            b"[replica 1 gpu GPU-a] Traceback (most recent call last):\n"
        );
        let t = LogTag::new(0, "0", false);
        assert_eq!(t.apply(b"{\"a\":1}\n"), b"[replica 0 gpu 0] {\"a\":1}\n");
    }

    #[test]
    fn a_report_from_another_process_is_ignored() {
        let dir = std::env::temp_dir().join(format!("oasr-report-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.json");
        let a = BoundAddrs {
            pid: 42,
            http: "127.0.0.1:1".parse().unwrap(),
            grpc: "127.0.0.1:2".parse().unwrap(),
        };
        std::fs::write(&path, serde_json::to_vec(&a).unwrap()).unwrap();
        assert_eq!(read_report(&path, 42), Some(a));
        assert_eq!(read_report(&path, 43), None);
        std::fs::write(&path, b"{\"pid\":42").unwrap();
        assert_eq!(
            read_report(&path, 42),
            None,
            "a partial file is not a report"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_first_starts_are_fatal_only_when_every_one_failed() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let f = FirstStarts::new(2, tx);
        f.failed(0, "exited with status 1");
        assert!(rx.try_recv().is_err());
        f.failed(1, "exited with status 1");
        assert!(rx.try_recv().is_ok());

        let (tx, mut rx) = mpsc::unbounded_channel();
        let f = FirstStarts::new(2, tx);
        f.ready();
        f.failed(1, "exited with status 1");
        assert!(rx.try_recv().is_err());
    }
}
