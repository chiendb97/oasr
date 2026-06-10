// Copyright 2024 OASR Authors / SPDX-License-Identifier: Apache-2.0
//
// Browser side of the OASR web demo.  Captures audio (file upload or mic) as
// 16 kHz mono Float32 PCM and sends it to the same-origin bridge:
//   - offline : POST /api/recognize  (whole clip)        -> JSON transcript
//   - stream  : WS   /api/stream      (config + chunks)   -> partial/final
"use strict";

const TARGET_SR = 16000;
let CHUNK_MS = 320; // overwritten from /api/info

// ---- DOM -------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const fileInput = $("file-input");
const transcribeBtn = $("transcribe-btn");
const recordBtn = $("record-btn");
const statusEl = $("status");
const committedEl = $("committed");
const partialEl = $("partial");
const uploadPanel = $("upload-panel");
const micPanel = $("mic-panel");
const recHint = $("rec-hint");

const getMode = () => document.querySelector('input[name="mode"]:checked').value;
const getSource = () => document.querySelector('input[name="source"]:checked').value;

// ---- transcript rendering --------------------------------------------------

let committed = "";

function resetTranscript() {
  committed = "";
  committedEl.textContent = "";
  partialEl.textContent = "";
}
function setPartial(text) {
  partialEl.textContent = committed && text ? " " + text : text;
}
function commitFinal(text) {
  if (text) committed += (committed ? " " : "") + text;
  committedEl.textContent = committed;
  partialEl.textContent = "";
}
function setOffline(text) {
  committed = text || "";
  committedEl.textContent = committed;
  partialEl.textContent = "";
}

// ---- status / UI state -----------------------------------------------------

function setStatus(msg, kind = "") {
  statusEl.textContent = msg;
  statusEl.className = "status" + (kind ? " " + kind : "");
  statusEl.classList.toggle("hidden", !msg);
}

let busy = false;
function setBusy(b) {
  busy = b;
  refreshControls();
}
function refreshControls() {
  const isMic = getSource() === "mic";
  uploadPanel.classList.toggle("hidden", isMic);
  micPanel.classList.toggle("hidden", !isMic);
  transcribeBtn.disabled = busy || !fileInput.files.length;
  const micOk = micUnavailableReason() === null;
  recordBtn.disabled = (busy && !recording) || (!recording && !micOk);
}

// ---- audio helpers ---------------------------------------------------------

function resampleTo16k(input, srcRate) {
  if (srcRate === TARGET_SR) return input.slice(); // always return a fresh copy
  const ratio = srcRate / TARGET_SR;
  const outLen = Math.max(0, Math.floor(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    out[i] = input[i0] * (1 - (pos - i0)) + input[i1] * (pos - i0);
  }
  return out;
}

function mergeFloat32(chunks) {
  let len = 0;
  for (const c of chunks) len += c.length;
  const out = new Float32Array(len);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function decodeFileTo16k(file) {
  const buf = await file.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx({ sampleRate: TARGET_SR });
  try {
    const audioBuf = await ctx.decodeAudioData(buf);
    return resampleTo16k(audioBuf.getChannelData(0), audioBuf.sampleRate);
  } finally {
    ctx.close();
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- offline (HTTP POST) ---------------------------------------------------

async function postOffline(pcm) {
  const resp = await fetch(`/api/recognize?sample_rate=${TARGET_SR}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: pcm.buffer,
  });
  let data;
  try { data = await resp.json(); } catch { data = {}; }
  if (!resp.ok || data.error) {
    throw new Error(data.error || `HTTP ${resp.status}`);
  }
  return data.transcript || "";
}

// ---- streaming (WebSocket) -------------------------------------------------

function openStream(onFinalClose) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/stream`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    if (m.type === "partial") setPartial(m.transcript);
    else if (m.type === "final") commitFinal(m.transcript);
    else if (m.type === "error") setStatus("Server error: " + m.error, "error");
    else if (m.type === "done") { try { ws.close(); } catch {} }
  };
  ws.onerror = () => setStatus("WebSocket error (is the bridge running?)", "error");
  ws.onclose = () => onFinalClose && onFinalClose();
  return ws;
}

function wsReady(ws) {
  return new Promise((resolve, reject) => {
    if (ws.readyState === WebSocket.OPEN) return resolve();
    ws.addEventListener("open", () => resolve(), { once: true });
    ws.addEventListener("error", () => reject(new Error("ws open failed")), { once: true });
  });
}

// ---- file flows ------------------------------------------------------------

async function runFileOffline(file) {
  setStatus("Decoding…", "busy");
  const pcm = await decodeFileTo16k(file);
  setStatus("Transcribing…", "busy");
  const text = await postOffline(pcm);
  setOffline(text);
  setStatus(text ? "Done." : "Done (empty transcript).", "ok");
}

async function runFileStreaming(file) {
  setStatus("Decoding…", "busy");
  const pcm = await decodeFileTo16k(file);
  resetTranscript();
  await new Promise((resolve) => {
    const ws = openStream(resolve);
    ws.addEventListener("open", async () => {
      setStatus("Streaming…", "busy");
      ws.send(JSON.stringify({ sample_rate: TARGET_SR, interim: true }));
      const step = Math.max(1, Math.floor((TARGET_SR * CHUNK_MS) / 1000));
      for (let i = 0; i < pcm.length; i += step) {
        if (ws.readyState !== WebSocket.OPEN) break;
        ws.send(pcm.slice(i, i + step).buffer);
        await sleep(CHUNK_MS); // pace ~realtime so partials are visible
      }
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "eof" }));
    }, { once: true });
  });
  setStatus("Done.", "ok");
}

// ---- mic capture -----------------------------------------------------------

// Browsers only expose getUserMedia in a secure context (HTTPS, or http on
// localhost/127.0.0.1).  Over plain HTTP to a remote host, navigator.mediaDevices
// is undefined.  Detect that up front and explain how to fix it.
function micUnavailableReason() {
  const hasModern = navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
  const hasLegacy = navigator.getUserMedia || navigator.webkitGetUserMedia ||
                    navigator.mozGetUserMedia;
  if (hasModern || hasLegacy) return null;
  if (window.isSecureContext === false) {
    return "Microphone needs a secure context. Open this page via " +
           "http://localhost (e.g. SSH-tunnel it: ssh -L 8000:localhost:8000 " +
           "<host>) or serve it over HTTPS. File upload still works.";
  }
  return "Microphone API is unavailable in this browser. File upload still works.";
}

function getUserMediaCompat(constraints) {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    return navigator.mediaDevices.getUserMedia(constraints);
  }
  const legacy = navigator.getUserMedia || navigator.webkitGetUserMedia ||
                 navigator.mozGetUserMedia;
  if (legacy) {
    return new Promise((res, rej) => legacy.call(navigator, constraints, res, rej));
  }
  return Promise.reject(new Error(micUnavailableReason() || "getUserMedia unavailable"));
}

let recording = false;
let micCtx = null;
let micStream = null;
let micNodes = null;
let micChunks = []; // for offline mic
let micWs = null; // for streaming mic

// ---- recording UI ----------------------------------------------------------

function enterRecordingUI() {
  recordBtn.classList.add("recording");
  recordBtn.setAttribute("aria-label", "Stop recording");
  if (recHint) recHint.textContent = "Recording… tap to stop";
}

function exitRecordingUI() {
  recordBtn.classList.remove("recording");
  recordBtn.setAttribute("aria-label", "Start recording");
  if (recHint) recHint.textContent = "Tap to record";
}

async function startMic() {
  const reason = micUnavailableReason();
  if (reason) { setStatus(reason, "error"); return; }
  const mode = getMode();
  resetTranscript();
  setStatus("Requesting microphone…", "busy");
  try {
    micStream = await getUserMediaCompat({ audio: true });
  } catch (err) {
    setStatus("Microphone access denied: " + ((err && err.message) || err), "error");
    return;
  }
  const Ctx = window.AudioContext || window.webkitAudioContext;
  micCtx = new Ctx({ sampleRate: TARGET_SR });
  try { await micCtx.resume(); } catch {}
  const source = micCtx.createMediaStreamSource(micStream);
  // ScriptProcessorNode is deprecated but universally available; fine for a demo.
  const proc = micCtx.createScriptProcessor(4096, 1, 1);
  const mute = micCtx.createGain();
  mute.gain.value = 0; // avoid echoing the mic to the speakers
  micChunks = [];

  if (mode === "streaming") {
    micWs = openStream(() => {});
    await wsReady(micWs).catch(() => {});
    if (micWs.readyState === WebSocket.OPEN) {
      micWs.send(JSON.stringify({ sample_rate: TARGET_SR, interim: true }));
    }
  }

  proc.onaudioprocess = (e) => {
    const frame = resampleTo16k(e.inputBuffer.getChannelData(0), micCtx.sampleRate);
    if (mode === "streaming") {
      if (micWs && micWs.readyState === WebSocket.OPEN) micWs.send(frame.buffer);
    } else {
      micChunks.push(frame);
    }
  };

  source.connect(proc);
  proc.connect(mute);
  mute.connect(micCtx.destination);
  micNodes = { source, proc, mute };

  recording = true;
  refreshControls();
  enterRecordingUI();
  setStatus("");
}

async function stopMic() {
  if (!recording) return;
  const mode = getMode();
  recording = false;
  exitRecordingUI();

  // Tear down the audio graph.
  try { micNodes.proc.disconnect(); micNodes.source.disconnect(); micNodes.mute.disconnect(); } catch {}
  try { micStream.getTracks().forEach((t) => t.stop()); } catch {}
  try { await micCtx.close(); } catch {}

  if (mode === "streaming") {
    if (micWs && micWs.readyState === WebSocket.OPEN) {
      micWs.send(JSON.stringify({ type: "eof" }));
    }
    setStatus("Finishing…", "busy");
  } else {
    setBusy(true);
    try {
      setStatus("Transcribing…", "busy");
      const text = await postOffline(mergeFloat32(micChunks));
      setOffline(text);
      setStatus(text ? "Done." : "Done (empty transcript).", "ok");
    } catch (err) {
      setStatus("Error: " + err.message, "error");
    } finally {
      setBusy(false);
    }
  }
  refreshControls();
}

// ---- event wiring ----------------------------------------------------------

document.querySelectorAll('input[name="source"]').forEach((el) =>
  el.addEventListener("change", () => {
    refreshControls();
    if (getSource() === "mic") {
      const reason = micUnavailableReason();
      setStatus(reason || "", reason ? "error" : "");
    } else {
      setStatus("");
    }
  }));
document.querySelectorAll('input[name="mode"]').forEach((el) =>
  el.addEventListener("change", resetTranscript));

fileInput.addEventListener("change", () => {
  const nameEl = $("file-name");
  if (nameEl) nameEl.textContent = fileInput.files.length
    ? fileInput.files[0].name
    : "Choose an audio file…";
  refreshControls();
});

transcribeBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  setBusy(true);
  resetTranscript();
  try {
    if (getMode() === "streaming") await runFileStreaming(file);
    else await runFileOffline(file);
  } catch (err) {
    setStatus("Error: " + err.message, "error");
  } finally {
    setBusy(false);
  }
});

recordBtn.addEventListener("click", () => { recording ? stopMic() : startMic(); });

// ---- init ------------------------------------------------------------------

(async function init() {
  try {
    const info = await (await fetch("/api/info")).json();
    if (info.chunk_ms) CHUNK_MS = info.chunk_ms;
  } catch { /* keep defaults */ }
  const reason = micUnavailableReason();
  if (reason) setStatus(reason);
  refreshControls();
})();
