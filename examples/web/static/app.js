// Copyright 2024 OASR Authors / SPDX-License-Identifier: Apache-2.0
//
// Browser side of the OASR web demo.  Speaks OASR's own HTTP API, never a
// demo-only protocol — nothing here needs translating:
//   - offline : POST {server}/v1/audio/transcriptions  (multipart upload)
//   - stream  : WS   {server}/v1/realtime              (session + PCM frames)
//
// Uploads are sent as the original file, whatever container it is: the server
// decodes MP3 / M4A / FLAC / OGG / WAV itself.  Only the streaming path decodes
// in-browser, because a live session is chunked PCM by definition.
//
// Two ways to run it, and the default is the quiet one:
//   examples/web/server.py  — serves this page and relays /v1/* to oasr-server,
//                             so every call is same-origin and CORS never enters
//                             into it.  Nothing to configure here.
//   ?server=http://host:port — talk to oasr-server directly, which is then
//                             cross-origin and needs --cors-allow-origin.
"use strict";

const TARGET_SR = 16000;
const CHUNK_MS = 320;

// Empty means "this origin", i.e. whatever served the page — the relay case.
// `file://` is the exception: it has no usable origin, so a page opened by
// double-clicking keeps the absolute default instead of deriving `wss://null`.
const SERVER = (new URLSearchParams(location.search).get("server") ||
                (location.protocol === "file:" ? "http://127.0.0.1:8080" : "")).replace(/\/$/, "");

// Base for the realtime socket.  Derived from SERVER when it is set, from the
// page's own origin otherwise; either way http->ws and https->wss, so an HTTPS
// page (which is what the microphone needs off localhost) gets a secure socket.
const WS_BASE = (SERVER || location.origin).replace(/^http/, "ws");

// What to call the endpoint in a message: an explicit ?server=, or wherever this
// page came from — which in the relay case is the thing that knows the upstream.
const SERVER_LABEL = SERVER || location.origin;

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

// Wrap raw f32 PCM in a WAV container so mic audio can go through the same
// upload endpoint as a file.  16 bytes of header work is cheaper than a second
// code path.
function wavFromFloat32(pcm, sampleRate) {
  const buf = new ArrayBuffer(44 + pcm.length * 4);
  const view = new DataView(buf);
  const ascii = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + pcm.length * 4, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);            // IEEE float
  view.setUint16(22, 1, true);            // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 32, true);
  ascii(36, "data");
  view.setUint32(40, pcm.length * 4, true);
  new Float32Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}

// ---- offline (OpenAI-compatible upload) ------------------------------------

async function postOffline(fileOrBlob, filename) {
  const form = new FormData();
  form.append("file", fileOrBlob, filename || fileOrBlob.name || "audio.wav");
  form.append("response_format", "json");
  const resp = await fetch(`${SERVER}/v1/audio/transcriptions`, { method: "POST", body: form });
  let data;
  try { data = await resp.json(); } catch { data = {}; }
  if (!resp.ok) {
    throw new Error((data.error && data.error.message) || `HTTP ${resp.status}`);
  }
  return data.text || "";
}

// ---- streaming (WebSocket) -------------------------------------------------

function openStream(onFinalClose) {
  const ws = new WebSocket(WS_BASE + "/v1/realtime");
  ws.binaryType = "arraybuffer";
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    // `delta` is the increment; `text` is the transcript so far, which is what
    // a caption line wants — a revised partial arrives with an empty delta.
    if (m.type && m.type.endsWith("input_audio_transcription.delta")) {
      setPartial(m.text !== undefined ? m.text : m.delta);
    } else if (m.type && m.type.endsWith("input_audio_transcription.completed")) {
      commitFinal(m.transcript);
      try { ws.close(); } catch {}
    } else if (m.type === "error") {
      setStatus("Server error: " + ((m.error && m.error.message) || "unknown"), "error");
    }
  };
  ws.onerror = () => setStatus(SERVER
    ? `WebSocket error — is oasr-server running at ${SERVER} with --cors-allow-origin?`
    : `WebSocket error — is ${SERVER_LABEL} still up, and can it reach its ` +
      `--oasr-server? Its log names the reason.`, "error");
  ws.onclose = () => onFinalClose && onFinalClose();
  return ws;
}

// The session frame the realtime endpoint expects before any audio.
function startSession(ws) {
  ws.send(JSON.stringify({
    type: "session.update",
    session: { sample_rate: TARGET_SR, encoding: "LINEAR32F", interim_results: true },
  }));
}

function commitSession(ws) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  }
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
  // No in-browser decode: the server reads the container itself, so an MP3 or
  // an M4A voice memo is uploaded exactly as it sits on disk.
  setStatus("Transcribing…", "busy");
  const text = await postOffline(file);
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
      startSession(ws);
      const step = Math.max(1, Math.floor((TARGET_SR * CHUNK_MS) / 1000));
      for (let i = 0; i < pcm.length; i += step) {
        if (ws.readyState !== WebSocket.OPEN) break;
        ws.send(pcm.slice(i, i + step).buffer);
        await sleep(CHUNK_MS); // pace ~realtime so partials are visible
      }
      commitSession(ws);
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
    return "Microphone needs a secure context, which this page is not. Either restart " +
           "server.py with --tls-self-signed, or allowlist this origin in the browser " +
           "(Chrome: chrome://flags/#unsafely-treat-insecure-origin-as-secure; Firefox: " +
           "about:config → media.devices.insecure.enabled). File upload still works.";
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
    if (micWs.readyState === WebSocket.OPEN) startSession(micWs);
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
    if (micWs) commitSession(micWs);
    setStatus("Finishing…", "busy");
  } else {
    setBusy(true);
    try {
      setStatus("Transcribing…", "busy");
      const wav = wavFromFloat32(mergeFloat32(micChunks), TARGET_SR);
      const text = await postOffline(wav, "recording.wav");
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
  const serverEl = document.getElementById("server-url");
  if (serverEl) serverEl.textContent = SERVER_LABEL;
  // Fail early and specifically: a dead server and a blocked origin look the
  // same from a failed fetch inside a transcription, and neither is the page's
  // fault — so name the two fixes rather than reporting "failed".
  try {
    const resp = await fetch(`${SERVER}/v1/models`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  } catch {
    setStatus(SERVER
      ? `Cannot reach oasr-server at ${SERVER}. Start it with ` +
        `--cors-allow-origin '*', or drop ?server= and run examples/web/server.py.`
      : `Cannot reach oasr-server through ${SERVER_LABEL}. Serve this page with ` +
        `examples/web/server.py --oasr-server http://host:8080 (its log says why ` +
        `the upstream failed), or pass ?server=http://host:8080 to call it directly.`,
      "error");
  }
  const reason = micUnavailableReason();
  if (reason) setStatus(reason);
  refreshControls();
})();
