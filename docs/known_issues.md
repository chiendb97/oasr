# Known issues

Tracked defects with reproductions. Each entry is written so it can be pasted
into the issue tracker verbatim. Fixes for kernel-level items belong on their
own branch, not on feature/refactor branches that merely surface them.

---

## GPU-DEC-1 — `ctc_gpu` `blank_threshold` frame-skip garbles transcripts

**Severity:** high (wrong transcripts, affects offline *and* streaming)
**Component:** `oasr/ctc_decode.py` → `csrc/` GPU CTC prefix-beam kernel
**Status:** open — needs CUDA-kernel fix
**Surfaced by:** `refactor/gpu-only-engine` (engine switched to `decoder_type="ctc_gpu"`)

### Symptom
With the default `GpuDecoderConfig.blank_threshold = 0.98`, the GPU prefix-beam
decoder duplicates subword pieces even on **clean, full-sequence offline**
log-probs:

| reference (CPU prefix-beam / GPU @ `blank_threshold=1.0`) | GPU @ `blank_threshold=0.98` (default) |
|---|---|
| `... EXHIBITION` | `... EXHIBIT EXHIBITIONION` |
| `... THE CHINESE TOOK ... WOOD BLOCKS ... WOOD CUTTERS ...` | `... THE CHIN CHINESESE TOOK ... WOOD BLOCKSS ... WOOD CUTTERTERS ...` |
| `... ALL THE ARTS AND CRAFTS ...` | `... ALL THE ARTS AND CRAFTS ...` (offline ok here, but streaming → `ARTSS`) |

### Root cause (evidence)
The decoder *algorithm* is correct: with the frame-skip disabled
(`blank_threshold = 1.0`) the GPU one-shot decoder is **bit-identical to the CPU
`prefix_beam` decoder** on every utterance, on both offline and streaming
log-probs. The duplication is introduced solely by the
"skip frames where `P(blank) > threshold`" optimization at `0.98` — it mishandles
repeat-token collapse at emission boundaries (a near-certain-blank frame that
should act as a repeat separator is dropped).

### Why it was hidden
The engine test compared two **CPU** `prefix_beam` decodes before the refactor;
the CPU `prefix_beam` path has no blank-skip, so the bug never ran. Removing the
CPU decoders from the engine forced the test onto `ctc_gpu`, exposing it.

### Repro
```python
import torch
from oasr.engine import ASREngine, EngineConfig
from oasr.ctc_decode import GpuDecoderConfig

cfg = lambda thr: EngineConfig(
    ckpt_dir=CKPT, device="cuda", dtype=torch.float16,
    service_mode="offline", decoder_type="ctc_gpu",
    gpu_decoder_config=GpuDecoderConfig(blank_threshold=thr))

print(ASREngine(cfg(0.98)).transcribe_offline([WAV]))  # garbled: EXHIBIT EXHIBITIONION
print(ASREngine(cfg(1.0)).transcribe_offline([WAV]))   # correct: EXHIBITION (== CPU)
```
(`CKPT = .../20210610_u2pp_conformer_exp_librispeech`,
 `WAV = .../ljspeech-sr16k-dataset/wavs/LJ001-0001.wav`)

### Notes for the fix
- The frame-skip is a ~5× speed lever (GPU decode 1.9 ms vs 9.1 ms/utt at T=240,
  V=5008), so the goal is to make skipping **lossless**, not to delete it.
- Verify against the CPU `prefix_beam` decoder as the oracle (it matches GPU
  exactly when skipping is off).

---

## GPU-DEC-2 — `GpuStreamingDecoder` truncates the tail when blank-skip is off

**Severity:** high (drops the end of streaming transcripts)
**Component:** `oasr/ctc_decode.py::GpuStreamingDecoder` (`decode_chunk` / `finalize_stream`) → `csrc/` streaming CTC kernel
**Status:** open — needs CUDA-kernel fix
**Depends on / interacts with:** GPU-DEC-1

### Symptom
Setting `blank_threshold = 1.0` fixes the offline garble (GPU-DEC-1) but makes the
**streaming** decoder drop the end of the transcript:

| offline @ `blank_threshold=1.0` (correct) | streaming(B=1) @ `blank_threshold=1.0` |
|---|---|
| `... REPRESENTED IN THE EXHIBITION` | `... ALL THE ARTS AND CRAFTS` (truncated) |
| `... OF THE NETHERLANDS BY A SIMILAR PROCESS` | `... OF THE NETHERLANDS` (truncated) |

So there is **no single `blank_threshold` that makes the engine correct
end-to-end**: `0.98` garbles, `1.0` truncates streaming. The one-shot decoder on
the *same* streaming log-probs at `1.0` produces the full correct text, so the
truncation lives in the incremental `decode_chunk` / `finalize_stream` state
machine, not in the logits.

### Repro
```python
on = ASREngine(EngineConfig(
    ckpt_dir=CKPT, device="cuda", dtype=torch.float16, decoder_type="ctc_gpu",
    chunk_size=16, num_left_chunks=-1, max_batch_size=1,
    gpu_decoder_config=GpuDecoderConfig(blank_threshold=1.0)))
print(on.transcribe([WAV]))  # tail dropped vs offline
```

### Test impact
`tests/test_engine.py::TestASREngine::test_streaming_matches_offline_single_stream`
asserts bit-exact streaming(B=1) == offline. That invariant is genuinely correct
for the *encoder* (the chunked paged forward agrees with the full-sequence
forward to within fp16 emission-boundary jitter that CTC collapse absorbs), but
it cannot hold while GPU-DEC-1/2 are open. The test is marked `xfail` referencing
this file until the kernel fixes land; remove the marker (restore exact-match)
once both are fixed — it should then pass bit-exactly.
