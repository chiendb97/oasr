// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//! Sample-rate conversion for f32 mono PCM.
//!
//! The engine is **waveform-only at the model's own sample rate** — nothing
//! downstream of PyO3 looks at `Request.sample_rate` except long-form window
//! arithmetic, and every frame count is derived from
//! `FeatureConfig.sample_rate`.  So audio that arrives at another rate has to be
//! converted here, in the front-end, before it crosses into Python.  Without
//! this a client posting 8 kHz telephony PCM or 44.1 kHz media — both of which
//! the documented `sampleRateHertz` field invites — got a confidently wrong
//! transcript with no error anywhere.
//!
//! Conversion is a windowed-sinc polyphase resample via [`rubato`], which scales
//! the filter cutoff by the ratio when downsampling, so the anti-alias filter is
//! handled for us.  Cost is a few hundred µs per audio-second on one core,
//! negligible next to the GPU step.
//!
//! Note that 8 kHz → 16 kHz is *correct but degraded*: the band above 4 kHz is
//! simply not in the source.  The honest fix for telephony is an 8 kHz model
//! variant, which the feature-frontend registry already anticipates.

use rubato::{
    calculate_cutoff, Resampler as _, SincFixedIn, SincInterpolationParameters,
    SincInterpolationType, WindowFunction,
};

use crate::audio::AudioError;

/// Lowest sample rate we will accept from a client.
pub const MIN_SAMPLE_RATE: u32 = 4_000;
/// Highest sample rate we will accept from a client.
pub const MAX_SAMPLE_RATE: u32 = 384_000;

/// Input frames consumed per resampler call.  Large enough that the per-call
/// overhead is amortised, small enough that a streaming chunk rarely has to
/// wait for a second one before producing output.
const CHUNK: usize = 1024;

/// Windowed-sinc filter length.  128 taps at 128× oversampling is ~64 KiB of
/// filter table per resampler — the streaming path keeps one per open stream,
/// so the 256/256 "high quality" preset would cost 256 KiB × N streams for
/// stopband depth that speech does not need.
const SINC_LEN: usize = 128;
const OVERSAMPLING: usize = 128;

/// Reject a client-declared sample rate that is outside the plausible range.
///
/// A rate of 0 means "unset" at the wire layer and is substituted with a
/// default before reaching here, so 0 arriving is a caller bug, not a client
/// one — it is still rejected rather than silently treated as 16 kHz.
pub fn validate_sample_rate(hz: u32) -> Result<(), AudioError> {
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&hz) {
        return Err(AudioError::UnsupportedSampleRate(hz));
    }
    Ok(())
}

/// Number of output frames a resample of `n_in` frames should produce.
fn expected_out(n_in: u64, from_hz: u32, to_hz: u32) -> u64 {
    ((n_in as f64) * (to_hz as f64) / (from_hz as f64)).round() as u64
}

/// A stateful mono f32 resampler.
///
/// State spans [`push`](Self::push) calls, making arbitrary chunking equivalent
/// to one-shot input. [`flush`](Self::flush) compensates filter delay and returns
/// exactly `round(n_in * to/from)` frames overall.
pub struct Resampler {
    inner: SincFixedIn<f32>,
    from_hz: u32,
    to_hz: u32,
    /// Input frames received but not yet handed to `inner`.
    pending: Vec<f32>,
    /// Chunk-sized staging so the steady-state path never allocates.
    stage_in: Vec<f32>,
    stage_out: Vec<Vec<f32>>,
    /// Output frames of filter group delay still to be discarded.
    delay_left: usize,
    /// Input frames accepted so far (across the whole stream).
    consumed: u64,
    /// Output frames emitted so far, after the group-delay trim.
    emitted: u64,
}

impl Resampler {
    /// Build a resampler from `from_hz` to `to_hz`.  Both rates are validated;
    /// equal rates are still accepted (the caller decides whether to bypass).
    pub fn new(from_hz: u32, to_hz: u32) -> Result<Self, AudioError> {
        validate_sample_rate(from_hz)?;
        validate_sample_rate(to_hz)?;
        let ratio = f64::from(to_hz) / f64::from(from_hz);
        let params = SincInterpolationParameters {
            sinc_len: SINC_LEN,
            f_cutoff: calculate_cutoff(SINC_LEN, WindowFunction::BlackmanHarris2),
            oversampling_factor: OVERSAMPLING,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::BlackmanHarris2,
        };
        let inner = SincFixedIn::<f32>::new(ratio, 1.0, params, CHUNK, 1)
            .map_err(|e| AudioError::Resample(e.to_string()))?;
        let out_cap = inner.output_frames_max();
        let delay_left = inner.output_delay();
        Ok(Self {
            inner,
            from_hz,
            to_hz,
            pending: Vec::with_capacity(CHUNK * 2),
            stage_in: vec![0.0; CHUNK],
            stage_out: vec![vec![0.0; out_cap]],
            delay_left,
            consumed: 0,
            emitted: 0,
        })
    }

    /// Source sample rate.
    pub fn from_hz(&self) -> u32 {
        self.from_hz
    }

    /// Target sample rate.
    pub fn to_hz(&self) -> u32 {
        self.to_hz
    }

    /// Feed `input` and append whatever output is ready to `out`.
    ///
    /// Output lags input by up to one [`CHUNK`]; the remainder comes out of
    /// [`flush`](Self::flush).
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), AudioError> {
        self.consumed += input.len() as u64;
        self.pending.extend_from_slice(input);
        let mut read = 0usize;
        while self.pending.len() - read >= CHUNK {
            self.stage_in
                .copy_from_slice(&self.pending[read..read + CHUNK]);
            read += CHUNK;
            self.run_chunk(out)?;
        }
        self.pending.drain(..read);
        Ok(())
    }

    /// Finish the stream: push the tail through and emit the frames still held
    /// inside the filter, so the total output is exactly
    /// `round(total_input * to/from)` frames.
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), AudioError> {
        let target = expected_out(self.consumed, self.from_hz, self.to_hz);

        // Remaining partial chunk, zero-padded to CHUNK.
        if !self.pending.is_empty() {
            let n = self.pending.len();
            self.stage_in[..n].copy_from_slice(&self.pending);
            self.stage_in[n..].fill(0.0);
            self.pending.clear();
            self.run_chunk(out)?;
        }

        // Silence chunks until the delayed tail has come out.  Bounded so a
        // ratio the arithmetic did not anticipate cannot spin here; the
        // truncate/pad below still makes the length exact.
        let mut guard = 0;
        while self.emitted < target && guard < 64 {
            self.stage_in.fill(0.0);
            self.run_chunk(out)?;
            guard += 1;
        }

        // Snap to the expected length.  Downstream frame arithmetic (and every
        // test) is much easier to reason about when a resample is a pure
        // function of the input length.
        let extra = self.emitted.saturating_sub(target) as usize;
        out.truncate(out.len() - extra.min(out.len()));
        let short = target.saturating_sub(self.emitted) as usize;
        out.resize(out.len() + short, 0.0);
        self.emitted = target;
        Ok(())
    }

    /// Run one `CHUNK`-sized input from `stage_in`, appending post-delay-trim
    /// output frames to `out`.
    fn run_chunk(&mut self, out: &mut Vec<f32>) -> Result<(), AudioError> {
        let (_, n_out) = self
            .inner
            .process_into_buffer(
                std::slice::from_ref(&self.stage_in),
                &mut self.stage_out,
                None,
            )
            .map_err(|e| AudioError::Resample(e.to_string()))?;
        let produced = &self.stage_out[0][..n_out];
        let skip = self.delay_left.min(produced.len());
        self.delay_left -= skip;
        out.extend_from_slice(&produced[skip..]);
        self.emitted += (produced.len() - skip) as u64;
        Ok(())
    }
}

/// One-shot resample of a whole clip.  Returns `round(n * to/from)` frames.
///
/// Equal rates short-circuit to a copy, so callers can call this
/// unconditionally rather than branching at every call site — the branch that
/// gets forgotten is the one that ships a wrong transcript.
pub fn resample_mono(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>, AudioError> {
    validate_sample_rate(from_hz)?;
    validate_sample_rate(to_hz)?;
    if from_hz == to_hz {
        return Ok(samples.to_vec());
    }
    let mut r = Resampler::new(from_hz, to_hz)?;
    let mut out =
        Vec::with_capacity(expected_out(samples.len() as u64, from_hz, to_hz) as usize + CHUNK);
    r.push(samples, &mut out)?;
    r.flush(&mut out)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(n: usize, freq_hz: f32, rate: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (TAU * freq_hz * (i as f32) / (rate as f32)).sin())
            .collect()
    }

    /// Correlate `a` against a reference sine at `freq_hz`, ignoring the first
    /// and last 10% (filter edges).  Returns the recovered amplitude.
    fn amplitude_at(a: &[f32], freq_hz: f32, rate: u32) -> f32 {
        let lo = a.len() / 10;
        let hi = a.len() - a.len() / 10;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &v) in a.iter().enumerate().take(hi).skip(lo) {
            let ph = (TAU as f64) * (freq_hz as f64) * (i as f64) / (rate as f64);
            re += (v as f64) * ph.cos();
            im += (v as f64) * ph.sin();
        }
        let n = (hi - lo) as f64;
        (2.0 * (re * re + im * im).sqrt() / n) as f32
    }

    #[test]
    fn output_length_is_exactly_the_rate_ratio() {
        for (from, to, n) in [
            (8_000u32, 16_000u32, 8_000usize),
            (44_100, 16_000, 44_100),
            (48_000, 16_000, 4_800),
            (22_050, 16_000, 3_307),
            (16_000, 16_000, 1_234),
        ] {
            let out = resample_mono(&sine(n, 440.0, from), from, to).unwrap();
            let want = ((n as f64) * f64::from(to) / f64::from(from)).round() as usize;
            assert_eq!(out.len(), want, "{from} -> {to}");
        }
    }

    /// The whole point: a tone in the passband must survive the conversion at
    /// its original frequency and amplitude.  A resampler that got the ratio
    /// backwards, or dropped the anti-alias filter, fails here.
    #[test]
    fn passband_tone_survives_downsampling() {
        // 1 kHz at 44.1 kHz -> 16 kHz.  Well inside the 8 kHz Nyquist.
        let src = sine(44_100, 1_000.0, 44_100);
        let out = resample_mono(&src, 44_100, 16_000).unwrap();
        let amp = amplitude_at(&out, 1_000.0, 16_000);
        assert!((amp - 1.0).abs() < 0.05, "amplitude {amp} != 1.0");
    }

    #[test]
    fn passband_tone_survives_upsampling() {
        // Telephony band: 1 kHz at 8 kHz -> 16 kHz.
        let src = sine(8_000, 1_000.0, 8_000);
        let out = resample_mono(&src, 8_000, 16_000).unwrap();
        let amp = amplitude_at(&out, 1_000.0, 16_000);
        assert!((amp - 1.0).abs() < 0.05, "amplitude {amp} != 1.0");
    }

    /// Content above the target Nyquist must be filtered out, not folded back
    /// into the speech band.  Without the anti-alias filter a 12 kHz tone
    /// resampled 44.1 -> 16 kHz reappears at 4 kHz at full amplitude, which is
    /// exactly the kind of damage a WER harness sees but a parity test cannot.
    #[test]
    fn out_of_band_tone_does_not_alias_back() {
        let src = sine(44_100, 12_000.0, 44_100);
        let out = resample_mono(&src, 44_100, 16_000).unwrap();
        // 12 kHz mirrors to |12000 - 16000| = 4 kHz.
        let folded = amplitude_at(&out, 4_000.0, 16_000);
        assert!(folded < 0.02, "aliased image amplitude {folded}");
    }

    /// A stream chopped into ragged chunks must resample to the same waveform
    /// as the whole clip — that is what the carried filter state buys, and the
    /// failure mode without it (a click per chunk boundary) is inaudible in a
    /// length assertion.
    #[test]
    fn streaming_matches_one_shot() {
        let src = sine(20_000, 700.0, 44_100);
        let one_shot = resample_mono(&src, 44_100, 16_000).unwrap();

        let mut r = Resampler::new(44_100, 16_000).unwrap();
        let mut streamed = Vec::new();
        // Deliberately ragged, and not a divisor of CHUNK.
        let sizes = [1usize, 333, 2048, 17, 4096, 900];
        let mut off = 0;
        let mut i = 0;
        while off < src.len() {
            let n = sizes[i % sizes.len()].min(src.len() - off);
            r.push(&src[off..off + n], &mut streamed).unwrap();
            off += n;
            i += 1;
        }
        r.flush(&mut streamed).unwrap();

        assert_eq!(streamed.len(), one_shot.len());
        for (i, (a, b)) in streamed.iter().zip(one_shot.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "chunked != one-shot at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn identical_rates_are_a_passthrough() {
        let src = sine(1_000, 440.0, 16_000);
        let out = resample_mono(&src, 16_000, 16_000).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn empty_input_gives_empty_output() {
        let out = resample_mono(&[], 8_000, 16_000).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn implausible_rates_are_rejected() {
        assert!(validate_sample_rate(0).is_err());
        assert!(validate_sample_rate(100).is_err());
        assert!(validate_sample_rate(1_000_000).is_err());
        assert!(validate_sample_rate(8_000).is_ok());
        assert!(validate_sample_rate(48_000).is_ok());
        assert!(resample_mono(&[0.0; 16], 0, 16_000).is_err());
        assert!(resample_mono(&[0.0; 16], 16_000, 3).is_err());
    }
}
