"""Harmonic-plus-noise renderer for the synthetic corpus.

The goal is not to sound human. It is to produce a signal on which the *real* extractors
behave the way they would on speech:

* F0 stays inside the mandatory 100-600 Hz child band and follows the syllable's tone
  contour, so torchcrepe / pYIN / Praat have something to track and tone features have
  something to recover;
* each syllable has an amplitude envelope, so pauses and syllable durations are visible to
  a rhythm analysis;
* the spectral envelope depends on a per-turn "voice mode", so ECAPA embeddings actually
  spread when the generator injects cartoon-voice variability (B7);
* per-turn level is set from a target dB, so intensity following (A12) is measurable.

Everything is vectorised over samples; a 12-subject corpus renders in a couple of seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..contracts import SyllableSpan

__all__ = ["VoiceProfile", "TONE_LEVELS", "render_turn", "render_session"]

# Chao tone letters (1 = lowest, 5 = highest) for the Mandarin inventory.
TONE_LEVELS: dict[int, tuple[float, ...]] = {
    1: (5.0, 5.0),            # high level
    2: (3.0, 3.6, 5.0),       # rising
    3: (2.2, 1.0, 1.2, 3.8),  # low dipping
    4: (5.0, 3.0, 1.0),       # falling
    0: (3.0, 2.8),            # neutral, short and mid
}


@dataclass(frozen=True)
class VoiceProfile:
    """One speaking voice. Different `mode` values give ECAPA something to separate."""

    base_f0: float
    formants: tuple[float, float, float]
    bandwidths: tuple[float, float, float]
    gains: tuple[float, float, float]
    rolloff: float = 1.3
    n_harmonics: int = 16
    noise_level: float = 0.02
    jitter: float = 0.006
    shimmer: float = 0.03

    def shifted(self, semitones: float) -> "VoiceProfile":
        return VoiceProfile(
            base_f0=self.base_f0 * 2.0 ** (semitones / 12.0),
            formants=self.formants,
            bandwidths=self.bandwidths,
            gains=self.gains,
            rolloff=self.rolloff,
            n_harmonics=self.n_harmonics,
            noise_level=self.noise_level,
            jitter=self.jitter,
            shimmer=self.shimmer,
        )


def _level_curve(tone: int, n: int) -> np.ndarray:
    """Interpolate the Chao levels of one tone across `n` samples."""
    levels = np.asarray(TONE_LEVELS[tone], dtype=np.float64)
    if n <= 1:
        return np.full(max(n, 1), levels[-1])
    src = np.linspace(0.0, 1.0, len(levels))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, levels)


def _formant_gain(freqs: np.ndarray, voice: VoiceProfile) -> np.ndarray:
    """Spectral envelope evaluated at (time-varying) harmonic frequencies."""
    gain = np.full_like(freqs, 0.02)
    for f_c, bw, g in zip(voice.formants, voice.bandwidths, voice.gains):
        gain = gain + g * np.exp(-0.5 * ((freqs - f_c) / bw) ** 2)
    return gain


def render_turn(
    spans: Sequence[SyllableSpan],
    voice: VoiceProfile,
    sr: int,
    intensity_db: float,
    residual_log2: np.ndarray | float = 0.0,
    rng: np.random.Generator | None = None,
    f0_floor: float = 70.0,
    f0_ceil: float = 600.0,
) -> tuple[np.ndarray, float]:
    """Render one turn. Returns (waveform, t_start of the first syllable)."""
    rng = rng or np.random.default_rng(0)
    if not spans:
        return np.zeros(0, dtype=np.float32), 0.0

    t0 = spans[0].t_start
    total = int(round((spans[-1].t_end - t0) * sr)) + 1
    buf = np.zeros(total, dtype=np.float64)

    resid = (
        np.asarray(residual_log2, dtype=np.float64)
        if np.ndim(residual_log2)
        else np.full(len(spans), float(residual_log2))
    )
    if resid.shape[0] != len(spans):
        resid = np.resize(resid, len(spans))

    for i, span in enumerate(spans):
        n = int(round(span.duration * sr))
        if n < 8:
            continue
        offset = int(round((span.t_start - t0) * sr))
        if offset + n > total:
            n = total - offset
            if n < 8:
                continue

        level = _level_curve(span.tone, n)
        # Chao level 3 is the speaker's mid; two semitones per level step.
        f0 = voice.base_f0 * 2.0 ** ((level - 3.0) / 6.0 + resid[i])
        f0 *= 1.0 + voice.jitter * rng.standard_normal(n)
        f0 = np.clip(f0, f0_floor, f0_ceil)

        phase = 2.0 * np.pi * np.cumsum(f0) / sr
        h = np.arange(1, voice.n_harmonics + 1, dtype=np.float64)[:, None]
        harm_freq = h * f0[None, :]
        amp = (h ** (-voice.rolloff)) * _formant_gain(harm_freq, voice)
        amp *= harm_freq < (sr / 2.0 - 200.0)          # anti-alias
        sig = (amp * np.sin(h * phase[None, :])).sum(axis=0)

        # amplitude envelope: raised-cosine attack/decay, plus shimmer
        ramp = max(4, int(0.015 * sr))
        env = np.ones(n)
        r = min(ramp, n // 2)
        if r > 1:
            w = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, r)))
            env[:r] = w
            env[-r:] = w[::-1]
        env *= 1.0 + voice.shimmer * rng.standard_normal(n)

        seg = sig * env
        seg += voice.noise_level * rng.standard_normal(n) * env
        buf[offset : offset + n] += seg

    rms = float(np.sqrt(np.mean(buf ** 2))) if total else 0.0
    if rms > 1e-9:
        target = 10.0 ** (intensity_db / 20.0)
        buf *= target / rms
    return buf.astype(np.float32), t0


def render_session(
    rendered: Sequence[tuple[np.ndarray, float]],
    sr: int,
    duration: float,
    background_level: float = 0.0012,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix pre-rendered turns into one mono session waveform.

    A constant room-noise floor is added across the whole session. Without it there is no
    silence to measure against and the SNR gate (audit P1-5) has nothing to estimate.
    """
    total = int(round(duration * sr)) + sr  # one second of tail padding
    rng = rng or np.random.default_rng(12345)
    out = (background_level * rng.standard_normal(total)).astype(np.float32)
    for wave, t0 in rendered:
        if wave.size == 0:
            continue
        start = int(round(t0 * sr))
        end = min(start + wave.size, total)
        out[start:end] += wave[: end - start]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out
