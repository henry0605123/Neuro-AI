"""B stream, hand-computed. openSMILE is replaced by a stub so the expected values are exact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asdpipe.contracts import Speaker
from asdpipe.features.b_voice import (ALL_LLD_CHANNELS, LLD_CHANNELS, PERTURBATION_CHANNELS,
                                      BStreamFeatures, ecapa_windows,
                                      extract_b)
from asdpipe.features.common import (frame_energy, noise_floor_energy, pool, rms_db,
                                     slice_turn, snr_db)
from conftest import make_turn


def test_rms_db_hand_example():
    """A constant signal of amplitude 0.1 has RMS 0.1, so 20*log10(0.1) = -20 dB."""
    assert rms_db(np.full(1000, 0.1)) == pytest.approx(-20.0, abs=1e-6)


def test_rms_db_of_a_square_wave_hand_example():
    """+-0.5 alternating -> RMS 0.5 -> 20*log10(0.5) = -6.0206 dB."""
    x = np.tile([0.5, -0.5], 500)
    assert rms_db(x) == pytest.approx(-6.0206, abs=1e-3)


def test_snr_is_measured_against_the_session_floor_hand_example():
    """Session: 1 s of amplitude-0.01 noise plus 1 s of amplitude-0.1 tone.

    Noise-floor energy ~ 0.01^2 = 1e-4, turn energy ~ 0.1^2 = 1e-2,
    so the segmental SNR is 10*log10(1e-2 / 1e-4) = 20 dB.
    """
    quiet = np.full(16000, 0.01)
    loud = np.full(16000, 0.1)
    session = np.concatenate([quiet, loud])
    floor = noise_floor_energy(session)
    assert floor == pytest.approx(1e-4, rel=1e-6)
    assert snr_db(loud, floor) == pytest.approx(20.0, abs=1e-3)


def test_frame_energy_hand_example():
    x = np.concatenate([np.full(512, 0.2), np.full(512, 0.4)])
    e = frame_energy(x, 512)
    assert e.tolist() == pytest.approx([0.04, 0.16])


def test_pool_functionals_hand_example():
    """values 1..5 -> mean 3, population sd sqrt(2)=1.41421, median 3, p25 2, p75 4."""
    out = pool(np.array([1.0, 2, 3, 4, 5]), ["mean", "std", "median", "p25", "p75"], "x")
    assert out["x_mean"] == pytest.approx(3.0)
    assert out["x_std"] == pytest.approx(np.sqrt(2.0))
    assert out["x_median"] == pytest.approx(3.0)
    assert out["x_p25"] == pytest.approx(2.0)
    assert out["x_p75"] == pytest.approx(4.0)


def test_pool_ignores_nan_and_survives_an_empty_series():
    out = pool(np.array([1.0, np.nan, 3.0]), ["mean"], "x")
    assert out["x_mean"] == pytest.approx(2.0)
    # Nothing to pool is "cannot be measured", not 0 (health check P12).
    empty = pool(np.array([]), ["mean", "std"], "x")
    assert set(empty) == {"x_mean", "x_std"} and all(np.isnan(v) for v in empty.values())
    # Below the minimum amount of data: also NaN.
    assert np.isnan(pool(np.array([1.0, 2.0]), ["mean"], "x", min_n=5)["x_mean"])


def test_b7_embedding_dispersion_hand_example():
    """Three unit vectors: (1,0), (0,1), (-1,0).

    Pairwise cosines: 0, -1, 0  ->  distances 1, 2, 1  ->  mean 4/3 = 1.3333
    """
    f = BStreamFeatures(embeddings=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                                            dtype=np.float32))
    b7 = f.b7()
    assert b7["b7_mean_pairwise_cosdist"] == pytest.approx(4 / 3, abs=1e-6)
    assert b7["b7_n_windows"] == 3.0


def test_b7_is_zero_for_identical_windows():
    f = BStreamFeatures(embeddings=np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (5, 1)))
    assert f.b7()["b7_mean_pairwise_cosdist"] == pytest.approx(0.0, abs=1e-6)


def test_ecapa_windows_are_fixed_length_and_child_only(cfg):
    """Audit P0-F4: 3 s windows at a 1.5 s hop, computed over child speech only.

    6 s of child speech -> starts at 0.0, 1.5, 3.0 -> 3 windows of 48000 samples.
    """
    sr = 16000
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 4.0, "問"),
             make_turn(1, Speaker.CHILD, 4.0, 10.0, "答")]
    audio = np.zeros(sr * 12, dtype=np.float32)
    w = ecapa_windows(audio, sr, turns, cfg)
    assert w.shape == (3, 48000)


class _StubSmile:
    """Returns one frame per 10 ms with constant, known LLD values."""

    def __init__(self, values):
        self.values = values

    def process_signal(self, x, sr):
        n = max(1, int(len(x) / sr / 0.01))
        cols = {"F0semitoneFrom27.5Hz_sma3nz": np.full(n, 30.0)}
        for ch in ALL_LLD_CHANNELS:
            cols[ch] = np.full(n, self.values[ch])
        return pd.DataFrame(cols)


def test_extract_b_pools_the_stub_values_exactly(cfg):
    """With every frame constant, the pooled mean must equal that constant and the sd 0."""
    sr = 16000
    values = {ch: float(i + 1) for i, ch in enumerate(ALL_LLD_CHANNELS)}
    audio = np.concatenate([
        np.full(sr, 0.001, dtype=np.float32),                 # silence -> noise floor
        (0.1 * np.sin(2 * np.pi * 250 * np.arange(sr * 2) / sr)).astype(np.float32),
    ])
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "問"),
             make_turn(1, Speaker.CHILD, 1.0, 3.0, "答")]
    out = extract_b(audio, sr, turns, _StubSmile(values), cfg)
    pooled = out.pooled(["mean", "std"])
    assert pooled["b_Loudness_mean"] == pytest.approx(values["Loudness_sma3"])
    assert pooled["b_Loudness_std"] == pytest.approx(0.0)
    # jitter/shimmer/HNR are still measured, but audit P1-5 v3.0 keeps them out of the
    # core feature set, so they must reach `lld` without reaching `pooled`.
    assert out.lld["jitterLocal_sma3nz"].size > 0
    assert not any(k.startswith(("b_jitterLocal", "b_shimmer", "b_HNR", "b_cpp"))
                   for k in pooled)
    assert out.quality["b_n_child_turns"] == 1.0
    assert out.turn_db.size == 1
    # 0.1 amplitude sine -> RMS 0.1/sqrt(2) -> about -23.01 dB
    assert out.turn_db[0] == pytest.approx(-23.0103, abs=0.2)


def test_the_snr_gate_stops_perturbation_measures_but_not_spectral_ones(cfg):
    """Audit P1-5 gates *perturbation* measures. Spectral balance and intensity are neither.

    The stimulus has to be voiced as well as noisy: the gate is evaluated on the stable
    voiced portion, so a signal with no periodicity takes the "no voiced run" exit and never
    reaches the SNR test. A constant-amplitude tone gives Praat something to find while
    making the session noise floor equal the signal, so the SNR lands at roughly 0 dB.

    The second half of this test is the point of the 2026-09-12 change: A7 聲音清亮度 and
    A11 音量 are both rated 高, and gating them cost a median of 43% of each child's turns.
    """
    sr = 16000
    values = {ch: 1.0 for ch in ALL_LLD_CHANNELS}
    t = np.arange(sr * 3) / sr
    audio = (0.05 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)   # no silence anywhere
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "問"),
             make_turn(1, Speaker.CHILD, 1.0, 3.0, "答")]
    out = extract_b(audio, sr, turns, _StubSmile(values), cfg)
    assert out.quality["b_n_below_snr"] == 1.0
    assert np.isnan(out.cpp_per_turn[0])
    for ch in PERTURBATION_CHANNELS:
        assert out.lld[ch].size == 0, ch
    for ch in LLD_CHANNELS:
        assert out.lld[ch].size > 0, ch
    assert out.quality["b_core_lld_coverage"] == 1.0


def test_extract_b_does_not_score_silence_as_a_clean_voice(cfg):
    """A turn with no periodicity must be recorded as unmeasurable, not measured anyway."""
    sr = 16000
    values = {ch: 1.0 for ch in ALL_LLD_CHANNELS}
    audio = np.full(sr * 3, 0.05, dtype=np.float32)   # DC: energy, but nothing voiced
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "問"),
             make_turn(1, Speaker.CHILD, 1.0, 3.0, "答")]
    out = extract_b(audio, sr, turns, _StubSmile(values), cfg)
    assert out.quality["b_n_no_voiced_run"] == 1.0
    assert np.isnan(out.cpp_per_turn[0])


def test_extract_b_measures_cpp_on_the_voiced_part_of_a_noisy_turn(cfg):
    """The point of the change: speech surrounded by noise still yields a CPP value.

    Under the old whole-turn gate this turn scored below the floor because the silence
    either side dragged its SNR down, and CPP came back NaN. On the real cohort that
    mechanism accounted for the missingness exactly (r = 1.00 with the gate).
    """
    sr = 16000
    values = {ch: 1.0 for ch in ALL_LLD_CHANNELS}
    rng = np.random.default_rng(0)
    audio = (0.002 * rng.standard_normal(sr * 4)).astype(np.float32)
    t = np.arange(int(0.6 * sr)) / sr
    speech = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    audio[int(1.2 * sr): int(1.2 * sr) + len(speech)] += speech.astype(np.float32)
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "問"),
             make_turn(1, Speaker.CHILD, 1.0, 3.0, "答")]
    out = extract_b(audio, sr, turns, _StubSmile(values), cfg)
    assert out.quality["b_voiced_seconds"] > 0.1
    assert np.isfinite(out.cpp_per_turn[0])
