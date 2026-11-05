"""Transcript-free recording audit (eval/audio_audit.py).

The audit's whole job is to say whether two cohorts were recorded under comparable
conditions. A detector that quietly mis-locates speech, or a report that silently drops a
broken file, would answer that question wrongly while looking fine -- so the tests drive it
with signals whose answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.eval.audio_audit import (RecordingAudit, audit_recording, speech_segments,
                                      write_audio_audit_report)
from asdpipe.features.common import noise_floor_energy

SR = 16000


def _room(seconds: float, noise: float = 0.002, seed: int = 0) -> np.ndarray:
    return (noise * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def _put(audio: np.ndarray, at_s: float, dur_s: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(dur_s * SR)) / SR
    tone = amp * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    a = int(at_s * SR)
    audio[a:a + len(tone)] += tone.astype(np.float32)
    return audio


def test_speech_segments_land_where_the_speech_actually_is():
    """Boundaries are reported in samples, so an index/frame mix-up shifts everything.

    Written after exactly that: the detector assumed a 25 ms analysis frame while
    `frame_energy` uses 512 samples, which put every segment 28% early -- a plausible-looking
    result that was wrong by a constant factor.
    """
    audio = _room(20.0)
    for at in (2.0, 8.0, 14.0):
        audio = _put(audio, at, 1.5)
    segs = speech_segments(audio, SR, noise_floor_energy(audio))
    assert len(segs) == 3
    for (a, b), at in zip(segs, (2.0, 8.0, 14.0)):
        assert a / SR == pytest.approx(at, abs=0.1)
        assert (b - a) / SR == pytest.approx(1.5, abs=0.2)


def test_speech_ratio_matches_the_speech_that_was_inserted():
    audio = _room(20.0)
    for at in (2.0, 8.0, 14.0):
        audio = _put(audio, at, 1.5)
    segs = speech_segments(audio, SR, noise_floor_energy(audio))
    ratio = sum(b - a for a, b in segs) / len(audio)
    assert ratio == pytest.approx(4.5 / 20.0, abs=0.03)


def test_a_silent_recording_is_reported_as_an_error_not_as_zero_speech(cfg, tmp_path):
    """Silence must not pass as a valid recording with speech_ratio 0.0.

    A cohort table full of zeros reads as a finding about the children; it is a finding
    about the files.
    """
    import soundfile as sf

    p = tmp_path / "silent.wav"
    sf.write(p, _room(5.0, noise=0.001), SR)
    out = audit_recording("case", "ASD", p, cfg, backends=None, cache_dir=None)
    assert not out.ok
    assert any("no speech" in e for e in out.errors)


def test_a_file_that_will_not_decode_is_a_finding(cfg, tmp_path):
    p = tmp_path / "broken.wav"
    p.write_bytes(b"not audio at all")
    out = audit_recording("case", "ASD", p, cfg, backends=None, cache_dir=None)
    assert not out.ok
    assert any("decode failed" in e or "decoded to" in e for e in out.errors)


def test_the_snr_gate_sees_a_noisy_recording_as_noisy(cfg, tmp_path):
    """The gate is the mechanism that removed CPP, so the audit has to reproduce it."""
    import soundfile as sf

    clean, noisy = _room(30.0, noise=0.0005), _room(30.0, noise=0.02, seed=1)
    for at in np.arange(1.0, 29.0, 3.0):
        clean, noisy = _put(clean, at, 1.0), _put(noisy, at, 1.0)
    res = {}
    for name, audio in (("clean", clean), ("noisy", noisy)):
        p = tmp_path / f"{name}.wav"
        sf.write(p, audio, SR)
        res[name] = audit_recording(name, "X", p, cfg, backends=None, cache_dir=None)
    assert res["clean"].ok and res["noisy"].ok
    assert res["clean"].metrics["snr_median_db"] > res["noisy"].metrics["snr_median_db"]
    assert (res["clean"].metrics["below_snr_floor_ratio"]
            <= res["noisy"].metrics["below_snr_floor_ratio"])


def test_the_report_de_identifies_cases_and_keeps_broken_files_visible(cfg, tmp_path):
    """Intake folders are named after the children, and this report gets forwarded."""
    audits = [
        # Placeholder identities, deliberately fake (year 2099, record 0000000x). This test
        # once used two real children's folder names -- the very thing it guards against --
        # and was caught before it was committed.
        RecordingAudit("20990101-00000001王小明", "TD",
                       metrics={"duration_min": 38.0, "snr_median_db": 12.0}),
        RecordingAudit("20990101-00000002李小華", "ASD", errors=["no media file"]),
    ]
    p = write_audio_audit_report(tmp_path / "r.md", audits, cfg)
    text = p.read_text(encoding="utf-8")
    assert "王小明" not in text and "李小華" not in text
    assert "00000001" not in text and "00000002" not in text
    assert "TD01" in text and "no media file" in text
    # the framing must survive edits: this report is a confound check, not a result
    assert "混淆檢查" in text
