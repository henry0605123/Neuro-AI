"""The checks that run before any model is fitted (research plan 3.4): split-half reliability, the
three-tracker F0 check, the F0 group rule and the convention gate for the segment measures.
Synthetic data and hand-built sessions only; nothing here touches the real recordings.

The first five tests are the task brief's. The block after them pins what the brief leaves to the
controller's rulings for this task, so that a later edit cannot change them unnoticed:

* a session passes the F0 check only if both agreements are strictly above 0.90 and both
  octave-error rates strictly below 0.03 (the plan's "> 90%" and "< 3%");
* the ratio windows that count as agreement and as an octave error are the ones of
  ``Backends.f0_cross_check``, and the pYIN pair reproduces that method's numbers;
* voicing: CREPE is voiced from periodicity 0.4, and pYIN and Praat wherever they report an F0
  (pYIN's own voiced flag marks exactly those frames, so no test can tell the two apart); both
  are interpolated onto CREPE's frame times; a pair is compared on the frames both call voiced,
  with fewer than 50 of them its values are NaN, and ``n_frames`` is the weaker pair's count;
* split-half: a speaker's turns in time order, the odd-numbered ones against the even-numbered
  ones, the examiner's turns split the same way, B7's windows cut inside each half, and each half
  measured under the floors of the real measure (15 windows, 5 turns of at least 0.1 s).

Where a test needs a tracker's answer to be known, ``FakeCrepe`` replaces ``track_f0`` alone and
pYIN and Praat run for real on a harmonic tone, which both follow to a fraction of a Hertz (pYIN
reports 220.64 Hz, the bin nearest 220, and Praat 220.00 Hz). A steady tone cannot show a wrong
clock or a wrong voicing rule, though, so one test runs the real trackers on a gliding voice with
a stretch of silence, with CREPE standing in as the exact F0, and one checks that Praat is
sampled once a hop. The tests that need a tracker's clock to be known replace ``_pyin_track`` and
``_praat_track`` instead, so they pin the interpolation but not those two helpers.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from asdpipe.config import load_decision_rules
from asdpipe.contracts import Label, Session, Speaker, SubjectRecord
from asdpipe.eval.convention_gate import TRANSFORMS
from asdpipe.features.backends import Backends
from asdpipe.features.common import slice_turn
from asdpipe.mil import checks
from asdpipe.mil.checks import (SPLIT_HALF_MEASURES, _child_means, _f0_at, _ratio_rates,
                                b7_a11_halves, f0_group_rule, f0_session_pass, f0_three_way,
                                segment_gate, split_half_r)
from asdpipe.mil.segments import CHILD_MEASURES, SEGMENT_MEASURES
from asdpipe.synth.generator import SynthConfig, generate_corpus, generate_subject
from conftest import hand_cfg, make_turn

SR = 16000
CHILD, EXAMINER = Speaker.CHILD, Speaker.EXAMINER
F0_KEYS = ("agree_pyin", "agree_praat", "octave_pyin", "octave_praat")


# ---------------------------------------------------------------- fixtures and stand-ins
@pytest.fixture(scope="module")
def rules():
    return load_decision_rules()


@pytest.fixture(scope="module")
def synthetic_subject():
    """30 child turns, about 85 s of child speech: far more than 15 windows of 3 s per half."""
    return generate_subject(0, Label.CONTROL, SynthConfig(seed=11))


def _punctuated(text: str, mark: str) -> str:
    mid = len(text) // 2
    return text[:mid] + "，" + text[mid:] + mark


@pytest.fixture(scope="module")
def synthetic_corpus():
    """Six synthetic children whose transcripts have their punctuation put back, the way a raw
    TD transcript has it: a comma in the middle of every turn and a full stop or question mark
    at its end. Stripping it is the convention under test."""
    raw = generate_corpus(SynthConfig(n_subjects=6, turns_per_subject=40, seed=13))
    subjects = []
    for s in raw.subjects:
        turns = tuple(dataclasses.replace(t, text=_punctuated(t.text, "？" if t.speaker is EXAMINER
                                                                else "。"))
                      for t in s.session.turns)
        subjects.append(dataclasses.replace(s, session=dataclasses.replace(s.session, turns=turns)))
    return dataclasses.replace(raw, subjects=tuple(subjects))


class SeededEmbedder:
    """Stands in for ``Backends.embed_windows``: seeded random 192-d vectors."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def embed_windows(self, windows):
        return self._rng.standard_normal((len(windows), 192)).astype(np.float32)


@pytest.fixture
def fake_backends():
    return SeededEmbedder()


class RecordingEmbedder:
    """``embed_windows`` that keeps what it was given and returns orthonormal vectors, so any two
    windows are a cosine distance of exactly 1 apart."""

    def __init__(self):
        self.calls: list[np.ndarray] = []

    def embed_windows(self, windows):
        self.calls.append(np.array(windows))
        return np.eye(192, dtype=np.float32)[: len(windows)]


class FakeCrepe:
    """Stands in for ``Backends.track_f0``: the test says what each frame's F0 and periodicity are
    (functions of the frame index), and every audio array it was handed is kept."""

    def __init__(self, f0, periodicity=lambda i: np.ones(len(i))):
        self._f0, self._periodicity = f0, periodicity
        self.heard: list[np.ndarray] = []

    def track_f0(self, audio, sr, fmin, fmax, hop_s):
        self.heard.append(np.array(audio))
        i = np.arange(1 + len(audio) // int(round(hop_s * sr)))
        return np.asarray(self._f0(i), dtype=float), np.asarray(self._periodicity(i), dtype=float)


@pytest.fixture(scope="module")
def real_backends(cfg):
    """The frozen toolchain on the light "smoke" profile, as cli.main._backends builds it, but
    with no cache directory so that nothing is written."""
    return Backends.from_config(cfg)


def _harmonic_tone(f0_hz: float, seconds: float) -> np.ndarray:
    """Five harmonics at 1/k amplitude, 0.3 peak: a clean voiced sound every tracker follows."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0_hz * k * t) / k for k in range(1, 6))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def _gliding_voice(seconds: float, silence: tuple[float, float]):
    """Five harmonics over an F0 that swings between 200 and 500 Hz, 1.5 times a second (about
    1,400 Hz a second at its fastest), with digital silence from ``silence[0]`` to ``silence[1]``.
    Returns the audio and the exact F0 as a function of time."""
    def f0_at(t):
        return 350.0 + 150.0 * np.sin(2 * np.pi * 1.5 * t)

    t = np.arange(int(seconds * SR)) / SR
    # the integral of 2 pi f0_at(t), so that harmonic k sits at k * f0_at(t)
    phase = 2 * np.pi * (350.0 * t - 150.0 / (2 * np.pi * 1.5) * np.cos(2 * np.pi * 1.5 * t))
    x = sum(np.sin(k * phase) / k for k in range(1, 6))
    x = (0.3 * x / np.abs(x).max()).astype(np.float32)
    x[int(silence[0] * SR): int(silence[1] * SR)] = 0.0
    return x, f0_at


def _three_way(cfg, crepe, seconds=5.0, **kw):
    """f0_three_way over a 220 Hz tone that is one child turn."""
    return f0_three_way(_harmonic_tone(220.0, seconds), SR,
                        [make_turn(0, CHILD, 0.0, seconds, "啊")], cfg, crepe, **kw)


# ---------------------------------------------------------------- the brief's five tests
def test_split_half_r():
    assert split_half_r(np.arange(10.), np.arange(10.) * 2) == pytest.approx(1.0)
    assert np.isnan(split_half_r(np.array([1., np.nan, 2, 3]), np.array([1., 2, np.nan, 3])))


def test_b7_a11_halves_split_turns_alternately(cfg, synthetic_subject, fake_backends):
    h = b7_a11_halves(synthetic_subject, cfg, fake_backends)
    assert all(np.isfinite(v).all() for v in map(np.array, h.values()))


@pytest.mark.slow
def test_f0_three_way_agrees_on_a_harmonic_tone(cfg, real_backends):   # 220 Hz, 5 s, 16 kHz
    tone = _harmonic_tone(220.0, 5.0)
    one_child_turn = make_turn(0, CHILD, 0.0, 5.0, "啊")
    c = f0_three_way(tone, 16000, [one_child_turn], cfg, real_backends)
    assert f0_session_pass(c)


def test_f0_group_rule():
    assert f0_group_rule(np.full(10, .1), np.r_[np.ones(5), np.zeros(5)]) == (0.5, True)
    assert f0_group_rule(np.r_[np.ones(5), np.zeros(5)], np.r_[np.ones(5), np.zeros(5)]) == (1.0, False)


def test_segment_gate_strip_is_a_no_op_after_normalization(cfg, rules, synthetic_corpus):
    res = segment_gate(synthetic_corpus, cfg, rules, words_of=lambda ts: [list(t) for t in ts])
    assert all(r.shifts.get("strip_punctuation", 0.0) == pytest.approx(0.0) for r in res)


# ---------------------------------------------------------------- split-half reliability
def test_split_half_r_is_the_uncorrected_pearson_over_children_with_both_halves():
    odd, even = np.array([1., 2, 3, 4, 5, 6]), np.array([2., 1, 4, 3, 6, 5])
    assert split_half_r(odd, even) == pytest.approx(29 / 35)             # 14.5 / 17.5, by hand
    assert split_half_r(odd, -even) == pytest.approx(-29 / 35)
    # a child with only one half is left out of both; the other six are what is correlated
    assert split_half_r(np.r_[odd, np.nan, 9.], np.r_[even, 7., np.nan]) == pytest.approx(29 / 35)
    # five children are enough, four are not
    five = split_half_r(np.array([1., 2, 3, 4, 5]), np.array([2., 1, 4, 3, 5]))
    assert five == pytest.approx(0.8)
    assert np.isnan(split_half_r(odd[:4], even[:4]))
    # a half with no spread has no correlation: NaN, not 0 and not an error
    assert np.isnan(split_half_r(odd, np.full(6, 3.0)))
    # one half per child, in the same order: two lengths mean two different sets of children
    with pytest.raises(ValueError):
        split_half_r(odd, even[:5])


def _levelled_subject(levels, turn_s=4.0, reverse=False):
    """Back-to-back turns, each a constant level, so that a window or a level says exactly which
    turns it came from. ``levels`` is (speaker, level) or (speaker, level, seconds) in time order,
    a turn lasting ``turn_s`` unless it says otherwise; ``reverse`` hands the turns over latest
    first."""
    turns, chunks, t = [], [], 0.0
    for i, (who, level, *seconds) in enumerate(levels):
        length = seconds[0] if seconds else turn_s
        chunks.append(np.full(int(round(length * SR)), level, dtype=np.float32))
        turns.append(make_turn(i, who, t, t + length, "x"))
        t += length
    if reverse:
        turns.reverse()
    return SubjectRecord("SUBJ", Label.CONTROL, Session("S", "SUBJ", tuple(turns), SR),
                         np.concatenate(chunks), SR)


@pytest.mark.parametrize("reverse", [False, True])
def test_halves_are_alternate_turns_of_each_speaker_in_time_order(cfg, reverse):
    c = hand_cfg(cfg)
    c["features"]["b_stream"]["ecapa_min_windows"] = 2
    # Examiner and child take turns, six each, and every level differs from the rest, so each way
    # of going wrong (the first against the second half of the session, the examiner's turns pooled
    # instead of split, the turns not put in time order) moves at least one of the medians below
    examiner = [0.06, 0.02, 0.04, 0.03, 0.08, 0.10]
    child = [0.10, 0.01, 0.30, 0.02, 0.20, 0.03]
    levels = [(who, v) for e, k in zip(examiner, child) for who, v in ((EXAMINER, e), (CHILD, k))]
    emb = RecordingEmbedder()
    h = b7_a11_halves(_levelled_subject(levels, reverse=reverse), c, emb)
    assert set(h) == set(SPLIT_HALF_MEASURES)

    # B7: each half is 3 child turns of 4 s, so it makes (12 - 3) / 1.5 + 1 = 7 windows of 3 s,
    # and every one of them holds only that half's child levels, never both halves'
    odd_levels = {np.float32(v) for v in child[0::2]}
    even_levels = {np.float32(v) for v in child[1::2]}
    held = [set(np.unique(w)) for w in np.concatenate(emb.calls)]
    assert len(held) == 14
    assert sum(v <= odd_levels for v in held) == 7
    assert sum(v <= even_levels for v in held) == 7
    assert h["b7_mean_pairwise_cosdist"] == pytest.approx((1.0, 1.0))   # orthonormal vectors
    # A11, by hand: the half's child median less the same half's examiner median, in dB.
    # odd: child 0.10, 0.30, 0.20 -> 0.20; examiner 0.06, 0.04, 0.08 -> 0.06
    # even: child 0.01, 0.02, 0.03 -> 0.02; examiner 0.02, 0.03, 0.10 -> 0.03
    assert h["b_level_rel_examiner_db"] == pytest.approx(
        (20 * math.log10(0.20 / 0.06), 20 * math.log10(0.02 / 0.03)), abs=1e-3)


def test_a_half_below_the_floors_of_the_real_measure_is_missing_as_the_measure_itself_is(cfg):
    # Six child turns of 4 s: each half holds 3 turns and makes 7 windows. A11 needs
    # features.min_data.values_per_statistic child turns (5) and B7 b_stream.ecapa_min_windows
    # windows (15), the floors the real measures apply to a whole session.
    subject = _levelled_subject([(who, 0.1) for _ in range(6) for who in (EXAMINER, CHILD)])
    no_turn_floor = hand_cfg(cfg)                                       # A11 asks for 1 turn
    h = b7_a11_halves(subject, no_turn_floor, RecordingEmbedder())
    assert np.isnan(h["b7_mean_pairwise_cosdist"]).all()                # 7 windows < 15
    assert np.isfinite(h["b_level_rel_examiner_db"]).all()
    no_window_floor = hand_cfg(cfg)
    no_window_floor["features"]["min_data"]["values_per_statistic"] = 5
    no_window_floor["features"]["b_stream"]["ecapa_min_windows"] = 2
    h = b7_a11_halves(subject, no_window_floor, RecordingEmbedder())
    assert np.isnan(h["b_level_rel_examiner_db"]).all()                 # 3 turns < 5
    assert np.isfinite(h["b7_mean_pairwise_cosdist"]).all()


def test_a11_ignores_a_child_turn_shorter_than_a_tenth_of_a_second(cfg):
    c = hand_cfg(cfg)
    c["features"]["b_stream"]["ecapa_min_windows"] = 2
    levels = [(EXAMINER, 0.05), (CHILD, 0.10),
              (EXAMINER, 0.05), (CHILD, 0.90, 0.05),            # a loud 50 ms click
              (EXAMINER, 0.05), (CHILD, 0.10)]
    odd, even = b7_a11_halves(_levelled_subject(levels), c, RecordingEmbedder())[
        "b_level_rel_examiner_db"]
    assert odd == pytest.approx(20 * math.log10(0.10 / 0.05), abs=1e-3)
    assert np.isnan(even)             # the half's only child turn is too short to have a level


def test_the_split_half_measures_are_two_of_the_child_measures():
    assert set(SPLIT_HALF_MEASURES) <= set(CHILD_MEASURES)


# ---------------------------------------------------------------- the F0 check
def _check(**kw):
    return {"agree_pyin": 0.95, "agree_praat": 0.95, "octave_pyin": 0.0, "octave_praat": 0.0,
            "n_frames": 500, **kw}


def test_f0_session_pass_is_strict_on_both_pairs():
    assert f0_session_pass(_check())
    for key in ("agree_pyin", "agree_praat"):
        assert not f0_session_pass(_check(**{key: 0.90}))         # "> 90%": 90% itself fails
        assert f0_session_pass(_check(**{key: 0.9001}))
    for key in ("octave_pyin", "octave_praat"):
        assert not f0_session_pass(_check(**{key: 0.03}))         # "< 3%": 3% itself fails
        assert f0_session_pass(_check(**{key: 0.0299}))
    # a pair that could not be compared never passes
    for key in F0_KEYS:
        assert not f0_session_pass(_check(**{key: float("nan")}))
    # the thresholds are the caller's to give (decision_rules.yaml: mil_analysis.f0_session)
    assert f0_session_pass(_check(agree_pyin=0.85), min_agreement=0.80)
    assert not f0_session_pass(_check(octave_praat=0.02), max_octave=0.02)


def test_agreement_and_octave_windows_include_their_edges():
    # Backends.f0_cross_check: agreement 0.95 to 1.05, octave 1.9 to 2.1 or 0.45 to 0.55
    other = np.full(12, 100.0)
    crepe = np.array([95.0, 105.0, 94.99, 105.01,         # two in, two just out
                      190.0, 210.0, 189.99, 210.01,       # an octave up: two in, two out
                      45.0, 55.0, 44.99, 55.01])          # an octave down: two in, two out
    agree, octave = _ratio_rates(crepe / other)
    assert agree == pytest.approx(2 / 12)
    assert octave == pytest.approx(4 / 12)


def test_f0_at_interpolates_between_voiced_neighbours_only():
    times = np.array([0.00, 0.01, 0.02, 0.03, 0.04])
    f0 = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    voiced = np.array([True, True, False, True, True])
    target = np.array([-0.001, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045])
    got = _f0_at(times, f0, voiced, target)
    nan = np.nan
    #                     before, 0, mid, 0.01, next to an unvoiced frame x3, 0.03, mid, 0.04, after
    expected = np.array([nan, 100.0, 105.0, 110.0, nan, nan, nan, 130.0, 135.0, 140.0, nan])
    assert np.allclose(got, expected, equal_nan=True)
    # the same frame times to the last bit are not needed: a tracker's clock may carry rounding
    # error, and a frame it calls voiced must not turn unvoiced for 1e-13 s
    jittered = _f0_at(times + 1e-13, f0, voiced, target)
    assert np.allclose(jittered, expected, equal_nan=True)
    # nothing to interpolate from
    assert np.isnan(_f0_at(np.zeros(0), np.zeros(0), np.zeros(0, bool), target)).all()


def test_f0_three_way_uses_the_windows_of_the_two_tracker_check(cfg):
    # CREPE's answer cycles over six frames: right, 4% high, 8% high, an octave up, an octave
    # down, and 54% of the tone. pYIN (220.64 Hz) and Praat (220.00 Hz) follow the tone, so 2
    # frames in 6 agree (CREPE / other within 5%), 3 in 6 are octave errors, and the frame 8% high
    # is neither. At 54% the ratio CREPE / other is 0.54, an octave error, while other / CREPE is
    # 1.85, which is not: the ratio is CREPE over the other tracker, as in f0_cross_check.
    cycle = 220.0 * np.array([1.0, 1.04, 1.08, 2.0, 0.5, 0.54])
    c = _three_way(cfg, FakeCrepe(lambda i: cycle[i % 6]))
    assert list(c) == [*F0_KEYS, "n_frames"]
    for pair in ("pyin", "praat"):
        assert c[f"agree_{pair}"] == pytest.approx(2 / 6, abs=0.01)
        assert c[f"octave_{pair}"] == pytest.approx(3 / 6, abs=0.01)
    assert 490 <= c["n_frames"] <= 501
    assert not f0_session_pass(c)


def test_f0_three_way_puts_pyin_and_praat_on_crepes_clock(cfg, monkeypatch):
    # Both trackers follow one ramp, F0 = 150 + 100 t Hz, but each sits on a clock of its own:
    # pYIN's frames start at 0.50 s, 10 ms apart; Praat's start at 1.00 s, 12.3 ms apart. A ramp
    # interpolated onto CREPE's clock is still the ramp, so every comparable frame agrees, and a
    # frame outside a tracker's span is not compared. Matching frames by index instead would put
    # the 150 + 100 t of one beside a value up to half a second ahead.
    def ramp(t):
        return 150.0 + 100.0 * t

    def on_own_clock(first, step, n):
        def track(*_):
            times = first + step * np.arange(n)
            return times, ramp(times), np.ones(n, dtype=bool)
        return track

    monkeypatch.setattr(checks, "_pyin_track", on_own_clock(0.5, 0.01, 400))      # 0.50 to 4.49 s
    monkeypatch.setattr(checks, "_praat_track", on_own_clock(1.0, 0.0123, 250))   # 1.00 to 4.06 s
    crepe = FakeCrepe(lambda i: ramp(0.01 * i))
    c = _three_way(cfg, crepe)
    assert c["agree_pyin"] == pytest.approx(1.0) and c["agree_praat"] == pytest.approx(1.0)
    assert c["octave_pyin"] == 0.0 and c["octave_praat"] == 0.0
    # CREPE's frames fall every 10 ms from 0: 400 of them (50 to 449) inside pYIN's span, 307
    # (100 to 406) inside Praat's, and n_frames is the weaker pair's count
    assert c["n_frames"] == 307

    # each pair has its floor to itself: Praat's span (1.00 to 1.36 s) holds 36 of CREPE's frames
    monkeypatch.setattr(checks, "_praat_track", on_own_clock(1.0, 0.0123, 30))
    c = _three_way(cfg, crepe)
    assert c["agree_pyin"] == pytest.approx(1.0)
    assert np.isnan(c["agree_praat"]) and np.isnan(c["octave_praat"])
    assert c["n_frames"] == 36 and not f0_session_pass(c)
    # and the other way round: pYIN's span (0.50 to 0.85 s) holds 36 and Praat's is long
    monkeypatch.setattr(checks, "_pyin_track", on_own_clock(0.5, 0.01, 36))
    monkeypatch.setattr(checks, "_praat_track", on_own_clock(1.0, 0.0123, 250))
    c = _three_way(cfg, crepe)
    assert np.isnan(c["agree_pyin"]) and np.isnan(c["octave_pyin"])
    assert c["agree_praat"] == pytest.approx(1.0)
    assert c["n_frames"] == 36 and not f0_session_pass(c)


def test_f0_three_way_real_trackers_follow_a_glide_and_skip_its_silence(cfg):
    # A steady tone cannot show a wrong clock (a constant pitch is the same at any time) or a wrong
    # voicing rule for Praat (it is voiced on every frame of a steady tone). So the real pYIN and
    # Praat get a voice whose F0 swings between 200 and 500 Hz and which is silent from 2.5 to
    # 3.5 s, and CREPE is replaced by the exact F0 with periodicity 1 on every frame, the silent
    # ones too: only the trackers' own voicing can keep the silence out of the comparison, and only
    # their own clocks, interpolated onto CREPE's, put their values beside the truth.
    # Measured on the real code: agreement 0.996 (pYIN) and 1.0 (Praat), no octave errors. With
    # Praat called voiced on every frame its agreement falls to 0.83; with its frames put at
    # k * 10 ms instead of on pitch.xs(), to 0.57; with pYIN's clock 15 ms out, to 0.61.
    audio, f0_at = _gliding_voice(6.0, silence=(2.5, 3.5))
    c = f0_three_way(audio, SR, [make_turn(0, CHILD, 0.0, 6.0, "啊")], cfg,
                     FakeCrepe(lambda i: f0_at(0.01 * i)))
    assert c["agree_pyin"] > 0.95 and c["agree_praat"] > 0.95
    assert c["octave_pyin"] < 0.03 and c["octave_praat"] < 0.03
    assert c["n_frames"] < 520          # CREPE has 601 frames, 100 of them in the silence


def test_praat_track_puts_its_frames_one_hop_apart(cfg):
    # Interpolation onto CREPE's clock hides how often Praat is sampled (a glide survives 20 ms
    # frames), so the step the brief asks for, time_step = the hop, is checked on the helper itself
    c_cfg = cfg["features"]["c_stream"]
    times, *_ = checks._praat_track(_harmonic_tone(220.0, 2.0), SR, c_cfg["f0_min_hz"],
                                    c_cfg["f0_max_hz"], c_cfg["f0_hop_s"])
    assert np.allclose(np.diff(times), c_cfg["f0_hop_s"])
    assert len(times) > 190              # 2 s at 10 ms, less the frames Praat cannot centre


def test_f0_three_way_voicing_from_periodicity_0_4_and_a_floor_of_50_frames(cfg):
    def run(periodicity):
        return _three_way(cfg, FakeCrepe(lambda i: np.full(len(i), 220.0), periodicity))

    # CREPE is voiced from periodicity 0.4 itself, not above it
    assert run(lambda i: np.full(len(i), 0.40))["n_frames"] >= 490
    unvoiced = run(lambda i: np.full(len(i), 0.39))
    assert unvoiced["n_frames"] == 0
    assert all(np.isnan(unvoiced[k]) for k in F0_KEYS) and not f0_session_pass(unvoiced)

    # a pair is compared from 50 frames both trackers call voiced; one fewer and it is NaN
    def only(k):         # periodicity high on k frames in the middle of the tone only
        return run(lambda i: np.where((i >= 100) & (i < 100 + k), 1.0, 0.0))

    enough, short = only(50), only(49)
    assert enough["n_frames"] == 50
    assert enough["agree_pyin"] == pytest.approx(1.0)
    assert enough["agree_praat"] == pytest.approx(1.0)
    assert short["n_frames"] == 49
    assert all(np.isnan(short[k]) for k in F0_KEYS) and not f0_session_pass(short)


def test_f0_three_way_hears_the_first_seconds_of_the_childs_own_speech_in_time_order(cfg):
    # child 0-2 s (200 Hz), examiner 2-4 s (300 Hz, never handed over), child 4-6 s (150 Hz);
    # the child's turns are given latest first and only the first 3 s of their speech is used
    audio = np.concatenate([_harmonic_tone(200.0, 2.0), _harmonic_tone(300.0, 2.0),
                            _harmonic_tone(150.0, 2.0)])
    kids = [make_turn(1, CHILD, 4.0, 6.0, "後"), make_turn(0, CHILD, 0.0, 2.0, "前")]
    crepe = FakeCrepe(lambda i: np.full(len(i), 200.0))
    f0_three_way(audio, SR, kids, cfg, crepe, max_seconds=3.0)
    (heard,) = crepe.heard
    assert np.array_equal(heard, np.concatenate([audio[:2 * SR], audio[4 * SR:5 * SR]]))


def test_f0_three_way_gives_nan_for_less_than_a_second_of_child_speech(cfg):
    crepe = FakeCrepe(lambda i: np.full(len(i), 220.0))
    tone = _harmonic_tone(220.0, 5.0)
    for kids in ([make_turn(0, CHILD, 0.0, 0.99, "啊")], []):
        c = f0_three_way(tone, SR, kids, cfg, crepe)
        assert c["n_frames"] == 0 and all(np.isnan(c[k]) for k in F0_KEYS)
        assert not f0_session_pass(c)
    assert crepe.heard == []                                  # not even tracked
    # a whole second is tracked
    c = f0_three_way(tone, SR, [make_turn(0, CHILD, 0.0, 1.0, "啊")], cfg, crepe)
    assert len(crepe.heard) == 1 and c["n_frames"] >= 90 and f0_session_pass(c)


@pytest.mark.slow
def test_three_way_pyin_pair_reproduces_the_two_tracker_check(cfg, real_backends):
    # 85 s of synthetic child speech, so the 60 s cut matters, and CREPE and pYIN disagree on a
    # few frames, so there is something to reproduce
    s = generate_subject(0, Label.CONTROL, SynthConfig(seed=11))
    kids = sorted(s.session.by_speaker(CHILD), key=lambda t: t.t_start)
    c_cfg = cfg["features"]["c_stream"]
    kid_audio = np.concatenate([slice_turn(s.audio, SR, t) for t in kids])
    old = real_backends.f0_cross_check(kid_audio, SR, c_cfg["f0_min_hz"], c_cfg["f0_max_hz"],
                                       c_cfg["f0_hop_s"], max_seconds=60.0)
    new = f0_three_way(s.audio, SR, kids, cfg, real_backends, max_seconds=60.0)
    assert 0.0 < new["agree_pyin"] < 1.0 and 0.0 < new["octave_pyin"] < 1.0
    assert new["agree_pyin"] == pytest.approx(old["agreement"])
    assert new["octave_pyin"] == pytest.approx(old["octave_error"])
    assert 0 < new["n_frames"] <= old["n_frames"]             # the weaker of the two pairs


# ---------------------------------------------------------------- the F0 group rule
def test_f0_group_rule_keeps_its_edges_and_cannot_keep_what_it_cannot_judge():
    # ten TD children scoring 0..9 and ten ASD children, listed with what each beats:
    # 100 beats all ten TD children, -100 none, 0.5 only the child at 0, 8.5 the nine below it
    td = np.arange(10.0)
    y = np.r_[np.ones(10), np.zeros(10)]
    for asd, auc, keep in (([100.] * 7 + [-100.] * 3, 0.70, True),        # on the edge: kept
                           ([100.] * 3 + [-100.] * 7, 0.30, True),
                           ([100.] * 7 + [-100.] * 2 + [0.5], 0.71, False),
                           ([100.] * 2 + [-100.] * 7 + [8.5], 0.29, False)):
        got_auc, got_keep = f0_group_rule(np.r_[asd, td], y)
        assert got_auc == pytest.approx(auc)
        assert got_keep is keep
    # 14 ASD against 25 TD children, 245 of the 350 pairs won: exactly 0.70, which roc_auc_score
    # reports as 0.7000000000000001. Rounding must not take the measures out of the model.
    asd = [1000.] * 9 + [19.5] + [-1000.] * 4          # nine beat all 25, one beats 20, four none
    auc, keep = f0_group_rule(np.r_[asd, np.arange(25.0)], np.r_[np.ones(14), np.zeros(25)])
    assert auc == pytest.approx(0.70) and keep is True
    # the range is the caller's to give (decision_rules.yaml: mil_analysis.f0_group_rule)
    assert f0_group_rule(np.r_[[100.] * 7 + [-100.] * 3, td], y, lo=0.40, hi=0.60)[1] is False
    # only one group: no AUC, and a rule that was not passed does not keep the measures
    auc, keep = f0_group_rule(np.arange(6.0), np.ones(6))
    assert np.isnan(auc) and keep is False
    # a child whose failure ratio is missing is left out, not scored as 0
    ratio = np.r_[np.full(10, 0.1), np.nan]
    assert f0_group_rule(ratio, np.r_[np.ones(5), np.zeros(5), 1]) == (0.5, True)
    with pytest.raises(ValueError):
        f0_group_rule(np.arange(5.0), np.ones(4))


# ---------------------------------------------------------------- the convention gate
def _chars(texts):
    return [list(t) for t in texts]


def test_segment_gate_judges_all_six_measures_under_all_three_transforms(cfg, rules,
                                                                          synthetic_corpus):
    res = segment_gate(synthetic_corpus, cfg, rules, words_of=_chars)
    assert [r.feature for r in res] == list(SEGMENT_MEASURES)
    assert all(np.isfinite(r.sd) and set(r.shifts) == set(TRANSFORMS) for r in res)


def test_a_childs_mean_skips_segments_without_a_value_and_is_missing_when_none_has_one():
    #                  A    B (never measured)    C
    rows = np.array([[1.0, np.nan, 3.0],
                     [3.0, np.nan, np.nan],
                     [5.0, np.nan, 5.0]])
    assert np.allclose(_child_means(rows), [3.0, np.nan, 4.0], equal_nan=True)   # not 0 for a gap
    assert np.isnan(_child_means(np.zeros((0, 3)))).all()                        # no segment at all


def test_segment_gate_follows_a_transform_that_normalization_does_not_undo(
        cfg, rules, synthetic_corpus, monkeypatch):
    # The three real transforms are all erased by normalization, so a gate that lost the variant
    # corpus on the way would look just the same. Doubling every text is not erased: characters
    # per utterance (C9) double, which is a shift of median / SD across the children.
    monkeypatch.setitem(TRANSFORMS, "strip_punctuation",
                        lambda turns, _rules: [dataclasses.replace(t, text=t.text * 2)
                                               for t in turns])
    res = {r.feature: r for r in segment_gate(synthetic_corpus, cfg, rules, words_of=_chars)}
    assert res["c_c9_chars_per_utt"].shifts["strip_punctuation"] > 0.20
    assert res["c_c9_chars_per_utt"].shifts["merge_same_speaker"] == pytest.approx(0.0)


def test_segment_gate_refuses_a_words_of_that_loses_texts(cfg, rules, synthetic_corpus):
    with pytest.raises(ValueError, match="words_of"):
        segment_gate(synthetic_corpus, cfg, rules, words_of=lambda ts: _chars(ts)[:-1])
