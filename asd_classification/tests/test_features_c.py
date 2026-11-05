"""C stream, hand-computed. Every number below is worked out by hand in the docstring."""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.contracts import Speaker
from asdpipe.features.c_prosody import CONTOUR_POINTS, SEQ_DIM, extract_c
from asdpipe.features.common import npvi
from conftest import hand_cfg, make_turn, syllables_from

@pytest.fixture(scope="module")
def cfg():
    """Hand-worked examples in this module run with the P12 floors at 1 (conftest.hand_cfg)."""
    from asdpipe.config import load_pipeline_config

    return hand_cfg(load_pipeline_config())



def test_npvi_hand_example():
    """durations 0.2, 0.4, 0.2

        |0.2-0.4| / 0.3 = 0.6667
        |0.4-0.2| / 0.3 = 0.6667
        nPVI = 100 * mean(0.6667, 0.6667) = 66.667
    """
    assert npvi([0.2, 0.4, 0.2]) == pytest.approx(66.6667, abs=1e-3)


def test_npvi_is_zero_for_a_perfectly_even_rhythm():
    assert npvi([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)


def test_npvi_needs_two_syllables():
    assert np.isnan(npvi([0.3]))


def _one_child_turn(durations, tones, gaps=None, t0=1.0):
    spans, t_end = syllables_from(durations, tones, t0=t0, gaps=gaps)
    turn = make_turn(1, Speaker.CHILD, t0, t_end, "x" * len(durations), spans)
    return [make_turn(0, Speaker.EXAMINER, 0.0, 0.9, "問句"), turn], {1: spans}


def _flat_f0(n_frames, hz=250.0, hop=0.01):
    return np.full(n_frames, hz), np.full(n_frames, 0.99)


def test_pause_counting_and_rates_hand_example(cfg):
    """Four syllables of 0.20 s, with gaps 0.05 / 0.30 / 0.05 s after the first three.

        turn duration     = 4*0.20 + 0.05 + 0.30 + 0.05 = 1.20 s
        pause threshold   = 0.20 s  ->  exactly one pause (the 0.30 s gap)
        speaking rate     = 4 / 1.20 = 3.333 syll/s          (includes pauses)
        articulation rate = 4 / (1.20 - 0.30) = 4.444 syll/s (excludes pauses)

    The two 0.05 s gaps stay in the denominator on purpose: they are below the pause
    threshold, so they are speech time, not pauses. Dividing by the summed syllable spans
    instead would give 4 / 0.80 = 5.0 and, on real recordings, inflate the rate to about
    10 syll/s -- far outside what children produce.
    """
    turns, syls = _one_child_turn([0.2] * 4, [1, 2, 3, 4], gaps=[0.05, 0.30, 0.05])
    f0, per = _flat_f0(400)
    out = extract_c(turns, syls, f0, per, 0.01, cfg)
    s = out.scalars
    assert s["c_pause_rate_per_min"] * (1.20 / 60.0) == pytest.approx(1.0, abs=1e-6)
    assert s["c_pause_mean_s"] == pytest.approx(0.30, abs=1e-9)
    assert s["c_articulation_rate"] == pytest.approx(4.0 / 0.90, abs=1e-6)
    assert s["c_speaking_rate"] == pytest.approx(4.0 / 1.20, abs=1e-6)
    assert s["c_articulation_rate"] > s["c_speaking_rate"]
    assert s["c_utt_dur_mean"] == pytest.approx(1.20, abs=1e-9)
    # P20: the pause-threshold sensitivity runs are reported, not features
    assert not any("_thr" in k for k in s)
    assert any(k.startswith("c_pause_rate_per_min_thr") for k in out.quality)


def test_syllable_duration_statistics_hand_example(cfg):
    """durations 0.10, 0.20, 0.30 -> mean 0.20, sd = sqrt(2/3)*0.1 = 0.08165, cv = 0.40825."""
    turns, syls = _one_child_turn([0.1, 0.2, 0.3], [1, 1, 1])
    f0, per = _flat_f0(200)
    s = extract_c(turns, syls, f0, per, 0.01, cfg).scalars
    assert s["c_syl_dur_mean"] == pytest.approx(0.2, abs=1e-9)
    assert s["c_syl_dur_cv"] == pytest.approx(0.4082483, abs=1e-6)


def test_neutral_tone_duration_ratio_hand_example(cfg):
    """Two lexical syllables of 0.30 s and one neutral of 0.15 s -> ratio 0.15/0.30 = 0.5."""
    turns, syls = _one_child_turn([0.3, 0.3, 0.15], [1, 2, 0])
    f0, per = _flat_f0(200)
    s = extract_c(turns, syls, f0, per, 0.01, cfg).scalars
    assert s["c_neutral_tone_dur_ratio"] == pytest.approx(0.5, abs=1e-9)


def test_sandhi_realisation_hand_example(cfg):
    """Underlying T3 T3 T3 with the first two produced as T2 -> 2 of 2 pairs realised.

    Pairs are (0,1) and (1,2). Position 0 is produced as T2 -> realised; position 1 is also
    produced as T2 -> realised. So the rate is 2/2 = 1.0.
    """
    from asdpipe.contracts import SyllableSpan

    spans = [
        SyllableSpan(1.0, 1.2, tone=2, underlying_tone=3),
        SyllableSpan(1.2, 1.4, tone=2, underlying_tone=3),
        SyllableSpan(1.4, 1.6, tone=3, underlying_tone=3),
    ]
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 0.9, "問"),
             make_turn(1, Speaker.CHILD, 1.0, 1.6, "三三三", spans)]
    f0, per = _flat_f0(200)
    # Reported as a diagnostic rather than a feature since 2026-09-12: MFA gives only the
    # produced tone, so on real data this is identically 0. The logic is kept and tested
    # because it works the moment `underlying_tone` is filled from the dictionary.
    q = extract_c(turns, {1: spans}, f0, per, 0.01, cfg).quality
    assert q["c_sandhi_realisation"] == pytest.approx(1.0)
    assert q["c_n_sandhi_pairs"] == 2.0


def test_pct_v_hand_example(cfg):
    """Two syllables of 0.10 s (10 frames each at a 10 ms hop).

    Frames 0-9 belong to syllable 1 and 10-19 to syllable 2. Making only the first 5 frames
    of each syllable voiced gives voiced time 10 frames * 0.01 = 0.10 s against 0.20 s of
    syllable time, so %V = 0.5.
    """
    turns, syls = _one_child_turn([0.1, 0.1], [1, 1], t0=0.0)
    per = np.zeros(60)
    per[0:5] = 0.99
    per[10:15] = 0.99
    f0 = np.full(60, 250.0)
    s = extract_c(turns, syls, f0, per, 0.01, cfg).scalars
    assert s["c_pct_v"] == pytest.approx(0.5, abs=1e-6)


def test_sequence_tensor_shape_and_tone_one_hot(cfg):
    turns, syls = _one_child_turn([0.2, 0.2, 0.2], [0, 3, 4])
    f0, per = _flat_f0(200)
    out = extract_c(turns, syls, f0, per, 0.01, cfg)
    assert out.sequence.shape == (3, SEQ_DIM) == (3, 8)
    assert int(cfg["features"]["c_stream"]["syllable_feature_dim"]) == SEQ_DIM
    # 0-2 are duration channels; 3..7 are the one-hot over tones (0,1,2,3,4)
    assert out.sequence[0, 3] == 1.0   # tone 0
    assert out.sequence[1, 6] == 1.0   # tone 3
    assert out.sequence[2, 7] == 1.0   # tone 4
    assert out.sequence[:, 3:].sum() == 3.0


def test_f0_range_is_locked_to_the_child_band(cfg):
    """Audit P1-3: an adult default range must be rejected outright, not silently used."""
    bad = {**cfg, "features": {**cfg["features"],
                               "c_stream": {**cfg["features"]["c_stream"],
                                            "f0_min_hz": 75.0, "f0_max_hz": 300.0}}}
    turns, syls = _one_child_turn([0.2, 0.2], [1, 1])
    f0, per = _flat_f0(200)
    with pytest.raises(ValueError, match="100-600"):
        extract_c(turns, syls, f0, per, 0.01, bad)


def test_tone_template_mode_is_guarded(cfg):
    bad = {**cfg, "features": {**cfg["features"],
                               "c_stream": {**cfg["features"]["c_stream"],
                                            "tone_template": "global"}}}
    turns, syls = _one_child_turn([0.2, 0.2], [1, 1])
    f0, per = _flat_f0(200)
    with pytest.raises(NotImplementedError):
        extract_c(turns, syls, f0, per, 0.01, bad)


def test_residual_is_zero_when_every_syllable_matches_its_template(cfg):
    """Identical flat F0 on every syllable of the same tone -> residual exactly 0."""
    turns, syls = _one_child_turn([0.2] * 6, [1] * 6)
    f0, per = _flat_f0(400)
    out = extract_c(turns, syls, f0, per, 0.01, cfg)
    assert np.allclose(out.residual_mean, 0.0, atol=1e-9)
    # Tone accuracy is a diagnostic now, not a feature -- see test_dropped_tone_contour_...
    assert out.quality["c_tone_accuracy"] == pytest.approx(1.0)


def test_dropped_tone_contour_features_are_absent_from_the_feature_vector(cfg):
    """The tone-contour features were removed on 2026-09-12 after three independent repair
    attempts all landed below the majority-class baseline. They must be *gone*, not zeroed:
    a constant feature reads as a finding rather than as a failure, which is exactly how
    `c_final_slope_question` survived a full run at 0.000 before anyone checked."""
    turns, syls = _one_child_turn([0.2] * 6, [1] * 6)
    f0, per = _flat_f0(400)
    pooled = extract_c(turns, syls, f0, per, 0.01, cfg).pooled(["mean", "std"])
    for gone in ("c_tone_accuracy", "c_tone_consistency_rmse", "c_sandhi_realisation",
                 "c_residual_f0_mean", "c_residual_f0_std", "c_residual_f0_sd_within"):
        assert gone not in pooled, gone
    assert "c_neutral_tone_dur_ratio" in pooled   # duration-only, survives


def test_contour_resampling_is_five_points():
    assert CONTOUR_POINTS == 5


# ---------------------------------------------------------------- alignment artefacts
def test_multi_second_syllables_are_dropped(cfg):
    """A syllable of several seconds is a stretched vowel, not speech.

    Real data: when a turn's transcript is far shorter than its audio, MFA fills the gap by
    extending one nucleus -- single `o` phones spanning 4-8.5 s were measured. Five normal
    0.2 s syllables plus one 5 s artefact must leave five, with the artefact counted.
    """
    from asdpipe.contracts import SyllableSpan

    spans = [SyllableSpan(1.0 + 0.2 * i, 1.2 + 0.2 * i, 1) for i in range(5)]
    spans.append(SyllableSpan(2.0, 7.0, 1))            # the artefact
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 0.9, "問"),
             make_turn(1, Speaker.CHILD, 1.0, 7.0, "六個字的句子", spans)]
    f0, per = _flat_f0(800)
    out = extract_c(turns, {1: spans}, f0, per, 0.01, cfg)
    assert out.quality["c_n_syllables"] == 5
    assert out.quality["c_implausible_syllable_ratio"] == pytest.approx(1 / 6)
    assert 0.05 <= out.scalars["c_syl_dur_mean"] <= 1.0


def test_sub_50ms_syllables_are_dropped_too(cfg):
    from asdpipe.contracts import SyllableSpan

    spans = [SyllableSpan(1.0, 1.2, 1), SyllableSpan(1.2, 1.21, 1), SyllableSpan(1.3, 1.5, 1)]
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 0.9, "問"),
             make_turn(1, Speaker.CHILD, 1.0, 1.5, "三個字", spans)]
    f0, per = _flat_f0(400)
    out = extract_c(turns, {1: spans}, f0, per, 0.01, cfg)
    assert out.quality["c_n_syllables"] == 2


def test_pct_v_uses_the_nucleus_split_when_the_aligner_supplies_it(cfg):
    """Three syllables of 0.20 s whose nuclei are 0.12 s -> %V = 0.6, dC = 0 (all 0.08 s)."""
    from asdpipe.contracts import SyllableSpan

    spans = [SyllableSpan(1.0 + 0.2 * i, 1.2 + 0.2 * i, 1, vocalic_s=0.12) for i in range(3)]
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 0.9, "問"),
             make_turn(1, Speaker.CHILD, 1.0, 1.6, "三個字", spans)]
    f0, per = _flat_f0(400)
    out = extract_c(turns, {1: spans}, f0, per, 0.01, cfg)
    assert out.scalars["c_pct_v"] == pytest.approx(0.6, abs=1e-6)
    assert out.scalars["c_delta_c"] == pytest.approx(0.0, abs=1e-9)
    assert out.quality["c_pct_v_from_alignment"] == 1.0


def test_pct_v_falls_back_to_voicing_without_a_nucleus_split(cfg):
    """The synthetic path has no phone tier, so the voicing proxy is used and flagged."""
    turns, syls = _one_child_turn([0.2, 0.2], [1, 1], t0=0.0)
    per = np.zeros(60)
    per[0:10] = 0.99
    f0 = np.full(60, 250.0)
    out = extract_c(turns, syls, f0, per, 0.01, cfg)
    assert out.quality["c_pct_v_from_alignment"] == 0.0
    assert out.scalars["c_pct_v"] == pytest.approx(0.25, abs=1e-6)
