"""Utterance-level intonation (assessment sheet A2/A3/B2/B5).

These tests drive the extractor with F0 tracks whose answer is known analytically, because
the whole reason this module exists is that a *plausible-looking* feature can still be
measuring nothing -- the syllable-level tone features looked fine and were below chance.
"""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.contracts import Speaker, Turn
from asdpipe.features.intonation import extract_intonation, frame_db

HOP = 0.01
SR = 16000


def _track(hz: np.ndarray):
    """An F0 track that is fully voiced, plus a flat level track on the same grid."""
    return hz, np.full(len(hz), 0.9), np.full(len(hz), -20.0)


def _turns(n: int, dur: float, question: bool = False):
    return [Turn(turn_id=i, speaker=Speaker.CHILD, t_start=0.0, t_end=dur,
                 text="測試", is_question=question) for i in range(n)]


def test_pitch_range_is_measured_in_semitones(cfg):
    """A sweep over a factor of 2 is one octave; P95-P5 recovers 90% of it."""
    t = np.arange(int(2.0 / HOP)) * HOP
    f0, per, db = _track(200.0 * 2.0 ** (t / 2.0))
    out = extract_intonation(_turns(6, 2.0), f0, per, db, HOP, cfg)
    assert out.scalars["c_pitch_range_st_median"] == pytest.approx(0.9 * 12.0, abs=0.5)


def test_monotone_speech_scores_below_varied_speech_on_every_a2_a3_measure(cfg):
    """A2/A3 are the whole point: a constant pitch must rank below a sweep on all of them.

    This used to assert on `c_flat_utterance_ratio` (B1), which was removed after failing
    split-half reliability. B1 was only the low tail of A2, so the property it checked is
    still checkable -- on the continuous measures that survived.
    """
    n = int(2.0 / HOP)
    flat = extract_intonation(_turns(6, 2.0), *_track(np.full(n, 220.0)), HOP, cfg)
    t = np.arange(n) * HOP
    wide = extract_intonation(_turns(6, 2.0), *_track(200.0 * 2.0 ** (t / 2.0)), HOP, cfg)
    for name in ("c_pitch_range_st_median", "c_pitch_range_st_p90", "c_pitch_sd_st_median",
                 "c_pitch_dynamism_st_per_s"):
        assert flat.scalars[name] < wide.scalars[name], name


def test_removed_features_are_gone_rather_than_silently_zero(cfg):
    """B1/B3 were dropped by PI decision. A feature that lingers as a constant 0.0 is worse
    than one that is absent: `c_final_slope_question` sat at exactly 0.000 for all 19
    subjects for a whole run before anyone noticed it was a bug, not a finding."""
    n = int(2.0 / HOP)
    out = extract_intonation(_turns(6, 2.0), *_track(np.full(n, 220.0)), HOP, cfg)
    for gone in ("c_flat_utterance_ratio", "c_final_slope_question", "c_final_slope_statement",
                 "c_final_slope_contrast", "c_final_rise_ratio_question"):
        assert gone not in out.scalars
    assert len(out.scalars) == 9


def test_thresholds_come_from_config_not_from_the_data(cfg):
    """R3: the B2 cut-off is an absolute constant. Changing the config must change the result.

    If they were cohort quantiles this test would pass trivially and the feature would be
    self-referential -- the exact failure that had to be repaired in the E stream (F1).
    """
    n = int(2.0 / HOP)
    t = np.arange(n) * HOP
    args = (_turns(6, 2.0), *_track(200.0 * 2.0 ** (t / 4.0)))   # ~3 semitones of excursion
    lenient = {**cfg["features"]["c_stream"]["intonation"], "exaggerated_range_st": 0.5}
    strict = {**cfg["features"]["c_stream"]["intonation"], "exaggerated_range_st": 20.0}
    mk = lambda i: {**cfg, "features": {**cfg["features"], "c_stream": {
        **cfg["features"]["c_stream"], "intonation": i}}}
    key = "c_exaggerated_utterance_ratio"
    assert extract_intonation(*args, HOP, mk(lenient)).scalars[key] == 1.0
    assert extract_intonation(*args, HOP, mk(strict)).scalars[key] == 0.0


def test_unvoiced_turns_are_skipped_rather_than_scored_as_monotone(cfg):
    """A turn with no usable F0 must not silently count as a perfectly flat one."""
    n = int(2.0 / HOP)
    f0, per, db = _track(np.full(n, 220.0))
    out = extract_intonation(_turns(6, 2.0), f0, np.zeros(n), db, HOP, cfg)
    assert out.quality["c_inton_n_turns"] == 0
    assert out.quality["c_inton_usable"] == 0.0


def test_frame_db_tracks_level_and_matches_the_f0_grid(cfg):
    quiet = 0.01 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    loud = 10.0 * quiet
    dq, dl = frame_db(quiet, SR, HOP), frame_db(loud, SR, HOP)
    assert len(dq) == int(np.ceil(SR / (HOP * SR)))
    assert np.median(dl) - np.median(dq) == pytest.approx(20.0, abs=0.5)
