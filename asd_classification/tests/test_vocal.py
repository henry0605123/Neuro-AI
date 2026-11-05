"""Untranscribed child vocalisation (features/vocal.py)."""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.contracts import Speaker
from asdpipe.features.vocal import (attribute, centroid, merge_close, subtract,
                                    untranscribed_segments, vocal_features)

from conftest import make_turn

SR = 16000
V_CFG = {"voice_over_floor_db": 6.0, "voice_min_run_s": 0.096, "merge_gap_s": 0.25,
         "min_segment_s": 0.3, "response_window_s": 2.0}


def test_subtract_keeps_only_what_no_turn_covers():
    assert subtract([(0.0, 10.0)], [(1.0, 2.0), (5.0, 6.0)]) == [(0.0, 1.0), (2.0, 5.0), (6.0, 10.0)]
    assert subtract([(1.2, 1.8)], [(1.0, 2.0)]) == []


def test_merge_close_joins_sounds_a_breath_apart():
    assert merge_close([(0.0, 1.0), (1.2, 2.0), (3.0, 4.0)], 0.25) == [(0.0, 2.0), (3.0, 4.0)]


def _session(voiced, seconds=20.0):
    a = (0.002 * np.random.default_rng(0).standard_normal(int(seconds * SR))).astype(np.float32)
    for s, e in voiced:
        t = np.arange(int((e - s) * SR)) / SR
        a[int(s * SR):int(s * SR) + len(t)] += (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return a


def test_untranscribed_segments_are_the_voiced_parts_nobody_wrote_down():
    """Voice at 2-3 s (transcribed), 6-7.5 s (not), 9-9.15 s (too short), 12-13 s (not)."""
    audio = _session([(2.0, 3.0), (6.0, 7.5), (9.0, 9.15), (12.0, 13.0)])
    turns = [make_turn(0, Speaker.EXAMINER, 1.0, 3.2, "你好"),
             make_turn(1, Speaker.CHILD, 14.0, 15.0, "好")]
    segs = untranscribed_segments(audio, SR, turns, V_CFG)
    assert [(round(a, 1), round(b, 1)) for a, b in segs] == [(6.0, 7.5), (12.0, 13.0)]


def test_a_segment_is_the_childs_only_when_clearly_closer_and_close_enough():
    child_c, exam_c = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    emb = np.array([[0.9, 0.1, 0.0],     # child
                    [0.1, 0.9, 0.0],     # examiner
                    [0.5, 0.45, 0.0],    # too close to call
                    [0.2, 0.0, 0.98]])   # nobody's voice (a toy, a chair)
    is_child, _, _ = attribute(emb, child_c, exam_c, margin=0.2, floor=0.5)
    assert is_child.tolist() == [True, False, False, False]


def test_centroid_needs_enough_windows():
    assert centroid(np.ones((1, 4)), min_windows=2) is None
    assert np.linalg.norm(centroid(np.ones((3, 4)), min_windows=2)) == pytest.approx(1.0)


def test_vocal_features_count_attributed_sounds_and_vocal_answers():
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 2.0, "你叫什麼名字", is_question=True),
             make_turn(1, Speaker.EXAMINER, 10.0, 12.0, "你幾歲", is_question=True),
             make_turn(2, Speaker.CHILD, 20.0, 22.0, "我五歲"),
             make_turn(3, Speaker.EXAMINER, 58.0, 60.0, "好")]
    segs = [(2.5, 3.5), (30.0, 31.0)]          # a vocal reply to the first question; babble
    out = vocal_features(turns, segs, [True, True], V_CFG, min_questions=2)
    s = out.scalars
    assert s["e_vocal_all_seconds_per_min"] == pytest.approx(4.0)       # (2 + 1 + 1) s in 1 min
    assert s["e_vocal_untranscribed_share"] == pytest.approx(0.5)
    assert s["e_vocal_response_rate"] == pytest.approx(0.5)             # only the first question
    none = vocal_features(turns, segs, None, V_CFG, min_questions=2).scalars
    assert all(np.isnan(v) for v in none.values())                      # voiceprint unavailable
