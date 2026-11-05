"""Assessment-sheet substitutes E5 / F2 / F4 / F7 (features/e_clinical.py).

These replace features the recording conditions made unusable, so the thing that matters
most is that they measure the construct rather than a property of the recording. Each test
constructs a transcript whose answer is known by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.contracts import Speaker
from asdpipe.features.e_clinical import extract_clinical

from conftest import hand_cfg, make_turn

@pytest.fixture(scope="module")
def cfg():
    """Hand-worked examples in this module run with the P12 floors at 1 (conftest.hand_cfg)."""
    from asdpipe.config import load_pipeline_config

    return hand_cfg(load_pipeline_config())



def _emb(turn_ids, vecs=None):
    """Unit vectors; identical ids share a vector, so cosine similarity is exactly 1."""
    if vecs is None:
        return {i: np.array([1.0, 0.0], dtype=np.float32) for i in turn_ids}
    return {i: v / np.linalg.norm(v) for i, v in zip(turn_ids, vecs)}


def _qa_session(n_pairs, child_text="我很好啊", question="你好嗎", child_dur=1.0):
    turns = []
    for k in range(n_pairs):
        turns.append(make_turn(2 * k, Speaker.EXAMINER, 10.0 * k, 10.0 * k + 1.0,
                               question, is_question=True))
        turns.append(make_turn(2 * k + 1, Speaker.CHILD, 10.0 * k + 2.0,
                               10.0 * k + 2.0 + child_dur, child_text))
    return turns


def test_e5_vocalisation_rates_are_per_minute_not_per_session(cfg):
    """A child who talks at the same rate for twice as long must score the same.

    Raw totals would reward a long appointment; E5 asks about 整體發出聲音的頻率.
    """
    short = extract_clinical(_qa_session(6), _emb(range(12)),
                             {i: ["我", "很", "好", "啊"] for i in range(12)}, cfg)
    long = extract_clinical(_qa_session(12), _emb(range(24)),
                            {i: ["我", "很", "好", "啊"] for i in range(24)}, cfg)
    for k in ("e_vocal_turns_per_min", "e_vocal_seconds_per_min", "e_vocal_words_per_min"):
        assert short.scalars[k] == pytest.approx(long.scalars[k], rel=0.1), k
    assert 0.0 < short.scalars["e_vocal_speech_share"] < 1.0


def test_f4_second_person_rate_is_a_rate_not_a_rare_event_proportion(cfg):
    """Only the rate survived reliability. The proportion version of the same construct --
    "how often is a 你-question answered with 你" -- scored r = -0.07 because it fires on
    2.9% of turns, so it was removed rather than kept as a noisy near-duplicate."""
    you = _qa_session(6, child_text="你要去哪裡玩")
    me = _qa_session(6, child_text="我要去公園玩")
    w = {i: ["詞"] for i in range(12)}
    a = extract_clinical(you, _emb(range(12)), w, cfg)
    b = extract_clinical(me, _emb(range(12)), w, cfg)
    assert a.scalars["e_pron_second_per_100c"] > b.scalars["e_pron_second_per_100c"]
    for gone in ("e_pron_second_in_reply_ratio", "e_pron_second_share",
                 "e_pron_first_per_100c", "e_f2_delayed_echo_ratio"):
        assert gone not in a.scalars
    assert len(a.scalars) == 8


def test_switch_rate_is_immune_to_merging_same_speaker_turns(cfg):
    """The reason it replaced the turn share. Merging consecutive same-speaker turns is what
    the TD annotators did at short pauses; on that cohort it moved the turn rate by 26-34%
    and the switch rate by 0.0%."""
    from dataclasses import replace as _replace

    fine = []
    tid = 0
    for k in range(6):
        fine.append(make_turn(tid, Speaker.EXAMINER, 10.0 * k, 10.0 * k + 1.0, "你好嗎",
                              is_question=True)); tid += 1
        for j in range(3):                         # one child utterance, split three ways
            fine.append(make_turn(tid, Speaker.CHILD, 10.0 * k + 2.0 + j,
                                  10.0 * k + 2.8 + j, "我很好")); tid += 1
    coarse = [t for t in fine if t.speaker is Speaker.EXAMINER]
    for k in range(6):
        coarse.append(make_turn(1000 + k, Speaker.CHILD, 10.0 * k + 2.0, 10.0 * k + 4.8,
                                "我很好我很好我很好"))
    coarse.sort(key=lambda t: t.t_start)
    w = {t.turn_id: ["詞"] for t in fine + coarse}
    a = extract_clinical(fine, _emb([t.turn_id for t in fine]), w, cfg)
    b = extract_clinical(coarse, _emb([t.turn_id for t in coarse]), w, cfg)
    assert a.scalars["e_init_switches_per_min"] == pytest.approx(
        b.scalars["e_init_switches_per_min"], rel=1e-6)
    # the turn rate, which this replaced, does move
    assert a.scalars["e_vocal_turns_per_min"] > 2 * b.scalars["e_vocal_turns_per_min"]


def test_f7_a_child_who_only_answers_questions_scores_zero_initiation(cfg):
    turns = _qa_session(8)
    out = extract_clinical(turns, _emb(range(16)), {i: ["詞"] for i in range(16)}, cfg)
    assert out.scalars["e_init_ratio"] == pytest.approx(0.0, abs=0.15)
    assert out.scalars["e_init_self_continuation_ratio"] == 0.0


def test_f7_self_continuation_and_unprompted_turns_raise_initiation(cfg):
    """Two child turns in a row, and a child turn after a non-question, both count."""
    turns = [
        make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "你好嗎", is_question=True),
        make_turn(1, Speaker.CHILD, 1.0, 2.0, "我很好"),
        make_turn(2, Speaker.CHILD, 2.0, 3.0, "然後我還要講"),      # self-continuation
        make_turn(3, Speaker.EXAMINER, 3.0, 4.0, "這是一張圖片"),    # not a question
        make_turn(4, Speaker.CHILD, 4.0, 5.0, "那是恐龍"),           # unprompted
    ]
    out = extract_clinical(turns, _emb(range(5)), {i: ["詞"] for i in range(5)}, cfg)
    # three scorable child turns, one of them elicited
    assert out.scalars["e_init_ratio"] == pytest.approx(2 / 3, abs=0.01)
    assert out.scalars["e_init_self_continuation_ratio"] == pytest.approx(1 / 3, abs=0.01)





def test_a_child_who_never_speaks_is_measured_where_possible_and_nan_elsewhere(cfg):
    """No child turns at all (health check P12).

    How much the child spoke is a real measurement -- zero. Ratios over the child's turns
    have nothing to divide and are NaN, "cannot be measured". The old code returned 1.0
    for the initiation ratio, i.e. "initiates everything", for a child who never spoke.
    """
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 60.0, "你好嗎", is_question=True)]
    out = extract_clinical(turns, _emb([0]), {0: ["詞"]}, cfg)
    s = out.scalars
    assert len(s) == 8
    for k in ("e_vocal_turns_per_min", "e_vocal_seconds_per_min", "e_vocal_words_per_min",
              "e_vocal_speech_share", "e_init_switches_per_min"):
        assert s[k] == 0.0, k
    for k in ("e_init_ratio", "e_init_self_continuation_ratio", "e_pron_second_per_100c"):
        assert np.isnan(s[k]), k
