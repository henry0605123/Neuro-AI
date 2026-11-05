"""Exchanges, segments and the six segment measures, hand-computed.

Every session here is a handful of hand-built turns, so each expected value can be checked
with a pencil: no CKIP, no model, no audio. ``words`` is passed in directly, the way
``extract_e`` takes it.

The first eight tests are the ones the task brief specifies. The block after them pins
decisions the brief leaves open, so that a later edit cannot change them unnoticed.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from asdpipe.contracts import Speaker
from asdpipe.features.e_text import _repeated_ngrams
from asdpipe.mil.segments import (CHILD_MEASURES, PITCH_MEASURES, SEGMENT_MEASURES, STREAMS,
                                  fixed_phrases, segment_index, segment_measures,
                                  split_exchanges)
from conftest import make_turn

E, C, O = Speaker.EXAMINER, Speaker.CHILD, Speaker.OVERLAP


def _turns(*rows):
    """(speaker, start, end, text) rows -> turns numbered 0, 1, 2... in the order given."""
    return [make_turn(i, who, t0, t1, text) for i, (who, t0, t1, text) in enumerate(rows)]


def _measures(turns, words, cfg, length=5):
    """Segment 0's six measures as a name -> value dict."""
    return dict(zip(SEGMENT_MEASURES, segment_measures(turns, words, cfg, length)[0]))


# ---------------------------------------------------------------- the brief's eight tests
def test_examiner_led_with_and_without_response():
    # E 0-2 asks and C 3-4 answers; E 20-22 speaks and nobody answers
    turns = _turns((E, 0, 2, "你叫什麼名字"), (C, 3, 4, "我叫小明"), (E, 20, 22, "好"))
    ex = split_exchanges(turns)
    assert [e.kind for e in ex] == ["examiner", "examiner"]
    assert ex[0].child.text == "我叫小明" and ex[1].child is None
    # the time range runs from the examiner opening to the later of the two ends
    assert [(e.t_start, e.t_end) for e in ex] == [(0, 4), (20, 22)]


def test_child_led_needs_ten_seconds_of_examiner_silence():
    # E 0-2, C 13-14: the examiner has been silent for 11 s, so the child opened this one
    led = split_exchanges(_turns((E, 0, 2, "你好"), (C, 13, 14, "看")))
    assert [e.kind for e in led] == ["examiner", "child"]
    assert led[0].child is None
    assert led[1].examiner is None and led[1].child.text == "看"
    assert (led[1].t_start, led[1].t_end) == (13, 14)
    # E 0-2, C 5-6: three seconds later is a response, not a second exchange
    replied = split_exchanges(_turns((E, 0, 2, "你好"), (C, 5, 6, "好")))
    assert [e.kind for e in replied] == ["examiner"]


def test_overlap_and_continuation_turns_are_not_exchanges():
    turns = _turns(
        (E, 0, 2, "你看"),
        (C, 1.5, 2.5, "嗯"),         # starts while the examiner is still speaking
        (E, 20, 22, "你說呢"),
        (O, 22.5, 22.9, "嗯"),       # an overlap turn takes no part at all
        (C, 23, 24, "我說好"),        # the reply
        (C, 25, 26, "然後呢"),        # the child's own next sentence
    )
    ex = split_exchanges(turns)
    assert sum(e.kind == "child" for e in ex) == 0
    assert [e.kind for e in ex] == ["examiner", "examiner"]
    assert ex[0].child is None                       # talking over the examiner is no reply
    assert ex[1].child.text == "我說好"                # the first turn after the prompt is


def test_segment_index():
    assert segment_index(7, 5) == [(0, 5), (1, 6), (2, 7)]
    assert segment_index(5, 5) == [(0, 5)]
    assert segment_index(3, 5) == [(0, 3)] and segment_index(0, 5) == []


def test_measures_hand_computed(cfg):
    # one segment: E 0-7 "你好", C 7-10 "你好嗎" (3 s child, 7 s examiner)
    turns = _turns((E, 0, 7, "你好"), (C, 7, 10, "你好嗎"))
    out = segment_measures(turns, {1: ["你好", "嗎"]}, cfg, 5)
    assert out.shape == (1, 6)
    m = dict(zip(SEGMENT_MEASURES, out[0]))
    assert m["e_e5_talk_share"] == pytest.approx(0.3)             # 3 s of the 10 s spoken
    assert m["e_f1_echo_bigram"] == pytest.approx(0.5)            # {你好,好嗎} vs {你好}
    assert m["e_f4_you_per_100c"] == pytest.approx(100 / 3)       # one 你 in three characters
    assert m["c_c9_chars_per_utt"] == pytest.approx(3.0)
    assert m["c_c1_chars_per_s"] == pytest.approx(1.0)            # three characters in 3 s
    assert m["e_f3_fixed_phrase_cover"] == 0.0                    # words exist, none repeated


def test_f3_cover_uses_session_level_phrases(cfg):
    said = ["恐龍", "好", "厲害"]
    turns = _turns(
        (E, 0, 1, "你看"), (C, 1, 2, "恐龍好厲害"),
        (E, 5, 6, "還有呢"), (C, 6, 7, "我要"),
        (E, 10, 11, "你看"), (C, 11, 12, "恐龍好厲害"),
        (E, 15, 16, "你看"), (C, 16, 17, "恐龍好厲害"),
    )
    words = {1: said, 3: ["我", "要"], 5: said, 7: said}
    # said three times across the session, so it is a fixed phrase ...
    phrases = fixed_phrases([said, ["我", "要"], said, said], (2, 5), 3)
    assert ("恐龍", "好", "厲害") in phrases and ("我", "要") not in phrases
    # ... although segment 0 (two exchanges) holds only one saying plus ["我", "要"]
    f3 = SEGMENT_MEASURES.index("e_f3_fixed_phrase_cover")
    assert segment_measures(turns, words, cfg, 2)[:, f3].tolist() == pytest.approx(
        [3 / 5, 3 / 5, 1.0])


def test_f3_ignores_function_word_only_ngrams():
    assert fixed_phrases([["然後", "就是"]] * 5, (2, 5), 3) == set()


def test_silent_child_gives_nan_but_e5_zero(cfg):
    examiner_only = _turns((E, 0, 2, "你好"), (E, 3, 5, "你叫什麼名字"), (E, 8, 9, "好"))
    row = segment_measures(examiner_only, {}, cfg, 5)[0]
    assert np.isnan(row[[0, 1, 3, 4, 5]]).all() and row[2] == 0.0


# ------------------------------------------- beyond the brief: decisions it leaves open
def test_a_reply_exactly_ten_seconds_late_is_not_also_child_led():
    # pair_turns accepts a gap of exactly 10 s. The child-led formula alone would call the
    # same turn child-led as well (the examiner's end is not *after* the window opens), and
    # one child turn would then sit in two exchanges.
    ex = split_exchanges(_turns((E, 0, 2, "你好"), (C, 12, 13, "好")))
    assert [e.kind for e in ex] == ["examiner"] and ex[0].child.text == "好"


def test_exchanges_starting_together_sort_examiner_first():
    # child at 5-7, examiner at 5-6: no examiner started *before* the child, so it is child-led
    rows = [(C, 5, 7, "我"), (E, 5, 6, "好")]
    for order in (rows, rows[::-1]):
        assert [e.kind for e in split_exchanges(_turns(*order))] == ["examiner", "child"]


def test_a_continuation_after_a_long_silence_is_child_led():
    # the reply is C 3-4; the child's next sentence at 14 comes 12 s after the examiner stopped
    turns = _turns((E, 0, 2, "你好"), (C, 3, 4, "好"), (C, 14, 15, "看"))
    ex = split_exchanges(turns)
    assert [e.kind for e in ex] == ["examiner", "child"]
    assert ex[0].child.text == "好" and ex[1].child.text == "看"


def test_continuation_counts_for_child_measures_but_not_for_echo(cfg):
    turns = _turns(
        (E, 0, 2, "你好嗎"),
        (C, 3, 4, "你好嗎"),          # the reply, a full echo
        (C, 5, 6, "我要吃飯飯"),       # the child's own next sentence: in the range, not a reply
        (E, 20, 21, "好"),
    )
    m = _measures(turns, {}, cfg)
    assert m["e_f1_echo_bigram"] == pytest.approx(1.0)            # only the reply is compared
    assert m["c_c9_chars_per_utt"] == pytest.approx(4.0)          # (3 + 5) / 2
    assert m["c_c1_chars_per_s"] == pytest.approx(4.0)            # eight characters in 2 s
    assert m["e_e5_talk_share"] == pytest.approx(2 / 5)           # 2 s of the 5 s spoken
    assert m["e_f4_you_per_100c"] == pytest.approx(12.5)          # one 你 in eight characters
    assert np.isnan(m["e_f3_fixed_phrase_cover"])                 # no words supplied


def test_f1_skips_replies_too_short_to_hold_a_bigram(cfg):
    turns = _turns((E, 0, 1, "你好嗎"), (C, 1, 2, "你好嗎"),    # full echo
                   (E, 5, 6, "好嗎"), (C, 6, 7, "好"))          # one character: no bigram at all
    assert _measures(turns, {}, cfg)["e_f1_echo_bigram"] == pytest.approx(1.0)
    only_short = _turns((E, 0, 1, "好嗎"), (C, 1, 2, "好"))
    assert np.isnan(_measures(only_short, {}, cfg)["e_f1_echo_bigram"])


def test_segments_slide_one_exchange_and_the_range_is_half_open(cfg):
    # seven touching exchanges: E [2k, 2k+1] says 好, C [2k+1, 2k+2] answers with k+1 characters
    rows = []
    for k in range(7):
        rows += [(E, 2 * k, 2 * k + 1, "好"), (C, 2 * k + 1, 2 * k + 2, "啊" * (k + 1))]
    out = segment_measures(_turns(*rows), {}, cfg, 5)
    assert out.shape == (3, 6)
    col = {name: out[:, j].tolist() for j, name in enumerate(SEGMENT_MEASURES)}
    # segment 0 covers exchanges 0-4 (replies of 1-5 characters), segment 1 covers 1-5, ...
    assert col["c_c9_chars_per_utt"] == pytest.approx([3.0, 4.0, 5.0])
    assert col["c_c1_chars_per_s"] == pytest.approx([3.0, 4.0, 5.0])
    # the next exchange's examiner starts at exactly the segment's latest end and must stay
    # out, and the first exchange's examiner starts at exactly its start and must stay in
    assert col["e_e5_talk_share"] == pytest.approx([0.5, 0.5, 0.5])


def test_max_gap_comes_from_the_config(cfg):
    # E 0-2 is answered, then the child carries on 9 s after the examiner stopped
    turns = _turns((E, 0, 2, "好"), (C, 3, 4, "嗯"), (C, 11, 12, "嗯"))
    assert segment_measures(turns, {}, cfg, 1).shape[0] == 1      # examiner spoke within 10 s
    tight = copy.deepcopy(cfg)
    tight["mil"]["max_gap_s"] = 5.0
    assert segment_measures(turns, {}, tight, 1).shape[0] == 2    # not within 5 s: child-led


def test_f3_settings_come_from_the_config(cfg):
    said = ["恐龍", "好", "厲害"]
    turns = _turns((E, 0, 1, "你看"), (C, 1, 2, "恐龍好厲害"),
                   (E, 5, 6, "你看"), (C, 6, 7, "恐龍好厲害"),
                   (E, 10, 11, "你看"), (C, 11, 12, "恐龍好厲害"))
    words = {1: said, 3: said, 5: said}
    assert segment_measures(turns, words, cfg, 5)[0, 0] == pytest.approx(1.0)
    stricter = copy.deepcopy(cfg)
    stricter["features"]["e_stream"]["ngram_min_repeats"] = 4     # said only three times
    assert segment_measures(turns, words, stricter, 5)[0, 0] == 0.0
    longer = copy.deepcopy(cfg)
    longer["features"]["e_stream"]["ngram_range"] = [4, 5]        # no 4-word phrase exists
    assert segment_measures(turns, words, longer, 5)[0, 0] == 0.0


def test_input_order_does_not_matter(cfg):
    turns = _turns((E, 0, 2, "你叫什麼名字"), (C, 3, 4, "我叫小明"),
                   (E, 20, 22, "好"), (C, 40, 41, "看"))
    assert split_exchanges(turns[::-1]) == split_exchanges(turns)
    np.testing.assert_array_equal(segment_measures(turns[::-1], {}, cfg, 5),
                                  segment_measures(turns, {}, cfg, 5))


def test_no_exchanges_gives_no_segments(cfg):
    assert segment_measures([], {}, cfg, 5).shape == (0, 6)
    assert segment_measures(_turns((O, 0, 1, "嗯")), {}, cfg, 5).shape == (0, 6)


def test_only_han_characters_are_counted_and_both_you_forms_count(cfg):
    turns = _turns((E, 0, 1, "嗯"), (C, 1, 2, "你好OK, 妳!"))
    m = _measures(turns, {}, cfg)
    assert m["c_c9_chars_per_utt"] == pytest.approx(3.0)          # 你 好 妳; OK , ! are not
    assert m["e_f4_you_per_100c"] == pytest.approx(100 * 2 / 3)   # 你 and 妳 both count


def test_a_child_turn_without_han_characters_is_skipped_by_c9_and_c1(cfg):
    # "OK" has no length in the plan's unit (Han characters), so C9 and C1 leave the turn out of
    # numerator and denominator alike, the way e_mlu_words leaves out a turn with no words
    # (decided 2026-11-01; counting it as a zero-character utterance would give 2.0 and 2.0)
    turns = _turns((E, 0, 1, "好"), (C, 1, 2, "我要吃飯"), (E, 5, 6, "好"), (C, 6, 7, "OK"))
    m = _measures(turns, {}, cfg)
    assert m["c_c9_chars_per_utt"] == pytest.approx(4.0)          # four characters in 1 utterance
    assert m["c_c1_chars_per_s"] == pytest.approx(4.0)            # four characters in 1 s
    # only C9 and C1 skip it: the child did speak for that second, so E5 still counts it
    assert m["e_e5_talk_share"] == pytest.approx(2 / 4)


def test_c9_and_c1_skip_every_kind_of_turn_without_han_characters(cfg):
    # digits, Latin letters and a turn emptied by punctuation stripping alike: none has a length
    # in the plan's unit, so neither its utterance nor its seconds enter C9 and C1
    turns = _turns(
        (E, 0, 1, "好"), (C, 1, 2, "你好嗎"),           # 3 characters, 1 s
        (E, 5, 6, "好"), (C, 6, 8, "我要吃飯飯"),        # 5 characters, 2 s
        (E, 10, 11, "好"), (C, 11, 15, "123"),           # digits only, 4 s
        (E, 20, 21, "好"), (C, 21, 24, ""),              # nothing left after stripping, 3 s
        (E, 30, 31, "好"), (C, 31, 32, "OK"),            # Latin letters only, 1 s
    )
    m = _measures(turns, {}, cfg)
    assert m["c_c9_chars_per_utt"] == pytest.approx(4.0)          # (3 + 5) characters / 2 turns
    assert m["c_c1_chars_per_s"] == pytest.approx(8 / 3)          # 8 characters in 1 s + 2 s
    assert m["e_e5_talk_share"] == pytest.approx(11 / 16)         # E5 still counts all 11 s
    assert m["e_f4_you_per_100c"] == pytest.approx(100 / 8)       # one 你 in eight characters
    # with no Han turn at all there is nothing to average: NaN, although the child did speak
    no_han = _measures(_turns((E, 0, 1, "好"), (C, 1, 2, "123"), (E, 5, 6, "好"), (C, 6, 7, "OK")),
                       {}, cfg)
    assert np.isnan([no_han[k] for k in ("c_c9_chars_per_utt", "c_c1_chars_per_s",
                                         "e_f4_you_per_100c")]).all()
    assert no_han["e_e5_talk_share"] == pytest.approx(0.5)


def test_fixed_phrases_match_the_e_stream_repeated_types():
    # one definition of "fixed phrase": the set built here has as many members as the
    # E stream's F3 counts repeated types, for any n-gram range and repeat threshold
    word_lists = ([["恐龍", "好", "厲害"]] * 3
                  + [["我", "要", "恐龍", "好"], ["然後", "就是"] * 2, ["紅色", "車車"]])
    for n_range, min_repeats in (((2, 5), 3), ((2, 3), 2), ((3, 5), 1)):
        assert (len(fixed_phrases(word_lists, n_range, min_repeats))
                == _repeated_ngrams(word_lists, n_range, min_repeats)[0])


def test_segment_index_rejects_a_non_positive_length():
    with pytest.raises(ValueError):
        segment_index(5, 0)


def test_measure_name_tables_are_consistent():
    assert len(SEGMENT_MEASURES) == 6 and len(set(SEGMENT_MEASURES)) == 6
    assert set(PITCH_MEASURES) <= set(CHILD_MEASURES)
    assert STREAMS["E"] == SEGMENT_MEASURES[:4]
    # the three streams split the ten measures between them, each name exactly once
    listed = [name for names in STREAMS.values() for name in names]
    assert sorted(listed) == sorted(SEGMENT_MEASURES + CHILD_MEASURES)
