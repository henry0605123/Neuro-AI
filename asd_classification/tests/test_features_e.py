"""E stream, hand-computed. Embeddings and word segmentation are supplied by hand, so no
model is loaded and every expected value is exact."""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.contracts import Speaker
from asdpipe.features.e_text import (FUNCTION_WORDS, _overlap_ratio, _repeated_ngrams,
                                     examiner_probe_features, extract_e, pair_turns)
from conftest import hand_cfg, make_turn

@pytest.fixture(scope="module")
def cfg():
    """Hand-worked examples in this module run with the P12 floors at 1 (conftest.hand_cfg)."""
    from asdpipe.config import load_pipeline_config

    return hand_cfg(load_pipeline_config())



def _emb(*vals) -> np.ndarray:
    return np.array(vals, dtype=np.float32)


def test_pairing_rule_hand_example(mini_dialogue):
    """Three examiner turns, three child turns, each child turn follows an examiner turn."""
    pairs = pair_turns(mini_dialogue)
    assert [(e.turn_id, c.turn_id) for e, c in pairs] == [(0, 1), (2, 3), (4, 5)]


def test_pairing_drops_a_child_turn_that_is_too_far_from_the_prompt():
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "問", is_question=True),
             make_turn(1, Speaker.CHILD, 20.0, 21.0, "答")]
    assert pair_turns(turns) == []


def test_overlap_ratio_hand_example():
    """child "我去公園" bigrams: 我去, 去公, 公園  (3 of them)
       examiner "你去公園嗎" bigrams: 你去, 去公, 公園, 園嗎
       shared: 去公, 公園 -> 2/3 = 0.6667
    """
    assert _overlap_ratio("我去公園", "你去公園嗎") == pytest.approx(2 / 3, abs=1e-9)


def test_overlap_ratio_is_one_for_a_verbatim_echo():
    assert _overlap_ratio("那你喜歡什麼顏色", "那你喜歡什麼顏色") == pytest.approx(1.0)


def test_repeated_ngrams_hand_example():
    """Three utterances, each ["紅色", "車車"].

    Bigram ("紅色","車車") occurs 3 times -> 1 repeated type out of 1 total type,
    contributing 3 repeated tokens.
    """
    words = [["紅色", "車車"]] * 3
    rep_types, all_types, rep_tokens = _repeated_ngrams(words, (2, 5), 3)
    assert (rep_types, all_types, rep_tokens) == (1, 1, 3)


def test_repeated_ngrams_excludes_pure_function_word_strings():
    """「然後然後」 repeated is developmentally normal, not stereotyped (audit P0-F3)."""
    assert "然後" in FUNCTION_WORDS and "就是" in FUNCTION_WORDS
    words = [["然後", "就是"]] * 4
    assert _repeated_ngrams(words, (2, 5), 3) == (0, 0, 0)


def test_below_the_repeat_threshold_nothing_counts():
    words = [["紅色", "車車"]] * 2      # only twice, threshold is 3
    assert _repeated_ngrams(words, (2, 5), 3)[0] == 0


def _run(turns, cfg, embeddings, words, db):
    return extract_e(turns, embeddings, words, db, cfg)


def test_g1_excludes_yes_no_and_verbatim_echo(cfg, mini_dialogue):
    """Of the three pairs: (2,3) is a yes/no question, (4,5) is a verbatim echo.
    Only pair (0,1) may enter G1."""
    e = {i: _emb(1.0, 0.0) for i in range(6)}
    e[1] = _emb(0.6, 0.8)               # child answer, cos with examiner = 0.6
    words = {i: ["詞"] for i in range(6)}
    db = {i: -20.0 for i in range(6)}
    out = _run(mini_dialogue, cfg, e, words, db)
    assert out.g1_similarity.size == 1
    assert out.g1_similarity[0] == pytest.approx(0.6, abs=1e-6)
    assert out.quality["e_excluded_yes_no"] == 1.0
    assert out.quality["e_excluded_verbatim_echo"] == 1.0


def test_f1_keeps_every_pair_including_the_echo(cfg, mini_dialogue):
    """F1 is the echolalia measure, so it must see all three pairs."""
    e = {i: _emb(1.0, 0.0) for i in range(6)}
    words = {i: ["詞"] for i in range(6)}
    db = {i: -20.0 for i in range(6)}
    out = _run(mini_dialogue, cfg, e, words, db)
    assert out.f1_embed_similarity.size == 3
    # pair (4,5) is a verbatim repetition -> literal overlap exactly 1.0
    assert out.f1_overlap[-1] == pytest.approx(1.0)


def test_response_latency_hand_example(cfg, mini_dialogue):
    """Latencies: 1.5-1.0 = 0.5, 4.2-4.0 = 0.2, 7.0-6.5 = 0.5 -> median 0.5."""
    e = {i: _emb(1.0, 0.0) for i in range(6)}
    words = {i: ["詞"] for i in range(6)}
    db = {i: -20.0 for i in range(6)}
    out = _run(mini_dialogue, cfg, e, words, db)
    assert sorted(np.round(out.response_latency, 6)) == [0.2, 0.5, 0.5]
    assert out.scalars["e_latency_median"] == pytest.approx(0.5)


def test_volume_following_slope_hand_example(cfg, mini_dialogue):
    """Examiner levels -20, -22, -24; child levels -30, -31, -32.

    OLS slope of child on examiner = cov/var. Examiner deviations (2,0,-2), child (1,0,-1)
    -> cov = (2*1 + 0 + (-2)*(-1))/3 = 4/3, var = (4+0+4)/3 = 8/3, slope = 0.5.
    """
    e = {i: _emb(1.0, 0.0) for i in range(6)}
    words = {i: ["詞"] for i in range(6)}
    db = {0: -20.0, 1: -30.0, 2: -22.0, 3: -31.0, 4: -24.0, 5: -32.0}
    out = _run(mini_dialogue, cfg, e, words, db)
    assert out.scalars["e_volume_following_slope"] == pytest.approx(0.5, abs=1e-9)


def test_mlu_and_ttr_hand_example(cfg, mini_dialogue):
    """Child utterances segment to 3, 1 and 2 words -> MLU = 2.0.

    Tokens: 我 去 公園 | 喜歡 | 那 你  -> 6 tokens, 6 distinct -> TTR = 1.0
    """
    e = {i: _emb(1.0, 0.0) for i in range(6)}
    words = {0: ["你", "今天"], 1: ["我", "去", "公園"], 2: ["你", "喜歡"],
             3: ["喜歡"], 4: ["那", "你"], 5: ["那", "你"]}
    db = {i: -20.0 for i in range(6)}
    out = _run(mini_dialogue, cfg, e, words, db)
    assert out.scalars["e_mlu_words"] == pytest.approx(2.0)
    assert out.scalars["e_child_total_words"] == 6.0
    # only child turns (1, 3, 5) count; 那 also appears in examiner turn 4 but that is
    # not part of the child's lexicon
    assert out.scalars["e_ttr"] == pytest.approx(1.0, abs=1e-9)


def test_f8_is_symmetric_by_construction(cfg):
    """Identical embeddings in both directions -> child-follows == examiner-follows == 1,
    so the asymmetry must be exactly 0. This is audit P0-F8's requirement that the two
    directions use the same algorithm."""
    turns = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "問一", is_question=True),
             make_turn(1, Speaker.CHILD, 1.2, 2.0, "答一"),
             make_turn(2, Speaker.EXAMINER, 2.2, 3.0, "問二", is_question=True),
             make_turn(3, Speaker.CHILD, 3.2, 4.0, "答二")]
    e = {i: _emb(1.0, 0.0) for i in range(4)}
    out = _run(turns, cfg, e, {i: ["詞"] for i in range(4)}, {i: -20.0 for i in range(4)})
    assert out.scalars["e_f8_child_follows"] == pytest.approx(1.0)
    assert out.scalars["e_f8_examiner_follows"] == pytest.approx(1.0)
    assert out.scalars["e_f8_asymmetry"] == pytest.approx(0.0)


def test_f7_reports_all_three_windows(cfg):
    """Audit P0-F7: 3, 5 and 7 must all be computed. Never just 5.

    F7 moved out of the feature vector into `quality` on 2026-09-12 (floor effect on the TD
    cohort: positive for 4 of 19, sign test p = 0.019, and the three windows correlate
    0.993/0.994 so they are one test rather than three). The audit still requires all three
    to be computed and reported, so this test follows it into `quality` rather than being
    deleted.
    """
    turns = []
    for k in range(12):
        turns.append(make_turn(2 * k, Speaker.EXAMINER, 2.0 * k, 2.0 * k + 0.8,
                               f"問{k}", is_question=True))
        turns.append(make_turn(2 * k + 1, Speaker.CHILD, 2.0 * k + 1.0, 2.0 * k + 1.8,
                               f"答{k}"))
    # child similarity rises linearly with turn index -> positive slope in every window
    e = {}
    for k in range(12):
        e[2 * k] = _emb(1.0, 0.0)
        ang = np.pi / 2 * (1.0 - k / 11.0) * 0.9
        e[2 * k + 1] = _emb(float(np.cos(ang)), float(np.sin(ang)))
    out = _run(turns, cfg, e, {i: ["詞"] for i in range(24)},
               {i: -20.0 for i in range(24)})
    for w in (3, 5, 7):
        assert f"e_f7_slope_w{w}" in out.quality
        assert out.quality[f"e_f7_slope_w{w}"] > 0
        assert f"e_f7_slope_w{w}" not in out.scalars


def test_examiner_probe_never_touches_child_turns(mini_dialogue):
    """Audit P0-0C: the probe must be computable from examiner speech alone."""
    e = {i: _emb(1.0, 0.0) for i in range(6)}
    words = {i: ["詞", "詞"] for i in range(6)}
    db = {i: -20.0 for i in range(6)}
    full = examiner_probe_features(mini_dialogue, e, words, db)
    # deleting every child entry must not change the result
    kids = {t.turn_id for t in mini_dialogue if t.speaker is Speaker.CHILD}
    stripped = examiner_probe_features(
        mini_dialogue, {k: v for k, v in e.items() if k not in kids},
        {k: v for k, v in words.items() if k not in kids},
        {k: v for k, v in db.items() if k not in kids},
    )
    for k in ("p_n_turns", "p_mean_chars", "p_question_ratio", "p_yes_no_ratio",
              "p_mean_dur", "p_mean_db", "p_self_similarity"):
        assert full[k] == pytest.approx(stripped[k])
    assert full["p_n_turns"] == 3.0
    assert full["p_yes_no_ratio"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- absolute thresholds
def _pairs_with_similarity(sims, cfg_):
    """Build a dialogue whose pair similarities are exactly `sims`."""
    turns, e = [], {}
    for k, cos in enumerate(sims):
        ex_id, ch_id = 2 * k, 2 * k + 1
        turns.append(make_turn(ex_id, Speaker.EXAMINER, 2.0 * k, 2.0 * k + 0.8,
                               f"請說說看第{k}件事", is_question=True))
        turns.append(make_turn(ch_id, Speaker.CHILD, 2.0 * k + 1.0, 2.0 * k + 1.8,
                               f"回答內容{k}"))
        ang = float(np.arccos(np.clip(cos, -1.0, 1.0)))
        e[ex_id] = _emb(1.0, 0.0)
        e[ch_id] = _emb(float(np.cos(ang)), float(np.sin(ang)))
    words = {i: ["詞"] for i in e}
    db = {i: -20.0 for i in e}
    return extract_e(turns, e, words, db, cfg_)


def test_high_and_low_ratios_use_an_absolute_threshold(cfg):
    """Eight pairs at 0.9 and two at 0.1, thresholds 0.80 / 0.20.

    high ratio = 8/10 = 0.8, low ratio = 2/10 = 0.2 -- NOT 0.25 each.
    """
    out = _pairs_with_similarity([0.9] * 8 + [0.1] * 2, cfg)
    assert out.scalars["e_f1_embed_high_ratio"] == pytest.approx(0.8, abs=1e-6)
    assert out.scalars["e_g1_low_ratio"] == pytest.approx(0.2, abs=1e-6)


def test_ratios_are_not_self_referential(cfg):
    """The bug this replaces: a per-subject quantile gives 0.25 whatever the data.

    Two subjects with completely different similarity levels must get different ratios.
    """
    high = _pairs_with_similarity([0.95] * 10, cfg)
    low = _pairs_with_similarity([0.05] * 10, cfg)
    assert high.scalars["e_f1_embed_high_ratio"] == pytest.approx(1.0)
    assert low.scalars["e_f1_embed_high_ratio"] == pytest.approx(0.0)
    assert high.scalars["e_g1_low_ratio"] == pytest.approx(0.0)
    assert low.scalars["e_g1_low_ratio"] == pytest.approx(1.0)
    assert high.scalars["e_f1_embed_high_ratio"] != pytest.approx(0.25)


def test_thresholds_come_from_config(cfg):
    assert 0.0 < cfg["features"]["e_stream"]["g1_low_similarity_threshold"]         < cfg["features"]["e_stream"]["f1_high_similarity_threshold"] <= 1.0
