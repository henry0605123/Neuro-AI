"""Shared fixtures. Deliberately model-free: every feature test builds its inputs by hand.

`extract_c` takes the F0 track as an argument and `extract_e` takes the embeddings and the
word segmentation as arguments, so the hand-computed tests can supply exact values and run
in milliseconds without loading SBERT, CKIP, ECAPA or CREPE.
"""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.config import load_pipeline_config
from asdpipe.contracts import Speaker, SyllableSpan, Turn


@pytest.fixture(scope="session")
def cfg():
    return load_pipeline_config()


def hand_cfg(cfg):
    """The config with the P12 minimum-data floors at 1.

    Hand-worked examples check a formula's arithmetic on a handful of values, far below
    the floors. The floors themselves are tested separately, on the real config.
    """
    import copy

    c = copy.deepcopy(cfg)
    c["features"]["min_data"]["values_per_statistic"] = 1
    c["features"]["e_stream"]["min_pairs_per_subject"] = 1
    c["features"]["e_stream"]["a12_min_pairs"] = 1
    return c


def make_turn(turn_id, speaker, t0, t1, text, syllables=(), **kw) -> Turn:
    return Turn(turn_id=turn_id, speaker=speaker, t_start=t0, t_end=t1, text=text,
                syllables=tuple(syllables), **kw)


def syllables_from(durations, tones, t0=0.0, gaps=None, texts=None):
    """Build syllable spans with explicit durations and (optionally) explicit gaps."""
    gaps = list(gaps or [0.0] * (len(durations) - 1))
    spans, t = [], t0
    for i, (d, tone) in enumerate(zip(durations, tones)):
        spans.append(SyllableSpan(t_start=t, t_end=t + d, tone=tone,
                                  text=(texts[i] if texts else ""),
                                  underlying_tone=tone))
        t += d
        if i < len(gaps):
            t += gaps[i]
    return spans, t


@pytest.fixture
def mini_dialogue():
    """A five-turn examiner/child exchange used by the E-stream hand-computed tests."""
    turns = [
        make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "你今天做了什麼", is_question=True),
        make_turn(1, Speaker.CHILD, 1.5, 2.5, "我去公園玩球"),
        make_turn(2, Speaker.EXAMINER, 3.0, 4.0, "你喜歡吃冰淇淋嗎",
                  is_question=True, is_yes_no=True),
        make_turn(3, Speaker.CHILD, 4.2, 5.0, "喜歡"),
        make_turn(4, Speaker.EXAMINER, 5.5, 6.5, "那你喜歡什麼顏色", is_question=True),
        make_turn(5, Speaker.CHILD, 7.0, 8.0, "那你喜歡什麼顏色"),   # verbatim echo
    ]
    return turns
