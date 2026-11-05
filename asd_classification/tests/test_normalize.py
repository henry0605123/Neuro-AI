"""Source normalization and the convention-invariance gate.

The property that matters is invariance: after normalization, a transcript produced under
one pipeline's conventions must look exactly like the same session produced under another's.
The tests build sessions whose audio is known, apply each convention by hand, and check that
normalization erases it.
"""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.contracts import Speaker
from asdpipe.eval.convention_gate import (TRANSFORMS, gate_shifts, stitch_speaker_changes)
from asdpipe.normalize import (merge_same_speaker, normalize_turns, strip_punctuation,
                               trim_to_voice)

from conftest import make_turn

SR = 16000


def _session(voice_spans, seconds=30.0, seed=0):
    """Room noise with 220 Hz tones where `voice_spans` say someone is speaking."""
    a = (0.002 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)
    for s, e in voice_spans:
        t = np.arange(int((e - s) * SR)) / SR
        a[int(s * SR):int(s * SR) + len(t)] += (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return a


def test_strip_punctuation_removes_marks_and_keeps_words():
    assert strip_punctuation("你好嗎？我很好，謝謝。OK! 3隻") == "你好嗎我很好謝謝OK3隻"
    assert strip_punctuation("～嗯…「對」") == "嗯對"


def test_merge_is_the_rulestitch_rule_and_is_idempotent():
    t = [make_turn(0, Speaker.CHILD, 0.0, 1.0, "我"),
         make_turn(1, Speaker.CHILD, 2.0, 3.0, "要"),          # gap 1.0 -> merged
         make_turn(2, Speaker.CHILD, 5.0, 6.0, "玩"),          # gap 2.0 -> not merged
         make_turn(3, Speaker.EXAMINER, 6.5, 7.0, "好")]
    once = merge_same_speaker(t, 1.5)
    assert [x.text for x in once] == ["我要", "玩", "好"]
    assert merge_same_speaker(once, 1.5) == once


def test_merged_turn_takes_the_last_members_question_flags():
    """A reply answers the last thing said."""
    t = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "你喜不喜歡", is_question=True, is_yes_no=True),
         make_turn(1, Speaker.EXAMINER, 1.2, 2.0, "那你今天做什麼", is_question=True)]
    m = merge_same_speaker(t, 1.5)[0]
    assert m.is_question and not m.is_yes_no


def test_trim_moves_edges_onto_the_voice_and_never_extends_an_end():
    a = _session([(2.0, 3.0)])
    t = [make_turn(0, Speaker.CHILD, 1.5, 3.8, "我")]
    out = trim_to_voice(t, a, SR, over_floor_db=6.0, min_run_s=0.096, min_keep_s=0.1,
                        search_back_s=2.0)[0]
    assert out.t_start == pytest.approx(2.0, abs=0.05)
    assert out.t_end == pytest.approx(3.0, abs=0.05)
    assert out.t_end <= 3.8


def test_a_turn_with_no_voice_keeps_its_span():
    """A quiet child's turn must not be discarded by a detector tuned on louder ones."""
    a = _session([])
    t = [make_turn(0, Speaker.CHILD, 1.0, 2.0, "嗯")]
    out = trim_to_voice(t, a, SR, over_floor_db=6.0, min_run_s=0.096, min_keep_s=0.1,
                        search_back_s=2.0)
    assert out[0].t_start == 1.0 and out[0].t_end == 2.0


def _dialogue():
    """Examiner speaks 1-2 s, child replies 2.6-3.4 s, examiner again 4.0-5.0 s."""
    audio = _session([(1.0, 2.0), (2.6, 3.4), (4.0, 5.0)])
    turns = [make_turn(0, Speaker.EXAMINER, 1.0, 2.0, "你叫什麼名字？", is_question=True),
             make_turn(1, Speaker.CHILD, 2.8, 3.4, "小明。"),     # annotator started late
             make_turn(2, Speaker.EXAMINER, 4.0, 5.0, "好。")]
    return turns, audio


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_normalization_erases_every_simulated_convention(cfg, name):
    """The gate's transforms, applied to a clean session, must leave no trace.

    Stitching is the hard case: it pulls the child's start back to the examiner's end. A
    shrink-only trim would then find the true onset (2.6 s) in the stitched copy but keep the
    annotator's late start (2.8 s) in the original -- two different answers for one session.
    Searching from the previous turn's end gives both the same onset.
    """
    from asdpipe.config import load_decision_rules

    turns, audio = _dialogue()
    base, _ = normalize_turns(turns, audio, SR, cfg)
    moved, _ = normalize_turns(TRANSFORMS[name](turns, load_decision_rules()), audio, SR, cfg)
    assert [(round(t.t_start, 3), round(t.t_end, 3), t.text) for t in moved] == \
           [(round(t.t_start, 3), round(t.t_end, 3), t.text) for t in base]
    child = [t for t in base if t.speaker is Speaker.CHILD][0]
    assert child.t_start == pytest.approx(2.6, abs=0.05)      # the voice, not the annotation


def test_stitch_only_touches_speaker_changes():
    t = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "問"),
         make_turn(1, Speaker.CHILD, 1.5, 2.0, "答"),          # speaker change, gap 0.5
         make_turn(2, Speaker.CHILD, 2.5, 3.0, "再答"),        # same speaker: untouched
         make_turn(3, Speaker.EXAMINER, 6.0, 7.0, "問")]       # gap 3.0 > 2.0: untouched
    out = stitch_speaker_changes(t, 2.0)
    assert [x.t_start for x in out] == [0.0, 1.0, 2.5, 6.0]


def test_gate_statistic_is_median_shift_over_sd():
    base = {"f": np.array([0.0, 1.0, 2.0, 3.0, 4.0]), "const": np.ones(5)}
    shifted = {"f": base["f"] + 0.5, "const": np.ones(5)}
    res = {r.feature: r for r in gate_shifts(base, {"t": shifted})}
    sd = np.std(base["f"], ddof=1)
    assert res["f"].shifts["t"] == pytest.approx(0.5 / sd)
    assert not res["const"].shifts                            # constant: cannot be judged
