"""Replacement features drawn from the clinical assessment sheet -- E5, F2, F4, F7.

Why these four. Six features were removed on 2026-09-12 because the data cannot support
them: CPP (17 of 19 subjects over audit P1-5's missingness ceiling), the tone-contour
features (three independent repair attempts, all below the majority-class baseline), and
multi-turn semantic adaptation (floor effect, positive for 4 of 19). The clinician's
instruction was to drop what the recording conditions make impossible and substitute other
items the assessment sheet rates 高.

Everything here is computed from the **transcript and the turn structure only**. That is
the point: the removals were all caused by recording quality or by forced-alignment
precision, and none of these four touches either. They survive a noisy room.

Assessment-sheet mapping, with the clinician's own wording:

    E5 發聲量的多寡    "整體發出聲音的頻率高低"          -> e_vocal_*
    F2 延宕仿說        "重複很久以前聽過的台詞、廣告詞"   -> e_f2_*      (rated 超級高)
    F4 代名詞反轉      "你我混用"                        -> e_pron_*
    F7 主動發話少      "幾乎只回答、很少主動起話題"       -> e_init_*

**F2 and three of the four F4 measures were implemented and then removed** (PI decision,
2026-09-12) after failing split-half reliability on the 19 TD subjects:
``e_f2_delayed_echo_ratio`` 0.38, ``e_f2_median_lag_turns`` 0.44,
``e_pron_first_per_100c`` 0.23, ``e_pron_second_share`` 0.40,
``e_pron_second_in_reply_ratio`` -0.07.

The failures share one shape, and it is the same shape as B1 and B3 before them: **every
measure that failed is the proportion of a rare event, and every measure that passed is a
continuous rate.** F4 makes the contrast cleanly -- "second-person pronouns per 100
characters" survives at r = 0.72 while "proportion of 你-questions answered with 你"
(median 2.9% of turns) lands at r = -0.07, measuring the same construct. At roughly 283
child turns per session, an event occurring in under 15% of them cannot be estimated
stably enough to rank children. Prefer rates to proportions when designing the next one.

**A naming warning.** The assessment sheet's F7 (主動發話少) and the research plan's F7
(多輪語意調適) are different constructs that share a label; the plan's F7 is the one with
the floor effect, and it is *not* what ``e_init_*`` measures. The features here are named
after what they measure rather than after a sheet number, so that the two can never be
confused in a results table. See the terminology table in CLAUDE.md.

**What F2 can and cannot see.** Delayed echolalia proper means reproducing a line heard
days or weeks ago -- an advertisement, a cartoon. Nothing in a single session can verify
that. What is measurable is its within-session shadow: an utterance that closely repeats
something the child said much earlier, far enough back that it is not conversational
repair. That is a proxy, and it is labelled as one. F1 covers immediate echoes of the
examiner; F3 covers short stereotyped n-grams; this covers long-lag self-repetition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..contracts import Speaker, Turn
from .common import min_values

__all__ = ["ClinicalFeatures", "extract_clinical"]

# Mandarin personal pronouns. 咱 is included for completeness; 人家 is not, because in
# child speech it is as often an indefinite ("someone") as a first-person substitute.
_FIRST = ("我", "咱")
_SECOND = ("你", "妳")
_THIRD = ("他", "她", "牠", "它")
_HAN = re.compile(r"[一-鿿]")


@dataclass
class ClinicalFeatures:
    scalars: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)


def _count(text: str, pronouns: Sequence[str]) -> int:
    return sum(text.count(p) for p in pronouns)


def extract_clinical(
    turns: Sequence[Turn],
    embeddings: Mapping[int, np.ndarray],
    words: Mapping[int, Sequence[str]],
    cfg: Mapping,
) -> ClinicalFeatures:
    """Turn-structure and text features for one subject. No audio, no alignment."""
    e_cfg = cfg["features"]["e_stream"]
    c_cfg = e_cfg.get("clinical", {})
    lag_min = int(c_cfg.get("f2_min_lag_turns", 20))
    f2_thr = float(c_cfg.get("f2_similarity_threshold", e_cfg["f1_high_similarity_threshold"]))
    f2_min_chars = int(c_cfg.get("f2_min_chars", 4))
    min_chars = int(e_cfg["min_answer_chars"])

    ordered = sorted(turns, key=lambda t: t.t_start)
    kids = [t for t in ordered if t.speaker is Speaker.CHILD]
    examiners = [t for t in ordered if t.speaker is Speaker.EXAMINER]
    session_s = (ordered[-1].t_end - ordered[0].t_start) if ordered else 0.0
    minutes = session_s / 60.0

    mv = min_values(cfg)
    nan = float("nan")
    scalars: dict[str, float] = {}
    quality: dict[str, float] = {"e_clinical_n_child_turns": float(len(kids))}

    # ---------------------------------------------------------------- E5 發聲量的多寡
    child_s = sum(t.duration for t in kids)
    child_words = sum(len([w for w in words.get(t.turn_id, ()) if _HAN.search(w)]) for t in kids)
    # Per-minute rates keep their real zeros: a child who says nothing has been measured,
    # not missed. Only a session with no time to divide by is unmeasurable (P12).
    scalars["e_vocal_turns_per_min"] = float(len(kids) / minutes) if minutes > 0 else nan
    scalars["e_vocal_seconds_per_min"] = float(child_s / minutes) if minutes > 0 else nan
    scalars["e_vocal_words_per_min"] = float(child_words / minutes) if minutes > 0 else nan
    # Share rather than a rate: robust to a session that runs long because the examiner
    # talks more, which is itself informative but is a different thing.
    total_s = child_s + sum(t.duration for t in examiners)
    scalars["e_vocal_speech_share"] = float(child_s / total_s) if total_s > 0 else nan

    # ---------------------------------------------------------------- F7 主動發話少
    # A child turn counts as *elicited* when the turn immediately before it is an examiner
    # question. Everything else -- following the examiner's statement, or following the
    # child's own turn -- is the child carrying the conversation.
    prev_speaker: list[Turn | None] = [None] + list(ordered[:-1])
    elicited = n_self_continued = n_scored_init = 0
    for t, prev in zip(ordered, prev_speaker):
        if t.speaker is not Speaker.CHILD or prev is None:
            continue
        # Count the denominator here rather than deriving it from len(kids): only a child
        # turn that *has* a predecessor can be classified, and how many of those there are
        # depends on who spoke first, not on an arithmetic adjustment.
        n_scored_init += 1
        if prev.speaker is Speaker.EXAMINER and prev.is_question:
            elicited += 1
        elif prev.speaker is Speaker.CHILD:
            n_self_continued += 1
    # With no classifiable child turn the old max(n, 1) guard returned 1.0 -- "initiates
    # everything" -- for a child who never spoke (P12).
    enough = n_scored_init >= mv
    scalars["e_init_ratio"] = float(1.0 - elicited / n_scored_init) if enough else nan
    scalars["e_init_self_continuation_ratio"] = (float(n_self_continued / n_scored_init)
                                                 if enough else nan)
    # Speaker switches rather than a turn share (PI decision, 2026-09-12). A turn share is
    # not robust to how the transcript was segmented: merging consecutive same-speaker turns
    # -- exactly what the TD annotators did at short pauses -- moves it by 26% to 34% on this
    # cohort. Simulated on the TD transcripts at merge gaps of 0.5/1.0/2.0 s, the turn rate
    # fell 25.6%/31.3%/33.6% while the switch rate moved 0.0% at every threshold, because
    # merging *same-speaker* turns cannot change how often the speaker changes.
    switches = sum(1 for a, b in zip(ordered, ordered[1:]) if a.speaker is not b.speaker)
    scalars["e_init_switches_per_min"] = float(switches / minutes) if minutes > 0 else nan
    quality["e_init_n_switches"] = float(switches)
    quality["e_init_n_elicited"] = float(elicited)
    quality["e_init_n_scorable"] = float(n_scored_init)

    # ---------------------------------------------------------------- F4 代名詞反轉
    # Two views. The rates say whether the child uses personal pronouns at all -- avoidance
    # is itself described in the literature. The reply ratio is the specific reversal
    # pattern: asked "你叫什麼名字", a child who reverses answers "你叫...".
    text_all = "".join(t.text for t in kids)
    n_han = len(_HAN.findall(text_all))
    n_chars = max(n_han, 1)
    first, second, third = (_count(text_all, p) for p in (_FIRST, _SECOND, _THIRD))
    scalars["e_pron_second_per_100c"] = float(100.0 * second / n_chars) if n_han >= mv else nan
    quality["e_pron_first_per_100c"] = float(100.0 * first / n_chars)
    quality["e_pron_third_per_100c"] = float(100.0 * third / n_chars)

    return ClinicalFeatures(scalars=scalars, quality=quality)
