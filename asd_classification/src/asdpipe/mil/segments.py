"""Exchanges, segments and the six transcript-only segment measures (research plan 3.2-3.3).

The segment-attention model scores a child from a *series* of short stretches of
conversation, not from one session-wide average, so the transcript is first cut into
comparable stretches.

An **exchange** is the unit of conversation, and there are two kinds:

* examiner-led: every examiner turn, with the child's reply as ``pair_turns`` defines it (the
  child's first turn after the prompt, within 10 s). That is the pairing rule G1, F1 and F8
  already use; it is called here, not rewritten, so "reply" means the same thing in every
  measure. A prompt nobody answered is still an exchange, with ``child = None``.
* child-led: a child turn that is not a reply and that no examiner speech precedes within
  ``max_gap_s``. The rule looks at time only. It never asks whether the examiner's last
  sentence was a question, because the pipeline cannot confirm question detection without a
  human (plan 3.3).

A child turn sits in at most one exchange. A child who starts talking while the examiner is
still speaking has not replied (the gap is negative) and has not opened anything (the
examiner was talking). Nor has a child who carries on with their own next sentence within
``max_gap_s`` of the examiner's last words: only the first turn after a prompt is its reply,
and the examiner spoke recently. Those turns are not exchanges of their own, but they are
still speech inside the segment that spans them. (After a longer silence the same sentence
does open a child-led exchange.)

A **segment** is ``length`` consecutive exchanges, advancing one exchange at a time. A
session with fewer exchanges than that is a single segment.

**What a segment contains** is every child or examiner turn that *starts* in
[start of its first exchange, latest end among its exchanges). The right edge is open on
purpose: a transcript stitched at speaker changes (see ``normalize``) has many turns that
start at exactly the moment the previous one ends, and the next exchange's opening turn
belongs to the next segment, not this one. Overlap turns are ignored throughout.

Five of the six measures read those turns. F1 is the exception: it describes a prompt and
its reply, so it averages over the segment's own answered exchanges. A continuation sentence
that falls inside the time range has no prompt to echo.

**Missing versus zero.** A measure that cannot be computed in a segment (the child said no
words, nobody replied) is NaN, and the model imputes it from the training folds only (R3).
Writing 0 would assert "measured, and none", which is not what happened. E5 is the one
exception: once anyone has spoken the share is defined, and a child who said nothing has a
share of exactly 0.

Characters are Han characters only (C9, C1, F4), so punctuation, Latin letters and digits in
the transcript never move them. C9 and C1 also skip a child turn that holds no Han character
at all ("OK", digits, a turn emptied by punctuation stripping): it has no length in this unit
and its seconds are not speech the rate is measured over, the way ``e_mlu_words`` leaves out a
turn with no words. The skip applies to C9 and C1 only: E5 still counts every second the child
spoke, and F1 and F3 still read the turn. F4's count of Han characters is unaffected, since
such a turn holds none.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..contracts import Speaker, Turn
from ..features.e_clinical import _SECOND
from ..features.e_text import FUNCTION_WORDS, _overlap_ratio, pair_turns

__all__ = ["Exchange", "split_exchanges", "segment_index", "fixed_phrases", "segment_measures",
           "segment_spans", "turns_in_span", "SEGMENT_MEASURES", "CHILD_MEASURES",
           "PITCH_MEASURES", "STREAMS"]

# Per-segment measures, in the column order of ``segment_measures``. The codes in the names
# are the clinical assessment sheet's item numbers (plan 3.3), not ADOS codes.
SEGMENT_MEASURES =("e_f3_fixed_phrase_cover", "e_f1_echo_bigram", "e_e5_talk_share",
                    "e_f4_you_per_100c", "c_c9_chars_per_utt", "c_c1_chars_per_s")

# Per-child measures, taken whole-session from the existing acoustic extraction.
CHILD_MEASURES = ("c_exaggerated_utterance_ratio", "b7_mean_pairwise_cosdist",
                  "c_pitch_range_st_median", "b_level_rel_examiner_db")

# The two that depend on a pitch track; the F0 check can blank them out (plan 3.4).
PITCH_MEASURES = ("c_exaggerated_utterance_ratio", "c_pitch_range_st_median")

# Which stream each of the ten measures belongs to (plan 3.3). Used to drop a whole stream.
STREAMS = {
    "E": SEGMENT_MEASURES[:4],
    "C": ("c_c9_chars_per_utt", "c_c1_chars_per_s", "c_exaggerated_utterance_ratio"),
    "B": ("b7_mean_pairwise_cosdist", "c_pitch_range_st_median", "b_level_rel_examiner_db"),
}

_HAN = re.compile(r"[一-鿿]")


@dataclass(frozen=True)
class Exchange:
    """One unit of conversation.

    ``kind == "examiner"``: an examiner turn and the child's reply to it (``child`` is None
    when nobody replied). ``kind == "child"``: a child turn the examiner did not prompt
    (``examiner`` is None). ``t_start`` and ``t_end`` span the turns it holds.
    """

    kind: str
    examiner: Turn | None
    child: Turn | None
    t_start: float
    t_end: float


def _speaking_turns(turns: Sequence[Turn]) -> list[Turn]:
    """Child and examiner turns in start order. Overlap turns take no part."""
    kept = [t for t in turns if t.speaker in (Speaker.CHILD, Speaker.EXAMINER)]
    return sorted(kept, key=lambda t: t.t_start)


def split_exchanges(turns: Sequence[Turn], max_gap_s: float = 10.0) -> list[Exchange]:
    """Cut a session into exchanges, ordered by start time.

    ``max_gap_s`` is how long the examiner must have been silent before a child turn for
    that turn to count as child-led. ``pair_turns`` has its own 10 s reply window built in,
    so the two are meant to be equal (``mil.max_gap_s`` in pipeline.yaml mirrors it); set
    above 10 s, a child turn between the two windows would be neither a reply nor child-led.

    A reply is never also child-led. The explicit exclusion matters at exactly 10 s:
    ``pair_turns`` accepts a gap of 10 s, while "no examiner turn ends after the window
    opens" would accept the same turn as child-led, putting it in two exchanges.

    Exchanges that start together sort examiner-led first, so the order is deterministic.
    """
    ordered = _speaking_turns(turns)
    examiners = [t for t in ordered if t.speaker is Speaker.EXAMINER]
    reply_to = {ex.turn_id: ch for ex, ch in pair_turns(ordered)}
    replies = {ch.turn_id for ch in reply_to.values()}

    out: list[Exchange] = []
    for ex in examiners:
        ch = reply_to.get(ex.turn_id)
        t_end = ex.t_end if ch is None else max(ex.t_end, ch.t_end)
        out.append(Exchange("examiner", ex, ch, ex.t_start, t_end))
    for c in ordered:
        if c.speaker is not Speaker.CHILD or c.turn_id in replies:
            continue
        prompted = any(e.t_start < c.t_start and e.t_end > c.t_start - max_gap_s
                       for e in examiners)
        if not prompted:
            out.append(Exchange("child", None, c, c.t_start, c.t_end))
    out.sort(key=lambda x: (x.t_start, 0 if x.kind == "examiner" else 1))
    return out


def segment_index(n_exchanges: int, length: int) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` index ranges of the sliding segments, stride one.

    ``length`` consecutive exchanges each; fewer than ``length`` exchanges make one segment
    of everything there is, and no exchanges make none.
    """
    if length < 1:
        raise ValueError(f"segment length must be at least 1, got {length}")
    if n_exchanges <= 0:
        return []
    if n_exchanges < length:
        return [(0, n_exchanges)]
    return [(i, i + length) for i in range(n_exchanges - length + 1)]


def fixed_phrases(word_lists: Sequence[Sequence[str]], n_range: tuple[int, int],
                  min_repeats: int) -> set[tuple[str, ...]]:
    """F3's fixed phrases: word n-grams the child repeats at least ``min_repeats`` times.

    The counting rule is the E stream's (``e_text._repeated_ngrams``): n-grams of
    ``n_range`` words within one utterance, never across two, counted with overlaps, and an
    n-gram made only of function words is skipped because repeating those is normal at 4-6
    years. That function reports counts only, so the set itself is built here; a test keeps
    the two in step.
    """
    counts: Counter[tuple[str, ...]] = Counter()
    for words in word_lists:
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(words) - n + 1):
                gram = tuple(words[i : i + n])
                if all(w in FUNCTION_WORDS for w in gram):
                    continue
                counts[gram] += 1
    return {gram for gram, c in counts.items() if c >= min_repeats}


def _covered_words(words: Sequence[str], phrases: set[tuple[str, ...]],
                   n_range: tuple[int, int]) -> int:
    """How many of ``words`` sit inside at least one fixed phrase (overlaps count once)."""
    covered = [False] * len(words)
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(words) - n + 1):
            if tuple(words[i : i + n]) in phrases:
                covered[i : i + n] = [True] * n
    return sum(covered)


def _segment_row(group: Sequence[Exchange], inside: Sequence[Turn],
                 words: Mapping[int, Sequence[str]], phrases: set[tuple[str, ...]],
                 n_range: tuple[int, int]) -> list[float]:
    """The six measures of one segment, in ``SEGMENT_MEASURES`` order."""
    nan = float("nan")
    kids = [t for t in inside if t.speaker is Speaker.CHILD]

    # F3: the share of the child's words that sit inside a fixed phrase
    kid_words = [list(words.get(t.turn_id, ())) for t in kids]
    n_words = sum(len(ws) for ws in kid_words)
    f3 = (sum(_covered_words(ws, phrases, n_range) for ws in kid_words) / n_words
          if n_words else nan)

    # F1: mean bigram overlap over this segment's own answered exchanges
    overlaps = [_overlap_ratio(x.child.text, x.examiner.text, 2)
                for x in group if x.kind == "examiner" and x.child is not None]
    overlaps = [r for r in overlaps if np.isfinite(r)]
    f1 = float(np.mean(overlaps)) if overlaps else nan

    # E5: the child's share of the seconds either person spoke. Defined whenever anyone
    # spoke, so a silent child gets a real 0 here and NaN everywhere else.
    child_s = sum(t.duration for t in kids)
    spoken_s = child_s + sum(t.duration for t in inside if t.speaker is Speaker.EXAMINER)
    e5 = child_s / spoken_s if spoken_s > 0 else nan

    # F4, C9, C1: Han characters only
    n_chars = sum(len(_HAN.findall(t.text)) for t in kids)
    n_you = sum(t.text.count(p) for t in kids for p in _SECOND)
    f4 = 100.0 * n_you / n_chars if n_chars else nan
    # C9 and C1 also skip a child turn with no Han character at all ("OK", digits, a turn
    # emptied by punctuation stripping): it has no length in this unit, and its seconds are not
    # speech the rate is measured over. e_mlu_words leaves out a turn with no words the same way.
    # E5 and F4 are not skipping anything: E5 counts every second the child spoke, and a turn
    # with no Han character adds nothing to F4's count of Han characters.
    han_kids = [t for t in kids if _HAN.search(t.text)]
    han_s = sum(t.duration for t in han_kids)
    c9 = n_chars / len(han_kids) if han_kids else nan
    c1 = n_chars / han_s if han_s > 0 else nan
    return [f3, f1, e5, f4, c9, c1]


def _windows(ordered: Sequence[Turn], cfg: Mapping,
             length: int) -> list[tuple[Sequence[Exchange], float, float]]:
    """Every segment of a session as (its exchanges, lo, hi), in ``segment_index`` order.

    ``ordered`` is ``_speaking_turns`` of the session. The segment spans [lo, hi): the start of
    its first exchange to the latest end among its exchanges (module docstring). This is the
    one place a segment's span is decided; ``segment_measures`` and ``segment_spans`` both
    read it, so a span reported for a segment is the span its measures were taken over.
    """
    exchanges = split_exchanges(ordered, float(cfg["mil"]["max_gap_s"]))
    out = []
    for first, stop in segment_index(len(exchanges), length):
        group = exchanges[first:stop]
        # exchanges are sorted, so the first starts earliest; any of them may end latest
        out.append((group, group[0].t_start, max(x.t_end for x in group)))
    return out


def _inside(ordered: Sequence[Turn], starts: Sequence[float], lo: float,
            hi: float) -> Sequence[Turn]:
    """The turns of ``ordered`` (sorted by start, ``starts`` their starts) that start in
    [lo, hi): one contiguous slice, since the list is sorted."""
    return ordered[bisect_left(starts, lo):bisect_left(starts, hi)]


def segment_spans(turns: Sequence[Turn], cfg: Mapping, length: int) -> list[tuple[float, float]]:
    """The time span [lo, hi) of every segment of a session, in the row order of
    ``segment_measures`` for the same ``length``. The run uses it to say where in the
    recording the segment a model weighed most lies."""
    return [(lo, hi) for _, lo, hi in _windows(_speaking_turns(turns), cfg, length)]


def turns_in_span(turns: Sequence[Turn], lo: float, hi: float) -> list[Turn]:
    """The turns a segment spanning [lo, hi) contains, in start order: every child or examiner
    turn that starts inside it, the turns its measures read (module docstring)."""
    ordered = _speaking_turns(turns)
    return list(_inside(ordered, [t.t_start for t in ordered], lo, hi))


def segment_measures(turns: Sequence[Turn], words: Mapping[int, Sequence[str]], cfg: Mapping,
                     length: int) -> np.ndarray:
    """The six segment measures for one session: shape ``(n_segments, 6)``.

    Rows follow ``segment_index`` (one per segment, in time order) and columns follow
    ``SEGMENT_MEASURES``. A measure that cannot be computed is NaN, except E5 (see the module
    docstring). ``words`` maps a turn id to that turn's word list, which is CKIP's output in
    production; a child turn with no entry contributes no words. ``max_gap_s`` comes from
    ``cfg["mil"]`` and the F3 settings from ``cfg["features"]["e_stream"]``.
    """
    e_cfg = cfg["features"]["e_stream"]
    n_range = tuple(e_cfg["ngram_range"])
    min_repeats = int(e_cfg["ngram_min_repeats"])

    ordered = _speaking_turns(turns)
    windows = _windows(ordered, cfg, length)

    # Fixed phrases are a property of the whole session, not of a segment: a phrase is fixed
    # because the child said it three times anywhere, even if one segment holds a single one.
    phrases = fixed_phrases(
        [list(words.get(t.turn_id, ())) for t in ordered if t.speaker is Speaker.CHILD],
        n_range, min_repeats)

    starts = [t.t_start for t in ordered]
    out = np.full((len(windows), len(SEGMENT_MEASURES)), np.nan)
    for row, (group, lo, hi) in enumerate(windows):
        out[row] = _segment_row(group, _inside(ordered, starts, lo, hi), words, phrases, n_range)
    return out
