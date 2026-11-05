"""Untranscribed child vocalisation, measured from the audio the same way in both arms.

Why. The ASD transcripts mark unintelligible child vocalisations as [UNINTELLIGIBLE]; the
TD transcribers left such sounds out altogether. Counting the label measures who made the
transcript, and dropping it erased about nine-tenths of one low-verbal child's voice
(health check P13: 6 intelligible sentences, about 12 s, against 18 unintelligible
vocalisations, about 116 s). The clinician rates babbling, non-communicative vocalisation
and guttural sounds (assessment sheet E3, E4, E6) 高.

So the label is never read. Both arms go through the same three steps:

1. **Voiced audio** -- frame energy above the session's own noise floor, the detector
   source normalization already uses to place turn edges.
2. **Untranscribed** -- voiced audio outside every transcribed turn. Whatever the
   normalizer dropped ([UNINTELLIGIBLE], third parties, overlaps) is therefore uncovered,
   exactly like the sounds the TD transcribers never wrote down.
3. **Whose voice** -- the ECAPA embedding of each untranscribed segment against the
   child's and the examiner's voiceprints, enrolled from their own transcribed speech in
   the same session. The decision rule is calibrated on TD alone (`calibrate`) and
   frozen in pipeline.yaml before any ASD session is scored.

Features (E stream, sheet E5 and the low-verbal additions of 2026-10):

* ``e_vocal_all_seconds_per_min`` -- (transcribed + attributed untranscribed) child seconds
  per minute of session;
* ``e_vocal_untranscribed_share`` -- attributed untranscribed / all child vocal seconds;
* ``e_vocal_response_rate`` -- share of examiner questions followed by any child
  vocalisation, words or not, within ``response_window_s``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..contracts import Speaker, Turn
from .common import frame_energy, noise_floor_energy

__all__ = ["voiced_intervals", "subtract", "merge_close", "untranscribed_segments",
           "centroid", "attribute", "vocal_features", "VocalResult", "calibrate"]

_FRAME = 512
Interval = tuple[float, float]


# ------------------------------------------------------------------------------ intervals
def voiced_intervals(audio: np.ndarray, sr: int, over_floor_db: float,
                     min_run_s: float) -> list[Interval]:
    """Stretches whose frame energy stays above the session noise floor for min_run_s."""
    e = frame_energy(audio, _FRAME)
    if e.size == 0:
        return []
    hop = _FRAME / sr
    on = e > noise_floor_energy(audio) * 10.0 ** (over_floor_db / 10.0)
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= max(1, int(round(min_run_s / hop)))
    return [(float(a * hop), float(b * hop)) for a, b in zip(starts[keep], ends[keep])]


def subtract(intervals: Sequence[Interval], covered: Sequence[Interval]) -> list[Interval]:
    """The parts of ``intervals`` that lie outside every ``covered`` interval."""
    cov = sorted(covered)
    starts = [c[0] for c in cov]
    out: list[Interval] = []
    for a, b in intervals:
        pieces = [(a, b)]
        # only covered intervals that can overlap [a, b]
        k = max(0, bisect.bisect_right(starts, a) - 1)
        while k < len(cov) and cov[k][0] < b:
            c0, c1 = cov[k]
            nxt = []
            for p0, p1 in pieces:
                if c1 <= p0 or c0 >= p1:
                    nxt.append((p0, p1))
                    continue
                if c0 > p0:
                    nxt.append((p0, c0))
                if c1 < p1:
                    nxt.append((c1, p1))
            pieces = nxt
            k += 1
        out += pieces
    return out


def merge_close(segments: Sequence[Interval], gap_s: float) -> list[Interval]:
    out: list[Interval] = []
    for a, b in sorted(segments):
        if out and a - out[-1][1] <= gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def untranscribed_segments(audio: np.ndarray, sr: int, turns: Sequence[Turn],
                           v_cfg: Mapping) -> list[Interval]:
    """Voiced audio inside the transcript's time span that no transcribed turn covers."""
    if not turns:
        return []
    span0 = min(t.t_start for t in turns)
    span1 = max(t.t_end for t in turns)
    covered = [(t.t_start, t.t_end) for t in turns]
    runs = [(max(a, span0), min(b, span1))
            for a, b in voiced_intervals(audio, sr, float(v_cfg["voice_over_floor_db"]),
                                         float(v_cfg["voice_min_run_s"]))
            if b > span0 and a < span1]
    free = merge_close(subtract(runs, covered), float(v_cfg["merge_gap_s"]))
    free = subtract(free, covered)             # merging must not bridge a transcribed turn
    return [(a, b) for a, b in free if b - a >= float(v_cfg["min_segment_s"])]


# ------------------------------------------------------------------------------ voiceprints
def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def centroid(embeddings: np.ndarray, min_windows: int) -> np.ndarray | None:
    """Unit-length mean of unit-length embeddings, or None with too few to trust."""
    if len(embeddings) < max(1, min_windows):
        return None
    return _unit(_unit(embeddings).mean(axis=0))


def attribute(seg_emb: np.ndarray, child_c: np.ndarray, exam_c: np.ndarray,
              margin: float, floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(is_child, similarity to child, similarity to examiner) for each segment.

    A segment is the child's when it is clearly closer to the child's voiceprint than to
    the examiner's (difference > margin) and close enough to the child's in absolute
    terms (>= floor) -- the second test turns away toys, chairs and other people.
    """
    z = _unit(seg_emb)
    s_child, s_exam = z @ child_c, z @ exam_c
    return (s_child - s_exam > margin) & (s_child >= floor), s_child, s_exam


# ------------------------------------------------------------------------------ features
@dataclass
class VocalResult:
    scalars: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)


def vocal_features(turns: Sequence[Turn], segments: Sequence[Interval],
                   is_child: Sequence[bool] | None, v_cfg: Mapping, min_questions: int
                   ) -> VocalResult:
    """``is_child`` None means the voiceprints could not be enrolled: unmeasurable."""
    nan = float("nan")
    ordered = sorted(turns, key=lambda t: t.t_start)
    kids = [t for t in ordered if t.speaker is Speaker.CHILD]
    quality = {"vocal_n_untranscribed_segments": float(len(segments)),
               "vocal_attribution_available": float(is_child is not None)}
    if is_child is None or not ordered:
        return VocalResult({"e_vocal_all_seconds_per_min": nan,
                            "e_vocal_untranscribed_share": nan,
                            "e_vocal_response_rate": nan}, quality)
    mine = [s for s, c in zip(segments, is_child) if c]
    child_s = sum(t.duration for t in kids)
    unt_s = sum(b - a for a, b in mine)
    minutes = (ordered[-1].t_end - ordered[0].t_start) / 60.0
    total = child_s + unt_s
    quality["vocal_untranscribed_child_s"] = float(unt_s)

    # a response is any child vocal onset after the question ends, inside the window and
    # before the examiner speaks again
    window = float(v_cfg["response_window_s"])
    onsets = sorted([t.t_start for t in kids] + [a for a, _ in mine])
    ex_starts = [t.t_start for t in ordered if t.speaker is Speaker.EXAMINER]
    questions = [t for t in ordered if t.speaker is Speaker.EXAMINER and t.is_question]
    answered = 0
    for q in questions:
        k = bisect.bisect_right(ex_starts, q.t_start)
        nxt = ex_starts[k] if k < len(ex_starts) else float("inf")
        lo, hi = q.t_end - 0.1, min(q.t_end + window, nxt)
        i = bisect.bisect_left(onsets, lo)
        answered += i < len(onsets) and onsets[i] <= hi
    quality["vocal_n_questions"] = float(len(questions))
    return VocalResult({
        "e_vocal_all_seconds_per_min": total / minutes if minutes > 0 else nan,
        "e_vocal_untranscribed_share": unt_s / total if total > 0 else nan,
        "e_vocal_response_rate": (answered / len(questions)
                                  if len(questions) >= min_questions else nan),
    }, quality)


# ------------------------------------------------------------------------------ calibration
def calibrate(sessions: Sequence[tuple[np.ndarray, int, Sequence[Turn]]],
              windows_of: Callable[[np.ndarray, int, Sequence[Turn], Speaker], np.ndarray],
              embed_windows: Callable[[np.ndarray], np.ndarray],
              embed_segments: Callable[[Sequence[np.ndarray]], np.ndarray],
              v_cfg: Mapping, min_windows: int, seed: int = 0,
              child_budgets_s: Sequence[float | None] = (None, 30.0, 12.0, 6.0)) -> dict:
    """Fix the attribution rule on TD sessions, where who spoke is known.

    In each session the transcribed turns of each speaker are split alternately: one half
    enrolls the voiceprints, the other half is cut into pieces as short as the
    untranscribed segments really are, and those pieces are attributed. The rule is then

    * margin = 95th percentile of (sim_child - sim_exam) over **examiner** pieces, so that
      at most 5% of examiner speech would be taken for the child;
    * floor = 5th percentile of sim_child over **child** pieces.

    ``child_budgets_s`` re-runs the same test with the child's voiceprint enrolled from
    only that many seconds of speech, as for a child who says very little; the rule from
    full enrolment is applied unchanged, so the table shows what it costs them.
    """
    rng = np.random.default_rng(seed)
    lo_s, hi_s = float(v_cfg["min_segment_s"]), 2.0
    rows: dict[object, dict[str, list]] = {b: {"child_diff": [], "child_sim": [], "exam_diff": [],
                                               "exam_sim": [], "child_dur": []}
                                           for b in child_budgets_s}
    for audio, sr, turns in sessions:
        ordered = sorted(turns, key=lambda t: t.t_start)
        by = {sp: [t for t in ordered if t.speaker is sp] for sp in (Speaker.CHILD, Speaker.EXAMINER)}
        enroll = {sp: v[0::2] for sp, v in by.items()}
        test = {sp: v[1::2] for sp, v in by.items()}
        exam_c = centroid(embed_windows(windows_of(audio, sr, enroll[Speaker.EXAMINER],
                                                   Speaker.EXAMINER)), min_windows)
        if exam_c is None:
            continue
        pieces = {}
        for sp in (Speaker.CHILD, Speaker.EXAMINER):
            clips, durs = [], []
            for t in test[sp]:
                d = float(rng.uniform(lo_s, hi_s))
                if t.duration < d:
                    d = t.duration
                if d < lo_s:
                    continue
                a = t.t_start + float(rng.uniform(0.0, t.duration - d))
                clips.append(np.asarray(audio[int(a * sr):int((a + d) * sr)], dtype=np.float32))
                durs.append(d)
            pieces[sp] = (embed_segments(clips) if clips else np.zeros((0, 192)), durs)
        for budget in child_budgets_s:
            kids = enroll[Speaker.CHILD]
            if budget is not None:
                kept, acc = [], 0.0
                for t in kids:
                    if acc >= budget:
                        break
                    kept.append(t)
                    acc += t.duration
                kids = kept
            child_c = centroid(embed_windows(windows_of(audio, sr, kids, Speaker.CHILD)), min_windows)
            if child_c is None:
                continue
            ce, cd = pieces[Speaker.CHILD]
            ee, _ = pieces[Speaker.EXAMINER]
            if len(ce):
                zc = _unit(ce)
                rows[budget]["child_diff"] += list(zc @ child_c - zc @ exam_c)
                rows[budget]["child_sim"] += list(zc @ child_c)
                rows[budget]["child_dur"] += list(cd)
            if len(ee):
                ze = _unit(ee)
                rows[budget]["exam_diff"] += list(ze @ child_c - ze @ exam_c)
                rows[budget]["exam_sim"] += list(ze @ child_c)
    full = rows[child_budgets_s[0]]
    margin = float(np.percentile(full["exam_diff"], 95))
    floor = float(np.percentile(full["child_sim"], 5))
    table = {}
    for budget, r in rows.items():
        cdiff, csim = np.asarray(r["child_diff"]), np.asarray(r["child_sim"])
        edf, esim = np.asarray(r["exam_diff"]), np.asarray(r["exam_sim"])
        cdur = np.asarray(r["child_dur"])
        hit = (cdiff > margin) & (csim >= floor)
        false_accept = (edf > margin) & (esim >= floor)
        bins = {}
        for lo, hi in ((lo_s, 0.5), (0.5, 1.0), (1.0, hi_s + 1e-9)):
            m = (cdur >= lo) & (cdur < hi)
            bins[f"{lo:.1f}-{min(hi, hi_s):.1f}s"] = (float(hit[m].mean()) if m.any() else float("nan"),
                                                      int(m.sum()))
        table[budget] = {"n_child": int(len(cdiff)), "n_exam": int(len(edf)),
                         "child_recall": float(hit.mean()) if len(hit) else float("nan"),
                         "exam_false_accept": float(false_accept.mean()) if len(edf) else float("nan"),
                         "recall_by_duration": bins}
    return {"margin": margin, "floor": floor, "table": table}
