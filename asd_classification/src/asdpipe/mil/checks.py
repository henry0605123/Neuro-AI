"""The checks that run before any model is fitted (research plan 3.4).

They decide, on TD recordings and before a label is looked at, which of the ten measures may enter
the model. A measure that fails its check is left empty; nothing is found to replace it (plan 3.4).
The pass rules that ``configs/decision_rules.yaml`` (``mil_analysis``) writes down are arguments
here with its values as the defaults (``f0_session_pass``, ``f0_group_rule``), so a rule is stated
once and the caller passes what it read; the 0.70 that ``split_half_r`` must reach is compared by
the caller. The functions measure and decide; they read no file.

**Split-half reliability of B7 and A11** (``b7_a11_halves``, ``split_half_r``). A child's turns are
put in time order and dealt out alternately, the odd-numbered ones to one half and the
even-numbered ones to the other. The examiner's turns are dealt out the same way, so each half's
A11 sets the child's level against that half's examiner level. The turns are split, not what is
made from them, and that matters for B7: its windows are 3 s long and advance 1.5 s, so
neighbouring windows share half their audio, and a window list cut in two would put the same
sound into both halves and raise r by construction. B7 cuts its windows inside each half.

Each half is measured the way the real measure is (the same window length and minimum window count,
the same 0.1 s floor on a turn, the same minimum number of turns), so that r says something about
the number the model will be given. The statistic is Pearson r between the halves across children,
without the Spearman-Brown correction: the code behind the older split-half values was never kept,
and the uncorrected r is the conservative reading, lower than the corrected one and so harder to
pass.

**F0 three-way check** (``f0_three_way``, ``f0_session_pass``). CREPE, pYIN and Praat's
autocorrelation tracker follow the first 60 s of the child's speech, and CREPE is compared with each
of the other two. A session passes only if both pairs agree on more than 90% of the frames and make
an octave error on fewer than 3%. A pair is compared on the frames that both of its trackers call
voiced: the question is whether two trackers that each found a pitch found the same one. A frame
one of them gives up on is a tracking failure, not a disagreement about pitch, and the failure
rate has a rule of its own (``f0_group_rule``). The trackers do not share a clock (Praat's first
frame is where its window first fits), so pYIN and Praat are interpolated onto CREPE's frame
times, linearly between the two frames either side and only if both are voiced; a time beside an
unvoiced frame has no pitch to compare. The ratio windows are those of ``Backends.f0_cross_check``,
which this extends to a third tracker. That method returns aggregates only, so the frame-level
comparison is restated here, and a test holds the pYIN pair to that method's numbers. A pair with
fewer than 50 comparable frames has NaN for both values and the session does not pass.

**F0 group rule** (``f0_group_rule``). The one function here that reads labels. If the per-child
tracking-failure ratio alone tells ASD from TD, the pitch measures would partly report how well the
trackers coped with one group's recordings, so A2 and B2 leave the model and are only described.
It belongs to the frozen analysis and is not run during the TD-only checks.

**Convention gate for the segment measures** (``segment_gate``). It is run the way
``eval.convention_gate.run_gate`` runs the older features: the baseline is the raw TD corpus
normalised, and a variant is a convention transform applied to the raw corpus and then normalised,
so a transform that normalisation undoes shows no shift. A child is summarised by the mean of each
measure over that child's segments, because the child is the unit (R2) and the gate states a shift
in units of the SD across children.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from ..contracts import Corpus, SubjectRecord, Turn
from ..eval.convention_gate import TRANSFORMS, GateResult, apply_transform, gate_shifts
from ..eval.metrics import safe_auc
from ..features import common
from ..features.b_voice import BStreamFeatures, _level_rel_examiner, ecapa_windows
from ..features.backends import Backends, defuse_speechbrain_lazy_modules
from ..normalize import normalize_corpus
from .segments import SEGMENT_MEASURES, segment_measures

__all__ = ["SPLIT_HALF_MEASURES", "split_half_r", "b7_a11_halves", "f0_three_way",
           "f0_session_pass", "f0_group_rule", "segment_gate"]

# The two measures the plan asks a split-half r for, under the names they carry as per-child
# measures (``segments.CHILD_MEASURES``) and in ``BStreamFeatures``.
_B7, _A11 = "b7_mean_pairwise_cosdist", "b_level_rel_examiner_db"
SPLIT_HALF_MEASURES = (_B7, _A11)

_MIN_CHILDREN = 5                # split_half_r: with fewer children than this r is not reported
_MIN_TURN_S = 0.1                # as extract_b: a child turn shorter than this has no level
_MIN_COMPARABLE_FRAMES = 50      # a tracker pair compared on fewer frames has NaN values

# Restated from Backends.f0_cross_check (features/backends.py), which gives aggregates only and
# stops at the pYIN pair. Keep the two in step; a slow test compares their numbers.
_CREPE_MIN_PERIODICITY = 0.4                  # CREPE calls a frame voiced from this periodicity
_AGREE_RATIO = (0.95, 1.05)                   # CREPE / other inside this: the two agree
_OCTAVE_RATIOS = ((1.9, 2.1), (0.45, 0.55))   # an octave up, an octave down
_PYIN_FRAME_LENGTH = 1024                     # librosa.pyin frame length, as f0_cross_check

# A target time this close to a tracker's frame is that frame. Frame times come from different
# clocks and carry rounding error of the order of 1e-13 s, which must not turn a voiced frame
# into an interpolation between its neighbours; a frame is 10 ms, so 1 microsecond cannot blur two.
_CLOCK_TOL_S = 1e-6

# An AUC is a multiple of 1 / (2 * n_ASD * n_TD), so it can sit exactly on 0.30 or 0.70, and
# roc_auc_score may report that one float away: 14 ASD against 25 TD children with 245 of the 350
# pairs won comes back as 0.7000000000000001. The plan keeps the measures unless the AUC is
# *outside* 0.30-0.70, and a rounding error must not decide that. As ``mil.evaluate.TIE_TOL``
# reasons: far below the grid step, far above a few ulps.
_AUC_TOL = 1e-9


# ------------------------------------------------------------------------------ split-half
def split_half_r(odd: np.ndarray, even: np.ndarray) -> float:
    """Pearson r between the two halves across children, uncorrected (module docstring).

    ``odd[i]`` and ``even[i]`` are child i's values in the two halves. Only children with both
    count. With fewer than five of them, or with no spread at all in one half, r is not defined
    and NaN comes back: a missing reliability, not a reliability of 0.
    """
    a = np.asarray(odd, dtype=float)
    b = np.asarray(even, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"one value per child in each half, in the same order; "
                         f"got shapes {a.shape} and {b.shape}")
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < _MIN_CHILDREN:
        return float("nan")
    a, b = a[both], b[both]
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:       # exact: np.std of a constant is not always 0
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _dealt(turns: Sequence[Turn]) -> tuple[list[Turn], list[Turn]]:
    """One speaker's turns in time order, dealt out alternately: ``turns[0::2]`` (the odd-numbered
    ones, counting from 1) and ``turns[1::2]`` (the even-numbered ones)."""
    ordered = sorted(turns, key=lambda t: (t.t_start, t.t_end))
    return ordered[0::2], ordered[1::2]


def _half_b7(audio: np.ndarray, sr: int, kids: Sequence[Turn], cfg: Mapping,
             backends: Backends) -> float:
    """B7 of one half: windows cut over that half's child speech alone, then the usual b7()."""
    windows = ecapa_windows(audio, sr, kids, cfg)
    embeddings = (backends.embed_windows(windows) if len(windows)
                  else np.zeros((0, 192), dtype=np.float32))
    min_windows = int(cfg["features"]["b_stream"]["ecapa_min_windows"])
    return BStreamFeatures(embeddings=embeddings, min_windows=min_windows).b7()[_B7]


def _half_a11(audio: np.ndarray, sr: int, kids: Sequence[Turn], examiners: Sequence[Turn],
              cfg: Mapping) -> float:
    """A11 of one half, made the way ``extract_b`` and ``BStreamFeatures.pooled`` make it for a
    whole session: the child's turn levels (turns of at least 0.1 s) against the examiner's, and
    missing when there are fewer child turns than ``features.min_data`` asks of any statistic."""
    floor = int(_MIN_TURN_S * sr)
    levels = [common.rms_db(x) for x in (common.slice_turn(audio, sr, t) for t in kids)
              if x.size >= floor]
    if int(np.isfinite(levels).sum()) < common.min_values(cfg):
        return float("nan")
    return _level_rel_examiner(audio, sr, examiners, levels)


def b7_a11_halves(subject: SubjectRecord, cfg: Mapping, backends: Backends
                  ) -> dict[str, tuple[float, float]]:
    """B7 and A11 on the two interleaved halves of one session, as ``{measure: (odd, even)}``.

    The keys are ``SPLIT_HALF_MEASURES``. A half that cannot be measured (too few windows for B7,
    too few turns or no examiner speech for A11) is NaN, and ``split_half_r`` leaves that child out.
    """
    turns = subject.session.turns
    kid_halves = _dealt(common.child_turns(turns))
    examiner_halves = _dealt(common.examiner_turns(turns))
    audio, sr = subject.audio, subject.sample_rate
    b7 = tuple(_half_b7(audio, sr, kids, cfg, backends) for kids in kid_halves)
    a11 = tuple(_half_a11(audio, sr, kids, examiners, cfg)
                for kids, examiners in zip(kid_halves, examiner_halves))
    return {_B7: b7, _A11: a11}


# ------------------------------------------------------------------------------ F0 check
def _f0_at(times: np.ndarray, f0: np.ndarray, voiced: np.ndarray, target: np.ndarray
           ) -> np.ndarray:
    """One tracker's F0 at the ``target`` times, NaN wherever it has no pitch to offer there.

    Between two frames the value is interpolated linearly, but only when both are voiced: a time
    beside an unvoiced frame, or outside the track, gets NaN and not a value made up across the
    gap. A time within ``_CLOCK_TOL_S`` of a frame takes that frame's value.
    """
    t = np.asarray(times, dtype=float)
    y = np.where(np.asarray(voiced, dtype=bool), np.asarray(f0, dtype=float), np.nan)
    target = np.asarray(target, dtype=float)
    out = np.full(target.shape, np.nan)
    if t.size == 0:
        return out
    after = np.clip(np.searchsorted(t, target), 0, t.size - 1)    # first frame at or after
    before = np.maximum(after - 1, 0)
    nearest = np.where(np.abs(target - t[before]) < np.abs(target - t[after]), before, after)
    on_frame = np.abs(target - t[nearest]) <= _CLOCK_TOL_S
    out[on_frame] = y[nearest[on_frame]]
    between = ~on_frame & (target > t[0]) & (target < t[-1])
    hi = after[between]
    lo = hi - 1
    w = (target[between] - t[lo]) / (t[hi] - t[lo])
    out[between] = (1.0 - w) * y[lo] + w * y[hi]                  # NaN if either is unvoiced
    return out


def _ratio_rates(ratio: np.ndarray) -> tuple[float, float]:
    """Share of frames in agreement and share that are octave errors, from the CREPE / other
    ratio of each comparable frame. Both windows include their edges, as in ``f0_cross_check``."""
    lo, hi = _AGREE_RATIO
    octave = np.zeros(len(ratio), dtype=bool)
    for low, high in _OCTAVE_RATIOS:
        octave |= (ratio >= low) & (ratio <= high)
    return float(np.mean((ratio >= lo) & (ratio <= hi))), float(np.mean(octave))


def _pyin_track(x: np.ndarray, sr: int, fmin: float, fmax: float, hop: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    defuse_speechbrain_lazy_modules()    # librosa's lazy loader walks sys.modules; see backends.py
    import librosa

    f0, flag, _ = librosa.pyin(x.astype(np.float64), fmin=fmin, fmax=fmax, sr=sr,
                               frame_length=_PYIN_FRAME_LENGTH, hop_length=hop)
    return np.arange(len(f0)) * hop / sr, f0, flag & np.isfinite(f0)


def _praat_track(x: np.ndarray, sr: int, fmin: float, fmax: float, hop_s: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import parselmouth

    sound = parselmouth.Sound(np.ascontiguousarray(x, dtype=np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch_ac(time_step=hop_s, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array["frequency"]                         # 0 where unvoiced
    return pitch.xs(), f0, f0 > 0


def f0_three_way(audio: np.ndarray, sr: int, child_turns: Sequence[Turn], cfg: Mapping,
                 backends: Backends, max_seconds: float = 60.0) -> dict:
    """CREPE against pYIN and against Praat on the first ``max_seconds`` of the child's speech.

    ``audio`` is the whole session and ``child_turns`` the child's turns; their speech is joined
    in time order, as ``extract_corpus`` joins it for the older check. Returns ``agree_pyin`` and
    ``agree_praat`` (the share of comparable frames with CREPE / other within 5%), ``octave_pyin``
    and ``octave_praat`` (the share an octave apart), and ``n_frames``: the number of comparable
    frames of the weaker pair, so that ``n_frames < 50`` is exactly "some pair is NaN". Under a
    second of child speech nothing is tracked and every value is NaN, as in ``f0_cross_check``.
    """
    c_cfg = cfg["features"]["c_stream"]
    fmin, fmax = float(c_cfg["f0_min_hz"]), float(c_cfg["f0_max_hz"])
    hop_s = float(c_cfg["f0_hop_s"])
    hop = max(1, int(round(hop_s * sr)))               # CREPE's hop in samples, as Backends

    nan = float("nan")
    out: dict = {"agree_pyin": nan, "agree_praat": nan, "octave_pyin": nan, "octave_praat": nan,
                 "n_frames": 0}
    ordered = sorted(child_turns, key=lambda t: (t.t_start, t.t_end))
    if not ordered:
        return out
    x = np.concatenate([common.slice_turn(audio, sr, t) for t in ordered])[: int(max_seconds * sr)]
    if x.size < sr:
        return out
    x = np.ascontiguousarray(x, dtype=np.float32)

    f0_crepe, periodicity = backends.track_f0(x, sr, fmin, fmax, hop_s)
    crepe_times = np.arange(len(f0_crepe)) * hop / sr       # frame i is centred on sample i * hop
    crepe_voiced = (periodicity >= _CREPE_MIN_PERIODICITY) & np.isfinite(f0_crepe)

    comparable_frames = []
    for name, (times, f0, voiced) in (("pyin", _pyin_track(x, sr, fmin, fmax, hop)),
                                      ("praat", _praat_track(x, sr, fmin, fmax, hop_s))):
        other = _f0_at(times, f0, voiced, crepe_times)
        both = crepe_voiced & np.isfinite(other)
        comparable_frames.append(int(both.sum()))
        if comparable_frames[-1] >= _MIN_COMPARABLE_FRAMES:
            out[f"agree_{name}"], out[f"octave_{name}"] = _ratio_rates(f0_crepe[both] / other[both])
    out["n_frames"] = min(comparable_frames)
    return out


def f0_session_pass(check: Mapping[str, float], min_agreement: float = 0.90,
                    max_octave: float = 0.03) -> bool:
    """Whether a session's pitch tracks can be trusted: both agreements strictly above
    ``min_agreement`` and both octave-error rates strictly below ``max_octave`` (the plan's
    "> 90%" and "< 3%", so 90% and 3% themselves fail). A NaN, a pair that could not be compared,
    fails too."""
    return bool(check["agree_pyin"] > min_agreement and check["agree_praat"] > min_agreement
                and check["octave_pyin"] < max_octave and check["octave_praat"] < max_octave)


# ------------------------------------------------------------------------------ F0 group rule
def f0_group_rule(failure_ratio: np.ndarray, y: np.ndarray, lo: float = 0.30,
                  hi: float = 0.70) -> tuple[float, bool]:
    """``(auc, keep_pitch_measures)``: the AUC of each child's F0 tracking-failure ratio against
    the label, and whether it lies within ``lo`` to ``hi`` (edges included; the plan drops the
    measures when the AUC is *outside* the range).

    ``y`` is 1 for ASD and 0 for TD. A child whose ratio is missing is left out. Where no AUC
    exists (one group only) the rule has not been passed, and the measures are not kept.
    """
    ratio = np.asarray(failure_ratio, dtype=float)
    labels = np.asarray(y)
    if ratio.shape != labels.shape:
        raise ValueError(f"one failure ratio per child; got {ratio.shape} ratios and "
                         f"{labels.shape} labels")
    seen = np.isfinite(ratio)
    auc = safe_auc(labels[seen], ratio[seen])
    return float(auc), bool(lo - _AUC_TOL <= auc <= hi + _AUC_TOL)    # NaN is in no range


# ------------------------------------------------------------------------------ convention gate
def _child_means(rows: np.ndarray) -> np.ndarray:
    """The mean of each column over a child's segments, NaN skipped; NaN for a column that has
    no value in any segment."""
    seen = np.isfinite(rows)
    n = seen.sum(axis=0)
    total = np.where(seen, rows, 0.0).sum(axis=0)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def _segment_means(corpus: Corpus, cfg: Mapping,
                   words_of: Callable[[list[str]], list[list[str]]]) -> dict[str, np.ndarray]:
    """Measure name -> each child's mean over their segments, from the normalised corpus.

    ``words_of`` is asked once, for every distinct text of the corpus, and the word lists are keyed
    back to the turns that carry that text, so a sentence said fifty times is segmented once.
    """
    normalized, _ = normalize_corpus(corpus, cfg)
    length = int(cfg["mil"]["segment_length"])
    texts = sorted({t.text for s in normalized.subjects for t in s.session.turns if t.text})
    segmented = words_of(texts) if texts else []
    if len(segmented) != len(texts):
        raise ValueError(f"words_of must return one word list per text; got {len(segmented)} "
                         f"for {len(texts)} texts")
    words_by_text = dict(zip(texts, segmented))

    means = np.full((len(normalized.subjects), len(SEGMENT_MEASURES)), np.nan)
    for i, subject in enumerate(normalized.subjects):
        turns = list(subject.session.turns)
        words = {t.turn_id: words_by_text[t.text] for t in turns if t.text}
        means[i] = _child_means(segment_measures(turns, words, cfg, length))
    return {name: means[:, j] for j, name in enumerate(SEGMENT_MEASURES)}


def segment_gate(raw_td: Corpus, cfg: Mapping, rules: Mapping,
                 words_of: Callable[[list[str]], list[list[str]]]) -> list[GateResult]:
    """The convention gate (plan 3.4) for the six segment measures, on the raw TD corpus.

    ``rules`` is ``decision_rules.yaml`` (the transforms' parameters) and ``words_of`` the word
    segmenter, a list of texts in and one word list per text out (``Backends.segment`` in
    production; memoise it if segmenting is slow, since the four passes share most texts). No
    label is read.
    """
    base = _segment_means(raw_td, cfg, words_of)
    variants = {name: _segment_means(apply_transform(raw_td, name, rules), cfg, words_of)
                for name in TRANSFORMS}
    return gate_shifts(base, variants)
