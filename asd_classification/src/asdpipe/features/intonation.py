"""Utterance-level intonation -- assessment-sheet items A2, A3, B2, B5.

Why this module exists. The C stream's *syllable-level* tone features do not work on this
corpus: no F0 tracker beats the majority-class baseline (experiment A, five trackers),
removing utterance declination makes separability worse rather than better, and measuring
the rime only -- which does repair the template shapes -- still leaves every window below
baseline. Two independent lines of evidence say this is not a tuning problem:

* in spontaneous spoken Taiwan Mandarin, word identity dominates canonical tone pattern in
  explaining the F0 contour (arXiv:2503.23163), so tone category carries little variance to
  begin with;
* the autism literature reports that Mandarin-speaking autistic children *do* produce
  distinct tone contours and are mostly judged correct by adult listeners; the group
  difference is in **tone height and F0 range**, described as a suppression of F0 range.

So the clinically-relevant signal the assessment sheet asks about is not "did the child get
the tone right" but "how wide and how varied is this child's pitch". That is measured here.

The decisive practical advantage: **none of these features needs an accurate syllable
boundary.** They need a reliable F0 track and utterance boundaries, both of which this
corpus has (99.8% turn yield). They therefore side-step the pipeline's weakest link.

Assessment-sheet mapping (all rated 高 by the clinician, none previously implemented):

    A2 音調起伏幅度        -> c_pitch_range_st_*        pitch excursion per utterance
    A3 音調變化的多寡      -> c_pitch_sd_st_*, c_pitch_dynamism_*, c_pitch_reversals_*
    B2 語調過度誇張        -> c_exaggerated_utterance_ratio  the high tail of A2
    B5 情緒語調表達不足    -> c_intensity_range_db_*, c_pitch_intensity_coupling

**B1 and B3 were implemented and then removed** (PI decision, 2026-09-12), because
odd/even split-half reliability across the 19-subject cohort failed for both in all three
independent runs:

* B1 語調平板單調 (`c_flat_utterance_ratio`), r = 0.40 / 0.46 / 0.43. It counts a rare
  event -- a median of 9% of utterances fall under the cut-off -- so the estimate is noisy,
  and by construction it is only the low tail of A2, whose continuous measures are stable
  (r = 0.73-0.87). Note what was *not* done: the cut-off was not loosened until the feature
  passed. Moving a pre-registered threshold after watching it fail is the error already
  disclosed in DEVIATIONS D1.
* B3 句尾語調不對 (`c_final_slope_*`, `c_final_rise_ratio_question`), r = -0.10 to 0.37.
  The statement half was borderline (0.52-0.61) but the question half is unmeasurable on
  this corpus: children ask a median of 29 questions each (range 4-84), so a split half
  holds ~14, whose binomial sampling SD of 0.128 exceeds the 0.085 between-child SD. The
  measurement noise is 1.5x the signal. This is a sample-size limit of the ADOS setting,
  not something a parameter change can fix; roughly 130+ questions per child would be
  needed.

Red line R3. Every quantity here is computed from a single subject's own audio: the pitch
reference is that subject's median F0, and the exaggeration cut-off is an **absolute
constant declared in configs/pipeline.yaml**, never a quantile of the cohort. Nothing is
fitted across subjects, so this module is safe to run outside a FoldScope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..contracts import Turn
from .common import child_turns, min_values

__all__ = ["IntonationFeatures", "extract_intonation", "frame_db"]

_VOICED_PERIODICITY = 0.40
_MIN_VOICED_FRAMES = 10          # 100 ms of voicing at a 10 ms hop
_MIN_TURNS = 5                   # below this the subject-level pooling is meaningless


@dataclass
class IntonationFeatures:
    scalars: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)


def frame_db(audio: np.ndarray, sr: int, hop_s: float, win_s: float = 0.025) -> np.ndarray:
    """RMS level in dBFS on the same frame grid as the F0 track.

    Computed here rather than reused from the B stream because B5 needs the intensity
    *contour* inside an utterance, not one number per turn.
    """
    hop = max(1, int(round(hop_s * sr)))
    win = max(hop, int(round(win_s * sr)))
    n = int(np.ceil(len(audio) / hop))
    pad = np.pad(np.asarray(audio, dtype=np.float64), (win // 2, win), mode="constant")
    # strided view: (n, win) without copying the audio
    frames = np.lib.stride_tricks.as_strided(
        pad, shape=(n, win), strides=(pad.strides[0] * hop, pad.strides[0]), writeable=False
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def _reversals(st: np.ndarray) -> int:
    """Direction changes in a lightly smoothed pitch track.

    Smoothed over 5 frames (50 ms) first: without it this counts tracker jitter, which is
    a property of the recording rather than of the child.
    """
    if len(st) < 7:
        return 0
    k = 5
    sm = np.convolve(st, np.ones(k) / k, mode="valid")
    d = np.diff(sm)
    d = d[np.abs(d) > 1e-3]
    if len(d) < 2:
        return 0
    return int(np.sum(np.diff(np.sign(d)) != 0))


def extract_intonation(
    turns: Sequence[Turn],
    f0: np.ndarray,
    periodicity: np.ndarray,
    db: np.ndarray,
    hop_s: float,
    cfg: Mapping,
) -> IntonationFeatures:
    """Utterance-level pitch and loudness dynamics for one subject.

    ``db`` is the frame-wise level track from :func:`frame_db`, on the same grid as ``f0``.
    """
    c_cfg = cfg["features"]["c_stream"]
    f0_min, f0_max = float(c_cfg["f0_min_hz"]), float(c_cfg["f0_max_hz"])
    i_cfg = c_cfg.get("intonation", {})
    exag_thr = float(i_cfg.get("exaggerated_range_st", 12.0))

    kids = child_turns(turns)
    valid = (periodicity >= _VOICED_PERIODICITY) & np.isfinite(f0) & (f0 >= f0_min) & (f0 <= f0_max)

    # ---- pitch reference: this subject's own median voiced F0 (R3: within-subject only) ----
    per_turn_idx: list[tuple[Turn, np.ndarray]] = []
    for t in kids:
        a = max(0, int(np.floor(t.t_start / hop_s)))
        b = min(len(f0), int(np.ceil(t.t_end / hop_s)))
        if b <= a:
            continue
        idx = np.arange(a, b)
        per_turn_idx.append((t, idx))
    all_voiced = np.concatenate([i[valid[i]] for _, i in per_turn_idx]) if per_turn_idx else np.zeros(0, int)
    ref_f0 = float(np.median(f0[all_voiced])) if all_voiced.size else 200.0

    st_track = np.full(len(f0), np.nan)
    st_track[valid] = 12.0 * np.log2(f0[valid] / ref_f0)

    ranges, sds, dynamism, reversal_rates = [], [], [], []
    int_ranges, couplings = [], []
    n_used = 0

    for _, idx in per_turn_idx:
        v = idx[valid[idx]]
        if len(v) < _MIN_VOICED_FRAMES:
            continue
        n_used += 1
        st = st_track[v]
        tm = v * hop_s

        # A2 -- pitch excursion. P95-P5 rather than min-max: one octave-doubled frame
        # would otherwise dominate, and 3/19 subjects fail the F0 cross-check (audit P1-3).
        ranges.append(float(np.percentile(st, 95) - np.percentile(st, 5)))

        # A3 -- how much the pitch moves, in three complementary senses
        sds.append(float(np.std(st)))
        gaps = np.diff(tm)
        steps = np.abs(np.diff(st))
        keep = gaps < 3 * hop_s          # do not count jumps across unvoiced gaps as movement
        if keep.any():
            dynamism.append(float(np.sum(steps[keep]) / max(np.sum(gaps[keep]), 1e-6)))
        span = tm[-1] - tm[0]
        if span > 0.1:
            reversal_rates.append(_reversals(st) / span)

        # B5 -- expressive prosody is pitch and loudness moving together
        d = db[idx]
        d = d[np.isfinite(d)]
        if len(d) >= _MIN_VOICED_FRAMES:
            int_ranges.append(float(np.percentile(d, 95) - np.percentile(d, 5)))
        dv = db[v]
        if len(dv) >= _MIN_VOICED_FRAMES and np.std(dv) > 1e-6 and np.std(st) > 1e-6:
            couplings.append(float(np.corrcoef(st, dv)[0, 1]))

    mv = min_values(cfg)
    nan = float("nan")

    def _m(x, f=np.median):
        return float(f(x)) if len(x) >= mv else nan       # P12: NaN, never 0

    r = np.asarray(ranges)
    scalars = {
        # A2 音調起伏幅度
        "c_pitch_range_st_median": _m(r),
        "c_pitch_range_st_p90": float(np.percentile(r, 90)) if r.size >= mv else nan,
        "c_pitch_range_st_iqr": (float(np.subtract(*np.percentile(r, [75, 25])))
                                 if r.size >= mv else nan),
        # A3 音調變化的多寡
        "c_pitch_sd_st_median": _m(sds),
        "c_pitch_dynamism_st_per_s": _m(dynamism),
        "c_pitch_reversals_per_s": _m(reversal_rates),
        # B2 語調過度誇張 -- absolute cut-off, see module docstring
        "c_exaggerated_utterance_ratio": float(np.mean(r > exag_thr)) if r.size >= mv else nan,
        # B5 情緒語調表達不足
        "c_intensity_range_db_median": _m(int_ranges),
        "c_pitch_intensity_coupling": _m(couplings, np.mean),
    }
    quality = {
        "c_inton_n_turns": float(n_used),
        "c_inton_turn_yield": float(n_used / max(len(per_turn_idx), 1)),
        "c_inton_ref_f0_hz": ref_f0,
        "c_inton_usable": float(n_used >= _MIN_TURNS),
    }
    return IntonationFeatures(scalars=scalars, quality=quality)
