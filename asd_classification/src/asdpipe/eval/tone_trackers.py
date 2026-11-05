"""Experiment A -- does a shorter-window F0 tracker recover Mandarin tone?

Why this experiment exists. The C stream's tone features sit below the majority-class
baseline on real data, and the per-tone mean contours are compressed inside ~1.2 semitones
with T2 falling where it should rise. F0 *accuracy* is not the cause (torchcrepe agrees
with pYIN on 96% of frames cohort-wide). The remaining suspect is **temporal resolution**:
torchcrepe's model input is 1024 samples = 64 ms regardless of the hop, while the median
aligned syllable in this corpus is 115 ms. Each F0 value therefore averages over more than
half a syllable and bleeds into its neighbours.

So: hold the alignment, the normalisation and the classifier fixed, swap only the tracker,
and measure how well the four lexical tones separate.

Two metrics, reported side by side because they answer different questions:

* **separability** = (largest distance between two tone templates) / (mean within-tone SD).
  Above 1 means the tone means are further apart than the scatter around them. This is a
  property of the representation and does not depend on a classifier.
* **LOO accuracy vs the majority-class baseline.** Accuracy alone is misleading -- tones are
  not uniformly distributed, so always predicting T4 already scores about 0.30 here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..contracts import SyllableSpan, Turn
from ..features.c_prosody import (CONTOUR_POINTS, TONE_CLASSES, _loo_templates, _resample)

__all__ = ["TrackerResult", "TRACKERS", "evaluate_tracker", "compare_trackers"]

_MIN_FRAMES = 3
_LEXICAL = (1, 2, 3, 4)


@dataclass
class TrackerResult:
    name: str
    window_ms: float
    n_syllables: int = 0
    accuracy: float = float("nan")
    majority: float = float("nan")
    separability: float = float("nan")
    template_spread_st: float = float("nan")
    within_sd_st: float = float("nan")
    voiced_ratio: float = float("nan")
    seconds: float = 0.0
    templates: dict[int, list[float]] = field(default_factory=dict)

    @property
    def lift(self) -> float:
        return self.accuracy - self.majority


# ---------------------------------------------------------------- trackers
def _crepe(audio: np.ndarray, sr: int, fmin: float, fmax: float, hop_s: float, backends):
    return backends.track_f0(audio, sr, fmin, fmax, hop_s)


def _pyin(frame_length: int) -> Callable:
    def run(audio, sr, fmin, fmax, hop_s, backends):
        from ..features.backends import defuse_speechbrain_lazy_modules

        defuse_speechbrain_lazy_modules()
        import librosa

        f0, voiced, prob = librosa.pyin(
            audio.astype(np.float64), fmin=fmin, fmax=fmax, sr=sr,
            frame_length=frame_length, hop_length=max(1, int(hop_s * sr)),
        )
        conf = np.where(voiced, np.nan_to_num(prob, nan=0.0), 0.0)
        return np.nan_to_num(f0, nan=0.0), conf
    return run


def _praat(audio, sr, fmin, fmax, hop_s, backends):
    """Praat autocorrelation. Its window is 3 / pitch_floor = 30 ms at a 100 Hz floor."""
    import parselmouth

    snd = parselmouth.Sound(np.ascontiguousarray(audio, dtype=np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=hop_s, pitch_floor=fmin, pitch_ceiling=fmax)
    values = pitch.selected_array["frequency"]
    strength = pitch.selected_array["strength"]
    n = int(np.ceil(len(audio) / (hop_s * sr)))
    f0 = np.zeros(n)
    conf = np.zeros(n)
    m = min(n, len(values))
    f0[:m] = np.nan_to_num(values[:m])
    conf[:m] = np.nan_to_num(strength[:m])
    conf[f0[:n] <= 0] = 0.0
    return f0, conf


TRACKERS: dict[str, tuple[Callable, float]] = {
    "torchcrepe (現行)": (_crepe, 64.0),
    "pYIN 1024": (_pyin(1024), 64.0),
    "pYIN 512": (_pyin(512), 32.0),
    "pYIN 256": (_pyin(256), 16.0),
    "Praat autocorr": (_praat, 30.0),
}


# ---------------------------------------------------------------- evaluation
def evaluate_tracker(name: str, tracker: Callable, window_ms: float,
                     audio: np.ndarray, sr: int, spans_by_turn: Mapping[int, Sequence[SyllableSpan]],
                     turns: Sequence[Turn], cfg: Mapping, backends,
                     voiced_threshold: float = 0.40) -> TrackerResult:
    c_cfg = cfg["features"]["c_stream"]
    fmin, fmax = float(c_cfg["f0_min_hz"]), float(c_cfg["f0_max_hz"])
    hop_s = float(c_cfg["f0_hop_s"])
    lo_s, hi_s = (float(x) for x in c_cfg.get("syllable_duration_bounds", (0.05, 1.0)))

    t0 = time.time()
    f0, conf = tracker(audio, sr, fmin, fmax, hop_s, backends)
    seconds = time.time() - t0
    valid = (conf >= voiced_threshold) & np.isfinite(f0) & (f0 >= fmin) & (f0 <= fmax)

    idx_all = np.concatenate([
        np.arange(max(0, int(np.floor(s.t_start / hop_s))), min(len(f0), int(np.ceil(s.t_end / hop_s))))
        for v in spans_by_turn.values() for s in v
    ]) if spans_by_turn else np.zeros(0, dtype=int)
    ref = float(np.median(f0[idx_all[valid[idx_all]]])) if idx_all.size and valid[idx_all].any() else 200.0

    contours, tones = [], []
    n_seen = n_voiced = 0
    for v in spans_by_turn.values():
        for s in v:
            if not (lo_s <= s.duration <= hi_s):
                continue
            n_seen += 1
            a = int(np.floor(s.t_start / hop_s))
            b = int(np.ceil(s.t_end / hop_s))
            idx = np.arange(max(0, a), min(len(f0), b))
            ok = idx[valid[idx]] if idx.size else idx
            if len(ok) < _MIN_FRAMES:
                continue
            n_voiced += 1
            contours.append(_resample(12.0 * np.log2(f0[ok] / ref)))
            tones.append(int(s.tone))

    res = TrackerResult(name=name, window_ms=window_ms, seconds=seconds,
                        voiced_ratio=n_voiced / max(n_seen, 1))
    if len(contours) < 50:
        return res
    C = np.stack(contours)
    T = np.array(tones)
    res.n_syllables = len(T)

    # separability: template spread against within-tone scatter, lexical tones only
    means = {t: C[T == t].mean(axis=0) for t in _LEXICAL if (T == t).sum() >= 20}
    sds = [C[T == t].std(axis=0).mean() for t in _LEXICAL if (T == t).sum() >= 20]
    if len(means) >= 2 and sds:
        spread = max(np.linalg.norm(means[a] - means[b])
                     for a in means for b in means if a < b)
        within = float(np.mean(sds))
        res.template_spread_st = float(spread)
        res.within_sd_st = within
        res.separability = float(spread / within) if within > 0 else float("nan")
        res.templates = {t: [round(float(x), 2) for x in m] for t, m in means.items()}

    _, loo = _loo_templates(C, T)
    pred = np.full(len(T), -1)
    for i in range(len(T)):
        d = np.array([np.mean((C[i] - loo[i, j]) ** 2) if np.isfinite(loo[i, j]).all() else np.nan
                      for j in range(len(TONE_CLASSES))])
        if np.isfinite(d).any():
            pred[i] = TONE_CLASSES[int(np.nanargmin(d))]
    m = pred >= 0
    if m.any():
        res.accuracy = float((T[m] == pred[m]).mean())
        counts = np.bincount(T[m], minlength=5)
        res.majority = float(counts.max() / m.sum())
    return res


def compare_trackers(audio: np.ndarray, sr: int,
                     spans_by_turn: Mapping[int, Sequence[SyllableSpan]],
                     turns: Sequence[Turn], cfg: Mapping, backends,
                     names: Sequence[str] | None = None) -> list[TrackerResult]:
    chosen = names or list(TRACKERS)
    return [evaluate_tracker(n, *TRACKERS[n], audio, sr, spans_by_turn, turns, cfg, backends)
            for n in chosen]
