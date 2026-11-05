"""C stream -- tone-normalised prosody (plan doc 5.4). The Mandarin-specific core.

All of the plan doc's C-stream features are computed **in parallel** -- "core / upgrading /
auxiliary" is an evidence ranking, not a processing order (see the terminology table in
CLAUDE.md).

Core (rhythm and duration, evidence A-):
    nPVI, %V, dC, syllable-duration distribution, pause distribution, speaking rate and
    articulation rate, utterance duration.
Auxiliary (tone realisation, evidence B/C):
    tone category accuracy, tone consistency, sandhi and neutral-tone realisation,
    residual F0 after the tone contour is stripped.

Two things the plan doc is emphatic about and that are enforced here:

* the F0 search range is the child band, 100-600 Hz -- never the adult default
  (section 9.2.1 / audit P1-3). The caller passes the tracker output; ``extract_c``
  asserts the configured range;
* the tone template is fitted **per subject and leave-one-out**, never pooled across
  subjects and never using diagnostic labels. See DEVIATIONS D3 for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..contracts import SyllableSpan, Turn
from .common import child_turns, min_values, npvi, pool, stat

__all__ = ["CStreamFeatures", "TONE_CLASSES", "CONTOUR_POINTS", "extract_c"]

TONE_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 4)
CONTOUR_POINTS = 5
# 3 duration channels + the tone one-hot. The three residual-F0 channels were dropped on
# 2026-09-12: a "residual" is what is left after the tone template is subtracted, and the
# template is the thing that failed -- every tracker, every normalisation and the rime-only
# window all land below the majority-class baseline. R5 pins the TCN's channels, layers and
# dilations, not its input width, so this is not an R5 change; `syllable_feature_dim` came
# from QUESTIONS Q16. The tone one-hot stays: it is the dictionary's citation tone, a label
# rather than a measurement, and duration genuinely depends on it.
SEQ_DIM = 3 + len(TONE_CLASSES)     # 8, matches configs/pipeline.yaml syllable_feature_dim
_VOICED_PERIODICITY = 0.40
_MIN_FRAMES_PER_SYLLABLE = 3


@dataclass
class CStreamFeatures:
    sequence: np.ndarray = field(default_factory=lambda: np.zeros((0, SEQ_DIM), dtype=np.float32))
    scalars: dict[str, float] = field(default_factory=dict)
    residual_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tone_true: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    tone_pred: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    quality: dict[str, float] = field(default_factory=dict)

    def pooled(self, functionals: Sequence[str]) -> dict[str, float]:
        """Rhythm and duration only. The tone-realisation scalars are gone (2026-09-12).

        ``residual_mean``, ``tone_true`` and ``tone_pred`` are still populated so audit
        P1-7 and the reference report can show *why* they were dropped, but nothing here
        reaches the feature vector.
        """
        return dict(self.scalars)


def _resample(v: np.ndarray, k: int = CONTOUR_POINTS) -> np.ndarray:
    if v.size == 0:
        return np.full(k, np.nan)
    if v.size == 1:
        return np.full(k, float(v[0]))
    return np.interp(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, v.size), v)


def _loo_templates(contours: np.ndarray, tones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-tone mean contour, plus the leave-one-out mean for each syllable.

    LOO matters: classifying a syllable against a template that contains that same
    syllable inflates tone-category accuracy, especially when a tone is rare.
    """
    n = len(contours)
    sums = {t: np.zeros(CONTOUR_POINTS) for t in TONE_CLASSES}
    counts = {t: 0 for t in TONE_CLASSES}
    for c, t in zip(contours, tones):
        sums[int(t)] += c
        counts[int(t)] += 1
    templates = np.stack([
        sums[t] / counts[t] if counts[t] else np.zeros(CONTOUR_POINTS) for t in TONE_CLASSES
    ])
    loo = np.zeros((n, len(TONE_CLASSES), CONTOUR_POINTS))
    for i in range(n):
        ti = int(tones[i])
        for j, t in enumerate(TONE_CLASSES):
            if t == ti and counts[t] > 1:
                loo[i, j] = (sums[t] - contours[i]) / (counts[t] - 1)
            elif counts[t]:
                loo[i, j] = sums[t] / counts[t]
            else:
                loo[i, j] = np.nan
    return templates, loo


def extract_c(
    turns: Sequence[Turn],
    syllables_by_turn: Mapping[int, Sequence[SyllableSpan]],
    f0: np.ndarray,
    periodicity: np.ndarray,
    hop_s: float,
    cfg: Mapping,
) -> CStreamFeatures:
    c_cfg = cfg["features"]["c_stream"]
    f0_min, f0_max = float(c_cfg["f0_min_hz"]), float(c_cfg["f0_max_hz"])
    if not (abs(f0_min - 100.0) < 1e-6 and abs(f0_max - 600.0) < 1e-6):
        raise ValueError(
            "audit P1-3 / section 9.2.1: the F0 search range must be the 100-600 Hz child "
            f"band, got {f0_min}-{f0_max}"
        )
    pause_thr = float(c_cfg["pause_threshold_s"])
    if c_cfg.get("tone_template") != "per_subject":
        raise NotImplementedError(
            "only tone_template=per_subject is wired up; the fold-internal variant is "
            "described in DEVIATIONS D3 and must be computed inside a FoldScope"
        )

    kids = child_turns(turns)
    voiced = periodicity >= _VOICED_PERIODICITY
    f0_valid = voiced & np.isfinite(f0) & (f0 >= f0_min) & (f0 <= f0_max)

    # ---- per-subject F0 reference: median of the child's own voiced frames ----
    child_frames: list[np.ndarray] = []
    for t in kids:
        a = int(np.floor(t.t_start / hop_s))
        b = int(np.ceil(t.t_end / hop_s))
        idx = np.arange(max(0, a), min(len(f0), b))
        child_frames.append(idx[f0_valid[idx]] if idx.size else idx)
    all_voiced = np.concatenate(child_frames) if child_frames else np.zeros(0, dtype=int)
    ref_f0 = float(np.median(f0[all_voiced])) if all_voiced.size else 200.0

    # ---- per-syllable measurements ----
    contours: list[np.ndarray] = []
    tones: list[int] = []
    underlying: list[int] = []
    durations: list[float] = []
    utt_id: list[int] = []
    n_unvoiced_syllables = 0

    per_utt_durations: list[list[float]] = []
    pauses: list[float] = []
    pauses_alt: dict[float, list[float]] = {p: [] for p in c_cfg.get("pause_sensitivity_s", [])}
    utt_durations: list[float] = []
    speaking_rates: list[float] = []
    articulation_rates: list[float] = []
    voiced_time = 0.0
    speech_time = 0.0
    unvoiced_runs: list[float] = []

    lo_s, hi_s = (float(x) for x in c_cfg.get("syllable_duration_bounds", (0.05, 1.0)))
    n_seen = n_dropped = 0
    vocalic_s = 0.0
    consonantal: list[float] = []

    # Utterances are bounded by their first and last syllable and split wherever two
    # syllables are more than `utterance_gap_s` apart -- never by the transcript's turn
    # boundaries. Those boundaries encode each pipeline's segmentation habits: merging
    # same-speaker segments at 1.5 s moved utterance length and pause rate by 1.4-1.6 SD on
    # TD (health check 0930 P3), and a turn's span also carries leading and trailing silence
    # into the speaking rate (P21). The gap equals the normalization merge gap, so an
    # utterance here is exactly what RuleStitch would have called one.
    utt_gap_s = float(c_cfg.get("utterance_gap_s", 1.5))
    u = -1
    for t in kids:
        raw_syls = sorted(syllables_by_turn.get(t.turn_id, ()), key=lambda x: x.t_start)
        n_seen += len(raw_syls)
        # A "syllable" of several seconds is a forced-alignment artefact, not a syllable:
        # when a turn's transcript is far shorter than its audio the aligner stretches one
        # vowel to fill (measured: single `o` phones spanning 4-8.5 s). Left in, they push
        # the duration CV above 2 and corrupt every rhythm statistic.
        syls = [s for s in raw_syls if lo_s <= s.duration <= hi_s]
        n_dropped += len(raw_syls) - len(syls)
        if not syls:
            continue
        groups: list[list[int]] = [[0]]
        for i in range(1, len(syls)):
            if syls[i].t_start - syls[i - 1].t_end > utt_gap_s:
                groups.append([i])
            else:
                groups[-1].append(i)
        utt_of = [0] * len(syls)
        for g in groups:
            u += 1
            utt = [syls[i] for i in g]
            for i in g:
                utt_of[i] = u
            u_dur = utt[-1].t_end - utt[0].t_start
            per_utt_durations.append([x.duration for x in utt])
            utt_durations.append(u_dur)
            speaking_rates.append(len(utt) / max(u_dur, 1e-6))

            utt_pause = 0.0
            for a, b in zip(utt[:-1], utt[1:]):
                gap = b.t_start - a.t_end
                if gap > pause_thr:
                    pauses.append(gap)
                    utt_pause += gap
                for p in pauses_alt:
                    if gap > p:
                        pauses_alt[p].append(gap)
            # Articulation rate is syllables per second of *speech*, i.e. the utterance
            # minus its pauses -- not syllables over the summed syllable spans, which drops
            # every sub-threshold gap and inflated the rate to about 10 syllables/s.
            articulation_rates.append(len(utt) / max(u_dur - utt_pause, 1e-6))

        for k, s in enumerate(syls):
            durations.append(s.duration)
            tones.append(int(s.tone))
            underlying.append(int(s.underlying_tone if s.underlying_tone is not None else s.tone))
            utt_id.append(utt_of[k])
            a = int(np.floor(s.t_start / hop_s))
            b = int(np.ceil(s.t_end / hop_s))
            idx = np.arange(max(0, a), min(len(f0), b))
            ok = idx[f0_valid[idx]] if idx.size else idx
            speech_time += s.duration
            voiced_time += len(ok) * hop_s
            if s.vocalic_s is not None:
                vocalic_s += s.vocalic_s
                consonantal.append(max(s.duration - s.vocalic_s, 0.0))
            if idx.size:
                run = 0.0
                for i in idx:
                    if f0_valid[i]:
                        if run > 0:
                            unvoiced_runs.append(run)
                        run = 0.0
                    else:
                        run += hop_s
                if run > 0:
                    unvoiced_runs.append(run)
            if len(ok) < _MIN_FRAMES_PER_SYLLABLE:
                n_unvoiced_syllables += 1
                contours.append(np.full(CONTOUR_POINTS, np.nan))
                continue
            semitones = 12.0 * np.log2(f0[ok] / ref_f0)
            contours.append(_resample(semitones))

    n_syl = len(durations)
    if n_syl == 0:
        return CStreamFeatures(quality={"c_n_syllables": 0.0, "c_usable": 0.0})

    contour_arr = np.stack(contours)
    tone_arr = np.asarray(tones, dtype=int)
    under_arr = np.asarray(underlying, dtype=int)
    dur_arr = np.asarray(durations, dtype=float)
    good = np.isfinite(contour_arr).all(axis=1)

    templates, loo = _loo_templates(contour_arr[good], tone_arr[good])
    # scatter LOO results back onto the full syllable index
    resid_mean = np.full(n_syl, np.nan)
    resid_slope = np.full(n_syl, np.nan)
    resid_sd = np.full(n_syl, np.nan)
    tone_pred = np.full(n_syl, -1, dtype=int)
    consistency: list[float] = []
    x_axis = np.linspace(0.0, 1.0, CONTOUR_POINTS)

    gi = 0
    for i in range(n_syl):
        if not good[i]:
            continue
        own = loo[gi, TONE_CLASSES.index(tone_arr[i])]
        r = contour_arr[i] - own
        resid_mean[i] = float(np.mean(r))
        resid_sd[i] = float(np.std(r))
        resid_slope[i] = float(np.polyfit(x_axis, r, 1)[0])
        dists = np.array([
            np.nan if not np.isfinite(loo[gi, j]).all()
            else float(np.mean((contour_arr[i] - loo[gi, j]) ** 2))
            for j in range(len(TONE_CLASSES))
        ])
        if np.isfinite(dists).any():
            tone_pred[i] = TONE_CLASSES[int(np.nanargmin(dists))]
        consistency.append(float(np.sqrt(np.mean(r ** 2))))
        gi += 1

    # ---- scalars ----
    # P12: too little data gives NaN, never 0. Rates keep real zeros once there is speech.
    mv = min_values(cfg)
    nan = float("nan")
    n_utt = len(per_utt_durations)
    scored = tone_pred >= 0
    tone_acc = float(np.mean(tone_pred[scored] == tone_arr[scored])) if scored.any() else np.nan

    sandhi_pairs, sandhi_realised = 0, 0
    for i in range(n_syl - 1):
        if under_arr[i] == 3 and under_arr[i + 1] == 3 and utt_id[i] == utt_id[i + 1]:
            sandhi_pairs += 1
            if tone_arr[i] != 3:
                sandhi_realised += 1
    neutral = dur_arr[tone_arr == 0]
    lexical = dur_arr[tone_arr != 0]
    neutral_ratio = (float(np.mean(neutral) / np.mean(lexical))
                     if neutral.size and lexical.size else np.nan)

    npvis = [npvi(d) for d in per_utt_durations if len(d) >= 2]
    total_child_time = float(sum(t.duration for t in kids)) or 1e-6

    scalars: dict[str, float] = {
        "c_npvi_duration": stat(npvis, np.mean, mv),
        # %V and dC use the aligner's nucleus/coda split when it is available (Ramus et al.);
        # the voicing-based proxy is the fallback for the synthetic path, and it runs high
        # because voiced consonants count as voiced but are not vocalic.
        "c_pct_v": ((float(vocalic_s / speech_time) if consonantal
                     else float(voiced_time / speech_time))
                    if n_syl >= mv and speech_time > 0 else nan),
        "c_delta_c": (stat(consonantal, np.std, mv) if consonantal
                      else stat(unvoiced_runs, np.std, mv)),
        "c_syl_dur_mean": stat(dur_arr, np.mean, mv),
        "c_syl_dur_cv": stat(dur_arr, lambda v: np.std(v) / max(np.mean(v), 1e-9), mv),
        "c_syl_dur_p90": stat(dur_arr, lambda v: np.percentile(v, 90), mv),
        "c_pause_rate_per_min": (float(60.0 * len(pauses) / total_child_time)
                                 if n_utt >= mv else nan),
        "c_pause_mean_s": stat(pauses, np.mean, mv),
        "c_pause_p90_s": stat(pauses, lambda v: np.percentile(v, 90), mv),
        "c_speaking_rate": stat(speaking_rates, np.mean, mv),
        "c_articulation_rate": stat(articulation_rates, np.mean, mv),
        "c_utt_dur_mean": stat(utt_durations, np.mean, mv),
        "c_utt_dur_std": stat(utt_durations, np.std, mv),
        # Tone *duration* survives the cull. It reads the dictionary's tone label, which was
        # verified against 媽/麻/馬/罵 and 八/拔/把/爸, and never touches F0 -- so the failure
        # that killed the contour features does not reach it. Assessment item D5 輕聲處理不當,
        # rated 高. Audit P1-7 7D asks for a sample count first, hence c_n_neutral below.
        "c_neutral_tone_dur_ratio": (float(neutral_ratio)
                                     if neutral.size >= mv and lexical.size >= mv else nan),
    }
    # Pause-threshold sensitivity runs: reported, not features (health check P20).
    pause_sensitivity = {
        f"c_pause_rate_per_min_thr{int(p * 1000)}ms":
            float(60.0 * len(lst) / total_child_time) if n_utt >= mv else nan
        for p, lst in pauses_alt.items()
    }

    # ---- per-syllable sequence tensor for the M1 TCN ----
    z_dur = (dur_arr - dur_arr.mean()) / max(dur_arr.std(), 1e-9)
    seq = np.zeros((n_syl, SEQ_DIM), dtype=np.float32)
    seq[:, 0] = dur_arr
    seq[:, 1] = np.log(np.maximum(dur_arr, 1e-4))
    seq[:, 2] = z_dur
    for j, t in enumerate(TONE_CLASSES):
        seq[:, 3 + j] = (tone_arr == t).astype(np.float32)

    quality = {
        **pause_sensitivity,
        "c_n_syllables": float(n_syl),
        "c_implausible_syllable_ratio": float(n_dropped / max(n_seen, 1)),
        "c_pct_v_from_alignment": float(bool(consonantal)),
        "c_n_utterances": float(len(per_utt_durations)),
        "c_unvoiced_syllable_ratio": float(n_unvoiced_syllables / n_syl),
        "c_n_neutral_syllables": float(int((tone_arr == 0).sum())),   # audit P1-7 7D
        # Diagnostics, not features. Sandhi realisation (D4 三聲變調, rated 高) needs the
        # citation tone as well as the produced one; MFA supplies only the latter, so on
        # real data this is identically 0 -- a constant masquerading as a measurement. It
        # is kept here rather than deleted because the logic is correct and tested, and the
        # feature becomes available the moment dictionary lookup fills `underlying_tone`.
        "c_sandhi_realisation": float(sandhi_realised / sandhi_pairs) if sandhi_pairs else 0.0,
        "c_n_sandhi_pairs": float(sandhi_pairs),
        "c_tone_accuracy": float(tone_acc) if np.isfinite(tone_acc) else 0.0,
        "c_f0_tracking_failure_ratio": float(1.0 - good.mean()),   # audit: group difference here leaks
        "c_ref_f0_hz": ref_f0,
        "c_usable": float(good.sum() >= 20),
    }
    return CStreamFeatures(
        sequence=seq,
        scalars=scalars,
        residual_mean=resid_mean[np.isfinite(resid_mean)],
        tone_true=tone_arr,
        tone_pred=tone_pred,
        quality=quality,
    )
