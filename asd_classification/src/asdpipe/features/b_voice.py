"""B stream -- voice quality (plan doc 5.3, 5.0.5).

Contents, in the order the plan doc prioritises them:

* **CPP** (Praat/Parselmouth) -- the robust clarity measure the clinician rated "high";
* eGeMAPS v02 low-level descriptors from openSMILE -- jitter / shimmer / HNR / spectral
  tilt / loudness. Reported, never direction-assumed (plan doc 5.3: the sign of the
  jitter/shimmer effect differs by language and age, so the model decides, not us);
* **ECAPA within-speaker embedding variability** (B7) -- the cartoon-voice proxy.

Audit hooks implemented here: SNR floor before computing perturbation measures (P1-5),
fixed 3 s ECAPA windows with a minimum count (P0-F4), and a per-subject quality dict that
feeds the missing-data table (P0-4).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..contracts import Speaker, Turn
from .common import (child_turns, min_values, noise_floor_energy, pool, rms_db, slice_turn,
                     snr_db)

__all__ = ["BStreamFeatures", "LLD_CHANNELS", "extract_b", "cpp_of", "ecapa_windows"]

# Frozen channel list (audit P0-4). "_sma3nz" channels are only valid on voiced frames.
# Spectral balance and intensity. These are **not** perturbation measures, so audit P1-5's
# SNR gate does not apply to them and they are computed on every child turn. Keeping them
# behind that gate cost real coverage: the gate rejects a median of 43% of turns (up to 89%
# for one case), which is the whole recording, not just its noisy parts.
LLD_CHANNELS: tuple[str, ...] = (
    "Loudness_sma3",              # A11 音量整體偏大或偏小 (rated 高)
    "alphaRatio_sma3",            # A7 聲音清亮度 (rated 高)
    "hammarbergIndex_sma3",       # A7
    "slope0-500_sma3",            # A7
    "slope500-1500_sma3",         # A7
    "spectralFlux_sma3",
)

# Computed and reported, but deliberately **not** pooled into the feature vector. Audit
# P1-5 v3.0 is explicit: "~~jitter/shimmer 全套~~(v3.0 降級:醫師判「低」＋文獻不穩;
# 僅作補充報告,不入核心特徵)". The clinician rated A4 聲音沙啞 and A5 音量忽大忽小 低.
# These do need the SNR gate, being perturbation measures.
PERTURBATION_CHANNELS: tuple[str, ...] = (
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
)

ALL_LLD_CHANNELS: tuple[str, ...] = LLD_CHANNELS + PERTURBATION_CHANNELS
_VOICED_GATE = "F0semitoneFrom27.5Hz_sma3nz"


@dataclass
class BStreamFeatures:
    lld: dict[str, np.ndarray] = field(default_factory=dict)   # channel -> frame values
    cpp_per_turn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    turn_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    turn_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 192), dtype=np.float32))
    quality: dict[str, float] = field(default_factory=dict)
    level_rel_examiner_db: float = float("nan")
    min_n: int = 1                  # P12 floor for frame and turn statistics
    min_windows: int = 2            # B7, set from b_stream.ecapa_min_windows by attach()

    def attach(self, embeddings: np.ndarray, min_windows: int) -> "BStreamFeatures":
        """Fill in GPU-computed ECAPA embeddings after a CPU-only extraction pass."""
        self.embeddings = embeddings
        self.min_windows = int(min_windows)
        self.quality["b_ecapa_windows"] = float(len(embeddings))
        self.quality["b_ecapa_windows_ok"] = float(len(embeddings) >= min_windows)
        return self

    def b7(self) -> dict[str, float]:
        """Within-speaker embedding dispersion -- the cartoon-voice proxy (P0-F4)."""
        e = self.embeddings
        n = len(e)
        if n < max(2, self.min_windows):
            # Too few windows is "cannot be measured". It used to be 0.0, which ranked the
            # child as the least cartoon-like voice in the cohort (health check P12).
            nan = float("nan")
            return {"b7_mean_pairwise_cosdist": nan, "b7_sd_pairwise_cosdist": nan,
                    "b7_centroid_cosdist": nan, "b7_n_windows": float(n)}
        z = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        sim = z @ z.T
        iu = np.triu_indices(n, k=1)
        d = 1.0 - sim[iu]
        centroid = z.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        return {
            "b7_mean_pairwise_cosdist": float(np.mean(d)),
            "b7_sd_pairwise_cosdist": float(np.std(d)),
            "b7_centroid_cosdist": float(np.mean(1.0 - z @ centroid)),
            "b7_n_windows": float(n),
        }

    def pooled(self, functionals: Sequence[str]) -> dict[str, float]:
        """The B-stream feature vector: spectral balance, intensity, ECAPA dispersion.

        CPP and the perturbation channels are computed and reported but **excluded here**
        (PI decision, 2026-09-12). CPP fails its own audit gate on this cohort -- 17 of 19
        subjects exceed P1-5's 20% missingness ceiling, worst case 98% -- and the missing
        values are not random: their correlation with the SNR gate is r = 0.99, so the
        missingness pattern itself is a recording-quality channel a classifier could learn.
        The perturbation channels were already downgraded out of the core set by P1-5 v3.0.
        Both stay in ``quality``/``cpp_per_turn`` so the audit trail survives.
        """
        out: dict[str, float] = {}
        for ch in LLD_CHANNELS:
            out.update(pool(self.lld.get(ch, np.zeros(0)), functionals, f"b_{_short(ch)}",
                            self.min_n))
        # A11 音量, gain-invariant. The absolute Loudness channels move with recording gain
        # and codec (health check 0930 P5: the same session re-encoded moved B-stream
        # features by up to 3.7 SD). A level *difference* within one recording cancels any
        # gain applied to the whole file. It does not cancel a difference in how far each
        # speaker sat from the microphone -- that is audit P0-F5's open question.
        enough = int(np.isfinite(self.turn_db).sum()) >= self.min_n
        out["b_level_rel_examiner_db"] = self.level_rel_examiner_db if enough else float("nan")
        out.update(self.b7())
        return out


def _short(name: str) -> str:
    return name.replace("_sma3nz", "").replace("_sma3", "").replace("-", "_")


def voiced_runs(x: np.ndarray, sr: int, f0_min: float = 100.0, f0_max: float = 600.0,
                min_run_s: float = 0.06) -> list[tuple[int, int]]:
    """Contiguous voiced stretches, as sample index pairs.

    Audit P1-5 asks for CPP "only on sustained vowels / stable voiced segments". Computing
    it over a whole turn instead mixes in breath, page-turning and inter-word silence, and
    those drag the turn's SNR below the gate: measured on this cohort, 57% of child turns
    fell below the 15 dB floor and CPP missingness tracked that at r = 1.00.
    """
    if x.size < int(min_run_s * sr):
        return []
    try:
        import parselmouth

        snd = parselmouth.Sound(np.ascontiguousarray(x, dtype=np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
        voiced = pitch.selected_array["frequency"] > 0
    except Exception:
        return []
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(np.append(voiced, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) * 0.01 >= min_run_s:
                runs.append((int(start * 0.01 * sr), min(len(x), int(i * 0.01 * sr))))
            start = None
    return runs


def cpp_of(x: np.ndarray, sr: int, f0_min: float = 100.0, f0_max: float = 600.0) -> float:
    """Smoothed cepstral peak prominence via Praat. NaN when the segment is too short."""
    if x.size < int(0.08 * sr):
        return float("nan")
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(np.ascontiguousarray(x, dtype=np.float64), sampling_frequency=sr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pc = call(snd, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
            return float(call(pc, "Get CPPS...", False, 0.02, 0.0005, f0_min, f0_max,
                              0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust"))
    except Exception:
        return float("nan")


def ecapa_windows(audio: np.ndarray, sr: int, turns: Sequence[Turn],
                  cfg: Mapping, speaker: Speaker = Speaker.CHILD) -> np.ndarray:
    """Fixed-length windows over one speaker's speech (audit P0-F4: never mixed lengths).

    The child's windows feed B7; the examiner's are the voiceprint the untranscribed-
    vocalisation detector compares against (features/vocal.py)."""
    b_cfg = cfg["features"]["b_stream"]
    n_win = int(round(float(b_cfg["ecapa_window_s"]) * sr))
    n_hop = int(round(float(b_cfg["ecapa_hop_s"]) * sr))
    kids = [t for t in turns if t.speaker is speaker]
    if not kids:
        return np.zeros((0, n_win), dtype=np.float32)
    child_audio = np.concatenate([slice_turn(audio, sr, t) for t in kids])
    starts = list(range(0, max(0, child_audio.size - n_win + 1), n_hop))
    if not starts:
        return np.zeros((0, n_win), dtype=np.float32)
    return np.stack([child_audio[s : s + n_win] for s in starts]).astype(np.float32)


def extract_b(
    audio: np.ndarray,
    sr: int,
    turns: Sequence[Turn],
    smile_lld,
    cfg: Mapping,
) -> BStreamFeatures:
    """CPU-only part of the B stream: openSMILE LLDs, CPP, per-turn level.

    ECAPA is deliberately not done here so this function can run inside a process pool;
    the caller batches the embeddings on the GPU and calls ``BStreamFeatures.attach``.
    """
    b_cfg = cfg["features"]["b_stream"]
    snr_floor = float(b_cfg["snr_floor_db"])
    min_windows = int(b_cfg["ecapa_min_windows"])
    on_voiced = bool(b_cfg.get("cpp_on_voiced_segments", True))
    min_run_s = float(b_cfg.get("cpp_min_voiced_run_s", 0.06))

    kids = child_turns(turns)
    noise_e = noise_floor_energy(audio)
    frames: dict[str, list[np.ndarray]] = {ch: [] for ch in ALL_LLD_CHANNELS}
    cpps, dbs, ids = [], [], []
    n_below_snr = 0
    n_no_voiced = 0
    n_core_lld = 0
    voiced_seconds = 0.0
    smile = smile_lld

    for t in kids:
        x = slice_turn(audio, sr, t)
        if x.size < int(0.1 * sr):
            continue
        ids.append(t.turn_id)
        dbs.append(rms_db(x))

        # ---- ungated: spectral balance and intensity are not perturbation measures ----
        df = smile.process_signal(np.ascontiguousarray(x, dtype=np.float64), sr)
        voiced = df[_VOICED_GATE].to_numpy() > 0 if _VOICED_GATE in df.columns else None

        def _take(channels):
            for ch in channels:
                if ch not in df.columns:
                    continue
                v = df[ch].to_numpy(dtype=np.float64)
                if ch.endswith("_sma3nz") and voiced is not None:
                    v = v[voiced]
                frames[ch].append(v)

        _take(LLD_CHANNELS)
        n_core_lld += 1

        # ---- gated: CPP and the perturbation channels (audit P1-5) ----
        # The gate is evaluated on the stable voiced portion, not the whole turn: breath,
        # page-turning and inter-word silence dragged whole turns below the floor.
        if on_voiced:
            runs = voiced_runs(x, sr, min_run_s=min_run_s)
            if not runs:
                n_no_voiced += 1
                cpps.append(np.nan)
                continue
            target = np.concatenate([x[a:b] for a, b in runs])
            voiced_seconds += target.size / sr
        else:
            target = x
        seg_snr = snr_db(target, noise_e)
        if not np.isnan(seg_snr) and seg_snr < snr_floor:
            n_below_snr += 1
            cpps.append(np.nan)
            continue
        cpps.append(cpp_of(target, sr))
        _take(PERTURBATION_CHANNELS)

    lld = {ch: (np.concatenate(v) if v else np.zeros(0)) for ch, v in frames.items()}

    child_speech_s = sum(slice_turn(audio, sr, t).size for t in kids) / sr
    embeddings = np.zeros((0, 192), dtype=np.float32)

    quality = {
        "b_n_child_turns": float(len(ids)),
        "b_n_below_snr": float(n_below_snr),
        "b_ecapa_windows": float(len(embeddings)),
        "b_ecapa_windows_ok": float(len(embeddings) >= min_windows),
        "b_child_speech_s": float(child_speech_s),
        "b_cpp_nan_ratio": float(np.mean(np.isnan(cpps))) if cpps else 1.0,
        "b_n_no_voiced_run": float(n_no_voiced),
        "b_voiced_seconds": float(voiced_seconds),
        "b_cpp_on_voiced": float(on_voiced),
        # Coverage of the ungated core channels. Under the old code this equalled the
        # gated count, so A7/A11 were measured on 57% of each child's speech.
        "b_core_lld_turns": float(n_core_lld),
        "b_core_lld_coverage": float(n_core_lld / max(len(ids), 1)),
    }
    return BStreamFeatures(
        lld=lld,
        cpp_per_turn=np.asarray(cpps, dtype=np.float64),
        turn_db=np.asarray(dbs, dtype=np.float64),
        turn_ids=np.asarray(ids, dtype=int),
        embeddings=embeddings,
        quality=quality,
        level_rel_examiner_db=_level_rel_examiner(audio, sr, turns, dbs),
        min_n=min_values(cfg),
    )


def _level_rel_examiner(audio: np.ndarray, sr: int, turns: Sequence[Turn],
                        child_dbs: Sequence[float]) -> float:
    """Median child turn level minus median examiner turn level, in dB. NaN if either is
    missing: a missing comparison is not a comparison of zero."""
    from ..contracts import Speaker

    ex = [rms_db(slice_turn(audio, sr, t)) for t in turns
          if t.speaker is Speaker.EXAMINER and t.duration >= 0.1]
    ex = [v for v in ex if np.isfinite(v)]
    ch = [v for v in child_dbs if np.isfinite(v)]
    if not ex or not ch:
        return float("nan")
    return float(np.median(ch) - np.median(ex))
