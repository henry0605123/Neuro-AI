"""What the pipeline can be told about a recording before anyone transcribes it.

The ASD arm arrived on 2026-09-12 as 21 raw video files with no transcripts. The C and E
streams cannot run without them -- C needs word-level alignment, E needs text -- and no
amount of engineering changes that. What *can* run is everything upstream of the
transcript: decoding, the recording-condition inventory that audit P0-0A asks for, the
SNR gate that audit P1-5 applies, F0 tracking and its cross-algorithm check (P1-3), and
the B stream's spectral machinery.

Why this matters more than a normal dry run. Every feature that had to be dropped from
this pipeline died of recording quality: CPP missing for 17 of 19 subjects, F0 cross-check
failing for 4, spectral measures gated out of 43% of turns. If the ASD recordings differ
from the TD recordings in noise floor or level -- different room, different microphone
placement, a year between cohorts -- then **every acoustic feature is confounded with
group**, and a classifier can reach high accuracy by learning the recording rather than the
child. Audit P0-0A exists for exactly this and wants the inventory taken up front.

So the comparison here is deliberately limited to *recording conditions*. It is a confound
check, not a result. Feature-level group comparisons are not run and must not be: the
analysis order is pre-registered (criterion validity first, then group difference; audit
P0-4), the transcripts that most features need do not exist yet, and reading group
differences off a convenience subset before that point is the failure mode the plan is
written to prevent.

Everything is computed with the pipeline's own functions -- ``noise_floor_energy``,
``snr_db``, ``Backends.track_f0``, ``Backends.f0_cross_check``, the frozen openSMILE
channel list -- so a pass here is evidence about the pipeline, not about a reimplementation
of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..features.common import frame_energy, noise_floor_energy, rms_db, snr_db
from ..realdata import resolve_excluded

__all__ = ["RecordingAudit", "speech_segments", "audit_recording"]

# `frame_energy` slices into non-overlapping blocks of this many samples. Reading it from
# the source function rather than restating it as a duration: assuming 25 ms here while the
# function used 512 samples put every detected segment 28% early, which looked like a
# plausible VAD rather than like an off-by-a-constant.
_ENERGY_FRAME = 512


@dataclass
class RecordingAudit:
    case_id: str
    cohort: str
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def speech_segments(audio: np.ndarray, sr: int, noise_e: float,
                    over_floor_db: float = 10.0, min_s: float = 0.30,
                    merge_gap_s: float = 0.20) -> list[tuple[int, int]]:
    """Energy-based speech detection, as sample index pairs.

    Not a substitute for the transcript's turns: it cannot tell the child from the examiner,
    so anything measured on these segments describes *the recording*, not the child. That is
    the right unit for a recording-condition audit and the wrong one for a feature.

    The threshold is relative to the session's own noise floor rather than absolute, because
    the quantity of interest -- how far speech sits above the room -- is what the SNR gate
    tests, and an absolute cut-off would just re-measure the recording level.
    """
    e = frame_energy(audio, _ENERGY_FRAME)
    if e.size == 0:
        return []
    hop = _ENERGY_FRAME
    thresh = noise_e * (10.0 ** (over_floor_db / 10.0))
    loud = e > thresh
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(np.append(loud, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start * hop, min(len(audio), i * hop)))
            start = None
    if not runs:
        return []
    merged = [list(runs[0])]
    gap = int(merge_gap_s * sr)
    for a, b in runs[1:]:
        if a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if (b - a) / sr >= min_s]


def audit_recording(case_id: str, cohort: str, path: Path, cfg: Mapping,
                    backends=None, cache_dir: Path | None = None,
                    sample_s: float = 180.0, seed: int = 20260912) -> RecordingAudit:
    """Decode one recording and measure everything that does not need a transcript.

    ``sample_s`` bounds the expensive stages (openSMILE, CREPE, pYIN) to a random sample of
    detected speech, drawn with a fixed seed so the numbers are reproducible. The cheap
    stages -- duration, level, noise floor, speech ratio, clipping -- always see the whole
    recording, because those are exactly the quantities a sample would distort.
    """
    from ..realdata import load_audio

    out = RecordingAudit(case_id=case_id, cohort=cohort)
    sr = int(cfg["synth"]["sample_rate"]) if "synth" in cfg else 16000
    try:
        audio = load_audio(path, sr, cache_dir)
    except Exception as exc:                       # a file that will not decode is a finding
        out.errors.append(f"decode failed: {type(exc).__name__}: {exc}")
        return out
    if audio.size < sr:
        out.errors.append(f"decoded to {audio.size} samples")
        return out

    m = out.metrics
    m["duration_min"] = audio.size / sr / 60.0
    m["level_dbfs"] = rms_db(audio)
    m["peak"] = float(np.abs(audio).max())
    m["clipped_ratio"] = float(np.mean(np.abs(audio) > 0.99))
    m["dc_offset"] = float(np.mean(audio))

    noise_e = noise_floor_energy(audio)
    m["noise_floor_dbfs"] = float(10.0 * np.log10(noise_e + 1e-20))

    segs = speech_segments(audio, sr, noise_e)
    speech_s = sum(b - a for a, b in segs) / sr
    m["n_speech_segments"] = float(len(segs))
    m["speech_ratio"] = float(speech_s * sr / audio.size)
    m["segment_mean_s"] = float(speech_s / len(segs)) if segs else 0.0
    if not segs:
        out.errors.append("no speech detected above the noise floor")
        return out

    # ---- the SNR gate that audit P1-5 applies, measured segment by segment ----
    snr_floor = float(cfg["features"]["b_stream"]["snr_floor_db"])
    snrs = np.array([snr_db(audio[a:b], noise_e) for a, b in segs])
    snrs = snrs[np.isfinite(snrs)]
    m["snr_median_db"] = float(np.median(snrs)) if snrs.size else float("nan")
    m["snr_p10_db"] = float(np.percentile(snrs, 10)) if snrs.size else float("nan")
    m["below_snr_floor_ratio"] = float(np.mean(snrs < snr_floor)) if snrs.size else float("nan")

    # ---- bounded sample for the expensive stages ----
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(segs))
    picked, total = [], 0.0
    for i in order:
        a, b = segs[i]
        picked.append((a, b))
        total += (b - a) / sr
        if total >= sample_s:
            break
    picked.sort()
    sample = np.concatenate([audio[a:b] for a, b in picked])
    m["sampled_s"] = float(sample.size / sr)

    if backends is None:
        return out

    c_cfg = cfg["features"]["c_stream"]
    fmin, fmax = float(c_cfg["f0_min_hz"]), float(c_cfg["f0_max_hz"])
    hop_s = float(c_cfg["f0_hop_s"])
    try:
        f0, per = backends.track_f0(sample, sr, fmin, fmax, hop_s)
        voiced = (per >= 0.40) & np.isfinite(f0) & (f0 >= fmin) & (f0 <= fmax)
        m["f0_voiced_ratio"] = float(np.mean(voiced))
        m["f0_median_hz"] = float(np.median(f0[voiced])) if voiced.any() else float("nan")
        # audit P1-3 reports a tracking-failure ratio; the complement of voicing is the
        # closest transcript-free analogue, and is reported as such rather than as P1-3.
        check = backends.f0_cross_check(sample, sr, fmin, fmax, hop_s)
        m["f0_agreement"] = float(check.get("agreement", float("nan")))
        m["f0_octave_error"] = float(check.get("octave_error", float("nan")))
    except Exception as exc:
        out.errors.append(f"f0 stage failed: {type(exc).__name__}: {exc}")

    # ---- B-stream spectral machinery, on the frozen channel list ----
    try:
        from ..features.b_voice import LLD_CHANNELS

        df = backends.smile_lld.process_signal(
            np.ascontiguousarray(sample, dtype=np.float64), sr)
        finite = []
        for ch in LLD_CHANNELS:
            if ch not in df.columns:
                out.errors.append(f"openSMILE did not return {ch}")
                continue
            v = df[ch].to_numpy(dtype=np.float64)
            finite.append(float(np.mean(np.isfinite(v))))
            if ch == "Loudness_sma3":
                m["lld_loudness_mean"] = float(np.nanmean(v))
            if ch == "alphaRatio_sma3":
                m["lld_alpha_ratio_mean"] = float(np.nanmean(v))
        m["lld_finite_ratio"] = float(np.min(finite)) if finite else 0.0
    except Exception as exc:
        out.errors.append(f"openSMILE stage failed: {type(exc).__name__}: {exc}")

    # ---- ECAPA windows: audit P0-F4 wants >= 15 fixed-length windows per subject ----
    try:
        b_cfg = cfg["features"]["b_stream"]
        n_win = int(round(float(b_cfg["ecapa_window_s"]) * sr))
        hop = int(round(float(b_cfg.get("ecapa_hop_s", b_cfg["ecapa_window_s"])) * sr))
        wins = [sample[i:i + n_win] for i in range(0, max(0, sample.size - n_win + 1), hop)]
        m["ecapa_windows"] = float(len(wins))
        if wins:
            emb = backends.embed_windows(np.stack(wins[:64]).astype(np.float32))
            m["ecapa_dim"] = float(emb.shape[1])
            m["ecapa_finite_ratio"] = float(np.mean(np.isfinite(emb)))
    except Exception as exc:
        out.errors.append(f"ECAPA stage failed: {type(exc).__name__}: {exc}")

    return out


def audit_cohort(root: Path, cohort: str, cfg: Mapping, backends=None,
                 cache_dir: Path | None = None, exclude: Sequence[str] = (),
                 limit: int | None = None, id_root: Path | None = None) -> list[RecordingAudit]:
    """Audit every recording under ``root/cohort``, transcript or not. ``exclude`` is
    ``realdata.exclude_cases``: ids or folder-name fragments (``resolve_excluded``)."""
    drop = resolve_excluded(exclude, id_root)
    base = Path(root) / cohort
    if not base.is_dir():
        return []
    media = (".mp3", ".wav", ".m4a", ".mp4")
    out: list[RecordingAudit] = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if any(x in d.name for x in drop):
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in media)
        if not files:
            out.append(RecordingAudit(d.name, cohort, errors=["no media file"]))
            continue
        out.append(audit_recording(d.name, cohort, files[0], cfg, backends, cache_dir))
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------- reporting
_ROWS: tuple[tuple[str, str, int], ...] = (
    ("duration_min", "時長(分)", 1),
    ("level_dbfs", "整體音量(dBFS)", 1),
    ("noise_floor_dbfs", "噪音底(dBFS)", 1),
    ("speech_ratio", "語音佔比", 3),
    ("n_speech_segments", "語音段數", 0),
    ("segment_mean_s", "段長均值(秒)", 2),
    ("snr_median_db", "SNR 中位(dB)", 1),
    ("snr_p10_db", "SNR 第10百分位(dB)", 1),
    ("below_snr_floor_ratio", "低於 SNR 門檻比例", 3),
    ("clipped_ratio", "削波比例", 5),
    ("f0_voiced_ratio", "F0 濁音幀比例", 3),
    ("f0_median_hz", "F0 中位(Hz)", 1),
    ("f0_agreement", "CREPE/pYIN 一致率", 3),
    ("f0_octave_error", "八度錯誤率", 3),
    ("lld_finite_ratio", "openSMILE 有效值比例", 3),
    ("ecapa_windows", "ECAPA 視窗數", 0),
)


def _col(audits: Sequence[RecordingAudit], key: str) -> np.ndarray:
    v = [a.metrics[key] for a in audits if key in a.metrics and np.isfinite(a.metrics[key])]
    return np.asarray(v, dtype=float)


def _fmt(x: float, nd: int) -> str:
    return "—" if not np.isfinite(x) else f"{x:.{nd}f}"


def write_audio_audit_report(path: Path, audits: Sequence[RecordingAudit],
                             cfg: Mapping) -> Path:
    """A recording-condition inventory for both cohorts (audit P0-0A).

    Case identifiers are replaced by cohort ordinals. The mapping is not written anywhere in
    the repository: this report is the kind that gets forwarded, and the intake folders are
    named after the children.
    """
    from datetime import datetime

    by_cohort: dict[str, list[RecordingAudit]] = {}
    for a in audits:
        by_cohort.setdefault(a.cohort, []).append(a)

    L: list[str] = []
    A = L.append
    A("# 錄音條件盤點（稽核 P0-0A）")
    A("")
    A(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}")
    A("")
    A("> 🔴 **這不是研究結果，是混淆檢查。** 這裡比較的只有**錄音條件**，不是特徵、更不是分組差異。")
    A("> 之所以必須先做：本管線被迫移除的特徵全部死於錄音品質。若兩組的噪音底或音量不同，")
    A("> 那麼**每一個聲學特徵都與組別混淆**，分類器可以靠「認錄音」而不是「認小孩」拿到高準確率。")
    A("")
    A("> ⚠ **C 流與 E 流在此完全無法執行**，因為它們需要逐字稿（C 流要詞層對齊，E 流要文字）。")
    A("> 下面測到的是解碼、音訊條件、SNR 閘門、F0 追蹤與 B 流頻譜機制——也就是逐字稿之前的全部管線。")
    A("")

    # ------------------------------------------------------------------ per cohort
    for cohort, group in sorted(by_cohort.items()):
        ok = [a for a in group if a.ok]
        A(f"## {cohort}（{len(group)} 筆錄音，{len(ok)} 筆可分析）")
        A("")
        bad = [a for a in group if not a.ok]
        if bad:
            A("| 個案 | 問題 |")
            A("|---|---|")
            for i, a in enumerate(bad, 1):
                A(f"| {cohort}-x{i:02d} | {'; '.join(a.errors)} |")
            A("")
        if not ok:
            A("**沒有任何一筆可分析。**")
            A("")
            continue
        A("| 個案 | " + " | ".join(t for _, t, _ in _ROWS) + " |")
        A("|---" * (len(_ROWS) + 1) + "|")
        for i, a in enumerate(sorted(ok, key=lambda x: x.case_id), 1):
            cells = [_fmt(a.metrics.get(k, float("nan")), nd) for k, _, nd in _ROWS]
            A(f"| {cohort}{i:02d} | " + " | ".join(cells) + " |")
        A("")

    # ------------------------------------------------------------------ comparison
    cohorts = sorted(by_cohort)
    if len(cohorts) >= 2:
        A("## 兩組錄音條件對照（中位數 [IQR]）")
        A("")
        A("| 指標 | " + " | ".join(cohorts) + " | 中位數差 |")
        A("|---" * (len(cohorts) + 2) + "|")
        for key, title, nd in _ROWS:
            cells, meds = [], []
            for c in cohorts:
                v = _col([a for a in by_cohort[c] if a.ok], key)
                if v.size == 0:
                    cells.append("—"); meds.append(float("nan")); continue
                q1, q3 = np.percentile(v, [25, 75])
                meds.append(float(np.median(v)))
                cells.append(f"{_fmt(np.median(v), nd)} [{_fmt(q1, nd)}, {_fmt(q3, nd)}]")
            diff = meds[0] - meds[1] if len(meds) >= 2 else float("nan")
            A(f"| {title} | " + " | ".join(cells) + f" | {_fmt(diff, nd)} |")
        A("")
        A("> **怎麼讀這張表。** 噪音底、整體音量與 SNR 若在兩組之間差得比組內 IQR 還多，")
        A("> 就表示兩組是在不同的錄音條件下收的，聲學特徵不能直接跨組比較，")
        A("> 必須先處理（同一房間重錄、或把錄音條件當共變量並如實揭露其後果）。")
        A("> 時長與語音佔比的差異則影響每人可用的樣本數，進而影響各特徵的抽樣誤差。")
        A("")

    A("## 這份報告不能拿來做什麼")
    A("")
    A("- **不能拿來比較兩組的特徵。** 這裡的語音段是用能量偵測切出來的，**分不出兒童與施測者**；")
    A("  任何在這些段落上算出的數字描述的是「這段錄音」，不是「這個小孩」。")
    A("- **不能當成分類結果。** 計畫書預先登錄的分析順序是效標效度優先、組間差異其次（稽核 P0-4）；")
    A("  在逐字稿到齊、主要分析執行之前讀取組間差異，正是該條款要防止的事。")
    A("- **不能用來判斷特徵是否可用。** 那需要逐字稿。")
    return _write(path, L)


def _write(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
