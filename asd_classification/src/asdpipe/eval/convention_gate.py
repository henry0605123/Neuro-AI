"""Convention-invariance gate -- decision_rules.yaml ``convention_invariance_gate``.

Question: if the TD transcripts had been produced the way the ASD ones were, how far would
each feature move? A feature that moves as much as the effect we are trying to detect cannot
tell "autism" from "which team made the file", whatever a classifier later does with it.

Method. Same audio, same words; only the transcript's conventions change. Each transform is
applied to the *raw* TD transcripts -- which is where the other pipeline's habits live -- and
the result then goes through the ordinary source normalization and extraction. The shift for
a feature is

    |median over subjects of (f_transformed - f_original)| / SD over subjects of f_original

which is the size, in the units a group comparison would use, of the spurious difference the
convention would create. Threshold and consequence are pre-registered; this module only
measures.

TD only. No label is read anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

import numpy as np

from ..contracts import Corpus, Turn
from ..normalize import merge_same_speaker, strip_punctuation

__all__ = ["TRANSFORMS", "stitch_speaker_changes", "GateResult", "gate_shifts"]


def stitch_speaker_changes(turns: Sequence[Turn], max_gap_s: float) -> list[Turn]:
    """At a change of speaker with a gap of at most ``max_gap_s``, start the later turn
    exactly where the earlier one ended. 36-92% of speaker-change gaps are 0.000 s in the
    chunk and RuleStitch transcripts, against 15-24% in the subtitle-derived ones."""
    ordered = sorted(turns, key=lambda t: (t.t_start, t.t_end))
    out: list[Turn] = []
    for t in ordered:
        if out and t.speaker is not out[-1].speaker:
            gap = t.t_start - out[-1].t_end
            if 0.0 < gap <= max_gap_s:
                t = replace(t, t_start=out[-1].t_end)
        out.append(t)
    return out


def _strip(turns: Sequence[Turn], _cfg: Mapping) -> list[Turn]:
    return [replace(t, text=strip_punctuation(t.text)) for t in turns]


def _merge(turns: Sequence[Turn], cfg: Mapping) -> list[Turn]:
    g = cfg["convention_invariance_gate"]["transforms"]["merge_same_speaker"]["max_gap_s"]
    return merge_same_speaker(turns, float(g))


def _stitch(turns: Sequence[Turn], cfg: Mapping) -> list[Turn]:
    g = cfg["convention_invariance_gate"]["transforms"]["stitch_speaker_changes"]["max_gap_s"]
    return stitch_speaker_changes(turns, float(g))


# name -> function(raw turns, decision rules) -> transformed raw turns
TRANSFORMS: dict[str, Callable[[Sequence[Turn], Mapping], list[Turn]]] = {
    "strip_punctuation": _strip,
    "merge_same_speaker": _merge,
    "stitch_speaker_changes": _stitch,
}


def apply_transform(corpus: Corpus, name: str, rules: Mapping) -> Corpus:
    fn = TRANSFORMS[name]
    subjects = tuple(replace(s, session=replace(s.session, turns=tuple(fn(s.session.turns, rules))))
                     for s in corpus.subjects)
    return replace(corpus, subjects=subjects)


@dataclass
class GateResult:
    feature: str
    sd: float
    shifts: dict[str, float] = field(default_factory=dict)       # transform -> |median| / sd
    spread: dict[str, float] = field(default_factory=dict)       # transform -> max |delta| / sd

    def worst(self) -> tuple[str, float]:
        if not self.shifts:
            return "", float("nan")
        k = max(self.shifts, key=lambda x: self.shifts[x])
        return k, self.shifts[k]

    def passes(self, threshold: float) -> bool:
        return bool(self.shifts) and all(v < threshold for v in self.shifts.values())


def gate_shifts(base: Mapping[str, np.ndarray], variants: Mapping[str, Mapping[str, np.ndarray]]
                ) -> list[GateResult]:
    """``base`` and each variant map feature name -> values over the same subjects, same order.

    A feature that is constant across subjects gets sd = 0 and no shifts: it cannot be
    judged, and a constant carries no information for a comparison anyway.
    """
    out: list[GateResult] = []
    for name, x0 in base.items():
        x0 = np.asarray(x0, dtype=float)
        sd = float(np.nanstd(x0, ddof=1)) if np.isfinite(x0).sum() > 2 else float("nan")
        res = GateResult(feature=name, sd=sd)
        if np.isfinite(sd) and sd > 1e-12:
            for tname, feats in variants.items():
                if name not in feats:
                    continue
                d = np.asarray(feats[name], dtype=float) - x0
                d = d[np.isfinite(d)]
                if d.size:
                    res.shifts[tname] = float(abs(np.median(d)) / sd)
                    res.spread[tname] = float(np.max(np.abs(d)) / sd)
        out.append(res)
    return out


# ------------------------------------------------------------------------------ runner
def feature_matrix(rows) -> dict[str, np.ndarray]:
    """SubjectRow list -> feature name -> values, over B, C and E streams."""
    from ..assemble import build_table

    table = build_table(rows)
    out: dict[str, np.ndarray] = {}
    for s in ("b", "c", "e"):
        X, names = table.stream(s), table.feature_names(s)
        for j, n in enumerate(names):
            out[n] = X[:, j].astype(float)
    return out


def run_gate(cfg: Mapping, rules: Mapping, backends, aligner, cache_dir,
             log=print) -> tuple[list[GateResult], list[GateResult], dict]:
    """Returns (transform results, format results, run metadata). TD only."""
    from ..extract import extract_corpus
    from ..normalize import normalize_corpus
    from ..realdata import CaseFiles, discover_cases, load_cohort, load_subject

    rd = cfg["realdata"]
    cfg = {**cfg, "realdata": {**rd, "cohort": "TD"}}
    raw, _ = load_cohort(rd["root"], cfg, cache_dir=cache_dir)
    order = [s.subject_id for s in raw.subjects]

    def features_of(corpus, ids) -> dict[str, np.ndarray]:
        norm, _ = normalize_corpus(corpus, cfg)
        rows = extract_corpus(norm, cfg, backends, aligner=aligner)
        by_id = {r.subject_id: r for r in rows}
        return feature_matrix([by_id[i] for i in ids if i in by_id])

    log("[gate] baseline (normalized TD)")
    base = features_of(raw, order)
    variants = {}
    for name in TRANSFORMS:
        log(f"[gate] transform: {name}")
        variants[name] = features_of(apply_transform(raw, name, rules), order)
    results = gate_shifts(base, variants)

    # ---- format check: the TD sessions that exist as both .mp3 and .mp4 ----
    cases = discover_cases(rd["root"], rd["exclude_cases"], cohort="TD")
    pairs = []
    for c in cases:
        mp3 = sorted(c.directory.glob("*.mp3"))
        mp4 = sorted(c.directory.glob("*.mp4"))
        if mp3 and mp4:
            pairs.append((c, mp3[0], mp4[0]))
    fmt_results: list[GateResult] = []
    meta = {"n_subjects": len(order), "n_format_pairs": len(pairs)}
    if pairs:
        log(f"[gate] format check on {len(pairs)} sessions")
        recs: dict[str, list] = {"mp3": [], "mp4": []}
        for c, a3, a4 in pairs:
            for tag, path in (("mp3", a3), ("mp4", a4)):
                rec, _ = load_subject(CaseFiles(c.case_id, c.directory, path, c.transcript),
                                      cfg, cache_dir=cache_dir)
                recs[tag].append(rec)
        ids = [r.subject_id for r in recs["mp4"]]
        f3 = features_of(replace(raw, subjects=tuple(recs["mp3"])), ids)
        f4 = features_of(replace(raw, subjects=tuple(recs["mp4"])), ids)
        # Standardise by the full TD cohort's SD, as for the transforms: the question is how
        # large the container effect is relative to the differences between children.
        sd_of = {r.feature: r.sd for r in results}
        for name, x4 in f4.items():
            sd = sd_of.get(name, float("nan"))
            res = GateResult(feature=name, sd=sd)
            if name in f3 and np.isfinite(sd) and sd > 1e-12:
                d = np.asarray(f3[name], float) - np.asarray(x4, float)
                d = d[np.isfinite(d)]
                if d.size:
                    res.shifts["mp3_vs_mp4"] = float(abs(np.median(d)) / sd)
                    res.spread["mp3_vs_mp4"] = float(np.max(np.abs(d)) / sd)
            fmt_results.append(res)
    return results, fmt_results, meta


def _cell(v, thr):
    if v is None or not np.isfinite(v):
        return "—"
    return f"**{v:.2f}**" if v >= thr else f"{v:.2f}"


def write_gate_report(path, results: Sequence[GateResult], fmt: Sequence[GateResult],
                      meta: Mapping, threshold: float, fmt_threshold: float):
    from datetime import datetime
    from pathlib import Path

    L: list[str] = []
    A = L.append
    A("# 慣例不變性關卡（decision_rules.yaml: convention_invariance_gate）")
    A("")
    A(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}｜TD {meta['n_subjects']} 人，不使用任何標籤")
    A("")
    A("每個特徵在每種轉換下的移動量＝｜受試者間中位數（轉換後－原始）｜÷ TD 的標準差。"
      f"任一轉換達 {threshold} 即未通過，依預先登錄的規則排除於確認性分析之外。")
    A("")
    judged = [r for r in results if r.shifts]
    fails = [r for r in judged if not r.passes(threshold)]
    fmt_by = {r.feature: r for r in fmt}
    fmt_fail = {r.feature for r in fmt if r.shifts and not r.passes(fmt_threshold)}
    A(f"**{len(judged)} 個可判定的特徵中，{len(judged) - len(fails)} 個通過慣例轉換、"
      f"{len(fails)} 個未通過；另有 {len(fmt_fail - {r.feature for r in fails})} 個只在檔案格式檢查"
      f"未通過。**（{len(results) - len(judged)} 個特徵在 TD 內為常數，無法判定）")
    A("")
    names = list(TRANSFORMS)
    A("## 各轉換的移動量")
    A("")
    A("| 轉換 | 有移動的特徵 | 中位移動量 | 最大移動量 | 最大者 |")
    A("|---|---|---|---|---|")
    for n in names:
        v = sorted(((r.shifts[n], r.feature) for r in judged if n in r.shifts), reverse=True)
        if v:
            moved = sum(x > 1e-12 for x, _ in v)
            top = f"`{v[0][1]}`" if moved else "—"
            A(f"| {n} | {moved}／{len(v)} | {np.median([x for x, _ in v]):.3f} | "
              f"{v[0][0]:.3f} | {top} |")
    A("")
    A("## 慣例轉換未通過")
    A("")
    if not fails:
        A("（無）")
    else:
        A("| 特徵 | " + " | ".join(names) + " | 格式 |")
        A("|---" * (len(names) + 2) + "|")
    for r in sorted(fails, key=lambda x: -x.worst()[1]):
        cells = [_cell(r.shifts.get(n), threshold) for n in names]
        fr = fmt_by.get(r.feature)
        cells.append(_cell(fr.shifts.get("mp3_vs_mp4") if fr else None, fmt_threshold))
        A(f"| `{r.feature}` | " + " | ".join(cells) + " |")
    A("")
    only_fmt = sorted(fmt_fail - {r.feature for r in fails})
    if only_fmt:
        A("## 只在檔案格式檢查未通過")
        A("")
        A(f"只用 {meta['n_format_pairs']} 場同時有 MP3 與 MP4 的錄音，屬篩檢而非估計。")
        A("")
        A("| 特徵 | MP3 對 MP4 移動量 | 兩場中最大 |")
        A("|---|---|---|")
        for n in only_fmt:
            r = fmt_by[n]
            A(f"| `{n}` | {r.shifts['mp3_vs_mp4']:.2f} | {r.spread['mp3_vs_mp4']:.2f} |")
        A("")
    passed = sorted(r.feature for r in judged if r.passes(threshold) and r.feature not in fmt_fail)
    A(f"## 全部通過（{len(passed)} 個）")
    A("")
    A("、".join(f"`{n}`" for n in passed))
    A("")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p
