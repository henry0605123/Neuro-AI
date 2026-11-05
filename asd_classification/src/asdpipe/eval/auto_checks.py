"""Checks a machine can do without a listener (health check step 5, the automatic part).

Sessions: every TD session, plus the six ASD sessions whose transcripts existed when
DEVIATIONS D10 was written -- their features have been seen already. ASD sessions
transcribed later stay untouched: nothing about them is computed before the analysis plan
is frozen. Every number below is about measurement quality; none compares the children's
features between groups.

1. Alignment screen    how much of each child utterance's voiced audio MFA's syllables
                       cover. A squeezed alignment (ten characters in four clumps) covers
                       little of the voice; a usable one covers most of it.
2. Question rule       the punctuation-blind question detector against the TD
                       transcribers' own question marks.
3. Topic relevance     do real question-answer pairs look more alike to the sentence
                       encoder than an answer paired with another question? If not, G1
                       does not measure relevance. Same test for the F8 directions.
4. Speaker labels      each long turn's voiceprint against leave-one-out voiceprints of the
                       two labelled speakers: turns closer to the other speaker are flagged.
5. F0 cross-check      three pitch trackers' agreement and octave errors (audit P1-3), TD.
6. Vocalisation recall the ASD transcribers' [UNINTELLIGIBLE] marks as an answer key for
                       the untranscribed-vocalisation detector, with the rule fixed on TD.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..contracts import Speaker, Turn

__all__ = ["run_auto_checks", "SEEN_BEFORE"]

# Transcripts older than this were in hand when D10 recorded that ASD features had been seen.
SEEN_BEFORE = datetime(2026, 10, 1)


@dataclass
class _S:
    group: str
    audio: np.ndarray
    sr: int
    raw: list[Turn]                       # as transcribed, punctuation and all
    turns: list[Turn]                     # normalized, as the features see them
    unintelligible: list[Turn] = field(default_factory=list)
    spans: dict = field(default_factory=dict)


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _load(cfg: Mapping, cache_dir: Path, backends, aligner, log) -> list[_S]:
    from ..extract import child_alignment_inputs
    from ..normalize import normalize_turns
    from ..realdata import discover_cases, load_subject

    rd = cfg["realdata"]
    out: list[_S] = []
    for group in ("TD", "ASD"):
        gcfg = {**cfg, "realdata": {**rd, "cohort": group}}
        cases = discover_cases(rd["root"], rd["exclude_cases"], cohort=group)
        if group == "ASD":
            cases = [c for c in cases
                     if datetime.fromtimestamp(c.transcript.stat().st_mtime) < SEEN_BEFORE]
            assert len(cases) == 6, f"expected the 6 already-seen ASD sessions, got {len(cases)}"
        for c in cases:
            rec, _ = load_subject(c, gcfg, cache_dir=cache_dir)
            keep_u = {**gcfg, "realdata": {**gcfg["realdata"], "drop_unintelligible": False}}
            with_u, _ = load_subject(c, keep_u, cache_dir=cache_dir, with_audio=False)
            turns, _ = normalize_turns(rec.session.turns, rec.audio, rec.sample_rate, gcfg)
            kids = [t for t in turns if t.speaker is Speaker.CHILD and t.text]
            words = dict(zip([t.turn_id for t in kids], backends.segment([t.text for t in kids])))
            alignable, align_words = child_alignment_inputs(turns, words)
            spans = aligner.align(rec.audio, rec.sample_rate, alignable, align_words) if alignable else {}
            out.append(_S(group, rec.audio, rec.sample_rate, list(rec.session.turns), list(turns),
                          [t for t in with_u.session.turns
                           if t.unintelligible and t.speaker is Speaker.CHILD], dict(spans)))
        log(f"[auto] {group}: {sum(1 for s in out if s.group == group)} sessions")
    return out


# ------------------------------------------------------------------------------ 1. alignment
def _alignment(sessions: Sequence[_S]) -> dict:
    from ..features.common import frame_energy, noise_floor_energy

    res = {}
    for g in ("TD", "ASD"):
        cov = []
        for s in (x for x in sessions if x.group == g):
            e = frame_energy(s.audio, 512)
            hop = 512 / s.sr
            on = e > noise_floor_energy(s.audio) * 10 ** 0.6
            for t in s.turns:
                sp = s.spans.get(t.turn_id, ())
                if t.speaker is not Speaker.CHILD or len(sp) < 3:
                    continue
                a, b = int(t.t_start / hop), int(np.ceil(t.t_end / hop))
                idx = np.arange(a, min(b, len(on)))
                voiced = idx[on[idx]]
                if len(voiced) < 5:
                    continue
                centre = (voiced + 0.5) * hop
                inside = np.zeros(len(voiced), bool)
                for x in sp:
                    inside |= (centre >= x.t_start) & (centre <= x.t_end)
                cov.append(float(inside.mean()))
        cov = np.asarray(cov)
        res[g] = {"n": int(len(cov)), "median": float(np.median(cov)) if len(cov) else float("nan"),
                  "low": float(np.mean(cov < 0.5)) if len(cov) else float("nan")}
    return res


# ------------------------------------------------------------------------------ 2. questions
def _questions(sessions: Sequence[_S]) -> dict:
    mark, rule = [], []
    for s in (x for x in sessions if x.group == "TD"):
        for t in s.raw:
            if t.speaker is Speaker.EXAMINER and t.text:
                mark.append("？" in t.text or "?" in t.text)
                rule.append(bool(t.is_question))
    mark, rule = np.asarray(mark), np.asarray(rule)
    return {"n": int(len(mark)), "marked": int(mark.sum()),
            "recall": float(rule[mark].mean()), "precision": float(mark[rule].mean()),
            "rule_only": int((rule & ~mark).sum())}


# ------------------------------------------------------------------------------ 3. relevance
def _relevance(sessions: Sequence[_S], backends, cfg: Mapping, seed: int = 0) -> dict:
    from ..features.e_text import _overlap_ratio, pair_turns, speech_blocks

    rng = np.random.default_rng(seed)
    e_cfg = cfg["features"]["e_stream"]
    cut = float(e_cfg.get("exclude_verbatim_echo_overlap", 0.8))
    min_chars = int(e_cfg["min_answer_chars"])
    g1_auc, f8_auc = [], []
    for s in (x for x in sessions if x.group == "TD"):
        turns = [t for t in s.turns if t.text]
        emb = dict(zip([t.turn_id for t in turns], backends.encode_sentences([t.text for t in turns])))
        cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        pairs = [(ex, ch) for ex, ch in pair_turns(turns)
                 if ex.is_question and not ex.is_yes_no and len(ch.text) >= min_chars
                 and not (_overlap_ratio(ch.text, ex.text) >= cut)]
        if len(pairs) >= 10:
            real = [cos(emb[ch.turn_id], emb[ex.turn_id]) for ex, ch in pairs]
            others = [ch for _, ch in pairs]
            fake = []
            for i, (ex, _) in enumerate(pairs):
                j = int(rng.integers(len(others) - 1))
                j += j >= i                                   # never the pair's own answer
                fake.append(cos(emb[others[j].turn_id], emb[ex.turn_id]))
            g1_auc.append(_auc(real, fake))
        blocks, bemb = speech_blocks(turns, emb)
        bp = [(ex, ch) for ex, ch in pair_turns(blocks) if ex.turn_id in bemb and ch.turn_id in bemb]
        if len(bp) >= 10:
            real = [cos(bemb[ch.turn_id], bemb[ex.turn_id]) for ex, ch in bp]
            kids = [ch for _, ch in bp]
            fake = []
            for i, (ex, _) in enumerate(bp):
                j = int(rng.integers(len(kids) - 1))
                j += j >= i
                fake.append(cos(bemb[kids[j].turn_id], bemb[ex.turn_id]))
            f8_auc.append(_auc(real, fake))
    summ = lambda v: {"n": len(v), "median": float(np.median(v)) if v else float("nan"),
                      "above": int(sum(x > 0.5 for x in v))}
    return {"g1": summ(g1_auc), "f8": summ(f8_auc)}


# ------------------------------------------------------------------------------ 4. speakers
def _speakers(sessions: Sequence[_S], backends, min_s: float = 2.0) -> dict:
    res = {}
    for g in ("TD", "ASD"):
        flagged = {Speaker.CHILD: [0, 0], Speaker.EXAMINER: [0, 0]}
        for s in (x for x in sessions if x.group == g):
            long_turns = [t for t in s.turns if t.duration >= min_s
                          and t.speaker in (Speaker.CHILD, Speaker.EXAMINER)]
            if not long_turns:
                continue
            clips = [s.audio[int(t.t_start * s.sr):int(t.t_end * s.sr)] for t in long_turns]
            z = backends.embed_segments(clips)
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
            lab = np.array([t.speaker is Speaker.CHILD for t in long_turns])
            sums = {True: z[lab].sum(0), False: z[~lab].sum(0)}
            cnt = {True: int(lab.sum()), False: int((~lab).sum())}
            for k, t in enumerate(long_turns):
                own, other = bool(lab[k]), not bool(lab[k])
                if cnt[own] < 2 or cnt[other] < 1:
                    continue
                c_own = sums[own] - z[k]                       # leave this turn out
                c_own /= max(np.linalg.norm(c_own), 1e-9)
                c_oth = sums[other] / max(np.linalg.norm(sums[other]), 1e-9)
                f = flagged[t.speaker]
                f[0] += 1
                f[1] += float(z[k] @ c_oth) > float(z[k] @ c_own)
        res[g] = {sp.name.lower(): {"n": n, "flag_rate": (k / n if n else float("nan"))}
                  for sp, (n, k) in flagged.items()}
    return res


# ------------------------------------------------------------------------------ 5. F0 (TD)
def _f0(cfg: Mapping, backends, aligner, cache_dir: Path) -> dict:
    from ..extract import extract_corpus
    from ..normalize import normalize_corpus
    from ..realdata import load_cohort

    rd = cfg["realdata"]
    tcfg = {**cfg, "realdata": {**rd, "cohort": "TD"}}
    raw, _ = load_cohort(rd["root"], tcfg, cache_dir=cache_dir)
    corpus, _ = normalize_corpus(raw, tcfg)
    rows = extract_corpus(corpus, tcfg, backends, aligner=aligner)
    agree = np.asarray([r.quality.get("c_f0_agreement", np.nan) for r in rows], float)
    octave = np.asarray([r.quality.get("c_f0_octave_error", np.nan) for r in rows], float)
    ok = (agree >= 0.90) & (octave < 0.03)
    return {"n": len(rows), "pass": int(ok.sum()), "agree_median": float(np.nanmedian(agree)),
            "octave_median": float(np.nanmedian(octave))}


# ------------------------------------------------------------------------------ 6. vocal recall
def _vocal_recall(sessions: Sequence[_S], cfg: Mapping, backends) -> dict:
    from ..features import b_voice
    from ..features.vocal import attribute, centroid, untranscribed_segments

    v = cfg["features"]["vocal"]
    n = det = att = 0
    sec = sec_det = sec_att = 0.0
    for s in (x for x in sessions if x.group == "ASD"):
        segs = untranscribed_segments(s.audio, s.sr, s.turns, v)
        cc = centroid(backends.embed_windows(b_voice.ecapa_windows(s.audio, s.sr, s.turns, cfg)),
                      int(v["min_enroll_windows"]))
        ec = centroid(backends.embed_windows(b_voice.ecapa_windows(
            s.audio, s.sr, s.turns, cfg, speaker=Speaker.EXAMINER)), int(v["min_enroll_windows"]))
        is_child = np.zeros(len(segs), bool)
        if segs and cc is not None and ec is not None:
            emb = backends.embed_segments([s.audio[int(a * s.sr):int(b * s.sr)] for a, b in segs])
            is_child = attribute(emb, cc, ec, float(v["margin"]), float(v["floor"]))[0]
        for u in s.unintelligible:
            d = u.t_end - u.t_start
            if d <= 0:
                continue
            ov = [max(0.0, min(b, u.t_end) - max(a, u.t_start)) for a, b in segs]
            hit = sum(ov) >= 0.5 * d
            mine = sum(o for o, c in zip(ov, is_child) if c) >= 0.5 * d
            n += 1
            det += hit
            att += mine
            sec += d
            sec_det += d * hit
            sec_att += d * mine
    return {"n": n, "detected": det / n if n else float("nan"),
            "attributed": att / n if n else float("nan"), "seconds": sec,
            "sec_detected": sec_det / sec if sec else float("nan"),
            "sec_attributed": sec_att / sec if sec else float("nan")}


# ------------------------------------------------------------------------------ driver
def run_auto_checks(cfg: Mapping, cache_dir: Path, out_path: Path, log=print,
                    with_f0: bool = False) -> Path:
    """Each check is written to ``out_path`` as soon as it finishes, so a crash later on
    loses nothing. The F0 cross-check re-runs feature extraction and is the heaviest on
    the GPU; it is off unless asked for."""
    from ..align import aligner_from_config
    from ..cli.main import _backends

    t0 = time.time()
    backends = _backends(cfg)
    aligner = aligner_from_config(cfg, cache_dir)
    pct = lambda x: "—" if not np.isfinite(x) else f"{100 * x:.1f}%"
    L = ["# 不需人工的自動檢查", "",
         f"產生時間：{datetime.now():%Y-%m-%d %H:%M}｜一般發展組 19 場、自閉症組 6 場（D10 時已有逐字稿者）；"
         "10/13 新轉錄的 4 場未使用。所有數字都是量測品質，不是兩組特徵的比較。", ""]

    def flush():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(L) + "\n", encoding="utf-8")

    sessions = _load(cfg, cache_dir, backends, aligner, log)

    log("[auto] 1 alignment")
    al = _alignment(sessions)
    L += ["## 1. MFA 切字：音節蓋住了孩子聲音的幾成", "",
          "| 組別 | 句數 | 覆蓋率中位數 | 覆蓋不到一半的句子 |", "|---|---|---|---|"]
    L += [f"| {g} | {al[g]['n']} | {pct(al[g]['median'])} | {pct(al[g]['low'])} |" for g in ("TD", "ASD")]
    L.append("")
    flush()

    log("[auto] 2 questions")
    qu = _questions(sessions)
    L += ["## 2. 問句判斷規則（只用一般發展組，以轉錄者標的問號當參考答案）", "",
          f"施測者 {qu['n']} 句，其中 {qu['marked']} 句有問號。有問號的句子被規則判為問句：**{pct(qu['recall'])}**；"
          f"規則判為問句的句子裡有問號：**{pct(qu['precision'])}**；規則判為問句但沒有問號：{qu['rule_only']} 句"
          "（可能是轉錄者漏標的問句，也可能是規則誤判）。", ""]
    flush()

    log("[auto] 3 relevance")
    rel = _relevance(sessions, backends, cfg)
    L += ["## 3. 語意相似度能不能反映「切題」「跟著話題」（只用一般發展組）", "",
          "做法：真的問答配對 vs 把回答換成同一場另一個回答。AUC 0.5＝分不出來，越接近 1 越能分。", "",
          "| 指標 | 孩子數 | AUC 中位數 | AUC > 0.5 的孩子 |", "|---|---|---|---|",
          f"| 切題（G1） | {rel['g1']['n']} | {rel['g1']['median']:.3f} | {rel['g1']['above']} |",
          f"| 孩子跟著施測者的話題（雙向配合） | {rel['f8']['n']} | {rel['f8']['median']:.3f} | {rel['f8']['above']} |", ""]
    flush()

    log("[auto] 4 speakers")
    spk = _speakers(sessions, backends)
    L += ["## 4. 說話者標記：聲紋比較像另一個人的句子（2 秒以上的句子）", "",
          "| 組別 | 標成孩子的句子 | 其中聲紋較像施測者 | 標成施測者的句子 | 其中聲紋較像孩子 |",
          "|---|---|---|---|---|"]
    for g in ("TD", "ASD"):
        c, e = spk[g]["child"], spk[g]["examiner"]
        L.append(f"| {g} | {c['n']} | {pct(c['flag_rate'])} | {e['n']} | {pct(e['flag_rate'])} |")
    L += ["", "被標出的句子不一定標錯，只是值得人工優先檢查；聲紋在短句上本來就不太準。", ""]
    flush()

    log("[auto] 6 vocal recall")
    voc = _vocal_recall(sessions, cfg, backends)
    L += ["## 6. 未轉錄發聲偵測：以自閉症組的 [UNINTELLIGIBLE] 標記當答案（門檻只用一般發展組訂好）", "",
          f"共 {voc['n']} 段、{voc['seconds']:.0f} 秒被標為聽不清楚的兒童發聲。", "",
          "| | 以段數計 | 以秒數計 |", "|---|---|---|",
          f"| 被偵測為「有聲音、沒轉成文字」 | {pct(voc['detected'])} | {pct(voc['sec_detected'])} |",
          f"| 而且被判斷為孩子的聲音 | {pct(voc['attributed'])} | {pct(voc['sec_attributed'])} |", ""]
    flush()

    if with_f0:
        log("[auto] 5 f0")
        f0 = _f0(cfg, backends, aligner, cache_dir)
        L += ["## 5. 音高追蹤的交叉比對（只用一般發展組；稽核標準：一致率 ≥ 90%、八度誤差 < 3%）", "",
              f"{f0['pass']}／{f0['n']} 位孩子達標；一致率中位數 {pct(f0['agree_median'])}、"
              f"八度誤差中位數 {pct(f0['octave_median'])}。", ""]
    else:
        L += ["## 5. 音高追蹤的交叉比對", "", "這次沒跑：要重跑整套特徵擷取，等顯示卡空出來再做。", ""]
    flush()
    log(f"[auto] done in {time.time() - t0:.0f}s -> {out_path}")
    return out_path
