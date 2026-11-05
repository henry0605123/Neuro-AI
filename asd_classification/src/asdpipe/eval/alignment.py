"""Alignment check on a real cohort (audit P1-1) and its report.

What P1-1 asks for is "human spot-check of 30-50 segments, median boundary error < 20 ms".
That threshold is defined against human-annotated boundaries and this cohort has none, so
this module never claims to have met it. It produces:

* automatic checks that stand on their own -- OOV rate, alignment yield, syllable recovery
  against the character count, duration plausibility;
* the disagreement between two independent alignments of the same audio (the Taiwan and
  mainland dictionaries), which *is* a real error bound and also measures how much the
  QUESTIONS Q14 choice matters;
* wav + TextGrid packages so a person can run the spot-check P1-1 actually mandates.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..align.evaluate import (AlignmentQuality, boundary_energy_agreement,
                              cross_alignment_agreement, duration_sanity, export_spotcheck,
                              implied_rate_profile, oov_report, read_dictionary_words)
from ..align.mfa import MFAAligner
from ..contracts import Speaker, SyllableSpan, Turn

__all__ = ["HAN", "segment_for_alignment", "align_case", "write_alignment_report"]

HAN = re.compile(r"[一-鿿]")


def segment_for_alignment(turns: Sequence[Turn], backends) -> dict[int, list[str]]:
    """CKIP segmentation, keeping only tokens that contain a Han character.

    Punctuation and bare digits dominate the raw OOV list (`!`, `...`, `1000`) and they
    have no acoustic realisation of their own, so feeding them to the aligner only creates
    spurious out-of-vocabulary tokens.
    """
    texts = [t.text for t in turns]
    segmented = backends.segment(texts)
    return {t.turn_id: [w for w in ws if HAN.search(w)] for t, ws in zip(turns, segmented)}


@dataclass
class CaseAlignment:
    case_id: str
    quality: AlignmentQuality
    spans: dict[int, tuple[SyllableSpan, ...]] = field(default_factory=dict)
    seconds: float = 0.0
    rate_profile: dict = field(default_factory=dict)


def align_case(case_id: str, audio: np.ndarray, sr: int, turns: Sequence[Turn],
               words: Mapping[int, Sequence[str]], aligner: MFAAligner,
               lexicon: set[str] | None = None) -> CaseAlignment:
    """Align one case's child turns and measure everything measurable without ground truth."""
    attempted = [t for t in turns if words.get(t.turn_id)]
    q = AlignmentQuality(case_id=case_id, n_turns_attempted=len(attempted))
    q.n_expected_syllables = sum(len(HAN.findall(t.text)) for t in attempted)

    if lexicon is not None:
        r = oov_report({t.turn_id: words[t.turn_id] for t in attempted}, lexicon)
        q.oov_types, q.vocab_types = r["oov_types"], r["vocab_types"]
        q.oov_tokens, q.total_tokens = r["oov_tokens"], r["total_tokens"]
        q.oov_examples = r["examples"][:12]

    t0 = time.time()
    spans = aligner.align(audio, sr, attempted, words)
    seconds = time.time() - t0

    q.n_turns_aligned = len(spans)
    flat = [s for v in spans.values() for s in v]
    q.n_syllables = len(flat)
    d = duration_sanity(flat)
    q.dur_median_ms, q.dur_p10_ms, q.dur_p90_ms = d["median_ms"], d["p10_ms"], d["p90_ms"]
    q.n_implausible = d["n_implausible"]

    energy = {"n": 0, "median_ms": float("nan"), "p90_ms": float("nan")}
    for turn_id, v in spans.items():
        t = next((x for x in attempted if x.turn_id == turn_id), None)
        if t is None or len(v) < 2:
            continue
        a, b = int(t.t_start * sr), int(t.t_end * sr)
        rebased = [SyllableSpan(s.t_start - t.t_start, s.t_end - t.t_start, s.tone) for s in v]
        e = boundary_energy_agreement(audio[a:b], sr, rebased)
        if e["n"]:
            energy["n"] += e["n"]
            energy.setdefault("_vals", []).append(e["median_ms"])
    vals = energy.get("_vals", [])
    if vals:
        q.energy_proxy_median_ms = float(np.median(vals))
        q.energy_proxy_p90_ms = float(np.percentile(vals, 90))
    return CaseAlignment(case_id=case_id, quality=q, spans=spans, seconds=seconds,
                         rate_profile=implied_rate_profile(spans))


def _f(x, nd=1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{v:.{nd}f}"


def write_alignment_report(path: Path, results: Sequence[CaseAlignment], cfg: Mapping,
                           cross: Mapping | None = None,
                           spotcheck_dir: Path | None = None) -> Path:
    mfa = cfg["models"]["mfa"]
    L: list[str] = []
    A = L.append
    q = [r.quality for r in results]

    A("# MFA 對齊品質報告（稽核 P1-1）")
    A("")
    A(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}　·　"
      f"聲學模型 `{mfa['acoustic_model']}`　·　字典 `{mfa['dictionary']}`")
    A("")
    A("> 🔴 **這份報告沒有、也不可能滿足 P1-1 的「邊界誤差中位數 < 20 ms」。**")
    A("> 那個門檻是對**人工標註的音節邊界**定義的，而這批語料沒有人工邊界標註。")
    A("> 下面給的是三類東西：(a) 本身就站得住的自動檢查；(b) 兩本字典各自對齊同一段音訊的")
    A("> **分歧程度**——那是誤差的真實下界；(c) 人工抽驗所需的 wav ＋ TextGrid 材料。")
    A("")

    yields = np.array([x.turn_yield for x in q])
    syl_yields = np.array([x.syllable_yield for x in q])
    A("## 一、對齊產率")
    A("")
    A(f"共 {len(q)} 個案。語句層級產率（MFA 有回傳結果的比例）"
      f"**{100 * yields.mean():.1f}%**（{100 * yields.min():.0f}–{100 * yields.max():.0f}%）；"
      f"音節回收率（實得音節數 ÷ 逐字稿漢字數）**{100 * np.nanmean(syl_yields):.1f}%**。")
    A("")
    A("| 個案 | 嘗試語句 | 對齊成功 | 語句產率 | 期望音節 | 實得音節 | 音節回收 | OOV詞次 | 耗時(s) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        x = r.quality
        A(f"| {x.case_id} | {x.n_turns_attempted} | {x.n_turns_aligned} | "
          f"{100 * x.turn_yield:.0f}% | {x.n_expected_syllables} | {x.n_syllables} | "
          f"{100 * x.syllable_yield:.0f}% | {100 * x.oov_token_rate:.1f}% | "
          f"{_f(r.seconds, 0)} |")
    A("")
    A("> **音節回收率低於 100% 是預期的，但要看是什麼原因。** 字典外的詞（OOV）會被對成 "
      "`spn`（不明語音），本管線的音節化在遇到 `spn`/靜音時會收束，因此那些字不會產生音節。"
      "換句話說，音節回收率的缺口主要是 OOV，不是對齊錯誤。")
    A("")

    A("## 二、時長合理性")
    A("")
    med = np.array([x.dur_median_ms for x in q], dtype=float)
    imp = np.array([x.n_implausible for x in q], dtype=float)
    tot = np.array([max(x.n_syllables, 1) for x in q], dtype=float)
    A(f"音節時長中位數各案為 {_f(np.nanmin(med))}–{_f(np.nanmax(med))} ms"
      f"（整體中位 {_f(np.nanmedian(med))} ms）。"
      f"不合理時長（< 50 ms 或 > 1 s）佔 **{100 * imp.sum() / tot.sum():.2f}%**。")
    A("")
    A("> 兒童華語音節一般落在 100–400 ms。中位數若明顯低於這個範圍，通常代表對齊把音節"
      "壓在了錯誤的位置；若明顯偏高，通常代表 OOV 造成的長段落被當成單一音節。")
    A("")

    A("## 三、兩本字典的分歧（誤差的真實下界）")
    A("")
    if cross and cross.get("n"):
        A(f"以 `{mfa['dictionary']}` 與對照字典各自對齊**同一段音訊**，"
          f"比較每個音節的起點。共比對 {cross['n_turns']} 個語句、{cross['n']} 個邊界：")
        A("")
        A(f"- 分歧中位數 **{_f(cross['median_ms'])} ms**")
        A(f"- 第 90 百分位 **{_f(cross['p90_ms'])} ms**")
        A(f"- 分歧 ≤ 20 ms 的比例 **{100 * cross['within_20ms']:.1f}%**")
        A("")
        A("> 兩本字典分歧的地方，至少有一本是錯的，所以這是誤差的**下界**而非估計值。"
          "反過來說，兩本一致也不保證都對——它們共用同一個聲學模型，會犯同樣的錯。"
          "這個數字的正確用途是回答 Q14：如果分歧很小，選哪本字典對結果無關緊要；"
          "如果很大，這個選擇就是必須先決定的事。")
    else:
        A("（未執行；需要同時安裝兩本字典並以 `--cross-dictionary` 啟用。）")
    A("")

    A("## 三b、隱含發音速率（壓縮對齊的自動偵測）")
    A("")
    A("若對齊把一句話的音節擠進實際語音的一小段，隱含的發音速率就會高到兒童做不到。"
      "**但這必須按語句長度分開讀**：短應答（「對啊」「會啊」）本來就講得快，"
      "而且依稽核 P1-2，nPVI 只用 ≥10 音節的語句。")
    A("")
    agg: dict[tuple, list] = {}
    long_v: list[float] = []
    for r in results:
        for bnd in r.rate_profile.get("bins", []):
            key = (bnd["min_syllables"], bnd["max_syllables"])
            agg.setdefault(key, []).append(bnd)
        lu = r.rate_profile.get("long_utterances") or {}
        if lu:
            long_v.append(lu)
    if agg:
        A("| 音節數 | 語句數 | 語速中位 | 音節時長中位 | 音節佔對齊區間 | >8/秒 的比例 |")
        A("|---|---|---|---|---|---|")
        for (lo, hi), items in sorted(agg.items()):
            label = f"{lo}–{hi - 1}" if hi else f"≥{lo}"
            A(f"| {label} | {sum(b['n'] for b in items)} | "
              f"{_f(np.median([b['median_rate'] for b in items]))} /秒 | "
              f"{_f(np.median([b['median_syllable_ms'] for b in items]), 0)} ms | "
              f"{100 * np.median([b['syllable_coverage'] for b in items]):.0f}% | "
              f"{100 * np.mean([b['frac_implausible'] for b in items]):.0f}% |")
        A("")
    if long_v:
        A(f"**節奏特徵實際用到的區間（≥10 音節）**："
          f"語速中位 {_f(np.median([x['median'] for x in long_v]))} /秒"
          f"（p10 {_f(np.median([x['p10'] for x in long_v]))}、"
          f"p90 {_f(np.median([x['p90'] for x in long_v]))}），"
          f"超過 8 音節/秒的比例 "
          f"{100 * np.mean([x['frac_implausible'] for x in long_v]):.0f}%。")
        A("")
    A("> **音節時長不可拿去跟文獻常模比。** 這裡的音節跨距只涵蓋 MFA 的音素序列，"
      "不含音節之間的靜音，所以會系統性短於文獻的華語音節值（約 200 ms）。"
      "**nPVI 是正規化指標、對尺度不變**，因此不受影響；但 `c_syl_dur_mean` 這類"
      "絕對時長特徵只能在本語料內部比較。")
    A("")

    A("## 四、能量低谷代理（**僅供偵測嚴重失敗**）")
    A("")
    prox = np.array([x.energy_proxy_median_ms for x in q], dtype=float)
    if np.isfinite(prox).any():
        A(f"音節邊界到最近能量低谷的距離，各案中位數 {_f(np.nanmin(prox))}–"
          f"{_f(np.nanmax(prox))} ms（整體 {_f(np.nanmedian(prox))} ms）。")
    else:
        A("（無足夠的相鄰音節邊界可比較。）")
    A("")
    A("> ⚠ **這不是準確度指標。** 華語音節邊界「常常」落在振幅低谷，但「常常」不是 "
      "ground truth。這個數字大到離譜（例如 > 50 ms）時代表對齊系統性偏移；"
      "數字小也**不能**用來宣稱達到 20 ms 精度。")
    A("")

    A("## 五、人工抽驗材料（P1-1 真正要求的東西）")
    A("")
    if spotcheck_dir:
        A(f"已輸出到 `{spotcheck_dir}`：每個抽樣語句一組 `.wav` ＋ `.TextGrid`，"
          f"邊界已重新對到片段起點，可直接用 Praat 開啟；每個案另有 `manifest.tsv` 列出"
          f"語句 id、原始時間、音節數與逐字稿內容。")
        A("")
        A("**做法**：在 Praat 同時開啟 wav 與 TextGrid，逐一檢視音節邊界是否落在正確位置，"
          "記錄明顯錯位的段落與偏移量。稽核 P1-1 要求抽驗 30–50 段、邊界誤差中位數 < 20 ms。"
          "**這一步只有人能做**，程式無法代勞。")
    else:
        A("（未輸出。）")
    A("")

    A("## 六、已知限制與尚未接上的東西")
    A("")
    A("- **變調實現（`c_sandhi_realisation`）目前恆為 0**。這個特徵需要比較「本調」與"
      "「實際產出的調」，MFA 給的是後者；本調要另外查字典中該詞的單獨發音。介面已預留"
      "（`SyllableSpan.underlying_tone`），但尚未接上。")
    A("- **MFA 內建的中文斷詞未使用**（`--no_tokenization`）。原因有二：它用的是 "
      "`spacy-pkuseg`（北京大學），屬 R4 禁用的中國來源；而且我們已用 CKIP 斷過詞，"
      "讓第二個斷詞器重切會使 E 流的詞單位與 C 流的對齊單位不一致。")
    A("- **OOV 目前直接落到 `spn`**。已下載 `mandarin_taiwan_mfa` 的 G2P 模型，"
      "可用 `--g2p_model_path` 為字典外詞產生發音，但尚未啟用——先量出 OOV 有多嚴重再決定。")
    A("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
