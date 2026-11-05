"""Turn the review answers into the numbers the audit asks for.

Reads ``key.json`` (never served) and ``answers.jsonl``; writes ``review_report.md`` with
groups and coded items only -- no case names. The figures are measurement-quality checks
the audit requires by group (P1-1 asks for alignment failure by group explicitly); none
of them compares the children's features between groups.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..normalize import strip_punctuation

__all__ = ["latest_answers", "char_edits", "agreement_ci", "score_package"]

GROUPS = ("TD", "ASD")


def latest_answers(path: Path) -> dict[str, dict[str, dict]]:
    """task -> item id -> the most recent answer. The log is append-only, so a corrected
    answer is simply a later line."""
    out: dict[str, dict[str, dict]] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.setdefault(rec["task"], {})[rec["id"]] = rec["answer"]
    return out


def char_edits(ref: str, hyp: str) -> int:
    """Levenshtein distance between the two texts with punctuation and spaces removed."""
    a, b = strip_punctuation(ref), strip_punctuation(hyp)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def agreement_ci(edits: Sequence[int], chars: Sequence[int], n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Character agreement 1 - sum(edits)/sum(chars), with a 95% bootstrap interval that
    resamples whole sentences -- characters within a sentence are not independent."""
    e, c = np.asarray(edits, float), np.asarray(chars, float)
    if c.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    point = 1.0 - e.sum() / c.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    boots = 1.0 - e[idx].sum(axis=1) / np.maximum(c[idx].sum(axis=1), 1e-9)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _pct(x: float) -> str:
    return "—" if not np.isfinite(x) else f"{100 * x:.1f}%"


# ------------------------------------------------------------------------------ report
def score_package(review_dir: Path) -> tuple[Path, dict]:
    from ..features.e_text import _overlap_ratio

    review_dir = Path(review_dir)
    key = json.loads((review_dir / "key.json").read_text(encoding="utf-8"))["key"]
    ans = latest_answers(review_dir / "answers.jsonl")
    L: list[str] = []
    A = L.append
    metrics: dict = {}

    A("# 第 5 步人工驗證結果")
    A("")
    A(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}。只列組別與片段代號，不含任何姓名。")
    A("")
    A("> 檢查者知道每位孩子的診斷，稽核清單要求的是對診斷設盲的檢查者。片段已打亂並以代號"
      "呈現，可以降低、但無法消除這個偏誤，引用時須一併揭露。")
    A("")

    # ---- mfa (P1-1): one judgement per utterance (D13) ----
    A("## 一、音節對齊（稽核 P1-1，精簡為每句一個判斷）")
    A("")
    A("| 組別 | 已判斷 | 大致對 | 明顯切錯 | 聽不清楚 | 切錯比例 |")
    A("|---|---|---|---|---|---|")
    for g in GROUPS:
        c = Counter()
        for code, k in key["mfa"].items():
            a = ans.get("mfa", {}).get(code)
            if k["group"] == g and a is not None:
                st = a.get("status")
                c["aligned" if st == "ok" else st] += 1      # "ok": the first version's save
        judged = c["aligned"] + c["misaligned"]
        rate = c["misaligned"] / judged if judged else float("nan")
        A(f"| {g} | {sum(c.values())} | {c['aligned']} | {c['misaligned']} | {c['cannot']} | {_pct(rate)} |")
        metrics[f"mfa_{g}"] = {"aligned": c["aligned"], "misaligned": c["misaligned"],
                               "cannot": c["cannot"], "misaligned_rate": rate}
    A("")
    A("切錯比例＝明顯切錯 ÷（大致對＋明顯切錯）。兩組差很多時，對齊失敗本身就會洩漏組別（稽核 P1-1）；"
      "切錯比例高時，依賴對齊的節奏類特徵不能當真。稽核原訂的毫秒級邊界誤差（中位數 < 20 ms）沒有做（D13）。")
    A("")

    # ---- speaker (P0-2) ----
    A("## 二、誰在說話（稽核 P0-2：一致率 ≥ 95%）")
    A("")
    A("| 組別 | 已判斷 | 與逐字稿一致 | 一致率 | 聽不出來 |")
    A("|---|---|---|---|---|")
    for g in GROUPS:
        n = agree = unsure = 0
        for code, k in key["speaker"].items():
            a = ans.get("speaker", {}).get(code)
            if k["group"] != g or a is None:
                continue
            if a["speaker"] == "unsure":
                unsure += 1
                continue
            n += 1
            agree += a["speaker"] == k["label"]
        A(f"| {g} | {n + unsure} | {agree} | {_pct(agree / n if n else float('nan'))} | {unsure} |")
        metrics[f"speaker_{g}"] = {"n": n, "agree": agree, "unsure": unsure}
    A("")

    # ---- echo (P0-1) ----
    A("## 三、仿說段落的轉錄獨立性（稽核 P0-1）")
    A("")
    A("「重疊度」＝孩子這句話的字詞有多少出現在施測者前一句裡（0–1）。原稿的重疊度若明顯高於"
      "你盲聽寫下的，表示原稿可能把施測者的話照抄進孩子那一句。")
    A("")
    A("| 片段 | 組別 | 原稿重疊度 | 你的重疊度 | 差距 | 原稿與你的字差 |")
    A("|---|---|---|---|---|---|")
    gaps = []
    for code, k in sorted(key["echo"].items(), key=lambda kv: kv[1]["group"]):
        a = ans.get("echo", {}).get(code)
        if a is None or not a.get("text", "").strip():
            continue
        mine = _overlap_ratio(a["text"], k["examiner_text"])
        gap = k["overlap"] - mine if np.isfinite(mine) else float("nan")
        gaps.append(gap)
        A(f"| {code} | {k['group']} | {k['overlap']:.2f} | {mine:.2f} | {gap:+.2f} | "
          f"{char_edits(k['child_text'], a['text'])} |")
    flagged = sum(1 for x in gaps if np.isfinite(x) and x >= 0.2)
    A("")
    A(f"差距 ≥ 0.20（可能照抄）的片段：{flagged}／{len(gaps)}。")
    A("")
    metrics["echo"] = {"n": len(gaps), "flagged": flagged}

    # ---- a12 (P0-F5) ----
    A("## 四、音量變化的來源（稽核 P0-F5 可行性檢查）")
    A("")
    labels = {"real": "真的改變說話音量", "distance": "離麥克風遠近不同",
              "both": "兩者都有", "unsure": "無法判斷"}
    A("| 組別 | " + " | ".join(labels.values()) + " |")
    A("|---" * (len(labels) + 1) + "|")
    for g in GROUPS:
        c = Counter(ans.get("a12", {}).get(code, {}).get("judgement")
                    for code, k in key["a12"].items() if k["group"] == g)
        A(f"| {g} | " + " | ".join(str(c.get(x, 0)) for x in labels) + " |")
        metrics[f"a12_{g}"] = {x: c.get(x, 0) for x in labels}
    A("")
    A("稽核規則：若多數是「距離」或「無法判斷」，音量跟隨（A12）整項擱置，並在報告記錄原因。")
    A("")

    # ---- transcribe (P0-1) ----
    if not key.get("transcribe"):
        A("## 五、雙人轉錄（稽核 P0-1）")
        A("")
        A("未做：轉錄信度未評估（DEVIATIONS D13）。")
        A("")
        out = review_dir / "review_report.md"
        out.write_text("\n".join(L) + "\n", encoding="utf-8")
        return out, metrics
    A("## 五、雙人轉錄（稽核 P0-1：字元一致率 ≥ 90%、語者一致率 ≥ 95%）")
    A("")
    A("| 組別 | 完成句數／總句數 | 字元一致率（95% 信賴區間） | 語者一致率 |")
    A("|---|---|---|---|")
    for g in GROUPS:
        total = done = spk_n = spk_ok = 0
        per_edits: list[int] = []
        per_chars: list[int] = []
        for code, k in key["transcribe"].items():
            if k["group"] != g:
                continue
            total += 1
            a = ans.get("transcribe", {}).get(code)
            if a is None:
                continue
            done += 1
            if a.get("speaker") not in (None, "unsure"):
                spk_n += 1
                spk_ok += a["speaker"] == k["label"]
            ref = strip_punctuation(k["text"])
            if ref and not a.get("unclear"):
                per_edits.append(char_edits(ref, a.get("text", "")))
                per_chars.append(len(ref))
        agree, lo, hi = agreement_ci(per_edits, per_chars)
        A(f"| {g} | {done}／{total} | {_pct(agree)}（{_pct(lo)}–{_pct(hi)}） | "
          f"{_pct(spk_ok / spk_n if spk_n else float('nan'))} |")
        metrics[f"transcribe_{g}"] = {"done": done, "total": total, "char_agreement": agree}
    A("")
    A("抽樣：每組 150 句，平均分散在該組每一場錄音（取代稽核原訂的 20–25% 場次，見 DEVIATIONS D13）。"
      "完成比例不足時，一致率只代表已完成的部分。")
    A("")

    out = review_dir / "review_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out, metrics
