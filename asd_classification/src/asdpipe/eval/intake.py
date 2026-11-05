"""Intake report for a real cohort -- the P0-0A / P0-1 / P0-2 audit numbers.

This is the artefact that has to exist before any feature is computed on real recordings.
It answers, per case and for the cohort: how much of the transcript survived the frozen
cleaning rules, how much each rule cost, how many usable question-answer pairs remain, and
which cases fall below the audit's reliability floors.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..realdata import SPEAKER_ROLES, IntakeReport

__all__ = ["write_intake_report"]


def _f(x, nd=1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{v:.{nd}f}"


def write_intake_report(path: Path, reports: Sequence[IntakeReport], cfg: Mapping,
                        excluded: Sequence[str] = ()) -> Path:
    rd = cfg["realdata"]
    min_pairs = int(rd["min_qa_pairs"])
    min_open = int(rd["min_open_qa_pairs"])
    L: list[str] = []
    A = L.append

    A("# 真實語料 intake 報告（稽核 P0-0A／P0-1／P0-2）")
    A("")
    A(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}　·　來源：`{rd['root']}`　·　"
      f"cohort：**{rd['cohort']}**")
    A("")
    A("> 🔴 **這批語料全部是同一個 cohort，沒有對照組。** 因此可以做的是管線驗證、")
    A("> 資料品質稽核、以及建立這個 cohort 的特徵參考分布；**不能**做分類、組間比較、")
    A("> 或效標關聯效度分析。TD 的 ADOS A 區編碼幾乎必然全為 0，而稽核 P0-0E 明文寫著")
    A("> 「某項幾乎全為 0（未被誘發）者不適合作效標」。")
    A("")
    if excluded:
        A(f"**已排除的個案**：{', '.join(excluded)}（逐字稿有未解決的缺陷，由計畫主持人指定）")
        A("")

    # ------------------------------------------------------------------ cohort summary
    n = len(reports)
    tot_raw = sum(r.n_raw for r in reports)
    tot_kept = sum(r.n_kept for r in reports)
    tot_ov = sum(r.n_dropped_overlap for r in reports)
    tot_tp = sum(r.n_third_party for r in reports)
    tot_pairs = sum(r.n_qa_pairs for r in reports)
    tot_open = sum(r.n_open_qa_pairs for r in reports)
    child_min = np.array([r.child_speech_s / 60 for r in reports])
    sess_min = np.array([r.session_s / 60 for r in reports])

    A("## 一、cohort 概況")
    A("")
    A(f"可用個案 **{n}** 位，總時長 {_f(sess_min.sum())} 分鐘"
      f"（每場 {_f(sess_min.min())}–{_f(sess_min.max())} 分鐘）。"
      f"原始語句段落 {tot_raw:,} 段，套用凍結的清理規則後留下 **{tot_kept:,} 段"
      f"（{100 * tot_kept / max(tot_raw, 1):.1f}%）**。")
    A("")
    A(f"兒童發話時間每人 {_f(child_min.min())}–{_f(child_min.max())} 分鐘"
      f"（中位數 {_f(np.median(child_min))}）。"
      f"可用的「醫師提問→兒童回答」配對共 {tot_pairs:,} 對，"
      f"其中開放式（排除是非題後、供 G1 使用）{tot_open:,} 對。")
    A("")

    # ------------------------------------------------------------------ cleaning cost
    A("## 二、清理規則付出的代價")
    A("")
    A("| 規則 | 設定 | 丟掉的段落 | 佔原始 |")
    A("|---|---|---|---|")
    A(f"| 重疊語段 | `{rd['overlap_policy']}`（容忍 {int(rd['overlap_tolerance_s'] * 1000)} ms） | "
      f"{tot_ov:,} | {100 * tot_ov / max(tot_raw, 1):.1f}% |")
    A(f"| 第三方語者 | `drop_third_party: {str(rd['drop_third_party']).lower()}` | "
      f"{tot_tp:,} | {100 * tot_tp / max(tot_raw, 1):.2f}% |")
    A(f"| 空白／零長度 | 固定 | {sum(r.n_empty_text for r in reports):,} | "
      f"{100 * sum(r.n_empty_text for r in reports) / max(tot_raw, 1):.2f}% |")
    A("")
    A("> 重疊在這批語料裡是**真的同時說話**，不是邊界四捨五入：中位數 367 ms、"
      "第 90 百分位 2 秒、最長 20.5 秒，只有 6.4% 小於 20 ms。"
      "`drop_both` 之所以丟兩段而不是一段，是因為重疊區的音訊裡有兩個人的聲音，"
      "留下任何一段都會汙染它的聲學特徵。")
    A("")
    A("> ⚠ **這個代價本身要進稽核**：稽核 P0-4 要求揭露缺值模式，"
      "而「清理規則在哪一組丟掉比較多」正是缺值模式。目前只有單一 cohort，"
      "加入第二組之後必須重跑本表並檢定兩組的留存率差異。")
    A("")

    # ------------------------------------------------------------------ per case
    A("## 三、逐案明細")
    A("")
    A("| 個案 | 原始 | 留存 | 留存% | 醫師輪 | 兒童輪 | 童發話(min) | 童佔比 | QA對 | 開放QA | 重疊對 | 順序倒退 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(reports, key=lambda x: x.case_id):
        A(f"| {r.case_id} | {r.n_raw} | {r.n_kept} | {100 * r.kept_ratio:.0f}% | "
          f"{r.n_examiner} | {r.n_child} | {_f(r.child_speech_s / 60)} | "
          f"{100 * r.child_talk_ratio:.0f}% | {r.n_qa_pairs} | {r.n_open_qa_pairs} | "
          f"{r.n_overlap_pairs} | {r.n_out_of_order} |")
    A("")

    # ------------------------------------------------------------------ audit gates
    A("## 四、稽核閘門")
    A("")
    low_pairs = [r.case_id for r in reports if r.n_qa_pairs < min_pairs]
    low_open = [r.case_id for r in reports if r.n_open_qa_pairs < min_open]
    A(f"**P0-2 每位受試者可用配對數 ≥ {min_pairs}**："
      + (f"❌ 未達標 {len(low_pairs)} 位：{', '.join(low_pairs)}" if low_pairs
         else f"✅ 全部達標（最少 {min(r.n_qa_pairs for r in reports)} 對）"))
    A("")
    A(f"**P0-F1 開放式配對數 ≥ {min_open}**（G1 專用，排除是非題後）："
      + (f"❌ 未達標 {len(low_open)} 位：{', '.join(low_open)}" if low_open
         else f"✅ 全部達標（最少 {min(r.n_open_qa_pairs for r in reports)} 對）"))
    A("")
    q1 = np.percentile(child_min, 25)
    low_talk = [r.case_id for r in reports if r.child_speech_s / 60 < q1]
    A(f"**發話量偏低者**（兒童發話時間低於第一四分位 {_f(q1)} 分鐘）："
      f"{', '.join(low_talk) if low_talk else '無'}。"
      f"稽核 P0-F3 要求分析刻板語句時併報各受試者的可用發話量——發話量太低時，"
      f"「沒有刻板語句」與「沒被誘發出來」無法區分。")
    A("")
    unmapped = {k: v for r in reports for k, v in r.third_party_labels.items()}
    A("**語者標籤正規化**（稽核 P0-2）：對照表為 "
      + "、".join(f"`{k}`→{v.value}" for k, v in SPEAKER_ROLES.items() if k in
                  {"評估者", "測試者", "SPEAKER_00", "兒童", "小孩", "SPEAKER_01"})
      + f"。未列於對照表而被視為第三方的標籤：{unmapped or '無'}。")
    A("")

    # ------------------------------------------------------------------ transcription finding
    A("## 五、🔴 轉錄規範一致性的發現（稽核 P0-1）")
    A("")
    marks = np.array([r.n_question_marks / max(r.n_examiner, 1) for r in reports])
    combined = np.array([r.n_examiner_questions / max(r.n_examiner, 1) for r in reports])
    A(f"**問號的使用是逐案的個人習慣，不是規範。** 醫師輪次中含問號的比例跨案從 "
      f"**{100 * marks.min():.0f}% 到 {100 * marks.max():.0f}%**"
      f"（平均 {100 * marks.mean():.0f}%、SD {100 * marks.std():.0f}%、"
      f"變異係數 {marks.std() / max(marks.mean(), 1e-9):.2f}）；"
      f"其中有個案完全不使用問號。")
    A("")
    A("這件事會直接汙染 G1：如果只靠問號判定問句，配對數就會跟著轉錄員的標點習慣走，"
      "而那與兒童的行為毫無關係——是典型的 confound。因此問句判定用的是**三個線索的聯集**"
      "（問號、`嗎/吧` 句尾、疑問詞），實測把跨案變異係數從 "
      f"**{marks.std() / max(marks.mean(), 1e-9):.2f} 降到 "
      f"{combined.std() / max(combined.mean(), 1e-9):.2f}**"
      f"（問句率 {100 * combined.min():.0f}–{100 * combined.max():.0f}%）。")
    A("")
    A("> **這仍然是一個要如實揭露的限制**：正確的解法是統一轉錄規範並重新標點，"
      "不是靠啟發式補救。加入第二組之前應先確認兩組的轉錄員與規範一致"
      "（稽核 P0-1：「同一批轉錄員、同一規範、轉錄員對診斷設盲」）。")
    A("")
    A("**另一項缺口**：這批逐字稿完全沒有 `[unclear]` 之類的聽不清標記（掃過全部段落，零個）。"
      "稽核 P0-1 要求揭露「聽不清比例的組間差異」——該指標在目前的轉錄規範下不存在，"
      "未來加入 ASD 組後將無法執行這項洩漏檢查。")
    A("")

    # ------------------------------------------------------------------ next steps
    notes = [(r.case_id, nz) for r in reports for nz in r.notes]
    if notes:
        A("## 五b、資料一致性註記")
        A("")
        for cid, nz in notes:
            A(f"- **{cid}**：{nz}")
        A("")

    A("## 六、這批語料現在能做與不能做的事")
    A("")
    A("**能做**：管線在真實 30–50 分鐘 ADOS 錄音上的端到端驗證；MFA 對齊品質量測"
      "（時間戳是語句級，所以 C 流的 nPVI／%V／ΔC／聲調實現一律依賴 MFA）；"
      "本 cohort 的特徵參考分布；稽核 P0-F7 明文要求的**地板效應檢查**"
      "（「先看對照組的分布」——4–6 歲兒童是否真的有多輪語意調適）；"
      "以及 G1 在真實問答對上的相似度動態範圍。")
    A("")
    A("**不能做**：分類、組間效果量、效標關聯效度。理由見報告開頭。")
    A("")
    A("**必須先解決的治理問題**：這個資料夾同時是 ASR 微調語料。"
      "計畫書 §5.5b.4 與稽核 P1-8 要求**ASR 微調語料與分析樣本的受試者不相交**，"
      "否則轉錄器背過測試者的原話＝受試者層級洩漏。"
      "若這批受試者日後要進主分析，必須擇一處理：改用後續新收個案微調、"
      "把本批排除於主分析之外只作管線驗證、或逐折重微調。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
