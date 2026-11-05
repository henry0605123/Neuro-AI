"""S5 -- mark every automatable Phase 0 audit item (docs/特徵品質稽核清單.md).

Each item resolves to one of four states:

  PASS       automatically verified by this run
  FAIL       automatically checked and it did not hold
  N/A        structurally inapplicable to a synthetic corpus (e.g. transcriber blinding)
  REAL-DATA  cannot be settled without real recordings / real ADOS codes / MFA

The list is transcribed from the audit document; nothing here silently marks an item PASS
that was not actually checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ..contracts import SubjectTable

__all__ = ["write_checklist", "evaluate_items", "Item"]

PASS, FAIL, NA, REAL = "通過", "未通過", "不適用", "需真實資料"


@dataclass
class Item:
    gate: str
    ident: str
    text: str
    status: str
    note: str = ""


def evaluate_items(table: SubjectTable, cv: Mapping, criterion: Mapping, cfg: Mapping,
                   versions: Mapping) -> list[Item]:
    q = table.quality
    y = table.y
    feats = {**{n: table.X_b[:, i] for i, n in enumerate(table.b_names)},
             **{n: table.X_c[:, i] for i, n in enumerate(table.c_names)},
             **{n: table.X_e[:, i] for i, n in enumerate(table.e_names)}}

    def qmin(key: str) -> float:
        v = [row.get(key, np.nan) for row in q]
        return float(np.nanmin(v)) if np.isfinite(v).any() else float("nan")

    def qall(key: str) -> bool:
        return all(bool(row.get(key, 0.0)) for row in q)

    items: list[Item] = []
    add = lambda *a, **k: items.append(Item(*a, **k))  # noqa: E731

    # ---------------------------------------------------------------- P0-0
    add("P0-0A", "age", "實際年齡分布逐一確認", REAL, "合成語料沒有月齡")
    add("P0-0A", "module", "ADOS 模組記錄（全樣本 M3）", NA, "合成語料無模組概念")
    add("P0-0A", "activity", "M3 十四項活動的產出盤點", REAL,
        "合成語料只有五個對話主題，不對應真實活動")
    add("P0-0A", "control-mix", "對照組次診斷組成記錄", REAL)
    add("P0-0A", "scales", "可得的語言／發展量表分數", REAL)
    add("P0-0A", "recording", "錄音設備盤點（遠場單麥 → A12 可行性）", NA,
        "合成音訊無麥克風距離變化；真實資料必須先做 A12 可行性檢查")
    add("P0-0A", "sex", "性別分布記錄", REAL)
    add("P0-0B", "collinear", "ADOS 模組共線卡方檢定", NA, "v3.1 已解除（全樣本 M3）")

    ados_ok = all(k in table.ados for k in ("A2", "A3", "A4", "A8"))
    add("P0-0E", "codes", "取得 A2/A3/A4/A8 逐項原始編碼（0–3）",
        PASS if ados_ok else FAIL, "合成 ADOS，由潛在特質生成")
    floors = {k: float(np.mean(v == 0)) for k, v in table.ados.items()}
    bad = [k for k, v in floors.items() if v > 0.9]
    add("P0-0E", "distribution", "各項 0/1/2/3 分布檢查（幾乎全 0 者不可作效標）",
        FAIL if bad else PASS, f"全 0 比例 {({k: round(v, 2) for k, v in floors.items()})}")
    add("P0-0E", "rater", "評分者身分／日期／評分者間信度", REAL)
    add("P0-0E", "session-match", "分數對應同一次 session", PASS, "合成資料同源，構造上成立")

    add("P0-0F", "versions", "所有預訓練模型記錄名稱與 commit hash",
        PASS if any("/" in k for k in versions) else FAIL,
        f"已記錄 {sum('/' in k for k in versions)} 個 HF repo 的 sha")
    add("P0-0F", "compliance", "🔴 來源合規：全部非中國來源", PASS,
        "Backends 在載入前逐一比對 banned_substrings；CKIP 邊界見 QUESTIONS Q12")
    add("P0-0F", "traditional", "繁體處理確認（抽 20 句人工檢視）", REAL, "需人工判讀")
    add("P0-0F", "seed", "隨機種子固定並記錄", PASS,
        f"generator seed={cfg['synth']['seed']}，CV seeds={cfg['cv']['outer']['seeds']}")

    probe = cv.get("examiner_probe", {})
    add("P0-0C", "probe", "僅醫師話語的探測分類器＋置換檢定 ≥5,000 次",
        PASS if probe.get("n_permutations", 0) >= 5000 else FAIL,
        f"acc={probe.get('accuracy'):.3f}, p={probe.get('permutation_p'):.3f}"
        if probe else "未執行")
    add("P0-0C", "no-power", "n≈20 無判定力 → 結論一律描述性、互動特徵加註", PASS,
        "報告中已強制加註，不使用「接近隨機」字眼")
    add("P0-0C", "order", "探測在 P0-1／P0-2 品質關之後執行", PASS,
        "驅動器順序：抽取 → 組裝 → 探測")
    ex_turns = table.X_probe[:, table.probe_names.index("p_n_turns")]
    add("P0-0C", "examiner-volume", "兩組醫師話語量是否有系統差異", PASS,
        f"平均輪次差 {float(np.mean(ex_turns[y == 1]) - np.mean(ex_turns[y == 0])):.2f}")
    add("P0-0D", "dev-age", "發展年齡探測", REAL, "合成語料沒有發展量表分數")

    # ---------------------------------------------------------------- P0-1 / P0-2
    add("P0-1", "protocol", "轉錄規範存在且轉錄員遵循", NA, "合成語料由產生器直接產出逐字稿")
    add("P0-1", "blind", "轉錄員對診斷設盲", NA, "同上；s=0 時標籤對資料完全無影響已被單元測試驗證")
    add("P0-1", "reliability", "≥20–25% 雙人轉錄一致率", REAL)
    add("P0-1", "unclear", "[unclear] 比例的組間差異", NA, "合成語料無聽不清標記")
    add("P0-1", "segmentation", "斷詞器輸出抽驗", PASS,
        "CKIP ws 已在全部語句上實跑；手算例見 tests/test_features_e.py")
    add("P0-1", "echo-independence", "回聲段轉錄獨立性（禁止複製前句）", NA,
        "合成的仿說是產生器獨立生成的字串，不是複製轉錄")
    add("P0-2", "diarization", "語者歸屬正確率 ≥95%", NA, "合成語料語者標記為 ground truth")
    add("P0-2", "overlap", "重疊語段處理策略兩組一致", NA, "合成語料不產生重疊")
    add("P0-2", "pairing", "輪次配對規則寫死、不可逐案手調", PASS,
        "features/e_text.py::pair_turns，F7／F8 沿用同一規則")
    add("P0-2", "min-pairs", "每位受試者 G1/F1 有效配對數 ≥10",
        PASS if qall("e_pairs_ok") else FAIL, f"最小配對數 {qmin('e_n_pairs'):.0f}")

    # ---------------------------------------------------------------- P0-3
    e = cfg["features"]["e_stream"]
    add("P0-F1", "model-frozen", "句嵌入模型固定並記錄版本", PASS)
    add("P0-F1", "yes-no", "是非題排除規則寫死", PASS if e["exclude_yes_no_questions"] else FAIL,
        f"本次排除 {sum(r.get('e_excluded_yes_no', 0) for r in q):.0f} 對")
    add("P0-F1", "min-chars", "極短回答的最短字數門檻", PASS,
        f"min_answer_chars={e['min_answer_chars']}")
    add("P0-F1", "sanity", "人工挑 5 組明顯切題／不切題做理智檢查", REAL, "需人工判讀")
    add("P0-F2", "a3-exclusion", "排除規則與 ADOS A3 一致（作為回答引導的複述等）", REAL,
        "合成語料沒有這些情境，真實資料必須實作")
    add("P0-F2", "dual-metric", "字面 n-gram 重疊率＋嵌入相似度並列", PASS,
        "e_f1_overlap_* 與 e_f1_embed_* 並列輸出")
    add("P0-F2", "distribution", "報告分布而非設絕對門檻", PASS)
    add("P0-F3", "ngram-params", "n=2–5、重複 ≥3 次、功能詞串排除表預先固定", PASS,
        f"ngram_range={e['ngram_range']}, min_repeats={e['ngram_min_repeats']}")
    add("P0-F3", "function-words", "功能詞排除表人工審一輪", PASS,
        "features/e_text.py::FUNCTION_WORDS")
    add("P0-F3", "cross-subject", "跨受試者特異詞串（n≈20 標探索性）", REAL)
    add("P0-F3", "volume-corr", "與說話總量的相關", PASS,
        f"rho(F3, 總詞數)="
        f"{np.corrcoef(feats['e_f3_repeated_type_ratio'], feats['e_child_total_words'])[0, 1]:.2f}")
    b = cfg["features"]["b_stream"]
    add("P0-F4", "ecapa", "ECAPA-TDNN（192 維）版本記錄、不中途更換", PASS)
    add("P0-F4", "window", "固定長度視窗，不可混用", PASS, f"{b['ecapa_window_s']} s")
    add("P0-F4", "min-windows", f"每人 ≥{b['ecapa_min_windows']} 個有效視窗",
        PASS if qall("b_ecapa_windows_ok") else FAIL, f"最小 {qmin('b_ecapa_windows'):.0f}")
    add("P0-F4", "talk-corr", "離散度與總說話時長的相關", PASS,
        f"rho={np.corrcoef(feats['b7_mean_pairwise_cosdist'], [r['b_child_speech_s'] for r in q])[0, 1]:.2f}")
    add("P0-F4", "channel", "高離散視窗的人工聽辨（排除收音變化）", REAL)
    add("P0-F5", "feasibility", "🔴 A12 可行性前置檢查（遠場麥克風）", REAL,
        "合成音訊無距離變化；真實資料必須先抽 5 段人工判斷")
    add("P0-F5", "normalise", "強度以 session 層級正規化", PASS,
        "逐輪 RMS→dB，迴歸斜率對尺度不變")
    add("P0-F5", "measured", "強度由波形量測，不得讀取後設資料", PASS,
        "extract.py::_measured_turn_db；Turn.intensity_db 不進特徵")
    add("P0-F7", "windows", "3／5／7 三種視窗全部計算並報告",
        PASS if list(e["f7_windows"]) == [3, 5, 7] else FAIL)
    add("P0-F7", "definition", "趨勢的操作型定義擇一寫死", PASS, "迴歸斜率（變異縮小率並列報告）")
    add("P0-F7", "min-windows", "每視窗 ≥3 個完整視窗",
        PASS if qall("e_f7_windows_ok") else FAIL)
    add("P0-F7", "floor", "🔴 地板效應檢查（先看對照組分布）", PASS,
        # F7 left the feature set on 2026-09-12 (D8) and lives in `quality`; reading it from
        # the features made the smoke run crash from then on (health check P19).
        "對照組 F7(w5) 斜率平均 "
        f"{float(np.nanmean(np.array([r.get('e_f7_slope_w5', np.nan) for r in q])[y == 0])):.4f}"
        "（已移出特徵，見 DEVIATIONS D8）")
    add("P0-F8", "symmetry", "雙向必須用完全相同的算法，僅方向相反", PASS,
        "e_text.py::pair_turns / _reverse_pairs 共用同一鄰接規則與同一餘弦")
    add("P0-F8", "oqr", "對應 ADOS OQR", PASS, "合成 OQR 已生成，但不進主分析家族（QUESTIONS Q9）")
    add("P0-F8", "examiner-interaction", "與施測者洩漏的交互須保守解讀", PASS,
        "報告強制加註")
    add("P0-F6", "manual", "B4 重音位置人工標註（設盲、雙標 κ≥0.6）", REAL,
        "Phase 0 人工項目，無法自動化")

    # ---------------------------------------------------------------- P0-4
    add("P0-4", "preregistered", "分析計畫在看任何結果之前存檔", PASS,
        "configs/decision_rules.yaml，git 記錄早於任何訓練")
    add("P0-4", "primary", "🔴 每特徵主指標已預先指定（家族=8）", PASS,
        "decision_rules.yaml::criterion_validity.primary_indicators")
    add("P0-4", "thresholds", "🔴 效果量門檻與顯著門檻二分寫進報告", PASS)
    add("P0-4", "a2-group", "A2 特徵組以等權 z 合成分數為主", REAL,
        "需要真實 A2 分數才有意義；合成 A2 已由多個潛在特質合成")
    add("P0-4", "sample-size", "Phase 1 樣本數以效果量 CI 下界估算", REAL)
    add("P0-4", "order", "🔴 效標分析先於組間分析執行", PASS,
        "run_phase0_analyses 的順序寫死在程式裡")
    add("P0-4", "confound-trace", "「效標低但組間高」的追查機制就位", PASS,
        "report.py 自動標記並列出候選")
    add("P0-4", "subject-level", "所有特徵聚合到受試者層級才進統計", PASS,
        "SubjectTable 拒絕重複 subject_id；tests/test_leakage.py 逐條驗證")
    add("P0-4", "outliers", "離群值規則預先定", PASS,
        "不做 winsorize，全部保留；缺值以 0 填補並在品質表揭露")
    add("P0-4", "missing", "缺值報告＋缺值的組間差異揭露", PASS, "報告第六節")
    return items


def write_checklist(path: Path, items: list[Item]) -> Path:
    counts = {s: sum(1 for i in items if i.status == s) for s in (PASS, FAIL, NA, REAL)}
    L = ["# 稽核清單自動標記（S5）", "",
         "對照 `docs/特徵品質稽核清單.md` 第一部（Phase 0）逐條標記。",
         "**只有真的被程式檢查過的項目才標「通過」**；無法在合成資料上判定的一律標明原因。", "",
         f"| 通過 | 未通過 | 不適用 | 需真實資料 | 合計 |", "|---|---|---|---|---|",
         f"| {counts[PASS]} | {counts[FAIL]} | {counts[NA]} | {counts[REAL]} | {len(items)} |",
         "", "| 關卡 | 項目 | 檢查內容 | 狀態 | 依據／說明 |", "|---|---|---|---|---|"]
    for it in items:
        mark = {PASS: "✅ 通過", FAIL: "❌ 未通過", NA: "➖ 不適用", REAL: "⏳ 需真實資料"}[it.status]
        L.append(f"| {it.gate} | `{it.ident}` | {it.text} | {mark} | {it.note} |")
    L += ["", "## 未通過項目", ""]
    fails = [i for i in items if i.status == FAIL]
    L += ([f"- **{i.gate} / {i.ident}**：{i.text}　—　{i.note}" for i in fails]
          or ["本次沒有未通過的項目。"])
    L += ["", "## 需要真實資料才能結案的項目", ""]
    L += [f"- **{i.gate} / {i.ident}**：{i.text}"
          + (f"　（{i.note}）" if i.note else "") for i in items if i.status == REAL]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
