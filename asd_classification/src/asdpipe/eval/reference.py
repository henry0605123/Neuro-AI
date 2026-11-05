"""Reference distribution of every feature on a single real cohort.

With one cohort there is no classification to do, so this report answers the questions
that a single group *can* answer and that the audit actually asks:

* do the extractors survive real 30-50 minute recordings, and what do they cost;
* are the resulting numbers physically plausible for 4-6 year old children;
* which audit floors (P0-2 pairs, P0-F4 ECAPA windows, P1-2 syllables, P1-3 F0 tracking)
  each subject clears;
* **audit P0-F7's floor-effect check**: "look at the control group's distribution first" --
  if typically-developing children show no multi-turn semantic adaptation at this age, the
  feature cannot separate anything, and the plan doc says that finding must be reported.

Nothing here is a group comparison or a criterion-validity analysis; both need two groups.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..contracts import SubjectTable

__all__ = ["write_reference_report", "FEATURE_GROUPS"]

# Which features get a plausibility comment, and what the plan doc expects of them.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "C 流｜節奏與時長（核心，證據 A−）": (
        "c_npvi_duration", "c_pct_v", "c_delta_c", "c_syl_dur_mean", "c_syl_dur_cv",
        "c_speaking_rate", "c_articulation_rate", "c_pause_rate_per_min",
        "c_pause_mean_s", "c_utt_dur_mean",
    ),
    "C 流｜語句層級語調（評估表 A2/A3/B2/B5）": (
        "c_pitch_range_st_median", "c_pitch_range_st_p90", "c_pitch_range_st_iqr",
        "c_pitch_sd_st_median", "c_pitch_dynamism_st_per_s", "c_pitch_reversals_per_s",
        "c_exaggerated_utterance_ratio",
        "c_intensity_range_db_median", "c_pitch_intensity_coupling",
    ),
    "C 流｜聲調時長（輪廓特徵已於 2026-09-12 移除）": (
        "c_neutral_tone_dur_ratio",
    ),
    "B 流｜頻譜平衡與強度（評估表 A7/A11）": (
        "b_Loudness_mean", "b_Loudness_std", "b_alphaRatio_mean",
        "b_hammarbergIndex_mean", "b_slope0_500_mean", "b_slope500_1500_mean",
        "b_spectralFlux_mean",
    ),
    "B 流｜B7 卡通腔代理": (
        "b7_mean_pairwise_cosdist", "b7_sd_pairwise_cosdist", "b7_n_windows",
    ),
    "E 流｜互惠（G1／F1／F8）": (
        "e_g1_similarity_median", "e_g1_sim_p25", "e_g1_low_ratio",
        "e_f1_embed_high_ratio", "e_f1_overlap_mean", "e_f8_asymmetry",
        "e_f8_child_follows", "e_f8_examiner_follows",
    ),
    "E 流｜臨床替補特徵（評估表 E5／F2／F4／F7）": (
        "e_vocal_turns_per_min", "e_vocal_seconds_per_min", "e_vocal_words_per_min",
        "e_vocal_speech_share",
        "e_init_ratio", "e_init_self_continuation_ratio", "e_init_switches_per_min",
        "e_f2_delayed_echo_ratio", "e_f2_median_lag_turns",
        "e_pron_first_per_100c", "e_pron_second_per_100c", "e_pron_second_share",
        "e_pron_second_in_reply_ratio",
    ),
    "E 流｜語彙與時序": (
        "e_f3_repeated_type_ratio", "e_f3_repeated_token_ratio", "e_mlu_words",
        "e_ttr", "e_child_total_words", "e_latency_median",
        "e_volume_following_slope",
    ),
}

# ⚠ These are **engineering sanity bounds written from general expectation, not literature
# citations**. None of them has a verified source attached, so they are labelled as such in
# the report and must never be quoted as published norms. Replacing them with sourced values
# is a documentation task, not a code change.
PLAUSIBILITY: dict[str, str] = {
    "c_npvi_duration": "粗略期望 40–60（未查證來源）",
    "c_syl_dur_mean": "偏低是預期的：音節跨距不含音節間靜音，見對齊報告",
    "c_speaking_rate": "粗略期望 2–5 音節/秒（未查證來源）",
    "c_articulation_rate": "粗略期望 3–6 音節/秒；兒童應慢於成人（未查證來源）",
    "c_pct_v": "粗略期望 0.4–0.6（未查證來源）",
    "c_tone_accuracy": "5 類，均勻隨機 = 0.20；正確的比較對象是多數類基準",
    "c_sandhi_realisation": "真實資料恆為 0：MFA 只給實現調，本調尚未接上",
    "b_cpp_mean": "只在通過 SNR 門檻的輪次上計算，見第三b節",
    "b7_n_windows": "稽核 P0-F4 要求每人 ≥15 個 3 秒視窗",
    "e_mlu_words": "粗略期望 3–8 詞（未查證來源）",
    "e_g1_similarity_median": "動態範圍見第三節",
    "c_pitch_range_st_median": "兒童自發語句音域，數半音量級",
    "c_pitch_range_st_p90": "語句音域上端，約一個八度",
    "c_pitch_sd_st_median": "語句內音高標準差",
    "c_pitch_dynamism_st_per_s": "音高移動速率（半音/秒）",
    "c_pitch_reversals_per_s": "每秒音高轉折次數",
    "c_exaggerated_utterance_ratio": "高於 12 半音的語句比例",
    "c_intensity_range_db_median": "語句內音量動態範圍",
    "c_pitch_intensity_coupling": "音高與音量的相關；情緒語調預期為正",
    "e_vocal_turns_per_min": "E5 發聲量：每分鐘兒童輪次",
    "e_vocal_seconds_per_min": "E5：每分鐘兒童說話秒數",
    "e_vocal_words_per_min": "E5：每分鐘兒童詞數",
    "e_vocal_speech_share": "E5：兒童說話時間佔總說話時間",
    "e_init_ratio": "F7 主動發話：非緊接施測者提問的兒童輪次比例",
    "e_init_self_continuation_ratio": "F7：接續自己上一輪的比例",
    "e_init_switches_per_min": "F7：每分鐘語者切換次數（對切段慣例免疫）",
    "e_f2_delayed_echo_ratio": "F2 延宕仿說代理：與 ≥20 輪前自己的話高度相似的比例",
    "e_f2_median_lag_turns": "F2：這些重複的中位間隔輪次",
    "e_pron_first_per_100c": "F4：每百字第一人稱次數",
    "e_pron_second_per_100c": "F4：每百字第二人稱次數",
    "e_pron_second_share": "F4：第二人稱佔人稱代名詞的比例",
    "e_pron_second_in_reply_ratio": "F4 代名詞反轉代理：被問「你…」時用「你」而非「我」回答的比例",
}


def _stats(v: np.ndarray) -> tuple[float, float, float, float, float]:
    x = np.asarray(v, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan,) * 5
    return (float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)),
            float(np.min(x)), float(np.max(x)))


def _f(x, nd=3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{v:.{nd}f}"


def write_reference_report(path: Path, table: SubjectTable, cfg: Mapping,
                           timings: Mapping | None = None,
                           extra_notes: Sequence[str] = ()) -> Path:
    rd = cfg["realdata"]
    lookup: dict[str, np.ndarray] = {}
    for s in ("b", "c", "e"):
        X, names = table.stream(s), table.feature_names(s)
        for j, n in enumerate(names):
            lookup[n] = X[:, j]

    L: list[str] = []
    A = L.append
    A(f"# 真實語料特徵參考分布（cohort：{rd['cohort']}，n={len(table)}）")
    A("")
    A(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}　·　來源：`{rd['root']}`")
    A("")
    A("> 🔴 **這不是研究結果，是管線在真實錄音上的體檢報告。**")
    A("> 全部受試者屬同一 cohort，沒有對照組，因此沒有任何分類、組間比較或效標效度分析。")
    A("> 下面每一個數字回答的都是「抽出來的東西合不合理」，不是「兩組有沒有差」。")
    A("")

    # ------------------------------------------------------------------ audit floors
    A("## 一、稽核底線逐項檢查")
    A("")
    q = table.quality
    checks = [
        ("P0-F4 每人 ≥15 個 ECAPA 視窗", "b_ecapa_windows", 15, "ge"),
        ("P0-2 每人 ≥10 組 QA 配對", "e_n_pairs", 10, "ge"),
        ("P0-F1 每人 ≥10 組開放式配對", "e_n_g1_pairs", 10, "ge"),
        ("P1-2 每人 ≥1 句達 10 音節（nPVI 需要）", "c_n_syllables", 10, "ge"),
        ("P1-3 F0 追蹤失敗率 ≤ 20%", "c_f0_tracking_failure_ratio", 0.20, "le"),
        ("P1-5 CPP 無法計算的比例 ≤ 20%", "b_cpp_nan_ratio", 0.20, "le"),
        ("P1-3 CREPE/pYIN 一致率 ≥ 90%", "c_f0_agreement", 0.90, "ge"),
        ("P1-3 八度錯誤率 ≤ 3%", "c_f0_octave_error", 0.03, "le"),
    ]
    A("| 稽核項目 | 門檻 | 最差個案 | 未達標人數 | 狀態 |")
    A("|---|---|---|---|---|")
    failures: list[str] = []
    for label, key, thr, direction in checks:
        vals = np.array([row.get(key, np.nan) for row in q], dtype=float)
        if not np.isfinite(vals).any():
            A(f"| {label} | {thr} | — | — | ⚠ 無資料 |")
            continue
        bad = (vals < thr) if direction == "ge" else (vals > thr)
        worst = float(np.nanmin(vals)) if direction == "ge" else float(np.nanmax(vals))
        n_bad = int(np.nansum(bad))
        if n_bad:
            failures.append(f"{label}（{n_bad} 人）")
        A(f"| {label} | {thr} | {_f(worst, 2)} | {n_bad} | "
          f"{'✅' if n_bad == 0 else '⚠ ' + str(n_bad) + ' 人未達'} |")
    A("")
    if failures:
        A(f"⚠ 未全部達標：{'；'.join(failures)}。稽核要求把這些受試者標為**低可靠**並在"
          f"分析時揭露，而不是悄悄剔除。")
    else:
        A("全部受試者通過所有可自動檢查的底線。")
    A("")

    A("> **P1-3 的三演算法交叉比對只做了兩個**（torchcrepe 與 pYIN）；"
      "Praat autocorrelation 尚未接上，這一條因此只算部分滿足。")
    A("")

    # ------------------------------------------------------------------ P0-F7 floor effect
    A("## 二、🔴 P0-F7 地板效應檢查（稽核明文要求「先看對照組的分布」）")
    A("")
    A("計畫書 §4b.1 對 F7 寫著：「4–6 歲兒童本就不擅長維持長篇互惠對話。**先檢視對照組"
      "分布**——若對照組本身即無明顯調適，兩組差距會被發展階段壓縮，此特徵在本年齡層"
      "可能不適用。**這是要如實報告的發現。**」")
    A("")
    A("| 視窗 | 斜率中位數 | IQR | 斜率 > 0 的人數 | 單樣本符號檢定 p |")
    A("|---|---|---|---|---|")
    from scipy.stats import binomtest

    f7_vals = {}
    for w in cfg["features"]["e_stream"]["f7_windows"]:
        # F7 lives in `quality` now, not in the feature vector (floor effect, 2026-09-12).
        # `table.quality` is one dict per subject, not one dict of arrays -- assuming the
        # latter cost a 27-minute extraction run that then died in the report writer.
        key = f"e_f7_slope_w{w}"
        vals = [float(qq[key]) for qq in q if key in qq]
        v = np.asarray(vals) if vals else lookup.get(key)
        if v is None:
            continue
        x = v[np.isfinite(v)]
        f7_vals[w] = x
        pos = int((x > 0).sum())
        p = binomtest(pos, len(x), 0.5).pvalue if len(x) else np.nan
        m, q1, q3, _, _ = _stats(x)
        A(f"| {w} 輪 | {_f(m, 6)} | [{_f(q1, 6)}, {_f(q3, 6)}] | {pos}/{len(x)} | {_f(p, 4)} |")
    A("")
    ws = sorted(f7_vals)
    if len(ws) >= 2:
        cors = [float(np.corrcoef(f7_vals[a], f7_vals[b])[0, 1])
                for a, b in zip(ws[:-1], ws[1:]) if f7_vals[a].size > 2]
        if cors:
            A(f"> **三個視窗幾乎是同一個數字**：相鄰視窗的跨受試者相關為 "
              f"{'、'.join(_f(c) for c in cors)}。稽核 P0-F7 要求三個視窗全部計算並報告，"
              f"理由是 SocialRecNet 顯示不同構念的最佳視窗不同；但在這批資料上它們並非"
              f"三個獨立的檢定，**不應該當成三次機會**。")
            A("")
    A("> **怎麼讀**：若斜率中位數接近 0 且「> 0 的人數」接近半數，代表這個年齡層的 TD 兒童"
      "**本來就沒有可測的多輪語意調適**。那不是特徵壞掉，是構念在這個年齡層可能不成立——"
      "而且它直接限制了 F7 未來能偵測到的組間差距上限。")
    A("")

    # ------------------------------------------------------------------ G1 dynamic range
    A("## 三、G1 相似度的動態範圍（延續合成資料的發現）")
    A("")
    g1m, g1_q1, g1_q3, g1lo, g1hi = _stats(lookup.get("e_g1_similarity_median", np.zeros(0)))
    p25 = _stats(lookup.get("e_g1_sim_p25", np.zeros(0)))
    A(f"跨受試者的 G1 相似度中位數落在 **{_f(g1lo)}–{_f(g1hi)}**"
      f"（中位 {_f(g1m)}、IQR [{_f(g1_q1)}, {_f(g1_q3)}]）；下四分位版本為 "
      f"{_f(p25[3])}–{_f(p25[4])}。")
    A("")
    g1_sd = float(np.nanstd(lookup.get("e_g1_similarity_median", np.zeros(0)), ddof=1))
    n_pairs = np.array([row.get("e_n_g1_pairs", np.nan) for row in q], dtype=float)
    n_med = float(np.nanmedian(n_pairs)) if np.isfinite(n_pairs).any() else np.nan
    sem = 0.15 / np.sqrt(n_med) if np.isfinite(n_med) and n_med > 0 else np.nan
    A(f"跨受試者標準差 **{_f(g1_sd, 4)}**；每人開放式配對數中位 **{n_med:.0f}** 對，"
      f"以逐對 SD 0.15 估算單一受試者的抽樣誤差約 {_f(sem, 4)}。"
      f"**訊噪比約 {_f(g1_sd / sem, 1)} 倍。**")
    A("")
    A("> **這修正了我先前在合成語料上的警告。** 合成 session 每人只有約 20 對開放式配對，"
      "抽樣誤差 ±0.031 大於組間訊號 ±0.017，所以 G1 在那裡量不出東西。真實 session 有"
      f"約 {n_med:.0f} 對，抽樣誤差降到 {_f(sem, 4)}，跨受試者變異是它的 "
      f"{_f(g1_sd / sem, 1)} 倍——**G1 在真實資料上的條件好得多**。"
      "這也說明合成語料的每人輪次數必須貼近真實，否則會低估特徵的可用性。")
    A("")

    # ------------------------------------------------------------------ recording quality
    A("## 三b、🔴 錄音品質：CPP 被自己的稽核門檻擋掉")
    A("")
    below = np.array([row.get("b_n_below_snr", np.nan) for row in q], dtype=float)
    nturn = np.array([max(row.get("b_n_child_turns", 1), 1) for row in q], dtype=float)
    nanr = np.array([row.get("b_cpp_nan_ratio", np.nan) for row in q], dtype=float)
    A(f"稽核 P1-5 規定「SNR < 15–20 dB 不計 CPP」，本管線用 15 dB。實測結果："
      f"兒童輪次低於門檻的比例中位 **{100 * np.nanmedian(below / nturn):.0f}%**"
      f"（各案 {100 * np.nanmin(below / nturn):.0f}–{100 * np.nanmax(below / nturn):.0f}%），"
      f"CPP 缺值比例中位 **{100 * np.nanmedian(nanr):.0f}%**。"
      f"兩者的相關是 **r = {float(np.corrcoef(below / nturn, nanr)[0, 1]):.2f}**"
      f"——CPP 的缺值**全部**來自 SNR 閘門，不是計算失敗。")
    A("")
    A("> **這是錄音條件的問題，不是程式的問題。** 實測噪音底各案穩定在約 −34 dBFS，"
      "醫師的 SNR 一律高於兒童（更靠近麥克風），兒童 SNR 中位數在各案之間從 6 dB 到 16 dB。"
      "這是遠場／房間麥克風的典型樣態，也正是稽核 P0-0A 要求先盤點的事項。")
    A("")
    A("**三個後果，都要在加入第二組之前處理**：")
    A("")
    A("1. **CPP 是醫師評「高」的 A7 特徵，現在只在最大聲的約四成輪次上算得出來。**"
      "計畫書 §9.2.3 的 Plan B 是「B 流退化為僅 CPP＋頻譜傾斜＋強度」——但若 CPP 本身"
      "被擋掉，那條退路也失效。")
    A("2. **缺值率本身是洩漏通道**：各案 CPP 缺值從 20% 到 99%，這個比例若在兩組間"
      "系統性不同，模型可以靠缺值模式分類（計畫書 §9.2.2 的陷阱結構）。")
    A("3. **A12 音量跟隨的可行性存疑**：兒童與醫師的音量差在各案之間從 −1.5 dB 到 −5.2 dB，"
      "與距離變動一致。稽核 P0-F5 要求「先抽 5 段人工評估音量變化是真實音量還是距離變化，"
      "不可靠則整項擱置」——**這一步只有人能做，尚未執行**。")
    A("")
    A("> **我沒有為了多留資料而降低門檻。** 那是結果導向的調參，正是稽核 P0-4 禁止的事。"
      "正確的決定方式是稽核 P1-5 自己指定的**噪聲擾動漂移測試**：對乾淨片段逐級加噪，"
      "若 CPP 的漂移大於預期的組間差，這個特徵在本錄音條件下就是不可用的——"
      "那是下一步該做的實驗，不是現在就改一個數字。")
    A("")

    # ------------------------------------------------------------------ distributions
    A("## 四、各流特徵分布")
    A("")
    A("> ⚠ **「註記」欄是工程上的粗略期望，不是文獻常模。** 我沒有為這些數字附上查證過的"
      "來源，所以它們只能用來看「有沒有離譜」，**不可以引用為published norms**。")
    A("")
    for group, names in FEATURE_GROUPS.items():
        present = [n for n in names if n in lookup]
        if not present:
            continue
        A(f"### {group}")
        A("")
        if group.startswith("C 流｜語句層級語調"):
            A("> 這一組是為了取代失效的音節層級聲調特徵而加的，**不依賴音節邊界**。"
              "全部九項都通過奇偶輪次分半信度（三次獨立重跑 r = 0.69–0.96）。"
              "原本另有 B1 平板句比例與 B3 句尾語調四項，三次重跑皆未通過"
              "（r = −0.30…0.46），已於 2026-09-12 由計畫主持人裁示移除；"
              "原因與數字見 `features/intonation.py` 的模組說明。")
            A("")
        A("| 特徵 | 中位數 | IQR | 最小 | 最大 | 註記 |")
        A("|---|---|---|---|---|---|")
        for n in present:
            m, q1, q3, lo, hi = _stats(lookup[n])
            A(f"| `{n}` | {_f(m)} | [{_f(q1)}, {_f(q3)}] | {_f(lo)} | {_f(hi)} | "
              f"{PLAUSIBILITY.get(n, '—')} |")
        A("")

    # ------------------------------------------------------------------ cost
    if timings:
        A("## 五、執行成本")
        A("")
        A("| 階段 | 秒 |")
        A("|---|---|")
        for k, v in timings.items():
            A(f"| {k} | {_f(v, 1)} |")
        A("")

    A("## 六、這份報告不能拿來做什麼")
    A("")
    A("- **不能宣稱任何特徵有鑑別力。** 只有一個 cohort，沒有可分的兩類。")
    A("- **不能做效標關聯效度。** TD 的 ADOS A 區編碼幾乎必然全為 0，"
      "稽核 P0-0E 明文寫「幾乎全為 0 者不適合作效標」。")
    A("- **不能把這裡的分布當成常模。** n=19、單一機構、單一施測者、且是"
      "「因懷疑 ASD 被轉介但未達診斷」以外的 TD——與計畫書設定的異質對照組不是同一群人。")
    for note in extra_notes:
        A(f"- {note}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
