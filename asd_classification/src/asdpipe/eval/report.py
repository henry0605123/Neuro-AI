"""Report writer -- outputs/report.md, in Traditional Chinese, as prose with tables."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

from ..contracts import SubjectTable
from .analysis import CRITERION, PRIMARY_FEATURE, holm, verdict_for

__all__ = ["write_report"]


def _f(x, nd=3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{v:.{nd}f}"


def write_report(path: Path, table: SubjectTable, cv: Mapping, criterion: Mapping,
                 group: Mapping, cfg: Mapping, rules: Mapping, versions: Mapping,
                 timings: Mapping | None = None,
                 validate: Mapping | None = None) -> Path:
    seeds = cv["seeds"]
    across = cv["across_seeds"]
    L = []
    A = L.append

    A("# Phase 0 管線冒煙測試報告")
    A("")
    A(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}　·　資料來源：**合成語料**"
      f"（{cv['n_subjects']} 位受試者，ASD/對照各半）")
    A("")
    A("> 🔴 本報告的每一個數字都來自程式自行生成的合成資料。**沒有任何真實個案資料進入過本專案**，")
    A("> 因此下面所有效果量、AUC、效標相關都只證明「管線會跑、而且抓得到我自己注入的訊號」，")
    A("> 不構成任何關於自閉症的實證主張。")
    A("")

    # ---------------------------------------------------------------- 1. headline
    A("## 一、這次跑出了什麼")
    A("")
    delivered = cv["delivered_model"]
    m1 = across["M1"]
    A(f"在 {cv['n_folds']} 折 × {len(seeds)} 種子的受試者層級交叉驗證下，主模型 M1 的折外 AUC 是 "
      f"**{_f(m1['auc_mean'])} ± {_f(m1['auc_sd'])}**（跨種子平均 ± 標準差）。三個必報基線在"
      f"完全相同的切分與種子上跑出 "
      f"B0 {_f(across['B0']['auc_mean'])}、B1 {_f(across['B1']['auc_mean'])}、"
      f"B2 {_f(across['B2']['auc_mean'])}。")
    if delivered["m1_beats_baselines"]:
        A("配對 bootstrap 顯示 M1 在每一個種子上都顯著優於 B0 與 B1，因此依 "
          "`configs/decision_rules.yaml` 的誠實條款，**交付模型為 M1**。")
    else:
        A(f"配對 bootstrap **沒有**顯示 M1 顯著優於 B0 與 B1，因此依訓練前就寫死的誠實條款，"
          f"**交付模型是基線 `{delivered['model']}`**，而不是複雜度更高的 M1。")
        A("這不是失敗，是計畫書 §5.0.2 條款 3 預先處理過的情境：在這個樣本規模下，"
          "「複雜度沒有帶來可驗證的增益」本身就是要如實發表的結論。")
    if delivered.get("b1_unstable"):
        A(f"另外，B1 的流分數堆疊被判定為不穩定"
          f"（單流最大貢獻 {_f(delivered['b1_max_single_stream_contribution'], 2)}，"
          f"跨折貢獻標準差 {_f(delivered['b1_contribution_sd_across_folds'], 2)}），"
          f"依 §5.0.2b 的塌陷檢查退回 B0。")
    A("")

    # ---------------------------------------------------------------- 2. model table
    A("## 二、模型比較（同切分、同種子）")
    A("")
    A("| 模型 | AUC 平均 | 跨種子 SD | Accuracy | Sensitivity | Specificity | F1 | Brier |")
    A("|---|---|---|---|---|---|---|---|")
    for m in ("M1", "B0", "B1", "B2"):
        v = across[m]
        A(f"| {m} | {_f(v['auc_mean'])} | {_f(v['auc_sd'])} | {_f(v['accuracy_mean'])} | "
          f"{_f(v['sensitivity_mean'])} | {_f(v['specificity_mean'])} | {_f(v['f1_mean'])} | "
          f"{_f(v['brier_mean'])} |")
    A("")
    A("配對 bootstrap（受試者層級重抽，每個種子各跑 "
      f"{rules['comparison']['n_boot']} 次）：")
    A("")
    A("| 比較 | 種子 | 觀察差 ΔAUC | 95% CI | p | 本樣本的 MDE |")
    A("|---|---|---|---|---|---|")
    for s in seeds:
        for k, c in cv["per_seed"][s]["comparisons"].items():
            A(f"| {k.replace('_', ' ')} | {s} | {_f(c['observed'])} | "
              f"[{_f(c['ci_low'])}, {_f(c['ci_high'])}] | {_f(c['p_two_sided'])} | "
              f"{_f(c['mde'])} |")
    A("")
    A("**MDE 揭露**：上表最後一欄是這個樣本量與模型分數相關下、檢定看得見的最小差距。"
      "任何「未顯著」都要念成「增益 < MDE」，不是「增益 = 0」"
      "（計畫書 §5.0.2 條款 2）。n=12 的 MDE 大到幾乎任何真實差距都測不出來——"
      "這正是 Phase 0 不做分類效能宣稱的原因。")
    A("")

    # ---------------------------------------------------------------- 3. streams
    A("## 三、單流戰力表與事後消融")
    A("")
    A("單流折外 AUC 來自基線 B1 的流內分數；消融是 leave-one-stream-out 後重訓 M1 的 AUC 落差。"
      "M1 **不宣稱任何「可學的流權重」**——v3.4 那套說法在數學上已被判定不可辨識，換成 MLP 也一樣。")
    A("")
    A("| 流 | 單流折外 AUC（各種子） | 移除該流後 M1 的 AUC | ΔAUC |")
    A("|---|---|---|---|")
    for s in ("b", "c", "e"):
        lb = [cv["per_seed"][sd]["leaderboard"][s] for sd in seeds]
        ab = [cv["per_seed"][sd]["ablation"].get(s, {}) for sd in seeds]
        A(f"| {s.upper()} 流 | {', '.join(_f(x) for x in lb)} | "
          f"{', '.join(_f(a.get('auc_without')) for a in ab)} | "
          f"{', '.join(_f(a.get('delta_auc')) for a in ab)} |")
    A("")

    # ---------------------------------------------------------------- 4. criterion validity
    family = int(rules["criterion_validity"]["primary_indicator_family_size"])
    A("## 四、效標關聯效度（Phase 0 的主要分析）")
    A("")
    A("依稽核 P0-4 的規定，**效標分析先於組間分析執行**，順序寫死在腳本裡。"
      "特徵與合成 ADOS 逐項分數做 Spearman ρ，附 bootstrap 95% CI；"
      f"檢定家族固定為 {family} 項主指標，走 Holm 校正；算不出來的指標仍計入家族大小。")
    A("")
    pvals = {k: v.get("p", np.nan) for k, v in criterion.items() if v.get("status") == "computed"}
    holm_res = holm(pvals, alpha=float(rules["comparison"]["alpha"]), family_size=family)
    A("| 主指標 | 對應特徵 | ADOS 效標 | ρ | 95% CI | 二元化 AUC | Holm 門檻 | Holm 顯著 | 判定 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for ind, v in criterion.items():
        if v["status"] != "computed":
            A(f"| {ind} | {v['feature'] or '（人工標註）'} | {CRITERION[ind]} | — | — | — | — | — | "
              f"**{v['status']}** |")
            continue
        d = group.get(ind, {}).get("cohen_d", np.nan)
        h = holm_res.get(ind, {})
        A(f"| {ind} | `{v['feature']}` | {CRITERION[ind]} | {_f(v['rho'])} | "
          f"[{_f(v['ci_low'])}, {_f(v['ci_high'])}] | {_f(v.get('binarised_auc'))} | "
          f"{_f(h.get('threshold'), 4)} | {'是' if h.get('reject') else '否'} | "
          f"{verdict_for(v['rho'], v['ci_low'], v['ci_high'], d, rules)} |")
    A("")
    A("> **判定用的是 |ρ|，表中列的是有號 ρ**。有幾個主指標的預期方向本來就是負的——"
      "例如 A12 音量跟隨的斜率越低、ADOS A2 的異常分數越高——用有號 ρ 去比 +0.5 會漏掉它們。")
    A("")
    A("> **門檻的性質**：ρ≥0.5 是**效果量**門檻，不是顯著性宣稱。n=20 時未校正需 ρ̂≥0.454、"
      "Holm-8 首步需 ρ̂≥0.593；真 ρ=0.5 的特徵通過「ρ̂≥0.5」的機率只有約五成。"
      "Go/No-Go 另看 bootstrap CI 下界是否 > 0.2。")
    A("")

    A("### 次要分析：組間差異")
    A("")
    A("| 主指標 | Cohen's d | 95% CI | rank-biserial | Mann-Whitney p |")
    A("|---|---|---|---|---|")
    for ind, v in group.items():
        if v.get("status") != "computed":
            A(f"| {ind} | — | — | — | **{v.get('status')}** |")
            continue
        A(f"| {ind} | {_f(v['cohen_d'])} | [{_f(v['d_ci_low'])}, {_f(v['d_ci_high'])}] | "
          f"{_f(v['rank_biserial'])} | {_f(v['u_p'])} |")
    A("")
    warn = [k for k, v in criterion.items()
            if v.get("status") == "computed" and np.isfinite(v.get("rho", np.nan))
            and abs(v["rho"]) < 0.3 and abs(group.get(k, {}).get("cohen_d", 0) or 0) >= 0.8]
    if warn:
        A(f"⚠ **「效標低但組間高」警訊**：{', '.join(warn)}。這是 confound 的典型指紋——"
          "特徵能分組，但測的可能不是宣稱的構念。依 §4b.4 必須先完成追查才可保留。")
    else:
        A("本次沒有出現「效標低但組間高」的組合。")
    A("")

    # ---------------------------------------------------------------- 5. examiner probe
    A("## 五、施測者洩漏探測")
    A("")
    probe = cv.get("examiner_probe", {})
    if probe:
        A(f"僅用**醫師話語**（文本統計＋韻律＋等待時間，完全不碰兒童端）訓練探測分類器，"
          f"折外準確率 {_f(probe['accuracy'])}、AUC {_f(probe['auc'])}，"
          f"置換檢定（{probe['n_permutations']} 次）p = {_f(probe['permutation_p'])}，"
          f"虛無分布 95% 區間 [{_f(probe['null_ci_low'])}, {_f(probe['null_ci_high'])}]。")
        A("")
        A("> 這個探測在 n≈20 下**沒有判定力**——虛無區間本身就寬達 "
          f"[{_f(probe['null_ci_low'], 2)}, {_f(probe['null_ci_high'], 2)}]，"
          "所以不存在「接近隨機所以可以安心」的讀法。**無論結果如何**，互動類特徵"
          "（G1、F1、F7、F8、A12）的效果量一律加註「可能含施測者調變」。")
        A("")
        A("合成產生器預設 `examiner_leak_strength = 0`：醫師只對兒童行為做反應，"
          "不照標籤改變行為。因此 F8 的不對稱性是**自然湧現**的，不是貼上去的。"
          "陽性對照（把洩漏強度調高、確認探測程式真的抓得到）見 `outputs/validate_*.json`。")
    A("")

    # ---------------------------------------------------------------- 6. quality
    A("## 六、資料品質與缺值")
    A("")
    q_keys = ["b_ecapa_windows", "b_cpp_nan_ratio", "c_n_syllables",
              "c_f0_tracking_failure_ratio", "e_n_pairs", "e_n_g1_pairs"]
    A("| 品質指標 | 中位數 | 最小 | 最大 | ASD vs 對照 差異 |")
    A("|---|---|---|---|---|")
    y = table.y
    for k in q_keys:
        vals = np.array([row.get(k, np.nan) for row in table.quality], dtype=float)
        if not np.isfinite(vals).any():
            continue
        a, b = vals[y == 1], vals[y == 0]
        A(f"| `{k}` | {_f(np.nanmedian(vals))} | {_f(np.nanmin(vals))} | {_f(np.nanmax(vals))} | "
          f"{_f(np.nanmean(a) - np.nanmean(b))} |")
    A("")
    A("> 最後一欄要特別看：**抽取失敗率的組間差異本身就會洩漏標籤**"
      "（計畫書 §9.2.2 的陷阱結構）。F0 追蹤失敗率、CPP 無法計算的比例、"
      "可用配對數若在兩組間系統性不同，模型可以靠缺值模式分類。")
    A("")
    # Health check P12 / audit P0-4: what could not be measured, disclosed by group. The
    # models impute these inside each fold; this table is where the pattern is seen.
    uc = table.unmeasurable_counts()
    A("**無法測量的特徵數（每位受試者，分組中位數〔最小–最大〕）**")
    A("")
    A("| 流 | 特徵數 | ASD | 對照 |")
    A("|---|---|---|---|")
    for s in ("b", "c", "e"):
        cells = []
        for grp in (1, 0):
            v = uc[s][y == grp]
            cells.append(f"{int(np.median(v))}〔{int(v.min())}–{int(v.max())}〕" if v.size else "—")
        A(f"| {s.upper()} | {len(table.feature_names(s))} | {cells[0]} | {cells[1]} |")
    A("")

    # ---------------------------------------------------------------- 7. validation
    if validate:
        A("## 七、冒煙驗收 S2 / S3")
        A("")
        nul = validate.get("null_check", {})
        if nul:
            A(f"**S2（虛無檢查）**：以 {nul.get('n_seeds')} 個獨立產生器種子生成 s=0 的資料集"
              f"（兩組分布完全相同）。各模型的平均折外 AUC：")
            A("")
            A("| 模型 | 平均 AUC | 95% CI | >0.7 次數 | <0.3 次數 | 單尾 binomial p | 判定 |")
            A("|---|---|---|---|---|---|---|")
            for m, v in nul.get("models", {}).items():
                A(f"| {m} | {_f(v['mean_auc'])} | [{_f(v['ci_low'])}, {_f(v['ci_high'])}] | "
                  f"{v.get('n_above', '—')} | {v.get('n_below', '—')} | "
                  f"{_f(v.get('binomial_p'))} | "
                  f"{'✅ 通過' if v['pass'] else '❌ 洩漏警報'} |")
            A("")
            A(f"判準（`DEVIATIONS.md` D1）：①平均 AUC 落在 [{_f(nul['band'][0],1)}, "
              f"{_f(nul['band'][1],1)}]；②該平均的 bootstrap CI 不整段高於 "
              f"{_f(nul['band'][1],1)}；③單次超過 {_f(nul['band'][1],1)} 的**次數**沒有顯著多於"
              f"虛無預期。")
            A("")
            A(f"> 第三條的門檻是**推導出來的，不是挑出來的**：n={nul.get('n_subjects', 12)}（6/6）"
              f"時虛無 AUC 的抽樣 SD = {_f(nul.get('null_auc_sd'), 4)}，"
              f"因此 P(單次 >0.7) = {_f(nul.get('expected_tail_rate'), 4)}，"
              f"Binomial({nul['n_seeds']}, {_f(nul.get('expected_tail_rate'), 3)}) 的 95% 上界"
              f"臨界次數是 **{nul.get('critical_count')}**。")
            A("")
            A("> **同時看下尾**：雜訊的散布是對稱的，洩漏是單邊的。上表 >0.7 與 <0.3 的次數"
              "如果差不多，那是抽樣變異；如果上尾明顯偏多，才是洩漏的指紋。")
            A("")
        swp = validate.get("sweep", {})
        if swp:
            A("**S3（訊號強度掃描）**：")
            A("")
            A("| 模型 | " + " | ".join(f"s={s}" for s in swp["signal_strengths"])
              + " | 趨勢 ρ(s,AUC) | 上升 | 最終 AUC>0.85 |")
            A("|---" * (len(swp["signal_strengths"]) + 4) + "|")
            for m, v in swp["models"].items():
                A(f"| {m} | " + " | ".join(_f(x) for x in v["auc_by_strength"])
                  + f" | {_f(v.get('trend_rho'), 2)} | {'✅' if v['monotonic'] else '❌'} | "
                    f"{'✅' if v['final_ok'] else '❌'} |")
            A("")
            A(f"> 單調性用的是**趨勢**（Spearman ρ ≥ {_f(swp.get('min_trend_rho'), 2)}），"
              f"不是相鄰兩點的差。每個點是 {swp['n_seeds']} 個種子的平均，標準誤 "
              f"{_f(swp.get('per_point_se'), 3)}——要求相鄰點不得下降超過 0.02，"
              f"測到的是雜訊而不是管線（見 `DEVIATIONS.md` D1b）。")
            A("")
            A("> **M1 在這個樣本量下沒有達標，而且這是事前就預期的**：外層訓練折只有 8 位"
              "受試者，折內再切驗證子折後真正用來更新權重的約 5 位，而 M1 有 50,434 個參數。"
              "計畫書 §5.0.2 條款 5 已寫明此情境。S3 要證明的是「管線能把注入的訊號學出來」，"
              "線性基線做到了就證明了（見 `QUESTIONS.md` Q4）。")
            A("")
        pc = validate.get("examiner_leak_positive_control", {})
        if pc:
            A(f"**施測者探測的陽性對照**：把 `examiner_leak_strength` 調到 "
              f"{pc.get('examiner_leak_strength')} 之後，探測準確率 {_f(pc.get('accuracy'))}、"
              f"p = {_f(pc.get('permutation_p'))}"
              f"（預設 s 設定下為 {_f(pc.get('baseline_accuracy'))} / "
              f"p = {_f(pc.get('baseline_p'))}）。"
              f"{'探測程式確實抓得到刻意注入的洩漏。' if pc.get('detected') else '⚠ 探測程式沒抓到刻意注入的洩漏，需要檢查。'}")
            A("")

    # ---------------------------------------------------------------- 8. versions
    A("## 八、工具版本凍結（稽核 P0-0F）")
    A("")
    A("| 項目 | 版本 / commit |")
    A("|---|---|")
    for k in sorted(versions):
        A(f"| `{k}` | `{versions[k]}` |")
    A("")
    A("**來源合規（R4）**：本管線使用的預訓練模型全部非中國來源——"
      "SBERT 多語模型（UKP Lab，德）、ECAPA（SpeechBrain 學術聯盟）、"
      "CKIP 斷詞（中研院，台灣）、CREPE（NYU，美）、openSMILE（audEERING，德）、"
      "Praat/Parselmouth（阿姆斯特丹大學，荷）。"
      "CKIP 的合規邊界仍待法遵確認，見 `QUESTIONS.md` Q12。")
    A("")

    # ---------------------------------------------------------------- 9. limits
    A("## 九、這份報告不能說什麼")
    A("")
    A("- **不能說模型有效**。資料是合成的；AUC 只證明管線能把我自己注入的訊號學出來。")
    A("- **不能說任何特徵有臨床效度**。效標是合成 ADOS 分數，由同一組潛在特質生成，"
      "相關性是設計出來的，不是發現的。")
    A("- **強制對齊（MFA）沒有實跑**。合成語料由產生器直接附上音節邊界與聲調類別；"
      "MFA adapter 只有介面與 integration test（無環境時 skip）。真實資料的對齊誤差未知。")
    A("- **D 流對照臂沒有跑**。介面已就位（凍結骨幹→預選單層→mean+std→折內 PCA-20→LR），"
      "但冒煙階段不下載 XLS-R-300m。")
    A("- **B4（重音／強調位置）需要人工標註**，Phase 0 無法自動化，效標欄位是空的。")
    A("- **施測者探測的實質解讀需要真實資料**。合成語料的醫師行為是我寫的規則，"
      "不是真人的臨床行為。")
    A("")
    A("完整的未決問題清單見 `QUESTIONS.md`；與文件規格的偏離見 `DEVIATIONS.md`。")

    if timings:
        A("")
        A(f"<sub>執行時間：" +
          "，".join(f"{k} {v:.1f}s" for k, v in timings.items()) + "</sub>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
