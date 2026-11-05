---
title: 任務：依三份研究文件建構 ASD 語音分類管線（Phase 0）＋合成資料冒煙測試

---

# 任務：依三份研究文件建構 ASD 語音分類管線（Phase 0）＋合成資料冒煙測試

## 0. 角色與任務邊界

你是本研究的**管線工程師**。`docs/` 內有三份文件：

- `華語ASD語音分類研究計畫書_v3.6.md` —— 完整技術規格（最高權威）
- `架構運作流程白話說明_v3.6.md` —— 管線流程與各步驟的理由
- `ASD特徵品質稽核清單_v3.6.md` —— 每一步的品質檢查與驗收條件

你的任務**不是執行研究**，而是：**按規格把 Phase 0 的程式管線蓋好，並在你自己生成的合成資料上完成端到端冒煙測試**。真實個案資料尚未到位，本階段也絕不會出現（見紅線 R1）。

以繁體中文溝通與撰寫文件；程式碼識別字與註解用英文。

## 1. 開工前（必做，先於任何程式碼）

1. **逐字讀完三份文件，不可跳讀。** 計畫書以 §5.0（主模型與必報基線）、§5.0.5（各流編碼器規格）、§5.3–§5.5b（B/C/E/D 流與互惠特徵）、§4b（Phase 0 七項特徵定義）、§6（驗證設計）、§9（風險與防洩漏）為核心；白話說明的步驟一～五就是管線順序；稽核清單的 P0-* 條目是驗收條件。
2. 產出 `CLAUDE.md`：(a) 管線總圖；(b) 每個模組的輸入／輸出契約；(c) 下方紅線清單**原文照抄**；(d) 名詞防混淆表（評估表編號 A1–F8 ≠ ADOS 編碼 A1–A8；C 流「角色」≠ E 流「三層」≠ 神經網路「層」）。
3. 產出 `QUESTIONS.md`：列出所有你認為模糊、矛盾、或無法實作之處。**遇到模糊處：採保守解釋、記錄在案，禁止沉默自行決定。**
4. 把第 3 節的里程碑寫進 `PLAN.md`，逐一開工；**每個里程碑驗收通過才准進下一個**。

## 2. 🔴 紅線（違反任何一條＝本次交付失敗）

- **R2 切分以受試者為單位**（subject-independent）。任何 fold 之間不得有同一受試者；**禁止把片段／幀當獨立樣本**（同一兒童的片段不獨立，這是文件的鐵律）。
- **R3 所有從資料估計的量一律折內 fit**：標準化、聲調模板、PCA、機率校準、超參數（巢套 CV）。任何 `fit` 只能見到訓練索引——這必須寫成可自動測試的約束（見 M4），不是註解。
- **R4 禁用中國來源模型與資料集**（text2vec、bge、chinese-hubert、ChildMandarin 等一律不得引入）。允許清單：torchcrepe、pYIN（librosa）、Praat/parselmouth、openSMILE（eGeMAPS v02）、MFA＋官方 Mandarin 模型、CKIP（中研院）、sentence-transformers 多語 SBERT（`paraphrase-multilingual-mpnet-base-v2`；輕量備選 `paraphrase-multilingual-MiniLM-L12-v2`）、speechbrain ECAPA（`spkrec-ecapa-voxceleb`）、XLS-R-300m（僅 D 對照臂）。要新增任何預訓練模型，先寫進 `QUESTIONS.md` 停下來問。
- **R5 模型規格照 §5.0.2，不得擅改**：
  - **M1（主模型）**＝C 流 TCN（64 通道×4 層擴張卷積，dilation 1/2/4/8，時間池化→嵌入）＋ B/E 統計池化 ＋ 三流串接 → MLP（64 隱藏單元）分類頭；損失＝class-weighted cross-entropy（**不用 Focal**）。
  - **B0**＝三流統計池化→串接（36–66 維）→ L2 邏輯迴歸／shrinkage-LDA。
  - **B1**＝每流折內 LR → 折外分數 f_s；融合層 logit = b + Σ w_s·f_s（3+1 參數）。
  - **B2**＝eGeMAPS functionals ＋ shrinkage-LDA／SVM。
  - 誠實條款的決策規則（「M1 未顯著優於 B0/B1 → 交付基線」、配對 bootstrap、同切分同種子）寫進 `configs/decision_rules.yaml`，先於任何真實訓練存檔。
- **R6 對齊（MFA）設計為可注入介面。** 合成資料由產生器直接附上音節邊界與聲調類別（ground-truth 邊界）；真實資料路徑才走 MFA adapter。**冒煙測試不得因 MFA 缺席而失敗**；MFA adapter 用獨立 integration test 驗證（無環境時標 skip）。

## 3. 里程碑

**M0｜骨架與環境**：repo 結構、venv/conda、requirements 鎖版、Makefile（`make smoke` / `make test`）、CLAUDE.md、QUESTIONS.md、PLAN.md。

**M1｜資料契約與合成資料產生器**：
- GT 逐字稿的 dataclass：語者（醫師/兒童/重疊）、起訖時間戳、文字內容；合成資料額外附：音節邊界、聲調類別、每輪強度。
- 產生器參數：受試者數、每人輪次數、隨機種子、**訊號強度 s**——s=0 時兩組（ASD/對照）分布完全相同；s>0 時按文件方向注入組間差（節奏變異↑、停頓異常↑、回應延遲↑、仿說相似度↑、答題切題↓、殘餘 F0 不預設方向）。同時生成合成 ADOS 逐項分數（A2/A3/A4/A8，0–3，與注入訊號相關）供效標分析用。
- 合成音訊：對每個兒童語段生成可供 F0／強度分析的訊號（諧波＋雜訊即可，F0 落在 100–600 Hz 兒童帶，不需要像人聲）。

**M2｜三流特徵抽取**（`features/b_voice.py`、`c_prosody.py`、`e_text.py`）：
- B：openSMILE eGeMAPS LLD（jitter/shimmer/HNR/CPP/spectral tilt/強度）＋ECAPA 語者內嵌入變異（卡通腔代理）。
- C：F0（torchcrepe，搜尋範圍 **100–600 Hz**，防八度錯誤）＋注入的音節邊界 → nPVI、%V/ΔC、音節時長分布、停頓分布、語速（含/不含停頓）、語句時長、聲調範疇正確率、聲調一致性、變調/輕聲實現、聲調剝離後殘餘 F0。
- E：CKIP 斷詞＋SBERT → MLU、刻板語句重複、即時仿說相似度（兒童句 vs 醫師前一句）、G1 答題切題、F7 多輪語意調適（3/5/7 輪視窗趨勢）、F8 雙向不對稱、回應延遲、音量跟隨（兒童輪強度對前一醫師輪強度的迴歸斜率）。
- **每個特徵配一個手算小例的單元測試**（3–5 輪的迷你逐字稿，手工算出期望值）。

**M3｜池化與組裝**：B/E functionals（均值/標準差/分位數等）；C 逐音節序列張量化（padding＋mask）；產出受試者級資料表。

**M4｜巢套 CV 框架＋🔴 防洩漏測試套件**（`tests/test_leakage.py`）：
- assert 折間受試者零重疊；
- 用 wrapper 記錄每個 transformer 的 fit 索引集合，assert ⊆ 訓練索引；
- assert 特徵表不含任何標籤衍生欄位；
- assert 每位受試者恰為一個樣本（無片段展開）。

**M5｜基線 B0/B1/B2**＋單流戰力表（B1 折外分數）。

**M6｜M1（PyTorch）**：TCN＋MLP 端到端、折內驗證 early stopping、AdamW＋weight decay＋dropout、多種子、與 B0/B1 同切分的配對 bootstrap 檢定。

**M7｜分析與報告**：效標相關（特徵 ↔ 合成 ADOS 分數，Spearman）、施測者探測（僅醫師話語的探測分類器＋置換檢定 ≥5,000 次）、機率校準、跨種子 mean±SD、自動產出 `report.md`。

**M8（選配，可延後）｜D 對照臂**：XLS-R-300m 凍結 → 預選單層 → mean+std → 折內 PCA-20 → LR。下載量大，冒煙階段可只留介面＋skip 標記。

## 4. 冒煙測試驗收（全部通過才算完成）

- **S1** `make smoke`：合成 12 位受試者（6/6），端到端 M2→M7，3-fold × 2 種子，產出 `report.md`，全程無未捕捉例外。
- **S2 訊號 s=0 時：所有模型折外 AUC 落在 [0.3, 0.7]。** 任何模型系統性 >0.7 ＝ 洩漏警報——停下、修復、重跑，並把根因寫進 DEVIATIONS.md。
- **S3 訊號調大時：折外 AUC 隨 s 單調上升，s 足夠大時 >0.85**（證明管線真的能把注入的訊號學出來）。
- **S4** `pytest` 全綠（防洩漏套件＋全部特徵手算例）。
- **S5** 對照稽核清單，把可自動化的 P0 條目逐條標記（通過／不適用／需真實資料），輸出 `checklist_report.md`。

## 5. 工作紀律

- 每個里程碑一個 git commit；驗收未過不得進下一個。
- 與文件規格的任何偏離，寫入 `DEVIATIONS.md`（含理由與影響範圍）。
- 完工報告需包含：已建成並通過測試的部分、被 stub 的部分、等真實資料才能做的部分（MFA 實跑、真 ADOS 分數、施測者探測的實質解讀）、以及 `QUESTIONS.md` 全文。
