# CLAUDE.md — ASD 語音分類管線（Phase 0 骨架 ＋ 合成資料冒煙測試）

> 本檔為本 repo 的**工程權威**。規格權威依序為：`docs/ASD分類研究計畫書.md`（最高）＞
> `docs/架構運作流程說明.md`（流程順序）＞ `docs/特徵品質稽核清單.md`（驗收條件）。
> 三者與本檔衝突時以計畫書為準，並把差異記入 `DEVIATIONS.md`。
>
> **本階段目標不是做研究**，是把 Phase 0 的程式管線蓋好。管線先在**自己生成的合成資料**上
> 完成端到端冒煙測試（S1–S5 全過）；2026-08-24 起，計畫主持人裁示接入真實 ADOS 錄音，
> 見下方 (a2)。真實資料**只讀不進 repo**，且目前手上的 cohort 全為 TD、沒有對照組，
> 因此**分類與組間分析一律不可做**（`QUESTIONS.md` Q1）。

---

## (a) 管線總圖

```
┌─ M1 合成資料產生器 (src/asdpipe/synth) ──────────────────────────────────────┐
│  參數：n_subjects、turns_per_subject、seed、訊號強度 s                        │
│  產出：SyntheticCorpus                                                       │
│    ├─ Session（每位受試者 1 場）                                             │
│    │    └─ Turn[]：speaker(EXAMINER/CHILD/OVERLAP)、t_start、t_end、text     │
│    │         ＋ 合成專屬：SyllableSpan[]（音節邊界＋聲調類別）、intensity_db  │
│    ├─ audio（16 kHz，諧波＋雜訊，F0 落在 100–600 Hz 兒童帶）                  │
│    ├─ label（ASD / CONTROL）＋ latent traits（注入的真值）                    │
│    └─ 合成 ADOS 逐項分數 A2/A3/A4/A8（0–3，與注入訊號相關）                   │
└──────────────────────────────────────────────────────────────────────────────┘
        │ 音訊                    │ 音訊＋邊界                 │ 逐字稿
        ▼                         ▼                            ▼
┌─ B 流 b_voice.py ──┐  ┌─ C 流 c_prosody.py ──┐  ┌─ E 流 e_text.py ──────────┐
│ eGeMAPS LLD        │  │ F0 (torchcrepe,      │  │ CKIP 斷詞 ＋ 多語 SBERT   │
│  jitter/shimmer/   │  │   100–600 Hz)        │  │ MLU、刻板語句 F3、        │
│  HNR/CPP/tilt/強度 │  │ ＋ 音節邊界(Aligner) │  │ 即時仿說 F1、G1 切題、    │
│ ＋ ECAPA 語者內    │  │ → nPVI、%V/ΔC、音節  │  │ F7 多輪調適(3/5/7)、      │
│   嵌入變異 (B7)    │  │   時長、停頓、語速、 │  │ F8 雙向不對稱、回應延遲、 │
│ ＋ 音量跟隨 (A12)  │  │   語句時長、聲調範疇 │  │ 音量跟隨(強度迴歸斜率)    │
│                    │  │   正確率/一致性/變調 │  │                           │
│                    │  │   /輕聲、殘餘 F0     │  │                           │
└────────────────────┘  └──────────────────────┘  └───────────────────────────┘
        │                         │                            │
        ▼  M3 assemble.py         ▼                            ▼
   B functionals            C 逐音節序列張量              E functionals
   (mean/std/分位數)        (padding ＋ mask)             (mean/std/分位數/趨勢)
        │                         │                            │
        └──────────────┬──────────┴────────────────────────────┘
                       ▼
        受試者級資料表 SubjectTable（一位受試者 ＝ 恰好一列）
                       │
        ┌──────────────┴───────────────────────────────────────┐
        ▼  M4 巢套 CV (cv/splits.py ＋ cv/guards.py)            │
   受試者層級分層 k-fold（外層）＋ 折內驗證子折（內層）        │
   所有 fit 只能見訓練索引 ← FoldScope 逐次記錄並 assert       │
        ▼                                                      ▼
┌─ M5 基線 (models/baselines.py) ─┐   ┌─ M6 主模型 M1 (models/m1.py) ────────┐
│ B0 三流池化→串接(36–66維)       │   │ C 流 TCN(64ch×4層,dilation 1/2/4/8) │
│    → L2 LR / shrinkage-LDA      │   │  → 時間池化 → 嵌入                   │
│ B1 每流折內 LR → 折外分數 f_s   │   │ ＋ B/E 統計池化                      │
│    融合 logit=b+Σ w_s·f_s       │   │ → 三流串接 → MLP(64) 分類頭          │
│ B2 eGeMAPS functionals          │   │ 損失＝class-weighted CE（不用 Focal）│
│    → shrinkage-LDA / SVM        │   │ AdamW＋weight decay＋dropout         │
│ ＋ 單流戰力表（B1 折外分數）    │   │ 折內驗證 early stopping、多種子      │
└─────────────────────────────────┘   └──────────────────────────────────────┘
        └──────────────┬───────────────────────────────────────┘
                       ▼  M7 分析與報告 (eval/)
   效標相關（特徵↔合成 ADOS，Spearman）｜施測者探測（僅醫師話語＋置換檢定 ≥5,000）
   機率校準（折內 Platt/isotonic ＋ Brier）｜配對 bootstrap（同切分同種子）
   跨種子 mean±SD ｜ report.md ｜ checklist_report.md
                       ▼
   〔M8 選配〕D 對照臂 d_ssl.py：XLS-R-300m 凍結 → 預選單層 → mean+std
              → 折內 PCA-20 → LR（冒煙階段只留介面＋skip 標記）
```

---

## (a2) 真實語料的進入點（2026-08-24 新增）

```
D:/自閉症/data/Fine_tune/<case>/  ──┐  只讀，原始檔案不進 repo
   ├─ <case>.mp3                    │
   └─ <case>_edited.json            │
                                    ▼
        realdata.py::load_transcript ── 凍結的清理規則（稽核 P0-1／P0-2）
          ├─ 語者角色正規化：評估者/測試者/SPEAKER_00 → EXAMINER
          │                  兒童/小孩/SPEAKER_01     → CHILD
          │                  其餘（個管師/家長/廣播/Unknown）→ 第三方，丟棄
          ├─ 段落依 start 排序（原始檔案有順序倒退）
          ├─ 重疊語段：drop_both（容忍 20 ms）——重疊區有兩個人的聲音
          ├─ 空白／零長度段落丟棄
          └─ 問句／是非題判定：問號 ∪ 嗎/吧句尾 ∪ 疑問詞
                                    ▼
        realdata.py::load_subject ── Session / Turn（與合成路徑同一契約）
          └─ MP3 → 16 kHz 單聲道，快取於 git-ignore 的 cache/audio/
                                    ▼
        eval/intake.py ── outputs/intake_report.md（P0-0A／P0-1／P0-2 稽核數字）
```

🔴 **兩個必須記住的限制**：
1. 目前手上的 19 案**全部是同一 cohort（TD）**，沒有對照組——
   分類、組間比較、效標關聯效度**一律不可做**（見 `QUESTIONS.md` Q1）。
2. 真實逐字稿的時間戳是**語句級**，所以 C 流的節奏與聲調特徵**全部依賴 MFA**；
   語速、語句間停頓、B 流與 E 流不需要對齊（`QUESTIONS.md` Q14）。

## (b) 模組輸入／輸出契約

所有 dataclass 定義於 `src/asdpipe/contracts.py`；序列化走 `to_dict()/from_dict()`。

| 模組 | 輸入 | 輸出 | 備註 |
|---|---|---|---|
| `synth/generator.py::generate_corpus` | `SynthConfig(n_subjects, turns_per_subject, seed, signal_strength, asd_ratio, sr)` | `SyntheticCorpus`（`subjects: list[SubjectRecord]`） | `signal_strength=0` 時兩組分布完全相同（同一組參數抽樣） |
| `synth/audio.py::render_session` | `list[Turn]`、`VoiceProfile`、`sr` | `np.ndarray[float32]`（單聲道 16 kHz） | 諧波＋雜訊；F0 由音節聲調曲線決定，恆落在 100–600 Hz |
| `align/base.py::Aligner` | `audio`、`list[Turn]` | `dict[turn_id, list[SyllableSpan]]` | 抽象介面。合成走 `GroundTruthAligner`（直接回傳產生器附的邊界）；真實走 `MFAAligner` |
| `features/b_voice.py::extract_b` | `audio`、`list[Turn]`、`Backends` | `BStreamFeatures`（LLD 序列 ＋ ECAPA 視窗嵌入 ＋ 逐輪強度） | eGeMAPS v02 LLD；ECAPA 固定 3 s 視窗 |
| `features/c_prosody.py::extract_c` | `audio`、`list[Turn]`、`syllables`、`Backends` | `CStreamFeatures`（逐音節序列 ＋ 逐輪韻律量） | F0 搜尋範圍**寫死 100–600 Hz** |
| `features/e_text.py::extract_e` | `list[Turn]`、`Backends` | `EStreamFeatures`（逐輪相似度序列 ＋ 詞彙統計 ＋ 輪次時序量） | CKIP 斷詞、SBERT 句嵌入 |
| `assemble.py::pool_subject` | 三流 `*StreamFeatures` | `SubjectRow`（B/E functionals dict、C 序列 `np.ndarray[T, D]` ＋ mask） | 池化統計量集合**預先寫死**於 `configs/features.yaml` |
| `assemble.py::build_table` | `list[SubjectRow]` | `SubjectTable`（`X_b`, `X_e`, `X_c`(padded), `mask`, `y`, `subject_ids`, `ados`, `meta`） | **每位受試者恰為一列**；不得片段展開 |
| `cv/splits.py::subject_folds` | `SubjectTable`、`n_folds`、`seed`、分層鍵 | `list[Fold(train_idx, test_idx)]` | 受試者為單位；折間零重疊 |
| `cv/guards.py::FoldScope` | `Fold`、全量 X/y | 受控的 `fit`／`transform` 呼叫 | 每次 fit 記錄索引集合，assert ⊆ train_idx |
| `models/baselines.py::fit_b0/b1/b2` | `FoldScope` | 折外分數 `np.ndarray[n_test]` | B1 另回傳單流折外分數 |
| `models/m1.py::fit_m1` | `FoldScope`、`M1Config` | 折外機率 ＋ 訓練紀錄 | 折內再切驗證子折做 early stopping |
| `eval/metrics.py::paired_bootstrap` | 兩組折外分數、`y`、`n_boot` | 觀察差、CI、p | 同切分同種子 |
| `eval/report.py::write_report` | 全部結果 | `outputs/report.md` | — |
| `eval/checklist.py::write_checklist` | 全部結果 | `outputs/checklist_report.md` | P0-* 逐條標記 通過／不適用／需真實資料 |
| `realdata.py::load_transcript` | 匯出的逐字稿 JSON、`cfg` | `list[Turn]` ＋ `IntakeReport` | 清理規則凍結於 `configs/pipeline.yaml: realdata`；受試者 id 取**資料夾名**（metadata.case_name 已知會漂移） |
| `realdata.py::load_cohort` | 語料根目錄、`cfg`、`labels` | `Corpus` ＋ `list[IntakeReport]` | 沒給 `labels` 時標籤是 placeholder 並標記 `label_is_placeholder`，避免單一 cohort 被當成兩組 |
| `eval/intake.py::write_intake_report` | `list[IntakeReport]` | `outputs/intake_report.md` | P0-0A／P0-1／P0-2 稽核數字 |

---

## (c) 🔴 紅線（原文照抄；違反任何一條＝本次交付失敗）

> 以下五條逐字抄自 `執行prompt.md` 第 2 節。**（註：該節從 R2 起編號，文件 §0 提及的 R1 未列出——已記入 `QUESTIONS.md` Q1。）**

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

---

## (d) 名詞防混淆表

| 名詞 | 這裡指什麼 | **不要**跟什麼搞混 |
|---|---|---|
| **評估表 A1–F8、G1、B4、B7、A12、C1–C9、D1–D5** | 合作醫師的 **49 項臨床特徵評估表**編號（計畫書附錄 E）。例：`B7`＝卡通腔／語者內嵌入變異、`A12`＝音量跟隨、`B4`＝重音位置、`C6`＝節奏不規律、`F3`＝刻板語句、`F1`＝即時回聲、`G1`＝答話文不對題 | ❌ **ADOS 編碼 A1–A8**。兩套代號完全不同 |
| **ADOS 編碼 A2/A3/A4/A8** | ADOS-2 Module 3 **A 區逐項臨床評分（0–3）**，Phase 0 的**效標**。A2＝語音異常（語調/音量/節奏/語速）、A3＝即時回聲語言、A4＝刻板/特異用語、A8＝對話互惠 | ❌ 評估表的 A1–A12。例：評估表 `A12` 是音量跟隨**特徵**，ADOS `A2` 是**效標分數** |
| **P0-F1 … P0-F8** | 稽核清單裡「Phase 0 第 N 項特徵」的**關卡編號**。⚠ 對應關係是錯開的：`P0-F1`＝評估表 G1、`P0-F2`＝評估表 F1、`P0-F3`＝評估表 F3、`P0-F4`＝評估表 B7、`P0-F5`＝評估表 A12、`P0-F6`＝評估表 B4、`P0-F7`＝多輪語意調適、`P0-F8`＝雙向不對稱 | ❌ 直接把 `P0-F1` 讀成評估表 `F1`（那其實是 `P0-F2`） |
| **C 流的「角色」** | 核心／升級中／輔助——C 流特徵的**證據強度與優先序**，三者**並行計算**，不是先後步驟。計畫書 §5.4 稱「三組」，內容相同 | ❌ E 流的「三層」；❌ 神經網路的「層」 |
| **E 流的「三層」** | Phase 1 依 **ASR 錯誤率**決定「哪些文本特徵可以啟用」的**門檻**：第一層任何 WER 可用、第二層 WER<35%、第三層 WER<15%。Phase 0 用人工轉錄，**全部啟用** | ❌ C 流的「角色／三組」；❌ 神經網路的「層」 |
| **神經網路的「層」** | TCN 的 4 層擴張卷積、MLP 的隱藏層 | ❌ 上述兩者 |
| **B0 / B1 / B2** | **必報基線**（B0 池化+LR、B1 流分數堆疊、B2 eGeMAPS+LDA） | ❌ **B 流**（Voice quality）。`B1` 是基線名，`B 流` 是特徵流 |
| **主模型 M1 / 里程碑 M1** | 兩個不同東西：①**主模型 M1**（TCN＋MLP）；②本 repo 的**里程碑 M1**（資料契約與合成資料產生器） | 文件中主模型一律寫「主模型 M1」，里程碑一律寫「里程碑 M1」 |
| **s（訊號強度）** | 合成產生器注入的組間差異強度。`s=0` ⇒ 兩組分布完全相同 | ❌ B1 融合式裡的流索引 `s ∈ {B,C,E}` |
| **fold / 折** | 外層受試者層級 CV 的一折 | ❌ 折內再切的**驗證子折**（early stopping 用） |

---

## 工作紀律

- 每個里程碑一個 git commit；**驗收未過不得進下一個**（進度見 `PLAN.md`）。
- 與文件規格的任何偏離 → `DEVIATIONS.md`（含理由與影響範圍）。
- 遇到模糊處：**採保守解釋、記錄在 `QUESTIONS.md`，禁止沉默自行決定**。
- 溝通與文件用繁體中文；程式碼識別字與註解用英文。

## 常用指令

```bash
make smoke     # S1：合成 12 位受試者（6/6），端到端 M2→M7，3-fold × 2 種子 → outputs/report.md
make validate  # S2/S3：s=0 虛無檢查、s 遞增的單調性檢查
make test      # S4：pytest（防洩漏套件＋全部特徵手算例）
make checklist # S5：稽核清單 P0-* 自動標記 → outputs/checklist_report.md
make all       # 上面全部
```
