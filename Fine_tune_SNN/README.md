# Whisper + ReLU Student 語者分離（知識蒸餾）

> **方法**: 以 **Bi-LSTM Teacher** 提供軟標籤，訓練 **ReLU 前饋 Student**；**`pipeline`** 在評估完 ANN Student 後，可將 **Conv 頭** 轉為 **snntorch Leaky LIF**（Whisper 與 WeightedAvg 仍為 ANN），並可選 **只微調 SNN 頭** 數個 epoch。  
> **入口**: `main.py`（子命令：`train` / `evaluate` / `infer` / `pipeline` / `compare`）  
> **日期**: 2026-04（含 Pipeline SNN 步驟與可選 SNN 頭微調）  

**`config.py` 預設路徑**（訓練資料、測試音訊與 SRT、輸出目錄）請依本機環境修改；`main.py` 建立的 `Config` 以 `config.py` 預設值為基礎，並可由命令列覆寫訓練相關參數。

---

## 目錄

1. [架構說明](#架構說明)  
2. [知識蒸餾與 Student 頭細節](#知識蒸餾與-student-頭細節)  
3. [完整 Pipeline 使用方式](#完整-pipeline-使用方式)（**建議先讀**）  
4. [資料集格式](#資料集格式)  
5. [快速開始（指令速查）](#快速開始指令速查)  
6. [檔案結構](#檔案結構)  
7. [與直接訓練 Bi-LSTM 之比較](#與直接訓練-bi-lstm-之比較)  
8. [命令列參數](#命令列參數)  
9. [Median Filter 使用說明](#median-filter-使用說明)  
10. [預估效能](#預估效能)  
11. [實驗建議](#實驗建議)  
12. [疑難排解](#疑難排解)  
13. [輸出說明](#輸出說明)  
14. [參考文獻](#參考文獻)  

---

## 架構說明

### Teacher（Bi-LSTM，凍結）

Teacher 為已訓練之 **Whisper + Weighted Average + Bi-LSTM + Conv1d 分類器** 完整權重（例如 `Fine_tune_LSTM` 產出之 `whisper_*_bilstm_best.pt`）。訓練 Student 時 **不更新** Teacher。

```
音訊 → Mel → Whisper Encoder (凍結) → Weighted Average → Bi-LSTM → 分類 → teacher_logits（軟標籤）
```

### Student（ReLU 前饋，可訓練）

Student 自 Teacher **複製並凍結** Whisper 與 Weighted Average；下游改為 **純前饋 + ReLU**（Conv 或 MLP），**僅此段與最後分類層**參與梯度更新。

```
音訊 → Mel → Whisper Encoder (凍結，與 Teacher 對齊)
    → Weighted Average (凍結)
    → ReLU Student 頭 (conv 或 mlp)
    → Conv1d 分類 (→ 3 類別)
```

### 損失函數

- **CE**：Student logits 對 **真實 frame 標籤** 的交叉熵。  
- **KD**：溫度 `T` 下 Student 與 Teacher 的 **KL 散度**（軟目標）。  
- **總損失**：`α × CE + (1 − α) × KD`，其中 `α` 為 `config.kd_alpha` / `--kd-alpha`。

推論與部署時 **僅使用 Student**，不需載入 Teacher。

### Pipeline 內的混合 SNN（僅 `student-head=conv`）

`pipeline` **Step 3**（在 Step 2 評估完純 ANN Student 之後）會：

1. 以 `load_state_dict` **複製**蒸餾後 **3×Conv1d** 與 **最後分類 Conv1d** 至 **snntorch** 的 Conv + **Leaky LIF** 鏈（與 Student 的 ANN 頭 **不共用參數**，避免微調 SNN 時改壞 ANN）。  
2. 以測試 DataLoader 前幾個 batch **校準**輸入與第三層發放率之縮放，使 SNN 域與原 ANN ReLU 特徵相近。  
3. （可選）`--snn-finetune-epochs N`：校準後再以 frame-wise **CrossEntropy** 僅訓練 SNN 頭 **N** 個 epoch，`--snn-ft-lr` 控制學習率；**Whisper 與 WeightedAvg 凍結**。  
4. 以 **Whisper + WeightedAvg（ANN）→ SNN 頭** 再跑 DER，結果寫入 `outputs/snn_eval/`，並儲存 `models/whisper_{size}_student_snn_head_bundle.pt`。  

**`student-head=mlp` 或未安裝 `snntorch`** 時會自動略過並於終端機提示。可用 **`--skip-snn`** 關閉此步驟。

---

## 知識蒸餾與 Student 頭細節

| 項目 | 說明 |
|------|------|
| `--student-head conv` | 論文式 **3×Conv1d (kernel=1) + ReLU + Dropout**，通道數由 `--conv-channels` 控制（預設 256） |
| `--student-head mlp` | **Linear + ReLU + Dropout** 堆疊，維度由 `--mlp-hidden` 控制（預設 `512,256`） |
| `--kd-temp` | KD 溫度 `T`（預設 4.0） |
| `--kd-alpha` | CE 權重 `α`（預設 0.5；KD 權重為 `1−α`） |
| `--teacher-path` | Bi-LSTM Teacher 的 `.pt` 路徑 |
| `--lstm-hidden` / `--lstm-layers` | 建立 Teacher 結構用；**實際維度會在載入時依 checkpoint 權重自動校正**（避免 hidden 與檔案不符） |

### 設計考量

1. **ReLU**：利於常見 ANN→SNN（發放率 / IF）轉換流程。  
2. **凍結前段**：與 Teacher 共用語意空間，Student 專注模仿 LSTM 的決策邊界。  
3. **conv 頭**：與論文 Conv1D 設定接近，**建議作為 SNN 轉換首選**。  

---

## 完整 Pipeline 使用方式

流程：**在 `config.py` 指好資料與測試路徑** → 透過 **`main.py`** 執行訓練 / 評估 / 推論（路徑以 `config.py` 為主，命令列可覆寫訓練與 KD 相關參數）。

### 步驟一：準備 Teacher

須先有與 **相同 `model-size` 的 Whisper** 對應之 Bi-LSTM 權重（例如自 `Fine_tune_LSTM` 訓練產物）。

### 步驟二：編輯 `config.py`

與資料、輸出相關的欄位：

| 欄位 | 用途 |
|------|------|
| `train_data_dir` | 訓練根目錄（每個**子資料夾** = 一個樣本：音訊 + SRT） |
| `val_split_ratio` | 從訓練樣本中**隨機**切出驗證集之比例（預設 0.25） |
| `test_audio_path` | `evaluate` / 預設測試 DataLoader 用之單一測試音檔 |
| `test_srt_path` | 測試用參考 SRT（計算 DER 等） |
| `teacher_checkpoint` | 預設 Teacher 路徑（可被 `train` / `pipeline` 的 `--teacher-path` 覆寫） |
| `model_save_dir` / `output_dir` / `log_dir` | 模型、評估圖表、訓練曲線目錄 |
| `snn_timesteps` / `snn_beta` | SNN 時間步 T、Leaky LIF 漏電 β（`pipeline` 可用 `--snn-timesteps` / `--snn-beta` 覆寫） |
| `snn_finetune_epochs` / `snn_finetune_lr` | 結構欄位預設；**`pipeline` 實際以** `--snn-finetune-epochs` / `--snn-ft-lr` **為準**（見下方「命令列參數」說明） |

本目錄之 `dataset.py` 目前為 **SRT 為主**之資料探索與切片（與 `Fine_tune_LSTM/code` 內含 RTTM、獨立 `val_data_dir` 的進階版**未必相同**）；若需 Playlogue / RTTM，請自行對齊或合併該版 `dataset.py` 與 `config` 欄位。

### 步驟三：訓練與完整 Pipeline

```powershell
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN

# 僅知識蒸餾訓練 Student（conv 頭）
python main.py train --model-size medium --student-head conv --teacher-path "C:\path\to\whisper_medium_bilstm_best.pt"

# 訓練 + ANN 評估 + Conv 頭轉 SNN 並再評估 + 產生 ANALYSIS_REPORT.md（須 pip install snntorch）
python main.py pipeline --model-size medium --student-head conv --teacher-path "C:\path\to\whisper_medium_bilstm_best.pt" --epochs 15

# 略過 SNN 步驟（僅訓練 + ANN 評估 + 報告）
python main.py pipeline --model-size medium --student-head conv --teacher-path "C:\path\to\teacher.pt" --skip-snn

# 自訂 SNN 時間步與 LIF 漏電係數
python main.py pipeline --model-size medium --student-head conv --teacher-path "C:\path\to\teacher.pt" --snn-timesteps 16 --snn-beta 0.85

# 轉 SNN 後再微調 SNN 頭 5 個 epoch（Whisper / WeightedAvg 凍結；使用 config 之訓練資料切 train/val）
python main.py pipeline --model-size medium --student-head conv --teacher-path "C:\path\to\teacher.pt" --snn-finetune-epochs 5 --snn-ft-lr 1e-4
```

**Pipeline 步驟對照**：Step 1 蒸餾訓練 → Step 2 純 ANN Student 評估 → Step 3 **ANN→SNN（Conv 頭）** → 可選 **SNN 頭微調**（`--snn-finetune-epochs>0`）→ 混合模型 DER 與存 bundle → Step 4 產生報告。若未 `--skip-snn` 且 Step 3 有執行，`ANALYSIS_REPORT.md` 含 §3.3（SNN DER；若有微調會註明 epoch / lr）。

### 步驟四：評估（載入已存 Student）

**必須**與訓練時一致指定 `--model-size` 與 `--student-head`（及 MLP 時之 `--mlp-hidden`），否則結構與權重不對齊。

```powershell
python main.py evaluate --model-path models\whisper_medium_student_relu_conv_best.pt --student-head conv
```

評估用之測試音訊與標註來自 **`config.test_audio_path` / `test_srt_path`**。

### 步驟五：對任意音檔推論

```powershell
python main.py infer --model-path models\whisper_medium_student_relu_conv_best.pt --student-head conv --audio-path path\to\audio.wav

# 可選：提供參考 SRT 以利評估輔助輸出
python main.py infer --model-path models\whisper_medium_student_relu_conv_best.pt --student-head conv --audio-path audio.wav --srt-path ref.srt
```

### 步驟六：比較多個 Whisper 大小的 Student

假設已分別訓練並存成 `models\whisper_{size}_student_relu_conv_best.pt`：

```powershell
python main.py compare --model-sizes tiny small medium --student-head conv
```

---

## 資料集格式

### 模型期望的標註語意（三類）

- **0 — Child（小孩）**  
- **1 — Adult（成人 / 測試者）**  
- **2 — Silence**（無標註覆蓋之 frame）  

**SRT**：字幕文字內需含語者標籤，例如 `小孩:`、`測試者:` 或 `child:`、`adult:`（詳見 `dataset.py` 的 `SRTParser`）。

### 目錄範例（訓練）

```text
train_data_dir/
  受試者A/
    xxx.mp3
    Aegisub.srt   （或 GTruth.srt、任意 *.srt）
  受試者B/
    ...
```

修改 `config.py` 後無需改 `main.py`，直接重新執行 `train` / `pipeline` / `evaluate` 即可。

---

## 快速開始（指令速查）

```powershell
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN

python main.py pipeline --model-size medium --student-head conv --teacher-path "..\Fine_tune_LSTM\models\whisper_medium_bilstm_best.pt" --epochs 15
```

```powershell
# 轉 SNN 後微調 SNN 頭（需 train_data_dir 可建 train/val）
python main.py pipeline --model-size medium --student-head conv --teacher-path "..\Fine_tune_LSTM\models\whisper_medium_bilstm_best.pt" --epochs 15 --snn-finetune-epochs 3 --snn-ft-lr 1e-4
```

```powershell
python main.py evaluate --model-path models\whisper_medium_student_relu_conv_best.pt --student-head conv
```

```powershell
python main.py infer --model-path models\whisper_medium_student_relu_conv_best.pt --student-head conv --audio-path path\to\audio.mp3
```

```powershell
python main.py compare --model-sizes tiny small medium --student-head conv
```

---

## 檔案結構

```text
Fine_tune_SNN/
├── main.py            # 統一入口（train / evaluate / infer / pipeline / compare）
├── config.py          # 路徑、KD、student 頭、SNN T/β 與可選 SNN 微調預設等
├── model.py           # ReLU Student（含 create_student_from_teacher、load_model）
├── teacher_lstm.py    # Bi-LSTM Teacher 定義與 load_teacher（含 LSTM 維度自權重推斷）
├── train.py           # 知識蒸餾訓練迴圈
├── ann_to_snn.py      # Conv 頭複製 → snntorch LIF、校準、可選 SNN 頭微調、混合評估、bundle
├── dataset.py         # SRT、DataLoader、訓練/驗證切分
├── evaluate.py        # DER、混淆矩陣、甘特圖、推論輔助
├── requirements.txt
├── README.md
├── models/            # Student 最佳權重、checkpoint
├── logs/              # training_history.json、training_curves.png
└── outputs/           # 評估圖表、ANALYSIS_REPORT.md（pipeline）
```

---

## 與直接訓練 Bi-LSTM 之比較

| 面向 | Bi-LSTM（Teacher） | ReLU Student（本專案） |
|------|-------------------|-------------------------|
| 時序建模 | LSTM 雙向隱狀態 | 僅依 Whisper 與 frame 級前饋（無 RNN） |
| 推論成本 | 較高（序列依賴） | 前饋，較易平行化 |
| SNN 轉換 | 門控與循環不利標準 ANN→SNN | ReLU + Conv/Linear 較易對接 |
| 準確度 | 通常較高（若資料吃時序） | 依 KD 與 α、T 調整，可能接近 Teacher |

---

## 命令列參數

### Student / KD 專用（`train`、`pipeline`）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--student-head` | conv | `conv` 或 `mlp` |
| `--conv-channels` | 256 | conv 頭通道數 |
| `--mlp-hidden` | 512,256 | MLP 隱藏層，逗號分隔 |
| `--teacher-path` | （見 main.py 預設） | Bi-LSTM Teacher `.pt` |
| `--lstm-hidden` | 256 | 僅供建立 Teacher 結構；載入時會依權重校正 |
| `--lstm-layers` | 2 | Teacher LSTM 層數 |
| `--kd-temp` | 4.0 | KD 溫度 T |
| `--kd-alpha` | 0.5 | CE 權重 α |

### 通用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model-size` | medium | Whisper：tiny / base / small / medium / large |
| `--epochs` | 15 | 訓練 epochs（`train` / `pipeline`） |
| `--batch-size` | 8 | 批次大小 |
| `--lr` | 5e-4 | 學習率 |
| `--window-size` | 20.0 | 音訊窗口（秒） |

### `evaluate` / `infer`

| 參數 | 說明 |
|------|------|
| `--model-path` | Student 權重（**必須**與 `--student-head` 等一致） |
| `--srt-path` | （`infer` 可選）參考 SRT |

### `compare`

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model-sizes` | tiny small medium | 要比較的 Whisper 大小列表 |

### `pipeline` 專用（SNN）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--skip-snn` | 關 | 若指定則不執行 Conv 頭轉 SNN 與混合評估 |
| `--snn-timesteps` | 8 | Leaky LIF 展開之時間步 T |
| `--snn-beta` | 0.9 | Leaky 漏電係數 β |
| `--snn-finetune-epochs` | 0 | 轉 SNN 並校準後，僅微調 **複製後** SNN 頭（Conv + LIF + 分類 Conv）之 epoch；0 表示不微調 |
| `--snn-ft-lr` | 1e-4 | 上述微調之 AdamW 學習率；`weight_decay` 沿用 `config.weight_decay` |

微調時會再次呼叫 `create_dataloaders(config)`（與 KD 訓練相同之 `train_data_dir` + `val_split_ratio` 切分），**Whisper、WeightedAvg、Student 的 ANN 頭與分類器皆凍結**，梯度僅進入 `snn_head`。微調使用 frame-wise CrossEntropy，**不啟用 AMP**（與 SNN 評估路徑一致）。

`config.py` 中有同名欄位 `snn_timesteps`、`snn_beta`、`snn_finetune_epochs`、`snn_finetune_lr` 供結構與其他程式碼參考。**`pipeline` 執行時**：T / β 來自 **`--snn-timesteps` / `--snn-beta`**；微調與否與 lr 來自 **`--snn-finetune-epochs` / `--snn-ft-lr`**（預設 0 / 1e-4），並會寫入 `ANALYSIS_REPORT.md`。僅修改 `config.py` 的 `snn_finetune_*` **不會**改變 `pipeline` 的微調行為，除非一併調整 `main.py` 的 argparse 預設或改為從 `Config` 讀取。

### Median Filter

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--use-median` | 停用 | 啟用中值濾波 |
| `--no-median` | — | 停用 |
| `--median-size` | 5 | 核大小（須為奇數） |

---

## Median Filter 使用說明

用於平滑**推論或評估**的 frame 級類別，**不參與 KD 訓練**。

### 基本原理

- 預設核大小 5 → 約 5 × 20ms = 100ms 平滑窗口  
- 可減少短片段雜訊；過大可能抹掉短語音事件  

### 使用範例

```powershell
python main.py evaluate --model-path models\whisper_medium_student_relu_conv_best.pt --student-head conv --use-median
python main.py infer --model-path models\whisper_medium_student_relu_conv_best.pt --student-head conv --audio-path audio.mp3 --use-median --median-size 7
python main.py pipeline --model-size medium --student-head conv --teacher-path "...\teacher.pt" --use-median
```

### 注意事項

- 核大小須為**奇數**（3, 5, 7, 9）  
- 建議先不做後處理建立基線，再比較啟用後差異  

---

## 預估效能

Student 可訓練參數主要來自 ReLU 頭與分類層，**通常少於完整 Bi-LSTM Teacher**；單 epoch 時間約與「僅訓練小頭」相當，但仍需 **每 batch 通過凍結 Whisper** 與 Teacher 前向，整體仍與 Whisper 尺寸強相關。

| 模型 | 粗估（依硬體而異） |
|------|-------------------|
| Whisper-tiny + KD Student | 較快 |
| Whisper-medium + KD Student | 常見實驗設定 |
| Whisper-large + KD Student | 較慢、佔 VRAM 較大 |

---

## 實驗建議

1. **先用 tiny 確認 Teacher 可載入、資料可讀**  
   ```powershell
   python main.py train --model-size tiny --student-head conv --teacher-path "...\whisper_tiny_bilstm_best.pt" --epochs 3
   ```

2. **調整 KD**  
   - 增大 `--kd-alpha`：更相信真實標籤  
   - 增大 `--kd-temp`：軟分布更平滑  

3. **conv 與 mlp 對照**  
   ```powershell
   python main.py pipeline --model-size medium --student-head conv --teacher-path "...\teacher.pt"
   python main.py pipeline --model-size medium --student-head mlp --mlp-hidden "512,256" --teacher-path "...\teacher.pt"
   ```

4. **與 Teacher DER 對照**  
   同一測試檔上比較 `Fine_tune_LSTM` 之 Bi-LSTM 與本專案 Student 之 `evaluate` 結果。  

5. **ANN→SNN 後 DER 偏差**  
   可先試調 `--snn-timesteps` / `--snn-beta` 與校準；若仍落差大，再以 **`--snn-finetune-epochs`** 小步數（例如 3～10）微調 SNN 頭，並觀察 `outputs/snn_eval/` 與 §3.3 報告段落。

---

## 疑難排解

### Teacher 載入出現 LSTM shape mismatch

已於 `teacher_lstm.py` 的 `load_teacher` 內依 **`bilstm_head.lstm.weight_ih_l0` 等權重形狀**自動推斷 `lstm_hidden_size` / `lstm_num_layers`。若仍失敗，請確認 `--teacher-path` 與 `--model-size` 與當初訓練 Teacher 時一致。

### Windows 與 DataLoader

若於 Epoch 開始時出現與 **多進程 DataLoader、預載入大資料集** 相關之 pickle / 路徑錯誤，可將 `config.py` 的 **`num_workers` 改為 0** 改為主進程載入（與 `Fine_tune_LSTM/code` README 建議類似）。

### `evaluate` / `infer` 結構不符

`load_model` 會依 checkpoint 內之 `config` 還原 `student_head_type` 等，但 **`--model-size` 仍須正確**（Whisper 結構與權重一致）。MLP 訓練者評估時須帶相同 `--mlp-hidden`。

### SNN 步驟無法執行

- 若出現「請安裝 snntorch」：執行 `pip install snntorch`（已列於 `requirements.txt`）。  
- 若使用 **`--student-head mlp`**：目前僅對 **Conv 頭** 實作 SNN；Pipeline 會略過 Step 3 並提示。  

### SNN 頭微調報錯或無資料

- **`--snn-finetune-epochs` > 0** 時，程式需能從 **`train_data_dir`** 建立 train/val（與 `train` 相同邏輯）。若目錄為空或格式不符，會在 `create_dataloaders` 階段失敗；請先確認 [資料集格式](#資料集格式)。  
- 若未裝 `snntorch`，不會進入微調（整段 Step 3 略過或中止依程式分支而定）。

---

## 輸出說明

| 產出 | 說明 |
|------|------|
| `models/whisper_{size}_student_relu_{conv\|mlp}_best.pt` | 驗證最佳 Student |
| `models/checkpoint_kd_latest.pt` | 最新 checkpoint |
| `models/whisper_{size}_student_snn_head_bundle.pt` | **Pipeline Step 3** 儲存之 SNN 頭 state + `input_scale` / `output_scale` / T / β；metadata 中可含 `snn_finetune_epochs` / `snn_finetune_lr`（若曾微調） |
| `logs/training_curves.png` | 訓練曲線（含 CE / KD） |
| `logs/training_history.json` | 數值歷史 |
| `outputs/confusion_matrix.png` 等 | Step 2 純 ANN Student 評估視覺化 |
| `outputs/snn_eval/` | **混合 SNN 頭** 評估之圖表與 JSON（若有執行 Step 3） |
| `outputs/ANALYSIS_REPORT.md` | **`pipeline` Step 4** 產生；含 §3.3 SNN DER（若有 Step 3） |

---

## 參考文獻

- Xu, A., et al. (2024). "Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions." *Interspeech 2024*.（Whisper + 加權層 + Conv 頭之論文設定；本專案 Student 之 `conv` 頭與此精神對齊。）  
- Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network." *NIPS Deep Learning Workshop*.（知識蒸餾概念。）  

本專案 **`pipeline` 已內建**以 **snntorch** 將蒸餾 **Conv 頭**（含分類 Conv 之權重複製）轉為 Leaky LIF，並可選 **僅訓練 SNN 支路**；更進階之晶片部署或不同神經元模型請再參考 [snntorch 文件](https://snntorch.readthedocs.io/) 調整。
