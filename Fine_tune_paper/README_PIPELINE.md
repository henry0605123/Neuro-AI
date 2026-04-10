# Pipeline 使用說明（`main.py`）

本文件說明 **Fine_tune_paper** 專案中 **`main.py` 的命令列架構**與 **完整 Pipeline** 流程。此結構與 **`Fine_tune_LSTM/main.py` 已對齊**，包含：

- 子命令：`train` / `evaluate` / `infer` / `pipeline` / `compare`
- `train` 與 `pipeline` 的 **`--epochs`**：命令列**未指定**時，訓練週期以 **`config.py` 的 `epochs`** 為準（與 LSTM 專案相同邏輯）
- `pipeline` 內部：**Step 1 訓練 → Step 2 評估 → Step 3 `generate_report` → `outputs/ANALYSIS_REPORT.md`**

差異僅在**下游架構**與**權重檔名**：本專案為論文 **Conv1D Head**、`whisper_*_diarization_best.pt`；`Fine_tune_LSTM` 為 **Bi-LSTM**、`whisper_*_bilstm_best.pt`。

---

## 1. 指令總覽

| 子命令 | 說明 |
|--------|------|
| `train` | 訓練 Whisper + Conv1D 模型 |
| `evaluate` | 載入權重，對 `config.py` 中設定的測試音訊 + SRT 評估 DER |
| `infer` | 對單一音檔推論並輸出 `.predicted.srt` |
| `pipeline` | **訓練 → 評估 → 產生 `outputs/ANALYSIS_REPORT.md`**（一次跑完） |
| `compare` | 依序評估多個 `model_size` 已存權重並列印 DER |

### 與 `Fine_tune_LSTM` 對齊的 CLI 參數（`train` / `pipeline`）

| 參數 | 說明 |
|------|------|
| `--model-size` | `tiny` / `base` / `small` / `medium` / `large` |
| `--epochs` | 未指定時沿用 **`config.py` 的 `epochs`** |
| `--batch-size` | 批次大小（預設 8） |
| `--lr` | 學習率（預設 5e-4） |
| `--window-size` | 音訊窗口秒數（預設 20） |
| `--use-median` / `--no-median` | 僅評估／推論／pipeline 的評估階段 |
| `--median-size` | 中值濾波核大小（奇數，預設 5） |

> 本專案 **無** `--lstm-hidden` / `--lstm-layers`（下游為 Conv1D 而非 Bi-LSTM）。

---

## 2. 執行前：設定 `config.py`

所有子命令透過 **`from config import Config`** 建立預設設定；路徑與硬體相關欄位請在 **`config.py`** 修改：

| 欄位 | 用途 |
|------|------|
| `train_data_dir` | 訓練資料根目錄（每子資料夾 = 一受試者；**RTTM 優先於 SRT**，與 `Fine_tune_LSTM` 相同） |
| `val_data_dir` | 可選；設為有效目錄時作為**獨立驗證集**，否則依 `val_split_ratio` 從訓練目錄隨機切分 |
| `test_data_dir` | 可選；設為有效目錄時為**多 clip 測試集**（掃描子資料夾），優先於單檔測試 |
| `test_audio_path` / `test_srt_path` / `test_rttm_path` | 單檔測試：音訊與標註；**RTTM 存在時優先於 SRT** |
| `val_split_ratio` | 未使用獨立 `val_data_dir` 時，從訓練目錄隨機切分驗證集的比例 |
| `output_dir` / `model_save_dir` / `log_dir` | 輸出、權重、日誌 |

**Windows 注意：** `Config.__post_init__` 會在 **win32** 上將 **`num_workers` 強制為 0**，避免預載入 Dataset 導致多進程 pickle 錯誤（與 Fine_tune_LSTM 相同處理）。

---

## 3. 完整 Pipeline（推薦）

等同於依序執行：**訓練 → 測試集評估 → 寫入分析報告**。

```powershell
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper

python main.py pipeline --model-size medium --epochs 15
# 或省略 --epochs，週期以 config.py 的 epochs 為準（與 Fine_tune_LSTM 相同）
python main.py pipeline --model-size medium
```

### Pipeline 內部步驟（與 `Fine_tune_LSTM/main.py` 一致）

1. **Step 1：訓練** — 呼叫 `train_model(config)`，最佳權重寫入 `models/whisper_{size}_diarization_best.pt`  
2. **Step 2：評估** — `create_test_dataloader(config)` + `Evaluator.evaluate()`，DER、混淆矩陣、甘特圖等於 `output_dir`  
3. **Step 3：報告** — `generate_report()` 產生 **`outputs/ANALYSIS_REPORT.md`**（含論文 baseline 對照表、訓練摘要、DER）

### 可選：評估階段 Median Filter

與 `evaluate` / `infer` 相同：`--use-median`、`--median-size`（須為奇數）。

```powershell
python main.py pipeline --model-size medium --epochs 15 --use-median --median-size 5
```

---

## 4. 分步指令

### 訓練

```powershell
python main.py train --model-size medium --epochs 15 --batch-size 8 --lr 5e-4 --window-size 20
```

### 評估（需已有權重）

```powershell
python main.py evaluate --model-path models/whisper_medium_diarization_best.pt --model-size medium
```

### 推論（單一音檔）

```powershell
python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path your.wav
# 可選：提供參考 SRT 以在推論結束後顯示 DER
python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path your.wav --srt-path ref.srt
```

### 多模型比較

```powershell
python main.py compare --model-sizes tiny small medium
```

---

## 5. 與 `Fine_tune_LSTM` 的對照

| 項目 | Fine_tune_paper（本專案） | Fine_tune_LSTM |
|------|---------------------------|----------------|
| 下游 Head | 3×Conv1D（論文） | 2×Bi-LSTM |
| 權重檔名 | `whisper_*_diarization_best.pt` | `whisper_*_bilstm_best.pt` |
| `pipeline` 流程 | 訓練 → 評估 → `ANALYSIS_REPORT.md` | 相同 |
| `config` | 已支援 `val_data_dir`、`test_data_dir`、`test_rttm_path`（與 LSTM 語意對齊） | 相同 |

兩專案皆可使用 **Playlogue 風格目錄 + `diarization.rttm`**；差異僅在下游 **Conv1D** vs **Bi-LSTM** 與權重檔名。

---

## 6. 產出檔案（Pipeline 成功後）

- `models/whisper_{size}_diarization_best.pt`  
- `logs/training_curves.png`、`logs/training_history.json`  
- `outputs/confusion_matrix.png`、`gantt_chart.png`、`gantt_comparison.png`  
- `outputs/ANALYSIS_REPORT.md`  
- `outputs/evaluation_results.json`（若評估程式有寫入）  

---

## 7. 論文與任務定義

- 論文：[Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions](https://arxiv.org/abs/2406.07890)（Interspeech 2024）  
- 本實作採 **3 分類**（Child / Adult / Silence），與目前 `config.py`、`dataset.py` 一致；標註請使用 `小孩:`、`測試者:` 等格式（見主 README「資料準備」）。
