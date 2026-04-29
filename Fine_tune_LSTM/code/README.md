# Whisper + Bi-LSTM 語者分離模型

> **架構變更**: 將原本的 3 層 Conv1D 替換為 2 層 Bi-LSTM  
> **Encoder 微調**: 預設在 **凍結 Whisper Encoder 基底**的前提下，於 FFN（`fc1` / `fc2`）套用 **PEFT LoRA**（對齊 [arXiv:2601.17640](https://arxiv.org/abs/2601.17640) 中 rank 16 的設定；見下方 [Whisper Encoder 與 LoRA](#whisper-encoder-與-lora)）  
> **日期**: 2026-01-24（README 更新：2026-04-16）  
> **資料**: 支援 **SRT** 與 **RTTM** 標註；可搭配 **Playlogue** 官方 train/val/test 目錄（2026-03-24）  
> **路徑**: 程式與本說明位於 **`Fine_tune_LSTM/code/`**；請在該目錄執行 `python main.py ...`。`config.py` 內「資料路徑設定」可依資料集修改（檔內附註解範例）。

---

## 目錄

1. [環境與依賴](#環境與依賴)  
2. [架構說明](#架構說明)  
3. [Whisper Encoder 與 LoRA](#whisper-encoder-與-lora)  
4. [Bi-LSTM 架構細節](#bi-lstm-架構細節)  
5. [完整 Pipeline 使用方式](#完整-pipeline-使用方式)（**建議先讀**；含 [K-fold 交叉驗證](#k-fold-交叉驗證pipeline)）  
6. [資料集格式與切換方式](#資料集格式與切換方式)  
7. [快速開始（指令速查）](#快速開始指令速查)  
8. [檔案結構](#檔案結構)  
9. [與 Conv1D 版本比較](#與-conv1d-版本比較)  
10. [命令列參數](#命令列參數)  
11. [Median Filter 使用說明](#median-filter-使用說明)  
12. [預估效能](#預估效能)  
13. [實驗建議](#實驗建議)  
14. [疑難排解（Windows DataLoader）](#疑難排解windows-dataloader)  
15. [輸出說明](#輸出說明)  
16. [參考文獻](#參考文獻)  

---

## 環境與依賴

1. 依 [PyTorch 官網](https://pytorch.org/) 安裝適合你 CUDA 版本的 `torch` / `torchaudio`。  
2. 其餘套件：

```powershell
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\code
pip install -r requirements.txt
```

其中包含 **`transformers`**（Whisper）與 **`peft`**（LoRA）。若僅補裝 LoRA：

```powershell
pip install peft
```

---

## 架構說明

### 原始架構 (Conv1D 版本)

```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
3 × Conv1D (channels=256, kernel=1) + ReLU + Dropout(0.2)
    ↓
Conv1D Classifier (→ 3 類別)
```

### 新架構 (Bi-LSTM + LoRA，預設)

```
Whisper Encoder 基底（凍結）+ LoRA 於各層 FFN 的 fc1 / fc2（可訓練）
    ↓
Weighted Average Layer
    ↓
2 × Bi-LSTM (hidden=256, bidirectional=True) + LayerNorm + Dropout(0.2)
    ↓
Conv1D Classifier (→ 3 類別)
```

若將 `config.use_lora = False` 且 `freeze_encoder = True`，則回到「整顆 Encoder 權重凍結、無 LoRA」行為；若兩者皆為 False，則 Encoder 可全量微調（不建議與論文設定混用，除非你有意為之）。

---

## Whisper Encoder 與 LoRA

本專案使用 [Hugging Face PEFT](https://github.com/huggingface/peft) 的 `LoraConfig` / `get_peft_model`，**僅對 `WhisperEncoder` 內的 `fc1`、`fc2`（FFN）** 注入 LoRA；基底權重維持凍結，與論文中「不解凍 Whisper encoder、改在 feed-forward 區塊做低秩適配」的敘述一致（實作細節仍以本 repo 為準）。

### `config.py` 相關欄位

| 欄位 | 預設 | 說明 |
|------|------|------|
| `use_lora` | `True` | 是否對 Encoder FFN 套用 LoRA |
| `lora_rank` | `16` | LoRA rank（論文為 16） |
| `lora_alpha` | `16` | LoRA scaling 用 alpha |
| `lora_dropout` | `0.05` | LoRA dropout |
| `freeze_encoder` | `True` | 僅在 **`use_lora == False`** 時會將整顆 Encoder `requires_grad=False` |

### 權重儲存與載入

- **訓練**（`train.py`）：checkpoint 內含 `model_state_dict`（含 LoRA 與 Bi-LSTM、Classifier）以及 `config` 中的 `use_lora`、`lora_rank`、`lora_alpha`、`lora_dropout`。  
- **載入**（`model.py` 的 `load_model`）：會先讀 checkpoint 的 `config`，還原 LoRA 相關設定再建模；**舊版 checkpoint 若有 `config` 但沒有 `use_lora` 欄位，會視為未使用 LoRA**，以便載入訓練於 LoRA 導入前的權重。

### 命令列

`train` / `evaluate` / `infer` / `pipeline` 皆支援 **`--use-lora`** 與 **`--no-use-lora`**；未指定時沿用 `config.py` 預設。評估與推論時若 checkpoint 已寫入 `use_lora`，**以 checkpoint 為準**，可避免架構與 `state_dict` 鍵名不一致。

**說明**：目前程式僅對 **Encoder** 做凍結或 LoRA；**Whisper Decoder 預設仍可訓練**（與舊版「只凍結 encoder」行為相同）。因此「可訓練參數」統計會包含 decoder；若只要訓練 LoRA + 頭部，需另行在程式中凍結 decoder（本 README 不展開）。

---

## Bi-LSTM 架構細節

| 參數 | 值 |
|------|-----|
| input_size | Whisper hidden dim (768/1024/1280) |
| hidden_size | 256 (單向) / 512 (雙向拼接) |
| num_layers | 2 |
| dropout | 0.2 (層間) |
| batch_first | True |
| bidirectional | True |

### 設計考量

1. **雙向 LSTM**: 同時利用過去和未來的上下文資訊  
2. **Layer Normalization**: 穩定訓練過程  
3. **正交初始化**: 用於循環權重，幫助捕捉長期依賴  
4. **Forget Gate Bias = 1**: 幫助 LSTM 記憶長期資訊  

---

## 完整 Pipeline 使用方式

本專案流程：**在 `code/config.py` 指好資料路徑** → **訓練 / 評估 / 推論** 皆讀同一套設定（`main.py` 子命令會建立 `Config`，路徑以 `config.py` 預設值為主，除非你在程式內另行覆寫）。

### 步驟一：準備資料

依你使用的資料集，將音訊與標註放在對應目錄（詳見下一節 [資料集格式與切換方式](#資料集格式與切換方式)）。

- **原本（自閉症 / 自訂 SRT）**：每個受試者一個子資料夾，內有 **一支音訊** + **一支 SRT**（語者行前綴須能被 `dataset.py` 辨識，例如 `小孩:`、`測試者:` 或 `child:`、`adult:`）。  
- **Playlogue**：在專案外層 `Playlogue` 完成裁切與對照後，執行  
  `Playlogue/scripts/prepare_playlogue_finetune_layout.py`  
  會在 **`Fine_tune_LSTM/data/playlogue_splits/`** 底下產生 `train/`、`val/`、`test/`，每個 clip 一個子資料夾：`audio.wav` + `diarization.rttm`。

### 步驟二：編輯 `config.py`

**目前儲存庫預設**已設為 Playlogue 經 `prepare_playlogue_finetune_layout.py` 產生之目錄（`train` / `val` / `test` 與一組測試用 `audio.wav` + `diarization.rttm`）。若目錄尚未建立，請先執行該腳本，或暫時改回你的路徑。

與資料相關的欄位：

| 欄位 | 用途 |
|------|------|
| `train_data_dir` | 訓練用根目錄（其下每個**子資料夾** = 一個樣本） |
| `val_data_dir` | **選填**。若設成有效目錄（例如 Playlogue 的 `val/`），驗證集**只從此目錄**掃描，**不再**使用 `val_split_ratio` 隨機切分。若改回 `None` 或空字串，則恢復「從 `train_data_dir` 隨機切驗證集」。 |
| `val_split_ratio` | 僅在 **未設定** `val_data_dir` 時生效。 |
| `enable_kfold_cv` | `pipeline` 是否改跑 K-fold；與獨立 `val_data_dir` **互斥**。 |
| `n_folds` | K-fold 折數（預設 5）；亦可由 `--n-folds` 覆寫。 |
| `test_audio_path` | 單一測試音檔（給 `evaluate` / 預設測試 DataLoader 用） |
| `test_srt_path` | 測試用 SRT（可與 RTTM擇一或並存） |
| `test_rttm_path` | 測試用 RTTM；若路徑有效，**優先於** SRT 載入標註 |

其餘：`model_save_dir`、`output_dir`、`log_dir`、Whisper 大小、`batch_size` 等維持原意。

### 步驟三：訓練與完整 Pipeline

```powershell
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\code

# 僅訓練（預設啟用 LoRA，見 config.use_lora）
python main.py train --model-size medium --epochs 15

# 關閉 LoRA（與舊 checkpoint / 全凍結 encoder 對齊時使用）
python main.py train --model-size medium --epochs 15 --no-use-lora

# 訓練 + 以 config 中的 test_* 評估並產生報告
python main.py pipeline --model-size medium --epochs 15
```

### K-fold 交叉驗證（`pipeline`）

當 **`config.enable_kfold_cv = True`**（或指令加上 **`--kfold`**）時，`pipeline` 會改為 **K 折交叉驗證**（預設 **5** 折，由 **`config.n_folds`** 或 **`--n-folds`** 指定）：

1. **切分**：在「訓練用樣本池」上以 `sklearn.model_selection.KFold` 做樣本級交叉；若 **`enable_fixed_split_from_train_dir=True`**，則池子為依 `train_val_test_ratio` 切出之 **train+val**（**測試集先保留**，不參與折疊）。  
2. **每折**：獨立訓練一個模型（該折內仍以驗證損失存**該折**最佳權重），並在**與單次 pipeline 相同之固定測試集**上算 DER 等指標。權重與圖表寫入 `model_save_dir/kfold_fold{k}`、`log_dir/kfold_fold{k}`、`output_dir/kfold_fold{k}`。  
3. **彙總輸出**（主結論）：寫入 **`output_dir/ANALYSIS_REPORT.md`** 與 **`output_dir/kfold_summary.json`**，報告主表為各折測試指標之 **平均 ± 標準差**；**不**另選「整體最佳折」覆寫為唯一對外模型。  

**限制**：與 **獨立 `val_data_dir` 目錄** 互斥；啟用 K-fold 時請將 `val_data_dir` 留空。樣本數須 **≥ `n_folds`**。

**不保留測試集（資料少、只報 CV 平均）**：可將 **`train_val_test_ratio` 設為例如 `(8, 2, 0)`**（第三項為 0）。此時 K-fold 的樣本池為 **train+val 合併（即全部樣本）**，不再先切出一塊測試；彙總 DER 改為各折在 **該折驗證集** 上計算後再平均（見 `kfold_summary.json` 的 `eval_metric_source`）。此做法與「另留固定測試集」不同，且驗證集亦用於 early stopping，解讀上略偏樂觀，屬小樣本常見取捨。**一般單次 `pipeline`（非 K-fold）** 在僅設 `train_data_dir`、無 `test_data_dir` 時仍須有測試來源；`(8,2,0)` 主要搭配 **K-fold** 使用。

```powershell
# 方式一：在 config.py 設 enable_kfold_cv = True 後直接 pipeline
python main.py pipeline --model-size medium --epochs 15

# 方式二：僅本次執行啟用（不改 config 檔）
python main.py pipeline --model-size medium --epochs 15 --kfold
# 指定 5 折以外（例如 10）
python main.py pipeline --model-size medium --epochs 15 --kfold --n-folds 10
```

### 步驟四：評估（載入已存模型）

```powershell
python main.py evaluate --model-path models/whisper_medium_bilstm_best.pt
```

評估時使用的測試音訊與標註來自 **`config.test_audio_path` / `test_srt_path` / `test_rttm_path`**。

### 步驟五：對任意音檔推論

```powershell
# 僅推論（可選參考標註以計算 DER）
python main.py infer --model-path models/whisper_medium_bilstm_best.pt --audio-path path\to\audio.wav

# 參考 SRT
python main.py infer --model-path models/whisper_medium_bilstm_best.pt --audio-path audio.wav --srt-path ref.srt

# 參考 RTTM（與 SRT 皆提供時，優先 RTTM）
python main.py infer --model-path models/whisper_medium_bilstm_best.pt --audio-path audio.wav --rttm-path ref.rttm
```

---

## 資料集格式與切換方式

### 模型期望的標註語意（三類）

- **0 — Child（小孩）**  
- **1 — Adult（成人 / 測試者）**  
- **2 — Silence**（無標註覆蓋的 frame）

**SRT**：字幕文字内需含語者標籤，例如 `小孩:`、`測試者:` 或 `child:`、`adult:`（詳見 `dataset.py` 的 `SRTParser`）。  
**RTTM**：目前將 **CHI*** 視為小孩、**ADT / INV / MOT / FAT** 等視為成人（Playlogue 為 **CHI / ADT**），其餘標籤略過（該時段為靜音）。

### 模式 A：原本資料集（每受試者一資料夾 + SRT）

目錄範例：

```text
train_data_dir/
  受試者A/
    xxx.mp3
    Aegisub.srt   （或 GTruth.srt、任意 *.srt）
  受試者B/
    ...
```

**切換回此模式時請檢查：**

1. `train_data_dir` 指到上述根目錄。  
2. `val_data_dir = None`（或 `""`），讓驗證集依 `val_split_ratio` 從訓練根目錄隨機切。  
3. `test_audio_path` / `test_srt_path` 指到你的測試音訊與 SRT。  
4. `test_rttm_path = None`（若沒有 RTTM）。  

每個子資料夾內若**同時**有 `.rttm` 與 `.srt`，**優先使用 RTTM**；若你只要 SRT，請勿在該資料夾放置 `.rttm`，或僅保留 SRT。

### 模式 B：Playlogue（官方 split + RTTM）

1. 在 `Playlogue` 內完成音訊與 `clip_to_raw_audio.csv`、裁切 WAV 等（見該目錄下 `scripts/`）。  
2. 執行：  
   `python Playlogue/scripts/prepare_playlogue_finetune_layout.py`  
3. **`config.py` 預設已對應**（本機路徑為 `Fine_tune_LSTM/data/playlogue_splits/...`）。若你的專案不在同一磁碟或想換測試 clip，只要改 `train_data_dir`、`val_data_dir`、`test_audio_path`、`test_rttm_path` 即可；`test_srt_path` 可留空字串。範例：

```python
train_data_dir = r"...\Fine_tune_LSTM\data\playlogue_splits\train"
val_data_dir   = r"...\Fine_tune_LSTM\data\playlogue_splits\val"
test_audio_path = r"...\playlogue_splits\test\<某clip_id>\audio.wav"
test_rttm_path  = r"...\playlogue_splits\test\<某clip_id>\diarization.rttm"
test_srt_path   = r""   # 有 RTTM 時以 RTTM 為準
```

**注意：** `test` 內的 clip **不要**再複製進 `train_data_dir`，以免洩漏。

### 模式 C：其他資料集

只要符合下列任一組合即可：

- 每個樣本一個子資料夾；內有 **一支音訊**（`mp3/wav/flac/m4a`）+ **SRT**（語者標籤需對應到 Child/Adult），**或**  
- 同左 + **RTTM**（語者欄位需能被 `RTTMParser` 映射，否則需改 `dataset.py` 對照表）。

若你的驗證集是「另一批人」而非隨機切分，可：

- 設 `train_data_dir` = 訓練根目錄、`val_data_dir` = 驗證根目錄（結構與訓練相同），**或**  
- 全部先放在 `train_data_dir`，並**不要**設 `val_data_dir`，改用 `val_split_ratio`（不適合需嚴格 subject-independent 時）。

### 切換資料集快速檢查清單

| 項目 | 改回原本 SRT 資料 | Playlogue |
|------|-------------------|-----------|
| `train_data_dir` | 你的 Fine_tune 根目錄 | `.../playlogue_splits/train` |
| `val_data_dir` | `None` | `.../playlogue_splits/val` |
| `val_split_ratio` | 依需求（例如 0.25） | 有 `val_data_dir` 時此比例**不**用於切分 |
| `test_audio_path` / 標註 | 你的測試音檔 + SRT | `test/某clip/audio.wav` + `diarization.rttm` |
| `test_rttm_path` | `None` | 對應 RTTM 路徑 |

修改 `config.py` 後無需改 `main.py`，直接重新執行 `train` / `pipeline` / `evaluate` 即可。

---

## 快速開始（指令速查）

```powershell
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\code

# 預設設定 pipeline（路徑以 config.py 為準；預設 LoRA 開啟）
python main.py pipeline --model-size medium --epochs 15

# 自訂 LSTM
python main.py pipeline --model-size medium --epochs 15 --lstm-hidden 256 --lstm-layers 2

# 明確指定 LoRA 開關（與 config 預設擇一即可）
python main.py train --model-size medium --epochs 15 --use-lora
python main.py train --model-size medium --epochs 15 --no-use-lora

# 較大／較新 Whisper 後端（需足夠 VRAM；HF 名稱見「命令列參數」一節）
python main.py pipeline --model-size large-v3 --epochs 15 --batch-size 8
python main.py pipeline --model-size large-v3-turbo --epochs 30 --batch-size 8
# 與 large-v3-turbo 為同一權重，checkpoint 檔名會是 whisper_large-turbo_...
python main.py pipeline --model-size large-turbo --epochs 30 --batch-size 8
```

```powershell
python main.py evaluate --model-path models/whisper_medium_bilstm_best.pt
```

```powershell
python main.py infer --model-path models/whisper_medium_bilstm_best.pt --audio-path path\to\audio.mp3
```

---

## 檔案結構

```text
Fine_tune_LSTM/
└── code/                      # 建議在此目錄執行 python main.py
    ├── config.py              # 路徑、超參數、LoRA、val_data_dir / test_rttm_path 等
    ├── model.py               # Whisper +（可選 LoRA）+ Bi-LSTM + Classifier
    ├── train.py               # 訓練與 checkpoint（含 LoRA 設定寫入 config）
    ├── evaluate.py            # 評估與推論輔助（含 RTTM 推論選項）
    ├── dataset.py             # SRT + RTTM、DataLoader、discover_training_samples*
    ├── main.py                # 入口
    ├── requirements.txt       # 含 peft、transformers 等
    └── README.md
    （可選）../data/playlogue_splits/ …
    （執行時產出）../models/、../logs/、../outputs/ — 實際路徑以 config 為準
```

上層專案中的 **`Playlogue/scripts/`** 可負責產生 `playlogue_splits` 等資料目錄（與 `code/` 分開維護）。

---

## 與 Conv1D 版本比較

執行完訓練後，可比較兩種架構的 DER：

| 架構 | 優點 | 缺點 |
|------|------|------|
| **Conv1D** | 平行計算、訓練較快 | 局部感受野、長距離依賴較難 |
| **Bi-LSTM** | 長距離依賴、雙向上下文 | 序列計算、訓練較慢 |

---

## 命令列參數

### Whisper 後端與 `--model-size`

`train`、`evaluate`、`infer`、`pipeline` 的 **`--model-size`** 會決定從 Hugging Face 載入的 Whisper 檢查點。實作上多數取值對應 `config.whisper_model_name`，即 **`openai/whisper-{model_size}`**；例外見下表最後一列。

| CLI `--model-size` | Hugging Face 模型 ID |
|--------------------|----------------------|
| `tiny` | `openai/whisper-tiny` |
| `base` | `openai/whisper-base` |
| `small` | `openai/whisper-small` |
| `medium` | `openai/whisper-medium` |
| `large` | `openai/whisper-large` |
| `large-v2` | `openai/whisper-large-v2` |
| `large-v3` | `openai/whisper-large-v3` |
| `large-v3-turbo` | `openai/whisper-large-v3-turbo` |
| `large-turbo` | **`openai/whisper-large-v3-turbo`**（簡寫；HF 上**沒有** `openai/whisper-large-turbo`） |

**Checkpoint 檔名**：`{model_save_dir}/whisper_{model_size}_bilstm_best.pt`，其中 `{model_size}` 與你在 CLI 寫的字串相同（故 `large-turbo` 與 `large-v3-turbo` 會存成不同檔名，若需對齊請固定使用其中一種）。

**相依性**：`large-v3`、`large-v3-turbo`、`large-turbo` 等需較新的 **`transformers`**；若 `from_pretrained` 失敗，請依 `requirements.txt` 升級後再試。

### LoRA（Whisper Encoder FFN）

| 參數 | 說明 |
|------|------|
| `--use-lora` | 啟用 Encoder 上之 LoRA（`fc1` / `fc2`）；未與 `--no-use-lora` 並用時，未指定則沿用 `config.py` |
| `--no-use-lora` | 停用 LoRA；等同關閉後依 `freeze_encoder` 決定是否整顆凍結 Encoder |

適用子命令：`train`、`evaluate`、`infer`、`pipeline`。

### LSTM 專用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--lstm-hidden` | 256 | LSTM 隱藏層維度（單向） |
| `--lstm-layers` | 2 | LSTM 層數 |

### 通用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model-size` | medium | Whisper 後端大小；可選值與 HF 對照見上一節 **「Whisper 後端與 `--model-size`」** |
| `--epochs` | 15 | 訓練 epochs |
| `--batch-size` | 8 | 批次大小 |
| `--lr` | 5e-4 | 學習率 |
| `--window-size` | 20.0 | 音訊窗口大小（秒） |

### 推論專用

| 參數 | 說明 |
|------|------|
| `--srt-path` | 參考 SRT（可選） |
| `--rttm-path` | 參考 RTTM（可選；與 SRT 皆存在時優先 RTTM） |

### Median Filter 後處理參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--use-median` | 停用 | 啟用中值濾波 |
| `--no-median` | 啟用 | 停用中值濾波 |
| `--median-size` | 5 | 核大小（須為奇數；別名 `--median-filter-size`） |

### `pipeline` 專用（K-fold）

| 參數 | 說明 |
|------|------|
| `--kfold` | 本次執行強制啟用 K-fold（等同 `config.enable_kfold_cv=True`） |
| `--no-kfold` | 本次執行強制停用 K-fold |
| `--n-folds` | 折數；未指定則沿用 `config.n_folds`（預設 5） |

---

## Median Filter 使用說明

中值濾波用於平滑推論結果，僅影響評估與推論，不影響訓練。

### 基本原理

- 預設核大小 5 → 約 5 × 20ms = 100ms 平滑窗口  
- 可減少短片段雜訊，但過大可能抹掉短語音事件  

### 使用範例

```powershell
python main.py evaluate --model-path models/whisper_medium_bilstm_best.pt --use-median
python main.py evaluate --model-path models/whisper_medium_bilstm_best.pt --use-median --median-size 7
python main.py infer --model-path models/whisper_medium_bilstm_best.pt --audio-path audio.mp3 --use-median
python main.py pipeline --model-size medium --epochs 15 --lstm-hidden 128 --use-median --median-size 5
```

### 注意事項

- 核大小須為**奇數**（3, 5, 7, 9）  
- 建議先不做後處理建立基線，再比較啟用後差異  

---

## 預估效能

下表為**歷史參考**（約略對應「僅頭部可訓練、Encoder 凍結」時的數量級）。**預設已啟用 LoRA**，且 **Whisper Decoder 仍可能參與訓練**，實際「可訓練參數」會**明顯大於**下表；請以訓練開始時 `model.py` 印出的參數統計為準。

| 模型 | 可訓練參數（參考，舊版行為） | 每 Epoch 時間 (RTX 5070 Ti) |
|------|---------------------------|----------------------------|
| Whisper-tiny + Bi-LSTM | ~2M | ~3 分鐘 |
| Whisper-small + Bi-LSTM | ~3M | ~6 分鐘 |
| Whisper-medium + Bi-LSTM | ~5M | ~12 分鐘 |
| Whisper-large + Bi-LSTM | ~8M | ~18 分鐘 |

Bi-LSTM 訓練通常比 Conv1D 長約 20–30%（序列難完全平行化）。啟用 LoRA 後 Encoder 前向會多一層適配計算，開銷通常仍可接受。

---

## 實驗建議

1. **先用 tiny 確認環境與資料可讀**  
   ```powershell
   python main.py pipeline --model-size tiny --epochs 3
   ```

2. **比較不同 hidden size**  
   ```powershell
   python main.py pipeline --model-size medium --epochs 15 --lstm-hidden 128
   python main.py pipeline --model-size medium --epochs 15 --lstm-hidden 256
   python main.py pipeline --model-size medium --epochs 15 --lstm-hidden 512
   ```

3. **與 Conv1D 版本比較**  
   - 跑 `Fine_tune_paper`（或 Conv1D 專案）與本專案 pipeline，比較 DER。  

---

## 疑難排解（Windows DataLoader）

若在 **Windows** 上於 Epoch 1 一開始出現類似：

`OSError: [Errno 22] Invalid argument`（發生在 `DataLoader` / `ForkingPickler.dump`），原因是：`SpeakerDiarizationDataset` 採 **預載入**（整段波形與標籤在記憶體），`num_workers > 0` 時子進程需 **pickle 整個 Dataset**，資料量過大時在 Windows 上容易失敗。

**處理方式：** `config.py` 的 `Config.__post_init__` 已在使用 **win32** 時自動將 **`num_workers` 改為 0**（單進程載入批次）。請重新執行 `train` / `pipeline` 即可。

在 **Linux** 上若未遇到錯誤，可維持 `num_workers` 為 4～8 以加速資料送入 GPU。

---

## 輸出說明

訓練與評估後常見產出：

- `models/whisper_{size}_bilstm_best.pt` — 最佳權重  
- `logs/training_curves.png` — 訓練曲線  
- `outputs/confusion_matrix.png` — 混淆矩陣  
- `outputs/gantt_chart.png` — 甘特圖  
- `outputs/ANALYSIS_REPORT.md` — 分析報告（`pipeline` 時產生）  

---

## 參考文獻

- Xu, A., Feng, T., Bishop, S., Lord, C., & Narayanan, S. (2026). "End-to-End Joint ASR and Speaker Role Diarization with Child-Adult Interactions." [arXiv:2601.17640](https://arxiv.org/abs/2601.17640)（文中 Whisper encoder 凍結並於 FFN 使用 **LoRA rank 16**；本 repo 為語者角色／分離頭部實驗，實作對齊程度請以程式為準。）  
- Xu, A., et al. (2024). "Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions." Interspeech 2024.  
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)  
- Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural computation.  

Playlogue 資料集說明見 [Hugging Face: playlogue/playlogue-v1](https://huggingface.co/datasets/playlogue/playlogue-v1)（授權與使用條款請自行遵守）。
