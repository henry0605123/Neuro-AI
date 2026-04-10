# Whisper 語者分離模型訓練 Pipeline

> **論文復現**：Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions
> 
> **論文連結**：https://arxiv.org/abs/2406.07890
> 
> **會議**：Interspeech 2024

---

## 📋 目錄

1. [專案簡介](#專案簡介)
2. [**Pipeline 與 main.py（完整說明）**](README_PIPELINE.md) — **建議閱讀**：與 `Fine_tune_LSTM/main.py` 指令結構對齊  
3. [環境需求](#環境需求)
4. [安裝步驟](#安裝步驟)
5. [資料準備](#資料準備)
6. [快速開始](#快速開始)
7. [詳細使用說明](#詳細使用說明)
8. [模型架構](#模型架構)
9. [RTX 5070 Ti 效能優化](#rtx-5070-ti-效能優化)
10. [常見問題](#常見問題)

---

## 專案簡介

本專案實作了基於 Whisper 語音基礎模型的 **child-adult speaker diarization（兒童-成人語者分離）** 系統。此系統能夠自動判斷「誰在何時說話」，特別針對兒童與成人的互動場景進行優化。

### 核心特色

- 🎯 **Whisper Encoder** 作為特徵提取器
- 🔧 **可學習的加權平均層** 整合所有 Hidden Layers
- 🧠 **Conv Head** 進行 Frame-level 分類
- ⚡ **混合精度訓練 (AMP)** 加速訓練
- 📊 **完整的評估工具**：DER、混淆矩陣、甘特圖視覺化

### 輸出類別（3 分類，與 `config.py` 一致）

| 標籤 | 類別 | 說明 |
|------|------|------|
| 0 | Child | 小孩語音 |
| 1 | Adult | 測試者/成人語音 |
| 2 | Silence | 靜音/背景 |

> 註：訓練管線以 **無 Overlap** 的 frame 標籤為主（與目前 `dataset` 之 Last-Write-Wins 策略一致）。若 SRT 中出現多語者同段，實作上可對應至特定標籤策略，請以程式為準。

---

## 環境需求

### 硬體需求

| 項目 | 最低需求 | 建議配置 |
|------|---------|---------|
| GPU | NVIDIA GTX 1080 Ti | NVIDIA RTX 5070 Ti |
| GPU 記憶體 | 8 GB | 16 GB |
| RAM | 16 GB | 64 GB |
| CPU | 4 核心 | AMD Ryzen 7 7700 |
| 儲存空間 | 20 GB | 50 GB |

### 軟體需求

- **Python**: 3.9 - 3.11
- **CUDA**: 11.8 或 12.x
- **cuDNN**: 8.x
- **作業系統**: Windows 10/11, Linux

---

## 安裝步驟

### 1. 建立虛擬環境

```powershell
# 使用 Conda（推薦）
conda create -n whisper_diarization python=3.10
conda activate whisper_diarization

# 或使用 venv
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. 安裝 PyTorch（CUDA 版本）

```powershell
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安裝其他依賴

```powershell
pip install -r requirements.txt
```

### 4. 安裝 FFmpeg（音訊處理必需）

```powershell
# 使用 Chocolatey
choco install ffmpeg

# 或使用 Conda
conda install -c conda-forge ffmpeg
```

### 5. 驗證安裝

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 資料準備

### 資料夾結構

請確保您的資料夾結構如下：

```
D:\自閉症\data\
├── Fine_tune\                    # 訓練資料
│   ├── 20250210-19683409吳苡恩\
│   │   ├── 20250210-19683409吳苡恩.mp3    # 音檔
│   │   └── Aegisub.srt                      # 標註
│   ├── 20250210-19683438吳忻芸\
│   │   ├── 20250210-19683438吳忻芸.mp3
│   │   └── Aegisub.srt
│   └── ...
└── ASD\                          # 測試資料
    └── 20250106\
        └── 20250106-蔡oo\
            ├── 20250106-蔡oo.mp3
            └── GroundTruth.srt
```

### SRT 標註格式

標註檔案需遵循以下格式：

```srt
1
00:00:02,475 --> 00:00:12,425
測試者: 好啊，老師這邊我們要玩拼圖喔

2
00:00:15,550 --> 00:00:17,325
測試者: 你是姐姐還是妹妹?

3
00:00:18,050 --> 00:00:18,750
小孩: 姐姐
```

#### 語者標籤對應

| 標註文字 | 對應類別 |
|---------|---------|
| `小孩:` `兒童:` `child:` | Child (0) |
| `測試者:` `大人:` `成人:` `adult:` | Adult (1) |
| 同時出現多個語者 | 依 `dataset` 策略（可視為 overlap 再映射） |
| 無標註區間 | Silence (2) |

---

## 快速開始

**完整 Pipeline 說明（步驟、config、`pipeline` 與 `Fine_tune_LSTM` 的對照）請見：** [**README_PIPELINE.md**](README_PIPELINE.md)

### 方法一：完整 Pipeline（推薦）

```powershell
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper
python main.py pipeline --model-size medium --epochs 15
```

內部為 **Step 1 訓練 → Step 2 評估 → Step 3 產生 `outputs/ANALYSIS_REPORT.md`**，與 `Fine_tune_LSTM/main.py` 的 `pipeline` **步驟與參數語意一致**（命令列未指定 `--epochs` 時，訓練週期以 **`config.py` 的 `epochs`** 為準；僅下游為 **Conv1D** 而非 **Bi-LSTM**）。

### 方法二：分步執行

```powershell
python main.py train --model-size medium --epochs 15
python main.py evaluate --model-path models/whisper_medium_diarization_best.pt
python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path path/to/audio.mp3
# 可選：--srt-path ref.srt 或 --rttm-path ref.rttm（兩者皆給時以 RTTM 為準，與 Fine_tune_LSTM 相同）
```

---

## 詳細使用說明

### 訓練模型

```powershell
python main.py train [OPTIONS]

選項：
  --model-size    Whisper 模型大小 [tiny/base/small/medium/large]，預設：medium
  --epochs        訓練週期數；**未指定時沿用 `config.py` 的 `epochs`**（與 `Fine_tune_LSTM/main.py` 相同）
  --batch-size    批次大小，預設：8
  --lr            學習率，預設：5e-4
  --window-size   音訊窗口大小（秒），預設：20.0
```

#### 範例

```powershell
# 使用 medium 模型訓練 15 個 epochs
python main.py train --model-size medium --epochs 15

# 使用 large 模型，較小的批次大小
python main.py train --model-size large --epochs 15 --batch-size 4

# 快速測試（使用 tiny 模型）
python main.py train --model-size tiny --epochs 3
```

### 評估模型

```powershell
python main.py evaluate [OPTIONS]

選項：
  --model-path    模型權重路徑（必填）
  --model-size    Whisper 模型大小，預設：medium
```

#### 範例

```powershell
python main.py evaluate --model-path models/whisper_medium_diarization_best.pt
```

### 推論單一音檔

```powershell
python main.py infer [OPTIONS]

選項：
  --model-path    模型權重路徑（必填）
  --audio-path    音訊檔案路徑（必填）
  --srt-path      SRT 標註路徑（可選，用於評估）
  --rttm-path     RTTM 標註路徑（可選；與 SRT 皆提供時優先 RTTM）
  --output-srt    輸出 SRT 路徑
  --output-dir    輸出目錄
  --model-size    Whisper 模型大小，預設：medium
  --use-median    啟用中值濾波後處理
  --no-median     停用中值濾波後處理
  --median-size   中值濾波器核大小（必須是奇數），預設：5
```

#### 範例

```powershell
# 基本推論
python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path audio.mp3

# 推論並評估（提供標註檔）
python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path audio.mp3 --srt-path annotation.srt

# 指定輸出路徑
python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path audio.mp3 --output-srt result.srt

# 推論時啟用 Median Filter
python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path audio.mp3 --use-median
```

### 比較多個模型

```powershell
python main.py compare --model-sizes tiny small medium
```

---

## 🔧 Median Filter 後處理

中值濾波器（Median Filter）可用於平滑推論結果，減少碎片化預測。此功能僅在評估與推論階段生效，不影響訓練。

### 基本原理

- **核大小**：預設為 5，對應 5 × 20ms = 100ms 的平滑窗口
- **作用**：消除短時間的雜訊預測，使語者切換更加平滑

### 命令列參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--use-median` | 停用 | 啟用中值濾波後處理 |
| `--no-median` | 啟用 | 停用中值濾波後處理 |
| `--median-size` | 5 | 中值濾波器核大小（必須是奇數）|

### 使用範例

```powershell
# 評估時啟用 Median Filter
python main.py evaluate --model-path models/whisper_medium_diarization_best.pt --use-median

# 使用自訂核大小（核大小 7，對應 140ms）
python main.py evaluate --model-path models/whisper_medium_diarization_best.pt --use-median --median-size 7

# 推論時啟用 Median Filter
python main.py infer --model-path models/whisper_medium_diarization_best.pt --audio-path audio.mp3 --use-median

# Pipeline 中啟用 Median Filter
python main.py pipeline --model-size medium --epochs 15 --use-median --median-size 5
```

### 注意事項

- 核大小必須為**奇數**（如 3, 5, 7, 9）
- 較大的核大小會更強烈地平滑結果，但可能丟失短時語音事件
- 建議先不使用 Median Filter 評估基準效能，再啟用後比較改善程度

---

## 模型架構

```
                    ┌─────────────────────┐
                    │   Raw Audio Input   │
                    │    (20s window)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Mel Spectrogram  │
                    │     Extraction      │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │              Whisper Encoder                 │
        │            (Frozen Weights)                  │
        │                                              │
        │  ┌─────────┐ ┌─────────┐     ┌─────────┐   │
        │  │ Layer 0 │ │ Layer 1 │ ... │ Layer N │   │
        │  └────┬────┘ └────┬────┘     └────┬────┘   │
        └───────┼───────────┼───────────────┼────────┘
                │           │               │
                └───────────┼───────────────┘
                            │
                ┌───────────▼───────────┐
                │   Weighted Average    │
                │  (Learnable Weights)  │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │      Conv Head        │
                │  (3 × Conv1D + ReLU   │
                │   + Dropout 0.2)      │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │      Classifier       │
                │    (Conv1D → 3)       │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │    Frame-level        │
                │    Predictions        │
                │ (Child / Adult /      │
                │      Silence)         │
                └───────────────────────┘
```

### 參數量統計

| 模型 | 總參數 | 可訓練參數 | GPU 記憶體需求 |
|------|--------|-----------|---------------|
| Whisper-tiny | ~39M | ~0.5M | ~4 GB |
| Whisper-base | ~74M | ~0.7M | ~6 GB |
| Whisper-small | ~244M | ~1.2M | ~8 GB |
| Whisper-medium | ~769M | ~2.0M | ~12 GB |
| Whisper-large | ~1550M | ~3.0M | ~16 GB |

---

## RTX 5070 Ti 效能優化

針對 NVIDIA RTX 5070 Ti 進行的專項優化：

### 1. 混合精度訓練 (AMP)

```python
# 已在 config.py 中啟用
use_amp: bool = True
```

AMP 可將訓練速度提升 **1.5-2x**，同時減少 GPU 記憶體使用約 **30-40%**。

### 2. cuDNN Benchmark

```python
# 已在 config.py 中啟用
cudnn_benchmark: bool = True
```

對於固定大小的輸入，cuDNN benchmark 可自動選擇最佳算法。

### 3. DataLoader 優化

```python
# Linux 可視 CPU 調整；Windows 上本專案 config 會將 num_workers 設為 0 以避免預載入 Dataset 多進程 pickle 失敗
num_workers: int = 8      # Linux 建議；Windows 實際為 0
pin_memory: bool = True   # 啟用記憶體鎖定
```

### 4. 批次大小建議

| 模型 | 建議批次大小 | 預估訓練時間/epoch |
|------|-------------|-------------------|
| tiny | 16 | ~2 分鐘 |
| base | 16 | ~3 分鐘 |
| small | 12 | ~5 分鐘 |
| medium | 8 | ~10 分鐘 |
| large | 4 | ~15 分鐘 |

### 5. 進階優化選項

```powershell
# 如果遇到記憶體不足，可降低批次大小
python main.py train --model-size medium --batch-size 4

# 使用梯度累積模擬大批次
# 在 config.py 中設定：
gradient_accumulation_steps: int = 2
```

---

## 輸出檔案說明

訓練完成後，會產生以下檔案：

```
Fine_tune_paper/
├── models/
│   ├── whisper_medium_diarization_best.pt   # 最佳模型權重
│   └── checkpoint_latest.pt                  # 最新檢查點
├── logs/
│   ├── training_history.json                 # 訓練歷史記錄
│   └── training_curves.png                   # 訓練曲線圖
├── outputs/
│   ├── confusion_matrix.png                  # 混淆矩陣
│   ├── gantt_chart.png                       # 甘特圖
│   ├── gantt_comparison.png                  # 多窗口比較圖
│   ├── evaluation_results.json               # 評估結果
│   └── ANALYSIS_REPORT.md                    # 分析報告
```

---

## 常見問題

### Q1: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解決方案**：
1. 減少批次大小：`--batch-size 4`
2. 使用較小的模型：`--model-size small`
3. 確保已啟用 AMP：`use_amp: bool = True`

### Q2: 找不到音訊檔案

```
FileNotFoundError: 找不到音訊檔案
```

**解決方案**：
1. 確認資料夾路徑正確
2. 確認音訊檔案格式為 `.mp3`、`.wav` 或 `.flac`
3. 檢查 `config.py` 中的路徑設定

### Q3: SRT 解析錯誤

```
ValueError: 無法解析時間格式
```

**解決方案**：
1. 確認 SRT 檔案使用 UTF-8 編碼
2. 檢查時間格式是否正確（`HH:MM:SS,mmm`）
3. 確認語者標籤格式（`小孩:` 或 `測試者:`）

### Q4: 模型載入失敗

```
RuntimeError: Error(s) in loading state_dict
```

**解決方案**：
1. 確認 `--model-size` 與訓練時相同
2. 檢查模型檔案是否完整
3. 嘗試重新訓練模型

### Q5: 訓練速度太慢

**解決方案**：
1. 確認 CUDA 正確安裝：`python -c "import torch; print(torch.cuda.is_available())"`
2. 確認使用 GPU：觀察 `nvidia-smi` 輸出
3. 增加 `num_workers`
4. 確認 `use_amp` 已啟用

---

## 引用

如果您使用本專案，請引用原始論文：

```bibtex
@inproceedings{xu2024exploring,
  title={Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions},
  author={Xu, Anfeng and Huang, Kevin and Feng, Tiantian and Shen, Lue and Tager-Flusberg, Helen and Narayanan, Shrikanth},
  booktitle={Interspeech},
  year={2024}
}
```

---

## 授權

本專案僅供學術研究使用。

---

## 聯絡方式

如有問題，請透過 GitHub Issues 提出。
