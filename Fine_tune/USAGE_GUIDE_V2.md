# 🚀 Pyannote Fine-tune V2 - 完整使用指南

## 📋 目錄

1. [快速開始](#快速開始)
2. [系統需求](#系統需求)
3. [資料準備](#資料準備)
4. [訓練流程](#訓練流程)
5. [模型使用](#模型使用)
6. [參數調整](#參數調整)
7. [疑難排解](#疑難排解)
8. [效能評估](#效能評估)

---

## 快速開始

### 5 分鐘快速上手

```bash
# 1. 進入專案目錄
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune

# 2. 啟動環境
conda activate ASD

# 3. 準備資料（將音檔切成片段）
python 01_prepare_dataset_v2.py

# 4. 訓練模型
python 02_train_model_v3.py

# 5. 使用模型進行轉錄
python 03_transcribe_v2.py --audio "你的音檔.mp3" --output "輸出.srt" --skip-alignment
```

---

## 系統需求

### 硬體需求

| 項目 | 最低需求 | 建議配置 |
|------|----------|----------|
| GPU | GTX 1660 (6GB) | RTX 3080+ (10GB+) |
| RAM | 16 GB | 32 GB |
| 儲存空間 | 10 GB | 50 GB |
| CPU | 4 核心 | 8 核心 |

### 軟體需求

- Python 3.10+
- PyTorch 2.0+ (支援 CUDA)
- pyannote.audio 3.0+
- whisperx 3.0+

### 安裝相依套件

```bash
conda activate ASD

# 核心套件
pip install pyannote.audio torch torchaudio
pip install whisperx

# 輔助套件
pip install tqdm numpy pandas
```

---

## 資料準備

### 資料目錄結構

```
C:\自閉症\data\Fine_tune\
├── 20250210-19683409吳苡恩/
│   ├── 20250210-19683409吳苡恩.mp3  # 音訊檔
│   └── Aegisub.srt                   # 標註檔
├── 20250210-19683438吳忻芸/
│   ├── 20250210-19683438吳忻芸.mp3
│   └── Aegisub.srt
└── ...
```

### SRT 標註格式

標註檔案應包含說話人標籤：

```srt
1
00:00:03,450 --> 00:00:04,950
小孩: 你好

2
00:00:05,500 --> 00:00:06,700
測試者: 你好啊

3
00:00:07,350 --> 00:00:09,450
小孩: 我叫小明
```

支援的說話人標籤：
- `測試者` → SPEAKER_00
- `小孩` → SPEAKER_01
- `Unknown` → SPEAKER_02

### 執行資料準備

```bash
python 01_prepare_dataset_v2.py
```

**輸出說明：**

```
[1/4] 處理: 20250210-19683409吳苡恩
  音訊檔: 20250210-19683409吳苡恩.mp3
  標註檔: Aegisub.srt
  時長: 1814.78 秒 (30.2 分鐘)
  段落數: 448
  正在切片...
  切片完成: 312 個片段
    含小孩語音: 187 個 (59.9%)
```

**產生的檔案：**

- `data_v2/segments_v2.json` - 所有片段資訊
- `data_v2/class_weights.json` - 自動計算的類別權重
- `data_v2/*.rttm` - RTTM 格式標註

---

## 訓練流程

### 基本訓練

```bash
python 02_train_model_v3.py
```

### 從中斷處繼續

```bash
python 02_train_model_v3.py --resume
```

### 除錯模式（使用少量資料）

```bash
python 02_train_model_v3.py --debug
```

### 訓練輸出說明

```
Epoch 5/30:
  Train Loss: 0.2341
  Val Loss: 0.2156
  測試者 - P: 85.32%, R: 78.45%, F1: 81.74%
  小孩 - P: 72.18%, R: 65.23%, F1: 68.53%
  ✓ 儲存最佳模型 (Val Loss: 0.2156)
```

**指標說明：**

| 指標 | 說明 | 目標 |
|------|------|------|
| P (Precision) | 預測為該說話人的準確率 | > 70% |
| R (Recall) | 該說話人被正確識別的比例 | > 60% |
| F1 | Precision 和 Recall 的調和平均 | > 65% |

### 模型儲存位置

```
models/finetuned_segmentation_v2/
├── best_model.pt    # 最佳模型（驗證損失最低）
├── last.pt          # 最終模型
├── checkpoint.pt    # 檢查點（可續訓）
└── history.json     # 訓練歷史
```

---

## 模型使用

### 基本轉錄

```bash
python 03_transcribe_v2.py \
    --audio "C:\path\to\audio.mp3" \
    --output "output.srt"
```

### 跳過時間對齊（避免網路問題）

```bash
python 03_transcribe_v2.py \
    --audio "audio.mp3" \
    --output "output.srt" \
    --skip-alignment
```

### 使用原始模型（比較用）

```bash
python 03_transcribe_v2.py \
    --audio "audio.mp3" \
    --use-original
```

### 同時輸出兩種模型結果

```bash
python 03_transcribe_v2.py \
    --audio "audio.mp3" \
    --compare \
    --skip-alignment
```

產生：
- `audio_original.srt` - 原始模型結果
- `audio_finetuned.srt` - 微調模型結果

### 指定模型路徑

```bash
python 03_transcribe_v2.py \
    --audio "audio.mp3" \
    --model "models/my_model/best_model.pt"
```

---

## 參數調整

### 配置檔案

所有參數都在 `config_v2.py` 中設定：

### 資料切片參數

```python
# 片段長度（秒）- 建議 3-10 秒
SEGMENT_DURATION = 5.0

# 重疊比例 - 0.5 表示 50% 重疊
# 較高的重疊可學習更多邊界資訊，但會增加資料量
SEGMENT_OVERLAP = 0.5

# 最小語音比例 - 過濾純靜音片段
MIN_SPEECH_RATIO = 0.1

# 是否要求片段包含小孩語音
# 設為 True 可提高小孩 Recall，但可能降低測試者準確度
REQUIRE_CHILD_SPEECH = True

# 小孩語音最小比例
MIN_CHILD_RATIO = 0.05
```

### 類別加權參數

```python
# 說話人權重 - 小孩權重更高因為樣本較少
SPEAKER_WEIGHTS = {
    "SPEAKER_00": 1.0,   # 測試者
    "SPEAKER_01": 3.0,   # 小孩（3 倍權重）
    "SPEAKER_02": 0.5,   # Unknown
}

# 自動計算權重（根據實際分佈）
AUTO_COMPUTE_WEIGHTS = True
```

### 訓練參數

```python
# 批次大小 - 根據 GPU 記憶體調整
BATCH_SIZE = 32  # RTX 3080: 32-64, RTX 2060: 16-24

# 訓練輪數
EPOCHS = 30

# 學習率 - 過高會震盪，過低收斂慢
LEARNING_RATE = 5e-5  # 建議範圍: 1e-5 ~ 1e-4

# 凍結層比例 - 資料越少應凍結越多
FREEZE_RATIO = 0.5  # 0.3-0.7

# 早停耐心值
EARLY_STOPPING_PATIENCE = 10
```

### 資料增強參數

```python
# 是否啟用增強
USE_AUGMENTATION = True

# 時間遮罩 - 隨機遮蓋部分時間
TIME_MASK_PROB = 0.3
TIME_MASK_MAX_LENGTH = 0.5  # 秒

# 音量變化
VOLUME_CHANGE_PROB = 0.3
VOLUME_CHANGE_RANGE = (0.8, 1.2)

# 加入噪音
ADD_NOISE_PROB = 0.2
NOISE_SNR_RANGE = (15, 30)  # dB
```

---

## 疑難排解

### 常見問題

#### 1. CUDA Out of Memory

**症狀：**
```
RuntimeError: CUDA out of memory
```

**解決方案：**
```python
# 在 config_v2.py 中降低批次大小
BATCH_SIZE = 16  # 或更低
```

#### 2. 小孩 Recall 為 0%

**原因：** 小孩語音樣本太少，模型傾向全部預測為測試者

**解決方案：**
1. 確保 `REQUIRE_CHILD_SPEECH = True`
2. 增加小孩權重：
   ```python
   SPEAKER_WEIGHTS = {
       "SPEAKER_01": 5.0,  # 提高到 5 倍
   }
   ```
3. 使用 Focal Loss（已內建）

#### 3. 訓練損失不下降

**原因：** 學習率可能太高或太低

**解決方案：**
```python
# 嘗試不同學習率
LEARNING_RATE = 1e-4  # 或 1e-5
```

#### 4. 網路下載失敗

**症狀：** 下載 Hugging Face 模型時中斷

**解決方案：**
```bash
# 使用 --skip-alignment 跳過對齊步驟
python 03_transcribe_v2.py --skip-alignment
```

#### 5. Tensor Size Mismatch

**症狀：**
```
RuntimeError: Sizes of tensors must match
```

**解決方案：** 已在程式碼中自動處理，會降低批次大小重試

### 日誌檢查

訓練歷史儲存在：
```
models/finetuned_segmentation_v2/history.json
```

可用以下程式碼視覺化：
```python
import json
import matplotlib.pyplot as plt

with open("history.json") as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['child_recall'], label='Child Recall')
plt.plot(history['tester_recall'], label='Tester Recall')
plt.title('Recall')
plt.legend()

plt.show()
```

---

## 效能評估

### 評估指標

#### DER (Diarization Error Rate)

```
DER = (Miss + False Alarm + Confusion) / Total Duration
```

- **Miss**：漏檢（說話人在說話但沒被偵測到）
- **False Alarm**：誤報（沒人說話但偵測到）
- **Confusion**：混淆（說話人標籤錯誤）

**目標：DER < 20%**

#### 說話人指標

| 說話人 | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| 測試者 | > 80% | > 70% | > 75% |
| 小孩 | > 60% | > 50% | > 55% |

### 預期改善

使用 V2 優化後，預期：

| 指標 | V1 (舊版) | V2 (優化版) | 改善 |
|------|-----------|-------------|------|
| DER | ~56% | ~30% | -26% |
| 小孩 Recall | 0% | > 50% | 顯著提升 |
| 測試者 Recall | 60% | > 70% | +10% |

### 評估方法

使用提供的評估腳本：
```bash
python evaluate_model.py \
    --prediction "output.srt" \
    --ground-truth "GTruth.srt"
```

---

## 檔案結構總覽

```
Fine_tune/
├── 主要腳本
│   ├── 01_prepare_dataset_v2.py  # 資料準備（V2）
│   ├── 02_train_model_v3.py      # 訓練腳本（V3）
│   └── 03_transcribe_v2.py       # 轉錄腳本（V2）
│
├── 配置檔
│   ├── config_v2.py              # V2 配置
│   └── config.py                 # V1 配置（舊版）
│
├── 工具模組
│   └── utils/
│       ├── srt_parser.py
│       ├── rttm_converter.py
│       └── audio_utils.py
│
├── 資料目錄
│   ├── data/                     # V1 資料
│   └── data_v2/                  # V2 資料
│       ├── segments_v2.json
│       ├── class_weights.json
│       └── *.rttm
│
├── 模型目錄
│   └── models/
│       ├── finetuned_segmentation/     # V1 模型
│       └── finetuned_segmentation_v2/  # V2 模型
│           ├── best_model.pt
│           ├── last.pt
│           └── history.json
│
└── 文檔
    ├── USAGE_GUIDE_V2.md         # 本文件
    └── OPTIMIZATION_REPORT.md    # 優化說明
```

---

## 進階使用

### 使用 TensorBoard 監控訓練

```bash
tensorboard --logdir=logs/
```

### 多 GPU 訓練

目前尚未支援，但可手動修改：
```python
model = nn.DataParallel(model)
```

### 自訂損失函數

在 `02_train_model_v3.py` 中修改 `FocalLoss` 類別的參數：
```python
self.criterion = FocalLoss(
    alpha=0.25,  # 正樣本權重
    gamma=2.0,   # 聚焦參數（越大越聚焦於困難樣本）
    class_weights=self.class_weights
)
```

---

## 常見問答

**Q: 需要多少訓練資料？**
A: 建議至少 4 個完整標註的音訊檔（每個 > 20 分鐘），總計 2 小時以上。

**Q: 訓練需要多長時間？**
A: 使用 RTX 3080，約 2000 個片段，30 epochs 約需 1-2 小時。

**Q: 可以用於其他語言嗎？**
A: 可以，但需要相應語言的標註資料。語者分離本身與語言無關。

**Q: 如何提高小孩的識別率？**
A: 
1. 增加小孩語音的權重
2. 確保資料中有足夠的小孩語音
3. 使用 `REQUIRE_CHILD_SPEECH = True`

---

## 版本歷史

| 版本 | 日期 | 主要改進 |
|------|------|----------|
| V1 | 2026-01-10 | 初始版本 |
| V2 | 2026-01-12 | 資料切片、類別加權、Focal Loss |

---

## 聯絡與支援

如有問題，請檢查：
1. 本使用指南
2. `OPTIMIZATION_REPORT.md` 優化說明
3. 程式碼中的註解

---

*最後更新：2026-01-12*
