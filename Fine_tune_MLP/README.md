# Whisper + MLP + Median Filter 語者分離模型

> **架構變更**: 將 Bi-LSTM 替換為 MLP + Median Filter
> 
> **日期**: 2026-01-26

---

## 📋 架構說明

### 原始架構 (Bi-LSTM 版本)
```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
2 × Bi-LSTM (hidden=256) + LayerNorm + Dropout
    ↓
Conv1D Classifier (→ 3 類別)
```

### 新架構 (MLP + Median Filter 版本)
```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
MLP (512 → 256) + GELU + LayerNorm + Dropout
    ↓
Conv1D Classifier (→ 3 類別)
    ↓
Median Filter (kernel=5, 僅推論時使用)
```

---

## 🔧 架構細節

### MLP Head

| 參數 | 值 |
|------|------|
| 輸入維度 | Whisper hidden dim (768/1024/1280) |
| 隱藏層 1 | 512 |
| 隱藏層 2 | 256 |
| 激活函數 | GELU |
| 正規化 | LayerNorm |
| Dropout | 0.2 |

### Median Filter

| 參數 | 值 |
|------|------|
| 核心大小 | 5（可調整）|
| 時間範圍 | 5 × 20ms = 100ms |
| 使用時機 | 僅推論時使用 |

### 設計考量

1. **MLP 優勢**: 
   - 完全平行計算，訓練快速
   - 記憶體效率高
   - 簡單易調

2. **Median Filter 作用**:
   - 平滑預測結果
   - 去除短暫抖動
   - 減少錯誤切換

3. **為何不在訓練時使用 Median Filter**:
   - 中值濾波不可微分
   - 作為後處理，不影響梯度傳播

---

## 🚀 快速開始

### 訓練模型

```powershell
cd C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP

# 使用預設設定 (medium 模型, MLP=[512,256], Median=5)
python main.py pipeline --model-size medium --epochs 15

# 自訂 MLP 設定
python main.py pipeline --model-size medium --epochs 15 --mlp-hidden 1024,512,256 --median-size 7
```

### 評估模型

```powershell
# 使用 Median Filter（預設）
python main.py evaluate --model-path models/whisper_medium_mlp_best.pt

# 不使用 Median Filter
python main.py evaluate --model-path models/whisper_medium_mlp_best.pt --no-median
```

### 推論

```powershell
python main.py infer --model-path models/whisper_medium_mlp_best.pt --audio-path path/to/audio.mp3
```

---

## 📁 檔案結構

```
Fine_tune_MLP/
├── config.py          # 配置檔（包含 MLP 和 Median Filter 設定）
├── model.py           # MLP + Median Filter 模型定義
├── train.py           # 訓練模組
├── evaluate.py        # 評估模組（支援 Median Filter）
├── dataset.py         # 資料集模組
├── main.py            # 主程式入口
├── requirements.txt   # 依賴套件
├── README.md          # 本文件
├── models/            # 模型權重儲存
├── logs/              # 訓練日誌
└── outputs/           # 評估結果輸出
```

---

## 📊 與其他版本比較

| 架構 | 優點 | 缺點 | 預期訓練速度 |
|------|------|------|-------------|
| **Conv1D** | 局部特徵提取 | 感受野有限 | ⭐⭐⭐⭐⭐ 最快 |
| **Bi-LSTM** | 長距離依賴 | 序列計算 | ⭐⭐⭐ 較慢 |
| **MLP + Median** | 快速、平滑輸出 | 無時序建模 | ⭐⭐⭐⭐ 快 |

---

## ⚙️ 命令列參數

### MLP 專用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--mlp-hidden` | "512,256" | MLP 隱藏層維度，逗號分隔 |
| `--median-size` | 5 | Median Filter 核心大小（必須為奇數）|
| `--use-median` | True | 評估/推論時使用 Median Filter |
| `--no-median` | - | 不使用 Median Filter |

### 通用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model-size` | medium | Whisper 模型大小 |
| `--epochs` | 15 | 訓練 epochs 數 |
| `--batch-size` | 8 | 批次大小 |
| `--lr` | 5e-4 | 學習率 |
| `--window-size` | 20.0 | 音訊窗口大小（秒）|

---

## 🔬 實驗建議

### 1. 先用 tiny 模型測試
```powershell
python main.py pipeline --model-size tiny --epochs 3
```

### 2. 調整 MLP 隱藏層維度
```powershell
# 較淺的 MLP
python main.py pipeline --model-size medium --epochs 15 --mlp-hidden 256

# 較深的 MLP
python main.py pipeline --model-size medium --epochs 15 --mlp-hidden 1024,512,256
```

### 3. 調整 Median Filter 大小
```powershell
# 較小的 kernel（更少平滑）
python main.py pipeline --model-size medium --epochs 15 --median-size 3

# 較大的 kernel（更多平滑）
python main.py pipeline --model-size medium --epochs 15 --median-size 9
```

### 4. 比較有無 Median Filter 的效果
```powershell
# 評估時使用 Median Filter
python main.py evaluate --model-path models/whisper_medium_mlp_best.pt --use-median

# 評估時不使用 Median Filter
python main.py evaluate --model-path models/whisper_medium_mlp_best.pt --no-median
```

---

## 📈 預估效能

| 模型 | 可訓練參數 | 每 Epoch 時間 (RTX 5070 Ti) |
|------|-----------|---------------------------|
| Whisper-tiny + MLP | ~1.5M | ~2 分鐘 |
| Whisper-small + MLP | ~2M | ~4 分鐘 |
| Whisper-medium + MLP | ~3M | ~8 分鐘 |
| Whisper-large + MLP | ~5M | ~12 分鐘 |

**注意**: MLP 版本通常比 Bi-LSTM 快 20-30%，因為完全可平行化。

---

## 📝 輸出說明

訓練完成後，以下檔案會產生：

- `models/whisper_{size}_mlp_best.pt` - 最佳模型權重
- `logs/training_curves.png` - 訓練曲線
- `outputs/confusion_matrix.png` - 混淆矩陣
- `outputs/gantt_chart.png` - 甘特圖
- `outputs/ANALYSIS_REPORT.md` - 分析報告

---

## 🎯 Median Filter 原理

Median Filter 是一種非線性濾波器，對於每個時間點，取其鄰近時間點的中值作為輸出。

例如，kernel_size = 5 時：
```
輸入:  [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]
輸出:  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
                 ↑               ↑
          短暫的 1 被平滑      短暫的 0 被平滑
```

這有助於：
- 去除預測中的短暫錯誤
- 使語者邊界更加清晰
- 減少不必要的語者切換

---

## 📚 參考文獻

- Xu, A., et al. (2024). "Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions." Interspeech 2024.
- 中值濾波在語音處理中的應用

---

## 📂 與其他版本的比較實驗

完成訓練後，建議比較三種架構：

1. **Fine_tune_paper** (Conv1D)
2. **Fine_tune_LSTM** (Bi-LSTM)
3. **Fine_tune_MLP** (MLP + Median Filter)

比較指標：
- DER (Diarization Error Rate)
- Speaker Confusion Rate
- 訓練時間
- 推論速度
