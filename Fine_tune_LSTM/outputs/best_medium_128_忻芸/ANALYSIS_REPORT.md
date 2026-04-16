# 語者分離模型實驗報告 (Whisper + Bi-LSTM)

**生成時間**: 2026-03-29 22:19:11

---

## 1. 實驗設置摘要

### 1.1 模型配置
| 參數 | 值 |
|------|------|
| Whisper 模型大小 | medium |
| 音訊窗口大小 | 20.0 秒 |
| Step Size | 20.0 ms |
| 批次大小 | 16 |
| 學習率 | 0.0005 |
| 權重衰減 | 0.001 |
| 最大訓練 Epochs | 30 |
| **分類類別數** | **3 (Child, Adult, Silence)** |

### 1.2 Bi-LSTM 架構設定
| 參數 | 值 |
|------|------|
| LSTM 隱藏層維度（單向） | 128 |
| LSTM 隱藏層維度（雙向拼接） | 256 |
| LSTM 層數 | 2 |
| Dropout | 0.2 |
| **中間層架構** | **2 層 Bi-LSTM（替換原本的 3 層 Conv1D）** |

### 1.3 資料集
- **訓練資料**: D:\自閉症\data\Fine_tune
- **測試資料**: 由 train_data_dir 固定比例隨機切分 (train/val/test = 2/1/1)
- **驗證集比例**: 25%

### 1.4 硬體配置
- **GPU**: NVIDIA RTX 5070 Ti
- **混合精度訓練 (AMP)**: 啟用
- **cuDNN Benchmark**: 啟用

---

## 2. 訓練結果分析

### 2.1 訓練摘要
| 指標 | 值 |
|------|------|
| 總訓練 Epochs | 11 |
| 最佳 Epoch | 6 |
| 最佳驗證損失 | 0.3867 |
| 最佳驗證準確率 | 85.95% |
| 總訓練時間 | 2.7 分鐘 |
| 平均每 Epoch 時間 | 14.7 秒 |

### 2.2 Loss 收斂情形
訓練曲線圖已儲存至: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\logs/training_curves.png`

---

## 3. 評估結果

### 3.1 Diarization Error Rate (DER)

| 指標 | 值 |
|------|------|
| **DER** | **42.62%** |
| False Alarm Rate | 27.02% |
| Missed Detection Rate | 13.68% |
| Speaker Confusion Rate | 1.92% |

### 3.2 架構比較

| 架構 | 描述 |
|------|------|
| 原始 (Conv1D) | 3 層 1D 卷積 + ReLU + Dropout |
| **Bi-LSTM** | **2 層雙向 LSTM + LayerNorm + Dropout** |

---

## 4. Bi-LSTM vs Conv1D 分析

### 4.1 理論優勢

**Bi-LSTM 的優勢**:
1. **長距離依賴**: LSTM 可以捕捉更長的時間依賴關係
2. **雙向上下文**: 同時利用過去和未來的資訊
3. **時序建模**: 更適合語音這種時序資料

**Conv1D 的優勢**:
1. **平行計算**: 可以完全平行化，訓練更快
2. **局部特徵**: 擅長捕捉局部模式
3. **參數效率**: 通常參數量較少

### 4.2 實驗結果對比

請比較此實驗的 DER 與之前 Conv1D 版本的結果，
以判斷哪種架構在您的資料集上表現更好。

---

## 5. 視覺化結果

以下視覺化圖表已生成：

1. **訓練曲線**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\logs/training_curves.png`
2. **混淆矩陣**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\outputs/confusion_matrix.png`
3. **甘特圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\outputs/gantt_chart.png`
4. **多窗口比較圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\outputs/gantt_comparison.png`

---

## 6. 結論

本實驗使用 Whisper-medium 作為特徵提取器，
將中間層的 3 層 Conv1D 替換為 2 層 Bi-LSTM。

### 架構摘要:
```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
2 層 Bi-LSTM (hidden_size=128, bidirectional)
    ↓
Layer Normalization + Dropout
    ↓
Conv1D Classifier (→ 3 類別)
```

---

## 附錄

### A. 類別定義 (3 分類)
- **0 (Child)**: 小孩語音
- **1 (Adult)**: 測試者/成人語音
- **2 (Silence)**: 靜音/背景噪音

### B. 評估指標說明
- **DER**: 語者分離總錯誤率
- **False Alarm**: 將非語音誤判為語音的比例
- **Missed Detection**: 將語音誤判為非語音的比例
- **Speaker Confusion**: 語音區段語者判斷錯誤的比例

