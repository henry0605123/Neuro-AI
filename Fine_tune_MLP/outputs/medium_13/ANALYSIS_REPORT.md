# 語者分離模型實驗報告 (Whisper + MLP + Median Filter)

**生成時間**: 2026-01-26 16:02:55

---

## 1. 實驗設置摘要

### 1.1 模型配置
| 參數 | 值 |
|------|------|
| Whisper 模型大小 | medium |
| 音訊窗口大小 | 20.0 秒 |
| Step Size | 20.0 ms |
| 批次大小 | 8 |
| 學習率 | 0.0005 |
| 權重衰減 | 0.0001 |
| 最大訓練 Epochs | 15 |
| **分類類別數** | **3 (Child, Adult, Silence)** |

### 1.2 MLP + Median Filter 架構設定
| 參數 | 值 |
|------|------|
| MLP 隱藏層維度 | [512, 256] |
| MLP 層數 | 2 |
| Dropout | 0.2 |
| Median Filter 核心大小 | 13 |
| **中間層架構** | **MLP + Median Filter（替換 Bi-LSTM）** |

### 1.3 資料集
- **訓練資料**: D:\自閉症\data\Fine_tune
- **測試資料**: D:\自閉症\data\TD\20250421-19752483鄭皓文\20250421-19752483鄭皓文.mp3
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
| 總訓練 Epochs | 15 |
| 最佳 Epoch | 13 |
| 最佳驗證損失 | 0.7885 |
| 最佳驗證準確率 | 71.90% |
| 總訓練時間 | 4.2 分鐘 |
| 平均每 Epoch 時間 | 16.7 秒 |

### 2.2 Loss 收斂情形
訓練曲線圖已儲存至: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP\logs/training_curves.png`

---

## 3. 評估結果

### 3.1 Diarization Error Rate (DER)

| 指標 | 值 |
|------|------|
| **DER** | **30.35%** |
| False Alarm Rate | 9.77% |
| Missed Detection Rate | 17.51% |
| Speaker Confusion Rate | 3.07% |

### 3.2 架構比較

| 架構 | 描述 |
|------|------|
| 原始 (Conv1D) | 3 層 1D 卷積 + ReLU + Dropout |
| Bi-LSTM | 2 層雙向 LSTM + LayerNorm + Dropout |
| **MLP + Median Filter** | **多層 MLP + GELU + Median Filter 後處理** |

---

## 4. MLP + Median Filter 分析

### 4.1 架構設計理念

**MLP 的優勢**:
1. **計算效率**: 完全平行化，訓練和推論速度快
2. **簡單架構**: 較少的超參數，容易調整
3. **記憶體友好**: 不需要儲存隱藏狀態

**Median Filter 的作用**:
1. **平滑預測**: 減少時間序列中的抖動
2. **去除噪音**: 過濾掉短暫的錯誤預測
3. **後處理增強**: 不影響訓練，只在推論時使用

### 4.2 與其他架構對比

| 架構 | 優點 | 缺點 |
|------|------|------|
| Conv1D | 局部特徵提取佳 | 感受野有限 |
| Bi-LSTM | 長距離依賴 | 訓練較慢 |
| **MLP + Median Filter** | **快速、平滑輸出** | **無時序建模能力** |

---

## 5. Median Filter 效果分析

Median Filter 核心大小 = 13

這意味著每個預測會考慮前後共 13 個 frame，
相當於 260.0 ms 的時間範圍。

---

## 6. 視覺化結果

以下視覺化圖表已生成：

1. **訓練曲線**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP\logs/training_curves.png`
2. **混淆矩陣**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP\outputs/confusion_matrix.png`
3. **甘特圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP\outputs/gantt_chart.png`
4. **多窗口比較圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP\outputs/gantt_comparison.png`

---

## 7. 結論

本實驗使用 Whisper-medium 作為特徵提取器，
將中間層替換為 MLP + Median Filter 架構。

### 架構摘要:
```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
MLP (隱藏層: [512, 256]) + GELU + Dropout
    ↓
Conv1D Classifier (→ 3 類別)
    ↓
Median Filter (kernel=13, 僅推論時使用)
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

### C. Median Filter 說明
Median Filter 是一種非線性濾波器，在時間序列上取滑動窗口內的中值。
這可以有效去除短暫的抖動和噪音，使預測結果更加平滑連貫。

