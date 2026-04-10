# 語者分離模型實驗報告 (Whisper + Conv1D · 論文復現)

**生成時間**: 2026-03-29 15:51:22

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

### 1.2 Conv Head 架構設定
| 參數 | 值 |
|------|------|
| Conv 通道數 | 128 |
| Dropout | 0.2 |
| **中間層** | **3 層 Conv1D + ReLU + Dropout（論文設定）** |

### 1.3 資料集
- **訓練資料**: D:\自閉症\data\Fine_tune
- **測試音檔**: 
- **驗證集比例**: 25%

### 1.4 硬體配置
- **GPU**: NVIDIA RTX 5070 Ti（或本機實際 GPU）
- **混合精度訓練 (AMP)**: 啟用
- **cuDNN Benchmark**: 啟用

---

## 2. 訓練結果分析

### 2.1 訓練摘要
| 指標 | 值 |
|------|------|
| 總訓練 Epochs | 17 |
| 最佳 Epoch | 12 |
| 最佳驗證損失 | 0.5487 |
| 最佳驗證準確率 | 78.86% |
| 總訓練時間 | 4.3 分鐘 |
| 平均每 Epoch 時間 | 15.1 秒 |

### 2.2 Loss 收斂情形
訓練曲線圖已儲存至: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\logs/training_curves.png`

---

## 3. 評估結果

### 3.1 Diarization Error Rate (DER)

| 指標 | 值 |
|------|------|
| **DER** | **20.43%** |
| False Alarm Rate | 5.30% |
| Missed Detection Rate | 12.61% |
| Speaker Confusion Rate | 2.52% |

### 3.2 與論文 Baseline 的比較

| 方法 | DER |
|------|-----|
| VBx (論文 Baseline) | ~25.8% |
| PyAnnote (論文 Baseline) | ~26.0% |
| WSP-Medium (論文) | 15.6% |
| **本實驗結果 (3 分類)** | **20.4%** |

---

## 4. 混淆矩陣與常見錯誤

混淆矩陣圖已儲存至: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs/confusion_matrix.png`

- **Child / Adult 混淆**: 語調或發聲特徵重疊時較易發生  
- **邊界誤差**: 語音起迄處 frame 標記易有偏差  
- **短語音**: 可能被誤判為 Silence  

---

## 5. 視覺化結果

1. **訓練曲線**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\logs/training_curves.png`
2. **混淆矩陣**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs/confusion_matrix.png`
3. **甘特圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs/gantt_chart.png`
4. **多窗口比較圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs/gantt_comparison.png`

---

## 6. 結論與架構摘要

本實驗復現論文方法：Whisper-medium 作為特徵提取器，下游為 **Conv1D Head**（與 Fine_tune_LSTM 專案中的 **Bi-LSTM** 替換版為不同架構選項）。

```
Whisper Encoder (凍結)
    ↓
Weighted Average Layer
    ↓
3 × Conv1D + ReLU + Dropout
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

