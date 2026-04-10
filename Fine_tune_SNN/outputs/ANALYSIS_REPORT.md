# 語者分離模型實驗報告 (Whisper + ReLU Student，KD)

**生成時間**: 2026-04-09 18:37:50

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
| 最大訓練 Epochs | 30 |
| **分類類別數** | **3 (Child, Adult, Silence)** |

### 1.2 知識蒸餾與 Student 頭
| 參數 | 值 |
|------|------|
| Teacher 權重 | `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\outputs\medium_128_filter\whisper_medium_bilstm_best.pt` |
| Teacher LSTM（載入時依 checkpoint 對齊） | hidden=128（單向）, layers=2 |
| Student 頭類型 | **conv** |
| Student 頭結構 | 3 層 Conv1d (k=1) + ReLU + Dropout（論文式） |
| KD 溫度 T | 4.0 |
| KD 混合係數 α（CE 權重） | 0.5 |
| KD 損失權重 (1-α) | 0.5000 |

### 1.3 資料集
- **訓練資料**: D:\自閉症\data\Fine_tune
- **測試資料**: D:\自閉症\data\TD\20250421-19752483鄭皓文\20250421-19752483鄭皓文.mp3
- **驗證集比例**: 25%

### 1.4 硬體配置
- **混合精度訓練 (AMP)**: 啟用
- **cuDNN Benchmark**: 啟用

---

## 2. 訓練結果分析

### 2.1 訓練摘要
| 指標 | 值 |
|------|------|
| 總訓練 Epochs | 19 |
| 最佳 Epoch | 14 |
| 最佳驗證損失 | 0.4361 |
| 最佳驗證準確率 | 84.43% |
| 總訓練時間 | 5.8 分鐘 |
| 平均每 Epoch 時間 | 18.3 秒 |

### 2.2 Loss 收斂情形
訓練曲線圖已儲存至: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\logs/training_curves.png`

---

## 3. 評估結果

### 3.1 Diarization Error Rate (DER)

| 指標 | 值 |
|------|------|
| **DER** | **29.55%** |
| False Alarm Rate | 12.55% |
| Missed Detection Rate | 13.92% |
| Speaker Confusion Rate | 3.08% |

### 3.2 架構說明

| 項目 | 描述 |
|------|------|
| **Teacher** | Bi-LSTM（凍結，僅提供軟標籤） |
| **Student** | ReLU 前饋頭（蒸餾目標） |


### 3.3 混合 SNN 頭評估（Pipeline Step 3）

Whisper Encoder + Weighted Average 仍為 **ANN**；蒸餾後之 **Conv 頭與最後 Conv1d 分類層** 以權重複製建立獨立 **snntorch Leaky LIF** 前向鏈，經校準縮放對齊 ANN 特徵域（SNN 支路與 Student 的 ANN 頭 **不共用模組**，可僅微調 SNN 頭而不改壞已蒸餾之 ANN）。

| 指標 | 值 |
|------|------|
| **DER（SNN 頭）** | **100.00%** |
| False Alarm Rate | 0.00% |
| Missed Detection Rate | 100.00% |
| Speaker Confusion Rate | 0.00% |

額外圖表（若已寫入）目錄：`C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\outputs/snn_eval/`


**SNN 頭微調**（轉換與校準後）：5 epochs，lr=0.0001；僅更新 SNN 內 Conv/LIF 路徑，Whisper 與 WeightedAvg 凍結。


---

## 4. 視覺化結果

以下視覺化圖表已生成：

1. **訓練曲線**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\logs/training_curves.png`
2. **混淆矩陣**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\outputs/confusion_matrix.png`
3. **甘特圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\outputs/gantt_chart.png`
4. **多窗口比較圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\outputs/gantt_comparison.png`

---

## 5. 結論

本實驗以 Whisper-medium 為特徵提取器，使用 **Bi-LSTM Teacher** 對 **ReLU Student** 進行知識蒸餾。

### 架構摘要:
```
Whisper Encoder (凍結，與 Teacher 對齊)
    ↓
Weighted Average Layer (凍結，自 Teacher 複製)
    ↓
ReLU Student 頭 (conv)
    ↓
Conv1d Classifier (→ 3 類別)
```

Pipeline 另可將 **Conv 頭** 轉為 **SNN（LIF）** 並評估混合模型（見上節 3.3，若已執行）。

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

