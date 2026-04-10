# 語者分離模型實驗報告 (3 分類任務)

**生成時間**: 2026-01-19 21:05:40

---

## 1. 實驗設置摘要

### 1.1 模型配置
| 參數 | 值 |
|------|------|
| Whisper 模型大小 | small |
| 音訊窗口大小 | 20.0 秒 |
| Step Size | 20.0 ms |
| 批次大小 | 8 |
| 學習率 | 0.0005 |
| 權重衰減 | 0.0001 |
| 最大訓練 Epochs | 30 |
| **分類類別數** | **3 (Child, Adult, Silence)** |

### 1.2 資料集
- **訓練資料**: D:\自閉症\data\Fine_tune
- **測試資料**: D:\自閉症\data\TD\20250421-19752483鄭皓文\20250421-19752483鄭皓文.mp3
- **驗證集比例**: 25%

### 1.3 硬體配置
- **GPU**: NVIDIA RTX 5070 Ti
- **混合精度訓練 (AMP)**: 啟用
- **cuDNN Benchmark**: 啟用

---

## 2. 訓練結果分析

### 2.1 訓練摘要
| 指標 | 值 |
|------|------|
| 總訓練 Epochs | 10 |
| 最佳 Epoch | 5 |
| 最佳驗證損失 | 0.7339 |
| 最佳驗證準確率 | 69.39% |
| 總訓練時間 | 1.6 分鐘 |
| 平均每 Epoch 時間 | 9.7 秒 |

### 2.2 Loss 收斂情形
訓練曲線圖已儲存至: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\logs/training_curves.png`

---

## 3. 評估結果

### 3.1 Diarization Error Rate (DER)

| 指標 | 值 |
|------|------|
| **DER** | **36.06%** |
| False Alarm Rate | 9.07% |
| Missed Detection Rate | 21.18% |
| Speaker Confusion Rate | 5.80% |

### 3.2 與論文 Baseline 的比較

| 方法 | DER |
|------|------|
| VBx (論文 Baseline) | ~25.8% |
| PyAnnote (論文 Baseline) | ~26.0% |
| WSP-Medium (論文) | 15.6% |
| **本實驗結果 (3 分類)** | **36.1%** |

### 3.3 分類準確率

混淆矩陣圖已儲存至: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs/confusion_matrix.png`

---

## 4. 混淆矩陣分析

### 4.1 各類別表現

根據混淆矩陣分析：

- **Child (小孩)**: 分析小孩語音的識別準確率
- **Adult (測試者)**: 分析成人語音的識別準確率  
- **Silence (靜音)**: 靜音區域的識別通常較準確

### 4.2 常見錯誤模式

1. **小孩與成人混淆**: 可能的原因包括語調相似、發聲特徵重疊
2. **邊界誤差**: 語音開始/結束處的判定容易產生誤差
3. **短暫語音遺漏**: 較短的發言可能被誤判為靜音

---

## 5. 視覺化結果

以下視覺化圖表已生成：

1. **訓練曲線**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\logs/training_curves.png`
   - Loss 曲線（訓練 vs 驗證）
   - 準確率曲線
   - 學習率變化

2. **混淆矩陣**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs/confusion_matrix.png`

3. **甘特圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs/gantt_chart.png`
   - 預測結果 vs 真實標籤的時間軸對比

4. **多窗口比較圖**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs/gantt_comparison.png`

---

## 6. 結論與建議

### 6.1 實驗結論

本實驗成功復現了論文《Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions》的主要方法，使用 Whisper-small 作為 speech foundation model 進行 child-adult speaker diarization。

**本實驗採用 3 分類任務（移除 Overlap）**，因為訓練資料集中沒有重疊語音的標註。

### 6.2 改進建議

1. **增加訓練資料**: 更多受試者資料可能提升模型泛化能力
2. **資料增強**: 考慮加入噪音增強、速度變化等技術
3. **模型架構**: 可嘗試不同的 Conv Head 配置
4. **後處理**: 加入平滑濾波器減少碎片化預測

---

## 附錄

### A. 類別定義 (3 分類)
- **0 (Child)**: 小孩語音
- **1 (Adult)**: 測試者/成人語音
- **2 (Silence)**: 靜音/背景噪音

### B. 評估指標說明
- **DER (Diarization Error Rate)**: 語者分離總錯誤率
- **False Alarm**: 將非語音誤判為語音的比例
- **Missed Detection**: 將語音誤判為非語音的比例
- **Speaker Confusion**: 語音區段語者判斷錯誤的比例

