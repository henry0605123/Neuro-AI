# 語者分離模型實驗報告 — K-fold 交叉驗證 (Whisper + Bi-LSTM)

**生成時間**: 2026-04-27 01:24:27

---

## 1. 實驗設置摘要

- **驗證方式**: 5-fold 交叉驗證（樣本層級 `sklearn.model_selection.KFold`，與獨立 `val_data_dir` 互斥）。
- **最終指標**: `train_val_test_ratio` 第三項為 **0**（無獨立測試桶）。各折在**該折驗證集**上計算 DER 等後，取**算術平均 ± 樣本標準差**（與「同一固定測試集」模式不同；驗證集亦用於該折內 early stopping，解讀時略偏樂觀屬常見限制）。
- **不**以整體最佳折作為唯一代表模型。
- **Median filter（評估）**: 啟用
- **訓練資料根目錄**: D:\自閉症\data\Fine_tune
- **評估資料來源說明**: 由 train_data_dir 固定比例隨機切分 (train/val/test = 8/2/0)（指標實際計算於：**該折驗證集**）
- **每折子目錄**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\models/kfold_fold0` … `kfold_fold4`（同樣結構於 `log_dir`、`output_dir`）
- **彙總 JSON**: `C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\outputs/kfold_summary.json`

### 1.1 模型與訓練（與單次 pipeline 相同之 CLI／config）
| 參數 | 值 |
|------|------|
| Whisper 模型大小 | large |
| 批次大小 | 8 |
| 學習率 | 0.0001 |
| Epochs | 30 |
| LSTM hidden | 128 |
| LSTM 層數 | 2 |
| Encoder LoRA | 停用 |

---

## 2. 各折於「該折驗證集」之 DER 分量

| 折 | DER | False Alarm | Missed Detection | Speaker Confusion |
|----|-----|-------------|------------------|-------------------|
| 1 | 25.22% | 16.74% | 3.85% | 4.63% |
| 2 | 40.09% | 8.66% | 11.67% | 19.76% |
| 3 | 36.43% | 14.83% | 8.89% | 12.71% |
| 4 | 51.99% | 21.03% | 15.25% | 15.71% |
| 5 | 50.66% | 8.46% | 8.21% | 33.99% |

---

## 3. 五折平均 ± 標準差（主要輸出）

| 指標 | 平均 ± 標準差 |
|------|----------------|
| **DER** | **40.88% ± 11.01%** |
| False Alarm | 13.94% ± 5.40% |
| Missed Detection | 9.57% ± 4.23% |
| Speaker Confusion | 17.36% ± 10.82% |

---

## 4. 各折訓練摘要（驗證損失最佳值，僅供參考）

各折內部仍以驗證損失選該折最佳 epoch 權重；**報告主結論請以上方第 3 節第 3 節「平均」為準。**

- 各折最佳驗證損失之平均 ± 標準差: **0.7083 ± 0.2387**

---

## 5. 結論

本實驗以 5 折交叉驗證訓練多個模型，並在各折驗證集上彙總平均表現（無另留測試集）。

