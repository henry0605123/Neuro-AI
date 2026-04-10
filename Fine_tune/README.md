# Pyannote 語者分離模型 Fine-tune Pipeline

## 📋 專案說明

本專案用於對 pyannote 語者分離（Speaker Diarization）模型進行微調（Fine-tune），以提升在特定場景（自閉症評估對話）的語者分離準確度。

## 📁 檔案結構

```
Fine_tune/
├── README.md                    # 本說明檔
├── config.py                    # 配置檔案
├── 01_prepare_dataset.py        # Step 1: 準備資料集
├── 02_train_model.py            # Step 2: 訓練模型
├── 03_transcribe_with_finetuned.py  # Step 3: 使用微調模型轉錄
├── utils/
│   ├── __init__.py
│   ├── srt_parser.py            # SRT 檔案解析工具
│   ├── rttm_converter.py        # RTTM 格式轉換工具
│   └── audio_utils.py           # 音訊處理工具
├── data/                        # 處理後的訓練資料
│   ├── train/
│   └── val/
├── models/                      # 微調後的模型
│   └── finetuned_segmentation/
└── outputs/                     # 轉錄輸出結果
```

## ⚡ 硬體優化配置

**本專案已針對以下硬體優化：**
- **GPU**: NVIDIA RTX 5070 Ti (16GB VRAM)
- **CPU**: AMD Ryzen 7 7700 (8核心16執行緒)

**優化項目：**
- ✅ 批次大小提升至 64（充分利用 16GB VRAM）
- ✅ 啟用混合精度訓練（Tensor Cores 加速 **2-3倍**）
- ✅ 8 執行緒資料載入（充分利用 Ryzen 7 7700）
- ✅ Pin Memory 和自動演算法優化
- ✅ 預期訓練速度提升：**2.5-3倍** 🚀

📖 詳細說明請參考：[HARDWARE_OPTIMIZATION.md](HARDWARE_OPTIMIZATION.md)

## 🔧 環境需求

### Python 版本
- Python 3.9 - 3.11

### 必要套件
```bash
pip install torch torchaudio
pip install pyannote.audio
pip install whisperx
pip install pydub
pip install pandas
pip install tensorboard
pip install lightning
```

### Hugging Face Token
需要一個有效的 Hugging Face token，並且需要接受以下模型的使用條款：
1. https://huggingface.co/pyannote/segmentation-3.0
2. https://huggingface.co/pyannote/speaker-diarization-3.1

## 📝 使用步驟

### Step 0: 硬體效能測試（建議）
```bash
python test_performance.py
```
驗證 GPU 和 CPU 效能，確認優化配置正常運作。

### Step 1: 設定配置
編輯 `config.py`，設定以下參數：
- `HF_TOKEN`: 你的 Hugging Face token
- `DATA_DIR`: 原始資料目錄路徑
- `OUTPUT_DIR`: 輸出目錄路徑

### Step 2: 準備資料集
```bash
python 01_prepare_dataset.py
```
此步驟會：
- 讀取所有 SRT 標註檔案
- 將標註轉換為 pyannote 訓練所需的 RTTM 格式
- 切分訓練集和驗證集
- 生成 protocol 配置檔

### Step 3: 訓練模型
```bash
python 02_train_model.py
```
此步驟會：
- 載入預訓練的 pyannote segmentation 模型
- 使用準備好的資料進行微調
- 儲存微調後的模型

訓練過程可以用 TensorBoard 監控：
```bash
tensorboard --logdir=Fine_tune/logs
```

### Step 4: 使用微調模型進行轉錄
```bash
python 03_transcribe_with_finetuned.py
```
此步驟會：
- 載入 WhisperX 進行語音轉錄
- 使用微調後的 pyannote 模型進行語者分離
- 輸出標準 SRT 格式的轉錄結果

## ⚙️ 參數調整

### 訓練參數（在 config.py 中調整）
- `BATCH_SIZE`: 批次大小（預設 32）
- `EPOCHS`: 訓練輪數（預設 20）
- `LEARNING_RATE`: 學習率（預設 1e-4）
- `DURATION`: 訓練樣本長度（預設 5.0 秒）

### 推論參數
- `MIN_SPEAKERS`: 最少說話人數（預設 2）
- `MAX_SPEAKERS`: 最多說話人數（預設 3）

## 📊 資料格式說明

### 輸入 SRT 格式
```
1
00:00:02,475 --> 00:00:12,425
測試者: 好啊，老師這邊我們要玩拼圖喔

2
00:00:15,550 --> 00:00:17,325
小孩: 姐姐
```

### 輸出 RTTM 格式（訓練用）
```
SPEAKER audio1 1 2.475 9.950 <NA> <NA> 測試者 <NA> <NA>
SPEAKER audio1 1 15.550 1.775 <NA> <NA> 小孩 <NA> <NA>
```

## 🔍 注意事項

1. **GPU 記憶體**: 訓練需要至少 8GB GPU 記憶體，建議使用 16GB 以上
2. **資料品質**: 標註的準確度直接影響模型效果
3. **類別平衡**: 確保「測試者」和「小孩」的發言時長相對平衡
4. **驗證集**: 建議保留 20% 資料作為驗證集

## 📧 問題排除

### 常見問題

1. **CUDA out of memory**
   - 減少 `BATCH_SIZE`
   - 減少 `DURATION`

2. **模型載入失敗**
   - 確認 HF_TOKEN 正確
   - 確認已接受模型使用條款

3. **SRT 解析錯誤**
   - 檢查 SRT 檔案格式是否正確
   - 確認說話人標籤格式一致

## 📚 參考資料

- [pyannote.audio 官方文檔](https://github.com/pyannote/pyannote-audio)
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Pyannote Fine-tuning Tutorial](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/training_a_model.ipynb)
