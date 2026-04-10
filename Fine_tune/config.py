"""
Pyannote Fine-tune 配置檔案
==========================
請根據你的環境修改以下參數
"""

import os
from pathlib import Path

# ============== 路徑配置 ==============

# 專案根目錄
PROJECT_ROOT = Path(r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune")

# 原始訓練資料目錄（包含 SRT 標註和 MP3 音檔）
RAW_DATA_DIR = Path(r"C:\自閉症\data\Fine_tune")

# 處理後的資料目錄
PROCESSED_DATA_DIR = PROJECT_ROOT / "data"

# 模型儲存目錄
MODEL_DIR = PROJECT_ROOT / "models"

# 輸出目錄
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 日誌目錄（TensorBoard）
LOG_DIR = PROJECT_ROOT / "logs"

# ============== Hugging Face 配置 ==============

# Hugging Face Token（需要有 pyannote 模型存取權限）
# 請設定環境變數 HF_TOKEN：https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN")

# ============== 模型配置 ==============

# 預訓練模型（用於微調）
PRETRAINED_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
PRETRAINED_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# ============== 訓練參數 ==============

# 批次大小（RTX 5070 Ti 16GB 可用更大批次）
BATCH_SIZE = 64  # 從 32 提升到 64

# 訓練輪數
EPOCHS = 20

# 學習率
LEARNING_RATE = 1e-4

# 每個訓練樣本的長度（秒）
DURATION = 5.0

# 驗證集比例
VAL_RATIO = 0.2

# 隨機種子
RANDOM_SEED = 42

# 工作執行緒數量（Ryzen 7 7700 有 8 核心）
# Windows 多進程限制，建議使用較小值
NUM_WORKERS = 4  # Windows 相容性

# 混合精度訓練（RTX 5070 Ti 支援 Tensor Cores）
USE_MIXED_PRECISION = True
COMPUTE_PRECISION = "float16"  # 或 "bfloat16"

# 梯度累積步數（如果記憶體不足可啟用）
ACCUMULATE_GRAD_BATCHES = 1

# Pin memory 加速資料傳輸
PIN_MEMORY = True

# ============== 推論參數 ==============

# 說話人數量範圍
MIN_SPEAKERS = 2
MAX_SPEAKERS = 3

# Whisper 模型配置（針對 RTX 5070 Ti 優化）
WHISPER_MODEL = "large-v3"
WHISPER_BATCH_SIZE = 32  # 從 16 提升到 32
WHISPER_COMPUTE_TYPE = "float16"  # RTX 5070 Ti 完美支援

# ============== 說話人標籤映射 ==============

# SRT 中的說話人標籤 -> 標準化標籤
SPEAKER_MAPPING = {
    "測試者": "SPEAKER_00",
    "小孩": "SPEAKER_01",
    "Unknown": "SPEAKER_02",
}

# 反向映射（用於輸出）
SPEAKER_REVERSE_MAPPING = {
    "SPEAKER_00": "測試者",
    "SPEAKER_01": "小孩",
    "SPEAKER_02": "Unknown",
}

# ============== 測試音檔 ==============

# 用於測試微調效果的音檔
TEST_AUDIO_FILE = Path(r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\20250106-蔡oo.mp3")

# ============== 建立必要目錄 ==============

def create_directories():
    """建立所有必要的目錄"""
    directories = [
        PROCESSED_DATA_DIR / "train",
        PROCESSED_DATA_DIR / "val",
        MODEL_DIR / "finetuned_segmentation",
        OUTPUT_DIR,
        LOG_DIR,
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 已建立目錄: {dir_path}")

if __name__ == "__main__":
    create_directories()
    print("\n✅ 所有目錄已建立完成！")
