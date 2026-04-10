"""
Pyannote Fine-tune 配置檔案 (優化版 V2)
======================================
針對小資料集進行優化的訓練配置

主要改進：
1. 資料切片配置 - 將長音檔切成小片段
2. 類別加權配置 - 解決標籤不平衡問題
3. 多採樣配置 - 增加有效訓練資料量
4. 資料增強配置 - 提高泛化能力
"""

import os
from pathlib import Path

# ============== 路徑配置 ==============

# 專案根目錄
PROJECT_ROOT = Path(r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune")

# 原始訓練資料目錄（包含 SRT 標註和 MP3 音檔）
RAW_DATA_DIR = Path(r"C:\自閉症\data\Fine_tune")

# 處理後的資料目錄
PROCESSED_DATA_DIR = PROJECT_ROOT / "data_v2"

# 切片資料目錄
SLICED_DATA_DIR = PROJECT_ROOT / "data_v2" / "sliced"

# 模型儲存目錄
MODEL_DIR = PROJECT_ROOT / "models"

# 輸出目錄
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 日誌目錄（TensorBoard）
LOG_DIR = PROJECT_ROOT / "logs"

# ============== Hugging Face 配置 ==============

# 請設定環境變數 HF_TOKEN（勿將 token 寫入版控）；https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN")

# ============== 模型配置 ==============

PRETRAINED_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
PRETRAINED_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# ============== 資料切片配置 (核心優化) ==============

# 訓練片段長度（秒）- 這是模型一次看到的音訊長度
SEGMENT_DURATION = 5.0

# 片段重疊比例 - 0.5 表示相鄰片段有 50% 重疊
# 重疊可以讓模型學習到更多邊界資訊
SEGMENT_OVERLAP = 0.5

# 最小語音比例 - 片段中必須有至少這麼多比例的語音
# 過濾掉純靜音片段
MIN_SPEECH_RATIO = 0.1

# 要求小孩語音 - 是否要求片段中必須包含小孩的語音
# 這可以解決小孩 recall 為 0 的問題
REQUIRE_CHILD_SPEECH = True

# 小孩語音最小比例 - 如果 REQUIRE_CHILD_SPEECH=True，
# 片段中小孩語音至少要佔這麼多比例
MIN_CHILD_RATIO = 0.05

# 每個原始檔案的採樣上限 - 防止單一檔案主導訓練
MAX_SAMPLES_PER_FILE = 500

# ============== 類別加權配置 (解決標籤不平衡) ==============

# 是否使用類別加權
USE_CLASS_WEIGHTS = True

# 說話人權重配置
# SPEAKER_00 = 測試者, SPEAKER_01 = 小孩, SPEAKER_02 = Unknown
# 小孩權重更高，因為小孩語音通常較少但更重要
SPEAKER_WEIGHTS = {
    "SPEAKER_00": 1.0,   # 測試者 - 基準權重
    "SPEAKER_01": 3.0,   # 小孩 - 3 倍權重（最重要）
    "SPEAKER_02": 0.5,   # Unknown - 較低權重
}

# 動態計算權重 - 根據實際標籤分佈自動計算
AUTO_COMPUTE_WEIGHTS = True

# ============== 訓練參數 ==============

# 批次大小（RTX 5070 Ti 16GB）
BATCH_SIZE = 32  # 因為有更多樣本，降低批次大小以穩定訓練

# 訓練輪數 - 因為資料量增加，可能需要更多輪
EPOCHS = 30

# 學習率
LEARNING_RATE = 5e-5  # 降低學習率，讓模型更穩定地學習

# 學習率預熱步數
WARMUP_STEPS = 500

# 早停耐心值 - 如果驗證損失連續這麼多輪沒改善就停止
EARLY_STOPPING_PATIENCE = 10

# 凍結層比例 - 0.5 表示凍結前 50% 的層
FREEZE_RATIO = 0.5  # 從 0.7 降到 0.5，訓練更多層

# 驗證集比例
VAL_RATIO = 0.15

# 隨機種子
RANDOM_SEED = 42

# 工作執行緒數量 (Windows 建議設為 0，避免多進程卡住)
NUM_WORKERS = 0

# 混合精度訓練
USE_MIXED_PRECISION = True
COMPUTE_PRECISION = "float16"

# 梯度累積步數
ACCUMULATE_GRAD_BATCHES = 2

# Pin memory 加速資料傳輸
PIN_MEMORY = True

# ============== 資料增強配置 ==============

# 是否啟用資料增強
USE_AUGMENTATION = True

# 時間遮罩 - 隨機遮蓋一小段時間
TIME_MASK_PROB = 0.3
TIME_MASK_MAX_LENGTH = 0.5  # 秒

# 頻率遮罩 - 隨機遮蓋一些頻率
FREQ_MASK_PROB = 0.3

# 音量變化
VOLUME_CHANGE_PROB = 0.3
VOLUME_CHANGE_RANGE = (0.8, 1.2)

# 加入背景噪音
ADD_NOISE_PROB = 0.2
NOISE_SNR_RANGE = (15, 30)  # dB

# ============== 推論參數 ==============

MIN_SPEAKERS = 2
MAX_SPEAKERS = 3

WHISPER_MODEL = "large-v3"
WHISPER_BATCH_SIZE = 32
WHISPER_COMPUTE_TYPE = "float16"

# ============== 說話人標籤映射 ==============

SPEAKER_MAPPING = {
    "測試者": "SPEAKER_00",
    "小孩": "SPEAKER_01",
    "Unknown": "SPEAKER_02",
}

SPEAKER_REVERSE_MAPPING = {
    "SPEAKER_00": "測試者",
    "SPEAKER_01": "小孩",
    "SPEAKER_02": "Unknown",
}

# 說話人名稱列表（用於標籤對應）
SPEAKER_NAMES = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

# ============== 測試音檔 ==============

TEST_AUDIO_FILE = Path(r"C:\自閉症\data\ASD\20250106\20250106-蔡oo\20250106-蔡oo.mp3")

# ============== 日誌配置 ==============

# TensorBoard 日誌
USE_TENSORBOARD = True

# 每隔多少步記錄一次
LOG_INTERVAL = 50

# 每隔多少步驗證一次
VAL_INTERVAL = 200

# ============== 建立必要目錄 ==============

def create_directories():
    """建立所有必要的目錄"""
    directories = [
        PROCESSED_DATA_DIR,
        SLICED_DATA_DIR,
        MODEL_DIR / "finetuned_segmentation",
        OUTPUT_DIR,
        LOG_DIR,
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 已建立目錄: {dir_path}")


def print_config():
    """打印當前配置"""
    print("\n" + "=" * 60)
    print("當前訓練配置 (V2 優化版)")
    print("=" * 60)
    
    print("\n【資料切片配置】")
    print(f"  片段長度: {SEGMENT_DURATION} 秒")
    print(f"  重疊比例: {SEGMENT_OVERLAP * 100}%")
    print(f"  最小語音比例: {MIN_SPEECH_RATIO * 100}%")
    print(f"  要求小孩語音: {REQUIRE_CHILD_SPEECH}")
    if REQUIRE_CHILD_SPEECH:
        print(f"  小孩語音最小比例: {MIN_CHILD_RATIO * 100}%")
    print(f"  每檔案採樣上限: {MAX_SAMPLES_PER_FILE}")
    
    print("\n【類別加權配置】")
    print(f"  使用類別加權: {USE_CLASS_WEIGHTS}")
    print(f"  自動計算權重: {AUTO_COMPUTE_WEIGHTS}")
    print(f"  說話人權重:")
    for speaker, weight in SPEAKER_WEIGHTS.items():
        name = SPEAKER_REVERSE_MAPPING.get(speaker, speaker)
        print(f"    {name}: {weight}")
    
    print("\n【訓練參數】")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  訓練輪數: {EPOCHS}")
    print(f"  學習率: {LEARNING_RATE}")
    print(f"  凍結層比例: {FREEZE_RATIO * 100}%")
    print(f"  早停耐心值: {EARLY_STOPPING_PATIENCE}")
    
    print("\n【資料增強】")
    print(f"  啟用增強: {USE_AUGMENTATION}")
    if USE_AUGMENTATION:
        print(f"  時間遮罩機率: {TIME_MASK_PROB * 100}%")
        print(f"  音量變化機率: {VOLUME_CHANGE_PROB * 100}%")
        print(f"  加噪音機率: {ADD_NOISE_PROB * 100}%")
    
    print("=" * 60)


if __name__ == "__main__":
    create_directories()
    print_config()
