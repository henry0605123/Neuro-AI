# -*- coding: utf-8 -*-
"""
論文復現：Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions
配置檔案：包含所有超參數與路徑設定

作者：AI Engineering Team
日期：2026-01-19
修訂：2026-01-19 - 改為 3 分類任務 (Child, Adult, Silence)
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class Config:
    """
    超參數配置類別
    ==================
    
    可調整的主要參數：
    - model_size: Whisper 模型大小 (tiny/base/small/medium/large)
    - window_size: 音訊窗口大小（秒）
    - batch_size: 批次大小
    - learning_rate: 學習率
    - epochs: 訓練週期數
    """
    
    # ==================== 模型設定 ====================
    # Whisper 模型大小: "tiny", "base", "small", "medium", "large"
    # 論文建議使用 medium 或 large 以獲得最佳效果
    model_size: str = "medium"
    
    # 是否凍結 Whisper Encoder 的權重 (論文中 encoder 是凍結的)
    freeze_encoder: bool = True
    
    # ==================== 音訊處理設定 ====================
    # 音訊窗口大小（秒）- 論文 Section 5.3 建議使用 20 秒
    window_size: float = 20.0
    
    # Step size（毫秒）- Whisper 的 step size 為 20ms
    step_size_ms: float = 20.0
    
    # 取樣率（Hz）- Whisper 使用 16kHz
    sample_rate: int = 16000
    
    # 訓練時音訊窗口重疊率（論文使用 50%）
    train_overlap_ratio: float = 0.5
    
    # ==================== 訓練設定 ====================
    # 批次大小 - 根據 RTX 5070 Ti 記憶體調整
    batch_size: int = 8
    
    # 學習率 - 論文建議 5e-4
    learning_rate: float = 5e-4
    
    # 權重衰減 - 論文建議 1e-4
    weight_decay: float = 1e-5
    
    # 最大訓練週期數 - 論文建議 15
    epochs: int = 15
    
    # Early Stopping 耐心值
    early_stopping_patience: int = 5
    
    # 梯度累積步數（適用於 GPU 記憶體不足時）
    gradient_accumulation_steps: int = 1
    
    # ==================== 下游模型架構設定 ====================
    # Conv Head 的輸出通道數
    conv_channels: int = 128
    
    # Dropout 機率
    dropout: float = 0.2
    
    # 輸出類別數 (0: Child, 1: Adult, 2: Silence) - 3 分類任務
    num_classes: int = 3
    
    # 類別名稱對應（順序與 Label ID 嚴格對應）
    # ID 0 -> Child, ID 1 -> Adult, ID 2 -> Silence
    class_names: List[str] = field(default_factory=lambda: ["Child", "Adult", "Silence"])
    
    # ==================== Median Filter 設定 ====================
    # Median Filter 核心大小（必須為奇數，用於推論時平滑預測）
    median_filter_size: int = 5
    
       # ==================== 資料路徑設定 ====================
    # 與 outputs/medium_128_filter/ANALYSIS_REPORT.md「1.3 資料集」一致：
    # 訓練資料為 Fine_tune 根目錄（每子資料夾 = 一個樣本），驗證集比例 25% 由 val_split_ratio 從訓練集隨機切分
    #train_data_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\data\playlogue_splits\train"
    # 若設為獨立 val 目錄則優先使用；None 表示依 val_split_ratio 從 train_data_dir 切分（與報告一致）
    #val_data_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\data\playlogue_splits\val"
    # 測試資料集根目錄（每子資料夾 = 一個 clip：音訊 + diarization.rttm）。勿把此路徑填進 test_audio_path。
    #test_data_dir: Optional[str] = (
    #    r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\data\playlogue_splits\test"
    #)
    # 單檔測試模式：僅在 test_data_dir 為 None 或 "" 時使用（須為 .wav/.mp3 等檔案，不可填資料夾）
    #test_audio_path: str = ""
    #test_srt_path: str = ""


    # 單檔模式可指定單一 .rttm；多 clip 模式以各子資料夾內 diarization.rttm 為準（勿填目錄路徑）
    test_rttm_path: Optional[str] = None

    # 測試資料集根目錄（每子資料夾 = 一個 clip：音訊 + 標註）；若未設定可用下方固定比例切分
    test_data_dir: Optional[str] = None

    # 單檔測試模式（僅在 test_data_dir 為空且未啟用固定比例切分時使用）
    test_audio_path: str = ""
    test_srt_path: str = ""

    # 是否從 train_data_dir 以固定比例隨機切分 train/val/test（避免寫死單一 test_audio_path）
    enable_fixed_split_from_train_dir: bool = True
    # 固定比例（train, val, test）
    train_val_test_ratio: Tuple[int, int, int] = (2, 1, 1)
    #
    # --- 改回「自閉症 / 原本」每受試者一資料夾 + SRT 時，可改為例如： ---
    train_data_dir = r"D:\自閉症\data\Fine_tune"
    val_data_dir = None
    test_data_dir = None
    test_audio_path = ""
    test_srt_path = ""
    test_rttm_path = None
    
    # 輸出目錄
    output_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\outputs"
    
    # 模型儲存目錄
    model_save_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\models"
    
    # 日誌目錄
    log_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_paper\logs"
    
    # ==================== 硬體優化設定 ====================
    # DataLoader worker 數量 - 針對 AMD Ryzen 7 7700 優化
    num_workers: int = 8
    
    # 是否使用 pin_memory（建議對 GPU 訓練啟用）
    pin_memory: bool = True
    
    # 是否使用混合精度訓練 (AMP)
    use_amp: bool = True
    
    # 是否啟用 cuDNN benchmark（對固定大小輸入有加速效果）
    cudnn_benchmark: bool = True
    
    # ==================== 評估設定 ====================
    # DER 計算的 collar（毫秒）- 論文使用 100ms
    der_collar_ms: float = 100.0
    
    # 驗證集比例
    val_split_ratio: float = 0.25
    
    # 隨機種子
    seed: int = 42
    
    def __post_init__(self):
        """初始化後處理：建立必要目錄、設定裝置"""
        # 建立輸出目錄
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 設定裝置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 設定 cuDNN benchmark
        if torch.cuda.is_available() and self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # Windows：預載入音訊的 Dataset 經多進程 DataLoader 傳遞時易 pickle 失敗，改為 num_workers=0
        if sys.platform == "win32" and self.num_workers > 0:
            self.num_workers = 0

        # 測試：目錄模式 vs 單檔模式（與 Fine_tune_LSTM 一致）
        td = getattr(self, "test_data_dir", None)
        if isinstance(td, str) and td.strip():
            self.test_data_dir = td.strip()
        else:
            self.test_data_dir = None

        ta = self.test_audio_path
        if isinstance(ta, str) and ta.strip() and os.path.isdir(ta.strip()):
            dir_path = ta.strip()
            if self.test_data_dir is None:
                self.test_data_dir = dir_path
                self.test_audio_path = ""
            elif os.path.normpath(self.test_data_dir) == os.path.normpath(dir_path):
                self.test_audio_path = ""
            else:
                raise ValueError(
                    "test_data_dir 與 test_audio_path（目錄）不一致；請只設定 test_data_dir，"
                    "或將 test_audio_path 改為單一音檔。"
                )
        elif self.test_data_dir is None:
            allow_empty_single_test = (
                bool(getattr(self, "enable_fixed_split_from_train_dir", False))
                and not (getattr(self, "val_data_dir", None) or "")
            )
            if allow_empty_single_test:
                self.test_audio_path = ""
                self.test_srt_path = (self.test_srt_path or "").strip() if isinstance(self.test_srt_path, str) else ""
                return
            if not isinstance(ta, str) or not str(ta).strip():
                raise ValueError(
                    "未設定 test_data_dir 時，test_audio_path 必須為單一音檔路徑（不可為資料夾）。"
                    "若要使用整個測試集目錄，請設定 test_data_dir。"
                )
            self.test_audio_path = str(ta).strip()

    @property
    def whisper_model_name(self) -> str:
        """取得 Hugging Face 上的 Whisper 模型名稱"""
        return f"openai/whisper-{self.model_size}"
    
    @property
    def window_samples(self) -> int:
        """計算音訊窗口的取樣點數"""
        return int(self.window_size * self.sample_rate)
    
    @property
    def step_samples(self) -> int:
        """計算 step size 的取樣點數"""
        return int(self.step_size_ms * self.sample_rate / 1000)
    
    @property
    def frames_per_window(self) -> int:
        """計算每個窗口的 frame 數 (用於標籤對齊)"""
        # Whisper encoder 每 20ms 產生一個 frame
        return int(self.window_size * 1000 / self.step_size_ms)
    
    @property
    def model_weights_path(self) -> str:
        """取得模型權重儲存路徑"""
        return os.path.join(self.model_save_dir, f"whisper_{self.model_size}_diarization_best.pt")
    
    def get_training_subjects(self) -> List[str]:
        """取得訓練資料夾中的受試者列表"""
        subjects = []
        if os.path.exists(self.train_data_dir):
            for folder in os.listdir(self.train_data_dir):
                folder_path = os.path.join(self.train_data_dir, folder)
                if os.path.isdir(folder_path):
                    subjects.append(folder)
        return subjects
    
    def print_config(self):
        """印出當前配置"""
        print("=" * 60)
        print("訓練配置摘要")
        print("=" * 60)
        print(f"\n【模型設定】")
        print(f"  Whisper 模型大小: {self.model_size}")
        print(f"  Hugging Face 模型名稱: {self.whisper_model_name}")
        print(f"  凍結 Encoder: {self.freeze_encoder}")
        
        print(f"\n【音訊處理設定】")
        print(f"  窗口大小: {self.window_size} 秒")
        print(f"  Step size: {self.step_size_ms} ms")
        print(f"  取樣率: {self.sample_rate} Hz")
        print(f"  每個窗口的 Frame 數: {self.frames_per_window}")
        
        print(f"\n【訓練設定】")
        print(f"  批次大小: {self.batch_size}")
        print(f"  學習率: {self.learning_rate}")
        print(f"  權重衰減: {self.weight_decay}")
        print(f"  最大訓練週期: {self.epochs}")
        print(f"  Early Stopping 耐心值: {self.early_stopping_patience}")
        
        print(f"\n【下游架構設定】")
        print(f"  Conv 通道數: {self.conv_channels}")
        print(f"  Dropout: {self.dropout}")
        print(f"  輸出類別數: {self.num_classes}")
        print(f"  類別名稱: {self.class_names}")
        print(f"  類別對應: 0=Child, 1=Adult, 2=Silence")
        
        print(f"\n【硬體優化設定】")
        print(f"  運算裝置: {self.device}")
        print(f"  DataLoader Workers: {self.num_workers}")
        print(f"  Pin Memory: {self.pin_memory}")
        print(f"  混合精度訓練 (AMP): {self.use_amp}")
        print(f"  cuDNN Benchmark: {self.cudnn_benchmark}")
        
        print(f"\n【資料路徑】")
        print(f"  訓練資料目錄: {self.train_data_dir}")
        print(f"  驗證資料目錄 (可選): {self.val_data_dir or '(未設定，使用隨機切分)'}")
        if getattr(self, "test_data_dir", None):
            print(f"  測試資料目錄 (多 clip): {self.test_data_dir}")
        else:
            print(f"  測試音檔 (單檔): {self.test_audio_path}")
            print(f"  測試標註 (SRT): {self.test_srt_path}")
            print(f"  測試標註 (RTTM): {self.test_rttm_path or '(未設定)'}")
        print(f"  輸出目錄: {self.output_dir}")
        
        print(f"\n【訓練受試者】")
        subjects = self.get_training_subjects()
        for i, subj in enumerate(subjects, 1):
            print(f"  {i}. {subj}")
        
        print("=" * 60)


# 預設配置實例
def get_default_config() -> Config:
    """取得預設配置"""
    return Config()


def get_config_for_model_size(model_size: str, **kwargs) -> Config:
    """
    根據模型大小取得建議配置
    
    Args:
        model_size: Whisper 模型大小
        **kwargs: 其他可覆寫的參數
    
    Returns:
        Config: 配置實例
    """
    # 根據模型大小調整批次大小（考慮 GPU 記憶體）
    batch_size_map = {
        "tiny": 16,
        "base": 16,
        "small": 12,
        "medium": 8,
        "large": 4,
        "large-v2": 4,
        "large-v3": 4,
    }
    
    suggested_batch_size = batch_size_map.get(model_size, 8)
    
    return Config(
        model_size=model_size,
        batch_size=kwargs.get("batch_size", suggested_batch_size),
        **{k: v for k, v in kwargs.items() if k != "batch_size"}
    )


if __name__ == "__main__":
    # 測試配置
    config = Config()
    config.print_config()
