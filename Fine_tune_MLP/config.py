# -*- coding: utf-8 -*-
"""
Whisper + MLP + Median Filter 語者分離模型
配置檔案：包含所有超參數與路徑設定

架構變更：將 Bi-LSTM 替換為 MLP + Median Filter
日期：2026-01-26
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
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
    model_size: str = "medium"
    
    # 是否凍結 Whisper Encoder 的權重
    freeze_encoder: bool = True
    
    # ==================== 音訊處理設定 ====================
    # 音訊窗口大小（秒）
    window_size: float = 20.0
    
    # Step size（毫秒）- Whisper 的 step size 為 20ms
    step_size_ms: float = 20.0
    
    # 取樣率（Hz）- Whisper 使用 16kHz
    sample_rate: int = 16000
    
    # 訓練時音訊窗口重疊率
    train_overlap_ratio: float = 0.5
    
    # ==================== 訓練設定 ====================
    # 批次大小 - 根據 RTX 5070 Ti 記憶體調整
    batch_size: int = 8
    
    # 學習率
    learning_rate: float = 5e-4
    
    # 權重衰減
    weight_decay: float = 1e-4
    
    # 最大訓練週期數
    epochs: int = 15
    
    # Early Stopping 耐心值
    early_stopping_patience: int = 5
    
    # 梯度累積步數
    gradient_accumulation_steps: int = 1
    
    # ==================== MLP 架構設定 ====================
    # MLP 隱藏層維度列表（可調整層數和大小）
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # Dropout 機率
    dropout: float = 0.2
    
    # ==================== Median Filter 設定 ====================
    # Median Filter 核心大小（必須為奇數）
    median_filter_size: int = 5
    
    # 是否在訓練時使用 Median Filter（通常只在推論時使用）
    use_median_filter_in_training: bool = False
    
    # ==================== 輸出設定 ====================
    # 輸出類別數 (0: Child, 1: Adult, 2: Silence) - 3 分類任務
    num_classes: int = 3
    
    # 類別名稱對應
    class_names: List[str] = field(default_factory=lambda: ["Child", "Adult", "Silence"])
    
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
    output_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP\outputs"
    
    # 模型儲存目錄
    model_save_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP\models"
    
    # 日誌目錄
    log_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_MLP\logs"
    
    # ==================== 硬體優化設定 ====================
    # DataLoader worker 數量
    num_workers: int = 8
    
    # 是否使用 pin_memory
    pin_memory: bool = True
    
    # 是否使用混合精度訓練 (AMP)
    use_amp: bool = True
    
    # 是否啟用 cuDNN benchmark
    cudnn_benchmark: bool = True
    
    # ==================== 評估設定 ====================
    # DER 計算的 collar（毫秒）
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
        
        # 確保 median filter size 為奇數
        if self.median_filter_size % 2 == 0:
            self.median_filter_size += 1
    
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
        """計算每個窗口的 frame 數"""
        return int(self.window_size * 1000 / self.step_size_ms)
    
    @property
    def model_weights_path(self) -> str:
        """取得模型權重儲存路徑"""
        return os.path.join(self.model_save_dir, f"whisper_{self.model_size}_mlp_best.pt")
    
    @property
    def mlp_num_layers(self) -> int:
        """取得 MLP 層數"""
        return len(self.mlp_hidden_dims)
    
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
        print("訓練配置摘要 (MLP + Median Filter 版本)")
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
        
        print(f"\n【MLP 架構設定】")
        print(f"  MLP 隱藏層維度: {self.mlp_hidden_dims}")
        print(f"  MLP 層數: {self.mlp_num_layers}")
        print(f"  Dropout: {self.dropout}")
        print(f"  輸出類別數: {self.num_classes}")
        print(f"  類別名稱: {self.class_names}")
        print(f"  類別對應: 0=Child, 1=Adult, 2=Silence")
        
        print(f"\n【Median Filter 設定】")
        print(f"  Median Filter 核心大小: {self.median_filter_size}")
        print(f"  訓練時使用 Median Filter: {self.use_median_filter_in_training}")
        
        print(f"\n【硬體優化設定】")
        print(f"  運算裝置: {self.device}")
        print(f"  DataLoader Workers: {self.num_workers}")
        print(f"  Pin Memory: {self.pin_memory}")
        print(f"  混合精度訓練 (AMP): {self.use_amp}")
        print(f"  cuDNN Benchmark: {self.cudnn_benchmark}")
        
        print(f"\n【資料路徑】")
        print(f"  訓練資料目錄: {self.train_data_dir}")
        print(f"  測試音檔: {self.test_audio_path}")
        print(f"  測試標註: {self.test_srt_path}")
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
    """
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
    config = Config()
    config.print_config()
