# -*- coding: utf-8 -*-
"""
Whisper + Bi-LSTM 語者分離模型
配置檔案：包含所有超參數與路徑設定

架構變更：將 3 層 Conv1D 替換為 2 層 Bi-LSTM
日期：2026-01-24

預設資料路徑：Playlogue（prepare_playlogue_finetune_layout.py 產生之 playlogue_splits）。
若改回自閉症等「每受試者一資料夾 + SRT」，請見下方「資料路徑設定」區塊註解範例。
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
    - model_size: Whisper 模型大小 (tiny…large-v3、large-v3-turbo、large-turbo 等)
    - window_size: 音訊窗口大小（秒）
    - batch_size: 批次大小
    - learning_rate: 學習率
    - epochs: 訓練週期數
    """
    
    # ==================== 模型設定 ====================
    # Whisper 模型大小: tiny…large-v3；large-v3-turbo；large-turbo（同 large-v3-turbo）
    model_size: str = "medium"
    
    # 是否凍結 Whisper Encoder 的權重
    freeze_encoder: bool = True
    # Whisper Encoder LoRA（與 train.py checkpoint、model.py 一致；未啟用時為凍結 Encoder + Bi-LSTM）
    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # 前向未使用 decoder；預設凍結以節省記憶體
    freeze_whisper_decoder: bool = True

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
    # 批次大小（與 medium_128_filter 實驗報告一致：Whisper-medium + Bi-LSTM hidden=128）
    batch_size: int = 32
    
    # 學習率
    learning_rate: float = 5e-4
    
    # 權重衰減
    weight_decay: float = 1e-3
    
    # 最大訓練週期數
    epochs: int = 1
    
    # Early Stopping 耐心值
    early_stopping_patience: int = 5
    
    # 梯度累積步數
    gradient_accumulation_steps: int = 1
    
    # ==================== Bi-LSTM 架構設定 ====================
    # LSTM 隱藏層維度（單向，雙向後會 *2）
    lstm_hidden_size: int = 128
    
    # LSTM 層數
    lstm_num_layers: int = 2
    
    # Dropout 機率
    dropout: float = 0.2
    
    # 輸出類別數 (0: Child, 1: Adult, 2: Silence) - 3 分類任務
    num_classes: int = 3
    
    # 類別名稱對應
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
    ## 單檔測試模式：僅在 test_data_dir 為 None 或 "" 時使用（須為 .wav/.mp3 等檔案，不可填資料夾）
    #test_audio_path: str = ""
    #test_srt_path: str = ""


    # 單檔模式可指定單一 .rttm；多 clip 模式以各子資料夾內 diarization.rttm 為準（勿填目錄路徑）
    test_rttm_path: Optional[str] = None

    # 測試資料集根目錄（每子資料夾 = 一個 clip：音訊 + 標註）；若未設定可用下方固定比例切分
    test_data_dir: Optional[str] = None

    # 單檔測試模式（僅在 test_data_dir 為空且未啟用固定比例切分時使用）
    test_audio_path: str = ""
    test_srt_path: str = ""
    # 單檔測試：編輯標註 JSON（與 test_audio_path 併用；標註優先 RTTM → JSON → SRT）
    test_json_path: Optional[str] = None

    # 是否從 train_data_dir 以固定比例隨機切分 train/val/test（避免寫死單一 test_audio_path）
    enable_fixed_split_from_train_dir: bool = True
    # 固定比例（train, val, test）
    train_val_test_ratio: Tuple[int, int, int] = (7, 2, 1)
    #
    # --- 改回「自閉症 / 原本」每受試者一資料夾 + SRT 時，可改為例如： ---
    train_data_dir = r"D:\自閉症\data\Fine_tune"
    val_data_dir = None
    test_data_dir = None
    test_audio_path = ""
    test_srt_path = ""
    test_json_path = None
    test_rttm_path = None

    # 輸出目錄
    output_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\outputs"
    
    # 模型儲存目錄
    model_save_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\models"
    
    # 日誌目錄
    log_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\logs"
    
    # ==================== 硬體優化設定 ====================
    # DataLoader worker 數量（Linux 建議 4～8）
    num_workers: int = 8
    # Windows 預設會將 num_workers 改為 0（見下方），因預載音訊的 Dataset 需 pickle 給子進程，易 OSError。
    # 若你接受風險並想嘗試多進程加速，改為 True 後可保留 num_workers=4/8（仍可能因資料量過大失敗）。
    allow_dataloader_workers_on_windows: bool = False
    
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
    
    # 隨機種子：None 表示每次執行自動產生新 seed；若要固定可重現，設為整數（如 42）
    seed: Optional[int] = None
    
    def __post_init__(self):
        """初始化後處理：建立必要目錄、設定裝置"""
        # 若未指定 seed，則每次執行自動產生，便於觀察模型在不同切分下的穩定度
        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(4), "little")
            print(f"🎲 本次自動 seed: {self.seed}")

        # 建立輸出目錄
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 設定裝置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 設定 cuDNN benchmark
        if torch.cuda.is_available() and self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # Windows：預載入整段音訊的 Dataset 體積大，多進程 DataLoader 需 pickle 整個 Dataset
        # 至子進程，易觸發 OSError: [Errno 22] Invalid argument / pickle truncated。
        # 預設改為 num_workers=0；若 allow_dataloader_workers_on_windows=True 則不覆寫（實驗用）。
        if (
            sys.platform == "win32"
            and self.num_workers > 0
            and not getattr(self, "allow_dataloader_workers_on_windows", False)
        ):
            print(
                "⚠️  Windows：已將 num_workers 改為 0（避免大型預載 Dataset 在子進程 pickle 失敗）。\n"
                "   若仍想使用多 worker，請在 config 設定 allow_dataloader_workers_on_windows=True，"
                "並可將 num_workers 設為 4 或 8（不保證穩定）。"
            )
            self.num_workers = 0
        elif sys.platform == "win32" and self.num_workers > 0 and getattr(
            self, "allow_dataloader_workers_on_windows", False
        ):
            print(
                f"⚠️  Windows：已啟用 allow_dataloader_workers_on_windows，使用 num_workers={self.num_workers}。"
                "若出現 OSError / pickle 相關錯誤，請改回 num_workers=0 或關閉此選項。"
            )

        # 測試路徑：目錄模式 vs 單檔模式
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
        # HF 實際為 openai/whisper-large-v3-turbo（無 openai/whisper-large-turbo）
        if self.model_size == "large-turbo":
            return "openai/whisper-large-v3-turbo"
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
        return os.path.join(self.model_save_dir, f"whisper_{self.model_size}_bilstm_best.pt")
    
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
        print("訓練配置摘要 (Bi-LSTM 版本)")
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
        
        print(f"\n【Bi-LSTM 架構設定】")
        print(f"  LSTM 隱藏層維度 (單向): {self.lstm_hidden_size}")
        print(f"  LSTM 隱藏層維度 (雙向): {self.lstm_hidden_size * 2}")
        print(f"  LSTM 層數: {self.lstm_num_layers}")
        print(f"  Dropout: {self.dropout}")
        print(f"  輸出類別數: {self.num_classes}")
        print(f"  類別名稱: {self.class_names}")
        print(f"  類別對應: 0=Child, 1=Adult, 2=Silence")
        
        print(f"\n【硬體優化設定】")
        print(f"  運算裝置: {self.device}")
        print(f"  DataLoader Workers: {self.num_workers}")
        print(
            f"  Windows 允許多進程 DataLoader: "
            f"{'是' if getattr(self, 'allow_dataloader_workers_on_windows', False) else '否（預設會將 workers 改為 0）'}"
        )
        print(f"  Pin Memory: {self.pin_memory}")
        print(f"  混合精度訓練 (AMP): {self.use_amp}")
        print(f"  cuDNN Benchmark: {self.cudnn_benchmark}")
        print(f"  隨機種子 (seed): {self.seed}")
        
        print(f"\n【資料路徑】")
        print(f"  訓練資料目錄: {self.train_data_dir}")
        print(f"  驗證資料目錄 (可選): {self.val_data_dir or '(未設定，使用隨機切分)'}")
        if getattr(self, "test_data_dir", None):
            print(f"  測試資料目錄 (多 clip): {self.test_data_dir}")
        elif getattr(self, "enable_fixed_split_from_train_dir", False) and not (self.test_audio_path or "").strip():
            ratio = getattr(self, "train_val_test_ratio", (2, 1, 1))
            print(
                "  測試資料來源: 由 train_data_dir 隨機固定比例切分"
                f" (train/val/test = {ratio[0]}/{ratio[1]}/{ratio[2]})"
            )
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
    """
    batch_size_map = {
        "tiny": 16,
        "base": 16,
        "small": 12,
        "medium": 8,
        "large": 4,
        "large-v2": 4,
        "large-v3": 4,
        "large-v3-turbo": 4,
        "large-turbo": 4,
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
