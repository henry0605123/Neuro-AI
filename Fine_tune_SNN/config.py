# -*- coding: utf-8 -*-
"""
Whisper + ReLU 前饋頭（知識蒸餾自 Bi-LSTM Teacher）— 配置
目的：訓練完成後可進行 ANN→SNN 轉換
日期：2026-03-24
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class Config:
    """超參數與路徑（與 LSTM 專案對齊，並新增 KD / 學生頭設定）"""

    model_size: str = "medium"
    freeze_encoder: bool = True

    window_size: float = 20.0
    step_size_ms: float = 20.0
    sample_rate: int = 16000
    train_overlap_ratio: float = 0.5

    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    early_stopping_patience: int = 5
    gradient_accumulation_steps: int = 1

    # Teacher（Bi-LSTM）— 須與訓練 teacher 時一致
    teacher_checkpoint: str = (
        r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_LSTM\models"
        r"\whisper_medium_bilstm_best.pt"
    )
    # 若與 checkpoint 不符，load_teacher() 會依權重自動覆寫
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2

    # Student 頭：「conv」= 論文 3 層 Conv1d+ReLU；「mlp」= 全連接+ReLU（無 GELU）
    student_head_type: str = "conv"  # "conv" | "mlp"
    conv_channels: int = 256
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])

    # 知識蒸餾
    kd_temperature: float = 4.0
    kd_alpha: float = 0.5  # loss = alpha * CE + (1-alpha) * KD

    # Pipeline：Conv 頭轉 SNN（snntorch，僅 student_head_type=conv）
    snn_timesteps: int = 8
    snn_beta: float = 0.9  # Leaky LIF 漏電係數
    snn_finetune_epochs: int = 0  # >0 時於轉 SNN 後只微調 SNN 頭（Whisper 凍結）
    snn_finetune_lr: float = 1e-4

    dropout: float = 0.2
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["Child", "Adult", "Silence"])

    median_filter_size: int = 5

    train_data_dir: str = r"D:\自閉症\data\Fine_tune"
    test_audio_path: str = (
        r"D:\自閉症\data\TD\20250421-19752483鄭皓文\20250421-19752483鄭皓文.mp3"
    )
    test_srt_path: str = (
        r"D:\自閉症\data\TD\20250421-19752483鄭皓文\Aegisub.srt"
    )

    output_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\outputs"
    model_save_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\models"
    log_dir: str = r"C:\Users\User\PycharmProjects\PythonProject1\Fine_tune_SNN\logs"

    num_workers: int = 8
    pin_memory: bool = True
    use_amp: bool = True
    cudnn_benchmark: bool = True

    der_collar_ms: float = 100.0
    val_split_ratio: float = 0.25
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    @property
    def whisper_model_name(self) -> str:
        return f"openai/whisper-{self.model_size}"

    @property
    def window_samples(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def frames_per_window(self) -> int:
        return int(self.window_size * 1000 / self.step_size_ms)

    @property
    def student_weights_path(self) -> str:
        tag = "conv" if self.student_head_type == "conv" else "mlp"
        return os.path.join(
            self.model_save_dir,
            f"whisper_{self.model_size}_student_relu_{tag}_best.pt",
        )

    def get_training_subjects(self) -> List[str]:
        subjects = []
        if os.path.exists(self.train_data_dir):
            for folder in os.listdir(self.train_data_dir):
                folder_path = os.path.join(self.train_data_dir, folder)
                if os.path.isdir(folder_path):
                    subjects.append(folder)
        return subjects

    def print_config(self):
        print("=" * 60)
        print("訓練配置 (KD → ReLU Student，供後續 SNN)")
        print("=" * 60)
        print(f"  Whisper: {self.model_size}")
        print(f"  Teacher checkpoint: {self.teacher_checkpoint}")
        print(f"  Student 頭: {self.student_head_type}")
        print(f"  KD: T={self.kd_temperature}, alpha={self.kd_alpha} (CE 權重)")
        print(f"  輸出模型: {self.student_weights_path}")
        print("=" * 60)


def get_default_config() -> Config:
    return Config()
