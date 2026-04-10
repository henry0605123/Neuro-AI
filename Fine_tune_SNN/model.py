# -*- coding: utf-8 -*-
"""
ReLU 前饋 Student：Whisper + WeightedAvg（自 Teacher 複製，推論用）+ Conv/MLP ReLU 頭
訓練時僅更新 head + classifier；Encoder 與加權層凍結。
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperProcessor

from config import Config


class WeightedAverageLayer(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        stacked_states = torch.stack(hidden_states, dim=0)
        normalized_weights = F.softmax(self.layer_weights, dim=0)
        weights = normalized_weights.view(-1, 1, 1, 1)
        return (stacked_states * weights).sum(dim=0)


class ConvHead(nn.Module):
    """論文：3 層 Conv1d k=1 + ReLU + Dropout"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_layers: int = 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, hidden_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class MLPHeadReLU(nn.Module):
    """MLP 改 ReLU：Linear → ReLU → Dropout（無 GELU、預設無 LayerNorm，利於 SNN）"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.classifier(x)
        return x.transpose(1, 2)


class MedianFilter1D(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必須為奇數")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def filter_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, seq_len = predictions.shape
        filtered = predictions.clone()
        for i in range(seq_len):
            start = max(0, i - self.padding)
            end = min(seq_len, i + self.padding + 1)
            for b in range(batch_size):
                window = predictions[b, start:end]
                filtered[b, i] = torch.median(window)
        if squeeze_output:
            filtered = filtered.squeeze(0)
        return filtered


class WhisperKDStudentModel(nn.Module):
    """
    Student：與 Teacher 共用相同 Whisper + WeightedAvg 權重（凍結），
    下游為 ReLU 前饋頭（conv 或 mlp）。
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        print(f"建立 Student: Whisper + ReLU head ({config.student_head_type})")
        self.whisper = WhisperModel.from_pretrained(config.whisper_model_name)
        self.processor = WhisperProcessor.from_pretrained(config.whisper_model_name)
        self.whisper_config = self.whisper.config
        self.hidden_dim = self.whisper_config.d_model
        self.num_encoder_layers = self.whisper_config.encoder_layers + 1

        if config.freeze_encoder:
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False

        self.weighted_avg = WeightedAverageLayer(self.num_encoder_layers)

        if config.student_head_type == "conv":
            self.head = ConvHead(
                input_dim=self.hidden_dim,
                hidden_dim=config.conv_channels,
                dropout=config.dropout,
                num_layers=3,
            )
            clf_in = config.conv_channels
        elif config.student_head_type == "mlp":
            self.head = MLPHeadReLU(
                input_dim=self.hidden_dim,
                hidden_dims=list(config.mlp_hidden_dims),
                dropout=config.dropout,
            )
            clf_in = self.head.output_dim
        else:
            raise ValueError(f"未知 student_head_type: {config.student_head_type}")

        self.classifier = ClassifierHead(input_dim=clf_in, num_classes=config.num_classes)
        self.median_filter = MedianFilter1D(kernel_size=config.median_filter_size)

    def _prepare_features(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size = audio.shape[0]
        device = audio.device
        features_list = []
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            features = self.processor(
                audio_np,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt",
            ).input_features
            features_list.append(features.squeeze(0))
        return torch.stack(features_list).to(device)

    def encode_weighted(self, audio: torch.Tensor) -> torch.Tensor:
        """回傳加權融合後特徵 (B, T, D)，供 KD 單次編碼使用。"""
        input_features = self._prepare_features(audio)
        encoder_outputs = self.whisper.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )
        return self.weighted_avg(encoder_outputs.hidden_states)

    def forward_from_weighted(self, weighted_features: torch.Tensor) -> torch.Tensor:
        """僅通過 ReLU 頭 + 分類器（weighted_features 可為已 detach 的 tensor）"""
        h = self.head(weighted_features)
        return self.classifier(h)

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w = self.encode_weighted(audio)
        return self.forward_from_weighted(w)

    def predict(
        self,
        audio: torch.Tensor,
        apply_median_filter: bool = False,
        return_probs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            if apply_median_filter:
                predictions = self.median_filter.filter_predictions(predictions)
            if return_probs:
                return predictions, probs
            return predictions


def create_student_from_teacher(config: Config, teacher: nn.Module) -> WhisperKDStudentModel:
    """自 Teacher 複製 Whisper 與 WeightedAverage 權重；凍結之，只訓練 head + classifier。"""
    student = WhisperKDStudentModel(config)
    student.whisper.load_state_dict(teacher.whisper.state_dict())
    student.weighted_avg.load_state_dict(teacher.weighted_avg.state_dict())
    for p in student.whisper.parameters():
        p.requires_grad = False
    for p in student.weighted_avg.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"Student 可訓練參數: {trainable:,} / 總計 {total:,}")
    return student


def load_model(config: Config, checkpoint_path: str) -> WhisperKDStudentModel:
    """載入已儲存的 Student（含完整 state_dict）。"""
    ckpt = torch.load(checkpoint_path, map_location=config.device)
    if "config" in ckpt:
        saved = ckpt["config"]
        if "student_head_type" in saved:
            config.student_head_type = saved["student_head_type"]
        if "conv_channels" in saved:
            config.conv_channels = saved["conv_channels"]
        if "mlp_hidden_dims" in saved:
            config.mlp_hidden_dims = saved["mlp_hidden_dims"]
    model = WhisperKDStudentModel(config)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"Student 權重已載入: {checkpoint_path}")
    return model
