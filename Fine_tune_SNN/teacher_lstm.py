# -*- coding: utf-8 -*-
"""
Bi-LSTM Teacher（與 Fine_tune_LSTM 一致，供載入權重與 KD 前向）
"""

import os
import re
from typing import Dict, Optional, Tuple

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


class BiLSTMHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.output_dim = hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        return self.dropout(lstm_out)


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


class WhisperBiLSTMModel(nn.Module):
    """與 Fine_tune_LSTM.model.WhisperBiLSTMModel 結構一致"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        print(f"載入 Whisper (Teacher): {config.whisper_model_name}")
        self.whisper = WhisperModel.from_pretrained(config.whisper_model_name)
        self.processor = WhisperProcessor.from_pretrained(config.whisper_model_name)
        self.whisper_config = self.whisper.config
        self.hidden_dim = self.whisper_config.d_model
        self.num_encoder_layers = self.whisper_config.encoder_layers + 1

        if config.freeze_encoder:
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False

        self.weighted_avg = WeightedAverageLayer(self.num_encoder_layers)
        self.bilstm_head = BiLSTMHead(
            input_dim=self.hidden_dim,
            hidden_dim=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout,
        )
        self.classifier = ClassifierHead(
            input_dim=config.lstm_hidden_size * 2,
            num_classes=config.num_classes,
        )

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

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_features = self._prepare_features(audio)
        encoder_outputs = self.whisper.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = encoder_outputs.hidden_states
        weighted_features = self.weighted_avg(hidden_states)
        lstm_features = self.bilstm_head(weighted_features)
        return self.classifier(lstm_features)


def _infer_lstm_from_state_dict(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    由 checkpoint 權重推斷 Bi-LSTM 的 hidden_size（單向）與層數。
    PyTorch LSTM 的 weight_ih_l* 第一維為 4 * hidden_size。
    """
    key0 = "bilstm_head.lstm.weight_ih_l0"
    if key0 not in state:
        raise KeyError(f"checkpoint 缺少 {key0}，無法推斷 Teacher LSTM 結構")
    hidden = state[key0].shape[0] // 4
    layer_ids = []
    for k in state:
        m = re.match(r"bilstm_head\.lstm\.weight_ih_l(\d+)$", k)
        if m:
            layer_ids.append(int(m.group(1)))
    num_layers = max(layer_ids) + 1 if layer_ids else 1
    return hidden, num_layers


def _apply_teacher_lstm_config(config: Config, state: Dict[str, torch.Tensor]) -> None:
    """以權重形狀為準，覆寫 config 的 lstm_hidden_size / lstm_num_layers（與訓練 Teacher 時一致）。"""
    h_inf, nl_inf = _infer_lstm_from_state_dict(state)
    if config.lstm_hidden_size != h_inf or config.lstm_num_layers != nl_inf:
        print(
            f"  Teacher LSTM 由權重推斷: hidden_size={h_inf}（單向）, num_layers={nl_inf}"
            f"（CLI/預設為 {config.lstm_hidden_size}/{config.lstm_num_layers}，已自動對齊）"
        )
    config.lstm_hidden_size = h_inf
    config.lstm_num_layers = nl_inf


def load_teacher(config: Config, checkpoint_path: Optional[str] = None) -> WhisperBiLSTMModel:
    path = checkpoint_path or config.teacher_checkpoint
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"找不到 Teacher 權重: {path}")

    try:
        ckpt = torch.load(path, map_location=config.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=config.device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    _apply_teacher_lstm_config(config, state)

    model = WhisperBiLSTMModel(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Teacher 已載入: {path}")
    return model
