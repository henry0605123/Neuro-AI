# -*- coding: utf-8 -*-
"""
Whisper + Bi-LSTM 語者分離模型
模型模組：Whisper Encoder + Weighted Average + Bi-LSTM + Classifier

架構變更說明：
==================================================
原本： 3 層 Conv1D (Channels=256, ReLU, Dropout)
替換為： 2 層 Bi-LSTM
    - input_size: Whisper hidden dim (768 for medium, 1024 for large)
    - hidden_size: 256 (單向，雙向拼接後 512)
    - num_layers: 2
    - dropout: 0.2
    - batch_first: True

輸出層：接一個 Conv1D, kernel=1 將維度壓到 3 (類別數)

日期：2026-01-24
"""

import math
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig, WhisperProcessor
from peft import LoraConfig, get_peft_model

from config import Config


def _merge_lora_settings_from_checkpoint(config: Config, checkpoint: dict) -> None:
    """
    依 checkpoint 內儲存的 config 還原 LoRA 相關設定，避免載入時架構與權重鍵不一致。
    若 checkpoint 含 config 但無 use_lora（舊版儲存格式），視為未使用 LoRA。
    """
    ckpt_cfg = checkpoint.get("config")
    if not isinstance(ckpt_cfg, dict):
        return
    if "use_lora" in ckpt_cfg:
        config.use_lora = bool(ckpt_cfg["use_lora"])
    else:
        config.use_lora = False
    if "lora_rank" in ckpt_cfg:
        config.lora_rank = int(ckpt_cfg["lora_rank"])
    if "lora_alpha" in ckpt_cfg:
        config.lora_alpha = int(ckpt_cfg["lora_alpha"])
    if "lora_dropout" in ckpt_cfg:
        config.lora_dropout = float(ckpt_cfg["lora_dropout"])


# ==================== Weighted Average Layer ====================

class WeightedAverageLayer(nn.Module):
    """
    可學習權重的加權平均層
    ========================
    
    對 Whisper Encoder 所有 Hidden Layers 的輸出進行加權平均。
    """
    
    def __init__(self, num_layers: int):
        """
        初始化加權平均層
        
        Args:
            num_layers: Hidden layers 的數量（包含 embedding layer）
        """
        super().__init__()
        self.num_layers = num_layers
        
        # 可學習的權重參數（初始化為均勻分布）
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        計算加權平均
        
        Args:
            hidden_states: Tuple of tensors, 每個 tensor shape: (batch, seq_len, hidden_dim)
            
        Returns:
            torch.Tensor: 加權平均後的輸出, shape: (batch, seq_len, hidden_dim)
        """
        # 將所有 hidden states 堆疊成一個 tensor
        # Shape: (num_layers, batch, seq_len, hidden_dim)
        stacked_states = torch.stack(hidden_states, dim=0)
        
        # 使用 softmax 正規化權重
        normalized_weights = F.softmax(self.layer_weights, dim=0)
        
        # 重塑權重以便廣播
        # Shape: (num_layers, 1, 1, 1)
        weights = normalized_weights.view(-1, 1, 1, 1)
        
        # 計算加權平均
        # Shape: (batch, seq_len, hidden_dim)
        weighted_avg = (stacked_states * weights).sum(dim=0)
        
        return weighted_avg
    
    def get_layer_importance(self) -> torch.Tensor:
        """
        取得各層的重要性權重（正規化後）
        
        Returns:
            torch.Tensor: 正規化後的權重
        """
        with torch.no_grad():
            return F.softmax(self.layer_weights, dim=0)


# ==================== Bi-LSTM Head (替換 ConvHead) ====================

class BiLSTMHead(nn.Module):
    """
    雙向 LSTM 頭部網路
    ===================
    
    替換原本的 3 層 Conv1D，使用 2 層 Bi-LSTM 來捕捉時間序列的上下文資訊。
    
    架構：
    - 2 層 Bidirectional LSTM
    - 輸入維度: Whisper hidden dim (768/1024/1280)
    - 隱藏維度: 256（單向），512（雙向拼接）
    - Dropout: 0.2
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        初始化 Bi-LSTM 頭部
        
        Args:
            input_dim: 輸入維度（Whisper hidden dimension）
            hidden_dim: LSTM 隱藏層維度（單向）
            num_layers: LSTM 層數
            dropout: Dropout 機率（層間）
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = hidden_dim * 2  # 雙向拼接
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # 只在多層時使用 dropout
            batch_first=True,
            bidirectional=True
        )
        
        # Layer Normalization（穩定訓練）
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # 輸出 Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化 LSTM 權重（使用正交初始化）"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 設定 forget gate bias 為 1（幫助記憶長期依賴）
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入 tensor, shape: (batch, seq_len, hidden_dim)
            
        Returns:
            torch.Tensor: 輸出 tensor, shape: (batch, seq_len, hidden_dim * 2)
        """
        # LSTM 前向傳播
        # output shape: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Layer Normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        return lstm_out


# ==================== Classifier Head ====================

class ClassifierHead(nn.Module):
    """
    分類器頭部
    ==========
    
    使用 1D 卷積將 Bi-LSTM 輸出映射到類別數
    """
    
    def __init__(self, input_dim: int, num_classes: int = 3):
        """
        初始化分類器
        
        Args:
            input_dim: 輸入維度（Bi-LSTM 輸出維度 = hidden_dim * 2）
            num_classes: 輸出類別數
        """
        super().__init__()
        
        self.classifier = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入 tensor, shape: (batch, seq_len, hidden_dim)
            
        Returns:
            torch.Tensor: 輸出 logits, shape: (batch, seq_len, num_classes)
        """
        # 轉換為 (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # 分類
        x = self.classifier(x)
        
        # 轉換回 (batch, seq_len, num_classes)
        x = x.transpose(1, 2)
        
        return x


# ==================== Median Filter ====================

class MedianFilter1D(nn.Module):
    """
    1D 中值濾波器
    =============
    
    用於平滑時間序列上的預測結果，去除短暂的錯誤預測。
    只在推論時使用，不影響訓練。
    """
    
    def __init__(self, kernel_size: int = 5):
        """
        初始化中值濾波器
        
        Args:
            kernel_size: 濾波器核心大小（必須為奇數）
        """
        super().__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必須為奇數")
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播（單純回傳輸入，不做濾波）
        
        中值濾波不可微分，因此訓練時不應用。
        使用 filter_predictions() 在推論時應用。
        """
        return x
    
    @torch.no_grad()
    def filter_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        對預測結果應用中值濾波
        
        Args:
            predictions: 預測標籤, shape: (batch, seq_len) 或 (seq_len,)
            
        Returns:
            torch.Tensor: 濾波後的預測
        """
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = predictions.shape
        filtered = predictions.clone()
        
        # 對每個時間點應用中值濾波
        for i in range(seq_len):
            start = max(0, i - self.padding)
            end = min(seq_len, i + self.padding + 1)
            
            for b in range(batch_size):
                window = predictions[b, start:end]
                # 取中值
                median_val = torch.median(window)
                filtered[b, i] = median_val
        
        if squeeze_output:
            filtered = filtered.squeeze(0)
        
        return filtered


# ==================== Whisper + Bi-LSTM Model ====================

class WhisperBiLSTMModel(nn.Module):
    """
    基於 Whisper + Bi-LSTM 的語者分離模型
    ======================================
    
    完整架構：
    1. Whisper Encoder (凍結)
    2. Weighted Average Layer (可學習)
    3. Bi-LSTM Head: 2 層 Bidirectional LSTM (可學習)
    4. Classifier: 1 層 Conv1D (可學習)
    
    輸入：原始音訊波形
    輸出：Frame-level 語者分類 logits
    """
    
    def __init__(self, config: Config):
        """
        初始化模型
        
        Args:
            config: 配置物件
        """
        super().__init__()
        
        self.config = config
        
        # 載入 Whisper 模型和處理器
        print(f"載入 Whisper 模型: {config.whisper_model_name}")
        self.whisper = WhisperModel.from_pretrained(config.whisper_model_name)
        self.processor = WhisperProcessor.from_pretrained(config.whisper_model_name)
        
        # 取得配置
        self.whisper_config = self.whisper.config
        self.hidden_dim = self.whisper_config.d_model
        self.num_encoder_layers = self.whisper_config.encoder_layers + 1
        
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  Encoder layers: {self.num_encoder_layers}")
        
        # Encoder：LoRA（FFN 的 fc1 / fc2）或凍結／全量微調
        if getattr(config, "use_lora", False):
            print(
                "  Encoder：套用 LoRA（target_modules: fc1, fc2；基底權重由 PEFT 凍結）"
            )
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["fc1", "fc2"],
                bias="none",
            )
            self.whisper.encoder = get_peft_model(self.whisper.encoder, lora_config)
        elif config.freeze_encoder:
            print("  凍結 Encoder 權重")
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
        else:
            print("  Encoder 權重可訓練（未使用 LoRA）")
        
        # 下游網路
        self.weighted_avg = WeightedAverageLayer(self.num_encoder_layers)
        
        # Bi-LSTM Head（替換 ConvHead）
        self.bilstm_head = BiLSTMHead(
            input_dim=self.hidden_dim,
            hidden_dim=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout
        )
        
        # 分類器
        # 注意：輸入維度是 lstm_hidden_size * 2（雙向）
        self.classifier = ClassifierHead(
            input_dim=config.lstm_hidden_size * 2,
            num_classes=config.num_classes
        )
        
        # 中值濾波器（僅推論時使用）
        self.median_filter = MedianFilter1D(kernel_size=config.median_filter_size)
        
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """印出可訓練參數數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n模型參數統計 (Bi-LSTM 版本):")
        print(f"  總參數數量: {total_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")
        print(f"  凍結參數: {frozen_params:,}")
        print(f"  可訓練比例: {trainable_params/total_params*100:.2f}%")
    
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            audio: 音訊波形, shape: (batch, num_samples)
            attention_mask: 注意力遮罩（可選）
            
        Returns:
            torch.Tensor: Frame-level logits, shape: (batch, num_frames, num_classes)
        """
        # 1. 準備輸入特徵
        input_features = self._prepare_features(audio)
        
        # 2. Whisper Encoder
        encoder_outputs = self.whisper.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = encoder_outputs.hidden_states
        
        # 3. Weighted Average
        weighted_features = self.weighted_avg(hidden_states)
        
        # 4. Bi-LSTM Head（替換原本的 Conv Head）
        lstm_features = self.bilstm_head(weighted_features)
        
        # 5. Classifier
        logits = self.classifier(lstm_features)
        
        return logits
    
    def _prepare_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        使用 WhisperProcessor 準備輸入特徵
        
        Args:
            audio: 原始音訊波形
            
        Returns:
            torch.Tensor: Mel spectrogram features
        """
        batch_size = audio.shape[0]
        device = audio.device
        
        # 處理每個樣本
        features_list = []
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            
            # 使用 processor 提取特徵
            features = self.processor(
                audio_np,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            ).input_features
            
            features_list.append(features.squeeze(0))
        
        # 堆疊成批次
        input_features = torch.stack(features_list).to(device)
        
        return input_features
    
    def get_layer_weights(self) -> Dict[str, float]:
        """取得各層的權重"""
        importance = self.weighted_avg.get_layer_importance()
        weights_dict = {}
        
        weights_dict["embedding"] = importance[0].item()
        for i in range(1, self.num_encoder_layers):
            weights_dict[f"encoder_layer_{i}"] = importance[i].item()
        
        return weights_dict
    
    def predict(
        self,
        audio: torch.Tensor,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """預測函式"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            if return_probs:
                return predictions, probs
            return predictions


# ==================== 模型工廠函式 ====================

def create_model(config: Config) -> nn.Module:
    """
    建立 Whisper + Bi-LSTM 模型
    
    Args:
        config: 配置物件
        
    Returns:
        nn.Module: 語者分離模型
    """
    model = WhisperBiLSTMModel(config)
    return model.to(config.device)


def load_model(config: Config, checkpoint_path: str) -> nn.Module:
    """
    載入已訓練的模型
    
    Args:
        config: 配置物件
        checkpoint_path: 檢查點路徑
        
    Returns:
        nn.Module: 載入權重的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    if isinstance(checkpoint, dict):
        _merge_lora_settings_from_checkpoint(config, checkpoint)

    model = create_model(config)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith("whisper.") for k in checkpoint):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat is None:
        missing, unexpected = [], []
    else:
        missing = list(incompat.missing_keys)
        unexpected = list(incompat.unexpected_keys)

    if missing or unexpected:
        print(f"⚠️ 載入 state_dict 時未完全對齊（strict=False）")
        if missing:
            preview = missing[:8]
            print(f"   缺失鍵 ({len(missing)}): {preview}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            preview = unexpected[:8]
            print(f"   多餘鍵 ({len(unexpected)}): {preview}{' ...' if len(unexpected) > 8 else ''}")
    else:
        print(f"模型權重已從 {checkpoint_path} 載入（鍵完全對齊）")
        return model

    print(f"模型權重已從 {checkpoint_path} 載入（請確認 LoRA / 架構設定與訓練時一致）")
    return model


if __name__ == "__main__":
    # 測試模型
    print("測試 Whisper + Bi-LSTM 模型...")
    
    config = Config()
    config.model_size = "tiny"  # 使用 tiny 進行測試
    
    # 建立模型
    print("\n建立 Bi-LSTM 模型:")
    model = create_model(config)
    
    # 測試前向傳播
    print("\n測試前向傳播:")
    batch_size = 2
    num_samples = int(config.window_size * config.sample_rate)
    
    dummy_audio = torch.randn(batch_size, num_samples).to(config.device)
    
    with torch.no_grad():
        logits = model(dummy_audio)
        print(f"輸入形狀: {dummy_audio.shape}")
        print(f"輸出形狀: {logits.shape}")
        
        # 預測
        predictions = model.predict(dummy_audio)
        print(f"預測形狀: {predictions.shape}")
    
    # 顯示各層權重
    print("\n各層權重:")
    weights = model.get_layer_weights()
    for layer_name, weight in weights.items():
        print(f"  {layer_name}: {weight:.4f}")
    
    # 顯示 Bi-LSTM 結構
    print("\nBi-LSTM Head 結構:")
    print(model.bilstm_head)
