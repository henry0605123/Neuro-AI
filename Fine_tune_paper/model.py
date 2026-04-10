# -*- coding: utf-8 -*-
"""
論文復現：Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions
模型模組：Whisper Encoder + Weighted Average + Conv Head + Classifier

作者：AI Engineering Team
日期：2026-01-19

模型架構說明（參考論文 Section 3.2）：
==================================================

1. Speech Foundation Model (Whisper Encoder)
   - 使用預訓練的 Whisper Encoder 提取音訊特徵
   - Encoder 權重凍結，僅訓練下游網路

2. Weighted Average Layer
   - 對 Whisper Encoder 所有 Hidden Layers 的輸出進行加權平均
   - 權重為可學習參數

3. Conv Head (3 層 1D Convolutional Layers)
   - 每層輸出通道數: 256
   - 激活函數: ReLU
   - Dropout: 0.2

4. Classifier (1 層 1D Convolutional Layer)
   - 輸出大小: 4 (對應 Child, Adult, Overlap, Silence)
"""

import math
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig, WhisperProcessor

from config import Config


# ==================== Weighted Average Layer ====================

class WeightedAverageLayer(nn.Module):
    """
    可學習權重的加權平均層
    ========================
    
    對 Whisper Encoder 所有 Hidden Layers 的輸出進行加權平均。
    
    論文原文：
    "We first take the weighted average of all hidden layers from each 
    pre-trained speech foundation model, where the weights are learnable."
    
    實作方式：
    - 為每一層分配一個可學習的 scalar weight
    - 使用 softmax 將權重正規化
    - 計算加權平均
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


# ==================== Convolutional Head ====================

class ConvHead(nn.Module):
    """
    卷積頭部網路
    ============
    
    論文架構：
    - 3 層 1D Convolutional Layers
    - 每層輸出通道數: 256
    - 激活函數: ReLU
    - Dropout: 0.2
    
    注意：使用 1D 卷積處理時間序列，kernel_size=1 等同於對每個 frame 獨立處理
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_layers: int = 3
    ):
        """
        初始化卷積頭部
        
        Args:
            input_dim: 輸入維度（Whisper hidden dimension）
            hidden_dim: 隱藏層維度
            dropout: Dropout 機率
            num_layers: 卷積層數量
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            
            self.layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,  # 逐 frame 處理
                    stride=1,
                    padding=0
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入 tensor, shape: (batch, seq_len, hidden_dim)
            
        Returns:
            torch.Tensor: 輸出 tensor, shape: (batch, seq_len, hidden_dim)
        """
        # 轉換為 (batch, channels, seq_len) 以符合 Conv1d 格式
        x = x.transpose(1, 2)
        
        for layer in self.layers:
            x = layer(x)
        
        # 轉換回 (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        return x


# ==================== Classifier Head ====================

class ClassifierHead(nn.Module):
    """
    分類器頭部
    ==========
    
    論文架構：
    - 1 層 1D Convolutional Layer
    - 輸出大小: 3 (Child, Adult, Silence) - 移除 Overlap
    """
    
    def __init__(self, input_dim: int, num_classes: int = 3):
        """
        初始化分類器
        
        Args:
            input_dim: 輸入維度
            num_classes: 輸出類別數 (default: 3 for 3-class task)
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


# ==================== Whisper Diarization Model ====================

class WhisperDiarizationModel(nn.Module):
    """
    基於 Whisper 的語者分離模型
    ============================
    
    完整架構：
    1. Whisper Encoder (凍結)
    2. Weighted Average Layer (可學習)
    3. Conv Head: 3 層 1D Conv (可學習)
    4. Classifier: 1 層 1D Conv (可學習)
    
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
        
        # 載入預訓練的 Whisper 模型
        print(f"載入 Whisper 模型: {config.whisper_model_name}")
        self.whisper = WhisperModel.from_pretrained(config.whisper_model_name)
        
        # 取得 Whisper 的配置資訊
        self.whisper_config = self.whisper.config
        self.hidden_dim = self.whisper_config.d_model
        self.num_encoder_layers = self.whisper_config.encoder_layers + 1  # +1 for embedding layer
        
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  Encoder layers: {self.num_encoder_layers}")
        
        # 凍結 Whisper Encoder
        if config.freeze_encoder:
            print("  凍結 Encoder 權重")
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
        
        # 初始化加權平均層
        self.weighted_avg = WeightedAverageLayer(self.num_encoder_layers)
        
        # 初始化卷積頭部
        self.conv_head = ConvHead(
            input_dim=self.hidden_dim,
            hidden_dim=config.conv_channels,
            dropout=config.dropout,
            num_layers=3
        )
        
        # 初始化分類器
        self.classifier = ClassifierHead(
            input_dim=config.conv_channels,
            num_classes=config.num_classes
        )
        
        # 中值濾波器（僅推論時使用）
        self.median_filter = MedianFilter1D(kernel_size=config.median_filter_size)
        
        # 計算可訓練參數數量
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """印出可訓練參數數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n模型參數統計:")
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
        # 1. Whisper Encoder 特徵提取
        # 注意：Whisper 期望輸入為 mel spectrogram，需要先轉換
        encoder_outputs = self.whisper.encoder(
            self._prepare_input(audio),
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 取得所有 hidden states
        # hidden_states[0] 是 embedding 輸出
        # hidden_states[1:] 是各層 encoder 輸出
        hidden_states = encoder_outputs.hidden_states
        
        # 2. Weighted Average
        weighted_features = self.weighted_avg(hidden_states)
        
        # 3. Conv Head
        conv_features = self.conv_head(weighted_features)
        
        # 4. Classifier
        logits = self.classifier(conv_features)
        
        return logits
    
    def _prepare_input(self, audio: torch.Tensor) -> torch.Tensor:
        """
        準備 Whisper Encoder 的輸入
        
        Whisper 期望的輸入是 log-mel spectrogram，
        這裡使用 Whisper 內建的特徵提取器
        
        Args:
            audio: 原始音訊波形, shape: (batch, num_samples)
            
        Returns:
            torch.Tensor: Mel spectrogram features
        """
        import whisper
        
        # 使用 Whisper 的特徵提取
        batch_size = audio.shape[0]
        device = audio.device
        
        # 確保音訊長度正確（Whisper 期望 30 秒的音訊）
        # 對於我們的 20 秒窗口，需要填充
        target_length = 30 * self.config.sample_rate  # 30 秒
        current_length = audio.shape[1]
        
        if current_length < target_length:
            # 用零填充
            padding = torch.zeros(
                batch_size, 
                target_length - current_length,
                device=device,
                dtype=audio.dtype
            )
            audio = torch.cat([audio, padding], dim=1)
        elif current_length > target_length:
            # 截斷
            audio = audio[:, :target_length]
        
        # 計算 mel spectrogram
        mel_features = []
        for i in range(batch_size):
            # 轉換為 numpy 並計算 mel spectrogram
            audio_np = audio[i].cpu().numpy()
            mel = whisper.log_mel_spectrogram(audio_np, n_mels=self.whisper_config.num_mel_bins)
            mel_features.append(mel)
        
        # 堆疊並轉換為 tensor
        mel_features = torch.stack([torch.from_numpy(m) for m in mel_features]).to(device)
        
        return mel_features
    
    def get_layer_weights(self) -> Dict[str, float]:
        """
        取得各層的權重
        
        Returns:
            Dict[str, float]: 層名稱到權重的映射
        """
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
        """
        預測函式
        
        Args:
            audio: 音訊波形
            return_probs: 是否同時返回機率
            
        Returns:
            預測的類別索引，以及可選的機率分布
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            if return_probs:
                return predictions, probs
            return predictions


# ==================== 模型變體：使用 HuggingFace 內建特徵提取 ====================

class WhisperDiarizationModelV2(nn.Module):
    """
    基於 Whisper 的語者分離模型（V2 版本）
    =====================================
    
    與 V1 的差異：
    - 使用 Hugging Face 的 WhisperProcessor 進行音訊處理
    - 更好的記憶體效率
    - 支援動態批次處理
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
        
        # 凍結 Encoder
        if config.freeze_encoder:
            print("  凍結 Encoder 權重")
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
        
        # 下游網路
        self.weighted_avg = WeightedAverageLayer(self.num_encoder_layers)
        
        self.conv_head = ConvHead(
            input_dim=self.hidden_dim,
            hidden_dim=config.conv_channels,
            dropout=config.dropout,
            num_layers=3
        )
        
        self.classifier = ClassifierHead(
            input_dim=config.conv_channels,
            num_classes=config.num_classes
        )
        
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """印出可訓練參數數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n模型參數統計:")
        print(f"  總參數數量: {total_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")
        print(f"  凍結參數: {frozen_params:,}")
    
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
            torch.Tensor: Frame-level logits
        """
        # 準備輸入特徵
        input_features = self._prepare_features(audio)
        
        # Encoder 前向傳播
        encoder_outputs = self.whisper.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = encoder_outputs.hidden_states
        
        # 下游網路
        weighted_features = self.weighted_avg(hidden_states)
        conv_features = self.conv_head(weighted_features)
        logits = self.classifier(conv_features)
        
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


# ==================== 高效版本：直接使用 Whisper 音訊特徵 ====================

class WhisperDiarizationModelFast(nn.Module):
    """
    高效版 Whisper 語者分離模型
    ============================
    
    特色：
    - 使用 openai-whisper 套件的原生 mel spectrogram 計算
    - 批次處理優化
    - 針對 RTX 5070 Ti 優化
    """
    
    def __init__(self, config: Config):
        """
        初始化模型
        
        Args:
            config: 配置物件
        """
        super().__init__()
        
        self.config = config
        
        # 載入 Whisper 模型
        print(f"載入 Whisper 模型: {config.whisper_model_name}")
        self.whisper = WhisperModel.from_pretrained(
            config.whisper_model_name,
            torch_dtype=torch.float16 if config.use_amp else torch.float32
        )
        
        # 配置
        self.whisper_config = self.whisper.config
        self.hidden_dim = self.whisper_config.d_model
        self.num_encoder_layers = self.whisper_config.encoder_layers + 1
        
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  Encoder layers: {self.num_encoder_layers}")
        
        # 凍結 Encoder
        if config.freeze_encoder:
            print("  凍結 Encoder 權重")
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
        
        # 下游網路（使用標準精度，在 forward 中處理）
        self.weighted_avg = WeightedAverageLayer(self.num_encoder_layers)
        
        self.conv_head = ConvHead(
            input_dim=self.hidden_dim,
            hidden_dim=config.conv_channels,
            dropout=config.dropout,
            num_layers=3
        )
        
        self.classifier = ClassifierHead(
            input_dim=config.conv_channels,
            num_classes=config.num_classes
        )
        
        # 預計算 mel 濾波器組
        self._init_mel_filters()
        
        self._print_trainable_params()
    
    def _init_mel_filters(self):
        """初始化 mel 濾波器組"""
        import librosa
        
        n_fft = 400
        hop_length = 160
        n_mels = self.whisper_config.num_mel_bins
        
        # 創建 mel 濾波器組
        mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=8000.0
        )
        
        self.register_buffer(
            "mel_filters",
            torch.from_numpy(mel_basis).float()
        )
        
        # 用於 STFT 的窗函數
        self.register_buffer(
            "hann_window",
            torch.hann_window(n_fft)
        )
        
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def _compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        在 GPU 上計算 mel spectrogram
        
        Args:
            audio: 音訊波形, shape: (batch, num_samples)
            
        Returns:
            torch.Tensor: log-mel spectrogram
        """
        # 填充音訊到 30 秒
        target_length = 30 * self.config.sample_rate
        current_length = audio.shape[1]
        
        if current_length < target_length:
            padding = torch.zeros(
                audio.shape[0],
                target_length - current_length,
                device=audio.device,
                dtype=audio.dtype
            )
            audio = torch.cat([audio, padding], dim=1)
        elif current_length > target_length:
            audio = audio[:, :target_length]
        
        # STFT
        window = self.hann_window.to(audio.device)
        
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # 計算功率譜
        magnitudes = stft.abs() ** 2
        
        # 應用 mel 濾波器
        mel_filters = self.mel_filters.to(audio.device)
        mel_spec = torch.matmul(mel_filters, magnitudes)
        
        # Log scaling
        log_mel = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        
        return log_mel
    
    def _print_trainable_params(self):
        """印出可訓練參數數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n模型參數統計:")
        print(f"  總參數數量: {total_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")
    
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            audio: 音訊波形, shape: (batch, num_samples)
            
        Returns:
            torch.Tensor: Frame-level logits
        """
        # 計算 mel spectrogram
        input_features = self._compute_mel_spectrogram(audio)
        
        # Encoder
        encoder_outputs = self.whisper.encoder(
            input_features,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = encoder_outputs.hidden_states
        
        # 下游網路
        weighted_features = self.weighted_avg(hidden_states)
        conv_features = self.conv_head(weighted_features)
        logits = self.classifier(conv_features)
        
        return logits
    
    def get_layer_weights(self) -> Dict[str, float]:
        """取得各層的權重"""
        importance = self.weighted_avg.get_layer_importance()
        weights_dict = {}
        
        weights_dict["embedding"] = importance[0].item()
        for i in range(1, self.num_encoder_layers):
            weights_dict[f"encoder_layer_{i}"] = importance[i].item()
        
        return weights_dict


# ==================== 模型工廠函式 ====================

def create_model(config: Config, model_version: str = "v2") -> nn.Module:
    """
    建立模型
    
    Args:
        config: 配置物件
        model_version: 模型版本 ("v1", "v2", "fast")
        
    Returns:
        nn.Module: 語者分離模型
    """
    if model_version == "v1":
        model = WhisperDiarizationModel(config)
    elif model_version == "v2":
        model = WhisperDiarizationModelV2(config)
    elif model_version == "fast":
        model = WhisperDiarizationModelFast(config)
    else:
        raise ValueError(f"不支援的模型版本: {model_version}")
    
    return model.to(config.device)


def load_model(config: Config, checkpoint_path: str, model_version: str = "v2") -> nn.Module:
    """
    載入已訓練的模型
    
    Args:
        config: 配置物件
        checkpoint_path: 檢查點路徑
        model_version: 模型版本
        
    Returns:
        nn.Module: 載入權重的模型
    """
    model = create_model(config, model_version)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"模型權重已從 {checkpoint_path} 載入")
    
    return model


if __name__ == "__main__":
    # 測試模型
    print("測試模型建立...")
    
    config = Config()
    config.model_size = "tiny"  # 使用 tiny 進行測試
    
    # 測試 V2 模型
    print("\n建立 V2 模型:")
    model = create_model(config, model_version="v2")
    
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
