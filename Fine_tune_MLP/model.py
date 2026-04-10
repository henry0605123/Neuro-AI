# -*- coding: utf-8 -*-
"""
Whisper + MLP + Median Filter 語者分離模型
模型模組：Whisper Encoder + Weighted Average + MLP + Median Filter + Classifier

架構變更說明：
==================================================
原本（LSTM版本）： 2 層 Bi-LSTM (hidden=256)
替換為： MLP + Median Filter
    - MLP: 多層全連接網路 (512 → 256)
    - Median Filter: 時間軸上的中值濾波，平滑預測結果

輸出層：接一個 Conv1D, kernel=1 將維度壓到 3 (類別數)

日期：2026-01-26
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


# ==================== MLP Head (替換 Bi-LSTM) ====================

class MLPHead(nn.Module):
    """
    多層感知器頭部網路
    ===================
    
    替換 Bi-LSTM，使用多層全連接網路進行特徵轉換。
    
    架構：
    - 多層 Linear + LayerNorm + ReLU + Dropout
    - 輸入維度: Whisper hidden dim (768/1024/1280)
    - 隱藏維度: [512, 256] (可配置)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2
    ):
        """
        初始化 MLP 頭部
        
        Args:
            input_dim: 輸入維度（Whisper hidden dimension）
            hidden_dims: 隱藏層維度列表
            dropout: Dropout 機率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # 建立 MLP 層
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # 全連接層
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Layer Normalization
            layers.append(nn.LayerNorm(hidden_dim))
            
            # 激活函數
            layers.append(nn.GELU())  # 使用 GELU 替代 ReLU，更適合 Transformer 架構
            
            # Dropout
            layers.append(nn.Dropout(p=dropout))
            
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入 tensor, shape: (batch, seq_len, hidden_dim)
            
        Returns:
            torch.Tensor: 輸出 tensor, shape: (batch, seq_len, output_dim)
        """
        # MLP 可以直接處理 (batch, seq_len, features) 形狀
        # PyTorch Linear 層會自動處理最後一個維度
        return self.mlp(x)


# ==================== Median Filter ====================

class MedianFilter1D(nn.Module):
    """
    一維中值濾波器
    ===============
    
    用於平滑時間序列預測，減少抖動和噪音。
    使用滑動窗口在時間軸上進行中值濾波。
    
    注意：這個操作在訓練時可能不可微分，
    因此建議只在推論時使用，或使用可微分的近似方法。
    """
    
    def __init__(self, kernel_size: int = 5):
        """
        初始化中值濾波器
        
        Args:
            kernel_size: 濾波器核心大小（必須為奇數）
        """
        super().__init__()
        
        # 確保 kernel_size 為奇數
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        應用中值濾波
        
        Args:
            x: 輸入 tensor, shape: (batch, seq_len) 或 (batch, seq_len, features)
            
        Returns:
            torch.Tensor: 濾波後的 tensor，形狀與輸入相同
        """
        # 如果輸入是 3D，我們對每個特徵通道分別進行濾波
        if x.dim() == 3:
            batch, seq_len, features = x.shape
            
            # 對預測類別進行濾波（argmax 後的結果）
            # 這裡我們假設輸入是 logits，先取 argmax
            predictions = torch.argmax(x, dim=-1)  # (batch, seq_len)
            
            # 應用中值濾波
            filtered = self._apply_median_filter(predictions)
            
            # 將濾波後的類別轉換回 one-hot 形式
            filtered_onehot = F.one_hot(filtered, num_classes=features).float()
            
            return filtered_onehot
        else:
            # 2D 輸入直接濾波
            return self._apply_median_filter(x)
    
    def _apply_median_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        應用中值濾波到 2D tensor
        
        Args:
            x: 輸入 tensor, shape: (batch, seq_len)
            
        Returns:
            torch.Tensor: 濾波後的 tensor
        """
        batch, seq_len = x.shape
        device = x.device
        
        # 使用 unfold 建立滑動窗口
        # 先進行 padding
        x_padded = F.pad(x.float(), (self.padding, self.padding), mode='replicate')
        
        # Unfold: (batch, seq_len, kernel_size)
        x_unfolded = x_padded.unfold(1, self.kernel_size, 1)
        
        # 計算中值
        median_values = x_unfolded.median(dim=-1).values
        
        return median_values.long()
    
    def filter_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        專門用於濾波預測結果的方法
        
        Args:
            predictions: 預測標籤, shape: (batch, seq_len) 或 (seq_len,)
            
        Returns:
            torch.Tensor: 濾波後的預測
        """
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
            result = self._apply_median_filter(predictions)
            return result.squeeze(0)
        return self._apply_median_filter(predictions)


class DifferentiableMedianFilter(nn.Module):
    """
    可微分的中值濾波近似
    =====================
    
    使用軟化的排序和選擇操作來近似中值濾波，
    使其在訓練時可以進行反向傳播。
    
    基於加權平均的軟中值計算。
    """
    
    def __init__(self, kernel_size: int = 5, temperature: float = 0.1):
        """
        初始化可微分中值濾波器
        
        Args:
            kernel_size: 濾波器核心大小（必須為奇數）
            temperature: 軟化溫度參數（越小越接近真實中值）
        """
        super().__init__()
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.temperature = temperature
        
        # 使用 1D 卷積實現平滑（作為中值的近似）
        self.smoother = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False
        )
        
        # 初始化為均勻權重（類似於均值濾波）
        nn.init.constant_(self.smoother.weight, 1.0 / kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        應用可微分的平滑濾波
        
        Args:
            x: 輸入 logits, shape: (batch, seq_len, num_classes)
            
        Returns:
            torch.Tensor: 濾波後的 logits
        """
        batch, seq_len, num_classes = x.shape
        
        # 對每個類別的 logits 分別進行平滑
        x_transposed = x.transpose(1, 2)  # (batch, num_classes, seq_len)
        
        # 使用 1D 卷積進行平滑
        smoothed = []
        for c in range(num_classes):
            channel = x_transposed[:, c:c+1, :]  # (batch, 1, seq_len)
            smoothed_channel = self.smoother(channel)
            smoothed.append(smoothed_channel)
        
        smoothed = torch.cat(smoothed, dim=1)  # (batch, num_classes, seq_len)
        smoothed = smoothed.transpose(1, 2)  # (batch, seq_len, num_classes)
        
        return smoothed


# ==================== Classifier Head ====================

class ClassifierHead(nn.Module):
    """
    分類器頭部
    ==========
    
    使用 1D 卷積將 MLP 輸出映射到類別數
    """
    
    def __init__(self, input_dim: int, num_classes: int = 3):
        """
        初始化分類器
        
        Args:
            input_dim: 輸入維度（MLP 輸出維度）
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


# ==================== Whisper + MLP + Median Filter Model ====================

class WhisperMLPModel(nn.Module):
    """
    基於 Whisper + MLP + Median Filter 的語者分離模型
    ==================================================
    
    完整架構：
    1. Whisper Encoder (凍結)
    2. Weighted Average Layer (可學習)
    3. MLP Head: 多層全連接網路 (可學習)
    4. Classifier: 1 層 Conv1D (可學習)
    5. Median Filter: 時間軸平滑（僅推論時使用）
    
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
        
        # 凍結 Encoder
        if config.freeze_encoder:
            print("  凍結 Encoder 權重")
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
        
        # 下游網路
        self.weighted_avg = WeightedAverageLayer(self.num_encoder_layers)
        
        # MLP Head（替換 Bi-LSTM）
        self.mlp_head = MLPHead(
            input_dim=self.hidden_dim,
            hidden_dims=config.mlp_hidden_dims,
            dropout=config.dropout
        )
        
        # 分類器
        self.classifier = ClassifierHead(
            input_dim=config.mlp_hidden_dims[-1],
            num_classes=config.num_classes
        )
        
        # Median Filter（用於推論時的平滑）
        self.median_filter = MedianFilter1D(kernel_size=config.median_filter_size)
        
        # 可微分的平滑濾波（可選，用於訓練時）
        self.differentiable_filter = DifferentiableMedianFilter(
            kernel_size=config.median_filter_size
        )
        
        self.use_median_in_training = config.use_median_filter_in_training
        
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """印出可訓練參數數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n模型參數統計 (MLP + Median Filter 版本):")
        print(f"  總參數數量: {total_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")
        print(f"  凍結參數: {frozen_params:,}")
        print(f"  可訓練比例: {trainable_params/total_params*100:.2f}%")
    
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        apply_smoothing: bool = False
    ) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            audio: 音訊波形, shape: (batch, num_samples)
            attention_mask: 注意力遮罩（可選）
            apply_smoothing: 是否應用平滑濾波（訓練時使用可微分版本）
            
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
        
        # 4. MLP Head（替換原本的 Bi-LSTM）
        mlp_features = self.mlp_head(weighted_features)
        
        # 5. Classifier
        logits = self.classifier(mlp_features)
        
        # 6. 可選的平滑濾波（訓練時使用可微分版本）
        if apply_smoothing and self.use_median_in_training:
            logits = self.differentiable_filter(logits)
        
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
        apply_median_filter: bool = True,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        預測函式（包含可選的中值濾波後處理）
        
        Args:
            audio: 音訊波形
            apply_median_filter: 是否應用中值濾波平滑預測
            return_probs: 是否返回機率
            
        Returns:
            預測結果，可選返回機率
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # 應用中值濾波
            if apply_median_filter:
                predictions = self.median_filter.filter_predictions(predictions)
            
            if return_probs:
                return predictions, probs
            return predictions


# ==================== 模型工廠函式 ====================

def create_model(config: Config) -> nn.Module:
    """
    建立 Whisper + MLP + Median Filter 模型
    
    Args:
        config: 配置物件
        
    Returns:
        nn.Module: 語者分離模型
    """
    model = WhisperMLPModel(config)
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
    # 先載入 checkpoint 以獲取儲存的配置
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # 從 checkpoint 中讀取訓練時的配置
    if "config" in checkpoint:
        saved_config = checkpoint["config"]
        
        # 更新 config 中的 MLP 和 Median Filter 相關設定
        if "mlp_hidden_dims" in saved_config:
            config.mlp_hidden_dims = saved_config["mlp_hidden_dims"]
            print(f"  從 checkpoint 載入 mlp_hidden_dims: {config.mlp_hidden_dims}")
        
        if "median_filter_size" in saved_config:
            config.median_filter_size = saved_config["median_filter_size"]
            print(f"  從 checkpoint 載入 median_filter_size: {config.median_filter_size}")
    
    # 使用更新後的 config 建立模型
    model = create_model(config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"模型權重已從 {checkpoint_path} 載入")
    
    return model


if __name__ == "__main__":
    # 測試模型
    print("測試 Whisper + MLP + Median Filter 模型...")
    
    config = Config()
    config.model_size = "tiny"  # 使用 tiny 進行測試
    
    # 建立模型
    print("\n建立 MLP 模型:")
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
        
        # 預測（不使用中值濾波）
        predictions_raw = model.predict(dummy_audio, apply_median_filter=False)
        print(f"預測形狀 (無濾波): {predictions_raw.shape}")
        
        # 預測（使用中值濾波）
        predictions_filtered = model.predict(dummy_audio, apply_median_filter=True)
        print(f"預測形狀 (有濾波): {predictions_filtered.shape}")
        
        # 比較濾波前後的差異
        diff = (predictions_raw != predictions_filtered).sum().item()
        total = predictions_raw.numel()
        print(f"濾波改變的 frame 數: {diff}/{total} ({diff/total*100:.2f}%)")
    
    # 顯示各層權重
    print("\n各層權重:")
    weights = model.get_layer_weights()
    for layer_name, weight in weights.items():
        print(f"  {layer_name}: {weight:.4f}")
    
    # 顯示 MLP 結構
    print("\nMLP Head 結構:")
    print(model.mlp_head)
    
    # 顯示 Median Filter 設定
    print(f"\nMedian Filter 核心大小: {model.median_filter.kernel_size}")
