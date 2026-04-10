"""
Step 2 (V3): 優化版模型訓練
============================
針對小資料集進行深度優化的訓練腳本

核心改進：
1. 使用所有切片片段進行訓練（而非每檔案只取 1 個）
2. 類別加權損失函數（解決標籤不平衡）
3. 焦點損失（Focal Loss）處理困難樣本
4. 資料增強提高泛化能力
5. 更完善的驗證和早停機制

執行方式：
    python 02_train_model_v3.py

可選參數：
    --resume  從上次中斷處繼續訓練
    --debug   除錯模式（使用較少資料）
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import torchaudio
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from config_v2 import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    HF_TOKEN,
    PRETRAINED_SEGMENTATION_MODEL,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEGMENT_DURATION,
    FREEZE_RATIO,
    EARLY_STOPPING_PATIENCE,
    USE_CLASS_WEIGHTS,
    USE_AUGMENTATION,
    USE_MIXED_PRECISION,
    ACCUMULATE_GRAD_BATCHES,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    SPEAKER_NAMES,
    TIME_MASK_PROB,
    TIME_MASK_MAX_LENGTH,
    VOLUME_CHANGE_PROB,
    VOLUME_CHANGE_RANGE,
    ADD_NOISE_PROB,
    NOISE_SNR_RANGE,
    LOG_INTERVAL,
    VAL_INTERVAL,
    create_directories
)


# ============== 資料增強 ==============

class AudioAugmentation:
    """音訊資料增強"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def time_mask(self, waveform: torch.Tensor, max_length: float = 0.5) -> torch.Tensor:
        """時間遮罩：隨機將一段時間設為 0"""
        if random.random() > TIME_MASK_PROB:
            return waveform
        
        length = waveform.shape[-1]
        mask_length = int(random.uniform(0.1, max_length) * self.sample_rate)
        mask_start = random.randint(0, max(0, length - mask_length))
        
        waveform = waveform.clone()
        waveform[..., mask_start:mask_start + mask_length] = 0
        return waveform
    
    def volume_change(self, waveform: torch.Tensor) -> torch.Tensor:
        """音量變化"""
        if random.random() > VOLUME_CHANGE_PROB:
            return waveform
        
        factor = random.uniform(*VOLUME_CHANGE_RANGE)
        return waveform * factor
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """加入高斯噪音"""
        if random.random() > ADD_NOISE_PROB:
            return waveform
        
        snr_db = random.uniform(*NOISE_SNR_RANGE)
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        
        return waveform + noise
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """應用所有增強"""
        waveform = self.time_mask(waveform)
        waveform = self.volume_change(waveform)
        waveform = self.add_noise(waveform)
        return waveform


# ============== 自訂資料集 ==============

class SegmentDataset(Dataset):
    """
    優化版資料集：使用預切片的片段
    
    每個樣本是一個 5 秒的音訊片段及其對應的標籤
    支援預載入音訊到記憶體以加速訓練
    """
    
    def __init__(
        self,
        segments: List[Dict],
        sample_rate: int = 16000,
        duration: float = SEGMENT_DURATION,
        augment: bool = False,
        num_speakers: int = 3,
        preload_audio: bool = True  # 新增：是否預載入音訊
    ):
        self.segments = segments
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(duration * sample_rate)
        self.num_speakers = num_speakers
        self.preload_audio = preload_audio
        
        self.augment = augment
        if augment:
            self.augmentation = AudioAugmentation(sample_rate)
        
        # 預先載入所有 RTTM 檔案
        self._load_annotations()
        
        # 預先載入所有音訊到記憶體
        if preload_audio:
            self._preload_audio_files()
    
    def _preload_audio_files(self):
        """預先載入所有音訊檔案到記憶體"""
        self.audio_cache = {}
        
        audio_files = set(seg['audio_path'] for seg in self.segments)
        
        print(f"  預載入 {len(audio_files)} 個音訊檔案到記憶體...")
        for audio_path in tqdm(audio_files, desc="  Loading audio"):
            waveform, sr = torchaudio.load(audio_path)
            
            # 重採樣到 16kHz
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 轉為單聲道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            self.audio_cache[audio_path] = waveform
        
        print(f"  ✓ 音訊載入完成")
    
    def _load_annotations(self):
        """預先載入所有標註"""
        self.annotations = {}
        
        rttm_files = set(seg['rttm_path'] for seg in self.segments)
        
        for rttm_path in rttm_files:
            annotations = []
            with open(rttm_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start = float(parts[3])
                        duration = float(parts[4])
                        speaker = parts[7]
                        annotations.append((start, start + duration, speaker))
            self.annotations[rttm_path] = annotations
    
    def __len__(self):
        return len(self.segments)
    
    def _create_labels(
        self,
        start_time: float,
        end_time: float,
        annotations: List[Tuple[float, float, str]],
        num_frames: int
    ) -> torch.Tensor:
        """
        從標註建立標籤
        
        Args:
            start_time: 片段開始時間
            end_time: 片段結束時間
            annotations: 標註列表
            num_frames: 輸出幀數
            
        Returns:
            形狀為 (num_frames, num_speakers) 的標籤
        """
        duration = end_time - start_time
        frame_duration = duration / num_frames
        
        # 說話人映射
        speaker_to_idx = {name: i for i, name in enumerate(SPEAKER_NAMES)}
        
        labels = torch.zeros(num_frames, self.num_speakers)
        
        for ann_start, ann_end, speaker in annotations:
            if speaker not in speaker_to_idx:
                continue
            
            speaker_idx = speaker_to_idx[speaker]
            if speaker_idx >= self.num_speakers:
                continue
            
            # 計算重疊區域
            overlap_start = max(start_time, ann_start)
            overlap_end = min(end_time, ann_end)
            
            if overlap_end > overlap_start:
                # 轉換為幀索引
                start_frame = int((overlap_start - start_time) / frame_duration)
                end_frame = int((overlap_end - start_time) / frame_duration)
                
                start_frame = max(0, min(start_frame, num_frames - 1))
                end_frame = max(0, min(end_frame, num_frames))
                
                labels[start_frame:end_frame, speaker_idx] = 1.0
        
        return labels
    
    def __getitem__(self, idx: int) -> Dict:
        segment = self.segments[idx]
        
        # 載入音訊片段
        audio_path = segment['audio_path']
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        # 使用預載入的音訊或從磁碟載入
        if self.preload_audio and audio_path in self.audio_cache:
            # 從快取取得音訊並切片
            full_waveform = self.audio_cache[audio_path]
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            waveform = full_waveform[:, start_sample:end_sample].clone()
        else:
            # 從磁碟載入
            info = torchaudio.info(audio_path)
            original_sr = info.sample_rate
            
            frame_offset = int(start_time * original_sr)
            num_frames_to_load = int((end_time - start_time) * original_sr)
            
            waveform, sr = torchaudio.load(
                audio_path,
                frame_offset=frame_offset,
                num_frames=num_frames_to_load
            )
            
            # 重採樣到 16kHz
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 轉為單聲道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        
        # 確保長度正確
        if waveform.shape[1] < self.num_samples:
            # 補零
            padding = self.num_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.num_samples:
            # 截斷
            waveform = waveform[:, :self.num_samples]
        
        # 資料增強
        if self.augment:
            waveform = self.augmentation(waveform)
        
        # 取得標註
        annotations = self.annotations.get(segment['rttm_path'], [])
        
        return {
            'waveform': waveform,
            'start_time': start_time,
            'end_time': end_time,
            'annotations': annotations,
            'segment_id': segment['segment_id'],
            'has_child_speech': segment.get('has_child_speech', False),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """自訂 collate function"""
    waveforms = torch.stack([item['waveform'] for item in batch])
    
    return {
        'waveforms': waveforms,
        'start_times': [item['start_time'] for item in batch],
        'end_times': [item['end_time'] for item in batch],
        'annotations': [item['annotations'] for item in batch],
        'segment_ids': [item['segment_id'] for item in batch],
        'has_child_speech': [item['has_child_speech'] for item in batch],
    }


# ============== 損失函數 ==============

class FocalLoss(nn.Module):
    """
    焦點損失（Focal Loss）
    
    對於困難樣本（預測不確定的樣本）給予更高的權重
    對於簡單樣本（預測很確定的樣本）降低權重
    
    這有助於模型專注於學習困難的情況，例如語音邊界
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, frames, speakers) 預測值（未經 sigmoid）
            targets: (batch, frames, speakers) 真實標籤
        """
        probs = torch.sigmoid(inputs)
        
        # 計算 BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 計算焦點權重
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # 應用類別權重
        if self.class_weights is not None:
            # class_weights shape: (speakers,)
            # 確保權重維度與目標匹配
            num_classes = targets.shape[-1]
            if self.class_weights.shape[0] != num_classes:
                # 如果權重數量不匹配，只取需要的部分或填充
                weights = self.class_weights[:num_classes] if self.class_weights.shape[0] > num_classes else self.class_weights
            else:
                weights = self.class_weights
            weight = weights.view(1, 1, -1).expand_as(targets)
            focal_loss = weight * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """
    加權 BCE 損失
    
    針對不同說話人給予不同權重
    """
    
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, frames, speakers)
            targets: (batch, frames, speakers)
        """
        # 計算每個類別的 BCE
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 應用權重
        weight = self.class_weights.view(1, 1, -1).expand_as(targets)
        weighted_bce = weight * bce
        
        return weighted_bce.mean()


# ============== 訓練邏輯 ==============

class Trainer:
    """訓練器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[Dict[str, float]] = None,
        device: str = "cuda",
        save_dir: Path = MODEL_DIR / "finetuned_segmentation_v2"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定類別權重
        if class_weights and USE_CLASS_WEIGHTS:
            weights = [class_weights.get(name, 1.0) for name in SPEAKER_NAMES]
            self.class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"類別權重: {dict(zip(SPEAKER_NAMES, weights))}")
        else:
            self.class_weights = None
        
        # 損失函數
        self.criterion = FocalLoss(
            alpha=0.25,
            gamma=2.0,
            class_weights=self.class_weights
        )
        
        # 優化器
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=0.01
        )
        
        # 學習率調度器
        total_steps = len(train_loader) * EPOCHS
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=LEARNING_RATE,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # 混合精度訓練
        self.scaler = GradScaler() if USE_MIXED_PRECISION and device == "cuda" else None
        
        # 訓練狀態
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 歷史記錄
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'child_recall': [],
            'tester_recall': [],
        }
    
    def train_epoch(self) -> float:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{EPOCHS} [Train]"
        )
        
        for batch_idx, batch in enumerate(pbar):
            waveforms = batch['waveforms'].to(self.device)
            
            # 前向傳播
            with autocast(enabled=USE_MIXED_PRECISION and self.device == "cuda"):
                outputs = self.model(waveforms)
                
                # 建立標籤
                batch_size = outputs.shape[0]
                num_frames = outputs.shape[1]
                # 只使用前 len(SPEAKER_NAMES) 個通道（我們的資料只有 3 類說話人）
                num_speakers = len(SPEAKER_NAMES)
                outputs = outputs[:, :, :num_speakers]  # 截取前 3 個通道
                
                targets = torch.zeros(batch_size, num_frames, num_speakers, device=self.device)
                
                for i in range(batch_size):
                    start_time = batch['start_times'][i]
                    end_time = batch['end_times'][i]
                    annotations = batch['annotations'][i]
                    
                    duration = end_time - start_time
                    frame_duration = duration / num_frames
                    
                    speaker_to_idx = {name: j for j, name in enumerate(SPEAKER_NAMES)}
                    
                    for ann_start, ann_end, speaker in annotations:
                        if speaker not in speaker_to_idx:
                            continue
                        speaker_idx = speaker_to_idx[speaker]
                        if speaker_idx >= num_speakers:
                            continue
                        
                        overlap_start = max(start_time, ann_start)
                        overlap_end = min(end_time, ann_end)
                        
                        if overlap_end > overlap_start:
                            start_frame = int((overlap_start - start_time) / frame_duration)
                            end_frame = int((overlap_end - start_time) / frame_duration)
                            
                            start_frame = max(0, min(start_frame, num_frames - 1))
                            end_frame = max(0, min(end_frame, num_frames))
                            
                            targets[i, start_frame:end_frame, speaker_idx] = 1.0
                
                loss = self.criterion(outputs, targets)
                loss = loss / ACCUMULATE_GRAD_BATCHES
            
            # 反向傳播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累積
            if (batch_idx + 1) % ACCUMULATE_GRAD_BATCHES == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUMULATE_GRAD_BATCHES
            num_batches += 1
            self.global_step += 1
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item() * ACCUMULATE_GRAD_BATCHES:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """驗證"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # 用於計算 recall
        speaker_tp = {name: 0 for name in SPEAKER_NAMES}
        speaker_fn = {name: 0 for name in SPEAKER_NAMES}
        speaker_fp = {name: 0 for name in SPEAKER_NAMES}
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{EPOCHS} [Val]"
        )
        
        for batch in pbar:
            waveforms = batch['waveforms'].to(self.device)
            
            with autocast(enabled=USE_MIXED_PRECISION and self.device == "cuda"):
                outputs = self.model(waveforms)
                
                # 建立標籤
                batch_size = outputs.shape[0]
                num_frames = outputs.shape[1]
                # 只使用前 len(SPEAKER_NAMES) 個通道
                num_speakers = len(SPEAKER_NAMES)
                outputs = outputs[:, :, :num_speakers]  # 截取前 3 個通道
                
                targets = torch.zeros(batch_size, num_frames, num_speakers, device=self.device)
                
                for i in range(batch_size):
                    start_time = batch['start_times'][i]
                    end_time = batch['end_times'][i]
                    annotations = batch['annotations'][i]
                    
                    duration = end_time - start_time
                    frame_duration = duration / num_frames
                    
                    speaker_to_idx = {name: j for j, name in enumerate(SPEAKER_NAMES)}
                    
                    for ann_start, ann_end, speaker in annotations:
                        if speaker not in speaker_to_idx:
                            continue
                        speaker_idx = speaker_to_idx[speaker]
                        if speaker_idx >= num_speakers:
                            continue
                        
                        overlap_start = max(start_time, ann_start)
                        overlap_end = min(end_time, ann_end)
                        
                        if overlap_end > overlap_start:
                            start_frame = int((overlap_start - start_time) / frame_duration)
                            end_frame = int((overlap_end - start_time) / frame_duration)
                            
                            start_frame = max(0, min(start_frame, num_frames - 1))
                            end_frame = max(0, min(end_frame, num_frames))
                            
                            targets[i, start_frame:end_frame, speaker_idx] = 1.0
                
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 計算 TP, FN, FP
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            for speaker_idx, speaker_name in enumerate(SPEAKER_NAMES):
                if speaker_idx >= num_speakers:
                    continue
                pred_speaker = preds[:, :, speaker_idx]
                target_speaker = targets[:, :, speaker_idx]
                
                tp = ((pred_speaker == 1) & (target_speaker == 1)).sum().item()
                fn = ((pred_speaker == 0) & (target_speaker == 1)).sum().item()
                fp = ((pred_speaker == 1) & (target_speaker == 0)).sum().item()
                
                speaker_tp[speaker_name] += tp
                speaker_fn[speaker_name] += fn
                speaker_fp[speaker_name] += fp
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 計算各說話人的 Precision, Recall, F1
        metrics = {}
        for speaker_name in SPEAKER_NAMES:
            tp = speaker_tp[speaker_name]
            fn = speaker_fn[speaker_name]
            fp = speaker_fp[speaker_name]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[speaker_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return avg_loss, metrics
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """儲存檢查點"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'history': self.history,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str = "checkpoint.pt"):
        """載入檢查點"""
        checkpoint_path = self.save_dir / filename
        if not checkpoint_path.exists():
            print(f"找不到檢查點: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        self.history = checkpoint['history']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"已從 epoch {self.current_epoch} 繼續訓練")
        return True
    
    def train(self, resume: bool = False):
        """完整訓練流程"""
        if resume:
            self.load_checkpoint()
        
        print(f"\n開始訓練 (共 {EPOCHS} epochs)")
        print(f"訓練樣本: {len(self.train_loader.dataset)}")
        print(f"驗證樣本: {len(self.val_loader.dataset)}")
        print("-" * 60)
        
        while self.current_epoch < EPOCHS:
            # 訓練
            train_loss = self.train_epoch()
            
            # 驗證
            val_loss, metrics = self.validate()
            
            # 記錄
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # 記錄各說話人的 recall
            child_recall = metrics.get('SPEAKER_01', {}).get('recall', 0)
            tester_recall = metrics.get('SPEAKER_00', {}).get('recall', 0)
            self.history['child_recall'].append(child_recall)
            self.history['tester_recall'].append(tester_recall)
            
            # 打印結果
            print(f"\nEpoch {self.current_epoch + 1}/{EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  測試者 - P: {metrics['SPEAKER_00']['precision']:.2%}, "
                  f"R: {metrics['SPEAKER_00']['recall']:.2%}, "
                  f"F1: {metrics['SPEAKER_00']['f1']:.2%}")
            print(f"  小孩 - P: {metrics['SPEAKER_01']['precision']:.2%}, "
                  f"R: {metrics['SPEAKER_01']['recall']:.2%}, "
                  f"F1: {metrics['SPEAKER_01']['f1']:.2%}")
            
            # 儲存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 儲存最佳模型
                torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics,
                }, self.save_dir / "best_model.pt")
                
                print(f"  ✓ 儲存最佳模型 (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  (無改善，耐心計數: {self.patience_counter}/{EARLY_STOPPING_PATIENCE})")
            
            # 定期儲存檢查點
            self.save_checkpoint()
            
            # 早停
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n早停：連續 {EARLY_STOPPING_PATIENCE} 個 epoch 無改善")
                break
            
            self.current_epoch += 1
        
        # 儲存最終模型
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
        }, self.save_dir / "last.pt")
        
        # 儲存訓練歷史
        with open(self.save_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n" + "=" * 60)
        print(f"✅ 訓練完成!")
        print(f"  最佳模型: {self.save_dir / 'best_model.pt'}")
        print(f"  最佳 Val Loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def main():
    """主程式"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='從上次中斷處繼續訓練')
    parser.add_argument('--debug', action='store_true', help='除錯模式')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 2 (V3): 優化版模型訓練")
    print("=" * 60)
    
    # 設定隨機種子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # 建立目錄
    create_directories()
    
    # 檢查裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 載入資料
    segments_json = PROCESSED_DATA_DIR / "segments_v2.json"
    if not segments_json.exists():
        print(f"❌ 找不到片段資料: {segments_json}")
        print("請先執行: python 01_prepare_dataset_v2.py")
        return
    
    with open(segments_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_segments = data['train']
    val_segments = data['val']
    
    # 除錯模式使用較少資料
    if args.debug:
        train_segments = train_segments[:100]
        val_segments = val_segments[:50]
        print(f"[除錯模式] 使用 {len(train_segments)} 訓練 / {len(val_segments)} 驗證")
    
    print(f"\n訓練片段: {len(train_segments)}")
    print(f"驗證片段: {len(val_segments)}")
    
    # 載入類別權重
    weights_json = PROCESSED_DATA_DIR / "class_weights.json"
    if weights_json.exists():
        with open(weights_json, 'r', encoding='utf-8') as f:
            class_weights = json.load(f)
        print(f"類別權重: {class_weights}")
    else:
        class_weights = None
    
    # 建立資料集
    train_dataset = SegmentDataset(
        train_segments,
        augment=USE_AUGMENTATION
    )
    val_dataset = SegmentDataset(
        val_segments,
        augment=False
    )
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    # 載入模型
    print(f"\n載入預訓練模型: {PRETRAINED_SEGMENTATION_MODEL}")
    from pyannote.audio import Model
    
    model = Model.from_pretrained(
        PRETRAINED_SEGMENTATION_MODEL,
        use_auth_token=HF_TOKEN
    )
    model = model.to(device)
    
    # 凍結部分層
    all_params = list(model.named_parameters())
    freeze_count = int(len(all_params) * FREEZE_RATIO)
    
    for name, param in all_params[:freeze_count]:
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可訓練參數: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    # 建立訓練器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device
    )
    
    # 開始訓練
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()
