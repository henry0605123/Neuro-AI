# -*- coding: utf-8 -*-
"""
Whisper + Bi-LSTM 語者分離模型
訓練模組：訓練迴圈、AMP 混合精度、Early Stopping、Loss/Accuracy 曲線

架構：Whisper Encoder + Weighted Average + 2 層 Bi-LSTM + Classifier
日期：2026-01-24
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

# 設定中文字體支援
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import Config
from model import create_model, WhisperBiLSTMModel


# ==================== Early Stopping ====================

class EarlyStopping:
    """
    Early Stopping 機制
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0001,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ==================== 訓練記錄器 ====================

class TrainingLogger:
    """訓練記錄器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
            "epoch_time": [],
        }
        
        self.start_time = None
    
    def start_training(self):
        self.start_time = time.time()
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float
    ):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["learning_rate"].append(lr)
        self.history["epoch_time"].append(epoch_time)
        
        self._save_history()
    
    def _save_history(self):
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
    
    def plot_curves(self, save_path: Optional[str] = None):
        if not self.history["train_loss"]:
            print("沒有訓練記錄可繪製")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # 1. Loss 曲線
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history["train_loss"], 'b-', label='訓練損失', linewidth=2)
        ax1.plot(epochs, self.history["val_loss"], 'r-', label='驗證損失', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('損失曲線 (Bi-LSTM)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy 曲線
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history["train_acc"], 'b-', label='訓練準確率', linewidth=2)
        ax2.plot(epochs, self.history["val_acc"], 'r-', label='驗證準確率', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('準確率曲線 (Bi-LSTM)', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning Rate 曲線
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history["learning_rate"], 'g-', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('學習率變化', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Epoch Time 曲線
        ax4 = axes[1, 1]
        ax4.bar(epochs, self.history["epoch_time"], color='purple', alpha=0.7)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Time (seconds)', fontsize=12)
        ax4.set_title('每 Epoch 訓練時間', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.log_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"訓練曲線已儲存至: {save_path}")
    
    def get_summary(self) -> Dict:
        if not self.history["train_loss"]:
            return {}
        
        best_val_idx = np.argmin(self.history["val_loss"])
        
        return {
            "total_epochs": len(self.history["train_loss"]),
            "best_epoch": best_val_idx + 1,
            "best_val_loss": self.history["val_loss"][best_val_idx],
            "best_val_acc": self.history["val_acc"][best_val_idx],
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "total_time": sum(self.history["epoch_time"]),
            "avg_epoch_time": np.mean(self.history["epoch_time"]),
        }


# ==================== 訓練器 ====================

class Trainer:
    """模型訓練器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        
        # 損失函數
        self.criterion = nn.CrossEntropyLoss()
        
        # 優化器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 學習率排程器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        # AMP 混合精度訓練
        self.scaler = GradScaler() if config.use_amp else None
        
        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode='min'
        )
        
        # 記錄器
        self.logger = TrainingLogger(config.log_dir)
        
        # 最佳模型狀態
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """訓練一個 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc="訓練中",
            leave=False,
            ncols=100
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast():
                    logits = self.model(audio)
                    
                    seq_len = min(logits.shape[1], labels.shape[1])
                    logits = logits[:, :seq_len, :]
                    labels = labels[:, :seq_len]
                    
                    loss = self.criterion(
                        logits.contiguous().view(-1, self.config.num_classes),
                        labels.contiguous().view(-1)
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(audio)
                
                seq_len = min(logits.shape[1], labels.shape[1])
                logits = logits[:, :seq_len, :]
                labels = labels[:, :seq_len]
                
                loss = self.criterion(
                    logits.contiguous().view(-1, self.config.num_classes),
                    labels.contiguous().view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.numel()
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*total_correct/total_samples:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * total_correct / total_samples
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """驗證模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="驗證中", leave=False, ncols=100):
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            if self.config.use_amp:
                with autocast():
                    logits = self.model(audio)
            else:
                logits = self.model(audio)
            
            seq_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :seq_len, :]
            labels = labels[:, :seq_len]
            
            loss = self.criterion(
                logits.contiguous().view(-1, self.config.num_classes),
                labels.contiguous().view(-1)
            )
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.numel()
        
        avg_loss = total_loss / max(1, len(self.val_loader))
        accuracy = 100.0 * total_correct / max(1, total_samples)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """儲存檢查點"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "model_size": self.config.model_size,
                "window_size": self.config.window_size,
                "num_classes": self.config.num_classes,
                "lstm_hidden_size": self.config.lstm_hidden_size,
                "lstm_num_layers": self.config.lstm_num_layers,
                "use_lora": self.config.use_lora,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "freeze_whisper_decoder": self.config.freeze_whisper_decoder,
            },
            "history": self.logger.history,
        }
        
        checkpoint_path = Path(self.config.model_save_dir) / "checkpoint_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = Path(self.config.model_save_dir) / f"whisper_{self.config.model_size}_bilstm_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  💾 最佳模型已儲存: {best_path}")
    
    def train(self) -> Dict:
        """執行完整訓練流程"""
        print("\n" + "=" * 60)
        print("開始訓練 (Whisper + Bi-LSTM)")
        print("=" * 60)
        
        self.logger.start_training()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            print(f"\n📌 Epoch {epoch}/{self.config.epochs}")
            print("-" * 40)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            self.logger.log_epoch(
                epoch, train_loss, train_acc,
                val_loss, val_acc, current_lr, epoch_time
            )
            
            print(f"  訓練損失: {train_loss:.4f} | 訓練準確率: {train_acc:.2f}%")
            print(f"  驗證損失: {val_loss:.4f} | 驗證準確率: {val_acc:.2f}%")
            print(f"  學習率: {current_lr:.2e} | 時間: {epoch_time:.1f}s")
            
            self.scheduler.step(val_loss)
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            self.save_checkpoint(epoch, is_best)
            
            if self.early_stopping(val_loss, epoch):
                print(f"\n⚠️ Early Stopping 觸發於 Epoch {epoch}")
                print(f"   最佳 Epoch: {self.early_stopping.best_epoch}")
                break
        
        self.logger.plot_curves()
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("\n✅ 已載入最佳模型權重")
        
        summary = self.logger.get_summary()
        
        print("\n" + "=" * 60)
        print("訓練完成 (Bi-LSTM)")
        print("=" * 60)
        print(f"  總 Epochs: {summary.get('total_epochs', 0)}")
        print(f"  最佳 Epoch: {summary.get('best_epoch', 0)}")
        print(f"  最佳驗證損失: {summary.get('best_val_loss', 0):.4f}")
        print(f"  最佳驗證準確率: {summary.get('best_val_acc', 0):.2f}%")
        print(f"  總訓練時間: {summary.get('total_time', 0)/60:.1f} 分鐘")
        
        return summary


# ==================== 訓練輔助函式 ====================

def calculate_class_weights(train_loader: DataLoader, num_classes: int = 3, child_boost: float = 1.5) -> torch.Tensor:
    """計算類別權重"""
    class_counts = torch.zeros(num_classes)
    
    for batch in tqdm(train_loader, desc="計算類別權重"):
        labels = batch["labels"]
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum().item()
    
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-6)
    weights[0] *= child_boost
    weights = weights / weights.sum() * num_classes
    
    print("\n類別權重 (3 分類):")
    class_names = ["Child", "Adult", "Silence"]
    for i, (name, weight) in enumerate(zip(class_names, weights)):
        print(f"  {name}: {weight:.4f} (count: {int(class_counts[i])})")
    
    return weights


def train_model(config: Config) -> Tuple[nn.Module, Dict]:
    """訓練模型的主函式"""
    from dataset import create_dataloaders
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    config.print_config()
    
    print("\n建立資料載入器...")
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")
    
    print("\n建立 Whisper + Bi-LSTM 模型...")
    model = create_model(config)
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    summary = trainer.train()
    
    return model, summary


# ==================== 主程式入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="訓練 Whisper + Bi-LSTM 語者分離模型")
    parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="訓練 epochs 數"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="學習率（Whisper-large+LoRA 建議勿高於 1e-4）"
    )
    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=256,
        help="LSTM 隱藏層維度（單向）"
    )
    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
        help="LSTM 層數"
    )
    
    args = parser.parse_args()
    
    config = Config(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers
    )
    
    model, summary = train_model(config)
