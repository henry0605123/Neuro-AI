# -*- coding: utf-8 -*-
"""
知識蒸餾訓練：Bi-LSTM Teacher（凍結）→ ReLU Student（只訓練頭）
單次編碼：使用 Student 的 encode_weighted（與 Teacher 權重一致）取得 w，再分別算 teacher / student logits。
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from config import Config
from model import WhisperKDStudentModel, create_student_from_teacher
from teacher_lstm import load_teacher


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0001, mode: str = "min"):
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


class TrainingLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "train_loss": [],
            "train_ce": [],
            "train_kd": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
            "epoch_time": [],
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_ce: float,
        train_kd: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float,
    ):
        self.history["train_loss"].append(train_loss)
        self.history["train_ce"].append(train_ce)
        self.history["train_kd"].append(train_kd)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["learning_rate"].append(lr)
        self.history["epoch_time"].append(epoch_time)
        with open(self.log_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def plot_curves(self, save_path=None):
        if not self.history["train_loss"]:
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        axes[0, 0].plot(epochs, self.history["train_loss"], label="train")
        axes[0, 0].plot(epochs, self.history["val_loss"], label="val")
        axes[0, 0].set_title("Total loss (KD)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(epochs, self.history["train_acc"], label="train acc")
        axes[0, 1].plot(epochs, self.history["val_acc"], label="val acc")
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(epochs, self.history["train_ce"], label="CE")
        axes[1, 0].plot(epochs, self.history["train_kd"], label="KD")
        axes[1, 0].set_title("CE vs KD")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].bar(epochs, self.history["epoch_time"], color="purple", alpha=0.7)
        axes[1, 1].set_title("Epoch time (s)")
        plt.tight_layout()
        out = save_path or self.log_dir / "training_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()

    def get_summary(self) -> Dict:
        if not self.history["train_loss"]:
            return {}
        best_val_idx = int(np.argmin(self.history["val_loss"]))
        return {
            "total_epochs": len(self.history["train_loss"]),
            "best_epoch": best_val_idx + 1,
            "best_val_loss": self.history["val_loss"][best_val_idx],
            "best_val_acc": self.history["val_acc"][best_val_idx],
            "total_time": sum(self.history["epoch_time"]),
            "avg_epoch_time": float(np.mean(self.history["epoch_time"])),
        }


def align_logits_labels(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = min(student_logits.shape[1], teacher_logits.shape[1], labels.shape[1])
    return (
        student_logits[:, :seq_len, :],
        teacher_logits[:, :seq_len, :],
        labels[:, :seq_len],
    )


def kd_batch_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float,
    alpha: float,
    num_classes: int,
) -> Tuple[torch.Tensor, float, float]:
    ls, lt, lb = align_logits_labels(student_logits, teacher_logits, labels)
    ls_flat = ls.contiguous().view(-1, num_classes)
    lt_flat = lt.contiguous().view(-1, num_classes)
    lb_flat = lb.contiguous().view(-1)
    ce = F.cross_entropy(ls_flat, lb_flat)
    kd = (
        F.kl_div(
            F.log_softmax(ls_flat / T, dim=-1),
            F.softmax(lt_flat / T, dim=-1),
            reduction="batchmean",
        )
        * (T * T)
    )
    loss = alpha * ce + (1.0 - alpha) * kd
    return loss, ce.item(), kd.item()


class KDTrainer:
    def __init__(
        self,
        teacher: nn.Module,
        student: WhisperKDStudentModel,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        params = list(student.head.parameters()) + list(student.classifier.parameters())
        self.optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.scaler = GradScaler() if config.use_amp else None
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="min")
        self.logger = TrainingLogger(config.log_dir)
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self) -> Tuple[float, float, float, float]:
        self.student.train()
        self.teacher.eval()

        trainable = list(self.student.head.parameters()) + list(self.student.classifier.parameters())
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_correct = 0
        total_samples = 0
        T = self.config.kd_temperature
        alpha = self.config.kd_alpha
        nc = self.config.num_classes

        for batch in tqdm(self.train_loader, desc="KD 訓練", leave=False, ncols=100):
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    with torch.no_grad():
                        w = self.student.encode_weighted(audio)
                        t_h = self.teacher.bilstm_head(w)
                        teacher_logits = self.teacher.classifier(t_h)
                    student_logits = self.student.forward_from_weighted(w)

                    loss, ce_b, kd_b = kd_batch_loss(
                        student_logits, teacher_logits, labels, T, alpha, nc
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad():
                    w = self.student.encode_weighted(audio)
                    t_h = self.teacher.bilstm_head(w)
                    teacher_logits = self.teacher.classifier(t_h)
                student_logits = self.student.forward_from_weighted(w)

                loss, ce_b, kd_b = kd_batch_loss(
                    student_logits, teacher_logits, labels, T, alpha, nc
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_ce += ce_b
            total_kd += kd_b

            ls, _, lb = align_logits_labels(student_logits, teacher_logits, labels)
            pred = torch.argmax(ls, dim=-1)
            total_correct += (pred == lb).sum().item()
            total_samples += lb.numel()

        n_batches = max(1, len(self.train_loader))
        return total_loss / n_batches, total_ce / n_batches, total_kd / n_batches, 100.0 * total_correct / total_samples

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        self.student.eval()
        self.teacher.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(self.val_loader, desc="驗證", leave=False, ncols=100):
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.config.use_amp:
                with autocast():
                    w = self.student.encode_weighted(audio)
                    student_logits = self.student.forward_from_weighted(w)
            else:
                w = self.student.encode_weighted(audio)
                student_logits = self.student.forward_from_weighted(w)

            sl = min(student_logits.shape[1], labels.shape[1])
            ls = student_logits[:, :sl, :]
            lb = labels[:, :sl]
            loss = criterion(ls.contiguous().view(-1, self.config.num_classes), lb.contiguous().view(-1))
            total_loss += loss.item()
            pred = torch.argmax(ls, dim=-1)
            total_correct += (pred == lb).sum().item()
            total_samples += lb.numel()

        n = max(1, len(self.val_loader))
        return total_loss / n, 100.0 * total_correct / max(1, total_samples)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        cfg_dict = {
            "model_size": self.config.model_size,
            "student_head_type": self.config.student_head_type,
            "conv_channels": self.config.conv_channels,
            "mlp_hidden_dims": self.config.mlp_hidden_dims,
            "num_classes": self.config.num_classes,
            "kd_temperature": self.config.kd_temperature,
            "kd_alpha": self.config.kd_alpha,
            "teacher_checkpoint": self.config.teacher_checkpoint,
        }
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": cfg_dict,
            "history": self.logger.history,
        }
        Path(self.config.model_save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, Path(self.config.model_save_dir) / "checkpoint_kd_latest.pt")
        if is_best:
            torch.save(checkpoint, self.config.student_weights_path)
            print(f"  💾 最佳 Student 已儲存: {self.config.student_weights_path}")

    def train(self) -> Dict:
        print("\n" + "=" * 60)
        print("知識蒸餾訓練 (Teacher LSTM → Student ReLU)")
        print("=" * 60)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            print(f"\nEpoch {epoch}/{self.config.epochs}")

            tr_loss, tr_ce, tr_kd, tr_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            ep_time = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            self.logger.log_epoch(epoch, tr_loss, tr_ce, tr_kd, tr_acc, val_loss, val_acc, lr, ep_time)
            print(f"  train loss={tr_loss:.4f} (CE={tr_ce:.4f}, KD={tr_kd:.4f}) acc={tr_acc:.2f}%")
            print(f"  val loss={val_loss:.4f} acc={val_acc:.2f}%  lr={lr:.2e} time={ep_time:.1f}s")

            self.scheduler.step(val_loss)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
            self.save_checkpoint(epoch, is_best=is_best)

            if self.early_stopping(val_loss, epoch):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.logger.plot_curves()
        if self.best_model_state is not None:
            self.student.load_state_dict(self.best_model_state)
            print("已載入驗證集最佳 Student 權重")

        return self.logger.get_summary()


def train_model(config: Config) -> Tuple[nn.Module, Dict]:
    from dataset import create_dataloaders

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    config.print_config()

    print("\n載入 Teacher (Bi-LSTM)...")
    teacher = load_teacher(config, config.teacher_checkpoint)
    teacher.to(config.device)

    print("\n建立 Student 並複製 Encoder + WeightedAvg...")
    student = create_student_from_teacher(config, teacher)
    student.to(config.device)

    print("\n建立 DataLoader...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"訓練批次: {len(train_loader)}, 驗證批次: {len(val_loader)}")

    trainer = KDTrainer(teacher, student, config, train_loader, val_loader)
    summary = trainer.train()
    return student, summary


if __name__ == "__main__":
    c = Config()
    train_model(c)
