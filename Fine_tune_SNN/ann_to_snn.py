# -*- coding: utf-8 -*-
"""
將蒸餾後的 Conv1d+ReLU 頭轉為脈衝網路（Whisper + WeightedAvg 仍為 ANN）。
依賴 snntorch；僅支援 student_head_type == "conv"。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import Config


def _clone_conv1d(module: nn.Conv1d) -> nn.Conv1d:
    c = nn.Conv1d(
        module.in_channels,
        module.out_channels,
        module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
    )
    c.load_state_dict(module.state_dict())
    return c

try:
    import snntorch as snn
    from snntorch import surrogate

    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False


def check_snntorch() -> None:
    if not HAS_SNNTORCH:
        raise ImportError(
            "請安裝 snntorch 以執行 ANN→SNN 轉換：pip install snntorch"
        )


class SNNConvDiarHead(nn.Module):
    """
    3×Conv1d + Leaky LIF + 最後 Conv1d 分類（權重自 Student 複製）。
    輸入為已除以 input_scale 的 weighted 特徵 (B, T_frames, D)，內部轉 (B, D, T_frames)。
    """

    def __init__(self, student: nn.Module, timesteps: int, beta: float):
        super().__init__()
        check_snntorch()
        if student.config.student_head_type != "conv":
            raise ValueError("SNN 轉換目前僅支援 student_head_type='conv'")

        self.T = timesteps
        self.beta = beta
        sg = surrogate.fast_sigmoid(slope=25.0)

        # 複製 Conv 權重，避免與 Student 的 ANN 頭共用模組（微調 SNN 時不會改壞原 Student）
        self.conv1 = _clone_conv1d(student.head.layers[0][0])
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg, threshold=1.0)
        self.conv2 = _clone_conv1d(student.head.layers[1][0])
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg, threshold=1.0)
        self.conv3 = _clone_conv1d(student.head.layers[2][0])
        self.lif3 = snn.Leaky(beta=beta, spike_grad=sg, threshold=1.0)
        self.conv_out = _clone_conv1d(student.classifier.classifier)

        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("output_scale", torch.tensor(1.0))

    def forward(self, w_btd: torch.Tensor) -> torch.Tensor:
        x = w_btd.transpose(1, 2)
        x = x / self.input_scale.clamp(min=1e-6)

        mem1 = torch.zeros(
            x.shape[0],
            self.conv1.out_channels,
            x.shape[2],
            device=x.device,
            dtype=x.dtype,
        )
        mem2 = torch.zeros(
            x.shape[0],
            self.conv2.out_channels,
            x.shape[2],
            device=x.device,
            dtype=x.dtype,
        )
        mem3 = torch.zeros(
            x.shape[0],
            self.conv3.out_channels,
            x.shape[2],
            device=x.device,
            dtype=x.dtype,
        )

        spk3_sum = torch.zeros(
            x.shape[0],
            self.conv3.out_channels,
            x.shape[2],
            device=x.device,
            dtype=x.dtype,
        )

        for _ in range(self.T):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_sum = spk3_sum + spk3

        rate = spk3_sum / float(self.T)
        rate = rate * self.output_scale.clamp(min=1e-6)
        logits = self.conv_out(rate)
        return logits.transpose(1, 2)


class WhisperANNWithSNNHead(nn.Module):
    """Whisper + WeightedAvg（ANN）+ SNN Conv 頭。"""

    def __init__(self, student: nn.Module, snn_head: SNNConvDiarHead):
        super().__init__()
        self.student = student
        self.snn_head = snn_head

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        w = self.student.encode_weighted(audio)
        return self.snn_head(w)

    def predict(
        self,
        audio: torch.Tensor,
        apply_median_filter: bool = False,
        return_probs: bool = False,
    ):
        self.eval()
        with torch.no_grad():
            logits = self.forward(audio)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            if apply_median_filter and hasattr(self.student, "median_filter"):
                predictions = self.student.median_filter.filter_predictions(predictions)
            if return_probs:
                return predictions, probs
            return predictions


@torch.no_grad()
def calibrate_scales(
    student: nn.Module,
    snn_head: SNNConvDiarHead,
    data_loader,
    device: torch.device,
    max_batches: int = 4,
) -> None:
    """以少數 batch 估計輸入縮放與第三層發放率→ANN 特徵的縮放。"""
    ws = []
    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        audio = batch["audio"].to(device)
        ws.append(student.encode_weighted(audio))

    in_s = torch.cat(ws, dim=0).abs().amax().clamp(min=1e-6)

    w0 = ws[0]
    ann_h = student.head(w0)
    ann_m = ann_h.abs().mean().clamp(min=1e-6)

    x_n = w0.transpose(1, 2) / in_s
    mem1 = torch.zeros(
        x_n.shape[0],
        snn_head.conv1.out_channels,
        x_n.shape[2],
        device=device,
        dtype=x_n.dtype,
    )
    mem2 = torch.zeros(
        x_n.shape[0],
        snn_head.conv2.out_channels,
        x_n.shape[2],
        device=device,
        dtype=x_n.dtype,
    )
    mem3 = torch.zeros(
        x_n.shape[0],
        snn_head.conv3.out_channels,
        x_n.shape[2],
        device=device,
        dtype=x_n.dtype,
    )
    spk3_sum = torch.zeros_like(mem3)
    for _ in range(snn_head.T):
        cur1 = snn_head.conv1(x_n)
        spk1, mem1 = snn_head.lif1(cur1, mem1)
        cur2 = snn_head.conv2(spk1)
        spk2, mem2 = snn_head.lif2(cur2, mem2)
        cur3 = snn_head.conv3(spk2)
        spk3, mem3 = snn_head.lif3(cur3, mem3)
        spk3_sum = spk3_sum + spk3
    rate = spk3_sum / float(snn_head.T)
    snn_m = rate.abs().mean().clamp(min=1e-6)
    out_s = (ann_m / snn_m).clamp(min=1e-4, max=1e4)

    snn_head.input_scale.copy_(in_s.to(snn_head.input_scale.device))
    snn_head.output_scale.copy_(out_s.to(snn_head.output_scale.device))


def build_hybrid_snn_model(
    student: nn.Module,
    timesteps: int,
    beta: float,
    data_loader,
    device: torch.device,
) -> WhisperANNWithSNNHead:
    snn_head = SNNConvDiarHead(student, timesteps=timesteps, beta=beta).to(device)
    calibrate_scales(student, snn_head, data_loader, device)
    return WhisperANNWithSNNHead(student, snn_head).to(device)


def save_snn_bundle(
    path: str,
    snn_head: SNNConvDiarHead,
    timesteps: int,
    beta: float,
    config_dict: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "snn_head_state_dict": snn_head.state_dict(),
            "timesteps": timesteps,
            "beta": beta,
            "config": config_dict,
        },
        path,
    )


def _freeze_student_backbone(student: nn.Module) -> None:
    """凍結 Whisper、WeightedAvg、Student 的 ANN 頭與分類器（僅訓練 snn_head 複本）。"""
    for p in student.whisper.parameters():
        p.requires_grad = False
    for p in student.weighted_avg.parameters():
        p.requires_grad = False
    if hasattr(student, "head"):
        for p in student.head.parameters():
            p.requires_grad = False
    if hasattr(student, "classifier"):
        for p in student.classifier.parameters():
            p.requires_grad = False


def finetune_snn_head(
    hybrid: WhisperANNWithSNNHead,
    config: Config,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> List[float]:
    """
    只更新 hybrid.snn_head 的參數；encode_weighted 無梯度。
    使用 surrogate gradient 通過 Leaky LIF。
    """
    check_snntorch()
    device = config.device
    _freeze_student_backbone(hybrid.student)

    params = list(hybrid.snn_head.parameters())
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    val_hist: List[float] = []
    for epoch in range(1, epochs + 1):
        hybrid.student.eval()
        hybrid.snn_head.train()
        running = 0.0
        n_batches = 0
        for batch in tqdm(
            train_loader,
            desc=f"SNN 微調 {epoch}/{epochs}",
            leave=False,
            ncols=100,
        ):
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                w = hybrid.student.encode_weighted(audio)
            logits = hybrid.snn_head(w)
            sl = min(logits.size(1), labels.size(1))
            logits_flat = logits[:, :sl, :].contiguous().view(-1, config.num_classes)
            lab_flat = labels[:, :sl].contiguous().view(-1)
            loss = criterion(logits_flat, lab_flat)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(1, n_batches)

        hybrid.snn_head.eval()
        vsum = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                labels = batch["labels"].to(device)
                w = hybrid.student.encode_weighted(audio)
                logits = hybrid.snn_head(w)
                sl = min(logits.size(1), labels.size(1))
                logits_flat = logits[:, :sl, :].contiguous().view(-1, config.num_classes)
                lab_flat = labels[:, :sl].contiguous().view(-1)
                vsum += criterion(logits_flat, lab_flat).item()
                vn += 1
        val_loss = vsum / max(1, vn)
        val_hist.append(val_loss)
        print(
            f"  SNN 微調 epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

    hybrid.snn_head.eval()
    return val_hist


def evaluate_hybrid_snn(
    hybrid: WhisperANNWithSNNHead,
    config: Config,
    data_loader,
    save_dir: Optional[str],
    apply_median_filter: bool,
) -> Dict[str, Any]:
    """沿用 Evaluator 邏輯；SNN 前向建議關閉 AMP。"""
    from evaluate import Evaluator

    cfg = config
    prev_amp = cfg.use_amp
    cfg.use_amp = False
    try:
        evaluator = Evaluator(cfg)
        results = evaluator.evaluate(
            model=hybrid,
            data_loader=data_loader,
            save_dir=save_dir,
            apply_median_filter=apply_median_filter,
        )
    finally:
        cfg.use_amp = prev_amp
    return results


def run_ann_to_snn_pipeline_step(
    student: nn.Module,
    config: Config,
    test_loader,
    timesteps: int,
    beta: float,
    apply_median_filter: bool,
    save_bundle: bool = True,
    finetune_epochs: int = 0,
    finetune_lr: float = 1e-4,
    train_loader=None,
    val_loader=None,
) -> Tuple[Optional[WhisperANNWithSNNHead], Optional[Dict[str, Any]]]:
    """
    Pipeline 用：建立混合 SNN、校準、可選 SNN 頭微調、評估、可選儲存 bundle。
    回傳 (hybrid_model, eval_results)；失敗時 (None, None)。
    """
    if config.student_head_type != "conv":
        print("  ⚠️ 目前僅對 Conv 頭執行 ANN→SNN；student-head=mlp 已跳過。")
        return None, None

    if not HAS_SNNTORCH:
        print("  ⚠️ 未安裝 snntorch，已跳過 SNN 轉換。請執行: pip install snntorch")
        return None, None

    device = config.device
    student.eval()
    print(f"  SNN 時間步 T={timesteps}, Leaky β={beta}，校準中…")
    hybrid = build_hybrid_snn_model(
        student, timesteps=timesteps, beta=beta, data_loader=test_loader, device=device
    )

    if finetune_epochs > 0:
        if train_loader is None or val_loader is None:
            raise ValueError(
                "SNN 頭微調需要 train_loader 與 val_loader（pipeline 請傳入 create_dataloaders）"
            )
        print(
            f"  SNN 頭微調 {finetune_epochs} epochs, lr={finetune_lr}, "
            f"weight_decay={config.weight_decay}（僅 snn_head，Whisper 凍結）"
        )
        finetune_snn_head(
            hybrid,
            config,
            train_loader,
            val_loader,
            epochs=finetune_epochs,
            lr=finetune_lr,
            weight_decay=config.weight_decay,
        )

    hybrid.eval()

    out_sub = os.path.join(config.output_dir, "snn_eval")
    print("  以混合模型（Whisper ANN + Conv SNN 頭）評估測試集…")
    eval_snn = evaluate_hybrid_snn(
        hybrid,
        config,
        test_loader,
        save_dir=out_sub,
        apply_median_filter=apply_median_filter,
    )
    der = eval_snn["der"]
    print(
        f"  SNN 頭 DER: {der.der:.2f}% "
        f"(FA {der.false_alarm:.2f}%, SC {der.speaker_confusion:.2f}%)"
    )

    if save_bundle:
        bundle_path = os.path.join(
            config.model_save_dir,
            f"whisper_{config.model_size}_student_snn_head_bundle.pt",
        )
        cfg_d = {
            "model_size": config.model_size,
            "student_head_type": config.student_head_type,
            "conv_channels": config.conv_channels,
            "num_classes": config.num_classes,
            "snn_timesteps": timesteps,
            "snn_beta": beta,
            "snn_finetune_epochs": finetune_epochs,
            "snn_finetune_lr": finetune_lr,
        }
        save_snn_bundle(bundle_path, hybrid.snn_head, timesteps, beta, cfg_d)
        print(f"  SNN 頭 bundle 已儲存: {bundle_path}")

    return hybrid, eval_snn
