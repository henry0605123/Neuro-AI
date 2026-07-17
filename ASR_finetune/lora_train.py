"""Step 3 — LoRA fine-tune helpers.

Provides a `train_fold` function that fine-tunes Belle-whisper-large-v3-zh
with LoRA on the train_cases of a single LOO fold and returns the trained
PeftModel (still in memory) plus paths to the saved adapter.

Designed to be called from `run_loo.py` once per fold.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_cosine_schedule_with_warmup,
)

from config import (
    BASE_MODEL, FEATURES_DIR, LORA_DIR, LANGUAGE, TASK,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    LEARNING_RATE, NUM_EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    WARMUP_RATIO, WEIGHT_DECAY, GRAD_ACCUM,
)


# ---------- Dataset ----------

class CachedFeaturesDataset(Dataset):
    """Concatenates pre-computed feature caches for a list of cases.

    If `role_filter` ("adult"/"child") is set, only samples whose audio_path
    maps to that role via `path_to_role` are kept. This supports 方案B
    (separate adult/child LoRA adapters).
    """

    def __init__(self, case_names: list[str], pad_token_id: int,
                 role_filter: str | None = None,
                 path_to_role: dict[str, str] | None = None):
        self.records: list[tuple[torch.Tensor, list[int], str, str]] = []
        self.pad_token_id = pad_token_id
        for case in case_names:
            blob = torch.load(FEATURES_DIR / f"{case}.pt", map_location="cpu", weights_only=False)
            feats = blob["features"]      # (N, 80, 3000) fp16
            labels = blob["labels"]        # list of token id lists
            speakers = blob["speakers"]
            paths = blob.get("audio_paths", [None] * feats.shape[0])
            for i in range(feats.shape[0]):
                if role_filter is not None:
                    if not path_to_role or path_to_role.get(paths[i]) != role_filter:
                        continue
                self.records.append((feats[i], labels[i], speakers[i], case))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        feat, tokens, _, _ = self.records[idx]
        return feat, tokens


def collate_fn(batch, pad_token_id: int):
    # Keep features in fp16 — the model is fp16 and autocast handles dtype.
    feats = torch.stack([b[0] for b in batch], dim=0)
    label_lists = [b[1] for b in batch]
    max_len = max(len(t) for t in label_lists)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, t in enumerate(label_lists):
        labels[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return feats, labels


# ---------- Train / eval ----------

@dataclass
class FoldArtifacts:
    fold: int
    test_case: str
    val_case: str
    adapter_dir: Path
    train_loss_curve: list[float]
    val_loss_curve: list[float]
    train_time_sec: float


def build_peft_model() -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    )
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    base.generation_config.language = LANGUAGE
    base.generation_config.task = TASK
    base.generation_config.forced_decoder_ids = None
    base.gradient_checkpointing_enable()
    base.config.use_cache = False

    # Note: leaving `task_type` unset — PeftModelForSeq2SeqLM injects
    # standard seq2seq kwargs like `input_ids` that Whisper's signature
    # doesn't accept; the bare PeftModel wrapper avoids that.
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.to("cuda")
    return model, processor


def _make_loader(ds: Dataset, batch_size: int, pad_id: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,                    # Windows: no multiproc
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )


@torch.no_grad()
def _eval_loss(model, loader) -> float:
    model.eval()
    total, count = 0.0, 0
    for feats, labels in loader:
        feats = feats.to("cuda", dtype=torch.float16, non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(input_features=feats, labels=labels)
        total += float(out.loss.item()) * feats.size(0)
        count += feats.size(0)
    return total / max(count, 1)


def train_fold(fold_def: dict, num_epochs: int = NUM_EPOCHS,
               role: str | None = None,
               path_to_role: dict[str, str] | None = None,
               patience: int | None = None,
               min_delta: float = 1e-3) -> FoldArtifacts:
    """Fine-tune one fold and save the LoRA adapter.

    If `role` is "adult" or "child", only that role's segments are used
    (方案B / separate adapters). adapter_dir gets the role suffix.
    """
    fold = fold_def["fold"]
    test_case = fold_def["test_case"]
    val_case = fold_def["val_case"]
    train_cases = fold_def["train_cases"]

    suffix = f"_{role}" if role else ""
    adapter_dir = LORA_DIR / f"fold{fold:02d}_test={test_case}{suffix}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    role_tag = f" | role={role}" if role else ""
    print(f"\n=== Fold {fold}{role_tag} | test={test_case} | val={val_case} | train={len(train_cases)} cases ===")
    model, processor = build_peft_model()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    train_ds = CachedFeaturesDataset(train_cases, pad_id,
                                     role_filter=role, path_to_role=path_to_role)
    val_ds = CachedFeaturesDataset([val_case], pad_id,
                                   role_filter=role, path_to_role=path_to_role)
    print(f"  train segments={len(train_ds)}  val segments={len(val_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError(f"Empty train set for fold={fold} role={role}")

    train_loader = _make_loader(train_ds, TRAIN_BATCH_SIZE, pad_id, shuffle=True)
    val_loader = _make_loader(val_ds, EVAL_BATCH_SIZE, pad_id, shuffle=False)

    total_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")  # for fp16 grad scaling

    train_loss_curve, val_loss_curve = [], []
    best_val, best_epoch, no_improve, early_stopped = float("inf"), 0, 0, False
    t0 = time.time()
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"  ep{epoch+1}/{num_epochs}", leave=False)
        for step, (feats, labels) in enumerate(pbar):
            feats = feats.to("cuda", dtype=torch.float16, non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_features=feats, labels=labels)
                loss = out.loss / GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            epoch_losses.append(float(out.loss.item()))
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{np.mean(epoch_losses[-50:]):.3f}")
        ep_loss = float(np.mean(epoch_losses))
        train_loss_curve.append(ep_loss)
        val_loss = _eval_loss(model, val_loader)
        val_loss_curve.append(val_loss)
        print(f"  epoch {epoch+1}: train_loss={ep_loss:.4f}  val_loss={val_loss:.4f}")

        if patience is not None:
            if val_loss < best_val - min_delta:
                best_val, best_epoch, no_improve = val_loss, epoch + 1, 0
                model.save_pretrained(adapter_dir)   # checkpoint best-val epoch
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  early stopping at epoch {epoch+1} "
                          f"(best epoch {best_epoch}, val_loss={best_val:.4f})")
                    early_stopped = True
                    break

    dt = time.time() - t0
    print(f"  fold {fold} train time: {dt:.1f}s")

    # Save adapter. With early stopping the best-val checkpoint is already on
    # disk (saved on each improvement); otherwise save the final-epoch weights.
    if patience is None:
        model.save_pretrained(adapter_dir)
    (adapter_dir / "fold_info.json").write_text(
        json.dumps({
            "fold": fold,
            "test_case": test_case,
            "val_case": val_case,
            "train_cases": train_cases,
            "train_loss_curve": train_loss_curve,
            "val_loss_curve": val_loss_curve,
            "train_time_sec": dt,
            "num_epochs": num_epochs,
            "patience": patience,
            "best_epoch": best_epoch if patience is not None else num_epochs,
            "early_stopped": early_stopped,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lr": LEARNING_RATE,
            "batch_size": TRAIN_BATCH_SIZE,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FoldArtifacts(
        fold=fold, test_case=test_case, val_case=val_case,
        adapter_dir=adapter_dir,
        train_loss_curve=train_loss_curve,
        val_loss_curve=val_loss_curve,
        train_time_sec=dt,
    )
