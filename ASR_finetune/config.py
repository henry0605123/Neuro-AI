"""Shared configuration for the ASR fine-tune pipeline.

All paths are absolute so individual scripts can be run from anywhere.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---- Roots ----
# PROJECT_ROOT 預設為本檔所在目錄;DATA_ROOT 請用環境變數指到你的原始資料夾。
PROJECT_ROOT = Path(os.environ.get("ASR_PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = Path(os.environ.get("ASR_DATA_ROOT", "data/Fine_tune"))

# ---- Pipeline artifacts ----
SEGMENTS_DIR = PROJECT_ROOT / "segments"
MANIFEST_FILE = PROJECT_ROOT / "manifest.jsonl"
FOLDS_FILE = PROJECT_ROOT / "loo_folds.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LORA_DIR = OUTPUTS_DIR / "lora_adapters"
FEATURES_DIR = PROJECT_ROOT / "features"   # pre-computed mel cache

# ---- Audio params ----
SAMPLE_RATE = 16_000
SEG_PAD_SEC = 0.15       # padding both sides of a segment (avoid cutoff)
MIN_SEG_DUR = 0.4        # drop segments shorter than this (after padding)
MAX_SEG_DUR = 29.0       # Whisper's 30s window; leave 1s headroom

# ---- Model ----
BASE_MODEL = "BELLE-2/Belle-whisper-large-v3-zh"
LANGUAGE = "zh"
TASK = "transcribe"

# ---- Speaker convention ----
# Each case's JSON has its own arbitrary `speaker_mapping` that maps speaker IDs
# (e.g. SPEAKER_00 / SPEAKER_02 / Unknown) to one of these two roles. We
# normalize to {adult, child} for the whole pipeline.
ROLE_ADULT_NAMES = {"評估者", "測試者", "老師", "個管師", "家長", "母親", "媽媽",
                    "父親", "爸爸", "家屬", "監護人", "adult", "Adult"}
ROLE_CHILD_NAMES = {"兒童", "小孩", "child", "Child"}
# 非對話者 / 無法歸屬 → 丟棄(不納入 child/adult)
ROLE_DROP_NAMES = {"廣播", "Unknown", "unknown", ""}


def resolve_role(case_speaker_mapping: dict, speaker_id: str) -> str | None:
    """Map a per-segment speaker to canonical 'adult' / 'child' / None.

    支援兩種 GT 格式:
      - 舊格式:speaker 是 SPEAKER_xx,透過 case 的 speaker_mapping 翻成角色。
      - 新格式:speaker_mapping 為空、speaker 欄位本身就是角色字串(兒童/評估者/…)
        → 直接拿 speaker 當角色判定。
    廣播 / Unknown / 無法辨識 → 回 None(呼叫端應丟棄該段)。
    """
    role = case_speaker_mapping.get(speaker_id, speaker_id)
    if role is None:
        return None
    role = str(role)
    if role in ROLE_CHILD_NAMES:
        return "child"
    if role in ROLE_ADULT_NAMES:
        return "adult"
    return None

# ---- Training (Step 3-4) ----
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
# Whisper module names (attention projections only — safe + light)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
WARMUP_RATIO = 0.1
GRAD_ACCUM = 1
WEIGHT_DECAY = 0.0


def ensure_dirs() -> None:
    for d in (SEGMENTS_DIR, OUTPUTS_DIR, PREDICTIONS_DIR, METRICS_DIR,
              LORA_DIR, FEATURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
