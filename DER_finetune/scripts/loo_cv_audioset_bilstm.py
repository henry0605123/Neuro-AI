"""
LOO Cross-Validation with ChangPN Chinese Pre-Training (from scratch).

Two-stage pipeline:
  Phase 1 — Pre-train on all 111 ChangPN sessions (Chinese TD child-adult, TalkBank).
             Starting point: base Whisper model (no AudioSet pre-training).
             Val: 11 held-out ChangPN sessions (early stopping only).
  Phase 2 — LOO CV on 13 ASD sessions, starting from ChangPN checkpoint.

Variant: Bi-LSTM classifier head (replaces the 1D-CNN model_seq) + Stage 1
         class-weighted NLLLoss (balances silence/child/adult/overlap frames,
         since ChangPN adult:child ≈ 67:33). Stage 2 keeps unweighted NLLLoss.

Resume-safe:
  - Phase 1 skipped if any .ckpt exists in CHANGPN_CKPT_DIR.
  - Completed LOO folds detected by JSON files in RESULTS_DIR and skipped.

Usage:
  python scripts/loo_cv_changpn_scratch.py
"""
import sys, os, json, re, glob, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch, torchaudio
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoFeatureExtractor

from models.whisper import WhisperWrapper
from dataset_classes.datasets import DatasetGenerator
from dataset_classes.preprocess import FRAME_SIZE, FRAME_STRIDE, WINDOW_OVERLAP
from scripts.convert_output import get_timestamps, majority_filter
from lightning_modules.classifier import Classifier

torch.set_float32_matmul_precision('high')

# ── Transformers compatibility shim ───────────────────────────────────────────
# Newer transformers changed WhisperAttention.forward:
#   (1) added `self.config._attn_implementation` check → AttributeError if config=None
#   (2) returns 2 values instead of 3; WhisperEncoderLayer expects 3
import transformers.models.whisper.modeling_whisper as _wm

_orig_wa_forward = _wm.WhisperAttention.forward

def _compat_wa_forward(self, *args, **kwargs):
    if getattr(self, 'config', None) is None:
        class _C:
            _attn_implementation = "eager"
        self.config = _C()
    result = _orig_wa_forward(self, *args, **kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1], None  # pad to legacy 3-tuple
    return result

_wm.WhisperAttention.forward = _compat_wa_forward

# ── Mel pre-computation (loaded once on import) ────────────────────────────────
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
MEL_LEN = 1000   # frames for 10s at hop_length=160: 160000/160=1000
ENC_LEN = 500    # after Whisper encoder CNN: (1000-1)//2+1=500


class WhisperWrapperFast(WhisperWrapper):
    """WhisperWrapper accepting pre-computed mel [B,80,MEL_LEN] or raw audio.

    New transformers requires mel to be exactly 3000 frames (line 1016-1019 in
    modeling_whisper.py).  We pad 1000-frame mel → 3000, run the encoder to get
    1500 output frames, then keep only the first ENC_LEN=500 frames (actual
    10-second audio).  Positional embeddings are restored to the full 1500-entry
    table so the addition `inputs_embeds + embed_pos` does not mis-shape.
    """

    def forward(self, x, length=None):
        if x.dim() == 3:
            # Pre-computed mel [B, 80, MEL_LEN=1000] → pad to 3000
            features = torch.nn.functional.pad(x, (0, 3000 - x.shape[2])).cuda()
        else:
            # Raw audio [B, T]: use built-in feature extractor (pads to 30s = 3000 frames)
            new_x = [x[i].detach().cpu().numpy() for i in range(len(x))]
            features = self.feature_extractor(
                new_x, return_tensors="pt", sampling_rate=16000
            ).input_features.cuda()

        # Full embed_positions [1500, D] matches 3000-frame mel → 1500 encoder frames
        self.backbone_model.encoder.embed_positions = \
            self.backbone_model.encoder.embed_positions.from_pretrained(
                self.embed_positions)

        hidden_states = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states

        # Encoder output: [B, 1500, D]; keep first ENC_LEN=500 (actual audio)
        stacked = torch.stack(hidden_states, dim=0)[1:]          # [L, B, 1500, D]
        stacked = stacked[:, :, :ENC_LEN, :].contiguous()        # [L, B, 500, D]

        _, *shape = stacked.shape  # [B, 500, D]
        stacked = stacked.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = torch.nn.functional.softmax(self.weights, dim=-1)
        weighted = (norm_weights.unsqueeze(-1) * stacked).sum(dim=0)
        return self.model_seq(weighted.view(*shape).transpose(1, 2))


# ── Constants ─────────────────────────────────────────────────────────────────
# Phase 1 — ChangPN pretrain (scratch)
CHANGPN_AUDIO_DIR      = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Talkbank/ChangPN/audio"
CHANGPN_TRANSCRIPT_DIR = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Talkbank/ChangPN/transcript"
CHANGPN_CKPT_DIR       = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling/checkpoints/changpn_pretrain_scratch_bilstm"
CHANGPN_VAL_N          = 11    # sessions held out for early stopping
CHANGPN_MAX_EPOCHS     = 30
CHANGPN_PATIENCE       = 10

# Phase 2 — ASD LOO
INPUT_DIR    = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Fine_tune"
RESULTS_DIR  = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/loo_results_audioset_bilstm_16sess"
SAVE_DIR     = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling"
AUDIOSET_PRETRAINED = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling/checkpoints/whisper_base_5k/epoch=17-val_loss=0.113.ckpt"
ASD_MAX_EPOCHS = 20
ASD_PATIENCE   = 7

# Shared
SAMPLE_RATE = 16000
WINDOW_SIZE = 10
COLLAR      = 0.1
BATCH_SIZE  = 16   # reduced from 64: encoder LoRA now actually trains (merge_weights fix) → more activation memory; 16 keeps comfortable VRAM headroom
LR          = 1e-4
LORA_RANK   = 8

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHANGPN_CKPT_DIR, exist_ok=True)

HPARAMS_BASE = {
    "embedding": "whisper_base", "lora_rank": LORA_RANK,
    "learning_rate": LR, "weight_decay": 1e-4,
    "optimizer": "Adam", "batch_size": BATCH_SIZE,
    "window_size": WINDOW_SIZE, "accumulate_grad_batches": 1,
    "max_epochs": ASD_MAX_EPOCHS, "devices": [0], "seed": 42, "debug": False,
    "num_enc_layers_dict": {"whisper_base": 6},
    "hid_dim_dict": {"whisper_base": 512},
    "head_type": "bilstm",
    "audio": "", "annotation": "", "name": "",
}


# ── Annotation parsers ─────────────────────────────────────────────────────────

def parse_cha(cha_path):
    """Parse TalkBank CHAT .cha file. Returns [(label, start_s, end_s)]."""
    BULLET = '\x15'  # U+0015, used as timestamp delimiter in CLAN/CHAT format
    segments = []
    with open(cha_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not (line.startswith('*CHI:') or line.startswith('*EXP:')):
                continue
            label = 'c' if line.startswith('*CHI:') else 'a'
            parts = line.split(BULLET)
            if len(parts) >= 3:   # [text_part, "start_end", trailing]
                ts = parts[1].strip()
                if '_' in ts:
                    try:
                        s_ms, e_ms = ts.split('_')
                        s, e = int(s_ms) / 1000.0, int(e_ms) / 1000.0
                        if e > s:
                            segments.append((label, s, e))
                    except ValueError:
                        pass
    return segments


def srt_time_to_sec(t):
    h, m, s = t.strip().split(":")
    s, ms = s.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000


def parse_srt(path):
    segs = []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for block in re.split(r"\n\n+", content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        times = lines[1].split("-->")
        s, e = srt_time_to_sec(times[0]), srt_time_to_sec(times[1])
        text = " ".join(lines[2:])
        if "小孩" in text:
            segs.append(("c", s, e))
        elif "測試者" in text or "評估者" in text:
            segs.append(("a", s, e))
    return segs


def parse_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mapping = data.get("speaker_mapping") or data.get("metadata", {}).get("speaker_mapping", {})
    role_map = {spk: ("c" if "兒童" in role or "小孩" in role else "a")
                for spk, role in mapping.items()}
    segs = []
    for seg in data["segments"]:
        label = role_map.get(seg.get("speaker", ""), "a")
        s, e = float(seg["start"]), float(seg["end"])
        if e > s:
            segs.append((label, s, e))
    return segs


# ── Label mapping ─────────────────────────────────────────────────────────────
def label_map(label):
    if label == "si": return 0
    if "c" in label:  return 1
    if "a" in label:  return 2
    return 3


# ── Class weighting (Stage 1 only) ──────────────────────────────────────────────
def compute_class_weights(windows, num_classes=4):
    """sklearn-style 'balanced' class weights from frame-level labels."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for w in windows:
        counts += torch.bincount(w["labels"].flatten(), minlength=num_classes).double()
    weights = torch.where(counts > 0, counts.sum() / (num_classes * counts), torch.zeros_like(counts))
    return weights.float()


# ── Audio / windowing ─────────────────────────────────────────────────────────
def load_audio(path):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio[0].numpy()


def make_windows(audio_np, segments):
    """Create 10s sliding windows with pre-computed mel spectrograms."""
    length = len(audio_np) / SAMPLE_RATE
    seg_list = sorted([{"start": s, "end": e, "label": lbl}
                       for lbl, s, e in segments], key=lambda x: x["start"])
    filled, prev_end = [], 0.0
    for seg in seg_list:
        if seg["start"] > prev_end + 0.01:
            filled.append({"start": prev_end, "end": seg["start"], "label": "si"})
        filled.append(seg)
        prev_end = seg["end"]
    if prev_end < length:
        filled.append({"start": prev_end, "end": length, "label": "si"})

    labels_all, curr_time, seg_idx = [], FRAME_SIZE / 2, 0
    while curr_time < length and seg_idx < len(filled):
        labels_all.append(label_map(filled[seg_idx]["label"]))
        if curr_time > filled[seg_idx]["end"]:
            seg_idx += 1
        curr_time += FRAME_STRIDE

    windows, start = [], 0.0
    while start + WINDOW_SIZE <= length:
        a_s = round(start * SAMPLE_RATE)
        a_e = round((start + WINDOW_SIZE) * SAMPLE_RATE)
        seg_np = audio_np[a_s:a_e]
        l_s = round(start / FRAME_STRIDE)
        l_e = l_s + round(WINDOW_SIZE / FRAME_STRIDE)
        lbl = torch.tensor(labels_all[l_s:l_e])
        if len(seg_np) == round(WINDOW_SIZE * SAMPLE_RATE) and len(lbl) > 0:
            mel = FEATURE_EXTRACTOR(seg_np, return_tensors="pt",
                                    sampling_rate=SAMPLE_RATE,
                                    max_length=len(seg_np)).input_features[0]
            windows.append({"audio": mel, "labels": lbl,
                            "start": start, "file_name": ""})
        start += WINDOW_SIZE * WINDOW_OVERLAP
    return windows


# ── DER & Inference ───────────────────────────────────────────────────────────
def compute_der(child_pred, adult_pred, child_gt, adult_gt, duration):
    step = 0.02
    n = int(duration / step) + 1
    pred = np.zeros(n, dtype=int)
    for s, e in child_pred:
        pred[int(s/step):int(e/step)] = 1
    for s, e in adult_pred:
        pred[int(s/step):int(e/step)] = 2
    gt = np.zeros(n, dtype=int)
    for s, e in child_gt:
        gt[int(s/step):int(e/step)] = 1
    for s, e in adult_gt:
        gt[int(s/step):int(e/step)] = 2
    collar_frames = int(COLLAR / step)
    collar_mask = np.zeros(n, dtype=bool)
    for arr in [child_gt, adult_gt]:
        for s, e in arr:
            fs, fe = int(s/step), int(e/step)
            collar_mask[max(0, fs-collar_frames):min(n, fs+collar_frames)] = True
            collar_mask[max(0, fe-collar_frames):min(n, fe+collar_frames)] = True
    mask = ~collar_mask
    ref = (gt[mask] > 0).sum()
    if ref == 0:
        return None
    fa = ((pred[mask] > 0) & (gt[mask] == 0)).sum()
    md = ((pred[mask] == 0) & (gt[mask] > 0)).sum()
    sc = ((pred[mask] > 0) & (gt[mask] > 0) & (pred[mask] != gt[mask])).sum()
    return {"FA": fa/ref*100, "MD": md/ref*100, "SC": sc/ref*100,
            "DER": (fa+md+sc)/ref*100}


def run_inference(mp3_path, whisper_model):
    audio, sr = torchaudio.load(mp3_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio.float()
    length = audio.size(1) / SAMPLE_RATE
    child_pred, adult_pred = [], []
    start = 0.0
    while start + WINDOW_SIZE <= length:
        window = audio[:, int(start*SAMPLE_RATE):int((start+WINDOW_SIZE)*SAMPLE_RATE)]
        pred = whisper_model.forward_eval(window)
        pred = majority_filter(pred)
        child, adult, _ = get_timestamps(pred)
        child_pred += [(start+s, start+e) for s, e in child]
        adult_pred += [(start+s, start+e) for s, e in adult]
        start += WINDOW_SIZE

    def merge(ivs):
        merged = []
        for s, e in sorted(ivs):
            if merged and s - merged[-1][1] <= 0.05:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged
    return merge(child_pred), merge(adult_pred), length


# ── Session loaders ───────────────────────────────────────────────────────────
def load_changpn_sessions():
    """Scan ChangPN directory and return [(session_name, mp3_path, cha_path)]."""
    sessions = []
    for age_folder in sorted(os.listdir(CHANGPN_TRANSCRIPT_DIR)):
        cha_dir   = os.path.join(CHANGPN_TRANSCRIPT_DIR, age_folder)
        audio_dir = os.path.join(CHANGPN_AUDIO_DIR, age_folder)
        if not os.path.isdir(cha_dir):
            continue
        for cha_file in sorted(os.listdir(cha_dir)):
            if not cha_file.endswith('.cha'):
                continue
            stem     = os.path.splitext(cha_file)[0]
            mp3_path = os.path.join(audio_dir, stem + '.mp3')
            cha_path = os.path.join(cha_dir, cha_file)
            if os.path.exists(mp3_path):
                sessions.append((f"{age_folder}_{stem}", mp3_path, cha_path))
    return sessions


def load_asd_sessions():
    """Scan ASD Fine_tune directory and return [(name, mp3, ann, ann_type)]."""
    sessions = []
    for d in sorted(os.listdir(INPUT_DIR)):
        full = os.path.join(INPUT_DIR, d)
        if not os.path.isdir(full):
            continue
        mp3s  = glob.glob(os.path.join(full, "*.mp3"))
        if not mp3s:
            continue
        jsons = glob.glob(os.path.join(full, "*_edited.json"))
        srts  = glob.glob(os.path.join(full, "*.srt"))
        if jsons:
            sessions.append((d, mp3s[0], jsons[0], "json"))
        elif srts:
            sessions.append((d, mp3s[0], srts[0], "srt"))
    return sessions


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(name, train_windows, val_windows, pretrained_state, class_weights=None):
    """Instantiate Classifier without calling __init__ (avoids preprocess_all_files)."""
    hparams = {**HPARAMS_BASE, "name": name}
    model = Classifier.__new__(Classifier)
    pl.LightningModule.__init__(model)
    model.save_hyperparameters(hparams)
    model.model           = WhisperWrapperFast(pretrained_model=hparams["embedding"],
                                               lora_rank=hparams["lora_rank"],
                                               head_type="bilstm")
    model.criterion       = (torch.nn.NLLLoss(weight=class_weights) if class_weights is not None
                              else torch.nn.NLLLoss())
    model.prediction_dict = defaultdict(list)
    model.training_save   = []
    model.validation_save = []
    model.testing_save    = []
    model.train_data      = train_windows
    model.val_data        = val_windows
    model.test_data       = val_windows
    model.load_state_dict(pretrained_state, strict=False)
    # Newer transformers requires WhisperAttention.config to be set; patch any None entries.
    _backbone_cfg = model.model.backbone_model.config
    for _layer in model.model.backbone_model.encoder.layers:
        if hasattr(_layer, 'self_attn') and getattr(_layer.self_attn, 'config', None) is None:
            _layer.self_attn.config = _backbone_cfg
    return model


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SKIPPED (AudioSet+BiLSTM baseline: NO ChangPN pre-training)
# ════════════════════════════════════════════════════════════════════════════════
# This is the BiLSTM control for Exp5'. Same code path as loo_cv_changpn_scratch.py
# (transformers shim, WhisperWrapperFast, head_type="bilstm", identical hparams), but
# we skip the ChangPN pretrain (and its Stage-1 class weighting, which only exists to
# pretrain on ChangPN) and start every LOO fold from the AudioSet checkpoint. The
# encoder/LoRA weights load from AudioSet; the BiLSTM head is freshly initialised and
# trained on the 16 ASD sessions during LOO. Only the pretrain source differs from
# Exp5' → isolates the value of ChangPN pre-training given the BiLSTM head.
print("\n" + "="*60)
print("PHASE 1 SKIPPED — AudioSet+BiLSTM baseline (no ChangPN pre-training)")
print("="*60)


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ASD LOO Cross-Validation
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 2: ASD LOO Cross-Validation")
print(f"Starting checkpoint (AudioSet): {os.path.basename(AUDIOSET_PRETRAINED)}")
print("="*60)

# Load AudioSet checkpoint once (shared starting point for all folds; BiLSTM head
# is not in this checkpoint and will be randomly initialised via strict=False).
print("\nLoading AudioSet pretrained checkpoint...")
_ckpt = torch.load(AUDIOSET_PRETRAINED, map_location="cpu")
changpn_state = {k: v for k, v in _ckpt["state_dict"].items()
                 if "embed_positions" not in k}
del _ckpt
torch.cuda.empty_cache()

# Load ASD sessions
print("Scanning ASD sessions...")
asd_sessions = load_asd_sessions()
print(f"Found {len(asd_sessions)} ASD sessions")

# Pre-process all ASD sessions once
print("\nPre-processing ASD sessions...")
asd_data = {}
asd_gt   = {}
for name, mp3, ann, ann_type in tqdm(asd_sessions):
    segs = parse_json(ann) if ann_type == "json" else parse_srt(ann)
    asd_data[name] = make_windows(load_audio(mp3), segs)
    asd_gt[name] = (
        [(s, e) for lbl, s, e in segs if lbl == "c"],
        [(s, e) for lbl, s, e in segs if lbl == "a"],
        mp3,
    )
print(f"Total ASD windows: {sum(len(v) for v in asd_data.values())}")

# LOO loop
all_results = []

for fold_idx, (test_name, test_mp3, _, _) in enumerate(asd_sessions):
    result_file = os.path.join(RESULTS_DIR, f"fold{fold_idx+1:02d}_{test_name}.json")

    # Resume: skip completed folds
    if os.path.exists(result_file):
        print(f"\nFold {fold_idx+1} ({test_name}) already done — loading cached result.")
        with open(result_file, encoding="utf-8") as f:
            all_results.append(json.load(f))
        continue

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/{len(asd_sessions)}: test = {test_name}")
    print(f"{'='*60}")

    train_windows = [w for name, wins in asd_data.items() if name != test_name for w in wins]
    val_windows   = asd_data[test_name]
    print(f"  Train: {len(train_windows)} | Val: {len(val_windows)} windows")

    train_dl = DataLoader(
        DatasetGenerator(train_windows, len(train_windows), is_test=False),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_dl = DataLoader(
        DatasetGenerator(val_windows, len(val_windows), is_test=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(f"loo_as16_bilstm_fold{fold_idx+1:02d}", train_windows, val_windows, changpn_state)

    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints", f"loo_as16_bilstm_fold{fold_idx+1:02d}")
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name=f"loo_as16_bilstm_fold{fold_idx+1:02d}"),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=ASD_PATIENCE, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                            dirpath=ckpt_dir, filename="{epoch}-{val_loss:.3f}"),
        ],
        max_epochs=ASD_MAX_EPOCHS,
        devices=[0], accelerator="gpu",
        precision="16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"  Best ckpt: {os.path.basename(best_ckpt)}")
    del model, trainer
    torch.cuda.empty_cache()

    # Evaluate on original test audio
    best_data = torch.load(best_ckpt, map_location="cpu")
    whisper_model = WhisperWrapperFast(pretrained_model=HPARAMS_BASE["embedding"],
                                       lora_rank=LORA_RANK, head_type="bilstm")
    model_state = {k[len("model."):]: v for k, v in best_data["state_dict"].items()
                   if k.startswith("model.") and "embed_positions" not in k}
    whisper_model.load_state_dict(model_state, strict=False)
    whisper_model = whisper_model.cuda()
    whisper_model.eval()

    child_gt, adult_gt, mp3_path = asd_gt[test_name]
    _audio, _sr = torchaudio.load(mp3_path)
    duration = _audio.shape[1] / _sr

    with torch.no_grad():
        child_pred, adult_pred, _ = run_inference(mp3_path, whisper_model)

    del whisper_model, best_data
    torch.cuda.empty_cache()

    result = compute_der(child_pred, adult_pred, child_gt, adult_gt, duration)
    if result:
        result["session"] = test_name
        all_results.append(result)
        print(f"  FA={result['FA']:.1f}%  MD={result['MD']:.1f}%  "
              f"SC={result['SC']:.1f}%  DER={result['DER']:.1f}%")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print("  WARNING: no reference speech found, skipping DER")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("LOO + ChangPN Pre-Training (from scratch) Results")
print(f"{'='*60}")
print(f"{'Session':<42} {'FA':>6} {'MD':>6} {'SC':>6} {'DER':>6}")
print("-"*62)
for r in all_results:
    print(f"{r['session']:<42} {r['FA']:>5.1f}% {r['MD']:>5.1f}% "
          f"{r['SC']:>5.1f}% {r['DER']:>5.1f}%")

if all_results:
    print("-"*62)
    avgs = {k: np.mean([r[k] for r in all_results]) for k in ["FA", "MD", "SC", "DER"]}
    print(f"{'Average':<42} {avgs['FA']:>5.1f}% {avgs['MD']:>5.1f}% "
          f"{avgs['SC']:>5.1f}% {avgs['DER']:>5.1f}%")
    summary = {
        "sessions": all_results,
        "average": avgs,
        "method": "AudioSet pretrain → ASD LOO fine-tune (fixed GT, 16 session, BiLSTM head, fresh head)",
        "start_checkpoint": os.path.basename(AUDIOSET_PRETRAINED),
        "n_sessions": len(asd_sessions),
        "note": "BiLSTM control for Exp5'; only pretrain source differs (AudioSet vs ChangPN)",
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {RESULTS_DIR}/summary.json")
