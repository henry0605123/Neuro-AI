"""
Optimized Leave-One-Out Cross-Validation.
Key optimizations vs loo_cv.py:
  1. Whisper model loaded ONCE per fold via weight reset (no subprocess)
  2. All audio pre-processed into windows ONCE at startup
  3. Larger batch size (64), fewer epochs (20)

Usage:
  python scripts/loo_cv_fast.py
"""
import sys, os, json, re, glob
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

from models.whisper import WhisperWrapper
from dataset_classes.datasets import DatasetGenerator
from dataset_classes.preprocess import FRAME_SIZE, FRAME_STRIDE, WINDOW_OVERLAP
from scripts.convert_output import get_timestamps, majority_filter
from lightning_modules.classifier import Classifier

torch.set_float32_matmul_precision('high')

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR   = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Fine_tune"
RESULTS_DIR = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/loo_results_fast"
PRETRAINED  = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling/checkpoints/whisper_base_5k/epoch=17-val_loss=0.113.ckpt"
SAVE_DIR    = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling"
SAMPLE_RATE = 16000
WINDOW_SIZE = 10
COLLAR      = 0.1
BATCH_SIZE  = 64
MAX_EPOCHS  = 20
PATIENCE    = 7
LR          = 1e-4
LORA_RANK   = 8

os.makedirs(RESULTS_DIR, exist_ok=True)

HPARAMS = {
    "embedding": "whisper_base", "lora_rank": LORA_RANK,
    "learning_rate": LR, "weight_decay": 1e-4,
    "optimizer": "Adam", "batch_size": BATCH_SIZE,
    "window_size": WINDOW_SIZE, "accumulate_grad_batches": 1,
    "max_epochs": MAX_EPOCHS, "devices": [0], "seed": 42,
    "debug": False,
    "num_enc_layers_dict": {"whisper_base": 6},
    "hid_dim_dict": {"whisper_base": 512},
    "audio": "", "annotation": "", "name": "",
}

# ── Annotation parsing ────────────────────────────────────────────────────────
def srt_time_to_sec(t):
    h, m, s = t.strip().split(":")
    s, ms = s.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def parse_srt(srt_path):
    segments = []
    with open(srt_path, encoding="utf-8") as f:
        content = f.read()
    for block in re.split(r"\n\n+", content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        times = lines[1].split("-->")
        s, e = srt_time_to_sec(times[0]), srt_time_to_sec(times[1])
        text = " ".join(lines[2:])
        if "小孩" in text:
            segments.append(("c", s, e))
        elif "測試者" in text or "評估者" in text:
            segments.append(("a", s, e))
    return segments

def parse_json(json_path):
    with open(json_path, encoding="utf-8") as f:
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

# ── Audio pre-processing ──────────────────────────────────────────────────────
def label_map(label):
    if label == "si": return 0
    if "c" in label:  return 1
    if "a" in label:  return 2
    return 3

def preprocess_session(mp3_path, segments):
    audio, sr = torchaudio.load(mp3_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio[0].float()
    length = len(audio) / SAMPLE_RATE

    seg_list = sorted([{"start": s, "end": e, "label": lbl} for lbl, s, e in segments],
                      key=lambda x: x["start"])
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
        seg_start = round(start * SAMPLE_RATE)
        seg_end   = round((start + WINDOW_SIZE) * SAMPLE_RATE)
        seg_audio = audio[seg_start:seg_end]
        lbl_start = round(start / FRAME_STRIDE)
        lbl_end   = lbl_start + round(WINDOW_SIZE / FRAME_STRIDE)
        lbl = torch.tensor(labels_all[lbl_start:lbl_end])
        if len(seg_audio) == round(WINDOW_SIZE * SAMPLE_RATE) and len(lbl) > 0:
            windows.append({"audio": seg_audio, "labels": lbl, "start": start, "file_name": ""})
        start += WINDOW_SIZE * WINDOW_OVERLAP
    return windows

# ── DER ───────────────────────────────────────────────────────────────────────
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
    return {"FA": fa/ref*100, "MD": md/ref*100, "SC": sc/ref*100, "DER": (fa+md+sc)/ref*100}

# ── Inference ─────────────────────────────────────────────────────────────────
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

# ── Gather sessions ───────────────────────────────────────────────────────────
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

print(f"Found {len(sessions)} sessions")

# ── Pre-process ALL sessions ONCE ─────────────────────────────────────────────
print("\nPre-processing all sessions (one-time)...")
session_data = {}
session_gt   = {}

for name, mp3, ann, ann_type in tqdm(sessions):
    segs = parse_json(ann) if ann_type == "json" else parse_srt(ann)
    session_data[name] = preprocess_session(mp3, segs)
    session_gt[name]   = (
        [(s, e) for lbl, s, e in segs if lbl == "c"],
        [(s, e) for lbl, s, e in segs if lbl == "a"],
        mp3,
    )

print(f"Total windows: {sum(len(v) for v in session_data.values())}")

# ── Load pretrained state dict ONCE ──────────────────────────────────────────
print("\nLoading pretrained checkpoint (one-time)...")
_ckpt = torch.load(PRETRAINED, map_location="cpu")
# embed_positions has a dynamic shape; forward() sets it automatically — skip it here
pretrained_state = {k: v for k, v in _ckpt["state_dict"].items()
                    if "embed_positions" not in k}
del _ckpt
torch.cuda.empty_cache()

# ── LOO loop ──────────────────────────────────────────────────────────────────
all_results = []

for fold_idx, (test_name, test_mp3, _, _) in enumerate(sessions):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/{len(sessions)}: test = {test_name}")
    print(f"{'='*60}")

    # Build dataloaders from pre-cached windows
    train_windows = [w for name, wins in session_data.items()
                     if name != test_name for w in wins]
    val_windows   = session_data[test_name]

    train_dl = DataLoader(DatasetGenerator(train_windows, len(train_windows), is_test=False),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dl   = DataLoader(DatasetGenerator(val_windows, len(val_windows), is_test=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build model without calling Classifier.__init__ (avoids preprocess_all_files)
    hparams = {**HPARAMS, "name": f"loo_fast_fold{fold_idx+1:02d}"}
    model = Classifier.__new__(Classifier)
    pl.LightningModule.__init__(model)
    model.save_hyperparameters(hparams)
    model.model            = WhisperWrapper(pretrained_model=hparams["embedding"],
                                            lora_rank=hparams["lora_rank"])
    model.criterion        = torch.nn.NLLLoss()
    model.prediction_dict  = defaultdict(list)
    model.training_save    = []
    model.validation_save  = []
    model.testing_save     = []
    # These are needed by train_val_test_split (called indirectly); set them directly
    model.train_data = train_windows
    model.val_data   = val_windows
    model.test_data  = val_windows
    # Load pretrained weights (embed_positions excluded — forward() handles it)
    model.load_state_dict(pretrained_state, strict=False)

    # Train
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints", f"loo_fast_fold{fold_idx+1:02d}")
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name=hparams["name"]),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                            dirpath=ckpt_dir, filename="{epoch}-{val_loss:.3f}"),
        ],
        max_epochs=MAX_EPOCHS,
        devices=[0], accelerator="gpu",
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"Best ckpt: {os.path.basename(best_ckpt)}")
    del model, trainer
    torch.cuda.empty_cache()

    # Evaluate: load best checkpoint weights into a fresh WhisperWrapper
    best_ckpt_data = torch.load(best_ckpt, map_location="cpu")
    whisper_model = WhisperWrapper(pretrained_model=HPARAMS["embedding"], lora_rank=LORA_RANK)
    # Extract only WhisperWrapper keys (strip "model." prefix, skip embed_positions)
    model_state = {k[len("model."):]: v for k, v in best_ckpt_data["state_dict"].items()
                   if k.startswith("model.") and "embed_positions" not in k}
    whisper_model.load_state_dict(model_state, strict=False)
    whisper_model = whisper_model.cuda()
    whisper_model.eval()

    child_gt, adult_gt, mp3_path = session_gt[test_name]
    _audio, _sr = torchaudio.load(mp3_path)
    duration = _audio.shape[1] / _sr

    with torch.no_grad():
        child_pred, adult_pred, _ = run_inference(mp3_path, whisper_model)

    del whisper_model, best_ckpt_data
    torch.cuda.empty_cache()

    result = compute_der(child_pred, adult_pred, child_gt, adult_gt, duration)
    if result:
        result["session"] = test_name
        all_results.append(result)
        print(f"  FA={result['FA']:.1f}%  MD={result['MD']:.1f}%  SC={result['SC']:.1f}%  DER={result['DER']:.1f}%")
        with open(os.path.join(RESULTS_DIR, f"fold{fold_idx+1:02d}_{test_name}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print("  WARNING: no reference speech found, skipping DER")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("LOO Cross-Validation Summary")
print(f"{'='*60}")
print(f"{'Session':<42} {'FA':>6} {'MD':>6} {'SC':>6} {'DER':>6}")
print("-"*62)
for r in all_results:
    print(f"{r['session']:<42} {r['FA']:>5.1f}% {r['MD']:>5.1f}% {r['SC']:>5.1f}% {r['DER']:>5.1f}%")

if all_results:
    print("-"*62)
    avgs = {k: np.mean([r[k] for r in all_results]) for k in ["FA", "MD", "SC", "DER"]}
    print(f"{'Average':<42} {avgs['FA']:>5.1f}% {avgs['MD']:>5.1f}% {avgs['SC']:>5.1f}% {avgs['DER']:>5.1f}%")
    summary = {"sessions": all_results, "average": avgs}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {RESULTS_DIR}/summary.json")
