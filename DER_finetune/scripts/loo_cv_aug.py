"""
LOO Cross-Validation with Pitch Augmentation.
Each training session is augmented into 4 variants:
  1. Original (no shift)
  2. Adult speech pitched down -4 semitones (adult sounds child-like: hard example)
  3. Child speech pitched down -3 semitones (child sounds adult-like: hard example)
  4. Entire session pitched up +3 semitones (pitch diversity)

Test sessions are never augmented.

Usage:
  python scripts/loo_cv_aug.py
"""
import sys, os, json, re, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch, torchaudio, librosa
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

# Pre-computed mel spectrogram extractor (loaded once, runs on CPU during preprocessing)
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
MEL_LEN = 1000   # frames for 10-second audio at hop=160: 160000/160=1000
ENC_LEN = 500    # after Whisper encoder CNN: (1000-1)//2+1=500


class WhisperWrapperFast(WhisperWrapper):
    """WhisperWrapper accepting pre-computed mel [B,80,MEL_LEN] or raw audio.

    New transformers requires mel to be exactly 3000 frames.  We pad 1000-frame
    mel → 3000, run encoder → 1500 frames, keep first ENC_LEN=500 (actual 10s).
    """

    def forward(self, x, length=None):
        if x.dim() == 3:
            # Pre-computed mel [B, 80, 1000] → pad to 3000
            features = torch.nn.functional.pad(x, (0, 3000 - x.shape[2])).cuda()
        else:
            # Raw audio: use built-in feature extractor (pads to 30s = 3000 frames)
            new_x = [x[i].detach().cpu().numpy() for i in range(len(x))]
            features = self.feature_extractor(
                new_x, return_tensors="pt", sampling_rate=16000
            ).input_features.cuda()

        # Full embed_positions [1500, D] for 3000-frame mel
        self.backbone_model.encoder.embed_positions = \
            self.backbone_model.encoder.embed_positions.from_pretrained(
                self.embed_positions)

        hidden_states = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states

        # Encoder outputs 1500 frames; keep first ENC_LEN=500 (actual audio)
        stacked = torch.stack(hidden_states, dim=0)[1:]          # [L, B, 1500, D]
        stacked = stacked[:, :, :ENC_LEN, :].contiguous()        # [L, B, 500, D]

        _, *shape = stacked.shape
        stacked = stacked.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = torch.nn.functional.softmax(self.weights, dim=-1)
        weighted = (norm_weights.unsqueeze(-1) * stacked).sum(dim=0)
        return self.model_seq(weighted.view(*shape).transpose(1, 2))

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR   = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Fine_tune"
RESULTS_DIR = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/loo_results_aug"
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

# ── Window creation from numpy array ─────────────────────────────────────────
def make_windows(audio_np, segments):
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
            # Pre-compute mel spectrogram on CPU once → store [80, MEL_LEN] tensor
            mel = FEATURE_EXTRACTOR(seg_np, return_tensors="pt",
                                    sampling_rate=SAMPLE_RATE,
                                    max_length=len(seg_np)).input_features[0]
            windows.append({"audio": mel, "labels": lbl,
                            "start": start, "file_name": ""})
        start += WINDOW_SIZE * WINDOW_OVERLAP
    return windows

# ── Pitch augmentation ────────────────────────────────────────────────────────
def pitch_shift_range(audio_np, sr, start_sec, end_sec, n_steps):
    s, e = int(start_sec * sr), min(int(end_sec * sr), len(audio_np))
    if e <= s:
        return audio_np
    result = audio_np.copy()
    result[s:e] = librosa.effects.pitch_shift(audio_np[s:e], sr=sr, n_steps=n_steps)[:e-s]
    return result

def augment(audio_np, segments, mode):
    """
    mode: 'adult_down' | 'child_down' | 'all_up'
    """
    if mode == 'all_up':
        return librosa.effects.pitch_shift(audio_np, sr=SAMPLE_RATE, n_steps=3)
    result = audio_np.copy()
    for lbl, s, e in segments:
        if mode == 'adult_down' and lbl == 'a':
            result = pitch_shift_range(result, SAMPLE_RATE, s, e, n_steps=-4)
        elif mode == 'child_down' and lbl == 'c':
            result = pitch_shift_range(result, SAMPLE_RATE, s, e, n_steps=-3)
    return result

def load_audio(mp3_path):
    audio, sr = torchaudio.load(mp3_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio[0].numpy()

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
    return {"FA": fa/ref*100, "MD": md/ref*100, "SC": sc/ref*100,
            "DER": (fa+md+sc)/ref*100}

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
print("\nPre-processing all sessions (original + 3 augmented variants)...")
session_orig = {}   # name → original windows (for test/val)
session_aug  = {}   # name → original + augmented windows (for train)
session_gt   = {}   # name → (child_gt, adult_gt, mp3_path)

for name, mp3, ann, ann_type in tqdm(sessions):
    segs = parse_json(ann) if ann_type == "json" else parse_srt(ann)
    audio_np = load_audio(mp3)
    # Original windows
    orig_wins = make_windows(audio_np, segs)
    session_orig[name] = orig_wins
    # Augmented windows: original + 3 variants
    aug_wins = list(orig_wins)
    for mode in ['adult_down', 'child_down', 'all_up']:
        aug_np = augment(audio_np, segs, mode)
        aug_wins.extend(make_windows(aug_np, segs))
    session_aug[name] = aug_wins
    session_gt[name]  = (
        [(s, e) for lbl, s, e in segs if lbl == "c"],
        [(s, e) for lbl, s, e in segs if lbl == "a"],
        mp3,
    )

orig_total = sum(len(v) for v in session_orig.values())
aug_total  = sum(len(v) for v in session_aug.values())
print(f"Original windows: {orig_total} | Augmented total: {aug_total} ({aug_total//orig_total}x)")

# ── Load pretrained state dict ONCE ──────────────────────────────────────────
print("\nLoading pretrained checkpoint (one-time)...")
_ckpt = torch.load(PRETRAINED, map_location="cpu")
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

    train_windows = [w for name, wins in session_aug.items()
                     if name != test_name for w in wins]
    val_windows   = session_orig[test_name]
    print(f"  Train: {len(train_windows)} windows | Val: {len(val_windows)} windows")

    train_dl = DataLoader(DatasetGenerator(train_windows, len(train_windows), is_test=False),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dl   = DataLoader(DatasetGenerator(val_windows, len(val_windows), is_test=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    hparams = {**HPARAMS, "name": f"loo_aug_fold{fold_idx+1:02d}"}
    model = Classifier.__new__(Classifier)
    pl.LightningModule.__init__(model)
    model.save_hyperparameters(hparams)
    model.model           = WhisperWrapperFast(pretrained_model=hparams["embedding"],
                                               lora_rank=hparams["lora_rank"])
    model.criterion       = torch.nn.NLLLoss()
    model.prediction_dict = defaultdict(list)
    model.training_save   = []
    model.validation_save = []
    model.testing_save    = []
    model.train_data      = train_windows
    model.val_data        = val_windows
    model.test_data       = val_windows
    model.load_state_dict(pretrained_state, strict=False)

    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints", f"loo_aug_fold{fold_idx+1:02d}")
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name=hparams["name"]),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                            dirpath=ckpt_dir, filename="{epoch}-{val_loss:.3f}"),
        ],
        max_epochs=MAX_EPOCHS,
        devices=[0], accelerator="gpu",
        precision="16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"Best ckpt: {os.path.basename(best_ckpt)}")
    del model, trainer
    torch.cuda.empty_cache()

    # Evaluate on original unaugmented test audio
    best_ckpt_data = torch.load(best_ckpt, map_location="cpu")
    whisper_model = WhisperWrapperFast(pretrained_model=HPARAMS["embedding"], lora_rank=LORA_RANK)
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
print("LOO + Pitch Augmentation Results")
print(f"{'='*60}")
print(f"{'Session':<42} {'FA':>6} {'MD':>6} {'SC':>6} {'DER':>6}")
print("-"*62)
for r in all_results:
    print(f"{r['session']:<42} {r['FA']:>5.1f}% {r['MD']:>5.1f}% {r['SC']:>5.1f}% {r['DER']:>5.1f}%")

if all_results:
    print("-"*62)
    avgs = {k: np.mean([r[k] for r in all_results]) for k in ["FA", "MD", "SC", "DER"]}
    print(f"{'Average':<42} {avgs['FA']:>5.1f}% {avgs['MD']:>5.1f}% {avgs['SC']:>5.1f}% {avgs['DER']:>5.1f}%")
    summary = {"sessions": all_results, "average": avgs,
               "augmentation": ["original", "adult_down_4st", "child_down_3st", "all_up_3st"]}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {RESULTS_DIR}/summary.json")
