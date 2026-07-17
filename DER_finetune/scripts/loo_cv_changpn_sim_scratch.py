"""
LOO Cross-Validation with ChangPN Segment-Level Simulated Conversations (from scratch).

Pipeline:
  Phase 0 — Load ChangPN segment pools (CHI + EXP utterance clips from .cha timestamps)
  Phase 1 — Pre-train on 50,000 Chinese simulated 10s conversations
             (Paper's Algorithm 1: exponential silence, MUSAN noise, p_overlap=0.1)
             Starting point: base Whisper model (NO AudioSet checkpoint)
  Phase 2 — ASD LOO CV from ChangPN-sim-scratch checkpoint

Variant: Bi-LSTM classifier head (replaces the 1D-CNN model_seq) + Stage 1
         class-weighted NLLLoss (balances silence/child/adult/overlap frames,
         estimated from the simulated-conversation distribution). Stage 2
         keeps unweighted NLLLoss.

Resume-safe:
  - Phase 1 skipped if any .ckpt exists in SIM_SCRATCH_CKPT_DIR
  - Completed LOO folds detected by JSON files in SIM_SCRATCH_RESULTS_DIR

Usage:
  cd child-adult-diarization/whisper-modeling
  D:/Anaconda/envs/ASD/python.exe scripts/loo_cv_changpn_sim_scratch.py
"""
import sys, os, json, re, glob, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch, torchaudio
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
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

# ── Mel pre-computation ────────────────────────────────────────────────────────
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
MEL_LEN = 1000   # frames for 10s at hop_length=160
ENC_LEN = 500    # after Whisper encoder CNN


# ── WhisperWrapperFast ─────────────────────────────────────────────────────────
class WhisperWrapperFast(WhisperWrapper):
    """Pad 1000-frame mel to 3000 (new transformers requirement), keep first 500 output frames."""

    def forward(self, x, length=None):
        if x.dim() == 3:
            features = torch.nn.functional.pad(x, (0, 3000 - x.shape[2])).cuda()
        else:
            new_x = [x[i].detach().cpu().numpy() for i in range(len(x))]
            features = self.feature_extractor(
                new_x, return_tensors="pt", sampling_rate=16000
            ).input_features.cuda()

        self.backbone_model.encoder.embed_positions = \
            self.backbone_model.encoder.embed_positions.from_pretrained(
                self.embed_positions)

        hidden_states = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states

        stacked = torch.stack(hidden_states, dim=0)[1:]           # [L, B, 1500, D]
        stacked = stacked[:, :, :ENC_LEN, :].contiguous()         # [L, B, 500, D]
        _, *shape = stacked.shape
        stacked = stacked.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = torch.nn.functional.softmax(self.weights, dim=-1)
        weighted = (norm_weights.unsqueeze(-1) * stacked).sum(dim=0)
        return self.model_seq(weighted.view(*shape).transpose(1, 2))


# ── Constants ─────────────────────────────────────────────────────────────────
CHANGPN_AUDIO_DIR      = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Talkbank/ChangPN/audio"
CHANGPN_TRANSCRIPT_DIR = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Talkbank/ChangPN/transcript"

MUSAN_DIR              = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/musan/noise/"
SIM_SCRATCH_CKPT_DIR   = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling/checkpoints/changpn_sim_pretrain_scratch_bilstm"
SIM_SCRATCH_RESULTS_DIR = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/loo_results_changpn_sim_scratch_bilstm"

INPUT_DIR  = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Fine_tune"
SAVE_DIR   = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling"

# Simulation hyper-parameters (matching paper's Algorithm 1)
SAMPLE_RATE    = 16000
MAX_LENGTH     = 10
P_OVERLAP      = 0.1
P_CHILD        = 0.4
P_START        = 0.5
BETA_INTRA     = 1.0
BETA_INTER     = 0.8
SNR_LIST       = [5, 10, 15, 20]
MIN_SEG_DUR    = 0.3
MAX_SEG_DUR    = 10.0

SIM_TRAIN_N    = 50_000
SIM_VAL_N      = 5_000
SIM_MAX_EPOCHS = 30
SIM_PATIENCE   = 10

# ASD LOO hyper-parameters
ASD_MAX_EPOCHS = 20
ASD_PATIENCE   = 7
WINDOW_SIZE    = 10
COLLAR         = 0.1
BATCH_SIZE     = 16
LR             = 1e-4
LORA_RANK      = 8

os.makedirs(SIM_SCRATCH_CKPT_DIR, exist_ok=True)
os.makedirs(SIM_SCRATCH_RESULTS_DIR, exist_ok=True)

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


# ── Label helpers ─────────────────────────────────────────────────────────────
def label_map(label):
    if label == "si": return 0
    if "c" in label:  return 1
    if "a" in label:  return 2
    return 3


# ── CHA parser ────────────────────────────────────────────────────────────────
def parse_cha(cha_path):
    BULLET = '\x15'
    segments = []
    with open(cha_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not (line.startswith('*CHI:') or line.startswith('*EXP:')):
                continue
            lbl = 'c' if line.startswith('*CHI:') else 'a'
            parts = line.split(BULLET)
            if len(parts) >= 3:
                ts = parts[1].strip()
                if '_' in ts:
                    try:
                        s_ms, e_ms = ts.split('_')
                        s, e = int(s_ms) / 1000.0, int(e_ms) / 1000.0
                        if e > s:
                            segments.append((lbl, s, e))
                    except ValueError:
                        pass
    return segments


# ── Session scanners ──────────────────────────────────────────────────────────
def load_changpn_sessions():
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
        lbl = role_map.get(seg.get("speaker", ""), "a")
        s, e = float(seg["start"]), float(seg["end"])
        if e > s:
            segs.append((lbl, s, e))
    return segs


def load_asd_sessions():
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


# ── Audio helpers ─────────────────────────────────────────────────────────────
def load_audio_np(path):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio[0].numpy()


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
            mel = FEATURE_EXTRACTOR(seg_np, return_tensors="pt",
                                    sampling_rate=SAMPLE_RATE,
                                    max_length=len(seg_np)).input_features[0]
            windows.append({"audio": mel, "labels": lbl, "start": start, "file_name": ""})
        start += WINDOW_SIZE * WINDOW_OVERLAP
    return windows


# ── Simulated conversation helpers ────────────────────────────────────────────
def combine_child_adult_rng(child_audio_set, adult_audio_set, rng):
    pos, audio, speaker_prev, speaker = 0, [], "None", None
    labels = []
    child_audio_set = list(child_audio_set)
    adult_audio_set = list(adult_audio_set)
    child_orig = list(child_audio_set)
    adult_orig = list(adult_audio_set)

    max_samples = MAX_LENGTH * SAMPLE_RATE

    if rng.rand() < 0.2:
        return torch.zeros(1, max_samples), []

    if rng.rand() < P_START:
        if rng.rand() < P_CHILD:
            idx = rng.randint(max(1, len(child_audio_set)))
            audio_add = child_audio_set.pop(idx % len(child_audio_set)).clone()
            speaker_prev = "c"
        else:
            idx = rng.randint(max(1, len(adult_audio_set)))
            audio_add = adult_audio_set.pop(idx % len(adult_audio_set)).clone()
            speaker_prev = "a"
        if audio_add.size(1) > 0:
            start_idx = rng.randint(max(1, audio_add.size(1)))
            audio.append(audio_add[:, start_idx:])
            pos += audio_add.size(1) - start_idx
            labels.append([speaker_prev, 0, round(pos / SAMPLE_RATE, 3)])

    while pos < max_samples:
        if not child_audio_set:
            child_audio_set = list(child_orig)
        if not adult_audio_set:
            adult_audio_set = list(adult_orig)

        if rng.rand() < P_CHILD:
            idx = rng.randint(len(child_audio_set))
            audio_add = child_audio_set.pop(idx).clone()
            speaker = "c"
        else:
            idx = rng.randint(len(adult_audio_set))
            audio_add = adult_audio_set.pop(idx).clone()
            speaker = "a"

        if speaker_prev == speaker:
            sil = int(rng.exponential(BETA_INTRA) * SAMPLE_RATE)
            audio.append(torch.zeros(1, sil))
            pos += sil
            if pos < max_samples:
                audio.append(audio_add)
                pos += audio_add.size(1)
                labels.append([speaker,
                               round((pos - audio_add.size(1)) / SAMPLE_RATE, 3),
                               min(MAX_LENGTH, round(pos / SAMPLE_RATE, 3))])
        elif rng.rand() > P_OVERLAP or speaker_prev in ("o", "None"):
            sil = int(rng.exponential(BETA_INTER) * SAMPLE_RATE)
            audio.append(torch.zeros(1, sil))
            pos += sil
            if pos < max_samples:
                audio.append(audio_add)
                pos += audio_add.size(1)
                labels.append([speaker,
                               round((pos - audio_add.size(1)) / SAMPLE_RATE, 3),
                               min(MAX_LENGTH, round(pos / SAMPLE_RATE, 3))])
        else:
            prev_size = audio[-1].size(1) if audio else 0
            if prev_size > 0:
                ov_start = rng.randint(prev_size)
                ov_end   = min(ov_start + audio_add.size(1), prev_size)
                s_idx    = pos - prev_size + ov_start
                e_idx    = s_idx + audio_add.size(1)
                audio[-1][:, ov_start:ov_end] += audio_add[:, :ov_end - ov_start]
                leftover = audio_add[:, ov_end - ov_start:]
                pos += leftover.size(1)
                audio.append(leftover)
                labels.append([speaker,
                               round(s_idx / SAMPLE_RATE, 3),
                               min(MAX_LENGTH, round(e_idx / SAMPLE_RATE, 3))])
                speaker = "o"
            else:
                audio.append(audio_add)
                pos += audio_add.size(1)
                labels.append([speaker,
                               round((pos - audio_add.size(1)) / SAMPLE_RATE, 3),
                               min(MAX_LENGTH, round(pos / SAMPLE_RATE, 3))])

        speaker_prev = speaker

    combined = torch.cat(audio, dim=1)
    combined = combined[:, :max_samples]
    return combined, labels


def add_noise_rng(audio, noise, snr, rng):
    audio_size, noise_size = audio.size(1), noise.size(1)
    non_sil = torch.sum(audio != 0) / audio_size
    if non_sil == 0:
        return audio
    audio_power = torch.sum(audio ** 2) / audio_size
    noise_power = torch.sum(noise ** 2) / noise_size
    if noise_size < audio_size:
        repeats = audio_size // noise_size + 1
        noise = torch.cat([noise] * repeats, dim=1)
        noise_size = noise.size(1)
    start = rng.randint(noise_size - audio_size) if noise_size > audio_size else 0
    noise = noise[:, start:start + audio_size]
    scale = torch.sqrt(audio_power / (noise_power + 1e-8)) / (10 ** (snr / 10))
    return audio + noise * scale


def sim_labels_to_frames(labels, duration=float(MAX_LENGTH)):
    seg_list = sorted([{"label": str(lbl), "start": float(s), "end": float(e)}
                      for lbl, s, e in labels if float(e) > float(s)],
                     key=lambda x: x["start"])
    filled, prev_end = [], 0.0
    for seg in seg_list:
        if seg["start"] > prev_end + 0.01:
            filled.append({"label": "si", "start": prev_end, "end": seg["start"]})
        filled.append(seg)
        prev_end = seg["end"]
    if prev_end < duration:
        filled.append({"label": "si", "start": prev_end, "end": duration})
    if not filled:
        filled = [{"label": "si", "start": 0.0, "end": duration}]

    frame_labels, curr_time, seg_idx = [], FRAME_SIZE / 2, 0
    while curr_time < duration and seg_idx < len(filled):
        frame_labels.append(label_map(filled[seg_idx]["label"]))
        if curr_time > filled[seg_idx]["end"]:
            seg_idx += 1
        curr_time += FRAME_STRIDE

    return torch.tensor(frame_labels, dtype=torch.long)


# ── Class weighting (Stage 1 only) ──────────────────────────────────────────────
def compute_class_weights_from_sim(child_pool, adult_pool, num_samples=2000, seed=42, num_classes=4):
    """sklearn-style 'balanced' class weights, estimated from simulated conversations.

    Reuses combine_child_adult_rng() + sim_labels_to_frames() to get frame labels
    without running add_noise_rng()/FEATURE_EXTRACTOR (cheap: no mel extraction).
    """
    child_keys = list(child_pool.keys())
    adult_keys = list(adult_pool.keys())
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for idx in range(num_samples):
        rng = np.random.RandomState(seed + idx)
        child_spk = child_keys[rng.randint(len(child_keys))]
        adult_spk = adult_keys[rng.randint(len(adult_keys))]
        child_set = list(child_pool[child_spk])
        adult_set = list(adult_pool[adult_spk])
        _, labels = combine_child_adult_rng(child_set, adult_set, rng)
        frame_labels = sim_labels_to_frames(labels)
        counts += torch.bincount(frame_labels, minlength=num_classes).double()
    weights = torch.where(counts > 0, counts.sum() / (num_classes * counts), torch.zeros_like(counts))
    return weights.float()


# ── Segment pool loader ───────────────────────────────────────────────────────
def load_changpn_pools():
    child_pool = defaultdict(list)
    adult_pool = defaultdict(list)
    total_chi = total_exp = skip_sessions = skip_segs = 0

    sessions = load_changpn_sessions()
    print(f"Found {len(sessions)} ChangPN sessions for pool extraction")

    for session_id, mp3_path, cha_path in tqdm(sessions, desc="Loading segments"):
        segs = parse_cha(cha_path)
        if not segs:
            skip_sessions += 1
            continue
        try:
            audio, sr = torchaudio.load(mp3_path)
        except Exception as exc:
            print(f"  Cannot load {mp3_path}: {exc}")
            skip_sessions += 1
            continue

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio_len = audio.size(1)

        for lbl, start_s, end_s in segs:
            dur = end_s - start_s
            if dur < MIN_SEG_DUR:
                skip_segs += 1
                continue
            end_s = min(end_s, start_s + MAX_SEG_DUR)
            s_f = int(start_s * SAMPLE_RATE)
            e_f = int(end_s   * SAMPLE_RATE)
            if e_f > audio_len or s_f >= audio_len:
                skip_segs += 1
                continue
            clip = audio[:, s_f:e_f].clone()
            if lbl == 'c':
                child_pool[session_id].append(clip)
                total_chi += 1
            else:
                adult_pool[session_id].append(clip)
                total_exp += 1

    child_pool = {k: v for k, v in child_pool.items() if v}
    adult_pool = {k: v for k, v in adult_pool.items() if v}

    print(f"\n=== Pool Statistics ===")
    print(f"Child pool : {len(child_pool):3d} speakers, {total_chi:5d} segments")
    print(f"Adult pool : {len(adult_pool):3d} speakers, {total_exp:5d} segments")
    print(f"Skipped    : {skip_sessions} sessions, {skip_segs} segments (too short / OOB)")
    return child_pool, adult_pool, total_chi, total_exp


def load_musan_noise():
    noise_files = glob.glob(MUSAN_DIR + '/**/*.wav', recursive=True)
    noise_list = []
    for f in noise_files:
        x, sr = torchaudio.load(f)
        if sr != SAMPLE_RATE:
            x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        if x.size(1) > 0:
            noise_list.append(x)
    print(f"Loaded {len(noise_list)} MUSAN noise files")
    return noise_list


# ── Simulated conversation dataset ────────────────────────────────────────────
class SimConvDataset(Dataset):
    def __init__(self, child_pool, adult_pool, noise_list, num_conversations, seed=42):
        self.child_pool   = child_pool
        self.adult_pool   = adult_pool
        self.noise_list   = noise_list
        self.n            = num_conversations
        self.seed         = seed
        self.child_keys   = list(child_pool.keys())
        self.adult_keys   = list(adult_pool.keys())

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx)

        child_spk = self.child_keys[rng.randint(len(self.child_keys))]
        adult_spk = self.adult_keys[rng.randint(len(self.adult_keys))]
        noise     = self.noise_list[rng.randint(len(self.noise_list))]

        child_set = list(self.child_pool[child_spk])
        adult_set = list(self.adult_pool[adult_spk])

        combined, labels = combine_child_adult_rng(child_set, adult_set, rng)
        snr          = rng.choice(SNR_LIST)
        final_audio  = add_noise_rng(combined, noise, snr, rng)

        mel = FEATURE_EXTRACTOR(
            final_audio[0].numpy(), return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
            max_length=MAX_LENGTH * SAMPLE_RATE
        ).input_features[0]

        label_tensor = sim_labels_to_frames(labels)

        return mel, label_tensor


# ── DER & inference ───────────────────────────────────────────────────────────
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


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(name, pretrained_state, train_data=None, val_data=None, class_weights=None):
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
    model.train_data      = train_data or []
    model.val_data        = val_data   or []
    model.test_data       = val_data   or []
    model.load_state_dict(pretrained_state, strict=False)
    # Patch any WhisperAttention layers that have config=None
    _backbone_cfg = model.model.backbone_model.config
    for _layer in model.model.backbone_model.encoder.layers:
        if hasattr(_layer, 'self_attn') and getattr(_layer.self_attn, 'config', None) is None:
            _layer.self_attn.config = _backbone_cfg
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Load ChangPN segment pools + MUSAN noise
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 0: Load ChangPN Segment Pools")
print("="*60)

child_pool, adult_pool, n_child_segs, n_adult_segs = load_changpn_pools()
noise_list = load_musan_noise()

print(f"\nSimulation plan:")
print(f"  Train conversations : {SIM_TRAIN_N:,}")
print(f"  Val   conversations : {SIM_VAL_N:,}")
print(f"  Child speakers      : {len(child_pool)}")
print(f"  Adult speakers      : {len(adult_pool)}")
print(f"  Child segments      : {n_child_segs}")
print(f"  Adult segments      : {n_adult_segs}")
print(f"  MUSAN noise files   : {len(noise_list)}")

# Class weights (Stage 1 only) — balance silence/child/adult/overlap frames,
# estimated from simulated conversations
class_weights = compute_class_weights_from_sim(child_pool, adult_pool)
print("\nStage 1 class weights (balanced, estimated from 2000 simulated conversations):")
for cname, w in zip(["silence", "child", "adult", "overlap"], class_weights.tolist()):
    print(f"  {cname:<8}: weight={w:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Pre-train on 50,000 simulated Chinese conversations (from scratch)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1: Pre-train on Chinese Simulated Conversations (base Whisper, no AudioSet)")
print("="*60)

existing_ckpts = glob.glob(os.path.join(SIM_SCRATCH_CKPT_DIR, "*.ckpt"))
if existing_ckpts:
    sim_best_ckpt = max(existing_ckpts, key=os.path.getmtime)
    print(f"Found existing checkpoint: {os.path.basename(sim_best_ckpt)}")
    print("Skipping Phase 1.")
else:
    print("Building model from base Whisper (no AudioSet checkpoint)...")
    model = build_model("changpn_sim_pretrain_scratch_bilstm", {}, class_weights=class_weights)

    train_ds = SimConvDataset(child_pool, adult_pool, noise_list,
                              num_conversations=SIM_TRAIN_N, seed=42)
    val_ds   = SimConvDataset(child_pool, adult_pool, noise_list,
                              num_conversations=SIM_VAL_N,   seed=99999)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name="changpn_sim_pretrain_scratch_bilstm"),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=SIM_PATIENCE, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                            dirpath=SIM_SCRATCH_CKPT_DIR,
                            filename="{epoch}-{val_loss:.3f}"),
        ],
        max_epochs=SIM_MAX_EPOCHS,
        devices=[0], accelerator="gpu",
        precision="16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    sim_best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nPhase 1 done. Best ckpt: {os.path.basename(sim_best_ckpt)}")
    del model, trainer, train_ds, val_ds
    torch.cuda.empty_cache()

# Free pool memory before LOO
del child_pool, adult_pool, noise_list


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ASD LOO Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 2: ASD LOO Cross-Validation")
print(f"Starting checkpoint: {os.path.basename(sim_best_ckpt)}")
print("="*60)

print("\nLoading ChangPN-sim-scratch checkpoint...")
_ckpt = torch.load(sim_best_ckpt, map_location="cpu")
sim_state = {k: v for k, v in _ckpt["state_dict"].items()
             if "embed_positions" not in k}
del _ckpt
torch.cuda.empty_cache()

print("Scanning ASD sessions...")
asd_sessions = load_asd_sessions()
print(f"Found {len(asd_sessions)} ASD sessions")

all_results = []
for fold_idx, (test_name, test_mp3, test_ann, ann_type) in enumerate(asd_sessions):
    result_path = os.path.join(SIM_SCRATCH_RESULTS_DIR, f"{test_name}.json")
    if os.path.exists(result_path):
        print(f"[{fold_idx+1:02d}/{len(asd_sessions)}] {test_name} — already done, loading.")
        with open(result_path) as f:
            all_results.append(json.load(f))
        continue

    print(f"\n[{fold_idx+1:02d}/{len(asd_sessions)}] LOO fold: test={test_name}")

    train_sessions = [s for s in asd_sessions if s[0] != test_name]
    random.seed(42)
    random.shuffle(train_sessions)
    val_n = max(1, len(train_sessions) // 6)
    val_sessions_fold   = train_sessions[:val_n]
    train_sessions_fold = train_sessions[val_n:]

    train_windows, val_windows = [], []
    for _, mp3, ann, atype in train_sessions_fold:
        segs = parse_json(ann) if atype == "json" else parse_srt(ann)
        if segs:
            train_windows.extend(make_windows(load_audio_np(mp3), segs))
    for _, mp3, ann, atype in val_sessions_fold:
        segs = parse_json(ann) if atype == "json" else parse_srt(ann)
        if segs:
            val_windows.extend(make_windows(load_audio_np(mp3), segs))

    print(f"  Train windows: {len(train_windows)} | Val windows: {len(val_windows)}")

    fold_name = f"loo_changpn_sim_scratch_bilstm_fold{fold_idx+1:02d}"
    model = build_model(fold_name, sim_state, train_windows, val_windows)

    train_dl = DataLoader(
        DatasetGenerator(train_windows, len(train_windows), is_test=False),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_dl = DataLoader(
        DatasetGenerator(val_windows, len(val_windows), is_test=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints", fold_name)
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name=fold_name),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=ASD_PATIENCE, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                            dirpath=ckpt_dir, filename="{epoch}-{val_loss:.3f}"),
        ],
        max_epochs=ASD_MAX_EPOCHS,
        devices=[0], accelerator="gpu",
        precision="16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=5,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = trainer.checkpoint_callback.best_model_path
    best_ckpt = torch.load(best_path, map_location="cpu")
    best_state = {k: v for k, v in best_ckpt["state_dict"].items()
                  if "embed_positions" not in k}
    del best_ckpt

    eval_model = build_model(fold_name + "_eval", best_state)
    eval_model.model.eval()
    eval_model.model.cuda()

    child_pred, adult_pred, dur = run_inference(test_mp3, eval_model.model)

    segs_gt = parse_json(test_ann) if ann_type == "json" else parse_srt(test_ann)
    child_gt = [(s, e) for lbl, s, e in segs_gt if lbl == "c"]
    adult_gt = [(s, e) for lbl, s, e in segs_gt if lbl == "a"]
    metrics = compute_der(child_pred, adult_pred, child_gt, adult_gt, dur)

    result = {
        "session": test_name,
        "fold": fold_idx + 1,
        "metrics": metrics,
        "num_train_windows": len(train_windows),
        "num_val_windows": len(val_windows),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    if metrics:
        print(f"  DER={metrics['DER']:.1f}%  FA={metrics['FA']:.1f}%  "
              f"MD={metrics['MD']:.1f}%  SC={metrics['SC']:.1f}%")
    all_results.append(result)

    del model, trainer, eval_model, train_windows, val_windows
    torch.cuda.empty_cache()


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS — ChangPN Sim (base Whisper, no AudioSet) → ASD LOO")
print("="*60)
valid = [r for r in all_results if r.get("metrics")]
if valid:
    avg_der = np.mean([r["metrics"]["DER"] for r in valid])
    avg_fa  = np.mean([r["metrics"]["FA"]  for r in valid])
    avg_md  = np.mean([r["metrics"]["MD"]  for r in valid])
    avg_sc  = np.mean([r["metrics"]["SC"]  for r in valid])
    print(f"{'Session':<25} {'DER':>6} {'FA':>6} {'MD':>6} {'SC':>6}")
    print("-" * 51)
    for r in valid:
        m = r["metrics"]
        print(f"{r['session']:<25} {m['DER']:6.1f} {m['FA']:6.1f} {m['MD']:6.1f} {m['SC']:6.1f}")
    print("-" * 51)
    print(f"{'Average':<25} {avg_der:6.1f} {avg_fa:6.1f} {avg_md:6.1f} {avg_sc:6.1f}")
    print(f"\nPool: {n_child_segs} child segs + {n_adult_segs} adult segs → "
          f"{SIM_TRAIN_N:,} train + {SIM_VAL_N:,} val conversations")
    print(f"Mean DER: {avg_der:.1f}%")

    summary = {
        "sessions": valid,
        "average": {"FA": avg_fa, "MD": avg_md, "SC": avg_sc, "DER": avg_der},
        "method": "ChangPN sim (base Whisper, BiLSTM head + Stage1 class weighting) → ASD LOO fine-tune",
        "sim_checkpoint": os.path.basename(sim_best_ckpt),
        "n_child_segs": n_child_segs,
        "n_adult_segs": n_adult_segs,
    }
    with open(os.path.join(SIM_SCRATCH_RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {SIM_SCRATCH_RESULTS_DIR}/summary.json")
