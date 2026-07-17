"""Breeze-as-backbone diarization — SMALL VALIDATION (1-2 folds).

Purpose: cheaply test whether swapping the diarization backbone from whisper-base
to Breeze-ASR-25 (whisper-large-v2, 32 layers / d=1280) is worth a full 16-fold run.
Compare the DER here against the same folds' whisper-base numbers
(see loo_results_audioset_bilstm_16sess/ for per-fold whisper-base DER).

This is the BiLSTM control's backbone-swap: Breeze encoder + fresh BiLSTM head +
ASD LOO (no ChangPN, no AudioSet ckpt — Breeze brings its own ASR-pretrained
encoder). Architecture support for "breeze" already exists in models/whisper.py.

PORTABLE: paths come from scripts.paths_config (env ASD_DATA_ROOT). Designed to
run on the workstation (single RTX 5090, 32GB) — NOT test-run on the 16GB local
card (Breeze is ~30x heavier per step; will likely OOM at large batch locally).
Tune BREEZE_DIAR_BATCH / BREEZE_DIAR_FOLDS via env vars.

Usage (workstation):
  ASD_DATA_ROOT=/data/asd python scripts/loo_cv_breeze_diar_val.py
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
from transformers import AutoFeatureExtractor

from models.whisper import WhisperWrapper
from dataset_classes.datasets import DatasetGenerator
from dataset_classes.preprocess import FRAME_SIZE, FRAME_STRIDE, WINDOW_OVERLAP
from scripts.convert_output import get_timestamps, majority_filter
from lightning_modules.classifier import Classifier
from scripts.paths_config import FINE_TUNE_DIR, CHECKPOINT_DIR, results_dir

torch.set_float32_matmul_precision('high')

# ── Transformers compatibility shim (same as loo_cv_changpn_scratch.py) ─────────
import transformers.models.whisper.modeling_whisper as _wm
_orig_wa_forward = _wm.WhisperAttention.forward

def _compat_wa_forward(self, *args, **kwargs):
    if getattr(self, 'config', None) is None:
        class _C:
            _attn_implementation = "eager"
        self.config = _C()
    result = _orig_wa_forward(self, *args, **kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1], None
    return result

_wm.WhisperAttention.forward = _compat_wa_forward

# ── Mel pre-computation ────────────────────────────────────────────────────────
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")  # 80-mel (Breeze=v2=80-mel)
ENC_LEN = 500    # first 500 of 1500 encoder frames = actual 10s


class WhisperWrapperFast(WhisperWrapper):
    """Pre-computed-mel / raw-audio wrapper with the mel->3000 padding shim.
    Generalises to Breeze: dims (num_hidden_layers, hidden_size) are read from
    backbone config, so 32 layers / d=1280 work unchanged."""

    def forward(self, x, length=None):
        _dt = next(self.backbone_model.parameters()).dtype  # bf16-true 下模型是 bf16;否則 fp32
        if x.dim() == 3:
            features = torch.nn.functional.pad(x, (0, 3000 - x.shape[2])).cuda().to(_dt)
        else:
            new_x = [x[i].detach().cpu().numpy() for i in range(len(x))]
            features = self.feature_extractor(
                new_x, return_tensors="pt", sampling_rate=16000
            ).input_features.cuda().to(_dt)

        self.backbone_model.encoder.embed_positions = \
            self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions.to(_dt))

        hidden_states = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states

        stacked = torch.stack(hidden_states, dim=0)[1:]
        stacked = stacked[:, :, :ENC_LEN, :].contiguous()
        _, *shape = stacked.shape
        stacked = stacked.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = torch.nn.functional.softmax(self.weights, dim=-1)
        weighted = (norm_weights.unsqueeze(-1) * stacked).sum(dim=0)
        return self.model_seq(weighted.view(*shape).transpose(1, 2))


# ── Config ──────────────────────────────────────────────────────────────────────
INPUT_DIR   = FINE_TUNE_DIR
RESULTS_DIR = results_dir("loo_results_breeze_diar_val")
SAVE_DIR    = os.path.dirname(CHECKPOINT_DIR)   # whisper-modeling dir
EMBEDDING   = os.environ.get("DIAR_EMBEDDING", "breeze")   # e.g. whisper_medium
HEAD_TYPE   = "bilstm"
SAMPLE_RATE = 16000
WINDOW_SIZE = 10
COLLAR      = 0.1
LR          = float(os.environ.get("DIAR_LR", "1e-4"))   # full-FT 大模型需更小(e.g. 1e-5)
LORA_RANK   = 8
MAX_EPOCHS  = int(os.environ.get("BREEZE_DIAR_EPOCHS", "20"))
PATIENCE    = int(os.environ.get("BREEZE_DIAR_PATIENCE", "7"))
BATCH_SIZE  = int(os.environ.get("BREEZE_DIAR_BATCH", "16"))   # smaller: Breeze is big
N_FOLDS     = int(os.environ.get("BREEZE_DIAR_FOLDS", "2"))     # validation: just a couple
# Optional: pick exact 0-based fold indices, e.g. "0,12" = best-fold + worst-fold.
# Defaults to the first N_FOLDS folds.
_idxs = os.environ.get("BREEZE_DIAR_FOLD_IDXS", "").strip()
FOLD_IDXS = ([int(x) for x in _idxs.split(",") if x.strip()] if _idxs
             else list(range(N_FOLDS)))

# ── Clean-subset mode: 排除經診斷確認自動分群品質不良的 session ──
# 用 CLEAN10=1 開啟;預設關閉,完全不影響原本 16 折行為。
# 實際名單請用 CLEAN10_NAMES 環境變數(逗號分隔)指定,本 repo 不內建任何真實 session 名稱。
CLEAN10 = os.environ.get("CLEAN10", "0") == "1"
EXCLUDE_NAMES = ([x for x in os.environ.get("CLEAN10_NAMES", "").split(",") if x]
                 if CLEAN10 else [])
# 無法自動修的壞 case(標註時間錯位 / 未完成 / 覆蓋率過低等):請用 DIAR_EXCLUDE 環境變數
# (逗號分隔的 session 資料夾名稱關鍵字)指定,本 repo 不內建任何真實 session 名稱。
EXCLUDE_CASES = tuple(x for x in os.environ.get("DIAR_EXCLUDE", "").split(",") if x)
# FREEZE_ENCODER=1: 凍結整個 Breeze encoder(含 LoRA),只訓練層加權 + 分類頭。
# 復刻原始 16 折 (41.98%) 的凍結行為,且省一半記憶體(不必保留 32 層 activation)。
# 用獨立輸出目錄/tag,不與訓練 encoder 版撞名。
FREEZE_ENCODER = os.environ.get("FREEZE_ENCODER", "0") == "1"
_frz = "_frozen" if FREEZE_ENCODER else ""
# FULL_FT=1: 全參數微調 —— 解凍整個 encoder 所有權重一起訓練(不只 LoRA)。
FULL_FT = os.environ.get("FULL_FT", "0") == "1"
# GRAD_CKPT: gradient checkpointing,大模型全微調抗 OOM(activation 大幅下降、結果零影響、慢 ~20-40%)。
# 預設在 FULL_FT 時自動開啟;可用 GRAD_CKPT=0 關閉、GRAD_CKPT=1 強制開啟。
GRAD_CKPT = os.environ.get("GRAD_CKPT", "1" if FULL_FT else "0") == "1"
# BF16=1: 純 bf16-true(權重/梯度/optimizer 全 bf16,無 fp32 master)→ 省一半固定記憶體,
# 但有 swamping 數值風險(交接 §5),務必盯 val_loss 是否正常下降。預設用 16-mixed(fp32 master)。
PRECISION = "bf16-true" if os.environ.get("BF16", "0") == "1" else "16-mixed"
# DIAR_TAG: 額外後綴,讓不同 backbone/設定的輸出目錄不撞名(e.g. _wmedium_fullft)。
_ext = os.environ.get("DIAR_TAG", "")
CKPT_TAG = ("_clean10" if CLEAN10 else "") + _frz + _ext
RESULTS_DIR = results_dir(
    ("loo_results_breeze_diar_clean10" if CLEAN10 else "loo_results_breeze_diar_val")
    + _frz + _ext)

os.makedirs(RESULTS_DIR, exist_ok=True)

HPARAMS_BASE = {
    "embedding": EMBEDDING, "lora_rank": LORA_RANK,
    "learning_rate": LR, "weight_decay": 1e-4,
    "optimizer": "Adam", "batch_size": BATCH_SIZE,
    "window_size": WINDOW_SIZE, "accumulate_grad_batches": 1,
    "max_epochs": MAX_EPOCHS, "devices": [0], "seed": 42, "debug": False,
    "head_type": HEAD_TYPE,
    "audio": "", "annotation": "", "name": "",
}


# ── Annotation parsers ──────────────────────────────────────────────────────────
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


_CHILD = {"兒童", "小孩", "child", "Child"}
_ADULT = {"評估者", "測試者", "老師", "個管師", "adult", "Adult"}


def _chunk_offsets(segments, tol=30.0):
    """偵測每個 chunk 是絕對還是相對時間,回傳每個 chunk 應加的 offset(秒)。
    若某 chunk 的 min(start) 明顯小於 chunk_start_ms/1000(差 > tol),判定為相對 → 補 offset。"""
    from collections import defaultdict
    by = defaultdict(list)
    for s in segments:
        by[s.get("chunk_id")].append(s)
    off = {}
    for cid, rows in by.items():
        csm = rows[0].get("chunk_start_ms")
        if csm and csm > 0:
            mn = min(float(r.get("start", 0)) for r in rows)
            off[cid] = csm / 1000.0 if mn < csm / 1000.0 - tol else 0.0
        else:
            off[cid] = 0.0
    return off


def parse_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mapping = data.get("speaker_mapping") or data.get("metadata", {}).get("speaker_mapping", {}) or {}
    segs_raw = data.get("segments", [])
    offs = _chunk_offsets(segs_raw)
    segs = []
    for seg in segs_raw:
        sp = seg.get("speaker", "")
        role = mapping.get(sp, sp)          # 新格式:speaker 本身就是角色;舊格式:用 mapping 翻譯
        if role in _CHILD:   label = "c"
        elif role in _ADULT: label = "a"
        else:                continue        # 廣播 / Unknown / 未指派角色 → 丟棄
        off = offs.get(seg.get("chunk_id"), 0.0)
        s, e = float(seg["start"]) + off, float(seg["end"]) + off
        if e > s:
            segs.append((label, s, e))
    return segs


def label_map(label):
    if label == "si": return 0
    if "c" in label:  return 1
    if "a" in label:  return 2
    return 3


# ── Audio / windowing ─────────────────────────────────────────────────────────
def load_audio(path):
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
            windows.append({"audio": mel, "labels": lbl,
                            "start": start, "file_name": ""})
        start += WINDOW_SIZE * WINDOW_OVERLAP
    return windows


# ── DER & inference ─────────────────────────────────────────────────────────────
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


# ── Session loader & model builder ───────────────────────────────────────────────
def load_asd_sessions():
    sessions = []
    for d in sorted(os.listdir(INPUT_DIR)):
        full = os.path.join(INPUT_DIR, d)
        if not os.path.isdir(full):
            continue
        if any(x in d for x in EXCLUDE_CASES):   # 無法自動修的壞 case(時間錯位/標註未完成)
            continue
        if any(x in d for x in EXCLUDE_NAMES):    # CLEAN10 病態折(預設空,新 GT 不需排除)
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


def build_model(name, train_windows, val_windows):
    """Fresh Breeze backbone + random BiLSTM head + LoRA (no checkpoint init)."""
    hparams = {**HPARAMS_BASE, "name": name}
    model = Classifier.__new__(Classifier)
    pl.LightningModule.__init__(model)
    model.save_hyperparameters(hparams)
    model.model           = WhisperWrapperFast(pretrained_model=hparams["embedding"],
                                               lora_rank=hparams["lora_rank"],
                                               head_type=HEAD_TYPE)
    model.criterion       = torch.nn.NLLLoss()
    model.prediction_dict = defaultdict(list)
    model.training_save   = []
    model.validation_save = []
    model.testing_save    = []
    model.train_data      = train_windows
    model.val_data        = val_windows
    model.test_data       = val_windows
    # Newer transformers requires WhisperAttention.config; patch any None entries.
    _backbone_cfg = model.model.backbone_model.config
    for _layer in model.model.backbone_model.encoder.layers:
        if hasattr(_layer, 'self_attn') and getattr(_layer.self_attn, 'config', None) is None:
            _layer.self_attn.config = _backbone_cfg
    if FREEZE_ENCODER:
        # 凍結整個 encoder(含 LoRA):無梯度穿過 32 層 → 不保留 activation → 省記憶體;
        # 只剩層加權 self.weights + BiLSTM head (model_seq) 可訓練。
        for _p in model.model.backbone_model.parameters():
            _p.requires_grad = False
    elif FULL_FT:
        # 全參數微調:解凍整個 encoder 所有權重(base weights + 任何 LoRA 一起訓練)。
        for _p in model.model.backbone_model.parameters():
            _p.requires_grad = True
    if GRAD_CKPT:
        # gradient checkpointing:抗 OOM 關鍵。use_reentrant=False 較穩(不需輸入 requires_grad)。
        enc = model.model.backbone_model.encoder
        try:
            model.model.backbone_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.model.backbone_model.gradient_checkpointing_enable()
        # 保險:直接在 encoder 上也設好旗標與函式(自訂 encoder.layers 仍走 HF 的 checkpoint 迴圈)
        import functools, torch.utils.checkpoint as _ckpt
        enc.gradient_checkpointing = True
        if getattr(enc, "_gradient_checkpointing_func", None) is None:
            enc._gradient_checkpointing_func = functools.partial(_ckpt.checkpoint, use_reentrant=False)
        print(f"  [build_model] gradient_checkpointing = {enc.gradient_checkpointing}")
    return model


# ════════════════════════════════════════════════════════════════════════════════
# LOO (validation: first N_FOLDS folds only)
# ════════════════════════════════════════════════════════════════════════════════
print("="*60)
print(f"Breeze diarization VALIDATION — {N_FOLDS} folds, head={HEAD_TYPE}, "
      f"batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}")
print(f"INPUT_DIR  = {INPUT_DIR}")
print(f"RESULTS_DIR= {RESULTS_DIR}")
print("="*60)

asd_sessions = load_asd_sessions()
# clean10 模式下,若未明確指定 fold,預設跑完全部(10)折。
if CLEAN10 and not _idxs and "BREEZE_DIAR_FOLDS" not in os.environ:
    FOLD_IDXS = list(range(len(asd_sessions)))
print(f"Found {len(asd_sessions)} ASD sessions; running fold indices {FOLD_IDXS}")

print("\nPre-processing ASD sessions (pre-computing mel)...")
asd_data, asd_gt = {}, {}
for name, mp3, ann, ann_type in tqdm(asd_sessions):
    segs = parse_json(ann) if ann_type == "json" else parse_srt(ann)
    asd_data[name] = make_windows(load_audio(mp3), segs)
    asd_gt[name] = ([(s, e) for lbl, s, e in segs if lbl == "c"],
                    [(s, e) for lbl, s, e in segs if lbl == "a"], mp3)

# ── SINGLE_SPLIT 模式:單一 train/val/test 分割(預設 13/2/1),訓練一次、存權重、評估 test ──
if os.environ.get("SINGLE_SPLIT", "0") == "1":
    import random
    N_VAL  = int(os.environ.get("SPLIT_VAL", "2"))
    N_TEST = int(os.environ.get("SPLIT_TEST", "1"))
    SEED   = int(os.environ.get("SPLIT_SEED", "42"))
    names = [s[0] for s in asd_sessions]
    random.Random(SEED).shuffle(names)
    test_names  = names[:N_TEST]
    val_names   = names[N_TEST:N_TEST + N_VAL]
    train_names = names[N_TEST + N_VAL:]
    print(f"\n{'='*60}\nSINGLE SPLIT (seed={SEED}): "
          f"train={len(train_names)} / val={len(val_names)} / test={len(test_names)}\n{'='*60}")
    print(f"  train: {train_names}")
    print(f"  val  : {val_names}")
    print(f"  test : {test_names}")
    train_windows = [w for nm in train_names for w in asd_data[nm]]
    val_windows   = [w for nm in val_names   for w in asd_data[nm]]
    print(f"  windows — train {len(train_windows)} | val {len(val_windows)}")

    train_dl = DataLoader(DatasetGenerator(train_windows, len(train_windows), is_test=False),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                          drop_last=True, pin_memory=True)
    val_dl = DataLoader(DatasetGenerator(val_windows, len(val_windows), is_test=False),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(f"breeze_diar{CKPT_TAG}_single", train_windows, val_windows)
    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"breeze_diar{CKPT_TAG}_single")
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name=f"breeze_diar{CKPT_TAG}_single"),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                            dirpath=ckpt_dir, filename="{epoch}-{val_loss:.3f}"),
        ],
        max_epochs=MAX_EPOCHS, devices=[0], accelerator="gpu",
        precision=PRECISION, enable_progress_bar=True, log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\n  ✅ Best checkpoint saved: {best_ckpt}")
    del model, trainer
    torch.cuda.empty_cache()

    # 評估 test sessions(可多個)
    best_data = torch.load(best_ckpt, map_location="cpu")
    whisper_model = WhisperWrapperFast(pretrained_model=EMBEDDING, lora_rank=LORA_RANK,
                                       head_type=HEAD_TYPE)
    model_state = {k[len("model."):]: v for k, v in best_data["state_dict"].items()
                   if k.startswith("model.") and "embed_positions" not in k}
    whisper_model.load_state_dict(model_state, strict=False)
    whisper_model = whisper_model.cuda().eval()
    test_results = []
    for test_name in test_names:
        child_gt, adult_gt, mp3_path = asd_gt[test_name]
        _audio, _sr = torchaudio.load(mp3_path)
        duration = _audio.shape[1] / _sr
        with torch.no_grad():
            child_pred, adult_pred, _ = run_inference(mp3_path, whisper_model)
        r = compute_der(child_pred, adult_pred, child_gt, adult_gt, duration) or {}
        r["session"] = test_name
        test_results.append(r)
        if "DER" in r:
            print(f"  test={test_name}  FA={r['FA']:.1f} MD={r['MD']:.1f} "
                  f"SC={r['SC']:.1f} DER={r['DER']:.1f}%")
    avgs = {k: float(np.mean([r[k] for r in test_results if k in r]))
            for k in ["FA", "MD", "SC", "DER"]}
    result = {"average": avgs, "test_results": test_results,
              "train_sessions": train_names, "val_sessions": val_names,
              "test_sessions": test_names, "best_ckpt": best_ckpt,
              "split": f"{len(train_names)}/{len(val_names)}/{len(test_names)}",
              "backbone": EMBEDDING, "head": HEAD_TYPE, "seed": SEED}
    out = os.path.join(RESULTS_DIR, "single_split_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  test 平均({len(test_names)} 個): FA={avgs['FA']:.1f} MD={avgs['MD']:.1f} "
          f"SC={avgs['SC']:.1f} DER={avgs['DER']:.1f}%")
    print(f"  結果 -> {out}\n  權重 -> {best_ckpt}\nDONE.")
    sys.exit(0)

all_results = []
for fold_idx, (test_name, test_mp3, _, _) in enumerate(asd_sessions):
    if fold_idx not in FOLD_IDXS:
        continue
    result_file = os.path.join(RESULTS_DIR, f"fold{fold_idx+1:02d}_{test_name}.json")
    if os.path.exists(result_file):
        print(f"\nFold {fold_idx+1} ({test_name}) already done — skip.")
        with open(result_file, encoding="utf-8") as f:
            all_results.append(json.load(f))
        continue

    print(f"\n{'='*60}\nFold {fold_idx+1} (idx {fold_idx}): test = {test_name}\n{'='*60}")
    train_windows = [w for nm, wins in asd_data.items() if nm != test_name for w in wins]
    val_windows   = asd_data[test_name]
    print(f"  Train: {len(train_windows)} | Val: {len(val_windows)} windows")

    train_dl = DataLoader(DatasetGenerator(train_windows, len(train_windows), is_test=False),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                          drop_last=True, pin_memory=True)
    val_dl = DataLoader(DatasetGenerator(val_windows, len(val_windows), is_test=False),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    torch.cuda.reset_peak_memory_stats()   # 每折開頭重設,量本折真實峰值顯存
    model = build_model(f"breeze_diar{CKPT_TAG}_fold{fold_idx+1:02d}", train_windows, val_windows)
    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"loo_breeze_diar{CKPT_TAG}_fold{fold_idx+1:02d}")
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs", name=f"loo_breeze_diar{CKPT_TAG}_fold{fold_idx+1:02d}"),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                            dirpath=ckpt_dir, filename="{epoch}-{val_loss:.3f}"),
        ],
        max_epochs=MAX_EPOCHS, devices=[0], accelerator="gpu",
        precision=PRECISION, enable_progress_bar=True, log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    _peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Best ckpt: {os.path.basename(best_ckpt)}  |  峰值顯存 = {_peak:.1f} GB "
          f"(grad_ckpt={GRAD_CKPT}, batch={BATCH_SIZE})")
    del model, trainer
    torch.cuda.empty_cache()

    # Evaluate
    best_data = torch.load(best_ckpt, map_location="cpu")
    whisper_model = WhisperWrapperFast(pretrained_model=EMBEDDING, lora_rank=LORA_RANK,
                                       head_type=HEAD_TYPE)
    model_state = {k[len("model."):]: v for k, v in best_data["state_dict"].items()
                   if k.startswith("model.") and "embed_positions" not in k}
    whisper_model.load_state_dict(model_state, strict=False)
    whisper_model = whisper_model.cuda().eval()

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

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nBreeze diarization VALIDATION results ({len(all_results)} folds)\n{'='*60}")
for r in all_results:
    print(f"{r['session']:<42} FA={r['FA']:5.1f} MD={r['MD']:5.1f} SC={r['SC']:5.1f} DER={r['DER']:5.1f}")
if all_results:
    avgs = {k: float(np.mean([r[k] for r in all_results])) for k in ["FA", "MD", "SC", "DER"]}
    print(f"{'mean':<42} FA={avgs['FA']:5.1f} MD={avgs['MD']:5.1f} SC={avgs['SC']:5.1f} DER={avgs['DER']:5.1f}")
    summary = {
        "sessions": all_results, "average": avgs,
        "method": f"Breeze-ASR-25 encoder + {HEAD_TYPE} head + ASD LOO ({'CLEAN-10' if CLEAN10 else 'VALIDATION'}, {len(all_results)} folds)",
        "excluded_sessions": EXCLUDE_NAMES,
        "backbone": EMBEDDING, "head": HEAD_TYPE, "n_folds": len(all_results),
        "note": "compare against loo_results_audioset_bilstm_16sess (whisper-base+BiLSTM) on the SAME sessions",
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {RESULTS_DIR}/summary.json")
print("\nDONE.")
