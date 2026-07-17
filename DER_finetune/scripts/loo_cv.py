"""
Leave-One-Out Cross-Validation for child-adult diarization.
For each fold: 1 session = test, remaining 12 = train (val = test session for early stopping).
Outputs DER per session and overall average.

Usage:
  python scripts/loo_cv.py
"""
import sys, os, csv, json, re, glob, random, subprocess, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch, torchaudio
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_DIR    = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/Fine_tune"
WORK_DIR     = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/loo_work"        # temp data for each fold
RESULTS_DIR  = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/data/loo_results"
PRETRAINED   = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling/checkpoints/whisper_base_5k/epoch=17-val_loss=0.113.ckpt"
SAVE_DIR     = f"{os.environ.get('ASD_DATA_ROOT', 'data')}/whisper-modeling"
PYTHON       = "D:/Anaconda/envs/paper/python.exe"
MAIN_SCRIPT  = "D:/PycharmProjects/paper/child-adult-diarization/whisper-modeling/scripts/main.py"
SAMPLE_RATE  = 16000
COLLAR       = 0.1

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data parsing (same as prepare_finetune.py) ───────────────────────────────
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
        start, end = srt_time_to_sec(times[0]), srt_time_to_sec(times[1])
        text = " ".join(lines[2:])
        if "小孩" in text:
            segments.append(("c", start, end))
        elif "測試者" in text or "評估者" in text:
            segments.append(("a", start, end))
    return segments

def parse_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    mapping = data.get("speaker_mapping") or data.get("metadata", {}).get("speaker_mapping", {})
    role_map = {spk: ("c" if "兒童" in role or "小孩" in role else "a")
                for spk, role in mapping.items()}
    segments = []
    for seg in data["segments"]:
        label = role_map.get(seg.get("speaker", ""), "a")
        s, e = float(seg["start"]), float(seg["end"])
        if e > s:
            segments.append((label, s, e))
    return segments

def convert_audio(mp3_path, wav_path):
    audio, sr = torchaudio.load(mp3_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    torchaudio.save(wav_path, audio, SAMPLE_RATE)

def save_csv(segments, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for label, s, e in segments:
            writer.writerow([label, round(s, 3), round(e, 3)])

# ── DER ──────────────────────────────────────────────────────────────────────
def compute_der(child_pred, adult_pred, child_gt, adult_gt, duration, collar=COLLAR):
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

    collar_frames = int(collar / step)
    collar_mask = np.zeros(n, dtype=bool)
    for arr in [child_gt, adult_gt]:
        for s, e in arr:
            fs, fe = int(s/step), int(e/step)
            collar_mask[max(0, fs-collar_frames):min(n, fs+collar_frames)] = True
            collar_mask[max(0, fe-collar_frames):min(n, fe+collar_frames)] = True

    eval_mask = ~collar_mask
    ref_speech = (gt[eval_mask] > 0).sum()
    if ref_speech == 0:
        return None

    fa = ((pred[eval_mask] > 0) & (gt[eval_mask] == 0)).sum()
    md = ((pred[eval_mask] == 0) & (gt[eval_mask] > 0)).sum()
    sc = ((pred[eval_mask] > 0) & (gt[eval_mask] > 0) & (pred[eval_mask] != gt[eval_mask])).sum()

    return {
        "FA":  fa / ref_speech * 100,
        "MD":  md / ref_speech * 100,
        "SC":  sc / ref_speech * 100,
        "DER": (fa + md + sc) / ref_speech * 100,
    }

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(wav_path, model):
    from scripts.convert_output import get_timestamps, majority_filter

    x, sr = torchaudio.load(wav_path)
    if x.size(0) > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
    x = x.float()
    length = x.size(1) / SAMPLE_RATE

    child_pred, adult_pred = [], []
    start = 0.0
    WINDOW = 10
    while start + WINDOW <= length:
        window = x[:, int(start*SAMPLE_RATE):int((start+WINDOW)*SAMPLE_RATE)]
        pred = model.forward_eval(window)
        pred = majority_filter(pred)
        child, adult, _ = get_timestamps(pred)
        child_pred += [(start+s, start+e) for s, e in child]
        adult_pred += [(start+s, start+e) for s, e in adult]
        start += WINDOW

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
    mp3s = glob.glob(os.path.join(full, "*.mp3"))
    if not mp3s:
        continue
    jsons = glob.glob(os.path.join(full, "*_edited.json"))
    srts  = glob.glob(os.path.join(full, "*.srt"))
    if jsons:
        sessions.append((d, mp3s[0], jsons[0], "json"))
    elif srts:
        sessions.append((d, mp3s[0], srts[0], "srt"))

print(f"Total sessions: {len(sessions)}")
all_results = []

# ── LOO loop ──────────────────────────────────────────────────────────────────
for fold_idx, test_session in enumerate(sessions):
    test_name, test_mp3, test_ann, test_type = test_session
    train_sessions = [s for s in sessions if s[0] != test_name]

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/{len(sessions)}: test = {test_name}")
    print(f"{'='*60}")

    # 1. Prepare data
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

    for split, split_sessions in [("train", train_sessions), ("val", [test_session])]:
        for name, mp3, ann, ann_type in split_sessions:
            wav_out = os.path.join(WORK_DIR, "audios", split, name + ".wav")
            csv_out = os.path.join(WORK_DIR, "labels", split, name + ".csv")
            segs = parse_json(ann) if ann_type == "json" else parse_srt(ann)
            convert_audio(mp3, wav_out)
            save_csv(segs, csv_out)

    # 2. Write fold config
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints", f"loo_fold{fold_idx+1:02d}")
    config = f"""name: loo_fold{fold_idx+1:02d}
save: {SAVE_DIR}
pretrained_checkpoint: {PRETRAINED}
audio: {WORK_DIR}/audios
annotation: {WORK_DIR}/labels
seed: 42
batch_size: 16
window_size: 10
accumulate_grad_batches: 1
learning_rate: 0.0001
weight_decay: 0.0001
lora_rank: 8
max_epochs: 30
early_stopping_patience: 10
devices: [0]
optimizer: Adam
embedding: whisper_base
num_enc_layers_dict:
    whisper_small: 12
    whisper_base: 6
    whisper_tiny: 4
hid_dim_dict:
    whisper_small: 768
    whisper_base: 512
    whisper_tiny: 384
"""
    config_path = os.path.join(WORK_DIR, "config.yaml")
    with open(config_path, "w") as f:
        f.write(config)

    # 3. Train
    print(f"Training fold {fold_idx+1}...")
    ret = subprocess.run(
        [PYTHON, MAIN_SCRIPT, "--config", config_path, "--debug", "f"],
        cwd="D:/PycharmProjects/paper/child-adult-diarization/whisper-modeling",
        capture_output=True, text=True
    )
    # find best checkpoint
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_files:
        print(f"  ERROR: no checkpoint found for fold {fold_idx+1}")
        print(ret.stderr[-1000:])
        continue
    best_ckpt = sorted(ckpt_files)[-1]
    print(f"  Best ckpt: {os.path.basename(best_ckpt)}")

    # 4. Evaluate
    print(f"Evaluating on {test_name}...")
    from lightning_modules.classifier import Classifier
    clf = Classifier.load_from_checkpoint(best_ckpt, strict=False)
    model = clf.model.cuda()
    model.eval()

    test_wav = os.path.join(WORK_DIR, "audios", "val", test_name + ".wav")
    test_csv = os.path.join(WORK_DIR, "labels", "val", test_name + ".csv")

    with torch.no_grad():
        child_pred, adult_pred, duration = run_inference(test_wav, model)

    child_gt, adult_gt = [], []
    with open(test_csv, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            label, s, e = row[0], float(row[1]), float(row[2])
            (child_gt if "c" in label else adult_gt).append((s, e))

    result = compute_der(child_pred, adult_pred, child_gt, adult_gt, duration)
    if result:
        result["session"] = test_name
        all_results.append(result)
        print(f"  FA={result['FA']:.1f}%  MD={result['MD']:.1f}%  SC={result['SC']:.1f}%  DER={result['DER']:.1f}%")

    # 5. Save fold result
    fold_result_path = os.path.join(RESULTS_DIR, f"fold{fold_idx+1:02d}_{test_name}.json")
    with open(fold_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    del clf, model
    torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("LOO Cross-Validation Results")
print(f"{'='*60}")
print(f"{'Session':<40} {'FA':>6} {'MD':>6} {'SC':>6} {'DER':>6}")
print("-" * 60)
for r in all_results:
    print(f"{r['session']:<40} {r['FA']:>5.1f}% {r['MD']:>5.1f}% {r['SC']:>5.1f}% {r['DER']:>5.1f}%")

if all_results:
    print("-" * 60)
    avg_fa  = np.mean([r["FA"]  for r in all_results])
    avg_md  = np.mean([r["MD"]  for r in all_results])
    avg_sc  = np.mean([r["SC"]  for r in all_results])
    avg_der = np.mean([r["DER"] for r in all_results])
    print(f"{'Average':<40} {avg_fa:>5.1f}% {avg_md:>5.1f}% {avg_sc:>5.1f}% {avg_der:>5.1f}%")

    summary = {"sessions": all_results, "average": {"FA": avg_fa, "MD": avg_md, "SC": avg_sc, "DER": avg_der}}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")
