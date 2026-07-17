"""
Inference + DER evaluation on a test session.
Usage:
  python scripts/evaluate.py --checkpoint <ckpt_path> --wav <wav_path> --label <csv_path>
"""
import sys, os, argparse, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch, torchaudio
import numpy as np
from scripts.convert_output import get_timestamps, majority_filter
from lightning_modules.classifier import Classifier

WINDOW_SIZE = 10
SAMPLE_RATE = 16000
COLLAR = 0.1  # 100ms forgiveness collar (same as paper)


# ── Inference ────────────────────────────────────────────────────────────────

def process_wav(wav_path, model):
    x, sr = torchaudio.load(wav_path)
    if x.size(0) > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
    x = x.float()
    length = x.size(1) / SAMPLE_RATE

    child_pred, adult_pred = [], []
    start = 0.0
    while start + WINDOW_SIZE <= length:
        window = x[:, int(start * SAMPLE_RATE):int((start + WINDOW_SIZE) * SAMPLE_RATE)]
        pred = model.forward_eval(window)
        pred = majority_filter(pred)
        child, adult, _ = get_timestamps(pred)
        child_pred += [(start + s, start + e) for s, e in child]
        adult_pred += [(start + s, start + e) for s, e in adult]
        start += WINDOW_SIZE

    return merge_intervals(child_pred), merge_intervals(adult_pred), length


def merge_intervals(intervals):
    merged = []
    for s, e in sorted(intervals):
        if merged and s - merged[-1][1] <= 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# ── Ground Truth ─────────────────────────────────────────────────────────────

def load_ground_truth(csv_path):
    child_gt, adult_gt = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            label, start, end = row[0], float(row[1]), float(row[2])
            if "c" in label.lower():
                child_gt.append((start, end))
            elif "a" in label.lower():
                adult_gt.append((start, end))
    return child_gt, adult_gt


# ── DER Calculation ───────────────────────────────────────────────────────────

def intervals_to_frames(intervals, duration, step=0.02):
    n = int(duration / step) + 1
    frames = np.zeros(n, dtype=int)
    for s, e in intervals:
        fs, fe = int(s / step), int(e / step)
        frames[fs:fe] = 1
    return frames


def compute_der(child_pred, adult_pred, child_gt, adult_gt, duration, collar=0.1):
    """Compute frame-level DER components."""
    step = 0.02
    n = int(duration / step) + 1

    # Build frame arrays: 1=child, 2=adult, 0=silence
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

    # Apply collar: ignore frames within collar of reference boundaries
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
        print("WARNING: no reference speech found")
        return None

    # False Alarm: predicted speech where reference is silence
    fa = ((pred[eval_mask] > 0) & (gt[eval_mask] == 0)).sum()
    # Missed Detection: reference speech not detected
    md = ((pred[eval_mask] == 0) & (gt[eval_mask] > 0)).sum()
    # Speaker Confusion: both speech but wrong speaker
    sc = ((pred[eval_mask] > 0) & (gt[eval_mask] > 0) & (pred[eval_mask] != gt[eval_mask])).sum()

    fa_r  = fa  / ref_speech * 100
    md_r  = md  / ref_speech * 100
    sc_r  = sc  / ref_speech * 100
    der   = fa_r + md_r + sc_r

    return {"FA": fa_r, "MD": md_r, "SC": sc_r, "DER": der}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav",        required=True)
    parser.add_argument("--label",      required=True)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    clf = Classifier.load_from_checkpoint(args.checkpoint, strict=False)
    model = clf.model.cuda()
    model.eval()

    print(f"Running inference on: {args.wav}")
    with torch.no_grad():
        child_pred, adult_pred, duration = process_wav(args.wav, model)

    print(f"\n── Predictions ──")
    print(f"Child segments : {len(child_pred)}")
    print(f"Adult segments : {len(adult_pred)}")

    print(f"\nLoading ground truth: {args.label}")
    child_gt, adult_gt = load_ground_truth(args.label)
    print(f"Child GT       : {len(child_gt)}")
    print(f"Adult GT       : {len(adult_gt)}")

    print(f"\n── DER Evaluation (collar={COLLAR}s) ──")
    result = compute_der(child_pred, adult_pred, child_gt, adult_gt, duration, COLLAR)
    if result:
        print(f"  FA  (False Alarm)        : {result['FA']:.1f}%")
        print(f"  MD  (Missed Detection)   : {result['MD']:.1f}%")
        print(f"  SC  (Speaker Confusion)  : {result['SC']:.1f}%")
        print(f"  DER                      : {result['DER']:.1f}%")
