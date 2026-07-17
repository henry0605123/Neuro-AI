"""Step 2 (Breeze) — Zero-shot baseline with MediaTek-Research/Breeze-ASR-25.

Loads audio directly from segment WAVs (existing features cache is 128-bin
v3-style and incompatible with Breeze's v2 80-bin mel front-end).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import (
    LANGUAGE, TASK, MANIFEST_FILE,
    PREDICTIONS_DIR, METRICS_DIR, SAMPLE_RATE, ensure_dirs,
)
from evaluate import evaluate_pairs

MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
BATCH = 8


def log(msg: str) -> None:
    print(msg, flush=True)


def load_manifest_grouped():
    by_case: dict[str, list[dict]] = {}
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_case.setdefault(r["case"], []).append(r)
    return by_case


@torch.no_grad()
def infer_case(processor, model, case: str, recs: list[dict]) -> list[dict]:
    preds: list[dict] = []
    t0 = time.time()
    n = len(recs)
    for bi in range(0, n, BATCH):
        batch_recs = recs[bi: bi + BATCH]
        wavs = []
        for r in batch_recs:
            wav, _ = librosa.load(r["audio_path"], sr=SAMPLE_RATE, mono=True)
            wavs.append(wav)
        inputs = processor(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats = inputs.input_features.to("cuda", dtype=torch.float16)
        ids = model.generate(
            input_features=feats,
            language=LANGUAGE, task=TASK,
            num_beams=1, max_new_tokens=200,
        )
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        for r, h in zip(batch_recs, hyps):
            preds.append({
                "case": case,
                "audio_path": r["audio_path"],
                "speaker": r["speaker"],
                "speaker_label": r["speaker_label"],
                "text": r["text"],
                "duration": r.get("duration"),
                "hyp": h.strip(),
            })
    dt = time.time() - t0
    log(f"  {case}: n={n}  {dt:.1f}s  ({n/max(dt,1e-6):.1f} seg/s)")
    return preds


def main():
    ensure_dirs()
    pred_path = PREDICTIONS_DIR / "breeze_baseline_predictions.jsonl"
    metrics_path = METRICS_DIR / "breeze_baseline_metrics.json"
    log(f"=== Breeze-ASR-25 zero-shot on user 13 cases ===")
    log(f"  model      : {MODEL_ID}")
    log(f"  predictions: {pred_path}")

    by_case = load_manifest_grouped()
    cases = sorted(by_case.keys())
    log(f"  cases: {len(cases)}, total segments: {sum(len(v) for v in by_case.values())}")

    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    log(f"  loaded in {time.time()-t0:.1f}s")

    f_out = pred_path.open("w", encoding="utf-8")
    all_preds: list[dict] = []
    t_start = time.time()
    for ci, case in enumerate(cases):
        log(f"[{ci+1}/{len(cases)}] {case}")
        preds = infer_case(processor, model, case, by_case[case])
        for p in preds:
            f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
        f_out.flush()
        all_preds.extend(preds)
    f_out.close()
    log(f"\nTotal inference: {time.time()-t_start:.1f}s for {len(all_preds)} segments")

    # ---- metrics ----
    refs = [p["text"] for p in all_preds]
    hyps = [p["hyp"] for p in all_preds]
    overall = evaluate_pairs(refs, hyps)

    by_speaker = {}
    for label in ("adult", "child"):
        idx = [i for i, p in enumerate(all_preds) if p["speaker_label"] == label]
        if idx:
            by_speaker[label] = evaluate_pairs(
                [refs[i] for i in idx], [hyps[i] for i in idx]
            )

    by_case_metrics: dict[str, dict] = {}
    for c in sorted({p["case"] for p in all_preds}):
        idx = [i for i, p in enumerate(all_preds) if p["case"] == c]
        d = {"overall": evaluate_pairs(
            [refs[i] for i in idx], [hyps[i] for i in idx]
        )}
        for label in ("adult", "child"):
            cidx = [i for i in idx if all_preds[i]["speaker_label"] == label]
            if cidx:
                d[label] = evaluate_pairs(
                    [refs[i] for i in cidx], [hyps[i] for i in cidx]
                )
        by_case_metrics[c] = d

    def m2d(m):
        return {"n": m.n, "CER_or_WER": m.cer, "BERTScore_F1": m.bertscore_f1}

    summary = {
        "model": MODEL_ID,
        "setting": "zero-shot",
        "n_segments": len(all_preds),
        "overall": m2d(overall),
        "by_speaker": {k: m2d(v) for k, v in by_speaker.items()},
        "by_case": {c: {k: m2d(v) for k, v in d.items()}
                    for c, d in by_case_metrics.items()},
    }
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\nMetrics -> {metrics_path}")
    log(f"\nOverall  CER={overall.cer:.4f}  BERTScore-F1={overall.bertscore_f1:.4f}  (n={overall.n})")
    for k, v in by_speaker.items():
        log(f"  {k:6s} CER={v.cer:.4f}  BERTScore-F1={v.bertscore_f1:.4f}  (n={v.n})")
    log("DONE")


if __name__ == "__main__":
    main()
