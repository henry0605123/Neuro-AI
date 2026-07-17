"""Step 2 (Breeze) — LoRA LOO on MediaTek-Research/Breeze-ASR-25.

Three phases in one script:
  Phase 1 — Precompute Breeze features (80-bin v2 mel) + traditional-Chinese
            labels to features_breeze/<case>.pt (no OpenCC normalization;
            Breeze outputs traditional natively)
  Phase 2 — Monkey-patch lora_train.{BASE_MODEL,FEATURES_DIR,LORA_DIR} +
            run 13-fold LoRA LOO
  Phase 3 — Inference + aggregate metrics
"""
from __future__ import annotations
import gc, json, os, sys, time, warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import librosa
import numpy as np
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import config as _config
import lora_train as _lt

BREEZE_MODEL = "MediaTek-Research/Breeze-ASR-25"
BREEZE_FEATURES_DIR = _config.PROJECT_ROOT / "features_breeze"
BREEZE_LORA_DIR = _config.LORA_DIR / "breeze"

# Patch lora_train module-level names
_lt.BASE_MODEL = BREEZE_MODEL
_lt.FEATURES_DIR = BREEZE_FEATURES_DIR
_lt.LORA_DIR = BREEZE_LORA_DIR
BREEZE_LORA_DIR.mkdir(parents=True, exist_ok=True)
BREEZE_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

from config import (
    FOLDS_FILE, MANIFEST_FILE, PREDICTIONS_DIR, METRICS_DIR,
    LANGUAGE, TASK, SAMPLE_RATE, ensure_dirs,
)
from evaluate import evaluate_pairs
from lora_train import train_fold

PRED_PATH = PREDICTIONS_DIR / "loo_breeze_predictions.jsonl"
FOLD_METRICS_PATH = METRICS_DIR / "loo_breeze_fold_metrics.json"
SUMMARY_PATH = METRICS_DIR / "loo_breeze_summary.json"
LOG_PATH = METRICS_DIR / "loo_breeze.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _free():
    gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Phase 1: precompute features
# ============================================================
def phase1_precompute() -> None:
    log(f"[Phase 1] Pre-compute Breeze features -> {BREEZE_FEATURES_DIR}")
    # Group manifest by case
    by_case: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            by_case[r["case"]].append((i, r))
    log(f"  cases: {len(by_case)}  total segs: {sum(len(v) for v in by_case.values())}")

    log(f"  loading processor {BREEZE_MODEL}")
    processor = WhisperProcessor.from_pretrained(BREEZE_MODEL)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    for ci, (case, items) in enumerate(by_case.items()):
        out_path = BREEZE_FEATURES_DIR / f"{case}.pt"
        if out_path.exists():
            log(f"  [{ci+1}/{len(by_case)}] {case} cached, skip")
            continue
        feats_list: list[np.ndarray] = []
        labels_list: list[list[int]] = []
        texts, speakers, paths, midx = [], [], [], []
        t0 = time.time()
        for i, r in items:
            wav, _ = librosa.load(r["audio_path"], sr=SAMPLE_RATE, mono=True)
            wav = wav.astype(np.float32)
            inp = processor.feature_extractor(
                wav, sampling_rate=SAMPLE_RATE, return_tensors="np"
            )
            mel = inp.input_features[0]   # (80, 3000) fp32
            feats_list.append(mel.astype(np.float16))
            # Breeze outputs traditional natively — keep labels traditional, no t2s
            tokens = processor.tokenizer(r["text"], add_special_tokens=True).input_ids
            labels_list.append(tokens)
            texts.append(r["text"])
            speakers.append(r["speaker"])
            paths.append(r["audio_path"])
            midx.append(i)
        features = torch.from_numpy(np.stack(feats_list, axis=0))
        torch.save({
            "features": features,
            "labels": labels_list,
            "texts": texts,
            "speakers": speakers,
            "audio_paths": paths,
            "manifest_idx": midx,
        }, out_path)
        size_mb = out_path.stat().st_size / 1024 / 1024
        log(f"  [{ci+1}/{len(by_case)}] {case}: N={features.shape[0]} {size_mb:.0f}MB ({time.time()-t0:.1f}s)")


# ============================================================
# Phase 2 + 3: LoRA LOO + metrics
# ============================================================
def _load_manifest_by_path() -> dict[str, dict]:
    out = {}
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["audio_path"]] = r
    return out


@torch.no_grad()
def infer_test_case(peft_model, processor, test_case: str) -> list[dict]:
    blob = torch.load(BREEZE_FEATURES_DIR / f"{test_case}.pt",
                      map_location="cpu", weights_only=False)
    feats_all = blob["features"]; texts = blob["texts"]
    speakers = blob["speakers"]; paths = blob["audio_paths"]
    peft_model.eval(); peft_model.config.use_cache = True
    bs = 8
    preds = []
    for i in tqdm(range(0, feats_all.shape[0], bs), desc=f"  infer {test_case[-8:]}", leave=False):
        chunk = feats_all[i: i + bs].to("cuda", dtype=torch.float16, non_blocking=True)
        ids = peft_model.generate(
            input_features=chunk,
            language=LANGUAGE, task=TASK,
            num_beams=1, max_new_tokens=200,
        )
        out_texts = processor.batch_decode(ids, skip_special_tokens=True)
        for j, h in enumerate(out_texts):
            preds.append({
                "case": test_case, "audio_path": paths[i + j],
                "speaker": speakers[i + j], "text": texts[i + j],
                "hyp": h.strip(),
            })
    return preds


def m2d(m): return {"n": m.n, "CER_or_WER": m.cer, "BERTScore_F1": m.bertscore_f1}


def evaluate_fold(preds: list[dict]) -> dict:
    refs = [p["text"] for p in preds]; hyps = [p["hyp"] for p in preds]
    overall = evaluate_pairs(refs, hyps)
    out = {"overall": m2d(overall)}
    for label in ("adult", "child"):
        idx = [i for i, p in enumerate(preds) if p["speaker_label"] == label]
        if idx:
            out[label] = m2d(evaluate_pairs([refs[i] for i in idx], [hyps[i] for i in idx]))
    return out


def phase2_3_loo():
    ensure_dirs()
    folds = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["folds"]
    log(f"\n[Phase 2-3] Breeze LoRA LOO ({len(folds)} folds)")
    log(f"  base       : {BREEZE_MODEL}")
    log(f"  features   : {BREEZE_FEATURES_DIR}")
    log(f"  adapter dir: {BREEZE_LORA_DIR}")
    log(f"  predictions: {PRED_PATH}")

    manifest_by_path = _load_manifest_by_path()
    fold_metrics: list[dict] = []
    all_preds: list[dict] = []

    # Resume
    completed: set[int] = set()
    if PRED_PATH.exists():
        with PRED_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line); all_preds.append(rec)
                    if rec.get("fold") is not None:
                        completed.add(rec["fold"])
                except json.JSONDecodeError:
                    pass
    if FOLD_METRICS_PATH.exists():
        try:
            fold_metrics = json.loads(FOLD_METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fold_metrics = []
    if completed:
        log(f"Resume: skip folds {sorted(completed)}")

    t_all = time.time()
    for fold_def in folds:
        if fold_def["fold"] in completed:
            log(f"Fold {fold_def['fold']} done, skip")
            continue
        artifacts = train_fold(fold_def)
        base = WhisperForConditionalGeneration.from_pretrained(BREEZE_MODEL, torch_dtype=torch.float16).to("cuda")
        peft_model = PeftModel.from_pretrained(base, artifacts.adapter_dir).to("cuda")
        processor = WhisperProcessor.from_pretrained(BREEZE_MODEL)
        processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
        preds = infer_test_case(peft_model, processor, artifacts.test_case)
        for p in preds:
            rec = manifest_by_path.get(p["audio_path"])
            if rec:
                p["speaker_label"] = rec["speaker_label"]
                p["speaker"] = rec["speaker"]
            p["fold"] = artifacts.fold
        with PRED_PATH.open("a", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        all_preds.extend(preds)

        m = evaluate_fold(preds)
        fold_metrics.append({
            "fold": artifacts.fold, "test_case": artifacts.test_case,
            "val_case": artifacts.val_case,
            "train_time_sec": artifacts.train_time_sec,
            "n_test_segments": len(preds), **m,
        })
        FOLD_METRICS_PATH.write_text(json.dumps(fold_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  fold {artifacts.fold} {artifacts.test_case}  "
            f"CER={m['overall']['CER_or_WER']:.4f}  BERT-F1={m['overall']['BERTScore_F1']:.4f}")
        del base, peft_model; _free()

    # Aggregate
    overall_pooled = evaluate_fold(all_preds)
    macro_overall = {
        "CER_or_WER": float(np.mean([f["overall"]["CER_or_WER"] for f in fold_metrics])),
        "BERTScore_F1": float(np.mean([f["overall"]["BERTScore_F1"] for f in fold_metrics])),
    }
    macro_by_role: dict[str, dict] = {}
    for label in ("adult", "child"):
        cers = [f[label]["CER_or_WER"] for f in fold_metrics if label in f]
        bs = [f[label]["BERTScore_F1"] for f in fold_metrics if label in f]
        if cers:
            macro_by_role[label] = {
                "CER_or_WER": float(np.mean(cers)),
                "BERTScore_F1": float(np.mean(bs)),
            }
    summary = {
        "model": BREEZE_MODEL,
        "method": "Breeze-ASR-25 + LoRA LOO (single stage)",
        "n_folds": len(folds),
        "total_time_sec": time.time() - t_all,
        "pooled": overall_pooled,
        "macro_avg_over_folds": {"overall": macro_overall, **macro_by_role},
        "per_fold": fold_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n=== Breeze LoRA LOO DONE ===")
    log(f"Pooled  CER={overall_pooled['overall']['CER_or_WER']:.4f}  BERT-F1={overall_pooled['overall']['BERTScore_F1']:.4f}")
    log(f"Macro   CER={macro_overall['CER_or_WER']:.4f}  BERT-F1={macro_overall['BERTScore_F1']:.4f}")
    for k, v in macro_by_role.items():
        log(f"  {k}: CER={v['CER_or_WER']:.4f} BERT-F1={v['BERTScore_F1']:.4f}")
    log(f"Summary -> {SUMMARY_PATH}")


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    t_all = time.time()
    log(f"=== Breeze LoRA LOO pipeline ===")
    phase1_precompute()
    _free()
    phase2_3_loo()
    log(f"\nTotal pipeline time: {(time.time()-t_all)/60:.1f} min")
    log("DONE")


if __name__ == "__main__":
    main()
