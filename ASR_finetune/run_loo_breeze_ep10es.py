"""Breeze LoRA LOO — verification run: 10 epochs + early stopping.

Same pipeline as run_loo_breeze.py, but each fold trains up to 10 epochs with
early stopping on the (separate) val_case loss (patience=2), keeping the
best-val checkpoint. Purpose: check whether the original 1-epoch setting was
actually sufficient, or whether more epochs lower CER.

Reuses the cached Breeze features (features_breeze/) — no recompute.
Writes to *_ep10es output paths + a separate adapter dir, so the original
1-epoch results (loo_breeze_*) are left untouched for comparison.
"""
from __future__ import annotations
import gc, json, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import config as _config
import lora_train as _lt
# Import run_loo_breeze FIRST — its module-level code patches _lt.{BASE_MODEL,
# FEATURES_DIR,LORA_DIR} to the original 'breeze' dir. We re-patch below so our
# adapters land in a separate dir and never overwrite the 1-epoch adapters.
import run_loo_breeze as base_run

BREEZE_MODEL = "MediaTek-Research/Breeze-ASR-25"
BREEZE_FEATURES_DIR = _config.PROJECT_ROOT / "features_breeze"
BREEZE_LORA_DIR = _config.LORA_DIR / "breeze_ep10es"

# Re-patch lora_train AFTER importing run_loo_breeze (separate adapter dir)
_lt.BASE_MODEL = BREEZE_MODEL
_lt.FEATURES_DIR = BREEZE_FEATURES_DIR
_lt.LORA_DIR = BREEZE_LORA_DIR
BREEZE_LORA_DIR.mkdir(parents=True, exist_ok=True)

from config import (
    FOLDS_FILE, MANIFEST_FILE, PREDICTIONS_DIR, METRICS_DIR,
    LANGUAGE, TASK, ensure_dirs,
)

NUM_EPOCHS = 10
PATIENCE = 2

PRED_PATH = PREDICTIONS_DIR / "loo_breeze_ep10es_predictions.jsonl"
FOLD_METRICS_PATH = METRICS_DIR / "loo_breeze_ep10es_fold_metrics.json"
SUMMARY_PATH = METRICS_DIR / "loo_breeze_ep10es_summary.json"
LOG_PATH = METRICS_DIR / "loo_breeze_ep10es.log"

# Point the shared helpers at our feature dir + output paths
base_run.BREEZE_FEATURES_DIR = BREEZE_FEATURES_DIR
base_run.BREEZE_LORA_DIR = BREEZE_LORA_DIR
base_run.PRED_PATH = PRED_PATH
base_run.FOLD_METRICS_PATH = FOLD_METRICS_PATH
base_run.SUMMARY_PATH = SUMMARY_PATH
base_run.LOG_PATH = LOG_PATH


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _free():
    gc.collect(); torch.cuda.empty_cache()


def phase2_3_loo():
    ensure_dirs()
    folds = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["folds"]
    log(f"\n[Phase 2-3] Breeze LoRA LOO — {NUM_EPOCHS} epochs + early stopping (patience={PATIENCE})")
    log(f"  base       : {BREEZE_MODEL}")
    log(f"  features   : {BREEZE_FEATURES_DIR}")
    log(f"  adapter dir: {BREEZE_LORA_DIR}")
    log(f"  predictions: {PRED_PATH}")

    manifest_by_path = base_run._load_manifest_by_path()
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
        artifacts = _lt.train_fold(fold_def, num_epochs=NUM_EPOCHS, patience=PATIENCE)
        base = WhisperForConditionalGeneration.from_pretrained(BREEZE_MODEL, torch_dtype=torch.float16).to("cuda")
        peft_model = PeftModel.from_pretrained(base, artifacts.adapter_dir).to("cuda")
        processor = WhisperProcessor.from_pretrained(BREEZE_MODEL)
        processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
        preds = base_run.infer_test_case(peft_model, processor, artifacts.test_case)
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

        m = base_run.evaluate_fold(preds)
        # best epoch chosen by early stopping (from saved fold_info.json)
        try:
            info = json.loads((artifacts.adapter_dir / "fold_info.json").read_text(encoding="utf-8"))
            best_epoch = info.get("best_epoch"); early_stopped = info.get("early_stopped")
        except Exception:
            best_epoch, early_stopped = None, None
        fold_metrics.append({
            "fold": artifacts.fold, "test_case": artifacts.test_case,
            "val_case": artifacts.val_case,
            "train_time_sec": artifacts.train_time_sec,
            "best_epoch": best_epoch, "early_stopped": early_stopped,
            "val_loss_curve": artifacts.val_loss_curve,
            "n_test_segments": len(preds), **m,
        })
        FOLD_METRICS_PATH.write_text(json.dumps(fold_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  fold {artifacts.fold} {artifacts.test_case}  best_ep={best_epoch}  "
            f"CER={m['overall']['CER_or_WER']:.4f}  BERT-F1={m['overall']['BERTScore_F1']:.4f}")
        del base, peft_model; _free()

    overall_pooled = base_run.evaluate_fold(all_preds)
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
        "method": f"Breeze-ASR-25 + LoRA LOO (single stage, {NUM_EPOCHS} epochs + early stopping patience={PATIENCE})",
        "n_folds": len(folds),
        "total_time_sec": time.time() - t_all,
        "pooled": overall_pooled,
        "macro_avg_over_folds": {"overall": macro_overall, **macro_by_role},
        "per_fold": fold_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n=== Breeze LoRA LOO (ep10+ES) DONE ===")
    log(f"Pooled  CER={overall_pooled['overall']['CER_or_WER']:.4f}  BERT-F1={overall_pooled['overall']['BERTScore_F1']:.4f}")
    log(f"Macro   CER={macro_overall['CER_or_WER']:.4f}  BERT-F1={macro_overall['BERTScore_F1']:.4f}")
    log(f"Summary -> {SUMMARY_PATH}")


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    t_all = time.time()
    log(f"=== Breeze LoRA LOO (10 epochs + early stopping) ===")
    base_run.phase1_precompute()   # reuses cached features, skips existing
    _free()
    phase2_3_loo()
    log(f"\nTotal pipeline time: {(time.time()-t_all)/60:.1f} min")
    log("DONE")


if __name__ == "__main__":
    main()
