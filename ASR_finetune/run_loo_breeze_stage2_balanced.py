"""Stage 2 — LoRA LOO on Breeze + ChangPN(BALANCED) merged base, user 13 cases.

Same as run_loo_breeze_stage2.py but points to the balanced Stage 1 merged base.
"""
from __future__ import annotations
import gc, json, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import config as _config
import lora_train as _lt

STAGE1_BASE = str(_config.PROJECT_ROOT / "models" / "breeze_changpn_stage1_balanced")
BREEZE_FEATURES_DIR = _config.PROJECT_ROOT / "features_breeze"
BREEZE_STAGE2_LORA_DIR = _config.LORA_DIR / "breeze_stage2_balanced"

_lt.BASE_MODEL = STAGE1_BASE
_lt.FEATURES_DIR = BREEZE_FEATURES_DIR
_lt.LORA_DIR = BREEZE_STAGE2_LORA_DIR
BREEZE_STAGE2_LORA_DIR.mkdir(parents=True, exist_ok=True)

from config import (
    FOLDS_FILE, MANIFEST_FILE, PREDICTIONS_DIR, METRICS_DIR,
    LANGUAGE, TASK, ensure_dirs,
)
from evaluate import evaluate_pairs
from lora_train import train_fold

PRED_PATH = PREDICTIONS_DIR / "loo_breeze_stage2_balanced_predictions.jsonl"
FOLD_METRICS_PATH = METRICS_DIR / "loo_breeze_stage2_balanced_fold_metrics.json"
SUMMARY_PATH = METRICS_DIR / "loo_breeze_stage2_balanced_summary.json"
LOG_PATH = METRICS_DIR / "loo_breeze_stage2_balanced.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _free():
    gc.collect(); torch.cuda.empty_cache()


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


def main():
    ensure_dirs()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    folds = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["folds"]
    log(f"=== Breeze Stage 2 BALANCED LoRA LOO ({len(folds)} folds) ===")
    log(f"  base       : {STAGE1_BASE}")
    log(f"  features   : {BREEZE_FEATURES_DIR}")
    log(f"  adapter dir: {BREEZE_STAGE2_LORA_DIR}")
    log(f"  predictions: {PRED_PATH}")

    manifest_by_path = _load_manifest_by_path()
    fold_metrics: list[dict] = []
    all_preds: list[dict] = []
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
        base = WhisperForConditionalGeneration.from_pretrained(STAGE1_BASE, torch_dtype=torch.float16).to("cuda")
        peft_model = PeftModel.from_pretrained(base, artifacts.adapter_dir).to("cuda")
        processor = WhisperProcessor.from_pretrained(STAGE1_BASE)
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
        "model": STAGE1_BASE,
        "method": "Breeze + ChangPN(BALANCED) Stage 1 + user Stage 2 LoRA LOO",
        "n_folds": len(folds),
        "total_time_sec": time.time() - t_all,
        "pooled": overall_pooled,
        "macro_avg_over_folds": {"overall": macro_overall, **macro_by_role},
        "per_fold": fold_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n=== Breeze Stage 2 BALANCED DONE ===")
    log(f"Pooled  CER={overall_pooled['overall']['CER_or_WER']:.4f}  BERT-F1={overall_pooled['overall']['BERTScore_F1']:.4f}")
    log(f"Macro   CER={macro_overall['CER_or_WER']:.4f}  BERT-F1={macro_overall['BERTScore_F1']:.4f}")
    for k, v in macro_by_role.items():
        log(f"  {k}: CER={v['CER_or_WER']:.4f} BERT-F1={v['BERTScore_F1']:.4f}")


if __name__ == "__main__":
    main()
