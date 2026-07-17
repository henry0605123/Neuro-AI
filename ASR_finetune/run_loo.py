"""Step 4 — LOO orchestrator.

For each of the 13 LOO folds:
  1. LoRA fine-tune Belle-whisper-large-v3-zh on the 12 training cases
  2. Infer on the held-out test case
  3. Compute WER (CER) and BERTScore-F1 — overall and split by adult/child
  4. Save fold metrics

Aggregates final metrics across folds and writes a summary JSON.
"""
from __future__ import annotations

import gc
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import torch
from tqdm import tqdm

from config import (
    BASE_MODEL, FOLDS_FILE, FEATURES_DIR, LORA_DIR, OUTPUTS_DIR,
    PREDICTIONS_DIR, METRICS_DIR, LANGUAGE, TASK, SAMPLE_RATE, ensure_dirs,
)
from evaluate import evaluate_pairs
from lora_train import train_fold


def _free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def infer_test_case(peft_model, processor, test_case: str) -> list[dict]:
    """Run inference on the test case's pre-computed features."""
    from transformers import WhisperForConditionalGeneration  # for type hints only

    blob = torch.load(FEATURES_DIR / f"{test_case}.pt", map_location="cpu", weights_only=False)
    feats_all = blob["features"]   # (N, 80, 3000) fp16
    texts = blob["texts"]
    speakers = blob["speakers"]
    paths = blob["audio_paths"]

    peft_model.eval()
    peft_model.config.use_cache = True

    bs = 8
    preds: list[dict] = []
    for i in tqdm(range(0, feats_all.shape[0], bs), desc=f"  infer test={test_case[-8:]}", leave=False):
        chunk = feats_all[i: i + bs].to("cuda", dtype=torch.float16, non_blocking=True)
        ids = peft_model.generate(
            input_features=chunk,
            language=LANGUAGE,
            task=TASK,
            num_beams=1,
            max_new_tokens=200,
        )
        out_texts = processor.batch_decode(ids, skip_special_tokens=True)
        for j, h in enumerate(out_texts):
            preds.append({
                "case": test_case,
                "audio_path": paths[i + j],
                "speaker": speakers[i + j],
                "speaker_label": "adult" if speakers[i + j] in {"SPEAKER_00"}
                                 else None,   # filled below from manifest
                "text": texts[i + j],
                "hyp": h.strip(),
            })
    peft_model.config.use_cache = False
    return preds


def _enrich_speaker_label(preds: list[dict], manifest_rows_by_path: dict[str, dict]) -> None:
    for p in preds:
        rec = manifest_rows_by_path.get(p["audio_path"])
        if rec is not None:
            p["speaker_label"] = rec["speaker_label"]
            p["speaker"] = rec["speaker"]


def _load_manifest_by_path() -> dict[str, dict]:
    out = {}
    with (Path(OUTPUTS_DIR).parent / "manifest.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["audio_path"]] = r
    return out


def m2d(m):
    return {"n": m.n, "CER_or_WER": m.cer, "BERTScore_F1": m.bertscore_f1}


def evaluate_fold(preds: list[dict]) -> dict:
    refs = [p["text"] for p in preds]
    hyps = [p["hyp"] for p in preds]
    overall = evaluate_pairs(refs, hyps)
    out = {"overall": m2d(overall)}
    for label in ("adult", "child"):
        idx = [i for i, p in enumerate(preds) if p["speaker_label"] == label]
        if idx:
            m = evaluate_pairs([refs[i] for i in idx], [hyps[i] for i in idx])
            out[label] = m2d(m)
    return out


def main():
    ensure_dirs()
    folds = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["folds"]
    print(f"Running LOO over {len(folds)} folds")

    manifest_by_path = _load_manifest_by_path()

    all_preds_path = PREDICTIONS_DIR / "loo_predictions.jsonl"
    fold_metrics_path = METRICS_DIR / "loo_fold_metrics.json"
    summary_path = METRICS_DIR / "loo_summary.json"

    fold_metrics = []
    all_preds: list[dict] = []
    t_start = time.time()

    # Resume support: skip folds whose adapter + predictions already exist.
    completed_preds_path = PREDICTIONS_DIR / "loo_predictions.jsonl"
    completed_folds: set[int] = set()
    if completed_preds_path.exists():
        for line in completed_preds_path.open(encoding="utf-8"):
            try:
                completed_folds.add(json.loads(line).get("fold"))
            except json.JSONDecodeError:
                pass
    if completed_folds:
        print(f"Resume: skipping already-finished folds {sorted(completed_folds)}", flush=True)

    for fold_def in folds:
        if fold_def["fold"] in completed_folds:
            print(f"Fold {fold_def['fold']} ({fold_def['test_case']}) already complete — skipping.", flush=True)
            continue
        artifacts = train_fold(fold_def)
        # After training, the LoRA-wrapped model is in memory inside train_fold.
        # But train_fold doesn't return it (to keep the API small). Reload here.
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        base = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16
        ).to("cuda")
        peft_model = PeftModel.from_pretrained(base, artifacts.adapter_dir).to("cuda")
        processor = WhisperProcessor.from_pretrained(BASE_MODEL)
        processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

        preds = infer_test_case(peft_model, processor, artifacts.test_case)
        _enrich_speaker_label(preds, manifest_by_path)
        # Tag fold and persist
        for p in preds:
            p["fold"] = artifacts.fold
        with all_preds_path.open("a", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        all_preds.extend(preds)

        m = evaluate_fold(preds)
        fold_metrics.append({
            "fold": artifacts.fold,
            "test_case": artifacts.test_case,
            "val_case": artifacts.val_case,
            "train_time_sec": artifacts.train_time_sec,
            "n_test_segments": len(preds),
            **m,
        })
        print(f"  fold {artifacts.fold} {artifacts.test_case}  "
              f"overall CER={m['overall']['CER_or_WER']:.4f}  "
              f"BERTScore-F1={m['overall']['BERTScore_F1']:.4f}")

        # cleanup before next fold
        del base, peft_model
        _free_gpu()

        # checkpoint fold metrics each fold
        fold_metrics_path.write_text(
            json.dumps(fold_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- aggregate ----
    # Pooled metrics: re-evaluate on all preds combined.
    overall_pooled = evaluate_fold(all_preds)
    macro_overall = {
        "CER_or_WER": float(np.mean([f["overall"]["CER_or_WER"] for f in fold_metrics])),
        "BERTScore_F1": float(np.mean([f["overall"]["BERTScore_F1"] for f in fold_metrics])),
    }
    macro_by_role = {}
    for label in ("adult", "child"):
        cers = [f[label]["CER_or_WER"] for f in fold_metrics if label in f]
        bs = [f[label]["BERTScore_F1"] for f in fold_metrics if label in f]
        if cers:
            macro_by_role[label] = {
                "CER_or_WER": float(np.mean(cers)),
                "BERTScore_F1": float(np.mean(bs)),
            }

    summary = {
        "model": BASE_MODEL,
        "method": "LoRA fine-tune (LOO)",
        "n_folds": len(folds),
        "total_time_sec": time.time() - t_start,
        "pooled": overall_pooled,
        "macro_avg_over_folds": {
            "overall": macro_overall,
            **macro_by_role,
        },
        "per_fold": fold_metrics,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSummary -> {summary_path}")
    print(f"Pooled overall   CER={overall_pooled['overall']['CER_or_WER']:.4f}  "
          f"BERTScore-F1={overall_pooled['overall']['BERTScore_F1']:.4f}")
    print(f"Macro-avg overall CER={macro_overall['CER_or_WER']:.4f}  "
          f"BERTScore-F1={macro_overall['BERTScore_F1']:.4f}")


if __name__ == "__main__":
    main()
