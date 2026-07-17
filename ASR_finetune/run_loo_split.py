"""方案B — Run LOO with SEPARATE adult/child LoRA adapters.

For each of the 13 folds:
  1. Train two LoRA adapters from Belle-whisper-large-v3-zh:
       * adult adapter — only segments with speaker_label == "adult"
       * child adapter — only segments with speaker_label == "child"
  2. On the held-out test case, route each segment to its adapter via the
     **ground-truth** speaker_label ("oracle routing" — the upper bound for
     this architecture; in production the diarization model would supply
     this label).
  3. Compute CER + BERTScore-F1 overall and per role.

Resume-safe: skips folds whose predictions already exist in the JSONL.
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

import numpy as np
import torch
from tqdm import tqdm

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import (
    BASE_MODEL, FOLDS_FILE, FEATURES_DIR, OUTPUTS_DIR,
    PREDICTIONS_DIR, METRICS_DIR, MANIFEST_FILE, LANGUAGE, TASK,
    NUM_EPOCHS, ensure_dirs,
)
from evaluate import evaluate_pairs
from lora_train import train_fold
from run_loo import evaluate_fold, m2d  # reuse


def _free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_path_to_role() -> dict[str, str]:
    out: dict[str, str] = {}
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["audio_path"]] = r["speaker_label"]
    return out


@torch.no_grad()
def infer_split(test_case: str, adapter_dirs: dict[str, Path],
                path_to_role: dict[str, str]) -> list[dict]:
    """Run inference on `test_case`, routing each segment to its role's
    adapter via ground-truth speaker_label."""
    blob = torch.load(FEATURES_DIR / f"{test_case}.pt",
                      map_location="cpu", weights_only=False)
    feats_all = blob["features"]
    texts = blob["texts"]
    speakers = blob["speakers"]
    paths = blob["audio_paths"]
    n = feats_all.shape[0]

    # Group indices by role
    role_indices: dict[str, list[int]] = {"adult": [], "child": []}
    for i, p in enumerate(paths):
        r = path_to_role.get(p)
        if r in role_indices:
            role_indices[r].append(i)
    unmatched = n - sum(len(v) for v in role_indices.values())
    if unmatched:
        print(f"  [warn] {unmatched} test segments have no role mapping", flush=True)

    preds: list[dict | None] = [None] * n
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    bs = 8

    for role in ("adult", "child"):
        idx_list = role_indices[role]
        if not idx_list:
            continue
        base = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16
        ).to("cuda")
        peft_model = PeftModel.from_pretrained(base, adapter_dirs[role]).to("cuda")
        peft_model.eval()
        peft_model.config.use_cache = True

        for i in tqdm(range(0, len(idx_list), bs),
                      desc=f"  infer {role} test={test_case[-8:]}",
                      leave=False):
            batch_idx = idx_list[i: i + bs]
            chunk = feats_all[batch_idx].to("cuda", dtype=torch.float16, non_blocking=True)
            ids = peft_model.generate(
                input_features=chunk,
                language=LANGUAGE, task=TASK,
                num_beams=1, max_new_tokens=200,
            )
            out_texts = processor.batch_decode(ids, skip_special_tokens=True)
            for k, j in enumerate(batch_idx):
                preds[j] = {
                    "case": test_case,
                    "audio_path": paths[j],
                    "speaker": speakers[j],
                    "speaker_label": role,
                    "text": texts[j],
                    "hyp": out_texts[k].strip(),
                }
        del base, peft_model
        _free_gpu()

    # drop any unmatched (no role) entries
    return [p for p in preds if p is not None]


def main():
    ensure_dirs()
    folds = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["folds"]
    print(f"方案B: LOO over {len(folds)} folds with separate adult/child LoRA", flush=True)

    path_to_role = load_path_to_role()

    all_preds_path = PREDICTIONS_DIR / "loo_split_predictions.jsonl"
    fold_metrics_path = METRICS_DIR / "loo_split_fold_metrics.json"
    summary_path = METRICS_DIR / "loo_split_summary.json"

    # Resume
    completed_folds: set[int] = set()
    if all_preds_path.exists():
        with all_preds_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    completed_folds.add(json.loads(line).get("fold"))
                except json.JSONDecodeError:
                    pass
    if completed_folds:
        print(f"Resume: skipping already-finished folds {sorted(completed_folds)}", flush=True)

    fold_metrics: list[dict] = []
    if fold_metrics_path.exists():
        try:
            fold_metrics = json.loads(fold_metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fold_metrics = []

    all_preds: list[dict] = []
    if all_preds_path.exists():
        with all_preds_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    all_preds.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    t_start = time.time()
    for fold_def in folds:
        if fold_def["fold"] in completed_folds:
            print(f"Fold {fold_def['fold']} already complete — skipping.", flush=True)
            continue

        # Train two adapters
        adapter_dirs: dict[str, Path] = {}
        train_times: dict[str, float] = {}
        for role in ("adult", "child"):
            art = train_fold(fold_def, num_epochs=NUM_EPOCHS,
                             role=role, path_to_role=path_to_role)
            adapter_dirs[role] = art.adapter_dir
            train_times[role] = art.train_time_sec
            _free_gpu()

        # Inference with oracle routing
        preds = infer_split(fold_def["test_case"], adapter_dirs, path_to_role)
        for p in preds:
            p["fold"] = fold_def["fold"]

        with all_preds_path.open("a", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        all_preds.extend(preds)

        m = evaluate_fold(preds)
        fold_metrics.append({
            "fold": fold_def["fold"],
            "test_case": fold_def["test_case"],
            "val_case": fold_def["val_case"],
            "n_test_segments": len(preds),
            "train_time_adult_sec": train_times["adult"],
            "train_time_child_sec": train_times["child"],
            **m,
        })
        fold_metrics_path.write_text(
            json.dumps(fold_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  fold {fold_def['fold']} {fold_def['test_case']}  "
            f"overall CER={m['overall']['CER_or_WER']:.4f}  "
            f"BERTScore-F1={m['overall']['BERTScore_F1']:.4f}",
            flush=True,
        )
        _free_gpu()

    # ---- Aggregate ----
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
        "model": BASE_MODEL,
        "method": "方案B: LoRA fine-tune (LOO) — separate adult/child adapters, oracle routing",
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
    print(f"\nSummary -> {summary_path}", flush=True)
    print(
        f"Pooled overall  CER={overall_pooled['overall']['CER_or_WER']:.4f}  "
        f"BERTScore-F1={overall_pooled['overall']['BERTScore_F1']:.4f}",
        flush=True,
    )
    print(
        f"Macro overall   CER={macro_overall['CER_or_WER']:.4f}  "
        f"BERTScore-F1={macro_overall['BERTScore_F1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
