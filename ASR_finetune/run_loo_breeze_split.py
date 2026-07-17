"""方案B for Breeze — single-stage Breeze-ASR-25 + SEPARATE adult/child LoRA adapters.

Per fold: train TWO adapters from Breeze (adult-only segments, child-only segments),
each with up to EPOCHS epochs + early stopping (patience). At inference, route each
test segment to its role's adapter via ground-truth speaker_label (oracle routing —
in production the diarization model supplies the label). Report overall/child/adult CER.

Compare against single-adapter Breeze single-stage:
  1 epoch       -> 39.07%   (child 59.09, adult 24.53)
  2 epoch (ES)  -> 37.67%   (loo_breeze_ep10es)
Hypothesis: a dedicated child adapter lowers child CER vs the shared adapter.

Env: EPOCHS (default 10), PATIENCE (default 2).
Adapted from run_loo_split.py (Belle 方案B). Outputs *_breeze_split (original split
results untouched). Single-stage by design (no ChangPN base).
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

BREEZE_MODEL = "MediaTek-Research/Breeze-ASR-25"
BREEZE_FEATURES_DIR = _config.PROJECT_ROOT / "features_breeze"
BREEZE_SPLIT_LORA_DIR = _config.LORA_DIR / "breeze_split"
_lt.BASE_MODEL = BREEZE_MODEL
_lt.FEATURES_DIR = BREEZE_FEATURES_DIR
_lt.LORA_DIR = BREEZE_SPLIT_LORA_DIR
BREEZE_SPLIT_LORA_DIR.mkdir(parents=True, exist_ok=True)

from config import (FOLDS_FILE, MANIFEST_FILE, PREDICTIONS_DIR, METRICS_DIR,
                    LANGUAGE, TASK, ensure_dirs)
from evaluate import evaluate_pairs
from lora_train import train_fold
from bootstrap_ci import bootstrap_ci_by_fold

EPOCHS = int(os.environ.get("EPOCHS", "10"))
PATIENCE = int(os.environ.get("PATIENCE", "2"))

PRED_PATH = PREDICTIONS_DIR / "loo_breeze_split_predictions.jsonl"
FOLD_METRICS_PATH = METRICS_DIR / "loo_breeze_split_fold_metrics.json"
SUMMARY_PATH = METRICS_DIR / "loo_breeze_split_summary.json"
LOG_PATH = METRICS_DIR / "loo_breeze_split.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _free():
    gc.collect(); torch.cuda.empty_cache()


def load_path_to_role() -> dict[str, str]:
    out = {}
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["audio_path"]] = r["speaker_label"]
    return out


def m2d(m): return {"n": m.n, "CER_or_WER": m.cer, "BERTScore_F1": m.bertscore_f1}


def evaluate_fold(preds: list[dict]) -> dict:
    refs = [p["text"] for p in preds]; hyps = [p["hyp"] for p in preds]
    out = {"overall": m2d(evaluate_pairs(refs, hyps))}
    for label in ("adult", "child"):
        idx = [i for i, p in enumerate(preds) if p.get("speaker_label") == label]
        if idx:
            out[label] = m2d(evaluate_pairs([refs[i] for i in idx], [hyps[i] for i in idx]))
    return out


@torch.no_grad()
def infer_split(test_case: str, adapter_dirs: dict[str, Path],
                path_to_role: dict[str, str]) -> list[dict]:
    blob = torch.load(BREEZE_FEATURES_DIR / f"{test_case}.pt", map_location="cpu", weights_only=False)
    feats_all = blob["features"]; texts = blob["texts"]
    speakers = blob["speakers"]; paths = blob["audio_paths"]
    n = feats_all.shape[0]

    role_indices = {"adult": [], "child": []}
    for i, p in enumerate(paths):
        r = path_to_role.get(p)
        if r in role_indices:
            role_indices[r].append(i)
    unmatched = n - sum(len(v) for v in role_indices.values())
    if unmatched:
        log(f"  [warn] {unmatched} test segments have no role mapping")

    preds = [None] * n
    processor = WhisperProcessor.from_pretrained(BREEZE_MODEL)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    bs = 8
    for role in ("adult", "child"):
        idx_list = role_indices[role]
        if not idx_list:
            continue
        base = WhisperForConditionalGeneration.from_pretrained(BREEZE_MODEL, torch_dtype=torch.float16).to("cuda")
        peft_model = PeftModel.from_pretrained(base, adapter_dirs[role]).to("cuda")
        peft_model.eval(); peft_model.config.use_cache = True
        for i in tqdm(range(0, len(idx_list), bs), desc=f"  infer {role} {test_case[-8:]}", leave=False):
            batch_idx = idx_list[i: i + bs]
            chunk = feats_all[batch_idx].to("cuda", dtype=torch.float16, non_blocking=True)
            ids = peft_model.generate(input_features=chunk, language=LANGUAGE, task=TASK,
                                      num_beams=1, max_new_tokens=200)
            out_texts = processor.batch_decode(ids, skip_special_tokens=True)
            for k, j in enumerate(batch_idx):
                preds[j] = {"case": test_case, "audio_path": paths[j], "speaker": speakers[j],
                            "speaker_label": role, "text": texts[j], "hyp": out_texts[k].strip()}
        del base, peft_model; _free()
    return [p for p in preds if p is not None]


def main():
    ensure_dirs()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    folds = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["folds"]
    log(f"=== Breeze 方案B: separate adult/child LoRA, oracle routing "
        f"({len(folds)} folds, EPOCHS={EPOCHS}, PATIENCE={PATIENCE}) ===")
    log(f"  base: {BREEZE_MODEL}  adapters: {BREEZE_SPLIT_LORA_DIR}")
    path_to_role = load_path_to_role()

    completed = set()
    all_preds = []
    if PRED_PATH.exists():
        with PRED_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line); all_preds.append(rec); completed.add(rec.get("fold"))
                except json.JSONDecodeError:
                    pass
    fold_metrics = []
    if FOLD_METRICS_PATH.exists():
        try:
            fold_metrics = json.loads(FOLD_METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fold_metrics = []
    if completed:
        log(f"Resume: skip folds {sorted(c for c in completed if c is not None)}")

    t0 = time.time()
    for fold_def in folds:
        if fold_def["fold"] in completed:
            log(f"Fold {fold_def['fold']} done, skip")
            continue
        adapter_dirs, train_times, best_epochs = {}, {}, {}
        for role in ("adult", "child"):
            art = train_fold(fold_def, num_epochs=EPOCHS, patience=PATIENCE,
                             role=role, path_to_role=path_to_role)
            adapter_dirs[role] = art.adapter_dir
            train_times[role] = art.train_time_sec
            try:
                info = json.loads((art.adapter_dir / "fold_info.json").read_text(encoding="utf-8"))
                best_epochs[role] = info.get("best_epoch")
            except Exception:
                best_epochs[role] = None
            _free()

        preds = infer_split(fold_def["test_case"], adapter_dirs, path_to_role)
        for p in preds:
            p["fold"] = fold_def["fold"]
        with PRED_PATH.open("a", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        all_preds.extend(preds)

        m = evaluate_fold(preds)
        fold_metrics.append({
            "fold": fold_def["fold"], "test_case": fold_def["test_case"],
            "val_case": fold_def["val_case"], "n_test_segments": len(preds),
            "best_epoch_adult": best_epochs.get("adult"), "best_epoch_child": best_epochs.get("child"),
            **m,
        })
        FOLD_METRICS_PATH.write_text(json.dumps(fold_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  fold {fold_def['fold']} {fold_def['test_case']}  overall={m['overall']['CER_or_WER']:.4f}"
            + (f"  child={m['child']['CER_or_WER']:.4f}" if 'child' in m else "")
            + (f"  adult={m['adult']['CER_or_WER']:.4f}" if 'adult' in m else ""))
        _free()

    overall_pooled = evaluate_fold(all_preds)
    macro_overall = {
        "CER_or_WER": float(np.mean([f["overall"]["CER_or_WER"] for f in fold_metrics])),
        "BERTScore_F1": float(np.mean([f["overall"]["BERTScore_F1"] for f in fold_metrics])),
    }
    macro_by_role = {}
    for label in ("adult", "child"):
        cers = [f[label]["CER_or_WER"] for f in fold_metrics if label in f]
        if cers:
            macro_by_role[label] = {"CER_or_WER": float(np.mean(cers)),
                                    "BERTScore_F1": float(np.mean([f[label]["BERTScore_F1"] for f in fold_metrics if label in f]))}
    log("\nBootstrap 重抽信賴區間(以 fold 為重抽單位,n_boot=1000)...")
    bootstrap = bootstrap_ci_by_fold(all_preds, n_boot=1000)
    summary = {
        "model": BREEZE_MODEL,
        "method": f"Breeze 方案B: separate adult/child LoRA adapters, oracle routing (single-stage, {EPOCHS}ep+ES)",
        "n_folds": len(folds), "total_time_sec": time.time() - t0,
        "pooled": overall_pooled,
        "macro_avg_over_folds": {"overall": macro_overall, **macro_by_role},
        "bootstrap_ci_pct25_50_75": bootstrap,
        "per_fold": fold_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n=== Breeze 方案B DONE ===")
    log(f"Pooled  CER={overall_pooled['overall']['CER_or_WER']:.4f}")
    for k, v in macro_by_role.items():
        log(f"  {k}: CER={v['CER_or_WER']:.4f}")
    log(f"Bootstrap CI (overall): p25={bootstrap['overall']['p25']:.4f} "
        f"p50={bootstrap['overall']['p50']:.4f} p75={bootstrap['overall']['p75']:.4f}")
    if bootstrap.get("child"):
        log(f"Bootstrap CI (child):   p25={bootstrap['child']['p25']:.4f} "
            f"p50={bootstrap['child']['p50']:.4f} p75={bootstrap['child']['p75']:.4f}")
    if bootstrap.get("adult"):
        log(f"Bootstrap CI (adult):   p25={bootstrap['adult']['p25']:.4f} "
            f"p50={bootstrap['adult']['p50']:.4f} p75={bootstrap['adult']['p75']:.4f}")
    log(f"Summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
