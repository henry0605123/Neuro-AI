"""Breeze-ASR-25 FULL fine-tune (no LoRA) — LOO on the 19 CLEAN folds.

Breeze = whisper-large-v2 backbone (1.543B params, 80-mel). Full FT of a 1.5B model
on a 16GB card is only feasible with the ACCELERATED recipe:
  - load weights in bf16 (torch_dtype=bfloat16) — no fp32 master copy
  - bitsandbytes AdamW8bit optimizer (8-bit optimizer states ~3GB vs fp32 ~12GB)
  - gradient_checkpointing (trade compute for activation memory)
  - batch=1 (VRAM is the hard limit)
Reuses features_breeze/ (80-mel, same cache as whisper-medium/breeze-split runs) so
results are directly comparable to those experiments.

WARNING: pure bf16 full FT (no fp32 master weights) can be numerically unstable — watch
val_loss for divergence. This is the price of fitting 1.5B full FT into 16GB.

Env:
  TAG(breeze_full_19fold) EPOCHS(4) PATIENCE(2) LR(1e-5) BATCH(1)
  GRAD_ACCUM(8)  — effective batch = BATCH*GRAD_ACCUM
  FOLD_LIMIT(0)  — 0 = all folds; N>0 = run only first N folds (smoke test: FOLD_LIMIT=1)
"""
from __future__ import annotations
import gc, json, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          get_cosine_schedule_with_warmup)
from bitsandbytes.optim import AdamW8bit

import config as _config
import lora_train as _lt
from config import (MANIFEST_FILE, PREDICTIONS_DIR, METRICS_DIR, FOLDS_FILE,
                    LANGUAGE, TASK, ensure_dirs)
from evaluate import evaluate_pairs
from bootstrap_ci import bootstrap_ci_by_fold

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

MODEL_NAME = os.environ.get("BREEZE_MODEL", "MediaTek-Research/Breeze-ASR-25")
TAG        = os.environ.get("TAG", "breeze_full_19fold")
EPOCHS     = int(os.environ.get("EPOCHS", "4"))
PATIENCE   = int(os.environ.get("PATIENCE", "2"))
LR         = float(os.environ.get("LR", "1e-5"))
BATCH      = int(os.environ.get("BATCH", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
FOLD_LIMIT = int(os.environ.get("FOLD_LIMIT", "0"))
GRAD_CKPT  = os.environ.get("GRAD_CKPT", "1") == "1"   # 關掉可加速(結果不變),顯存換速度
FINAL       = os.environ.get("FINAL", "0") == "1"      # 用全部 case 訓練一個交付模型(無 val/early-stop)
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", "3"))  # LOO 的 best_epoch 中位數=2、平均 2.5 → 固定 3

# 與 Breeze/whisper-medium 用同一份 80-mel 特徵,確保公平對照
_lt.FEATURES_DIR = _config.PROJECT_ROOT / "features_breeze"

MODEL_DIR = _config.PROJECT_ROOT / "models" / TAG
PRED_PATH = PREDICTIONS_DIR / f"loo_{TAG}_predictions.jsonl"
FOLD_METRICS_PATH = METRICS_DIR / f"loo_{TAG}_fold_metrics.json"
SUMMARY_PATH = METRICS_DIR / f"loo_{TAG}_summary.json"
LOG_PATH = METRICS_DIR / f"loo_{TAG}.log"


def log(msg):
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _free():
    gc.collect(); torch.cuda.empty_cache()


def _vram():
    return (f"alloc={torch.cuda.memory_allocated()/1e9:.1f}G "
            f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}G")


def build_clean_folds():
    payload = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))
    cases = payload["all_cases"]
    folds = []
    for i, tc in enumerate(cases):
        tr = [c for c in cases if c != tc]
        folds.append({"fold": i, "test_case": tc, "val_case": tr[0], "train_cases": tr[1:]})
    return folds, cases


def load_role_map():
    out = {}
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["audio_path"]] = r.get("speaker_label")
    return out


def build_model():
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16)          # bf16 weights, no fp32 master
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    if GRAD_CKPT:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model.to("cuda"), processor


@torch.no_grad()
def eval_loss(model, loader):
    model.eval(); tot = torch.zeros((), device="cuda"); n = 0
    for feats, labels in loader:
        feats = feats.to("cuda", dtype=torch.bfloat16, non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)
        out = model(input_features=feats, labels=labels)
        tot += out.loss.detach() * feats.size(0); n += feats.size(0)
    return (tot / max(n, 1)).item()


@torch.no_grad()
def infer(model, processor, test_case, role_map):
    blob = torch.load(_lt.FEATURES_DIR / f"{test_case}.pt", map_location="cpu", weights_only=False)
    feats_all = blob["features"]; texts = blob["texts"]
    speakers = blob["speakers"]; paths = blob["audio_paths"]
    model.eval(); model.config.use_cache = True
    preds = []
    for i in tqdm(range(0, feats_all.shape[0], 8), desc=f"  infer {test_case[-8:]}", leave=False):
        chunk = feats_all[i:i + 8].to("cuda", dtype=torch.bfloat16, non_blocking=True)
        ids = model.generate(input_features=chunk, language=LANGUAGE, task=TASK,
                             num_beams=1, max_new_tokens=150,
                             repetition_penalty=1.3, no_repeat_ngram_size=5)
        outs = processor.batch_decode(ids, skip_special_tokens=True)
        for j, h in enumerate(outs):
            preds.append({"case": test_case, "audio_path": paths[i + j],
                          "speaker": speakers[i + j], "text": texts[i + j], "hyp": h.strip(),
                          "speaker_label": role_map.get(paths[i + j])})
    model.config.use_cache = False
    return preds


def m2d(m): return {"n": m.n, "CER_or_WER": m.cer, "BERTScore_F1": m.bertscore_f1}


def evaluate_fold(preds):
    refs = [p["text"] for p in preds]; hyps = [p["hyp"] for p in preds]
    out = {"overall": m2d(evaluate_pairs(refs, hyps))}
    for label in ("adult", "child"):
        idx = [i for i, p in enumerate(preds) if p.get("speaker_label") == label]
        if idx:
            out[label] = m2d(evaluate_pairs([refs[i] for i in idx], [hyps[i] for i in idx]))
    return out


def train_one_fold(fold_def):
    model, processor = build_model()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    train_ds = _lt.CachedFeaturesDataset(fold_def["train_cases"], pad_id)
    val_ds = _lt.CachedFeaturesDataset([fold_def["val_case"]], pad_id)
    train_loader = _lt._make_loader(train_ds, BATCH, pad_id, True)
    val_loader = _lt._make_loader(val_ds, BATCH, pad_id, False)
    log(f"    train segs={len(train_ds)}  val segs={len(val_ds)}  batch={BATCH} accum={GRAD_ACCUM}")

    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    optim = AdamW8bit(model.parameters(), lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optim, int(total_steps * 0.1), total_steps)

    best_val, best_epoch, no_improve, best_state = float("inf"), 0, 0, None
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); loss_acc = torch.zeros((), device="cuda"); nb = 0
        optim.zero_grad(set_to_none=True)
        for bi, (feats, labels) in enumerate(tqdm(train_loader, desc=f"    ep{epoch+1}/{EPOCHS}", leave=False)):
            feats = feats.to("cuda", dtype=torch.bfloat16, non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            out = model(input_features=feats, labels=labels)
            (out.loss / GRAD_ACCUM).backward()
            loss_acc += out.loss.detach(); nb += 1
            if (bi + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            if bi == 0 and epoch == 0:
                log(f"    [vram after 1st step] {_vram()}")
        vl = eval_loss(model, val_loader)
        tl = (loss_acc / max(nb, 1)).item()
        log(f"    epoch {epoch+1}: train_loss={tl:.4f}  val_loss={vl:.4f}  [{_vram()}]")
        if vl < best_val - 1e-3:
            best_val, best_epoch, no_improve = vl, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log(f"    early stop at epoch {epoch+1} (best={best_epoch}, val={best_val:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, processor, best_epoch, time.time() - t0, best_val


def train_final(cases, epochs):
    """用全部 cases 訓練一個交付模型:固定 epochs、無 val / 無 early-stop、存最後權重。
    epochs 取自 LOO 交叉驗證得到的最佳點(best_epoch 中位數),而非再靠 held-out val。"""
    model, processor = build_model()
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    train_ds = _lt.CachedFeaturesDataset(cases, pad_id)
    train_loader = _lt._make_loader(train_ds, BATCH, pad_id, True)
    log(f"  FINAL: train on ALL {len(cases)} cases, {len(train_ds)} segs, "
        f"{epochs} epochs (no val / no early-stop), batch={BATCH} accum={GRAD_ACCUM}")
    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    total_steps = steps_per_epoch * epochs
    optim = AdamW8bit(model.parameters(), lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optim, int(total_steps * 0.1), total_steps)
    t0 = time.time()
    for epoch in range(epochs):
        model.train(); loss_acc = torch.zeros((), device="cuda"); nb = 0
        optim.zero_grad(set_to_none=True)
        for bi, (feats, labels) in enumerate(tqdm(train_loader, desc=f"    ep{epoch+1}/{epochs}", leave=False)):
            feats = feats.to("cuda", dtype=torch.bfloat16, non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            out = model(input_features=feats, labels=labels)
            (out.loss / GRAD_ACCUM).backward()
            loss_acc += out.loss.detach(); nb += 1
            if (bi + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            if bi == 0 and epoch == 0:
                log(f"    [vram after 1st step] {_vram()}")
        log(f"    epoch {epoch+1}: train_loss={(loss_acc / max(nb, 1)).item():.4f}  [{_vram()}]")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)
    log(f"  FINAL model saved -> {MODEL_DIR}  ({(time.time()-t0)/60:.1f} min)")
    return MODEL_DIR


def save_fold_checkpoint(model, processor, fold_def):
    ckpt_dir = MODEL_DIR / f"fold{fold_def['fold']:02d}_test={fold_def['test_case']}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    return ckpt_dir


def main():
    ensure_dirs(); LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PRED_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    folds, cases = build_clean_folds()

    if FINAL:
        torch.cuda.reset_peak_memory_stats()
        log(f"=== Breeze FULL finetune FINAL model | {MODEL_NAME} | ALL {len(cases)} cases ===")
        log(f"  recipe: bf16 + AdamW8bit + grad-ckpt={GRAD_CKPT} | "
            f"FINAL_EPOCHS={FINAL_EPOCHS} LR={LR} BATCH={BATCH} GRAD_ACCUM={GRAD_ACCUM}")
        train_final(cases, FINAL_EPOCHS)
        log("=== FINAL DONE ===")
        return

    if FOLD_LIMIT > 0:
        folds = folds[:FOLD_LIMIT]
    role_map = load_role_map()
    log(f"=== Breeze FULL finetune LOO | {MODEL_NAME} | {len(folds)} folds "
        f"(of {len(cases)} cases) | SMOKE={FOLD_LIMIT>0} ===")
    log(f"  recipe: bf16 weights + AdamW8bit + grad-ckpt={GRAD_CKPT} | "
        f"EPOCHS={EPOCHS} PATIENCE={PATIENCE} LR={LR} BATCH={BATCH} GRAD_ACCUM={GRAD_ACCUM}")

    completed, all_preds, fold_metrics = set(), [], []
    if PRED_PATH.exists():
        for line in PRED_PATH.open(encoding="utf-8"):
            try:
                r = json.loads(line); all_preds.append(r)
                if r.get("fold") is not None: completed.add(r["fold"])
            except json.JSONDecodeError:
                pass
    if FOLD_METRICS_PATH.exists():
        try: fold_metrics = json.loads(FOLD_METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError: fold_metrics = []
    if completed: log(f"  resume: skip folds {sorted(completed)}")

    t_all = time.time()
    for fd in folds:
        if fd["fold"] in completed:
            log(f"Fold {fd['fold']} done, skip"); continue
        torch.cuda.reset_peak_memory_stats()
        log(f"\n=== Fold {fd['fold']} | test={fd['test_case']} | val={fd['val_case']} "
            f"| train={len(fd['train_cases'])} cases ===")
        model, processor, best_epoch, dt, best_val = train_one_fold(fd)
        ckpt_dir = save_fold_checkpoint(model, processor, fd)
        preds = infer(model, processor, fd["test_case"], role_map)
        for p in preds: p["fold"] = fd["fold"]
        with PRED_PATH.open("a", encoding="utf-8") as f:
            for p in preds: f.write(json.dumps(p, ensure_ascii=False) + "\n")
        all_preds.extend(preds)
        m = evaluate_fold(preds)
        fold_metrics.append({"fold": fd["fold"], "test_case": fd["test_case"],
                             "val_case": fd["val_case"], "best_epoch": best_epoch,
                             "best_val_loss": best_val, "checkpoint_dir": str(ckpt_dir),
                             "train_time_sec": dt, "n_test_segments": len(preds), **m})
        FOLD_METRICS_PATH.write_text(json.dumps(fold_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  fold {fd['fold']} {fd['test_case']} best_ep={best_epoch} "
            f"CER={m['overall']['CER_or_WER']:.4f}"
            + (f"  child={m['child']['CER_or_WER']:.4f}" if 'child' in m else "")
            + f"  train_time={dt/60:.1f}min  peak_vram={torch.cuda.max_memory_allocated()/1e9:.1f}G")
        del model; _free()

    if FOLD_LIMIT > 0:
        log(f"\n=== SMOKE TEST DONE ({len(folds)} fold) — 檢查上面的 peak_vram / val_loss / train_time ===")
        return

    pooled = evaluate_fold(all_preds)
    macro = {"CER_or_WER": float(np.mean([f["overall"]["CER_or_WER"] for f in fold_metrics])),
             "BERTScore_F1": float(np.mean([f["overall"]["BERTScore_F1"] for f in fold_metrics]))}
    macro_role = {}
    for label in ("adult", "child"):
        cers = [f[label]["CER_or_WER"] for f in fold_metrics if label in f]
        if cers:
            macro_role[label] = {"CER_or_WER": float(np.mean(cers)),
                                 "BERTScore_F1": float(np.mean([f[label]["BERTScore_F1"] for f in fold_metrics if label in f]))}
    log("\nBootstrap 重抽信賴區間(以 fold 為重抽單位,n_boot=1000)...")
    bootstrap = bootstrap_ci_by_fold(all_preds, n_boot=1000)
    summary = {"model": MODEL_NAME,
               "method": f"Breeze FULL finetune LOO (bf16+AdamW8bit+gradckpt, {EPOCHS}ep+ES)",
               "n_folds": len(folds), "total_time_sec": time.time() - t_all,
               "pooled": pooled, "macro_avg_over_folds": {"overall": macro, **macro_role},
               "bootstrap_ci_pct25_50_75": bootstrap, "per_fold": fold_metrics}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n=== {TAG} DONE ===  Pooled CER={pooled['overall']['CER_or_WER']:.4f}")
    for k, v in macro_role.items(): log(f"  {k}: CER={v['CER_or_WER']:.4f}")
    log(f"Bootstrap CI (overall): p25={bootstrap['overall']['p25']:.4f} "
        f"p50={bootstrap['overall']['p50']:.4f} p75={bootstrap['overall']['p75']:.4f}")
    log(f"Summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
