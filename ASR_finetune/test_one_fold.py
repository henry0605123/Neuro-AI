"""Quick LoRA single-fold sanity check (fold 0 only, 1 epoch).

Run this once before launching `run_loo.py` to make sure the LoRA training
loop and the inference plumbing work end-to-end without burning the full
13-fold compute budget.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import BASE_MODEL, FOLDS_FILE, LANGUAGE, TASK, ensure_dirs
from evaluate import evaluate_pairs
from lora_train import train_fold
from run_loo import infer_test_case, _load_manifest_by_path, _enrich_speaker_label


def main():
    ensure_dirs()
    folds = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["folds"]
    fold0 = folds[0]
    print(f"Testing fold 0: test={fold0['test_case']}", flush=True)

    t0 = time.time()
    artifacts = train_fold(fold0, num_epochs=1)
    print(f"Train done in {time.time()-t0:.1f}s", flush=True)

    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    ).to("cuda")
    peft_model = PeftModel.from_pretrained(base, artifacts.adapter_dir).to("cuda")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    preds = infer_test_case(peft_model, processor, fold0["test_case"])
    manifest = _load_manifest_by_path()
    _enrich_speaker_label(preds, manifest)

    refs = [p["text"] for p in preds]
    hyps = [p["hyp"] for p in preds]
    m = evaluate_pairs(refs, hyps)
    print(f"\nFold 0 (1 epoch) — n={m.n}  CER={m.cer:.4f}  BERTScore-F1={m.bertscore_f1:.4f}", flush=True)

    for label in ("adult", "child"):
        idx = [i for i, p in enumerate(preds) if p["speaker_label"] == label]
        if idx:
            ml = evaluate_pairs([refs[i] for i in idx], [hyps[i] for i in idx])
            print(f"  {label}: n={ml.n}  CER={ml.cer:.4f}  BERT-F1={ml.bertscore_f1:.4f}", flush=True)

    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
