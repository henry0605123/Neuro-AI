"""Replace the cached label token ids in features/<case>.pt with tokenizations
of the **Simplified-Chinese** version of each reference text.

Why: Belle-whisper-large-v3-zh natively outputs Simplified.  Training on the
original Traditional labels would force the model to also re-learn the script
change, which is wasteful with limited data.  We instead train on Simplified
labels and evaluate with both sides normalized to Simplified — this isolates
the actual ASR error.

Audio features are not touched.
"""
from __future__ import annotations

import json

import torch
from opencc import OpenCC
from tqdm import tqdm

from transformers import WhisperProcessor

from config import BASE_MODEL, FEATURES_DIR, LANGUAGE, TASK


def main():
    cc = OpenCC("t2s")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

    pt_files = sorted(FEATURES_DIR.glob("*.pt"))
    for pt in pt_files:
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        texts = blob["texts"]
        new_labels = []
        new_texts_simp = []
        for t in texts:
            simp = cc.convert(t)
            new_texts_simp.append(simp)
            new_labels.append(processor.tokenizer(simp, add_special_tokens=True).input_ids)
        # Keep both versions for traceability
        blob["labels"] = new_labels
        blob["texts_simplified"] = new_texts_simp
        torch.save(blob, pt)
        print(f"  updated {pt.name}  (N={len(new_labels)})")
    print("Done.")


if __name__ == "__main__":
    main()
