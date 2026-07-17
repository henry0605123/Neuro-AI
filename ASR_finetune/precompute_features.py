"""Pre-compute Whisper input features (log-mel) for every segment.

Storing features upfront removes the CPU-side feature extraction bottleneck
during LoRA training (matches `feedback_training_efficiency` guidance).

Output:
  features/<case>.pt    -- dict with
      'features':   fp16 tensor (N, 80, 3000)
      'labels':     list of List[int] (variable-length token ids)
      'texts':      list[str]
      'speakers':   list[str]
      'audio_paths':list[str]
      'manifest_idx': list[int]
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from opencc import OpenCC
from tqdm import tqdm
from transformers import WhisperProcessor

from config import (
    BASE_MODEL, LANGUAGE, TASK, MANIFEST_FILE, FEATURES_DIR,
    SAMPLE_RATE, ensure_dirs,
)


def load_manifest_grouped():
    by_case: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            by_case[r["case"]].append((i, r))
    return by_case


def main():
    ensure_dirs()
    print(f"Loading processor {BASE_MODEL} ...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    cc = OpenCC("t2s")    # Traditional -> Simplified (Belle outputs simplified)

    by_case = load_manifest_grouped()
    print(f"Cases: {len(by_case)}; total segments: {sum(len(v) for v in by_case.values())}")

    for case, items in by_case.items():
        out_path = FEATURES_DIR / f"{case}.pt"
        if out_path.exists():
            print(f"[skip] {case} already cached -> {out_path.name}")
            continue

        feats_list: list[np.ndarray] = []
        labels_list: list[list[int]] = []
        texts, texts_simp, speakers, paths, midx = [], [], [], [], []

        for i, r in tqdm(items, desc=f"{case[-10:]}", leave=False):
            wav, _ = librosa.load(r["audio_path"], sr=SAMPLE_RATE, mono=True)
            wav = wav.astype(np.float32)
            inp = processor.feature_extractor(
                wav, sampling_rate=SAMPLE_RATE, return_tensors="np"
            )
            mel = inp.input_features[0]
            feats_list.append(mel.astype(np.float16))

            # Convert reference to Simplified Chinese for training labels
            # (Belle-whisper-large-v3-zh emits Simplified natively).
            simp_text = cc.convert(r["text"])
            tokens = processor.tokenizer(simp_text, add_special_tokens=True).input_ids
            labels_list.append(tokens)

            texts.append(r["text"])           # original (traditional)
            texts_simp.append(simp_text)       # simplified
            speakers.append(r["speaker"])
            paths.append(r["audio_path"])
            midx.append(i)

        features = torch.from_numpy(np.stack(feats_list, axis=0))
        torch.save({
            "features": features,
            "labels": labels_list,
            "texts": texts,
            "texts_simplified": texts_simp,
            "speakers": speakers,
            "audio_paths": paths,
            "manifest_idx": midx,
        }, out_path)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"[ok ] {case}: N={len(feats_list)}  ->  {out_path.name} ({size_mb:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
