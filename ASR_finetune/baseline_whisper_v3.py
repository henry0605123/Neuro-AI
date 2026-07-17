"""Zero-shot baseline with vanilla openai/whisper-large-v3 (no Chinese FT).

Reuses the pre-computed mel features (whisper-large-v3 and Belle share the
same 128-bin mel front-end, so the cache is compatible).  Writes to
outputs/predictions/baseline_whisper_v3_predictions.jsonl and
outputs/metrics/baseline_whisper_v3_metrics.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import (
    LANGUAGE, TASK, FEATURES_DIR, MANIFEST_FILE,
    PREDICTIONS_DIR, METRICS_DIR, ensure_dirs,
)
from evaluate import evaluate_pairs


MODEL_ID = "openai/whisper-large-v3"
BATCH_SIZE_INFER = 8


def log(msg: str) -> None:
    print(msg, flush=True)


def load_model():
    t0 = time.time()
    log(f"Loading {MODEL_ID} ...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
    log(f"  processor loaded ({time.time()-t0:.1f}s)")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    log(f"  model on GPU ({time.time()-t0:.1f}s)")
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    return processor, model


def load_manifest_by_path():
    out = {}
    with MANIFEST_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["audio_path"]] = r
    return out


@torch.no_grad()
def infer_case(processor, model, case: str) -> list[dict]:
    blob = torch.load(FEATURES_DIR / f"{case}.pt",
                      map_location="cpu", weights_only=False)
    feats_all = blob["features"]
    texts = blob["texts"]
    speakers = blob["speakers"]
    paths = blob["audio_paths"]
    n = feats_all.shape[0]

    preds: list[dict] = []
    t0 = time.time()
    for i in range(0, n, BATCH_SIZE_INFER):
        chunk = feats_all[i: i + BATCH_SIZE_INFER].to(
            "cuda", dtype=torch.float16, non_blocking=True
        )
        ids = model.generate(
            input_features=chunk,
            language=LANGUAGE, task=TASK,
            num_beams=1, max_new_tokens=200,
        )
        out_texts = processor.batch_decode(ids, skip_special_tokens=True)
        for j, h in enumerate(out_texts):
            preds.append({
                "case": case,
                "audio_path": paths[i + j],
                "speaker": speakers[i + j],
                "text": texts[i + j],
                "hyp": h.strip(),
            })
    log(f"  {case}: n={n}  {time.time()-t0:.1f}s  ({n/(time.time()-t0):.1f} seg/s)")
    return preds


def main():
    ensure_dirs()
    cases = sorted({p.stem for p in FEATURES_DIR.glob("*.pt")})
    log(f"Cases with cached features: {len(cases)}")

    manifest_by_path = load_manifest_by_path()
    processor, model = load_model()

    all_preds: list[dict] = []
    out_preds_path = PREDICTIONS_DIR / "baseline_whisper_v3_predictions.jsonl"
    f_out = out_preds_path.open("w", encoding="utf-8")
    t_start = time.time()

    for ci, case in enumerate(cases):
        log(f"[{ci+1}/{len(cases)}] {case}")
        preds = infer_case(processor, model, case)
        for p in preds:
            rec = manifest_by_path.get(p["audio_path"])
            if rec is not None:
                p["speaker_label"] = rec["speaker_label"]
                p["duration"] = rec["duration"]
        for p in preds:
            f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
        f_out.flush()
        all_preds.extend(preds)
    f_out.close()
    log(f"\nTotal inference time: {time.time()-t_start:.1f}s for {len(all_preds)} segments")

    # metrics
    refs = [p["text"] for p in all_preds]
    hyps = [p["hyp"] for p in all_preds]
    overall = evaluate_pairs(refs, hyps)

    by_speaker = {}
    for label in ("adult", "child"):
        idx = [i for i, p in enumerate(all_preds) if p.get("speaker_label") == label]
        if idx:
            by_speaker[label] = evaluate_pairs(
                [refs[i] for i in idx], [hyps[i] for i in idx]
            )

    by_case = {}
    for c in sorted({p["case"] for p in all_preds}):
        idx = [i for i, p in enumerate(all_preds) if p["case"] == c]
        d = {"overall": evaluate_pairs(
            [refs[i] for i in idx], [hyps[i] for i in idx]
        )}
        for label in ("adult", "child"):
            cidx = [i for i in idx if all_preds[i].get("speaker_label") == label]
            if cidx:
                d[label] = evaluate_pairs(
                    [refs[i] for i in cidx], [hyps[i] for i in cidx]
                )
        by_case[c] = d

    def m2d(m):
        return {"n": m.n, "CER_or_WER": m.cer, "BERTScore_F1": m.bertscore_f1}

    summary = {
        "model": MODEL_ID,
        "setting": "zero-shot",
        "n_segments": len(all_preds),
        "overall": m2d(overall),
        "by_speaker": {k: m2d(v) for k, v in by_speaker.items()},
        "by_case": {c: {k: m2d(v) for k, v in d.items()}
                    for c, d in by_case.items()},
    }
    out_metrics_path = METRICS_DIR / "baseline_whisper_v3_metrics.json"
    out_metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\nWrote predictions: {out_preds_path}")
    log(f"Wrote metrics:     {out_metrics_path}")
    log(f"\nOverall  CER={overall.cer:.4f}  BERTScore-F1={overall.bertscore_f1:.4f}")
    for k, v in by_speaker.items():
        log(f"  {k:6s} CER={v.cer:.4f}  BERTScore-F1={v.bertscore_f1:.4f}  (n={v.n})")


if __name__ == "__main__":
    main()
