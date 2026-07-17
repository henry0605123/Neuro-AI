"""Re-compute metrics from existing predictions JSONL.

Useful when the evaluation logic (e.g. trad→simp normalization) changes and we
want to avoid re-running expensive ASR inference.

Usage:
    python recompute_metrics.py outputs/predictions/baseline_predictions.jsonl \
                                outputs/metrics/baseline_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from config import BASE_MODEL
from evaluate import evaluate_pairs


def m2d(m):
    return {"n": m.n, "CER_or_WER": m.cer, "BERTScore_F1": m.bertscore_f1}


def main():
    if len(sys.argv) != 3:
        print("usage: recompute_metrics.py <preds.jsonl> <out_metrics.json>")
        sys.exit(1)
    preds_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    preds = []
    with preds_path.open(encoding="utf-8") as f:
        for line in f:
            preds.append(json.loads(line))
    print(f"Loaded {len(preds)} predictions from {preds_path}", flush=True)

    refs = [p["text"] for p in preds]
    hyps = [p["hyp"] for p in preds]
    overall = evaluate_pairs(refs, hyps)
    print(f"Overall  CER={overall.cer:.4f}  BERT-F1={overall.bertscore_f1:.4f}", flush=True)

    by_speaker = {}
    for label in ("adult", "child"):
        idx = [i for i, p in enumerate(preds) if p.get("speaker_label") == label]
        if idx:
            by_speaker[label] = evaluate_pairs(
                [refs[i] for i in idx], [hyps[i] for i in idx]
            )
            print(f"  {label}: CER={by_speaker[label].cer:.4f}  "
                  f"BERT-F1={by_speaker[label].bertscore_f1:.4f}  "
                  f"(n={by_speaker[label].n})", flush=True)

    by_case = {}
    for c in sorted({p["case"] for p in preds}):
        idx = [i for i, p in enumerate(preds) if p["case"] == c]
        d = {"overall": evaluate_pairs(
            [refs[i] for i in idx], [hyps[i] for i in idx]
        )}
        for label in ("adult", "child"):
            cidx = [i for i in idx if preds[i].get("speaker_label") == label]
            if cidx:
                d[label] = evaluate_pairs(
                    [refs[i] for i in cidx], [hyps[i] for i in cidx]
                )
        by_case[c] = d

    summary = {
        "model": BASE_MODEL,
        "source_predictions": str(preds_path),
        "n_segments": len(preds),
        "overall": m2d(overall),
        "by_speaker": {k: m2d(v) for k, v in by_speaker.items()},
        "by_case": {c: {k: m2d(v) for k, v in d.items()} for c, d in by_case.items()},
    }
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
