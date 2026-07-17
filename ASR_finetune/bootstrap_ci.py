"""Bootstrap resampling confidence intervals for pooled CER.

`evaluate.compute_cer` computes POOLED CER (total edit distance / total ref length
across all given segments via jiwer.wer), not a mean of per-segment CERs. Pooled CER
is decomposable: CER(S) = sum(edits_i for i in S) / sum(ref_len_i for i in S). So we
precompute per-segment (edits, ref_len) ONCE, then bootstrap resampling is pure
arithmetic (no repeated jiwer calls) — orders of magnitude faster than recomputing
jiwer.wer from scratch every iteration.

We resample at the FOLD level (each fold = one held-out LOO test session), since a
session is the true independent sampling unit here (utterances within a session are
highly correlated — same two speakers, same room/mic). For each of `n_boot` iterations:
  1. Sample len(fold_ids) fold ids WITH replacement from the observed folds.
  2. Sum (edits, ref_len) over all segments belonging to the (possibly repeated) folds.
  3. CER = sum(edits) / sum(ref_len).
Report the 25th / 50th(median) / 75th percentiles of the bootstrap distribution.
"""
from __future__ import annotations
import jiwer
import numpy as np
from evaluate import _to_char_string


def _segment_stats(refs: list[str], hyps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment (edit_distance, ref_char_len) via jiwer.process_words on the
    space-separated character strings (matches evaluate.compute_cer's tokenization).
    Segments with empty ref are excluded (ref_char_len=0), matching compute_cer's
    'skip empty ref' behavior.
    """
    edits = np.zeros(len(refs), dtype=np.float64)
    lens = np.zeros(len(refs), dtype=np.float64)
    for i, (r, h) in enumerate(zip(refs, hyps)):
        rc, hc = _to_char_string(r), _to_char_string(h)
        if not rc.strip():
            continue
        out = jiwer.process_words(rc, hc)
        edits[i] = out.substitutions + out.deletions + out.insertions
        lens[i] = len(rc.split())
    return edits, lens


def bootstrap_ci_by_fold(preds: list[dict], n_boot: int = 1000, seed: int = 42) -> dict:
    """preds: list of dicts with keys 'fold', 'text' (ref), 'hyp', and optionally
    'speaker_label' ('adult'/'child'). Returns:
      {"overall": {"p25":.., "p50":.., "p75":..}, "child": {...}, "adult": {...}}
    """
    refs = [p["text"] for p in preds]
    hyps = [p["hyp"] for p in preds]
    edits, lens = _segment_stats(refs, hyps)
    fold_arr = np.array([p["fold"] for p in preds])
    role_arr = np.array([p.get("speaker_label") for p in preds])

    fold_ids = sorted(set(p["fold"] for p in preds))
    n_folds = len(fold_ids)
    # 每個 fold 各自的整體/child/adult 段落 index(供重抽時快速取用)
    fold_idx = {fid: np.where(fold_arr == fid)[0] for fid in fold_ids}

    rng = np.random.default_rng(seed)
    boot = {"overall": [], "child": [], "adult": []}
    for _ in range(n_boot):
        sampled = rng.choice(fold_ids, size=n_folds, replace=True)
        idx = np.concatenate([fold_idx[fid] for fid in sampled]) if n_folds else np.array([], dtype=int)
        tot_e, tot_l = edits[idx].sum(), lens[idx].sum()
        boot["overall"].append(tot_e / tot_l if tot_l > 0 else np.nan)
        for label in ("adult", "child"):
            mask = role_arr[idx] == label
            l = lens[idx][mask].sum()
            boot[label].append(edits[idx][mask].sum() / l if l > 0 else np.nan)

    def pct(vals):
        a = np.asarray(vals, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return None
        p25, p50, p75 = np.percentile(a, [25, 50, 75])
        return {"p25": float(p25), "p50": float(p50), "p75": float(p75)}

    return {"overall": pct(boot["overall"]), "child": pct(boot["child"]), "adult": pct(boot["adult"]),
            "n_boot": n_boot, "n_folds": n_folds}
