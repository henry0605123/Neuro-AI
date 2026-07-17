"""Evaluation utilities — WER (computed as CER for Chinese), BERTScore-F1.

For Chinese ASR the standard practice is character-level error rate (CER); we
expose it under both names so the reported metric matches the user's request
("WER") while reflecting Chinese-language convention.

Belle-whisper-large-v3-zh emits **Simplified** Chinese while our JSON
ground-truth is **Traditional** Chinese (Taiwan).  Without normalization
nearly every character would mismatch.  We convert both sides to Simplified
via OpenCC before scoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import jiwer
from opencc import OpenCC

_T2S = OpenCC("t2s")    # Traditional -> Simplified

# BERTScore is heavy to import — defer until first call
_BERT_SCORER = None


def normalize_zh(text: str) -> str:
    """Traditional -> Simplified; also strip whitespace at edges."""
    if not text:
        return ""
    return _T2S.convert(text).strip()


def _strip_punct(text: str) -> str:
    # Remove common Chinese + ASCII punctuation; keep Hanzi, latin, digits.
    return re.sub(r"[\s　，。、？！；：「」『』《》（）\(\)\[\]\.\?!,;:\"'`~\-—…·]+",
                  "", text or "")


def _to_char_string(text: str) -> str:
    """Insert spaces between every character so jiwer treats each char as a token.
    This gives CER under jiwer.wer; equivalent to jiwer.cer for Chinese text.
    Both ref and hyp are normalized to Simplified Chinese first.
    """
    return " ".join(list(_strip_punct(normalize_zh(text))))


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    """Character Error Rate (with traditional→simplified normalization)."""
    refs_c = [_to_char_string(r) for r in refs]
    hyps_c = [_to_char_string(h) for h in hyps]
    # Skip pairs where ref is empty (avoid div-by-zero)
    kept_r, kept_h = [], []
    for r, h in zip(refs_c, hyps_c):
        if r.strip():
            kept_r.append(r)
            kept_h.append(h)
    if not kept_r:
        return float("nan")
    return float(jiwer.wer(kept_r, kept_h))


def _get_bert_scorer():
    global _BERT_SCORER
    if _BERT_SCORER is None:
        import torch
        from bert_score import BERTScorer
        # bert-base-chinese is the standard choice for Chinese BERTScore;
        # it's reasonably small and covers the relevant vocabulary.
        _BERT_SCORER = BERTScorer(
            model_type="bert-base-chinese",
            lang="zh",
            num_layers=8,
            device="cuda" if torch.cuda.is_available() else "cpu",
            rescale_with_baseline=False,
        )
    return _BERT_SCORER


def compute_bertscore_f1(refs: list[str], hyps: list[str]) -> float:
    """BERTScore-F1 averaged across non-empty reference samples.
    bert-base-chinese is trained on Simplified Chinese, so we normalize both
    sides to Simplified before scoring.
    """
    pairs = [(normalize_zh(r), normalize_zh(h))
             for r, h in zip(refs, hyps) if (r or "").strip()]
    if not pairs:
        return float("nan")
    refs_k, hyps_k = zip(*pairs)
    scorer = _get_bert_scorer()
    P, R, F1 = scorer.score(list(hyps_k), list(refs_k), batch_size=64)
    return float(F1.mean().item())


@dataclass
class MetricResult:
    n: int
    cer: float       # = WER for Chinese
    bertscore_f1: float


def evaluate_pairs(refs: list[str], hyps: list[str]) -> MetricResult:
    if len(refs) != len(hyps):
        raise ValueError(f"len mismatch: refs={len(refs)} hyps={len(hyps)}")
    return MetricResult(
        n=len(refs),
        cer=compute_cer(refs, hyps),
        bertscore_f1=compute_bertscore_f1(refs, hyps),
    )


def evaluate_by_group(records: Iterable[dict], key: str) -> dict[str, MetricResult]:
    """records: dicts with at least {ref, hyp, <key>}. Group by `key` value."""
    buckets: dict[str, list[dict]] = {}
    for r in records:
        buckets.setdefault(r[key], []).append(r)
    out: dict[str, MetricResult] = {}
    for k, rows in buckets.items():
        out[k] = evaluate_pairs([r["ref"] for r in rows], [r["hyp"] for r in rows])
    return out
