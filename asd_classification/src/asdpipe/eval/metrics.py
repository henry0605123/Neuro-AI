"""Subject-level metrics, paired comparison and calibration (plan doc 7.3, 7.6).

Everything is reported at the **subject** level -- chunk-level reporting is the inflation
trap in plan doc 9.4 and there are no chunks in this pipeline anyway.

The paired bootstrap resamples *subjects*, which is the correct unit, and is always run on
the same split and the same seed list for every model (decision_rules.yaml).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

__all__ = ["safe_auc", "subject_metrics", "paired_bootstrap", "bootstrap_ci",
           "platt_calibrate", "empirical_mde", "fold_center"]


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def fold_center(scores: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    """Centre out-of-fold scores within each fold before pooling (audit P1-9).

    With LOSO the training prevalence is negatively correlated with the held-out label, and
    pooling raw scores gives a below-chance AUC even for a null feature. Centring per fold
    removes that offset; with balanced k-fold it is a no-op in expectation.
    """
    out = np.array(scores, dtype=float)
    for f in np.unique(fold_ids):
        m = fold_ids == f
        if m.sum() > 1:
            out[m] = out[m] - np.mean(out[m])
    return out


def subject_metrics(y: np.ndarray, scores: np.ndarray, probs: np.ndarray | None = None,
                    threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    p = np.asarray(probs, dtype=float) if probs is not None else 1.0 / (1.0 + np.exp(-s))
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "auc": safe_auc(y, s),
        "accuracy": (tp + tn) / max(len(y), 1),
        "sensitivity": sens,
        "specificity": spec,
        "f1": 2 * prec * sens / max(prec + sens, 1e-9),
        "brier": float(brier_score_loss(y, np.clip(p, 0.0, 1.0))) if len(np.unique(y)) > 1 else float("nan"),
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[draws].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                     n_boot: int = 10000, seed: int = 0,
                     alpha: float = 0.05) -> dict[str, float]:
    """Paired subject-level bootstrap of AUC(a) - AUC(b) on identical splits and seeds."""
    y = np.asarray(y)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    observed = safe_auc(y, a) - safe_auc(y, b)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs[i] = safe_auc(y[idx], a[idx]) - safe_auc(y[idx], b[idx])
    d = diffs[np.isfinite(diffs)]
    if d.size == 0:
        return {"observed": observed, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_two_sided": float("nan"), "n_valid": 0}
    lo = float(np.percentile(d, 100 * alpha / 2))
    hi = float(np.percentile(d, 100 * (1 - alpha / 2)))
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2.0 * min(float(np.mean(d <= 0.0)), float(np.mean(d >= 0.0)))
    return {"observed": float(observed), "ci_low": lo, "ci_high": hi,
            "p_two_sided": float(min(1.0, p)), "n_valid": int(d.size),
            "se": float(np.std(d))}


def empirical_mde(se: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable effect from the observed paired-bootstrap SE (plan doc 5.0.2-2).

    A non-significant result means "gain < MDE", never "gain = 0"; the report must say so.
    """
    z_alpha, z_beta = 1.959963985, 0.8416212336
    return float((z_alpha + z_beta) * se) if np.isfinite(se) else float("nan")


def platt_calibrate(train_scores: np.ndarray, train_y: np.ndarray,
                    target_scores: np.ndarray) -> np.ndarray:
    """Platt scaling fitted **inside the training fold** (plan doc 5.7, decision_rules.yaml)."""
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(train_scores, dtype=float).reshape(-1, 1)
    if len(np.unique(train_y)) < 2:
        return 1.0 / (1.0 + np.exp(-np.asarray(target_scores, dtype=float)))
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(s, np.asarray(train_y))
    return lr.predict_proba(np.asarray(target_scores, dtype=float).reshape(-1, 1))[:, 1]
