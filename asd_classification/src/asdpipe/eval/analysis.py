"""Phase 0 analyses (plan doc 4b.2, 4b.3; audit P0-0C, P0-4).

Order matters and is enforced by `run_phase0_analyses`: **criterion validity first, group
differences second** (audit P0-4). Seeing the group difference first and then adjusting a
feature definition is precisely the p-hacking route the analysis plan forbids.

The examiner probe is descriptive, never a pass/fail gate: at n~20 it has no decisive
power, so it is reported as a permutation p with a CI and the interaction features carry
the "may contain examiner modulation" caveat regardless of the result.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from ..contracts import SubjectTable

__all__ = ["spearman_with_ci", "holm", "criterion_validity", "group_differences",
           "examiner_probe", "verdict_for"]


def spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000,
                     seed: int = 0) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"rho": float("nan"), "p": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": int(x.size)}
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        xi, yi = x[draws[i]], y[draws[i]]
        if np.std(xi) == 0 or np.std(yi) == 0:
            continue
        boots[i] = spearmanr(xi, yi)[0]
    b = boots[np.isfinite(boots)]
    return {
        "rho": float(rho), "p": float(p), "n": int(x.size),
        "ci_low": float(np.percentile(b, 2.5)) if b.size else float("nan"),
        "ci_high": float(np.percentile(b, 97.5)) if b.size else float("nan"),
    }


def holm(pvalues: Mapping[str, float], alpha: float = 0.05,
         family_size: int | None = None) -> dict[str, dict]:
    """Holm-Bonferroni over the pre-registered family (plan doc 4b.2).

    ``family_size`` is the registered size (decision_rules.yaml
    ``criterion_validity.primary_indicator_family_size``). An indicator that could not be
    computed still counts towards it -- as if its p were 1 -- so a missing indicator never
    loosens the correction for the others. Without it the family silently shrank to the
    indicators that happened to be computable (health check P19).
    """
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items) if family_size is None else max(int(family_size), len(items))
    out: dict[str, dict] = {}
    rejected_so_far = True
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        reject = rejected_so_far and p <= thr
        rejected_so_far = reject
        out[k] = {"p": float(p), "threshold": float(thr), "reject": bool(reject), "rank": i + 1}
    for k, v in pvalues.items():
        out.setdefault(k, {"p": float(v), "threshold": float("nan"),
                           "reject": False, "rank": -1})
    return out


def verdict_for(rho: float, ci_low: float, ci_high: float, cohen_d: float,
                rules: Mapping) -> str:
    """The pre-registered decision table (plan doc 4b.2-5). Never re-derived from results.

    Two implementation points the table leaves implicit:

    * the threshold is on the **magnitude**. Several indicators have a negative expected
      direction (A12 volume following goes down as the A2 abnormality score goes up), and
      comparing a signed rho against +0.5 would silently fail those.
    * "rho >= 0.5 but the CI includes 0" is not a row in the table. It falls between rows 1
      and 2, so it is reported as a candidate rather than being dropped -- a large point
      estimate with an unstable interval is exactly the winner's-curse case section 4b.2-6
      warns about.
    """
    thr = float(rules["criterion_validity"]["effect_size_threshold_rho"])
    if not np.isfinite(rho):
        return "not computable"
    mag = abs(rho)
    excludes_zero = np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low > 0 or ci_high < 0)
    if mag >= thr and excludes_zero:
        return "Phase 1 core feature"
    if mag >= thr:
        return "Phase 1 candidate; effect size met but the CI includes 0"
    if 0.3 <= mag < thr:
        return "Phase 1 candidate; review operational definition"
    if mag < 0.3 and np.isfinite(cohen_d) and abs(cohen_d) >= 0.8:
        return "WARNING: separates groups but may measure something else -- trace confound"
    return "demote or drop"


def _feature_lookup(table: SubjectTable) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for s in ("b", "c", "e"):
        X, names = table.stream(s), table.feature_names(s)
        for j, n in enumerate(names):
            out[n] = X[:, j]
    return out


# Which feature carries each pre-registered primary indicator (decision_rules.yaml).
PRIMARY_FEATURE: dict[str, str] = {
    "G1": "e_g1_similarity_median",
    "F1": "e_f1_embed_high_ratio",
    "F3": "e_f3_repeated_type_ratio",
    "B7": "b7_mean_pairwise_cosdist",
    "A12": "e_volume_following_slope",
    "F8": "e_f8_asymmetry",
    "B4": None,     # manual annotation, needs real data (audit P0-F6)
}

# ADOS criterion for each indicator (plan doc 4b.1). F8/G1 share A8 (QUESTIONS Q9).
# F7 left the family on 2026-09-12 (DEVIATIONS D8) and is no longer listed (P19).
CRITERION: dict[str, str] = {
    "G1": "A8", "F1": "A3", "F3": "A4", "B7": "A2",
    "A12": "A2", "F8": "A8", "B4": "A2",
}


def criterion_validity(table: SubjectTable, rules: Mapping, n_boot: int = 10000,
                       seed: int = 0) -> dict[str, dict]:
    """PRIMARY Phase 0 analysis: feature vs its ADOS item, Spearman + bootstrap CI."""
    registered = set(rules["criterion_validity"]["primary_indicators"])
    if registered != set(PRIMARY_FEATURE):
        raise ValueError(f"PRIMARY_FEATURE {sorted(PRIMARY_FEATURE)} does not match the "
                         f"registered family {sorted(registered)} (health check P19)")
    feats = _feature_lookup(table)
    results: dict[str, dict] = {}
    for indicator, feature in PRIMARY_FEATURE.items():
        item = CRITERION[indicator]
        entry: dict = {"feature": feature, "criterion": item}
        if feature is None or feature not in feats or item not in table.ados:
            entry.update({"status": "needs real data" if feature is None else "unavailable",
                          "rho": float("nan"), "p": float("nan"),
                          "ci_low": float("nan"), "ci_high": float("nan")})
        else:
            entry.update(spearman_with_ci(feats[feature], table.ados[item], n_boot, seed))
            entry["status"] = "computed"
            # binarised AUC: 0 (no abnormality) vs >=1, per Li/Zou (2019)
            binary = (table.ados[item] >= 1).astype(int)
            if len(np.unique(binary)) > 1:
                from .metrics import safe_auc

                entry["binarised_auc"] = safe_auc(binary, feats[feature])
            else:
                entry["binarised_auc"] = float("nan")
        results[indicator] = entry
    return results


def group_differences(table: SubjectTable, n_boot: int = 10000, seed: int = 0) -> dict[str, dict]:
    """SECONDARY analysis. Runs only after criterion validity (audit P0-4)."""
    feats = _feature_lookup(table)
    y = table.y
    out: dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    for indicator, feature in PRIMARY_FEATURE.items():
        if feature is None or feature not in feats:
            out[indicator] = {"feature": feature, "status": "needs real data"}
            continue
        v = feats[feature]
        a, b = v[y == 1], v[y == 0]
        if a.size < 2 or b.size < 2 or np.std(v) == 0:
            out[indicator] = {"feature": feature, "status": "not computable"}
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rank_biserial = 1.0 - 2.0 * u / (a.size * b.size)
        pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                         / max(a.size + b.size - 2, 1))
        d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")
        boots = np.empty(n_boot)
        for i in range(n_boot):
            ai = a[rng.integers(0, a.size, a.size)]
            bi = b[rng.integers(0, b.size, b.size)]
            pl = np.sqrt(((ai.size - 1) * ai.var(ddof=1) + (bi.size - 1) * bi.var(ddof=1))
                         / max(ai.size + bi.size - 2, 1))
            boots[i] = (ai.mean() - bi.mean()) / pl if pl > 0 else np.nan
        bb = boots[np.isfinite(boots)]
        out[indicator] = {
            "feature": feature, "status": "computed", "u_p": float(p),
            "rank_biserial": float(-rank_biserial), "cohen_d": d,
            "d_ci_low": float(np.percentile(bb, 2.5)) if bb.size else float("nan"),
            "d_ci_high": float(np.percentile(bb, 97.5)) if bb.size else float("nan"),
        }
    return out


def examiner_probe(table: SubjectTable, n_permutations: int = 5000, seed: int = 0,
                   n_folds: int = 3) -> dict:
    """Audit P0-0C: can a classifier trained on EXAMINER speech alone predict the diagnosis?

    Descriptive by construction. At n~20 the 95% interval under no leakage is roughly
    [0.30, 0.70] accuracy, so neither a high nor a low value settles anything -- which is
    why the interaction features carry their caveat either way.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from .metrics import safe_auc
    from ..cv.splits import subject_folds

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    X = np.array(table.X_probe, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan            # imputed inside each fold (health check P12)
    y = table.y

    def _cv_score(labels: np.ndarray) -> tuple[float, float]:
        oof = np.zeros(len(labels))
        for fold in subject_folds(labels, n_folds, seed):
            clf = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(X[fold.train_idx], labels[fold.train_idx])
            oof[fold.test_idx] = clf.decision_function(X[fold.test_idx])
        pred = (oof > 0).astype(int)
        return float(np.mean(pred == labels)), safe_auc(labels, oof)

    acc, auc = _cv_score(y)
    rng = np.random.default_rng(seed + 4242)
    null_acc = np.empty(n_permutations)
    for i in range(n_permutations):
        null_acc[i] = _cv_score(rng.permutation(y))[0]
    p = float((1.0 + np.sum(null_acc >= acc)) / (n_permutations + 1.0))
    return {
        "accuracy": acc, "auc": auc, "permutation_p": p,
        "null_mean": float(null_acc.mean()),
        "null_ci_low": float(np.percentile(null_acc, 2.5)),
        "null_ci_high": float(np.percentile(null_acc, 97.5)),
        "n_permutations": int(n_permutations),
        "interpretation": "descriptive only -- n is far too small for this probe to decide "
                          "anything; interaction features (G1/F1/F8/A12) carry the "
                          "'may contain examiner modulation' caveat regardless",
    }
