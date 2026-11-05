"""Mandatory baselines B0 / B1 / B2 (R5, plan doc 5.0.2 and 5.0.2b).

B0  three-stream statistical pooling -> concatenation (36-66 dims) -> L2 logistic
    regression / shrinkage LDA. The linear reference and the final fallback.
B1  stacked generalisation (Wolpert 1992): one in-fold logistic regression per stream
    produces an out-of-fold score f_s, then the fusion layer learns logit = b + sum_s w_s
    f_s -- exactly 3 + 1 parameters. Also the source of the single-stream leaderboard.
B2  eGeMAPS functionals + shrinkage LDA. The external reference point.

Deliberately *not* implemented: the pre-v3.5 "global stream weights multiplied onto feature
blocks" design. It is unidentifiable -- the likelihood depends on the parameters only
through w_s * beta_s -- so a paired test against B0 has nothing to test. B1 avoids this by
acting on a 3-dimensional score space.

Every fit goes through a FoldScope, so R3 is enforced rather than asserted in a comment.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..assemble import b0_column_mask, concat_features
from ..contracts import SubjectTable
from ..cv.guards import FoldScope
from ..cv.splits import inner_folds

__all__ = ["fit_predict_b0", "fit_predict_b1", "fit_predict_b2", "STREAMS"]

STREAMS: tuple[str, ...] = ("b", "c", "e")


def _clean(X: np.ndarray) -> np.ndarray:
    """Infinities become NaN; NaN stays NaN until `_scaled` imputes it inside the fold.

    Unmeasurable features used to become 0 here, before scaling, which placed a child who
    barely spoke at an arbitrary and often extreme value (health check P12).
    """
    X = np.array(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return X


def _class_weight(y: np.ndarray) -> dict[int, float]:
    """Inverse prevalence, fixed in advance (decision_rules.yaml, plan doc 6.2)."""
    n = len(y)
    return {c: n / (2.0 * max(1, int((y == c).sum()))) for c in (0, 1)}


def _scaled(scope: FoldScope, X: np.ndarray, stage: str) -> np.ndarray:
    """Median imputation, then standardisation -- both fitted on training rows only."""
    imputer = scope.fit(SimpleImputer(strategy="median", keep_empty_features=True), X, None,
                        stage=f"impute:{stage}")
    X = imputer.transform(X)
    scaler = scope.fit(StandardScaler(), X, None, stage=f"scale:{stage}")
    return scaler.transform(X)


def _pick_C(scope: FoldScope, X: np.ndarray, y: np.ndarray, grid: Sequence[float],
            n_inner: int, stage: str) -> float:
    """Nested selection of the regularisation strength (plan doc 6.2)."""
    folds = inner_folds(scope.train_idx, y, n_inner, scope.seed)
    best_C, best_auc = grid[len(grid) // 2], -np.inf
    for C in grid:
        aucs = []
        for f in folds:
            inner = scope.inner(f)
            clf = LogisticRegression(C=C, penalty="l2", max_iter=2000,
                                     class_weight=_class_weight(y[f.train_idx]))
            scope.fit(clf, X, y, f.train_idx, stage=f"{stage}:C={C}")
            if len(np.unique(y[f.test_idx])) < 2:
                continue
            aucs.append(roc_auc_score(y[f.test_idx], clf.decision_function(X[f.test_idx])))
            _ = inner
        if aucs and float(np.mean(aucs)) > best_auc:
            best_auc, best_C = float(np.mean(aucs)), C
    return best_C


def fit_predict_b0(scope: FoldScope, table: SubjectTable, cfg: Mapping) -> dict:
    b0_cfg = cfg["cv"]["baselines"]["b0"]
    mask = b0_column_mask(table, cfg)
    X = _clean(concat_features(table))[:, mask]
    y = table.y
    Xz = _scaled(scope, X, "b0")

    if b0_cfg["estimator"] == "shrinkage_lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        scope.fit(clf, Xz, y, stage="b0:lda")
        all_scores = clf.decision_function(Xz)
        n_params = int(Xz.shape[1] + 1)
    else:
        C = _pick_C(scope, Xz, y, b0_cfg["C_grid"], cfg["cv"]["inner"]["n_folds"], "b0:inner")
        clf = LogisticRegression(C=C, penalty="l2", max_iter=5000,
                                 class_weight=_class_weight(y[scope.train_idx]))
        scope.fit(clf, Xz, y, stage="b0:lr")
        all_scores = clf.decision_function(Xz)
        n_params = int(Xz.shape[1] + 1)
    return {"scores": all_scores[scope.test_idx], "scores_all": all_scores,
            "n_params": n_params, "dim": int(mask.sum())}


def fit_predict_b1(scope: FoldScope, table: SubjectTable, cfg: Mapping) -> dict:
    """Stream-score stacking. Fusion is 3 + 1 parameters; the streams contribute ~40-70 more."""
    b1_cfg = cfg["cv"]["baselines"]["b1"]
    n_inner = cfg["cv"]["inner"]["n_folds"]
    y = table.y
    Xs = {s: _scaled(scope, _clean(table.stream(s)), f"b1:{s}") for s in STREAMS}

    # ---- level 0: out-of-fold scores for the TRAINING subjects, via an inner CV ----
    folds = inner_folds(scope.train_idx, y, n_inner, scope.seed)
    oof = {s: np.full(len(y), np.nan) for s in STREAMS}
    for f in folds:
        for s in STREAMS:
            clf = LogisticRegression(C=1.0, penalty="l2", max_iter=5000,
                                     class_weight=_class_weight(y[f.train_idx]))
            scope.fit(clf, Xs[s], y, f.train_idx, stage=f"b1:{s}:inner")
            oof[s][f.test_idx] = clf.decision_function(Xs[s][f.test_idx])

    # ---- level 0 refit on the full training fold, then score the test fold ----
    test_scores = {}
    stream_params = 0
    for s in STREAMS:
        clf = LogisticRegression(C=1.0, penalty="l2", max_iter=5000,
                                 class_weight=_class_weight(y[scope.train_idx]))
        scope.fit(clf, Xs[s], y, stage=f"b1:{s}:outer")
        test_scores[s] = clf.decision_function(Xs[s][scope.test_idx])
        stream_params += Xs[s].shape[1] + 1

    # ---- level 1: the identifiable 3+1 fusion, fitted on out-of-fold scores only ----
    F_train = np.column_stack([oof[s][scope.train_idx] for s in STREAMS])
    ok = np.isfinite(F_train).all(axis=1)
    fuse_idx = scope.train_idx[ok]
    F_all = np.zeros((len(y), len(STREAMS)))
    F_all[scope.train_idx] = np.nan_to_num(F_train)
    F_all[scope.test_idx] = np.column_stack([test_scores[s] for s in STREAMS])

    fusion = LogisticRegression(C=1.0 / float(b1_cfg["fusion_l2"]), penalty="l2",
                                max_iter=5000, class_weight=_class_weight(y[fuse_idx]))
    scope.fit(fusion, F_all, y, fuse_idx, stage="b1:fusion")
    all_scores = fusion.decision_function(F_all)

    w = fusion.coef_[0]
    sd = np.array([np.std(F_all[fuse_idx][:, i]) for i in range(len(STREAMS))])
    denom = float(np.sum(np.abs(w) * sd)) or 1.0
    contribution = {s: float(abs(w[i]) * sd[i] / denom) for i, s in enumerate(STREAMS)}

    return {
        "scores": all_scores[scope.test_idx],
        "scores_all": all_scores,
        "stream_scores": test_scores,
        "weights": {s: float(w[i]) for i, s in enumerate(STREAMS)},
        "contribution": contribution,
        "n_params": len(STREAMS) + 1,
        "n_params_including_streams": len(STREAMS) + 1 + stream_params,
    }


def fit_predict_b2(scope: FoldScope, table: SubjectTable, cfg: Mapping) -> dict:
    """External reference: eGeMAPS functionals + shrinkage LDA (Du et al. 2026 reached 0.85)."""
    X = _clean(table.X_egemaps)
    y = table.y
    Xz = _scaled(scope, X, "b2")
    if cfg["cv"]["baselines"]["b2"]["estimator"] == "svm":
        from sklearn.svm import SVC

        clf = SVC(kernel="linear", C=1.0, class_weight="balanced")
        scope.fit(clf, Xz, y, stage="b2:svm")
    else:
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        scope.fit(clf, Xz, y, stage="b2:lda")
    all_scores = clf.decision_function(Xz)
    return {"scores": all_scores[scope.test_idx], "scores_all": all_scores,
            "n_params": int(Xz.shape[1] + 1)}
