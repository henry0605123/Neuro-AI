"""Subject-level cross-validation splits (R2).

Every fold boundary is drawn over *subjects*. There is no segment expansion anywhere in
this repo -- `SubjectTable` physically cannot hold two rows for one subject -- so the
classic "chunks of the same child land in train and test" failure is impossible by
construction rather than by convention.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..contracts import Fold

__all__ = ["subject_folds", "inner_folds", "stratification_key"]


def stratification_key(y: np.ndarray, extra: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Combine the label with any additional stratifiers (age band / sex on real data)."""
    key = y.astype(str)
    for arr in extra:
        key = np.char.add(np.char.add(key, "|"), np.asarray(arr).astype(str))
    return key


def subject_folds(y: np.ndarray, n_folds: int, seed: int,
                  extra_stratifiers: Sequence[np.ndarray] = ()) -> list[Fold]:
    n = len(y)
    if n_folds > n:
        raise ValueError(f"{n_folds} folds requested for {n} subjects")
    key = stratification_key(y, extra_stratifiers)
    # fall back to the plain label if a combined key leaves a class with fewer members
    # than folds -- stratifying into empty strata is worse than stratifying coarsely
    _, counts = np.unique(key, return_counts=True)
    if counts.min() < n_folds:
        key = y
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        Fold(fold_id=i, train_idx=np.asarray(tr), test_idx=np.asarray(te))
        for i, (tr, te) in enumerate(skf.split(np.zeros(n), key))
    ]


def inner_folds(train_idx: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> list[Fold]:
    """Inner validation sub-folds, used for early stopping and hyper-parameters.

    These indices are positions into the **full** table, but every one of them is drawn
    from ``train_idx`` -- the outer test fold is never visible (plan doc 6.2, audit P1-11).
    """
    y_tr = y[train_idx]
    k = min(n_folds, int(np.min(np.bincount(y_tr, minlength=2))))
    if k < 2:
        mid = max(1, len(train_idx) // 2)
        return [Fold(0, train_idx[:mid], train_idx[mid:])]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + 977)
    return [
        Fold(fold_id=i, train_idx=train_idx[tr], test_idx=train_idx[te])
        for i, (tr, te) in enumerate(skf.split(np.zeros(len(train_idx)), y_tr))
    ]
