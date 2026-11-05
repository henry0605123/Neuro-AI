"""R3 as executable code, not a comment.

Every quantity estimated from data -- standardisation, PCA, regularisation strength,
probability calibration, stacking weights -- must be fitted on training indices only. This
module makes that mechanically checkable:

* nothing in `models/` calls ``estimator.fit`` directly; everything goes through
  ``FoldScope.fit``;
* ``FoldScope.fit`` refuses index sets that are not a subset of what the scope allows, and
  records every call in a global log;
* ``tests/test_leakage.py`` then asserts (a) the recorded index sets are all inside the
  training fold, and (b) a static scan finds no direct ``.fit(`` call outside this module.

The second check is what stops the guarantee from decaying: someone can add a new model,
but they cannot bypass the scope without the test going red.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..contracts import Fold

__all__ = ["FitEvent", "LeakageLog", "LEAKAGE_LOG", "FoldScope"]


@dataclass(frozen=True)
class FitEvent:
    estimator: str
    stage: str
    fold_id: int
    seed: int
    indices: frozenset[int]


@dataclass
class LeakageLog:
    events: list[FitEvent] = field(default_factory=list)

    def record(self, event: FitEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()

    def for_fold(self, fold_id: int, seed: int) -> list[FitEvent]:
        return [e for e in self.events if e.fold_id == fold_id and e.seed == seed]


LEAKAGE_LOG = LeakageLog()


class FoldScope:
    """A window onto the data that only ever exposes training rows to a ``fit`` call."""

    def __init__(self, fold: Fold, seed: int, allowed: np.ndarray | None = None,
                 log: LeakageLog | None = None, parent: "FoldScope | None" = None):
        self.fold = fold
        self.seed = seed
        self.allowed = np.asarray(fold.train_idx if allowed is None else allowed)
        self.log = log if log is not None else LEAKAGE_LOG
        self.parent = parent
        if parent is not None:
            extra = set(self.allowed.tolist()) - set(parent.allowed.tolist())
            if extra:
                raise AssertionError(
                    f"R3 violated: inner scope would see rows {sorted(extra)} that the "
                    "outer training fold does not contain"
                )

    # ------------------------------------------------------------------ index handling
    def check(self, idx: Iterable[int], stage: str = "") -> np.ndarray:
        idx = np.asarray(list(idx), dtype=int)
        illegal = set(idx.tolist()) - set(self.allowed.tolist())
        if illegal:
            raise AssertionError(
                f"R3 violated at stage {stage!r}: fit would see rows {sorted(illegal)}, "
                f"which are outside the training indices of fold {self.fold.fold_id}"
            )
        return idx

    @property
    def train_idx(self) -> np.ndarray:
        return self.allowed

    @property
    def test_idx(self) -> np.ndarray:
        return np.asarray(self.fold.test_idx)

    def inner(self, fold: Fold) -> "FoldScope":
        """Sub-scope for an inner validation fold (early stopping, lambda search)."""
        return FoldScope(fold, self.seed, allowed=fold.train_idx, log=self.log, parent=self)

    # ------------------------------------------------------------------ guarded fitting
    def fit(self, estimator: Any, X: np.ndarray, y: np.ndarray | None,
            idx: Sequence[int] | None = None, stage: str = "") -> Any:
        """Fit ``estimator`` on ``X[idx]``. ``idx`` defaults to this scope's training rows."""
        idx = self.check(self.allowed if idx is None else idx, stage or type(estimator).__name__)
        self.log.record(FitEvent(
            estimator=type(estimator).__name__,
            stage=stage or type(estimator).__name__,
            fold_id=self.fold.fold_id,
            seed=self.seed,
            indices=frozenset(int(i) for i in idx),
        ))
        if y is None:
            estimator.fit(X[idx])
        else:
            estimator.fit(X[idx], y[idx])
        return estimator

    def fit_callable(self, fn, idx: Sequence[int] | None = None, stage: str = "", **kwargs):
        """Same guarantee for a hand-written fit (the M1 torch loop, the B1 fusion layer)."""
        idx = self.check(self.allowed if idx is None else idx, stage)
        self.log.record(FitEvent(
            estimator=getattr(fn, "__name__", "callable"), stage=stage,
            fold_id=self.fold.fold_id, seed=self.seed,
            indices=frozenset(int(i) for i in idx),
        ))
        return fn(idx, **kwargs)
