"""Repeated cross-validation of the segment-attention model and of the logistic baseline, the
95% interval, and the permutation test (research plan 3.5, steps 1-8).

For one label set the procedure is the plan's: split the children into ``n_folds`` folds
stratified on those labels (step 1); train ``seeds`` models per fold on the other folds'
children, which alone decide the imputation and the standardisation (steps 2-4); give every
child the mean probability of the models that never saw it (step 5); and do all of it again on
``repeats`` different splits (step 6). The AUC is the mean over the repeats of the AUC of each
repeat's out-of-fold probabilities.

**Probabilities, not logits** (plan 3.5, steps 4-6). A model's logit becomes a probability (the
sigmoid) before anything is averaged, over the seeds of a fold and then over the repeats. The
two orders rank children differently: in a mean of logits one confident seed can move a child
without bound, while it moves a mean of five probabilities by at most a fifth. The logistic
baseline's scores are its probabilities as well.

**The whole procedure under shuffled labels.** The permutation test (step 7) asks how often a
study of this design, with labels that mean nothing, scores at least as high as the real one.
So a shuffled label set goes through every step the truth goes through: its folds are drawn
again, stratified on the shuffled labels; every model is trained again, on them; and its AUC
is computed against them. Re-using the truth's folds or models would test a smaller procedure
than the one reported. ``label_sets`` puts the truth in row 0 and the shuffles after it, and
every row is run the same way, the logistic baseline included.

**Batches.** All models of all label sets (1,001 x 5 repeats x 5 folds x 5 seeds, about
125,000 in the real run) are laid out once, in the order label set, repeat, fold, seed, and
handed to ``fit_predict`` ``chunk_models`` at a time. A model's seed is
``perm_seed + 100_000 * p + 1_000 * r + 10 * fold + s``: it says which model this is, not where
the model falls in a batch, and ``fit_predict`` makes a model's result independent of its
batch-mates. So ``chunk_models`` changes how long a run takes and how much memory it needs,
never a number it reports, and for the same reason a batch that runs out of GPU memory is
halved and retried instead of ending the run; the smaller size is then kept for the rest of
the call. ``CVSettings`` keeps each term of the seed below the next one, so no two models share
a seed.

**R2 and R3.** A fold's test children are left out of the training masks of that fold's models,
and a child's out-of-fold probability is read only from the models whose mask left it out.
Every child is held out exactly once per repeat, which is checked, not assumed. Imputation and
standardisation happen inside ``fit_predict`` and inside each fold's logistic pipeline, from
the training children only. ``child_means`` averages within one child, so it estimates nothing
across children.

**The 95% interval** (step 6) is a stratified bootstrap of the children for the statistic
reported as ``auc``: each draw redraws the ASD children and the TD children separately, with
replacement, so it keeps the group sizes and has an AUC; it computes every repeat's AUC on the
drawn children and averages them. The interval runs from the 2.5th to the 97.5th percentile of
those means. Both models use the same draws (seed ``perm_seed``).
"""

from __future__ import annotations

import logging
import operator
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..cv.splits import subject_folds
from ..eval.metrics import safe_auc
from .bags import Bags
from .model import MILConfig, fit_predict

__all__ = ["CVSettings", "label_sets", "oof_mil", "child_means", "oof_logreg", "mean_auc",
           "bootstrap_ci", "permutation_p", "evaluate"]

_log = logging.getLogger(__name__)

# model seed = perm_seed + _P_STEP * p + _R_STEP * r + _FOLD_STEP * fold + s. CVSettings keeps
# s < 10, fold < 100 and r < 100, so every term stays below the next step and seeds are unique.
_P_STEP, _R_STEP, _FOLD_STEP = 100_000, 1_000, 10
_CV_KEYS = ("n_folds", "repeats", "seeds")
_LOGREG_MAX_ITER = 2000            # as eval/analysis.py's examiner_probe

# How far apart two AUCs may be and still count as one value in permutation_p. A repeat's AUC
# is a multiple of 1 / (2 * n_ASD * n_TD) (a tie counts one half), so a mean over R repeats
# lies on a grid of step 1 / (2 * R * n_ASD * n_TD): 1/3,800 for 20 ASD, 19 TD and 5 repeats,
# 1/1,900 with 10 ASD. roc_auc_score and the mean can put one exact value on neighbouring
# floats, a few 1e-16 apart, and a tie lost that way would only ever make p smaller. 1e-9 lies
# far between the two scales: closer than that is the same grid point.
TIE_TOL = 1e-9


@dataclass(frozen=True)
class CVSettings:
    """How the cross-validation runs (plan 3.5; ``mil`` in ``configs/pipeline.yaml``).

    ``n_folds``, ``repeats`` and ``seeds`` are the frozen ``mil.cv`` settings, and ``n_boot``
    the number of bootstrap draws behind the 95% interval (``mil.n_boot``). ``chunk_models``
    (models per ``fit_predict`` call, fewer after running out of GPU memory; see ``oof_mil``)
    and ``device`` (``"cpu"``, ``"cuda"`` or ``"auto"``) are execution settings: they change
    how long a run takes, never a number it reports. The defaults are the frozen values, on
    the CPU.
    """

    n_folds: int = 5
    repeats: int = 5
    seeds: int = 5
    n_boot: int = 2000
    chunk_models: int = 250
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("n_folds", "repeats", "seeds", "n_boot", "chunk_models"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        # the bounds the seed formula needs (see _P_STEP); StratifiedKFold needs 2 folds
        if not (2 <= self.n_folds < 100 and 1 <= self.repeats < 100 and 1 <= self.seeds < 10):
            raise ValueError(f"need 2 <= n_folds < 100, 1 <= repeats < 100 and 1 <= seeds < 10 "
                             f"(the model seed formula); got {self.n_folds}, {self.repeats} and "
                             f"{self.seeds}")
        if self.n_boot < 1 or self.chunk_models < 1:
            raise ValueError(f"n_boot and chunk_models must be at least 1, got {self.n_boot} "
                             f"and {self.chunk_models}")

    @classmethod
    def from_cfg(cls, cfg: Mapping) -> "CVSettings":
        """Read ``cfg["mil"]``: ``cv`` must hold exactly n_folds, repeats and seeds, and
        ``n_boot``, ``chunk_models`` and ``device`` must be there. A misspelt key would
        otherwise leave a default in force without anyone noticing."""
        mil = cfg["mil"]
        given = mil["cv"]
        unknown, missing = sorted(set(given) - set(_CV_KEYS)), sorted(set(_CV_KEYS) - set(given))
        if unknown or missing:
            raise ValueError(f"mil.cv: unknown key(s) {unknown}, missing key(s) {missing}")
        absent = sorted({"n_boot", "chunk_models", "device"} - set(mil))
        if absent:
            raise ValueError(f"mil: missing key(s) {absent}")
        return cls(**{name: operator.index(given[name]) for name in _CV_KEYS},
                   n_boot=operator.index(mil["n_boot"]),
                   chunk_models=operator.index(mil["chunk_models"]), device=str(mil["device"]))


def _label_rows(labels: np.ndarray, n_children: int) -> np.ndarray:
    lab = np.asarray(labels)
    if (lab.ndim != 2 or lab.shape[0] == 0 or lab.shape[1] != n_children
            or not np.isin(lab, (0, 1)).all()):
        raise ValueError(f"labels must be a (label sets, {n_children}) array of 0 (TD) and "
                         f"1 (ASD) with at least one row; got shape {lab.shape}")
    return lab.astype(np.int64)


def label_sets(y: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """The true labels in row 0 and ``n_perm`` shuffles of them after it, ``(1 + n_perm, N)``.

    A shuffle keeps the numbers of ASD and TD children, so every row describes a study of the
    same design. One generator draws the shuffles in order, so the first k shuffles are the
    same whatever ``n_perm`` is.
    """
    y = _label_rows(np.asarray(y)[None], len(y))[0]
    n_perm = operator.index(n_perm)
    if n_perm < 0:
        raise ValueError(f"n_perm cannot be negative, got {n_perm}")
    rng = np.random.default_rng(seed)
    out = np.empty((1 + n_perm, len(y)), dtype=np.int64)
    out[0] = y
    for row in out[1:]:
        row[:] = rng.permutation(y)
    return out


def _held_out(y: np.ndarray, cv: CVSettings, repeat: int) -> np.ndarray:
    """One label set's folds in repeat ``repeat``, ``(n_folds, N)``: True on a fold's test
    children.

    The folds are ``subject_folds(y, n_folds, seed=repeat)``, stratified on these labels, and
    both models take them from here, so they share every split. A fold's models train on its
    ``train_idx`` and every other child is held out of them; every child must be held out of
    exactly one fold, or a repeat's out-of-fold scores would be incomplete or doubled.
    """
    folds = subject_folds(y, cv.n_folds, seed=repeat)
    held = np.ones((len(folds), len(y)), dtype=bool)
    for k, fold in enumerate(folds):
        held[k, fold.train_idx] = False
    if len(folds) != cv.n_folds or not (held.sum(axis=0) == 1).all():
        raise RuntimeError(f"repeat {repeat}: the folds do not hold out every child exactly once")
    return held


def oof_mil(bags: Bags, labels: np.ndarray, mcfg: MILConfig, cv: CVSettings, *,
            perm_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The segment-attention model's out-of-fold probabilities ``(P, repeats, N)`` and gates
    ``(P, repeats, n_folds, Ds + Dc)``, for every label set (row) of ``labels`` ``(P, N)``.

    A child's probability in a repeat is the mean over the ``seeds`` models of the fold that
    held it out of each model's probability (the sigmoid of its logit), and a fold's gates are
    the mean of its models' gates (segment measures first, in the Bags' column order). Model
    ``s`` of fold ``k`` in repeat ``r`` of label set ``p`` has seed
    ``perm_seed + 100_000 * p + 1_000 * r + 10 * k + s``, whatever the batch.

    The models train ``cv.chunk_models`` at a time on ``cv.device``, in order. A batch that
    runs out of CUDA memory is halved and retried, down to a single model, which raises if it
    still does not fit. The smaller size is kept for the rest of the call, and each new size is
    logged once: every model needs the same memory, so each of the real run's 501 batches of
    250 would otherwise fail once before running. Neither changes a number (module docstring).
    """
    n = len(bags.ids)
    labels = _label_rows(labels, n)
    perm_seed = operator.index(perm_seed)
    shape = (len(labels), cv.repeats, cv.n_folds, cv.seeds)
    held = np.stack([_held_out(y, cv, r) for y in labels for r in range(cv.repeats)])
    # one row per model, in the order label set, repeat, fold, seed (the last varies fastest)
    train = np.repeat(~held.reshape(-1, n), cv.seeds, axis=0)
    p, r, k, s = np.indices(shape).reshape(len(shape), -1)
    seeds = perm_seed + _P_STEP * p + _R_STEP * r + _FOLD_STEP * k + s

    oof = np.zeros((len(labels), cv.repeats, n))
    gates = np.zeros(shape[:3] + (bags.seg.shape[2] + bags.child.shape[1],))
    size, start = cv.chunk_models, 0
    while start < len(seeds):
        part = slice(start, min(start + size, len(seeds)))
        try:
            fit = fit_predict(bags, train[part], labels[p[part]], seeds[part].tolist(), mcfg,
                              device=cv.device)
        except torch.cuda.OutOfMemoryError:
            if part.stop - part.start == 1:
                raise
            fit = None
        if fit is None:
            # retried only once the except clause is over: until then the traceback keeps the
            # failed attempt's frames, and with them its tensors, alive
            size = (part.stop - part.start) // 2
            _log.warning("models %d-%d ran out of GPU memory; training %d at a time from here "
                         "on (the results do not change)", part.start, part.stop - 1, size)
            torch.cuda.empty_cache()                  # a no-op unless CUDA is in use
            continue
        # a model adds its probability only for the children it was not trained on (R2/R3)
        np.add.at(oof, (p[part], r[part]), np.where(train[part], 0.0, expit(fit.scores)))
        np.add.at(gates, (p[part], r[part], k[part]), fit.gates)
        start = part.stop
    return oof / cv.seeds, gates / cv.seeds


def child_means(bags: Bags) -> np.ndarray:
    """The logistic baseline's input (plan 3.5, step 8), ``(N, Ds + Dc)``: each child's mean of
    every segment measure over its real segments, skipping the ones it could not be measured
    in, followed by its child measures, in the Bags' column order.

    A measure missing from all of a child's segments stays NaN, for each fold's pipeline to
    impute from its training children (R3). Every mean is within one child.
    """
    seg = np.where(bags.mask[:, :, None], bags.seg.astype(np.float64), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # "Mean of empty slice"
        means = np.nanmean(seg, axis=1)
    return np.concatenate([means, bags.child.astype(np.float64)], axis=1)


def _logreg(C: float) -> Pipeline:
    """One fold's baseline. ``keep_empty_features``: a measure no training child has turns into
    a column of 0 instead of disappearing, so the columns stay the same in every fold."""
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=_LOGREG_MAX_ITER))


def oof_logreg(bags: Bags, labels: np.ndarray, cv: CVSettings, C: float) -> np.ndarray:
    """The logistic baseline's out-of-fold probabilities of ASD ``(P, repeats, N)`` for every
    label set of ``labels``, on the folds ``oof_mil`` uses.

    Each fold fits its own median imputer, scaler and L2 logistic regression (strength ``C``,
    classes weighted to balance) on its training children's ``child_means`` only (R3), and
    scores the held-out children with ``predict_proba``.
    """
    n = len(bags.ids)
    labels = _label_rows(labels, n)
    if not C > 0:
        raise ValueError(f"C must be positive, got {C}")
    x = child_means(bags)
    oof = np.full((len(labels), cv.repeats, n), np.nan)
    for p, y in enumerate(labels):
        for r in range(cv.repeats):
            for test in _held_out(y, cv, r):
                fitted = _logreg(C).fit(x[~test], y[~test])
                oof[p, r, test] = fitted.predict_proba(x[test])[:, 1]     # classes_ is [0, 1]
    return oof


def mean_auc(y: np.ndarray, oof: np.ndarray) -> float:
    """The mean over repeats of the AUC of each repeat's out-of-fold scores, ``oof`` being
    ``(repeats, N)`` (plan 3.5, step 6)."""
    y = np.asarray(y)
    oof = np.asarray(oof, dtype=np.float64)
    if oof.ndim != 2 or oof.shape[1] != len(y):
        raise ValueError(f"oof must be (repeats, {len(y)}); got shape {oof.shape}")
    return float(np.mean([safe_auc(y, row) for row in oof]))


def bootstrap_ci(y: np.ndarray, score: np.ndarray, n_boot: int,
                 seed: int) -> tuple[float, float]:
    """The 95% interval of an AUC from ``n_boot`` bootstrap draws stratified on ``y`` (plan
    3.5, step 6): the 2.5th and 97.5th percentiles.

    ``score`` is one score per child, ``(N,)``, for the AUC of that score; or one per repeat
    and child, ``(repeats, N)``, for ``mean_auc``: every draw then computes each repeat's AUC on
    the drawn children and averages them. A draw takes as many ASD children as there are, with
    replacement, from the ASD children, and as many TD children from the TD children, so it
    keeps the group sizes and has an AUC.

    A repeat's AUC on a draw is the share of drawn (ASD, TD) pairs, counted with multiplicity,
    in which the ASD child scores higher, a tie counting one half: what ``safe_auc`` returns on
    the drawn children. That share is linear in how often each child was drawn, so its mean
    over the repeats is the same share taken from the repeats' mean comparison matrix, and no
    draw is ever built child by child.
    """
    y = np.asarray(y)
    s = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or s.ndim not in (1, 2) or s.shape[-1] != len(y) or s.size == 0:
        raise ValueError(f"score must be (N,) or (repeats, N) with N = len(y) = {len(y)} and at "
                         f"least one repeat; got y {y.shape} and score {s.shape}")
    if not (np.isin(y, (0, 1)).all() and (y == 1).any() and (y == 0).any()):
        raise ValueError("y must be 0 (TD) and 1 (ASD), with at least one child of each")
    if not np.isfinite(s).all():
        raise ValueError("every score must be finite")
    n_boot = operator.index(n_boot)
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    rows = np.atleast_2d(s)                                     # a 1-D score is one repeat
    pos, neg = rows[:, y == 1], rows[:, y == 0]
    above = pos[:, :, None] > neg[:, None, :]
    tied = pos[:, :, None] == neg[:, None, :]
    wins = (above + 0.5 * tied).mean(axis=0)                    # (n_ASD, n_TD)
    n_pos, n_neg = wins.shape
    rng = np.random.default_rng(seed)
    times_pos = rng.multinomial(n_pos, np.full(n_pos, 1.0 / n_pos), size=n_boot)
    times_neg = rng.multinomial(n_neg, np.full(n_neg, 1.0 / n_neg), size=n_boot)
    aucs = ((times_pos @ wins) * times_neg).sum(axis=1) / (n_pos * n_neg)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_p(observed: float, null: np.ndarray) -> float:
    """``(1 + #{null >= observed - TIE_TOL}) / (1 + len(null))`` (plan 3.5, step 7).

    The share of label sets, the true one counted among them, whose AUC is at least the true
    one's: never below ``1 / (1 + len(null))``, and a tie counts against the truth, including
    one that rounding has put a float below it (see ``TIE_TOL``). With no shuffle it is 1.
    """
    null = np.asarray(null, dtype=np.float64).reshape(-1)
    if not (np.isfinite(observed) and np.isfinite(null).all()):
        raise ValueError("permutation_p needs finite AUCs")
    return float((1 + np.count_nonzero(null >= observed - TIE_TOL)) / (1 + null.size))


def evaluate(bags: Bags, mcfg: MILConfig, cv: CVSettings, n_perm: int, perm_seed: int,
             C: float) -> dict:
    """The plan's procedure (3.5, steps 1-8) for both models, on the true labels ``bags.y`` and
    on ``n_perm`` shuffles of them (``label_sets(bags.y, n_perm, perm_seed)``).

    Returns ``{"mil": {...}, "logreg": {...}}``, each holding ``auc`` (``mean_auc`` of the
    out-of-fold probabilities on the true labels), ``ci`` (``bootstrap_ci`` of those
    ``(repeats, N)`` probabilities: an interval for that same mean AUC, ``cv.n_boot`` draws,
    seed ``perm_seed``), ``p`` (``permutation_p`` of ``auc`` against ``null``), ``null``
    (``(n_perm,)``, one mean AUC per shuffled label set) and ``oof_mean`` (``(N,)``, each
    child's out-of-fold probability on the true labels, averaged over seeds and repeats).
    ``mil`` also holds ``gates_by_fold``, ``(repeats, n_folds, Ds + Dc)``: every fold's gates on
    the true labels, averaged over seeds, the segment measures first. With ``n_perm = 0`` the
    null is empty and ``p`` is 1.
    """
    labels = label_sets(bags.y, n_perm, perm_seed)
    # the baseline first: it is quick next to the model, so a bad input fails before the long part
    scores = {"logreg": oof_logreg(bags, labels, cv, C)}
    scores["mil"], gates = oof_mil(bags, labels, mcfg, cv, perm_seed=perm_seed)
    out: dict[str, dict] = {}
    for name in ("mil", "logreg"):
        aucs = np.array([mean_auc(y, oof) for y, oof in zip(labels, scores[name])])
        truth = scores[name][0]                                 # (repeats, N)
        out[name] = {"auc": float(aucs[0]),
                     "ci": bootstrap_ci(labels[0], truth, cv.n_boot, perm_seed),
                     "p": permutation_p(aucs[0], aucs[1:]), "null": aucs[1:].copy(),
                     "oof_mean": truth.mean(axis=0)}
    out["mil"]["gates_by_fold"] = gates[0].copy()
    return out
