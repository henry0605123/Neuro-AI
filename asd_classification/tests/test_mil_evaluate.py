"""Repeated cross-validation, the logistic baseline, the interval and the permutation test: the
brief's seven tests, then the controller's rulings R19-R21.

They run on synthetic bags on the CPU with the frozen settings of ``configs/pipeline.yaml``
(``MILConfig.from_cfg`` and ``CVSettings.from_cfg``: 5 folds x 5 repeats x 5 seeds, 300 epochs,
2,000 bootstrap draws), with only ``device`` set to ``cpu``, so a passing test speaks about the
procedure the real run uses. The tests that need to see inside the cross-validation replace
``fit_predict`` with a recorder, which checks what is handed over without training anything.

The rulings: R19, probabilities are averaged (plan 3.5, steps 4-6), never logits; R20, the
interval is for the same statistic as the point estimate, the mean over repeats of the AUC;
R21 and R22, a batch that runs out of GPU memory is halved and retried, and the smaller size
is kept for the rest of the call; R23, permutation_p counts as a tie two AUCs that differ only
by rounding.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest
import torch
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asdpipe.cv.splits import subject_folds
from asdpipe.eval.metrics import safe_auc
from asdpipe.mil import evaluate as mil_evaluate
from asdpipe.mil.bags import synthetic_bags
from asdpipe.mil.evaluate import (CVSettings, bootstrap_ci, child_means, evaluate, label_sets,
                                  mean_auc, oof_logreg, oof_mil, permutation_p)
from asdpipe.mil.model import MILConfig, MILFit


@pytest.fixture(scope="module")
def mcfg(cfg) -> MILConfig:
    return MILConfig.from_cfg(cfg)


@pytest.fixture(scope="module")
def cv(cfg) -> CVSettings:
    """The frozen cross-validation settings, on the CPU."""
    return dataclasses.replace(CVSettings.from_cfg(cfg), device="cpu")


def _check_contract(r: dict, n: int, cv: CVSettings, n_perm: int, d: int) -> None:
    """The keys and shapes Task 7 reads from an ``evaluate`` result."""
    assert set(r) == {"mil", "logreg"}
    assert set(r["mil"]) == {"auc", "ci", "p", "null", "oof_mean", "gates_by_fold"}
    assert set(r["logreg"]) == {"auc", "ci", "p", "null", "oof_mean"}
    for part in r.values():
        lo, hi = part["ci"]
        assert 0.0 <= lo <= hi <= 1.0
        assert part["null"].shape == (n_perm,) and part["oof_mean"].shape == (n,)
        assert ((part["oof_mean"] >= 0.0) & (part["oof_mean"] <= 1.0)).all()   # probabilities
    assert r["mil"]["gates_by_fold"].shape == (cv.repeats, cv.n_folds, d)


# ---------------------------------------------------------------- the brief's seven tests
def test_null_auc_is_near_half(mcfg, cv):
    aucs = [evaluate(synthetic_bags(20, 19, signal=0.0, seed=k), mcfg, cv, 0, 0, 1.0)["mil"]["auc"]
            for k in range(6)]
    assert abs(np.mean(aucs) - 0.5) < 0.08


def test_signal_auc_is_high_for_both_models(mcfg, cv):
    b = synthetic_bags(20, 19, signal=1.5, seed=1)
    r = evaluate(b, mcfg, cv, 0, 0, 1.0)
    assert r["mil"]["auc"] > 0.8 and r["logreg"]["auc"] > 0.8
    _check_contract(r, len(b.ids), cv, 0, b.seg.shape[2] + b.child.shape[1])


def test_every_child_is_tested_once_per_repeat(monkeypatch, mcfg, cv):
    # R2/R3: built from Fold, asserted on the masks
    """Every model, as fit_predict receives it, against subject_folds(labels[p], n_folds,
    seed=r): its held-out children are that Fold's test children, its labels are label set p,
    its seed follows the formula, and per repeat and seed every child is held out exactly once.

    The recorder gives a model's held-out children the logit of a probability that names the
    model, its training children NaN, and every gate the model's seed. So the out-of-fold
    result shows which models scored a child: a training child's score reaching it would be
    NaN (R2/R3), and the value must be the mean of the names of the fold that held the child
    out. Batches of 7 models, not a multiple of the 5 seeds, make the seeds of one fold
    straddle two batches.
    """
    b = synthetic_bags(20, 19, signal=0.0, seed=4)
    labels = label_sets(b.y, 2, seed=0)                      # the truth and two shuffles
    perm_seed = 20261101
    calls = []

    def name_of(seeds: np.ndarray) -> np.ndarray:            # a probability, one per model
        return (seeds - perm_seed + 1) / 1e6

    def record(bags, train_mask, y, seeds, cfg, device="cpu"):
        seeds = np.array(seeds, dtype=np.int64)
        calls.append((np.array(train_mask), np.array(y), seeds))
        name = name_of(seeds)[:, None]
        n, t, d_seg = bags.seg.shape
        return MILFit(scores=np.where(train_mask, np.nan, np.log(name / (1 - name))),
                      gates=np.repeat(seeds[:, None].astype(np.float64),
                                      d_seg + bags.child.shape[1], axis=1),
                      attention=np.zeros((len(seeds), n, t)))

    monkeypatch.setattr(mil_evaluate, "fit_predict", record)
    oof, gates = oof_mil(b, labels, mcfg, dataclasses.replace(cv, chunk_models=7),
                         perm_seed=perm_seed)

    assert max(len(seeds) for _, _, seeds in calls) == 7
    train = np.concatenate([mask for mask, _, _ in calls])
    y = np.concatenate([lab for _, lab, _ in calls])
    seeds = np.concatenate([s for _, _, s in calls])
    n_p, n_r, n_k, n_s = len(labels), cv.repeats, cv.n_folds, cv.seeds
    assert len(seeds) == n_p * n_r * n_k * n_s == len(set(seeds.tolist()))
    local = seeds - perm_seed
    p, r, k, s = local // 100_000, local % 100_000 // 1_000, local % 1_000 // 10, local % 10

    expected_oof = np.full_like(oof, np.nan)
    expected_gates = np.full_like(gates, np.nan)
    for pi in range(n_p):
        for ri in range(n_r):
            for fold in subject_folds(labels[pi], n_k, seed=ri):
                models = np.flatnonzero((p == pi) & (r == ri) & (k == fold.fold_id))
                assert sorted(s[models].tolist()) == list(range(n_s))
                for m in models:
                    assert np.array_equal(np.flatnonzero(~train[m]), np.sort(fold.test_idx))
                    assert np.array_equal(y[m], labels[pi])
                expected_oof[pi, ri, fold.test_idx] = name_of(seeds[models]).mean()
                expected_gates[pi, ri, fold.fold_id] = seeds[models].mean()
            for si in range(n_s):
                held_out = ~train[(p == pi) & (r == ri) & (s == si)]
                assert (held_out.sum(axis=0) == 1).all()
    assert oof.shape == (n_p, n_r, len(b.ids))
    np.testing.assert_allclose(oof, expected_oof, rtol=0, atol=1e-12)
    np.testing.assert_allclose(gates, expected_gates, rtol=0, atol=1e-6)


def test_chunk_size_does_not_change_results(mcfg, cv):
    b = synthetic_bags(20, 19, signal=1.0, seed=3)
    labels = label_sets(b.y, 1, seed=0)                      # 2 label sets x 125 models
    a = oof_mil(b, labels, mcfg, dataclasses.replace(cv, chunk_models=25))[0]
    c = oof_mil(b, labels, mcfg, dataclasses.replace(cv, chunk_models=125))[0]
    np.testing.assert_allclose(a, c, atol=1e-5)


def test_permutation_p():
    assert permutation_p(0.9, np.full(99, 0.5)) == pytest.approx(0.01)
    assert permutation_p(0.5, np.full(99, 0.5)) == pytest.approx(1.0)


def test_strong_signal_gives_smallest_p(mcfg, cv):
    b = synthetic_bags(20, 19, signal=2.5, seed=2)
    r = evaluate(b, mcfg, cv, 19, 0, 1.0)
    assert r["mil"]["p"] == pytest.approx(1 / 20)
    assert r["logreg"]["p"] == pytest.approx(1 / 20)      # its null is rerun on the shuffles too
    _check_contract(r, len(b.ids), cv, 19, b.seg.shape[2] + b.child.shape[1])


def test_bootstrap_ci_brackets_point():
    y = np.repeat([1, 0], [20, 19])
    s = y + np.random.default_rng(0).standard_normal(len(y))
    lo, hi = bootstrap_ci(y, s, 2000, 0)
    assert lo <= safe_auc(y, s) <= hi
    assert 0.0 <= lo < hi <= 1.0


# ---------------------------------------------------------------- the controller's rulings
def test_seeds_and_repeats_average_probabilities(monkeypatch, mcfg, cv):
    """R19 and R20 through oof_mil and evaluate: probabilities are averaged, over a fold's
    seeds and then over the repeats, and the interval is for the mean over repeats of the AUC.

    The recorder hands out logits whose mean ranks the children differently from their mean
    probability. An ASD child gets 10 and -2 from its fold's two seeds, in both repeats: mean
    probability 0.560, mean logit 4. A TD child gets 3 from both seeds in repeat 0 and -1 in
    repeat 1: probability 0.953, then 0.269. As probabilities, repeat 0 ranks every TD child
    above every ASD child (AUC 0) and repeat 1 the reverse (AUC 1), so ``auc`` is 0.5, and as
    every draw of children keeps that, the interval is exactly (0.5, 0.5). Mean logits would
    give AUC 1 in both repeats; an interval for the AUC of ``oof_mean`` (ASD 0.560, TD 0.611)
    would be (0, 0).
    """
    b = synthetic_bags(20, 19, signal=0.0, seed=6)
    small = dataclasses.replace(cv, repeats=2, seeds=2)

    def record(bags, train_mask, y, seeds, cfg, device="cpu"):
        seeds = np.asarray(seeds)[:, None]                   # perm_seed 0: r and s read off
        r, s = seeds // 1_000 % 100, seeds % 10
        logit = np.where(np.asarray(y) == 1, np.where(s == 0, 10.0, -2.0),
                         np.where(r == 0, 3.0, -1.0))
        n, t, d_seg = bags.seg.shape
        return MILFit(scores=logit, gates=np.zeros((len(seeds), d_seg + bags.child.shape[1])),
                      attention=np.zeros((len(seeds), n, t)))

    monkeypatch.setattr(mil_evaluate, "fit_predict", record)
    oof, _ = oof_mil(b, label_sets(b.y, 0, seed=0), mcfg, small)
    asd = (expit(10.0) + expit(-2.0)) / 2
    want = np.stack([np.where(b.y == 1, asd, expit(3.0)), np.where(b.y == 1, asd, expit(-1.0))])
    np.testing.assert_allclose(oof[0], want, rtol=0, atol=1e-12)

    r = evaluate(b, mcfg, small, 0, 0, 1.0)
    np.testing.assert_allclose(r["mil"]["oof_mean"], want.mean(axis=0), rtol=0, atol=1e-12)
    assert r["mil"]["auc"] == pytest.approx(0.5)
    assert r["mil"]["ci"] == pytest.approx((0.5, 0.5))


def test_permutation_p_counts_ties_that_differ_only_by_rounding():
    """R23: mean_auc can put one exact mean AUC on two neighbouring floats (here 557/950, the
    reviewer's case); that is still a tie, and a tie counts against the truth. A real
    difference, one step of the grid a mean AUC lies on (1/3,800 for 20 ASD, 19 TD and 5
    repeats), still is one."""
    assert permutation_p(0.5863157894736843, [0.5863157894736842]) == 1.0
    assert permutation_p(0.5 + 1 / 3800, [0.5]) == 0.5


def test_logreg_scores_are_held_out_probabilities(cv):
    """R19 for the baseline, and its folds: each fold's pipeline is fitted on its training
    children's child_means only (R3) and gives the held-out children predict_proba, on the
    folds subject_folds(labels[p], n_folds, seed=r) that the segment model gets too."""
    b = synthetic_bags(20, 19, signal=1.0, seed=7)
    labels = label_sets(b.y, 1, seed=3)
    x = child_means(b)
    want = np.full((len(labels), cv.repeats, len(b.ids)), np.nan)
    for p, y in enumerate(labels):
        for r in range(cv.repeats):
            for fold in subject_folds(y, cv.n_folds, seed=r):
                clf = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                    LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
                clf.fit(x[fold.train_idx], y[fold.train_idx])
                want[p, r, fold.test_idx] = clf.predict_proba(x[fold.test_idx])[:, 1]
    np.testing.assert_array_equal(oof_logreg(b, labels, cv, 1.0), want)


def test_bootstrap_ci_of_repeats_brackets_mean_auc():
    """R20: with a (repeats, N) score every draw averages each repeat's AUC on the drawn
    children, so the interval is for mean_auc, the reported point estimate. Repeats that are
    all alike give the interval of the single score."""
    y = np.repeat([1, 0], [20, 19])
    s = y + np.random.default_rng(1).standard_normal((5, len(y)))
    lo, hi = bootstrap_ci(y, s, 2000, 0)
    assert lo <= mean_auc(y, s) <= hi
    assert 0.0 <= lo < hi <= 1.0
    assert bootstrap_ci(y, np.stack([s[0]] * 5), 2000, 0) == pytest.approx(
        bootstrap_ci(y, s[0], 2000, 0), abs=1e-12)


def test_out_of_memory_halves_the_batch_for_the_rest_of_the_call(monkeypatch, caplog, mcfg, cv):
    """R21 and R22: a batch that runs out of CUDA memory is halved and retried, down to single
    models, and the smaller size is kept for the rest of the call, each new size logged once;
    every number is still the unsplit run's. The error is faked on the CPU: any batch of more
    than two models raises it. With 10 models in batches of 4, the first batch fails at 4 and
    runs at 2, and the second and third batches start at 2: one failure in the whole call, not
    one per batch. A single model that does not fit raises to the caller."""
    b = synthetic_bags(20, 19, signal=1.0, seed=5)
    labels = label_sets(b.y, 0, seed=0)
    quick = dataclasses.replace(mcfg, epochs=20)
    small = dataclasses.replace(cv, repeats=1, seeds=2, chunk_models=10)    # 10 models
    whole = oof_mil(b, labels, quick, small)

    real, attempts = mil_evaluate.fit_predict, []

    def tight(bags, train_mask, y, seeds, cfg, device="cpu"):
        attempts.append(list(seeds))
        if len(seeds) > 2:
            raise torch.cuda.OutOfMemoryError("fake: CUDA out of memory")
        return real(bags, train_mask, y, seeds, cfg, device=device)

    monkeypatch.setattr(mil_evaluate, "fit_predict", tight)
    with caplog.at_level(logging.WARNING, logger=mil_evaluate.__name__):
        split = oof_mil(b, labels, quick, dataclasses.replace(small, chunk_models=4))
    for got, want in zip(split, whole):                      # probabilities, then gates
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    # model i has seed 10 * (i // 2) + i % 2 (perm_seed 0, one repeat, two seeds a fold);
    # models 4-7 are the second batch of 4 and 8-9 the third
    assert attempts == [[0, 1, 10, 11], [0, 1], [10, 11], [20, 21], [30, 31], [40, 41]]
    logged = [rec.getMessage() for rec in caplog.records if rec.name == mil_evaluate.__name__]
    assert len(logged) == 1 and "2 at a time" in logged[0]

    sizes = []

    def never(bags, train_mask, y, seeds, cfg, device="cpu"):
        sizes.append(len(seeds))
        raise torch.cuda.OutOfMemoryError("fake: CUDA out of memory")

    monkeypatch.setattr(mil_evaluate, "fit_predict", never)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        oof_mil(b, labels, quick, small)
    assert sizes == [10, 5, 2, 1]
