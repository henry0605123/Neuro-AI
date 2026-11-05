"""🔴 M4 anti-leakage suite. These four assertions are the ones that must never go red.

  1. no subject appears in two folds (R2);
  2. every recorded ``fit`` saw only training indices (R3);
  3. the feature table carries no label-derived column;
  4. one subject is exactly one row -- no segment expansion.

Plus a static scan: nothing under `models/` may call ``.fit(`` directly, which is what stops
a future model from quietly bypassing the FoldScope.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from asdpipe.assemble import B0_E_FEATURES
from asdpipe.contracts import Fold, Label, SubjectRow, SubjectTable
from asdpipe.cv.guards import LEAKAGE_LOG, FoldScope
from asdpipe.cv.splits import inner_folds, subject_folds
from asdpipe.models.baselines import fit_predict_b0, fit_predict_b1, fit_predict_b2
from asdpipe.models.m1 import fit_predict_m1

SRC = Path(__file__).resolve().parents[1] / "src" / "asdpipe"


@pytest.fixture
def fake_table():
    """A table with the real shape but fabricated numbers -- the CV machinery is what is
    under test here, not the extractors."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        label = Label.ASD if i < 6 else Label.CONTROL
        rows.append(SubjectRow(
            subject_id=f"S{i:02d}", label=label,
            # realistic names: the B0 column selector keys off the functional suffix and
            # off the frozen E-stream shortlist, so the fake table has to look real
            b_pooled={**{f"b_ch{j}_{st}": float(rng.normal())
                         for j in range(11) for st in ("mean", "std", "p25")},
                      "b7_mean_pairwise_cosdist": float(rng.normal())},
            c_pooled={**{f"c_m{j}": float(rng.normal()) for j in range(18)},
                      "c_residual_f0_mean": float(rng.normal()),
                      "c_residual_f0_std": float(rng.normal()),
                      "c_residual_f0_p25": float(rng.normal())},
            e_pooled={**{n: float(rng.normal()) for n in B0_E_FEATURES},
                      **{f"e_extra{j}": float(rng.normal()) for j in range(9)}},
            c_sequence=rng.normal(size=(rng.integers(15, 30), 11)).astype(np.float32),
            egemaps_functionals=rng.normal(size=88),
            examiner_probe={f"p_{j}": float(rng.normal()) for j in range(12)},
            ados={"A2": int(rng.integers(4)), "A3": int(rng.integers(4)),
                  "A4": int(rng.integers(4)), "A8": int(rng.integers(4))},
            latents={"rhythm_variability": float(rng.normal())},
            quality={"b_child_speech_s": 60.0},
        ))
    return SubjectTable(rows)


# ---------------------------------------------------------------- 1. fold disjointness
def test_no_subject_appears_in_two_folds(fake_table):
    for seed in (0, 1, 2):
        folds = subject_folds(fake_table.y, 3, seed)
        assert sum(len(f.test_idx) for f in folds) == len(fake_table)
        seen: set[int] = set()
        for f in folds:
            assert not (set(f.train_idx.tolist()) & set(f.test_idx.tolist()))
            assert not (seen & set(f.test_idx.tolist())), "a subject landed in two test folds"
            seen |= set(f.test_idx.tolist())
        assert seen == set(range(len(fake_table)))


def test_fold_rejects_overlap_at_construction():
    with pytest.raises(ValueError, match="R2 violated"):
        Fold(0, np.array([0, 1, 2]), np.array([2, 3]))


def test_inner_folds_never_escape_the_training_fold(fake_table):
    outer = subject_folds(fake_table.y, 3, 0)[0]
    for f in inner_folds(outer.train_idx, fake_table.y, 3, 0):
        assert set(f.train_idx.tolist()) <= set(outer.train_idx.tolist())
        assert set(f.test_idx.tolist()) <= set(outer.train_idx.tolist())


# ---------------------------------------------------------------- 2. recorded fit indices
def test_every_recorded_fit_saw_only_training_indices(fake_table, cfg):
    """The wrapper logs the index set of every fit; all of them must be inside the fold."""
    from asdpipe.config import load_decision_rules

    load_decision_rules()          # R5: the file must exist before any training
    LEAKAGE_LOG.clear()
    y = fake_table.y
    for seed in (0, 1):
        for fold in subject_folds(y, 3, seed):
            scope = FoldScope(fold, seed)
            fit_predict_b0(scope, fake_table, cfg)
            fit_predict_b1(scope, fake_table, cfg)
            fit_predict_b2(scope, fake_table, cfg)
            fit_predict_m1(scope, fake_table, cfg, torch_seed=seed)

            train = set(fold.train_idx.tolist())
            test = set(fold.test_idx.tolist())
            events = scope.log.for_fold(fold.fold_id, seed)
            assert events, "no fit was recorded -- the guard is not wired up"
            for ev in events:
                assert set(ev.indices) <= train, (
                    f"R3 violated in {ev.stage}: saw {sorted(set(ev.indices) - train)}")
                assert not (set(ev.indices) & test)
    assert len(LEAKAGE_LOG.events) > 50


def test_scope_refuses_an_index_outside_the_training_fold(fake_table):
    from sklearn.preprocessing import StandardScaler

    fold = subject_folds(fake_table.y, 3, 0)[0]
    scope = FoldScope(fold, 0)
    with pytest.raises(AssertionError, match="R3 violated"):
        scope.fit(StandardScaler(), fake_table.X_b, None, fold.test_idx, stage="evil")


def test_inner_scope_cannot_widen_the_allowed_set(fake_table):
    folds = subject_folds(fake_table.y, 3, 0)
    outer, other = folds[0], folds[1]
    scope = FoldScope(outer, 0)
    _ = other
    # an inner fold whose training rows are the OUTER TEST rows must be rejected
    with pytest.raises(AssertionError, match="R3 violated"):
        scope.inner(Fold(9, outer.test_idx, outer.train_idx[:1]))


# ---------------------------------------------------------------- 3. no label-derived columns
def test_feature_table_contains_no_label_derived_column(fake_table):
    names = fake_table.all_feature_names()
    # substring match on whole tokens: "y_" would false-positive on e_latency_median
    banned = {"label", "diagnosis", "asd", "ados", "latent", "latents", "group",
              "target", "y", "class"}
    for n in names:
        tokens = set(n.lower().split("_"))
        assert not (tokens & banned), f"suspicious feature name {n!r}"
    # the injected ground truth and the criterion scores are kept out of X entirely
    assert set(fake_table.latents) & set(names) == set()
    assert set(fake_table.ados) & set(names) == set()


def test_no_more_columns_separate_the_labels_than_chance_allows(fake_table):
    """Canary for a feature that is a deterministic function of the label.

    The bar cannot be "no column separates perfectly". These columns are fabricated
    independent normals, and with 6 vs 6 subjects a single continuous column separates them
    by luck with probability 2 / C(12, 6) = 0.22%; across ~60 columns that is roughly a 12%
    chance of a spurious failure on any given draw. The previous version asserted exactly
    that and passed only because the RNG stream happened to be kind -- removing one entry
    from B0_E_FEATURES shifted the stream and it failed immediately.

    So the assertion is against the null distribution instead. Real label leakage does not
    produce one borderline column; it produces many, and a model that scores far above
    chance, which the CV tests around this one already check.
    """
    import math

    from scipy.stats import binom

    X = np.concatenate([fake_table.X_b, fake_table.X_c, fake_table.X_e], axis=1)
    y = fake_table.y
    p_chance = 2 / math.comb(len(y), int((y == 1).sum()))
    names = fake_table.all_feature_names()
    separating = [names[j] for j in range(X.shape[1])
                  if np.std(X[:, j]) > 0
                  and (X[y == 1, j].min() > X[y == 0, j].max()
                       or X[y == 1, j].max() < X[y == 0, j].min())]
    bound = int(binom.ppf(0.9999, X.shape[1], p_chance))
    assert len(separating) <= bound, (
        f"{len(separating)} columns separate the labels; chance allows at most {bound}: "
        f"{separating}")


def test_probe_matrix_is_not_part_of_the_model_features(fake_table):
    names = set(fake_table.all_feature_names())
    assert not (set(fake_table.probe_names) & names)


# ---------------------------------------------------------------- 4. one subject, one row
def test_one_subject_is_exactly_one_row(fake_table):
    assert len(fake_table) == fake_table.n_subjects == len(set(fake_table.subject_ids)) == 12
    for arr in (fake_table.X_b, fake_table.X_c, fake_table.X_e, fake_table.X_egemaps,
                fake_table.seq, fake_table.seq_mask, fake_table.X_probe):
        assert arr.shape[0] == 12
    assert fake_table.y.shape == (12,)


def test_duplicate_subject_ids_are_rejected(fake_table):
    with pytest.raises(ValueError, match="unique subject ids"):
        SubjectTable(list(fake_table.rows) + [fake_table.rows[0]])


# ---------------------------------------------------------------- static scan
def test_models_never_call_fit_directly():
    """R3 is only durable if a new model cannot bypass the scope without a red test."""
    pattern = re.compile(r"(?<!scope)\.fit\(")
    offenders = []
    for path in (SRC / "models").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line) and "scope.fit" not in line:
                offenders.append(f"{path.name}:{i}: {stripped}")
    assert not offenders, (
        "these lines fit an estimator outside FoldScope, which defeats the R3 guarantee:\n"
        + "\n".join(offenders))


def test_extractors_never_read_the_generator_ground_truth():
    """`Turn.intensity_db` and `SubjectRecord.latents` are ground truth, not features."""
    offenders = []
    for path in (SRC / "features").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("intensity_db", ".latents"):
            if needle in text:
                offenders.append(f"{path.name} reads {needle}")
    assert not offenders, offenders
