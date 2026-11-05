"""Gated segment-attention model: the brief's six tests, then the decisions it leaves open.

Every test trains on synthetic bags on the CPU with the frozen settings of
``configs/pipeline.yaml`` (``MILConfig.from_cfg``), so a passing test speaks about the model the
real run uses. The block after the brief's six pins what Task 5 and the analysis lean on: a
model's result does not depend on its neighbours or its position in the batch; held-out labels
are never read (R3); a model with every gate shut scores everyone exactly alike, at the
even-odds logit; padding changes nothing; the reported gates carry no training noise; the
outputs keep their documented shapes; and the transcript-only variant (no child measures)
trains.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from asdpipe.eval.metrics import safe_auc
from asdpipe.mil.bags import Bags, synthetic_bags
from asdpipe.mil.model import MILConfig, fit_predict, n_parameters, prepare


@pytest.fixture(scope="module")
def mcfg(cfg) -> MILConfig:
    return MILConfig.from_cfg(cfg)


def _small() -> tuple[Bags, np.ndarray, np.ndarray]:
    """20 ASD and 19 TD, the size of the real cohort; two training masks, two label sets.

    Mask 0 holds out rows 0, 5, 10, ...; mask 1 holds out rows 3, 8, 13, ... Label set 0 is the
    truth and label set 1 a shuffle of it, the way the permutation test will call the model.
    """
    b = synthetic_bags(20, 19, signal=1.0, seed=1)
    rows = np.arange(len(b.ids))
    tm = np.stack([rows % 5 != 0, rows % 5 != 3])
    lab = np.stack([b.y, np.random.default_rng(0).permutation(b.y)])
    return b, tm, lab


def _single_segment_and_unmeasured(where: str) -> Bags:
    """Plan edge cases 1 and 3, on 10 ASD and 10 TD.

    Child 0 has a single segment and nothing measured in it, nor at child level. Segment
    measure 1 and child measure 2 were measured for nobody. ``where="one_child"`` keeps the
    other bags long, so child 0's bag is mostly padding; ``"all_children"`` gives every child
    one segment (T = 1).
    """
    n_seg = (1, 1) if where == "all_children" else (20, 60)
    b = synthetic_bags(10, 10, signal=1.0, seed=2, n_seg=n_seg)
    seg, mask, child = b.seg.copy(), b.mask.copy(), b.child.copy()
    mask[0, 1:] = False
    seg[0] = 0.0
    seg[0, 0] = np.nan
    child[0] = np.nan
    seg[mask, 1] = np.nan
    child[:, 2] = np.nan
    return dataclasses.replace(b, seg=seg, mask=mask, child=child)


# ---------------------------------------------------------------- the brief's six tests
def test_parameter_count_is_about_100():
    assert n_parameters(6, 4, 8) == 88


def test_models_in_a_batch_are_independent(mcfg):
    b, tm, lab = _small()
    alone = fit_predict(b, tm[:1], lab[:1], [7], mcfg)
    pair = fit_predict(b, tm[[0, 0]], lab[[0, 0]], [7, 8], mcfg)
    np.testing.assert_allclose(alone.scores[0], pair.scores[0], atol=1e-5)


def test_learns_planted_signal_on_held_out_children(mcfg):
    b = synthetic_bags(40, 40, signal=1.5, seed=0)
    rows = np.arange(len(b.ids))
    train = rows % 2 == 0
    f = fit_predict(b, train[None], b.y[None], [0], mcfg)
    assert safe_auc(b.y[~train], f.scores[0, ~train]) > 0.8


def test_gates_keep_signal_and_close_noise(mcfg):
    b = synthetic_bags(40, 40, signal=1.5, seed=0)
    Ds = b.seg.shape[2]
    all_rows = np.ones((5, len(b.ids)), dtype=bool)
    f = fit_predict(b, all_rows, b.y[None].repeat(5, 0), range(5), mcfg)
    g = f.gates.mean(0)
    assert g[0] > 0.5 and g[Ds] > 0.5                      # informative seg / child gate open
    assert np.delete(g, [0, Ds]).mean() < 0.5               # noise gates mostly closed


def test_prepare_uses_training_rows_only():
    b, tm, _ = _small()
    x1, _ = prepare(b, tm)
    seg = b.seg.copy()
    seg[np.flatnonzero(~tm[0])[0]] *= 1000                  # a test child of model 0
    b2 = dataclasses.replace(b, seg=seg)
    x2, _ = prepare(b2, tm)
    np.testing.assert_allclose(x1[0][tm[0]], x2[0][tm[0]])


@pytest.mark.parametrize("where", ["one_child", "all_children"])
def test_all_nan_column_and_single_segment_bag(mcfg, where):
    b = _single_segment_and_unmeasured(where)
    rows = np.arange(len(b.ids))
    tm = np.stack([rows % 2 == 0, rows % 2 == 1])          # child 0 trains in model 0 only
    f = fit_predict(b, tm, np.stack([b.y, b.y]), [0, 1], mcfg)
    assert np.isfinite(f.scores).all()
    assert np.allclose(f.attention[0, 0, 0], 1.0)
    assert np.allclose(f.attention[:, 0, 0], 1.0) and (f.attention[:, 0, 1:] == 0).all()


# ---------------------------------------------------------------- decisions the brief leaves open
def test_config_is_the_frozen_settings(cfg):
    """from_cfg reads mil.model, which holds the defaults; a misspelt or missing key raises."""
    assert MILConfig.from_cfg(cfg) == MILConfig()
    typo = copy.deepcopy(cfg)
    typo["mil"]["model"]["hiden"] = typo["mil"]["model"].pop("hidden")
    with pytest.raises(ValueError, match="hiden"):
        MILConfig.from_cfg(typo)


def test_neighbours_and_position_do_not_change_a_model(mcfg):
    """Model 0 alone, then in the middle of other masks, labels and seeds: the same model.

    Also the outputs' contract: the shapes, gates in [0, 1], attention exactly 0 on padding and
    summing to 1 over each child's real segments, and a rerun reproduces every number.
    """
    b, tm, lab = _small()
    alone = fit_predict(b, tm[:1], lab[:1], [7], mcfg)
    mixed = fit_predict(b, tm[[1, 0, 1]], lab[[1, 0, 1]], [8, 7, 9], mcfg)
    # 1e-9, far inside Task 5's 1e-5: the model computes in float64 (float32 reached 1e-5)
    for name in ("scores", "gates", "attention"):
        np.testing.assert_allclose(getattr(mixed, name)[1], getattr(alone, name)[0], atol=1e-9)
    assert not np.allclose(mixed.scores[0], mixed.scores[2])  # different seeds, different models

    n, t, d_seg = b.seg.shape
    assert mixed.scores.shape == (3, n) and mixed.attention.shape == (3, n, t)
    assert mixed.gates.shape == (3, d_seg + b.child.shape[1])
    assert ((mixed.gates >= 0) & (mixed.gates <= 1)).all()
    assert (mixed.attention[:, ~b.mask] == 0).all()
    np.testing.assert_allclose(mixed.attention.sum(-1), 1.0, atol=1e-6)

    again = fit_predict(b, tm[:1], lab[:1], [7], mcfg)
    for name in ("scores", "gates", "attention"):
        np.testing.assert_array_equal(getattr(again, name), getattr(alone, name))


def test_held_out_labels_are_never_read(mcfg):
    """R3: flipping every held-out child's label changes nothing, class weights included."""
    b, tm, lab = _small()
    quick = dataclasses.replace(mcfg, epochs=30)
    seen = fit_predict(b, tm, lab, [0, 1], quick)
    flipped = fit_predict(b, tm, np.where(tm, lab, 1 - lab), [0, 1], quick)
    for name in ("scores", "gates", "attention"):
        np.testing.assert_array_equal(getattr(flipped, name), getattr(seen, name))


def test_a_model_with_every_gate_shut_scores_everyone_alike(mcfg):
    """Gates that start far below 0 never open, so the model sees no measure: one score for
    all. Rounding must not split it by bag length, or an AUC would rank the children by their
    number of segments (under permuted labels most models end with every gate shut).

    After the fixed epochs that one score has settled at the even-odds logit 0 even on a
    lopsided training fold (20 ASD, 8 TD): weighting ASD by n_TD / n_ASD makes both classes
    count the same. Unweighted it settles at log(20 / 8) = 0.92, inverted at twice that."""
    b, _, _ = _small()
    rows = np.arange(len(b.ids))
    lopsided = (b.y == 1) | (rows < 28)                    # all 20 ASD, 8 of the 19 TD
    shut = dataclasses.replace(mcfg, gate_init=-5.0)
    f = fit_predict(b, np.stack([lopsided, rows % 5 != 0]), np.stack([b.y, b.y]), [0, 1], shut)
    assert (f.gates == 0).all()
    assert [np.unique(row).size for row in f.scores] == [1, 1]
    np.testing.assert_allclose(f.scores, 0.0, atol=0.05)


def test_padding_is_invisible(mcfg):
    """Seven more padding rows change no number: the score pools over real segments only and
    prepare's statistics ignore padding. The added positions get attention exactly 0."""
    b, tm, lab = _small()
    n, t, d_seg = b.seg.shape
    longer = dataclasses.replace(
        b, seg=np.concatenate([b.seg, np.zeros((n, 7, d_seg), np.float32)], axis=1),
        mask=np.concatenate([b.mask, np.zeros((n, 7), bool)], axis=1))
    quick = dataclasses.replace(mcfg, epochs=50)
    short = fit_predict(b, tm, lab, [0, 1], quick)
    padded = fit_predict(longer, tm, lab, [0, 1], quick)
    np.testing.assert_allclose(padded.scores, short.scores, atol=1e-9)
    np.testing.assert_allclose(padded.gates, short.gates, atol=1e-9)
    np.testing.assert_allclose(padded.attention[:, :, :t], short.attention, atol=1e-9)
    assert (padded.attention[:, :, t:] == 0).all()


def test_reported_gates_carry_no_training_noise(mcfg):
    """Gates are reported as clamp(mu, 0, 1): after one Adam step, which moves a parameter by
    at most lr, every gate is still within lr of gate_init. Noise would spread them over [0, 1]."""
    b, tm, lab = _small()
    f = fit_predict(b, tm, lab, [0, 1], dataclasses.replace(mcfg, epochs=1))
    np.testing.assert_allclose(f.gates, mcfg.gate_init, atol=1.001 * mcfg.lr)


def test_prepare_fills_and_scales_from_the_training_fold():
    """Child layer from training rows only; padding 0; a measure no training child has is 0
    for everyone, even a test child who has it; the training segments come out mean 0, sd 1."""
    b, tm, _ = _small()
    test_child = np.flatnonzero(~tm[0])[0]
    child = b.child.copy()
    child[test_child] *= 1000
    _, c1 = prepare(b, tm)
    _, c2 = prepare(dataclasses.replace(b, child=child), tm)
    np.testing.assert_allclose(c1[0][tm[0]], c2[0][tm[0]])

    seg, child = b.seg.copy(), b.child.copy()
    seg[b.mask, 1] = np.nan
    seg[test_child, b.mask[test_child], 1] = 5.0
    child[:, 3] = np.nan
    child[test_child, 3] = 5.0
    x, c = prepare(dataclasses.replace(b, seg=seg, child=child), tm)
    assert np.isfinite(x).all() and np.isfinite(c).all()
    assert (x[:, ~b.mask] == 0).all()
    assert (x[0, :, :, 1] == 0).all() and (c[0, :, 3] == 0).all()

    kept = [0, 2, 3, 4, 5]                                 # every measured column
    fold = x[0][tm[0]][b.mask[tm[0]]]                       # model 0's training segments
    np.testing.assert_allclose(fold[:, kept].mean(0), 0.0, atol=1e-9)
    np.testing.assert_allclose(fold[:, kept].std(0), 1.0, atol=1e-9)


def test_a_column_constant_in_training_is_zero_for_everyone():
    """A column in which all of a model's training children share one value carries nothing
    in that fold: it is 0 for every child, a held-out child who differs included. Divided by
    the sd floor instead, that child's difference would enter a million times over. In a fold
    where the same column varies it is scaled as usual."""
    b, tm, _ = _small()
    held_out = np.flatnonzero(~tm[0])[0]                   # held out by model 0, trained by 1
    seg, child = b.seg.copy(), b.child.copy()
    seg[b.mask, 2] = 3.0
    seg[held_out, b.mask[held_out], 2] = 7.0
    child[:, 1] = 2.0
    child[held_out, 1] = 9.0
    x, c = prepare(dataclasses.replace(b, seg=seg, child=child), tm)
    assert (x[0, :, :, 2] == 0).all() and (c[0, :, 1] == 0).all()
    assert np.abs(x[1][b.mask][:, 2]).max() > 0 and np.abs(c[1, :, 1]).max() > 0


def test_transcript_only_variant_and_refused_inputs(mcfg):
    """No child measures (the transcript-only run) trains; inconsistent inputs raise."""
    b, tm, lab = _small()
    quick = dataclasses.replace(mcfg, epochs=5)
    f = fit_predict(b.select(child_names=()), tm, lab, [0, 1], quick)
    assert f.gates.shape == (2, 6) and np.isfinite(f.scores).all()
    assert n_parameters(6, 0, 8) == 80

    with pytest.raises(ValueError, match="seed"):
        fit_predict(b, tm, lab, [0], quick)
    with pytest.raises(ValueError, match="labels"):
        fit_predict(b, tm, lab[:1], [0, 1], quick)
    with pytest.raises(ValueError, match="both"):
        fit_predict(b, tm & (lab == 0), lab, [0, 1], quick)  # no ASD child in training
