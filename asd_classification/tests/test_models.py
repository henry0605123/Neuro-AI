"""M3 / M5 / M6: assembly, baselines and the primary model's pinned architecture."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asdpipe.assemble import b0_column_mask, build_table, concat_features, stream_blocks
from asdpipe.config import load_decision_rules
from asdpipe.cv.guards import FoldScope
from asdpipe.cv.splits import subject_folds
from asdpipe.eval.metrics import empirical_mde, paired_bootstrap, safe_auc, subject_metrics
from asdpipe.models.baselines import STREAMS, fit_predict_b0, fit_predict_b1, fit_predict_b2
from asdpipe.models.m1 import M1Net, assert_architecture, count_parameters, fit_predict_m1
from test_leakage import fake_table  # noqa: F401  -- reuse the fabricated table


def test_b0_dimension_stays_inside_the_36_66_budget(fake_table, cfg):
    """R5 pins B0 at 36-66 dimensions; the frozen reduced set must respect it."""
    mask = b0_column_mask(fake_table, cfg)
    lo, hi = cfg["features"]["b0"]["target_dim_range"]
    assert lo <= int(mask.sum()) <= hi


def test_b0_raises_when_the_reduced_set_falls_out_of_budget(fake_table, cfg):
    tiny = {**cfg, "features": {**cfg["features"],
                                "b0": {**cfg["features"]["b0"], "target_dim_range": [200, 300]}}}
    with pytest.raises(ValueError, match="R5"):
        b0_column_mask(fake_table, tiny)


def test_stream_blocks_partition_the_feature_matrix(fake_table):
    blocks = stream_blocks(fake_table)
    X = concat_features(fake_table)
    assert blocks["e"][1] == X.shape[1]
    assert np.allclose(X[:, slice(*blocks["b"])], fake_table.X_b)
    assert np.allclose(X[:, slice(*blocks["c"])], fake_table.X_c)


def test_b1_fusion_has_exactly_three_plus_one_parameters(fake_table, cfg):
    """R5: the identifiable fusion layer is logit = b + sum_s w_s f_s."""
    fold = subject_folds(fake_table.y, 3, 0)[0]
    out = fit_predict_b1(FoldScope(fold, 0), fake_table, cfg)
    assert out["n_params"] == 4
    assert set(out["weights"]) == set(STREAMS)
    assert pytest.approx(sum(out["contribution"].values()), abs=1e-9) == 1.0


def test_every_baseline_scores_exactly_the_test_subjects(fake_table, cfg):
    fold = subject_folds(fake_table.y, 3, 0)[0]
    for fn in (fit_predict_b0, fit_predict_b1, fit_predict_b2):
        out = fn(FoldScope(fold, 0), fake_table, cfg)
        assert out["scores"].shape == (len(fold.test_idx),)
        assert out["scores_all"].shape == (len(fake_table),)


def test_m1_architecture_is_pinned(cfg):
    m = dict(cfg["cv"]["m1"])
    assert_architecture(m)
    for key, bad in (("tcn_channels", 32), ("mlp_hidden", 128)):
        with pytest.raises(ValueError, match="R5"):
            assert_architecture({**m, key: bad})
    with pytest.raises(ValueError, match="R5"):
        assert_architecture({**m, "tcn_dilations": [1, 2, 3, 4]})


def test_m1_parameter_count_matches_the_documented_budget(cfg):
    """Plan doc 5.0.2 puts M1 at roughly 50-95k trainable parameters."""
    net = M1Net(seq_dim=11, tab_dim=96, cfg=cfg["cv"]["m1"])
    n = count_parameters(net)
    assert 40_000 <= n <= 95_000, n
    # the TCN alone should be in the 42-49k single-conv-block range the doc quotes
    tcn = count_parameters(net.tcn)
    assert 35_000 <= tcn <= 50_000, tcn


def test_m1_tcn_dilations_are_actually_applied(cfg):
    net = M1Net(seq_dim=11, tab_dim=4, cfg=cfg["cv"]["m1"])
    dil = [blk[0].dilation[0] for blk in net.tcn.blocks]
    assert dil == [1, 2, 4, 8]
    assert all(blk[0].out_channels == 64 for blk in net.tcn.blocks)


def test_m1_masked_pooling_ignores_padding(cfg):
    """Padded positions must not move the pooled embedding."""
    torch.manual_seed(0)
    net = M1Net(seq_dim=11, tab_dim=2, cfg=cfg["cv"]["m1"]).eval()
    seq = torch.randn(1, 20, 11)
    mask = torch.zeros(1, 20)
    mask[0, :8] = 1.0
    with torch.inference_mode():
        a = net.tcn(seq, mask)
        seq2 = seq.clone()
        seq2[0, 8:] = 99.0                # garbage in the padded region
        b = net.tcn(seq2, mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_m1_uses_class_weighted_cross_entropy_not_focal():
    """Section 5.7: Focal is an ablation only -- it must not appear in the model code."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "asdpipe" / "models" / "m1.py")
    import ast

    tree = ast.parse(src.read_text(encoding="utf-8"))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "CrossEntropyLoss" in called, "M1 must train with class-weighted cross-entropy"
    assert not {"FocalLoss", "focal_loss", "sigmoid_focal_loss"} & called


def test_m1_runs_and_respects_the_fold(fake_table, cfg):
    fold = subject_folds(fake_table.y, 3, 0)[0]
    scope = FoldScope(fold, 0)
    fast = {**cfg, "cv": {**cfg["cv"], "m1": {**cfg["cv"]["m1"], "max_epochs": 5, "patience": 3}}}
    out = fit_predict_m1(scope, fake_table, fast, torch_seed=0)
    assert out["scores"].shape == (len(fold.test_idx),)
    assert out["epochs_run"] <= 5
    for ev in scope.log.for_fold(fold.fold_id, 0):
        assert set(ev.indices) <= set(fold.train_idx.tolist())


def test_m1_ablation_drops_the_requested_stream(fake_table, cfg):
    fold = subject_folds(fake_table.y, 3, 0)[0]
    fast = {**cfg, "cv": {**cfg["cv"], "m1": {**cfg["cv"]["m1"], "max_epochs": 3, "patience": 2}}}
    for s in STREAMS:
        out = fit_predict_m1(FoldScope(fold, 0), fake_table, fast, torch_seed=0, drop_stream=s)
        assert out["dropped_stream"] == s


# ---------------------------------------------------------------- metrics
def test_auc_hand_example():
    """Perfect separation -> 1.0; reversed -> 0.0; tied -> 0.5."""
    y = np.array([0, 0, 1, 1])
    assert safe_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert safe_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert safe_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_subject_metrics_hand_example():
    """y = [0,0,1,1], probabilities [0.1,0.6,0.4,0.9] at threshold 0.5.

    predictions -> [0,1,0,1]; tp=1, tn=1, fp=1, fn=1
    sensitivity = 1/2, specificity = 1/2, accuracy = 2/4
    """
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = subject_metrics(y, np.log(p / (1 - p)), p)
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.5)


def test_paired_bootstrap_reports_a_zero_difference_for_identical_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    out = paired_bootstrap(y, s, s, n_boot=500, seed=0)
    assert out["observed"] == pytest.approx(0.0)
    assert out["ci_low"] == pytest.approx(0.0)
    assert out["ci_high"] == pytest.approx(0.0)


def test_empirical_mde_scales_with_the_standard_error():
    assert empirical_mde(0.025) == pytest.approx(2.8016 * 0.025, rel=1e-3)


def test_decision_rules_file_exists_and_pins_the_honesty_clause():
    """R5: this file has to be on disk before any training run."""
    rules = load_decision_rules()
    assert rules["primary_model"] == "M1"
    assert set(rules["mandatory_baselines"]) == {"B0", "B1", "B2"}
    assert rules["loss"]["focal"] is False
    assert rules["overfit_guards"]["early_stopping"] == "inner_fold_only"
    assert "baseline" in rules["honesty_clause"]["rule"].lower()
    # 7 since 2026-09-12: F7 (多輪語意調適) left the family on a measured floor effect.
    # The size is asserted because it sets the Holm correction; a silent change there would
    # quietly relax every threshold in the primary analysis.
    cv = rules["criterion_validity"]
    assert cv["primary_indicator_family_size"] == 7
    assert len(cv["primary_indicators"]) == cv["primary_indicator_family_size"]
    assert "F7" not in cv["primary_indicators"]
    # The assessment-sheet substitutes must stay outside the family until the PI promotes
    # them: adding features to the correction set is not a code decision.
    assert set(cv["exploratory_candidates"]) & set(cv["primary_indicators"]) == set()


# ---------------------------------------------------------------- decision table
def test_verdict_table_matches_the_pre_registered_rules():
    """Plan doc 4b.2-5, plus the two implicit points documented in verdict_for."""
    from asdpipe.eval.analysis import verdict_for

    rules = load_decision_rules()
    # row 1: strong effect, CI excludes zero
    assert verdict_for(0.7, 0.2, 0.9, 0.5, rules) == "Phase 1 core feature"
    # magnitude is what counts: a negative expected direction still qualifies
    assert verdict_for(-0.7, -0.9, -0.2, 0.5, rules) == "Phase 1 core feature"
    # rho >= 0.5 but the CI includes 0 -- not a row in the table, must not be dropped
    assert "CI includes 0" in verdict_for(0.55, -0.09, 0.94, 0.2, rules)
    # row 2
    assert verdict_for(0.35, -0.1, 0.7, 0.1, rules).startswith("Phase 1 candidate; review")
    # row 3: the confound fingerprint
    assert verdict_for(0.1, -0.4, 0.5, 1.2, rules).startswith("WARNING")
    # row 4
    assert verdict_for(0.1, -0.4, 0.5, 0.2, rules) == "demote or drop"
    assert verdict_for(float("nan"), 0.0, 0.0, 0.0, rules) == "not computable"


def test_holm_correction_hand_example():
    """p = 0.001, 0.02, 0.04 with a family of 3.

    step 1: 0.001 <= 0.05/3 = 0.01667 -> reject
    step 2: 0.020 <= 0.05/2 = 0.025   -> reject
    step 3: 0.040 <= 0.05/1 = 0.05    -> reject
    """
    from asdpipe.eval.analysis import holm

    out = holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
    assert [out[k]["reject"] for k in "abc"] == [True, True, True]
    assert out["a"]["threshold"] == pytest.approx(0.05 / 3)
    # once a step fails, everything after it fails too
    out2 = holm({"a": 0.001, "b": 0.03, "c": 0.04}, alpha=0.05)
    assert [out2[k]["reject"] for k in "abc"] == [True, False, False]
