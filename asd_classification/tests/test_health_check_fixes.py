"""Health check 0930, step 4: fixes that change what a measure means or how a missing one
is treated. Each test names the problem it guards (P4, P10, P12, P16, P19, P20, P25)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from asdpipe.config import load_decision_rules, load_pipeline_config
from asdpipe.contracts import Fold, Label, Speaker, SubjectRow, SubjectTable
from asdpipe.cv.guards import FoldScope
from asdpipe.features.e_text import extract_e, pair_turns

from conftest import hand_cfg, make_turn


def _e_inputs(turns, seed=0):
    rng = np.random.default_rng(seed)
    emb = {t.turn_id: rng.normal(size=8).astype(np.float32) for t in turns}
    words = {t.turn_id: list(t.text) for t in turns}
    db = {t.turn_id: 60.0 + t.turn_id for t in turns}
    return emb, words, db


def _three_kinds_of_examiner_turn():
    return [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "我們來玩積木"),                # statement
            make_turn(1, Speaker.CHILD, 1.5, 2.5, "我要紅色的"),
            make_turn(2, Speaker.EXAMINER, 3.0, 4.0, "你喜歡嗎", is_question=True,
                      is_yes_no=True),
            make_turn(3, Speaker.CHILD, 4.5, 5.5, "很喜歡啊"),
            make_turn(4, Speaker.EXAMINER, 6.0, 7.0, "你昨天去哪裡", is_question=True),
            make_turn(5, Speaker.CHILD, 7.5, 8.5, "去阿嬤家")]


# ---------------------------------------------------------------- P10
def test_a_continuation_is_not_a_response():
    t = [make_turn(0, Speaker.EXAMINER, 0.0, 1.0, "你今天做了什麼", is_question=True),
         make_turn(1, Speaker.CHILD, 1.5, 2.5, "我去公園"),
         make_turn(2, Speaker.CHILD, 4.5, 5.5, "還有溜滑梯"),        # the child carrying on
         make_turn(3, Speaker.EXAMINER, 6.0, 7.0, "好棒"),
         make_turn(4, Speaker.CHILD, 7.5, 8.0, "對")]
    assert [(e.turn_id, c.turn_id) for e, c in pair_turns(t)] == [(0, 1), (3, 4)]


def test_g1_takes_answers_to_open_questions_only():
    t = _three_kinds_of_examiner_turn()
    q = extract_e(t, *_e_inputs(t), hand_cfg(load_pipeline_config())).quality
    assert q["e_n_pairs"] == 3
    assert (q["e_excluded_not_question"], q["e_excluded_yes_no"], q["e_n_g1_pairs"]) == (1, 1, 1)


# ---------------------------------------------------------------- P12
def test_too_few_pairs_is_unmeasurable_not_zero(cfg):
    t = _three_kinds_of_examiner_turn()                    # 3 pairs, floor is 10
    out = extract_e(t, *_e_inputs(t), cfg)
    for k in ("e_g1_similarity_median", "e_f1_embed_high_ratio", "e_latency_median",
              "e_volume_following_slope", "e_f8_asymmetry"):
        assert np.isnan(out.scalars[k]), k
    assert np.isnan(out.pooled(["mean"])["e_latency_mean"])


def test_b7_with_too_few_windows_is_nan_not_zero():
    from asdpipe.features.b_voice import BStreamFeatures

    b7 = BStreamFeatures().attach(np.eye(4, 192, dtype=np.float32), min_windows=15).b7()
    assert np.isnan(b7["b7_mean_pairwise_cosdist"]) and b7["b7_n_windows"] == 4.0


def _row(i, label, c_pooled):
    return SubjectRow(subject_id=f"S{i}", label=label, b_pooled={"b_x_mean": 1.0},
                      c_pooled=c_pooled, e_pooled={"e_y": 1.0},
                      c_sequence=np.zeros((0, 3), np.float32), egemaps_functionals=np.zeros(88),
                      examiner_probe={"p": 0.0}, ados={}, latents={}, quality={})


def test_table_takes_the_union_of_columns_and_marks_gaps_nan():
    """A child with no syllables returns no C features at all; that must not crash the
    table or shift every other column."""
    t = SubjectTable([_row(0, Label.ASD, {}), _row(1, Label.CONTROL, {"c_a": 1.0, "c_b": 2.0})])
    assert t.c_names == ["c_a", "c_b"]
    assert np.isnan(t.X_c[0]).all() and t.X_c[1].tolist() == [1.0, 2.0]
    assert t.unmeasurable_counts()["c"].tolist() == [2, 0]
    assert t.without_features(["c_a"]).c_names == ["c_b"]


def test_m1_imputes_from_training_rows_only():
    from asdpipe.models.m1 import _impute_median

    X = np.array([[1.0], [3.0], [np.nan], [100.0]])
    assert _impute_median(np.array([0, 1, 2]), X)[2, 0] == 2.0     # not pulled by row 3


def test_baseline_imputation_is_fitted_inside_the_fold():
    from asdpipe.models.baselines import _scaled

    X = np.array([[1.0], [3.0], [np.nan], [100.0]])
    scope = FoldScope(Fold(0, np.array([0, 1, 2]), np.array([3])), seed=0)
    Z = _scaled(scope, X, "t")
    assert np.isfinite(Z).all()
    assert Z[2, 0] == pytest.approx(0.0)       # the training median is the training mean here


# ---------------------------------------------------------------- P16
def test_calibration_scores_come_from_models_that_never_saw_the_row(monkeypatch, cfg):
    import asdpipe.run as run

    seen = []

    def fake_fit_all(scope, table, cfg_, with_ablation):
        seen.append((set(scope.train_idx.tolist()), set(scope.test_idx.tolist())))
        return {m: {"scores": np.ones(len(scope.test_idx))} for m in run.MODELS}

    monkeypatch.setattr(run, "_fit_all", fake_fit_all)
    table = type("T", (), {"y": np.array([0, 1] * 10)})()
    scope = FoldScope(Fold(0, np.arange(15), np.arange(15, 20)), seed=0)
    out = run._inner_oof_scores(scope, table, cfg)
    assert np.isfinite(out["M1"][:15]).all() and np.isnan(out["M1"][15:]).all()
    for train, test in seen:
        assert not (train & test) and (train | test) <= set(range(15))


# ---------------------------------------------------------------- P19
def test_real_cv_is_five_repeats_of_five_folds_and_refuses_less(cfg):
    from asdpipe.run import cv_settings

    assert cv_settings(cfg, real=True) == {"n_folds": 5, "seeds": [0, 1, 2, 3, 4]}
    bad = copy.deepcopy(cfg)
    bad["cv"]["real"]["seeds"] = [0, 1]
    with pytest.raises(ValueError):
        cv_settings(bad, real=True)


def test_holm_counts_indicators_that_could_not_be_computed():
    from asdpipe.eval.analysis import holm

    out = holm({"a": 0.001, "b": 0.02}, alpha=0.05, family_size=7)
    assert out["a"]["threshold"] == pytest.approx(0.05 / 7)
    assert out["b"]["threshold"] == pytest.approx(0.05 / 6)


def test_primary_indicator_map_matches_the_registered_family():
    from asdpipe.eval.analysis import CRITERION, PRIMARY_FEATURE

    registered = set(load_decision_rules()["criterion_validity"]["primary_indicators"])
    assert set(PRIMARY_FEATURE) == registered == set(CRITERION)


# ---------------------------------------------------------------- P20, P4, gate
def test_g1_median_is_one_column_not_two():
    t = _three_kinds_of_examiner_turn()
    pooled = extract_e(t, *_e_inputs(t), hand_cfg(load_pipeline_config())).pooled(
        ["mean", "std", "p25", "median", "p75"])
    assert "e_g1_similarity_median" in pooled and "e_g1_sim_median" not in pooled


def test_b0_has_no_latency_and_confirmatory_models_lose_the_gate_exclusions():
    from asdpipe.assemble import B0_E_FEATURES
    from asdpipe.run import confirmatory_exclusions, confirmatory_table

    assert not any("latency" in n for n in B0_E_FEATURES)
    excluded = confirmatory_exclusions(load_decision_rules())
    assert excluded
    t = SubjectTable([_row(0, Label.ASD, {excluded[0]: 1.0, "c_kept": 1.0}),
                      _row(1, Label.CONTROL, {excluded[0]: 2.0, "c_kept": 2.0})])
    kept = confirmatory_table(t, load_decision_rules()).all_feature_names()
    assert not any(n.split("/", 1)[1] in excluded for n in kept) and "c/c_kept" in kept


# ---------------------------------------------------------------- P25
def test_a_session_saved_in_parts_is_read_end_to_end(tmp_path, cfg):
    import soundfile as sf

    from asdpipe.realdata import discover_cases, load_subject

    sr = int(cfg["synth"]["sample_rate"])           # what load_subject decodes to
    d = tmp_path / "ASD" / "case1"
    d.mkdir(parents=True)
    sf.write(d / "a_part1.wav", np.full(sr, 0.1, np.float32), sr)
    sf.write(d / "b_part2.wav", np.full(sr // 2, -0.1, np.float32), sr)
    (d / "case1.json").write_text(json.dumps({"segments": [
        {"start": 0.2, "end": 0.8, "speaker": "Tester", "text": "你好嗎"},
        {"start": 1.1, "end": 1.4, "speaker": "Child", "text": "很好"}]}, ensure_ascii=False),
        encoding="utf-8")
    case = discover_cases(tmp_path, cohort="ASD")[0]
    assert [p.name for p in case.audio_parts] == ["a_part1.wav", "b_part2.wav"]
    rec, rep = load_subject(case, cfg)
    assert len(rec.audio) == sr + sr // 2
    assert any("joined end to end" in n for n in rep.notes)
