"""Cross-validation driver: M1 and the three mandatory baselines on identical splits.

The comparison protocol comes straight from `configs/decision_rules.yaml`, which was frozen
before any run: same folds, same seed list, paired subject-level bootstrap, non-significant
results reported, and the honesty clause deciding what gets delivered.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .assemble import build_table
from .contracts import SubjectTable
from .cv.guards import LEAKAGE_LOG, FoldScope
from .cv.splits import inner_folds, subject_folds
from .eval import analysis as an
from .eval import metrics as mt
from .models.baselines import STREAMS, fit_predict_b0, fit_predict_b1, fit_predict_b2
from .models.m1 import fit_predict_m1

__all__ = ["run_cv", "MODELS", "cv_settings", "confirmatory_exclusions", "confirmatory_table"]

MODELS = ("M1", "B0", "B1", "B2")


def _fit_all(scope: FoldScope, table: SubjectTable, cfg: Mapping,
             with_ablation: bool) -> dict[str, dict]:
    out = {
        "B0": fit_predict_b0(scope, table, cfg),
        "B1": fit_predict_b1(scope, table, cfg),
        "B2": fit_predict_b2(scope, table, cfg),
        "M1": fit_predict_m1(scope, table, cfg, torch_seed=scope.seed),
    }
    if with_ablation:
        for s in STREAMS:
            out[f"M1-no-{s}"] = fit_predict_m1(scope, table, cfg, torch_seed=scope.seed,
                                               drop_stream=s)
    return out


def cv_settings(cfg: Mapping, real: bool) -> dict:
    """Outer CV for this run. Real recordings take ``cv.real`` and refuse anything below the
    registration: >= 5 seeds (plan doc 5.7) of stratified 5-fold (audit P1-9). The smoke
    settings in ``cv.outer`` exist only to keep synthetic runs fast (health check P19)."""
    block = cfg["cv"]["real" if real else "outer"]
    n_folds, seeds = int(block["n_folds"]), list(block["seeds"])
    if real and (n_folds < 5 or len(seeds) < 5):
        raise ValueError(f"real-data CV needs >= 5 folds and >= 5 seeds; cv.real has "
                         f"{n_folds} folds and {len(seeds)} seeds")
    return {"n_folds": n_folds, "seeds": seeds}


def confirmatory_exclusions(rules: Mapping) -> list[str]:
    """Features the convention-invariance gate excluded from confirmatory analysis (D11)."""
    gate = rules.get("convention_invariance_gate", {})
    return list(gate.get("outcome", {}).get("excluded_from_confirmatory", []))


def confirmatory_table(table: SubjectTable, rules: Mapping) -> SubjectTable:
    """The table every confirmatory model sees: the gate's exclusions removed."""
    return table.without_features(confirmatory_exclusions(rules))


def _inner_oof_scores(scope: FoldScope, table: SubjectTable, cfg: Mapping
                      ) -> dict[str, np.ndarray]:
    """Out-of-fold scores for the training rows of one outer fold (health check P16).

    Platt scaling fitted on a model's scores for its own training rows learns how sure the
    model is about subjects it was fitted to, and carries that confidence to new ones:
    under shuffled labels the four models scored Brier 0.265-0.363, worse than the 0.182
    of always predicting the base rate. Here every training row is scored by a model that
    never saw it, refitted on the rest of the training rows inside the outer fold.
    """
    y = table.y
    out = {m: np.full(len(y), np.nan) for m in MODELS}
    for f in inner_folds(scope.train_idx, y, int(cfg["cv"]["inner"]["n_folds"]), scope.seed):
        fitted = _fit_all(scope.inner(f), table, cfg, with_ablation=False)
        for m in MODELS:
            out[m][f.test_idx] = fitted[m]["scores"]
    return out


def run_cv(table: SubjectTable, cfg: Mapping, rules: Mapping,
           with_ablation: bool = True, run_probe: bool = True, real: bool = False) -> dict:
    """``real=True`` for real recordings: registered CV settings (`cv_settings`)."""
    # The gate's exclusions apply to every model input -- synthetic runs included, so the
    # smoke test exercises exactly the configuration a real run will use.
    table = confirmatory_table(table, rules)
    y = table.y
    n = len(y)
    outer = cv_settings(cfg, real)
    n_folds, seeds = outer["n_folds"], outer["seeds"]
    n_boot = int(rules["comparison"]["n_boot"])
    LEAKAGE_LOG.clear()

    model_names = list(MODELS) + ([f"M1-no-{s}" for s in STREAMS] if with_ablation else [])
    per_seed: dict[int, dict] = {}

    for seed in seeds:
        folds = subject_folds(y, n_folds, seed)
        oof = {m: np.full(n, np.nan) for m in model_names}
        # Ablations are judged on AUC alone, which needs no calibration.
        prob = {m: np.full(n, np.nan) for m in MODELS}
        stream_oof = {s: np.full(n, np.nan) for s in STREAMS}
        fold_ids = np.full(n, -1)
        fold_meta = []

        for fold in folds:
            scope = FoldScope(fold, seed)
            fitted = _fit_all(scope, table, cfg, with_ablation)
            inner = _inner_oof_scores(scope, table, cfg)
            fold_ids[fold.test_idx] = fold.fold_id
            for m, res in fitted.items():
                oof[m][fold.test_idx] = res["scores"]
                if m not in inner:
                    continue
                # Platt calibration on out-of-fold scores of the training rows (P16)
                s_in, s_test = inner[m], np.asarray(res["scores"], dtype=float)
                tr = scope.train_idx[np.isfinite(s_in[scope.train_idx])]
                prob[m][fold.test_idx] = scope.fit_callable(
                    lambda idx, s_in=s_in, s_test=s_test: mt.platt_calibrate(
                        s_in[idx], y[idx], s_test),
                    idx=tr, stage=f"calibrate:{m}",
                )
            for s in STREAMS:
                stream_oof[s][fold.test_idx] = fitted["B1"]["stream_scores"][s]
            fold_meta.append({
                "fold_id": fold.fold_id,
                "n_train": int(len(fold.train_idx)),
                "n_test": int(len(fold.test_idx)),
                "b1_contribution": fitted["B1"]["contribution"],
                "b1_weights": fitted["B1"]["weights"],
                "m1_epochs": fitted["M1"]["epochs_run"],
            })

        centred = {m: mt.fold_center(oof[m], fold_ids) for m in model_names}
        metrics = {m: mt.subject_metrics(y, centred[m], prob.get(m)) for m in model_names}
        leaderboard = {s: mt.safe_auc(y, mt.fold_center(stream_oof[s], fold_ids))
                       for s in STREAMS}

        comparisons = {}
        for baseline in ("B0", "B1", "B2"):
            cmp_ = mt.paired_bootstrap(y, centred["M1"], centred[baseline],
                                       n_boot=n_boot, seed=seed)
            cmp_["mde"] = mt.empirical_mde(cmp_.get("se", float("nan")))
            comparisons[f"M1_vs_{baseline}"] = cmp_

        ablation = {}
        if with_ablation:
            for s in STREAMS:
                ablation[s] = {
                    "auc_without": metrics[f"M1-no-{s}"]["auc"],
                    "delta_auc": metrics["M1"]["auc"] - metrics[f"M1-no-{s}"]["auc"],
                }

        per_seed[seed] = {
            "metrics": metrics, "leaderboard": leaderboard, "comparisons": comparisons,
            "ablation": ablation, "folds": fold_meta,
            "oof_scores": {m: centred[m].tolist() for m in model_names},
            "probabilities": {m: p.tolist() for m, p in prob.items()},   # calibrated models only
        }

    # ---- across seeds ----
    across = {}
    for m in model_names:
        vals = [per_seed[s]["metrics"][m]["auc"] for s in seeds]
        across[m] = {
            "auc_mean": float(np.nanmean(vals)),
            "auc_sd": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "auc_per_seed": [float(v) for v in vals],
        }
    for m in model_names:
        for k in ("accuracy", "sensitivity", "specificity", "f1", "brier"):
            across[m][f"{k}_mean"] = float(
                np.nanmean([per_seed[s]["metrics"][m][k] for s in seeds])
            )

    delivered = _apply_honesty_clause(per_seed, seeds, rules)

    result = {
        "n_subjects": n, "n_folds": n_folds, "seeds": seeds,
        "per_seed": per_seed, "across_seeds": across,
        "delivered_model": delivered,
        "leakage_events": len(LEAKAGE_LOG.events),
        "labels": y.tolist(),
        "subject_ids": [str(s) for s in table.subject_ids],
    }
    if run_probe:
        result["examiner_probe"] = an.examiner_probe(
            table, n_permutations=int(cfg["analysis"]["n_permutations"]), seed=0,
            n_folds=n_folds,
        )
    return result


def _apply_honesty_clause(per_seed: Mapping, seeds, rules: Mapping) -> dict:
    """R5 / decision_rules.yaml: M1 must beat BOTH B0 and B1 to be the delivered model."""
    alpha = float(rules["comparison"]["alpha"])
    wins = {}
    for baseline in ("B0", "B1"):
        ps = [per_seed[s]["comparisons"][f"M1_vs_{baseline}"] for s in seeds]
        beat = all(
            np.isfinite(c["p_two_sided"]) and c["p_two_sided"] < alpha and c["observed"] > 0
            for c in ps
        )
        wins[baseline] = {
            "significant_in_every_seed": bool(beat),
            "observed_mean": float(np.nanmean([c["observed"] for c in ps])),
            "mde_mean": float(np.nanmean([c["mde"] for c in ps])),
        }
    m1_wins = wins["B0"]["significant_in_every_seed"] and wins["B1"]["significant_in_every_seed"]

    # B1 collapse check (plan doc 5.0.2b, audit P1-11): a single stream taking more than
    # 90% of the contribution, or contributions swinging across seeds, means the stacking
    # layer is fitting noise -- fall back to B0's equal-weight concatenation.
    contribs = [f["b1_contribution"] for s in seeds for f in per_seed[s]["folds"]]
    max_share = max((max(c.values()) for c in contribs), default=0.0)
    share_sd = float(np.mean([
        np.std([c[k] for c in contribs]) for k in (contribs[0].keys() if contribs else [])
    ])) if contribs else 0.0
    b1_auc = [per_seed[s]["metrics"]["B1"]["auc"] for s in seeds]
    b1_unstable = bool(max_share > 0.9 or share_sd > 0.25
                       or (len(b1_auc) > 1 and np.nanstd(b1_auc, ddof=1) > 0.15))

    if m1_wins:
        model = rules["honesty_clause"]["delivered_when_m1_wins"]
    elif b1_unstable:
        model = rules["honesty_clause"]["fallback_chain"][-1]      # B0
    else:
        model = rules["honesty_clause"]["delivered_when_m1_ties"]  # B1

    return {
        "model": model,
        "m1_beats_baselines": bool(m1_wins),
        "b1_unstable": b1_unstable,
        "b1_max_single_stream_contribution": float(max_share),
        "b1_contribution_sd_across_folds": share_sd,
        "detail": wins,
        "rule": rules["honesty_clause"]["rule"],
        "note": ("Non-significant means 'gain < MDE', not 'gain = 0'. The MDE at this "
                 "sample size is reported alongside every comparison."),
    }
