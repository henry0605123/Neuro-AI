"""Calibration of the gate penalty for the segment-attention model (DEVIATIONS D17, 2026-11-01).

Synthetic data only: no real recording or transcript is read. The grid, the pass thresholds and
the selection rule below were fixed (controller ruling, Task 4) before the grid was run.

Grid: gate_lambda in {0.005, 0.01, 0.02, 0.05} x epochs in {300, 500}; the other settings stay
at hidden 8, lr 0.01, weight_decay 1e-4, gate_sigma 0.5, gate_init 0.5.

For each grid point and data seed k = 0..9, at signal 1.5 and at signal 0.0, on
``synthetic_bags(40, 40, signal, seed=k)``:
  held-out setup: train on the even rows, test on the odd rows, model seed 0;
  gate setup:     5 models (seeds 0-4) on all rows, gates averaged over the 5.

Pass when, over the 10 data seeds,
  null (signal 0):   |mean held-out AUC - 0.5| <= 0.08, and the mean gate is below 0.5, both
                     over all 10 gates and over the 8 without seg 0 and child 0;
  signal (1.5):      held-out AUC > 0.8 in at least 9 of 10 seeds, and in all 10 seeds the
                     seg 0 or the child 0 gate (5-seed mean) is above 0.5.
Choice: the largest passing gate_lambda, ties going to fewer epochs. (Fallback, not needed on
2026-11-01: the most seeds meeting both signal conditions among the points passing the null.)

Run from the project directory (asd_classification); the output is deterministic:
    D:\\Anaconda\\envs\\paper\\python.exe docs\\superpowers\\calibration\\mil_calibrate.py
        > docs\\superpowers\\calibration\\mil_calibrate.log
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from asdpipe.eval.metrics import safe_auc  # noqa: E402
from asdpipe.mil.bags import synthetic_bags  # noqa: E402
from asdpipe.mil.model import MILConfig, fit_predict  # noqa: E402

FIXED = {"hidden": 8, "lr": 0.01, "weight_decay": 1e-4, "gate_sigma": 0.5, "gate_init": 0.5}
LAMBDAS, EPOCHS, DATA_SEEDS = (0.005, 0.01, 0.02, 0.05), (300, 500), range(10)


def one(cfg: MILConfig, signal: float, k: int) -> dict:
    b = synthetic_bags(40, 40, signal=signal, seed=k)
    n, d_seg = len(b.ids), b.seg.shape[2]
    even = np.arange(n) % 2 == 0
    # row 0: held-out model (seed 0, even rows); rows 1-5: gate models (seeds 0-4, all rows).
    # One batched call: the models are independent (checked against separate calls below).
    tm = np.vstack([even[None], np.ones((5, n), dtype=bool)])
    f = fit_predict(b, tm, np.repeat(b.y[None], 6, 0), [0, 0, 1, 2, 3, 4], cfg)
    g = f.gates[1:].mean(0)
    return {"auc": safe_auc(b.y[~even], f.scores[0, ~even]), "g_seg0": float(g[0]),
            "g_child0": float(g[d_seg]), "noise8": float(np.delete(g, [0, d_seg]).mean()),
            "all10": float(g.mean())}


def main() -> None:
    print(f"fixed settings: {FIXED}")
    check_cfg = MILConfig(**FIXED, gate_lambda=0.05, epochs=300)
    b = synthetic_bags(40, 40, signal=1.5, seed=0)
    n = len(b.ids)
    even = np.arange(n) % 2 == 0
    alone_h = fit_predict(b, even[None], b.y[None], [0], check_cfg)
    alone_g = fit_predict(b, np.ones((5, n), bool), b.y[None].repeat(5, 0), range(5), check_cfg)
    batch = fit_predict(b, np.vstack([even[None], np.ones((5, n), bool)]),
                        np.repeat(b.y[None], 6, 0), [0, 0, 1, 2, 3, 4], check_cfg)
    gap = max(np.abs(batch.scores[0] - alone_h.scores[0]).max(),
              np.abs(batch.gates[1:] - alone_g.gates).max())
    assert gap < 1e-9, gap
    print(f"batched == separate (max diff {gap:.1e})", flush=True)

    rows = []
    for lam in LAMBDAS:
        for ep in EPOCHS:
            cfg = MILConfig(**FIXED, gate_lambda=lam, epochs=ep)
            sig = [one(cfg, 1.5, k) for k in DATA_SEEDS]
            nul = [one(cfg, 0.0, k) for k in DATA_SEEDS]
            auc_s = np.array([r["auc"] for r in sig])
            inf_ok = np.array([max(r["g_seg0"], r["g_child0"]) > 0.5 for r in sig])
            null_auc = float(np.mean([r["auc"] for r in nul]))
            null_all10 = float(np.mean([r["all10"] for r in nul]))
            null_noise8 = float(np.mean([r["noise8"] for r in nul]))
            null_ok = abs(null_auc - 0.5) <= 0.08 and null_all10 < 0.5 and null_noise8 < 0.5
            n_auc, n_inf = int((auc_s > 0.8).sum()), int(inf_ok.sum())
            passed = null_ok and n_auc >= 9 and n_inf == 10
            rows.append({"lam": lam, "ep": ep, "pass": passed, "null_ok": null_ok,
                         "both": int(((auc_s > 0.8) & inf_ok).sum())})
            print(f"lambda {lam:<5} epochs {ep}: signal AUC>0.8 {n_auc}/10, inf gate {n_inf}/10, "
                  f"AUC mean {auc_s.mean():.3f} min {auc_s.min():.3f}, gates seg0 "
                  f"{np.mean([r['g_seg0'] for r in sig]):.2f} child0 "
                  f"{np.mean([r['g_child0'] for r in sig]):.2f} noise "
                  f"{np.mean([r['noise8'] for r in sig]):.2f} | null AUC {null_auc:.3f}, gates all "
                  f"{null_all10:.2f} noise8 {null_noise8:.2f} | {'PASS' if passed else 'fail'}",
                  flush=True)
            for name, runs in (("signal", sig), ("null", nul)):
                print(f"    {name:6s} AUC   " + " ".join(f"{r['auc']:.3f}" for r in runs))
                print(f"    {name:6s} gates " + " ".join(
                    f"{r['g_seg0']:.2f}/{r['g_child0']:.2f}/{r['noise8']:.2f}" for r in runs)
                    + "   (seg0/child0/noise8 per data seed)")

    passing = [r for r in rows if r["pass"]]
    if passing:
        best = max(passing, key=lambda r: (r["lam"], -r["ep"]))
        how = "largest passing gate_lambda, ties -> fewer epochs"
    else:
        best = max((r for r in rows if r["null_ok"]), key=lambda r: (r["both"], r["lam"], -r["ep"]))
        how = "NO grid point passed; most signal successes among null-passing points"
    print(f"CHOSEN: gate_lambda {best['lam']}, epochs {best['ep']} ({how})")


if __name__ == "__main__":
    main()
