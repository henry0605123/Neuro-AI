"""Read Phase 1 val_score (accuracy) from TensorBoard event files."""
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_BASE = "D:/PycharmProjects/paper/child-adult-diarization/whisper-modeling/logs"

EXPERIMENTS = [
    {
        "name": "A：ChangPN session（AudioSet 起點）",
        "tb_dir": f"{LOG_BASE}/changpn_pretrain",
        "best_epoch": 27,
        "best_val_loss": 0.466,
    },
    {
        "name": "B：ChangPN sim 50k（AudioSet 起點）",
        "tb_dir": f"{LOG_BASE}/changpn_sim_pretrain",
        "best_epoch": 29,
        "best_val_loss": 0.189,
    },
    {
        "name": "C：ChangPN session（base Whisper）",
        "tb_dir": f"{LOG_BASE}/changpn_pretrain_scratch",
        "best_epoch": 29,
        "best_val_loss": 0.478,
    },
    {
        "name": "D：ChangPN sim 50k（base Whisper）",
        "tb_dir": f"{LOG_BASE}/changpn_sim_pretrain_scratch",
        "best_epoch": 29,
        "best_val_loss": 0.191,
    },
]


def get_scalars(tb_dir, tag):
    """Return {step: value} from the latest version in tb_dir."""
    versions = sorted([
        d for d in os.listdir(tb_dir) if d.startswith("version_")
    ], key=lambda x: int(x.split("_")[1]))

    # Use the last version that has events
    for version in reversed(versions):
        v_path = os.path.join(tb_dir, version)
        ea = EventAccumulator(v_path)
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            return {e.step: e.value for e in ea.Scalars(tag)}
    return {}


print(f"\n{'='*70}")
print("Phase 1 Pre-training Accuracy (val_score at best epoch)")
print(f"{'='*70}")
print(f"{'實驗':<40} {'Best Epoch':>10} {'val_loss':>10} {'val_score (acc)':>16}")
print("-" * 78)

results = []
for exp in EXPERIMENTS:
    scores = get_scalars(exp["tb_dir"], "val_score")

    best_epoch = exp["best_epoch"]

    # TB logs step (= cumulative batch count), not epoch index.
    # Sort steps; step at index best_epoch corresponds to epoch best_epoch (0-indexed).
    if scores:
        sorted_steps = sorted(scores.keys())
        if best_epoch < len(sorted_steps):
            best_step = sorted_steps[best_epoch]
            val_score_at_best = scores[best_step]
        else:
            val_score_at_best = None
        best_score_overall = max(scores.values())
    else:
        val_score_at_best = None
        best_score_overall = None

    print(f"{exp['name']:<40} {best_epoch:>10} {exp['best_val_loss']:>10.3f} "
          f"{(val_score_at_best*100 if val_score_at_best is not None else float('nan')):>15.2f}%")

    results.append({
        "name": exp["name"],
        "best_epoch": best_epoch,
        "val_loss": exp["best_val_loss"],
        "val_score_at_best_epoch": val_score_at_best,
        "val_score_best_overall": best_score_overall,
        "sorted_steps": sorted(scores.keys()) if scores else [],
        "all_scores": scores,
    })

print()
print("All epoch val_scores per experiment:")
for r in results:
    print(f"\n{r['name']}")
    if r["all_scores"]:
        for idx, step in enumerate(r["sorted_steps"]):
            marker = " ← best ckpt" if idx == r["best_epoch"] else ""
            print(f"  epoch {idx:>2} (step {step:>6}): val_score={r['all_scores'][step]*100:.2f}%{marker}")
    else:
        print("  (no val_score found)")
